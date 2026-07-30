"""Restart-safe MONAI Swin UNETR development runner for the frozen v2 design."""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from monai.inferers.utils import sliding_window_inference
from monai.networks.nets.swin_unetr import SwinUNETR
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bratsarticle.data.dataset import BraTSSliceDataset, build_cv_fold_dataset
from bratsarticle.data.preprocessing import (
    PreprocessingConfig,
    load_preprocessing_config,
)
from bratsarticle.experiments.pilot_runner import PatientGroupedSampler
from bratsarticle.experiments.registry import ResourceTracker
from bratsarticle.models.resource_profile import profile_torch_module
from bratsarticle.training.checkpoint import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from bratsarticle.training.loss_catalog import (
    ConfiguredSegmentationLoss,
    build_loss,
    load_loss_catalog,
)
from bratsarticle.training.losses import class_indices_to_labels
from bratsarticle.training.reproducibility import (
    seed_dataloader_worker,
    seed_everything,
)
from bratsarticle.training.schedule import build_warmup_cosine_scheduler
from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
)
from evaluation import (
    CentralEvaluator,
    load_evaluation_config,
    summarize_patient_metrics,
)

SWIN_MODEL_ID = "swin_unetr"


@dataclass(frozen=True)
class SwinRunSpec:
    """Immutable identity for one Swin fold-seed development run."""

    model_id: str
    fold: int
    seed: int
    loss_name: str
    maximum_optimizer_steps: int
    warmup_optimizer_steps: int

    @property
    def run_id(self) -> str:
        """Return a deterministic filesystem-safe run identifier."""
        return (
            f"main_convergence__{self.model_id}__f{self.fold}"
            f"__s{self.seed}__{self.loss_name}"
        )

    @property
    def sha256(self) -> str:
        """Hash the canonical run identity."""
        return text_digest(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        )


def load_swin_runner_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen Swin runner configuration."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Swin runner configuration must be a mapping")
    config = cast(dict[str, Any], loaded)
    if config.get("status") != "frozen_before_first_reportable_development_run":
        raise PermissionError("Swin runner configuration is not frozen")
    hardware = cast(dict[str, Any], config["hardware"])
    if hardware.get("backend") != "mps":
        raise ValueError("The M1 Swin runner requires the MPS backend")
    if (
        hardware.get("deterministic_algorithms")
        != "warn_only_with_repeat_tolerance_audit"
    ):
        raise ValueError("Swin MPS determinism policy is not frozen")
    guards = cast(dict[str, Any], config["guards"])
    prohibited = (
        bool(guards["allow_external_data"])
        or bool(guards["allow_legacy_internal_test"])
        or bool(guards["allow_silent_seed_replacement"])
    )
    if prohibited:
        raise PermissionError("Swin runner enables prohibited data or seed conduct")
    training = cast(dict[str, Any], config["training"])
    microbatch = int(training["microbatch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    if microbatch * accumulation != int(training["effective_batch_size"]):
        raise ValueError("Swin effective batch size does not match accumulation")
    if microbatch != 1 or accumulation != 2:
        raise ValueError("The frozen M1 Swin microbatch/accumulation is 1 x 2")
    return config


def _load_selected_loss(path: Path) -> tuple[str, Path]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PermissionError("Selected-loss configuration must be a mapping")
    selected = cast(dict[str, Any], loaded)
    if selected.get("status") != "frozen_from_complete_development_cv":
        raise PermissionError("Architecture-attribution loss is not frozen")
    if selected.get("external_data_used_for_selection") is not False:
        raise PermissionError("Selected loss used external data")
    if selected.get("legacy_internal_test_used_for_selection") is not False:
        raise PermissionError("Selected loss used the legacy internal test")
    evidence_path = Path(str(selected.get("selection_artifact", "")))
    if not evidence_path.is_absolute():
        evidence_path = Path.cwd() / evidence_path
    if not evidence_path.is_file():
        raise PermissionError("Selected-loss evidence artifact is missing")
    if file_digest(evidence_path) != str(selected.get("selection_artifact_sha256")):
        raise PermissionError("Selected-loss evidence hash does not match")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    loss_name = str(selected["selected_loss"])
    if (
        evidence.get("status") != "selected_from_complete_development_cv"
        or str(evidence.get("selected_loss")) != loss_name
        or evidence.get("external_data_accessed") is not False
        or evidence.get("legacy_internal_test_accessed") is not False
    ):
        raise PermissionError("Selected-loss evidence is invalid")
    return loss_name, evidence_path


def swin_convergence_specs(
    config_path: Path,
    selected_loss_path: Path,
) -> tuple[SwinRunSpec, ...]:
    """Expand the frozen five-fold, equal-five-seed Swin matrix."""
    config = load_swin_runner_config(config_path)
    loss_name, _ = _load_selected_loss(selected_loss_path)
    matrix = cast(dict[str, Any], config["matrix"])
    training = cast(dict[str, Any], config["training"])
    specs = tuple(
        SwinRunSpec(
            model_id=SWIN_MODEL_ID,
            fold=int(fold),
            seed=int(seed),
            loss_name=loss_name,
            maximum_optimizer_steps=int(training["maximum_optimizer_steps"]),
            warmup_optimizer_steps=int(training["warmup_optimizer_steps"]),
        )
        for fold in cast(list[Any], matrix["folds"])
        for seed in cast(list[Any], matrix["seeds"])
    )
    if len(specs) != 25 or len({spec.sha256 for spec in specs}) != 25:
        raise ValueError("Frozen Swin matrix must contain 25 unique runs")
    if {spec.fold for spec in specs} != {1, 2, 3, 4, 5}:
        raise ValueError("Frozen Swin matrix must contain folds 1 through 5")
    if len({spec.seed for spec in specs}) != 5:
        raise ValueError("Frozen Swin matrix must contain five equal seeds")
    return specs


def resolve_swin_convergence_spec(
    config_path: Path,
    selected_loss_path: Path,
    *,
    fold: int,
    seed: int,
) -> SwinRunSpec:
    """Resolve a requested run only if it belongs to the frozen matrix."""
    matches = [
        spec
        for spec in swin_convergence_specs(config_path, selected_loss_path)
        if spec.fold == fold and spec.seed == seed
    ]
    if len(matches) != 1:
        raise PermissionError("Requested run is outside the frozen Swin matrix")
    return matches[0]


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return commit, bool(status.strip())


def _loss(catalog_path: Path, name: str) -> ConfiguredSegmentationLoss:
    matches = [
        config for config in load_loss_catalog(catalog_path) if config.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one loss named {name}")
    return build_loss(matches[0])


def _model(config_path: Path) -> tuple[nn.Module, tuple[int, int, int]]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not isinstance(loaded.get("model"), dict):
        raise ValueError("Swin model configuration is invalid")
    raw = cast(dict[str, Any], loaded["model"])
    patch = tuple(int(value) for value in cast(list[Any], raw["patch_size"]))
    if len(patch) != 3 or any(value <= 0 for value in patch):
        raise ValueError("Swin patch size must contain three positive values")
    model = SwinUNETR(
        in_channels=int(raw["input_channels"]),
        out_channels=int(raw["output_channels"]),
        feature_size=int(raw["feature_size"]),
        use_checkpoint=bool(raw["use_checkpoint"]),
        spatial_dims=int(raw["spatial_dims"]),
    )
    return model, (patch[0], patch[1], patch[2])


def _patch_bounds(
    shape: tuple[int, int, int],
    patch: tuple[int, int, int],
    center: tuple[int, int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for length, width, coordinate in zip(shape, patch, center, strict=True):
        if width > length:
            raise ValueError("Frozen Swin patch exceeds a BraTS volume dimension")
        start = min(max(coordinate - width // 2, 0), length - width)
        output.append((start, start + width))
    return (output[0], output[1], output[2])


def _sample_patch(
    image: np.ndarray,
    label: np.ndarray,
    *,
    patch_size: tuple[int, int, int],
    generator: np.random.Generator,
    tumor_probability: float,
    preprocessing: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 4 or image.shape[0] != 4 or image.shape[1:] != label.shape:
        raise ValueError("Swin sampling expects image [4,X,Y,Z] and label [X,Y,Z]")
    choose_tumor = bool(np.any(label)) and float(generator.random()) < float(
        tumor_probability
    )
    if choose_tumor:
        coordinates = np.argwhere(label != 0)
        selected = coordinates[int(generator.integers(0, len(coordinates)))]
        center = (int(selected[0]), int(selected[1]), int(selected[2]))
    else:
        center = (
            int(generator.integers(0, label.shape[0])),
            int(generator.integers(0, label.shape[1])),
            int(generator.integers(0, label.shape[2])),
        )
    volume_shape = (label.shape[0], label.shape[1], label.shape[2])
    bounds = _patch_bounds(
        volume_shape,
        patch_size,
        center,
    )
    slices = tuple(slice(start, stop) for start, stop in bounds)
    image_patch = np.array(
        image[(slice(None), *slices)],
        dtype=np.float32,
        copy=True,
    )
    label_patch = np.array(label[slices], dtype=np.int16, copy=True)

    if preprocessing.spatial_augmentation.enabled:
        for spatial_axis in range(3):
            if float(generator.random()) < (
                preprocessing.spatial_augmentation.flip_probability
            ):
                image_patch = np.flip(image_patch, axis=spatial_axis + 1)
                label_patch = np.flip(label_patch, axis=spatial_axis)
        if preprocessing.spatial_augmentation.rotate_90:
            rotations = int(generator.integers(0, 4))
            image_patch = np.rot90(
                image_patch,
                k=rotations,
                axes=(1, 2),
            )
            label_patch = np.rot90(
                label_patch,
                k=rotations,
                axes=(0, 1),
            )

    intensity = preprocessing.intensity_augmentation
    if intensity.enabled:
        for modality in range(image_patch.shape[0]):
            if float(generator.random()) >= intensity.apply_probability_per_modality:
                continue
            scale = float(generator.uniform(*intensity.scale_range))
            shift = float(generator.uniform(*intensity.shift_range))
            image_patch[modality] = image_patch[modality] * scale + shift
    return (
        np.ascontiguousarray(image_patch, dtype=np.float32),
        np.ascontiguousarray(label_patch, dtype=np.int64),
    )


class SwinPatchDataset(Dataset[dict[str, Any]]):
    """Deterministic patient-grouped 3D patch view over authorized fold data."""

    def __init__(
        self,
        volume_dataset: BraTSSliceDataset,
        *,
        patch_size: tuple[int, int, int],
        samples_per_patient: int,
        tumor_probability: float,
        seed: int,
        preprocessing: PreprocessingConfig,
    ) -> None:
        if samples_per_patient < 1:
            raise ValueError("samples_per_patient must be positive")
        if not 0.0 <= tumor_probability <= 1.0:
            raise ValueError("tumor_probability must be in [0, 1]")
        self.volume_dataset = volume_dataset
        self.patch_size = patch_size
        self.samples_per_patient = samples_per_patient
        self.tumor_probability = tumor_probability
        self.seed = seed
        self.preprocessing = preprocessing
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic patch-sampling epoch."""
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.volume_dataset.manifest) * self.samples_per_patient

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        patient_index = index // self.samples_per_patient
        generator = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        volume = self.volume_dataset.subject_volume(patient_index)
        image, label = _sample_patch(
            volume.image,
            volume.label,
            patch_size=self.patch_size,
            generator=generator,
            tumor_probability=self.tumor_probability,
            preprocessing=self.preprocessing,
        )
        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(label),
            "subject_id": str(
                self.volume_dataset.manifest.iloc[patient_index]["subject_id"]
            ),
        }


def _loader(
    dataset: SwinPatchDataset,
    *,
    microbatch_size: int,
    workers: int,
    seed: int,
    sampler: PatientGroupedSampler,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=microbatch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
        pin_memory=False,
        persistent_workers=workers > 0,
    )


def _batches_from_resume(
    loader: DataLoader[dict[str, Any]],
    consumed: int,
) -> Iterator[tuple[int, dict[str, Any]]]:
    if consumed < 0 or consumed > len(loader):
        raise ValueError("Invalid batches_consumed_in_epoch in checkpoint")
    for batch_index, batch in enumerate(loader):
        if batch_index >= consumed:
            yield batch_index, batch


@dataclass(frozen=True)
class SwinValidationResult:
    """Full-volume validation rows and selection values."""

    patient_rows: tuple[dict[str, Any], ...]
    mean_regional_dice: float
    validation_loss: float


def _selection_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    both_empty: float,
    one_empty: float,
) -> tuple[float, float, float]:
    prediction_regions = (
        prediction != 0,
        (prediction == 1) | (prediction == 4),
        prediction == 4,
    )
    target_regions = (
        target != 0,
        (target == 1) | (target == 4),
        target == 4,
    )
    values: list[float] = []
    for predicted_region, target_region in zip(
        prediction_regions,
        target_regions,
        strict=True,
    ):
        prediction_count = int(np.count_nonzero(predicted_region))
        target_count = int(np.count_nonzero(target_region))
        if prediction_count == 0 and target_count == 0:
            values.append(both_empty)
            continue
        if (prediction_count == 0) != (target_count == 0):
            values.append(one_empty)
            continue
        intersection = int(np.count_nonzero(predicted_region & target_region))
        values.append(2.0 * intersection / (prediction_count + target_count))
    return (values[0], values[1], values[2])


def validate_swin_full_volumes(
    model: nn.Module,
    dataset: BraTSSliceDataset,
    *,
    device: torch.device,
    evaluator: CentralEvaluator,
    loss_function: nn.Module,
    patch_size: tuple[int, int, int],
    overlap: float,
    mode: str,
    sliding_window_batch_size: int,
) -> SwinValidationResult:
    """Run untouched full-volume fold validation through the central evaluator."""
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    weighted_loss = 0.0
    voxel_count = 0
    with torch.no_grad():
        for patient_index, (_, manifest_row) in enumerate(dataset.manifest.iterrows()):
            volume = dataset.subject_volume(patient_index)
            image = torch.from_numpy(
                np.ascontiguousarray(volume.image[None], dtype=np.float32)
            )
            logits = cast(
                torch.Tensor,
                sliding_window_inference(
                    image,
                    roi_size=patch_size,
                    sw_batch_size=sliding_window_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                    sw_device=device,
                    device=torch.device("cpu"),
                ),
            )
            target = torch.from_numpy(
                np.ascontiguousarray(volume.label[None], dtype=np.int64)
            )
            patient_loss = loss_function(logits, target)
            if not torch.isfinite(patient_loss):
                raise FloatingPointError(
                    "Swin full-volume validation produced non-finite loss"
                )
            count = int(target.numel())
            weighted_loss += float(patient_loss) * count
            voxel_count += count
            predicted = class_indices_to_labels(torch.argmax(logits, dim=1))
            regional = _selection_dice(
                predicted.numpy()[0],
                volume.label,
                both_empty=evaluator.config.empty_masks.overlap_both_empty,
                one_empty=evaluator.config.empty_masks.overlap_one_empty,
            )
            rows.append(
                {
                    "patient_id": str(manifest_row["subject_id"]),
                    "evaluation_stage": "raw",
                    "wt_dice": regional[0],
                    "tc_dice": regional[1],
                    "et_dice": regional[2],
                    "mean_regional_dice": float(np.mean(regional)),
                }
            )
            del logits, target, predicted
    if was_training:
        model.train()
    if not rows or voxel_count == 0:
        raise RuntimeError("Swin validation produced no patient results")
    return SwinValidationResult(
        patient_rows=tuple(rows),
        mean_regional_dice=float(
            np.mean([float(row["mean_regional_dice"]) for row in rows])
        ),
        validation_loss=weighted_loss / voxel_count,
    )


def evaluate_swin_full_metrics(
    model: nn.Module,
    dataset: BraTSSliceDataset,
    *,
    device: torch.device,
    evaluator: CentralEvaluator,
    patch_size: tuple[int, int, int],
    overlap: float,
    mode: str,
    sliding_window_batch_size: int,
) -> list[dict[str, Any]]:
    """Evaluate best-checkpoint Swin predictions with every central metric."""
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for patient_index, (_, manifest_row) in enumerate(dataset.manifest.iterrows()):
            volume = dataset.subject_volume(patient_index)
            image = torch.from_numpy(
                np.ascontiguousarray(volume.image[None], dtype=np.float32)
            )
            logits = cast(
                torch.Tensor,
                sliding_window_inference(
                    image,
                    roi_size=patch_size,
                    sw_batch_size=sliding_window_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode=mode,
                    sw_device=device,
                    device=torch.device("cpu"),
                ),
            )
            predicted = class_indices_to_labels(torch.argmax(logits, dim=1))
            rows.extend(
                evaluator.evaluate_batch(
                    predicted.numpy(),
                    volume.label[None],
                    patient_ids=[str(manifest_row["subject_id"])],
                    spacings_mm=[volume.spacing_mm],
                )
            )
            del logits, predicted
    if was_training:
        model.train()
    if not rows:
        raise RuntimeError("Swin full-metric evaluation produced no patient rows")
    return rows


def _improved(
    metric: float,
    validation_loss: float,
    progress: Mapping[str, Any],
) -> bool:
    best_metric = progress["best_metric"]
    if best_metric is None or metric > float(best_metric):
        return True
    if not np.isclose(metric, float(best_metric), rtol=0.0, atol=1e-12):
        return False
    best_loss = progress["best_validation_loss"]
    return best_loss is None or validation_loss < float(best_loss)


def _progress() -> dict[str, Any]:
    return {
        "status": "running",
        "best_metric": None,
        "best_validation_loss": None,
        "best_step": None,
        "early_stopping_reference_metric": None,
        "validation_checks_without_minimum_improvement": 0,
        "completed_validation_checks": 0,
        "stop_reason": None,
        "cumulative_elapsed_seconds": 0.0,
    }


def _repeat_tolerance_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    report = json.loads(path.read_text(encoding="utf-8"))
    return (
        report.get("status") == "pass"
        and report.get("model_id") == SWIN_MODEL_ID
        and report.get("external_data_accessed") is False
        and report.get("legacy_internal_test_accessed") is False
    )


def run_swin_development(
    *,
    runner_config_path: Path,
    selected_loss_path: Path,
    spec: SwinRunSpec,
    dataset_root: Path,
    allow_reportable_development_training: bool,
    resume: bool,
    repeat_tolerance_path: Path = Path(
        "reports/q1q2_v2/swin_mps_repeat_tolerance.json"
    ),
) -> Path:
    """Run or resume one frozen reportable Swin development job."""
    if not allow_reportable_development_training:
        raise PermissionError(
            "Swin development training requires explicit authorization"
        )
    config = load_swin_runner_config(runner_config_path)
    allowed = swin_convergence_specs(runner_config_path, selected_loss_path)
    if spec.sha256 not in {candidate.sha256 for candidate in allowed}:
        raise PermissionError("Run specification is outside the frozen Swin matrix")
    runtime_root = Path("artifacts/q1q2_v2/queue_runtime")
    if (runtime_root / "loss_screen.lock").exists():
        raise RuntimeError("Loss-screen queue is still active")
    guards = cast(dict[str, Any], config["guards"])
    if bool(guards["require_mps_repeat_tolerance_audit_before_main"]) and not (
        _repeat_tolerance_passed(repeat_tolerance_path)
    ):
        raise PermissionError("Swin MPS repeat-tolerance audit has not passed")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Swin M1 development runs require available MPS")
    git_commit, dirty = _git_state()
    hardware = cast(dict[str, Any], config["hardware"])
    if bool(hardware["require_clean_git"]) and dirty:
        raise RuntimeError("Reportable Swin training requires a clean worktree")

    data = cast(dict[str, Any], config["data"])
    model_raw = cast(dict[str, Any], config["model"])
    loss_raw = cast(dict[str, Any], config["loss"])
    training = cast(dict[str, Any], config["training"])
    validation = cast(dict[str, Any], config["validation"])
    resources = cast(dict[str, Any], config["resource_profiling"])
    artifacts = cast(dict[str, Any], config["artifacts"])
    fold_path = Path(str(data["fold_pattern"]).format(fold=spec.fold))
    canonical_path = Path(str(data["canonical_manifest"]))
    preprocessing_path = Path(str(data["preprocessing"]))
    evaluation_path = Path(str(data["evaluation"]))
    resource_profile_path = Path(str(resources["protocol"]))
    resource_protocol_raw = yaml.safe_load(
        resource_profile_path.read_text(encoding="utf-8")
    )
    if not isinstance(resource_protocol_raw, dict):
        raise ValueError("Resource profile protocol must be a mapping")
    resource_protocol = cast(dict[str, Any], resource_protocol_raw)
    timing = cast(dict[str, Any], resource_protocol["timing"])
    timing_warmup = int(timing["warmup_iterations"])
    timing_measurements = int(timing["measured_iterations"])
    if timing_warmup < 0 or timing_measurements < 1:
        raise ValueError("Resource timing protocol is invalid")
    model_config_path = Path(str(model_raw["config"]))
    catalog_path = Path(str(loss_raw["catalog"]))
    evidence_path = _load_selected_loss(selected_loss_path)[1]
    output_dir = Path(str(artifacts["root"])).resolve() / spec.run_id
    assert_output_paths_safe([output_dir], [dataset_root.resolve()])
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    metadata = {
        "schema_version": 1,
        "run_id": spec.run_id,
        "run_spec": asdict(spec),
        "run_spec_sha256": spec.sha256,
        "stage": "main_convergence",
        "scientific_role": "reportable_development_cross_validation",
        "git_commit": git_commit,
        "repository_dirty_at_start": False,
        "model_id": spec.model_id,
        "fold": spec.fold,
        "seed": spec.seed,
        "loss": spec.loss_name,
        "hardware": {
            "backend": "mps",
            "device": "Apple M1 Max",
            "determinism": "warn_only_with_passed_repeat_tolerance_audit",
            "memory_terminology": ("MPS framework-reported allocated unified memory"),
        },
        "hashes": {
            "runner_config": file_digest(runner_config_path),
            "model_matrix": file_digest(Path(str(model_raw["matrix"]))),
            "model_config": file_digest(model_config_path),
            "loss_catalog": file_digest(catalog_path),
            "selected_loss_config": file_digest(selected_loss_path),
            "loss_selection_artifact": file_digest(evidence_path),
            "fold_manifest": file_digest(fold_path),
            "canonical_manifest": file_digest(canonical_path),
            "preprocessing": file_digest(preprocessing_path),
            "evaluation": file_digest(evaluation_path),
            "resource_profile_protocol": file_digest(resource_profile_path),
            "repeat_tolerance_audit": file_digest(repeat_tolerance_path),
            "environment_lock": file_digest(
                Path("environment/q1q2_v2-environment.json")
            ),
            "requirements_lock": file_digest(
                Path("environment/q1q2_v2-requirements-lock.txt")
            ),
            "hardware_preflight": file_digest(
                Path("reports/q1q2_v2/hardware_preflight.json")
            ),
        },
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
        "status": "running",
    }
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    recovery_path = checkpoint_dir / "recovery.pt"
    if resume:
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored["run_spec_sha256"] != spec.sha256:
            raise ValueError("Resume run-spec hash differs")
        if stored["hashes"] != metadata["hashes"]:
            raise ValueError("Resume scientific-input hashes differ")
        progress = cast(
            dict[str, Any],
            json.loads(progress_path.read_text(encoding="utf-8")),
        )
    else:
        progress = _progress()
        atomic_write_json(metadata_path, metadata)
        atomic_write_json(output_dir / "run_spec.json", asdict(spec))
        atomic_write_json(progress_path, progress)

    seed_everything(spec.seed, deterministic=True, deterministic_warn_only=True)
    device = torch.device("mps")
    preprocessing = load_preprocessing_config(preprocessing_path)
    train_volumes = build_cv_fold_dataset(
        fold_path,
        canonical_path,
        "train",
        dataset_root,
        preprocessing,
        seed=spec.seed,
    )
    validation_volumes = build_cv_fold_dataset(
        fold_path,
        canonical_path,
        "validation",
        dataset_root,
        preprocessing,
        seed=spec.seed,
    )
    model, patch_size = _model(model_config_path)
    static_profile = profile_torch_module(
        model,
        input_shape=(1, 4, *patch_size),
    )
    static_profile.update(
        {
            "receptive_field_proxy_definition": (
                "full_declared_input_patch_via_hierarchical_shifted_windows"
            ),
            "receptive_field_proxy_voxels_per_axis": list(patch_size),
        }
    )
    metadata["parameter_count"] = int(
        cast(int, static_profile["parameter_count"])
    )
    metadata["static_profile"] = static_profile
    atomic_write_json(metadata_path, metadata)
    train_dataset = SwinPatchDataset(
        train_volumes,
        patch_size=patch_size,
        samples_per_patient=int(data["patches_per_patient_per_epoch"]),
        tumor_probability=float(data["tumor_patch_probability"]),
        seed=spec.seed,
        preprocessing=preprocessing,
    )
    sampler = PatientGroupedSampler(
        patient_count=len(train_volumes.manifest),
        samples_per_patient=int(data["patches_per_patient_per_epoch"]),
        seed=spec.seed,
    )
    train_loader = _loader(
        train_dataset,
        microbatch_size=int(training["microbatch_size"]),
        workers=int(data["training_workers"]),
        seed=spec.seed,
        sampler=sampler,
    )
    accumulation = int(training["gradient_accumulation_steps"])
    if len(train_loader) % accumulation:
        raise ValueError("Swin training batches must divide gradient accumulation")
    loss_function = _loss(catalog_path, spec.loss_name)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=spec.warmup_optimizer_steps,
        total_steps=spec.maximum_optimizer_steps,
        minimum_fraction=float(training["minimum_learning_rate_fraction"]),
    )
    model.to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    state = TrainingState()
    if resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(
                "Resume requested but recovery checkpoint is absent"
            )
        state, checkpoint_metadata = load_checkpoint(
            recovery_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            map_location=device,
        )
        if checkpoint_metadata["run_spec_sha256"] != spec.sha256:
            raise ValueError("Resume checkpoint run-spec hash differs")
    evaluator = CentralEvaluator(load_evaluation_config(evaluation_path))
    tracker = ResourceTracker(device)
    elapsed_before = float(progress["cumulative_elapsed_seconds"])
    validation_frequency = int(training["validation_frequency_optimizer_steps"])
    minimum_delta = float(training["early_stopping_minimum_delta"])
    patience = int(training["early_stopping_patience_validation_checks"])
    minimum_steps_before_early_stopping = int(
        training["minimum_optimizer_steps_before_early_stopping"]
    )
    milestone_steps = {
        int(value)
        for value in cast(
            list[Any],
            training["budget_sensitivity_checkpoint_steps"],
        )
    }
    progress.setdefault("budget_sensitivity_checkpoints", {})
    progress.setdefault("synchronized_training_step_seconds", [])
    stop = False
    try:
        while state.global_step < spec.maximum_optimizer_steps and not stop:
            train_dataset.set_epoch(state.epoch)
            sampler.set_epoch(state.epoch)
            pending_losses: list[float] = []
            optimizer.zero_grad(set_to_none=True)
            optimizer_step_started: float | None = None
            for batch_index, batch in _batches_from_resume(
                train_loader,
                state.batches_consumed_in_epoch,
            ):
                step_samples = cast(
                    list[float],
                    progress["synchronized_training_step_seconds"],
                )
                if (
                    not pending_losses
                    and state.global_step >= timing_warmup
                    and len(step_samples) < timing_measurements
                ):
                    torch.mps.synchronize()
                    optimizer_step_started = time.perf_counter()
                model.train()
                image = batch["image"].to(device, dtype=torch.float32)
                label = batch["label"].to(device, dtype=torch.long)
                logits = cast(torch.Tensor, model(image))
                loss = loss_function(logits, label)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Swin training produced non-finite loss")
                (loss / accumulation).backward()
                pending_losses.append(float(loss.detach().cpu()))
                state.batches_consumed_in_epoch = batch_index + 1
                if len(pending_losses) < accumulation:
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                state.global_step += 1
                if optimizer_step_started is not None:
                    torch.mps.synchronize()
                    step_samples.append(time.perf_counter() - optimizer_step_started)
                    optimizer_step_started = None
                training_loss = float(np.mean(pending_losses))
                pending_losses.clear()
                at_validation = (
                    state.global_step % validation_frequency == 0
                    or state.global_step >= spec.maximum_optimizer_steps
                )
                if not at_validation:
                    continue
                selection = validate_swin_full_volumes(
                    model,
                    validation_volumes,
                    device=device,
                    evaluator=evaluator,
                    loss_function=loss_function.cpu(),
                    patch_size=patch_size,
                    overlap=float(validation["sliding_window_overlap"]),
                    mode=str(validation["sliding_window_mode"]),
                    sliding_window_batch_size=int(
                        validation["sliding_window_batch_size"]
                    ),
                )
                loss_function.to(device)
                is_best = _improved(
                    selection.mean_regional_dice,
                    selection.validation_loss,
                    progress,
                )
                if is_best:
                    progress["best_metric"] = selection.mean_regional_dice
                    progress["best_validation_loss"] = selection.validation_loss
                    progress["best_step"] = state.global_step
                    atomic_write_csv(
                        output_dir / "best_validation_per_patient.csv",
                        list(selection.patient_rows),
                    )
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        state=state,
                        metadata=metadata,
                    )
                if state.global_step in milestone_steps:
                    milestone_path = (
                        checkpoint_dir / f"budget_step_{state.global_step}.pt"
                    )
                    save_checkpoint(
                        milestone_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        state=state,
                        metadata=metadata,
                    )
                    milestone_metrics = output_dir / (
                        f"validation_step_{state.global_step}_per_patient.csv"
                    )
                    atomic_write_csv(
                        milestone_metrics,
                        list(selection.patient_rows),
                    )
                    progress["budget_sensitivity_checkpoints"][
                        str(state.global_step)
                    ] = {
                        "checkpoint": milestone_path.as_posix(),
                        "checkpoint_sha256": file_digest(milestone_path),
                        "patient_metrics": milestone_metrics.as_posix(),
                        "patient_metrics_sha256": file_digest(milestone_metrics),
                    }
                reference = progress["early_stopping_reference_metric"]
                if (
                    reference is None
                    or selection.mean_regional_dice > float(reference) + minimum_delta
                ):
                    progress["early_stopping_reference_metric"] = (
                        selection.mean_regional_dice
                    )
                    progress["validation_checks_without_minimum_improvement"] = 0
                else:
                    progress["validation_checks_without_minimum_improvement"] = (
                        int(progress["validation_checks_without_minimum_improvement"])
                        + 1
                    )
                progress["completed_validation_checks"] = (
                    int(progress["completed_validation_checks"]) + 1
                )
                append_jsonl(
                    output_dir / "metrics_per_validation.jsonl",
                    {
                        "record_type": "validation_check",
                        "optimizer_step": state.global_step,
                        "epoch": state.epoch,
                        "batches_consumed_in_epoch": (state.batches_consumed_in_epoch),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "last_training_loss": training_loss,
                        "validation_patient_mean_regional_dice": (
                            selection.mean_regional_dice
                        ),
                        "validation_loss": selection.validation_loss,
                        "is_best": is_best,
                        "checks_without_minimum_improvement": progress[
                            "validation_checks_without_minimum_improvement"
                        ],
                    },
                )
                save_checkpoint(
                    recovery_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler=scheduler,
                    state=state,
                    metadata=metadata,
                )
                if state.global_step >= minimum_steps_before_early_stopping and (
                    int(progress["validation_checks_without_minimum_improvement"])
                    >= patience
                ):
                    progress["stop_reason"] = "early_stopping_patience"
                    stop = True
                elif state.global_step >= spec.maximum_optimizer_steps:
                    progress["stop_reason"] = "maximum_optimizer_steps"
                    stop = True
                progress["cumulative_elapsed_seconds"] = (
                    elapsed_before + tracker.elapsed_seconds()
                )
                atomic_write_json(progress_path, progress)
                if stop:
                    break
            if pending_losses:
                raise RuntimeError("Swin epoch ended with incomplete accumulation")
            if not stop:
                state.epoch += 1
                state.batches_consumed_in_epoch = 0
        save_checkpoint(
            checkpoint_dir / "terminal.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            state=state,
            metadata=metadata,
        )
        best_payload = cast(
            dict[str, Any],
            torch.load(
                checkpoint_dir / "best.pt",
                map_location=device,
                weights_only=False,
            ),
        )
        best_metadata = cast(dict[str, Any], best_payload["metadata"])
        if best_metadata.get("run_spec_sha256") != spec.sha256:
            raise ValueError("Best Swin checkpoint run-spec hash differs")
        model.load_state_dict(best_payload["model"])
        full_metric_rows = evaluate_swin_full_metrics(
            model,
            validation_volumes,
            device=device,
            evaluator=evaluator,
            patch_size=patch_size,
            overlap=float(validation["sliding_window_overlap"]),
            mode=str(validation["sliding_window_mode"]),
            sliding_window_batch_size=int(validation["sliding_window_batch_size"]),
        )
        full_metric_path = output_dir / "best_checkpoint_full_metrics.csv"
        full_metric_summary_path = (
            output_dir / "best_checkpoint_full_metric_summary.csv"
        )
        checkpoint_sha256 = file_digest(checkpoint_dir / "best.pt")
        atomic_write_csv(
            full_metric_path,
            [
                {
                    "run_id": spec.run_id,
                    "model_id": spec.model_id,
                    "fold": spec.fold,
                    "seed": spec.seed,
                    "checkpoint_role": "best_development",
                    "checkpoint_sha256": checkpoint_sha256,
                    **row,
                }
                for row in full_metric_rows
            ],
        )
        atomic_write_csv(
            full_metric_summary_path,
            summarize_patient_metrics(full_metric_rows),
        )
        progress["full_metric_evaluation"] = {
            "checkpoint": (checkpoint_dir / "best.pt").as_posix(),
            "checkpoint_sha256": checkpoint_sha256,
            "patient_metrics": full_metric_path.as_posix(),
            "patient_metrics_sha256": file_digest(full_metric_path),
            "metric_summary": full_metric_summary_path.as_posix(),
            "metric_summary_sha256": file_digest(full_metric_summary_path),
            "patient_count": len(full_metric_rows),
        }
        acceptance = {
            "best_checkpoint": (checkpoint_dir / "best.pt").is_file(),
            "terminal_checkpoint": (checkpoint_dir / "terminal.pt").is_file(),
            "validation_checks": int(progress["completed_validation_checks"]) > 0,
            "stop_reason": progress["stop_reason"]
            in {"early_stopping_patience", "maximum_optimizer_steps"},
            "budget_sensitivity_checkpoints": set(
                progress["budget_sensitivity_checkpoints"]
            )
            == {str(value) for value in milestone_steps},
            "full_metric_evaluation": (
                full_metric_path.is_file() and full_metric_summary_path.is_file()
            ),
        }
        progress["status"] = "completed" if all(acceptance.values()) else "invalid"
        progress["acceptance"] = acceptance
        metadata["status"] = progress["status"]
    except Exception:
        progress["status"] = "failed"
        progress["error"] = traceback.format_exc()
        metadata["status"] = "failed"
        metadata["error"] = progress["error"]
        raise
    finally:
        resource = tracker.snapshot()
        resource["cumulative_elapsed_seconds_before_session"] = elapsed_before
        resource["cumulative_elapsed_seconds_including_session"] = (
            elapsed_before + float(resource["elapsed_seconds"])
        )
        resource["completed_optimizer_steps"] = state.global_step
        resource["completed_validation_checks"] = int(
            progress.get("completed_validation_checks", 0)
        )
        resource["checkpoint_size_bytes"] = (
            (checkpoint_dir / "terminal.pt").stat().st_size
            if (checkpoint_dir / "terminal.pt").is_file()
            else None
        )
        step_samples_array = np.asarray(
            progress.get("synchronized_training_step_seconds", []),
            dtype=np.float64,
        )
        resource["resource_profile_protocol"] = resource_profile_path.as_posix()
        resource["resource_profile_protocol_sha256"] = file_digest(
            resource_profile_path
        )
        resource["training_step_timing_warmup_iterations"] = timing_warmup
        resource["synchronized_training_step_seconds"] = step_samples_array.tolist()
        resource["synchronized_training_step_measurement_count"] = len(
            step_samples_array
        )
        resource["synchronized_training_step_mean_seconds"] = (
            float(step_samples_array.mean()) if len(step_samples_array) else None
        )
        resource["synchronized_training_step_p50_seconds"] = (
            float(np.quantile(step_samples_array, 0.5))
            if len(step_samples_array)
            else None
        )
        resource["synchronized_training_step_p95_seconds"] = (
            float(np.quantile(step_samples_array, 0.95))
            if len(step_samples_array)
            else None
        )
        atomic_write_json(output_dir / "resource_profile.json", resource)
        atomic_write_json(progress_path, progress)
        atomic_write_json(metadata_path, metadata)
    return output_dir


__all__ = [
    "SWIN_MODEL_ID",
    "SwinPatchDataset",
    "SwinRunSpec",
    "SwinValidationResult",
    "evaluate_swin_full_metrics",
    "load_swin_runner_config",
    "resolve_swin_convergence_spec",
    "run_swin_development",
    "swin_convergence_specs",
    "validate_swin_full_volumes",
]
