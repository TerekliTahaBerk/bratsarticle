"""Restart-safe native 2D/2.5D development runner for the frozen v2 design."""

from __future__ import annotations

import json
import subprocess
import traceback
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from bratsarticle.data.dataset import (
    BraTSSliceDataset,
    build_cv_fold_dataset,
)
from bratsarticle.data.preprocessing import load_preprocessing_config
from bratsarticle.experiments.pilot_runner import PatientGroupedSampler
from bratsarticle.experiments.registry import ResourceTracker
from bratsarticle.models.configurable_unet import (
    load_model_config,
    model_from_config,
)
from bratsarticle.training.checkpoint import load_checkpoint, save_checkpoint
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.loss_catalog import (
    ConfiguredSegmentationLoss,
    build_loss,
    load_loss_catalog,
)
from bratsarticle.training.reproducibility import (
    seed_dataloader_worker,
    seed_everything,
)
from bratsarticle.training.schedule import build_warmup_cosine_scheduler
from bratsarticle.training.validation import validate_selection_dice
from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
)
from evaluation import CentralEvaluator, load_evaluation_config

NativeStage = Literal[
    "loss_screen",
    "main_convergence",
    "main_compute_matched",
    "loss_interaction",
]


@dataclass(frozen=True)
class NativeRunSpec:
    """Immutable identity of one model-fold-seed-loss development run."""

    stage: NativeStage
    model_id: str
    fold: int
    seed: int
    loss_name: str
    maximum_optimizer_steps: int
    warmup_optimizer_steps: int
    full_metric_evaluation: bool

    @property
    def run_id(self) -> str:
        """Return a deterministic filesystem-safe identifier."""
        return (
            f"{self.stage}__{self.model_id}__f{self.fold}"
            f"__s{self.seed}__{self.loss_name}"
        )

    @property
    def sha256(self) -> str:
        """Hash the canonical run identity."""
        return text_digest(
            json.dumps(
                asdict(self),
                sort_keys=True,
                separators=(",", ":"),
            )
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], payload)


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


def _model_entry(
    matrix_path: Path,
    model_id: str,
) -> dict[str, Any]:
    matrix = _load_yaml(matrix_path)
    matches = [
        raw
        for raw in cast(list[dict[str, Any]], matrix["main_models"])
        if str(raw["id"]) == model_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one model-matrix entry for {model_id}")
    entry = matches[0]
    if str(entry["adapter"]) != "native_configurable_unet":
        raise ValueError(f"{model_id} is not supported by the native runner")
    return entry


def _loss(catalog_path: Path, name: str) -> ConfiguredSegmentationLoss:
    matches = [
        config for config in load_loss_catalog(catalog_path) if config.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one loss named {name}")
    return build_loss(matches[0])


def loss_screen_specs(config_path: Path) -> tuple[NativeRunSpec, ...]:
    """Expand the frozen 3-loss by 5-fold by 1-seed screening matrix."""
    config = _load_yaml(config_path)
    stages = cast(dict[str, Any], config["stages"])
    stage = cast(dict[str, Any], stages["loss_screen"])
    specs = tuple(
        NativeRunSpec(
            stage="loss_screen",
            model_id=str(model),
            fold=int(fold),
            seed=int(seed),
            loss_name=str(loss_name),
            maximum_optimizer_steps=int(stage["maximum_optimizer_steps"]),
            warmup_optimizer_steps=int(stage["warmup_optimizer_steps"]),
            full_metric_evaluation=bool(stage["full_metric_evaluation"]),
        )
        for model in cast(list[Any], stage["models"])
        for fold in cast(list[Any], stage["folds"])
        for seed in cast(list[Any], stage["seeds"])
        for loss_name in cast(list[Any], stage["losses"])
    )
    if len(specs) != 15 or len({spec.run_id for spec in specs}) != 15:
        raise ValueError("Frozen loss screen must expand to 15 unique runs")
    return specs


def resolve_loss_screen_spec(
    config_path: Path,
    *,
    model_id: str,
    fold: int,
    seed: int,
    loss_name: str,
) -> NativeRunSpec:
    """Resolve a requested run only if it belongs to the frozen matrix."""
    matches = [
        spec
        for spec in loss_screen_specs(config_path)
        if (
            spec.model_id == model_id
            and spec.fold == fold
            and spec.seed == seed
            and spec.loss_name == loss_name
        )
    ]
    if len(matches) != 1:
        raise PermissionError("Requested run is outside the frozen loss-screen matrix")
    return matches[0]


def main_convergence_specs(
    config_path: Path,
    selected_loss_path: Path,
) -> tuple[NativeRunSpec, ...]:
    """Expand the nine-native-model, five-fold, equal-five-seed matrix."""
    config = _load_yaml(config_path)
    selected = _load_yaml(selected_loss_path)
    if selected.get("status") != "frozen_from_complete_development_cv":
        raise PermissionError("Architecture-attribution loss is not frozen")
    if selected.get("external_data_used_for_selection") is not False:
        raise PermissionError("Selected loss used external data")
    if selected.get("legacy_internal_test_used_for_selection") is not False:
        raise PermissionError("Selected loss used the legacy internal test")
    selected_loss = str(selected["selected_loss"])
    selection_artifact = Path(str(selected.get("selection_artifact", "")))
    if not selection_artifact.is_absolute():
        selection_artifact = Path.cwd() / selection_artifact
    if not selection_artifact.is_file():
        raise PermissionError("Selected-loss evidence artifact is missing")
    if (
        file_digest(selection_artifact)
        != str(selected.get("selection_artifact_sha256"))
    ):
        raise PermissionError("Selected-loss evidence hash does not match")
    selection_evidence = cast(
        dict[str, Any],
        json.loads(selection_artifact.read_text(encoding="utf-8")),
    )
    if (
        selection_evidence.get("status")
        != "selected_from_complete_development_cv"
        or str(selection_evidence.get("selected_loss")) != selected_loss
        or selection_evidence.get("external_data_accessed") is not False
        or selection_evidence.get("legacy_internal_test_accessed") is not False
    ):
        raise PermissionError("Selected-loss evidence is invalid")
    stages = cast(dict[str, Any], config["stages"])
    stage = cast(dict[str, Any], stages["main_convergence"])
    configured_loss = str(stage["loss"])
    if configured_loss not in {"pending_development_cv", selected_loss}:
        raise PermissionError("Main convergence config conflicts with frozen loss")
    specs = tuple(
        NativeRunSpec(
            stage="main_convergence",
            model_id=str(model),
            fold=int(fold),
            seed=int(seed),
            loss_name=selected_loss,
            maximum_optimizer_steps=int(stage["maximum_optimizer_steps"]),
            warmup_optimizer_steps=int(stage["warmup_optimizer_steps"]),
            full_metric_evaluation=bool(stage["full_metric_evaluation"]),
        )
        for model in cast(list[Any], stage["models"])
        for fold in cast(list[Any], stage["folds"])
        for seed in cast(list[Any], stage["seeds"])
    )
    expected_count = 9 * 5 * 5
    if (
        len(specs) != expected_count
        or len({spec.run_id for spec in specs}) != expected_count
    ):
        raise ValueError(
            f"Frozen native main matrix must expand to {expected_count} unique runs"
        )
    return specs


def resolve_main_convergence_spec(
    config_path: Path,
    selected_loss_path: Path,
    *,
    model_id: str,
    fold: int,
    seed: int,
) -> NativeRunSpec:
    """Resolve a main run only if it belongs to the frozen native matrix."""
    matches = [
        spec
        for spec in main_convergence_specs(config_path, selected_loss_path)
        if (
            spec.model_id == model_id
            and spec.fold == fold
            and spec.seed == seed
        )
    ]
    if len(matches) != 1:
        raise PermissionError("Requested run is outside the frozen native main matrix")
    return matches[0]


def main_compute_matched_specs(
    config_path: Path,
    selected_loss_path: Path,
) -> tuple[NativeRunSpec, ...]:
    """Expand the eight-model, five-fold, five-seed compute-matched matrix."""
    selected_loss = main_convergence_specs(
        config_path,
        selected_loss_path,
    )[0].loss_name
    config = _load_yaml(config_path)
    stage = cast(
        dict[str, Any],
        cast(dict[str, Any], config["stages"])["main_compute_matched"],
    )
    if str(stage["loss"]) not in {"pending_development_cv", selected_loss}:
        raise PermissionError(
            "Compute-matched config conflicts with the frozen selected loss"
        )
    if float(stage["maximum_accelerator_hours"]) != 4.0:
        raise ValueError("Compute-matched runs must use the frozen four-hour budget")
    specs = tuple(
        NativeRunSpec(
            stage="main_compute_matched",
            model_id=str(model),
            fold=int(fold),
            seed=int(seed),
            loss_name=selected_loss,
            maximum_optimizer_steps=int(stage["maximum_optimizer_steps"]),
            warmup_optimizer_steps=int(stage["warmup_optimizer_steps"]),
            full_metric_evaluation=bool(stage["full_metric_evaluation"]),
        )
        for model in cast(list[Any], stage["models"])
        for fold in cast(list[Any], stage["folds"])
        for seed in cast(list[Any], stage["seeds"])
    )
    expected_count = 8 * 5 * 5
    if (
        len(specs) != expected_count
        or len({spec.run_id for spec in specs}) != expected_count
    ):
        raise ValueError(
            "Frozen native compute-matched matrix must contain "
            f"{expected_count} unique runs"
        )
    return specs


def resolve_main_compute_matched_spec(
    config_path: Path,
    selected_loss_path: Path,
    *,
    model_id: str,
    fold: int,
    seed: int,
) -> NativeRunSpec:
    """Resolve a compute-matched run only if it belongs to the frozen matrix."""
    matches = [
        spec
        for spec in main_compute_matched_specs(config_path, selected_loss_path)
        if (
            spec.model_id == model_id
            and spec.fold == fold
            and spec.seed == seed
        )
    ]
    if len(matches) != 1:
        raise PermissionError(
            "Requested run is outside the frozen native compute-matched matrix"
        )
    return matches[0]


def loss_interaction_specs(
    config_path: Path,
    selected_loss_path: Path,
) -> tuple[NativeRunSpec, ...]:
    """Expand the four-finalist alternative-loss interaction matrix."""
    selected_loss = main_convergence_specs(
        config_path,
        selected_loss_path,
    )[0].loss_name
    config = _load_yaml(config_path)
    loss_protocol_path = Path(
        str(cast(dict[str, Any], config["losses"])["protocol"])
    )
    loss_protocol = _load_yaml(loss_protocol_path)
    interaction = cast(
        dict[str, Any],
        loss_protocol["interaction_sensitivity"],
    )
    alternative_rule = cast(
        dict[str, Any],
        interaction["alternative_loss_rule"],
    )
    priority = [
        str(value)
        for value in cast(
            list[Any],
            alternative_rule["choose_first_not_equal_to_selected"],
        )
    ]
    alternatives = [name for name in priority if name != selected_loss]
    if not alternatives:
        raise ValueError("Loss-interaction rule has no alternative to selected loss")
    alternative_loss = alternatives[0]
    stage = cast(
        dict[str, Any],
        cast(dict[str, Any], config["stages"])["loss_interaction"],
    )
    if str(stage["loss"]) != "deterministic_alternative_to_selected":
        raise PermissionError("Loss-interaction alternative rule changed")
    if int(interaction["minimum_distinct_losses_per_finalist"]) != 2:
        raise ValueError("Loss interaction must contain exactly two loss settings")
    configured_models = [str(value) for value in cast(list[Any], stage["models"])]
    protocol_models = [
        str(value) for value in cast(list[Any], interaction["finalists"])
    ]
    if configured_models != protocol_models:
        raise ValueError("Loss-interaction runner differs from the loss protocol")
    specs = tuple(
        NativeRunSpec(
            stage="loss_interaction",
            model_id=model,
            fold=int(fold),
            seed=int(seed),
            loss_name=alternative_loss,
            maximum_optimizer_steps=int(stage["maximum_optimizer_steps"]),
            warmup_optimizer_steps=int(stage["warmup_optimizer_steps"]),
            full_metric_evaluation=bool(stage["full_metric_evaluation"]),
        )
        for model in configured_models
        for fold in cast(list[Any], stage["folds"])
        for seed in cast(list[Any], stage["seeds"])
    )
    expected_count = 4 * 5 * 5
    if (
        len(specs) != expected_count
        or len({spec.run_id for spec in specs}) != expected_count
    ):
        raise ValueError(
            "Frozen loss-interaction matrix must contain "
            f"{expected_count} unique runs"
        )
    return specs


def resolve_loss_interaction_spec(
    config_path: Path,
    selected_loss_path: Path,
    *,
    model_id: str,
    fold: int,
    seed: int,
) -> NativeRunSpec:
    """Resolve an alternative-loss run only within the frozen matrix."""
    matches = [
        spec
        for spec in loss_interaction_specs(config_path, selected_loss_path)
        if (
            spec.model_id == model_id
            and spec.fold == fold
            and spec.seed == seed
        )
    ]
    if len(matches) != 1:
        raise PermissionError(
            "Requested run is outside the frozen loss-interaction matrix"
        )
    return matches[0]


def _loader(
    dataset: BraTSSliceDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    sampler: PatientGroupedSampler | None,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
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
        if batch_index < consumed:
            continue
        yield batch_index, batch


def _metadata(
    *,
    spec: NativeRunSpec,
    runner_config_path: Path,
    matrix_path: Path,
    model_config_path: Path,
    loss_catalog_path: Path,
    fold_path: Path,
    canonical_manifest_path: Path,
    preprocessing_path: Path,
    evaluation_path: Path,
    git_commit: str,
    extra_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    scientific_role = (
        "development_only_loss_selection"
        if spec.stage == "loss_screen"
        else (
            "reportable_compute_matched_development_cross_validation"
            if spec.stage == "main_compute_matched"
            else (
                "reportable_architecture_by_loss_sensitivity"
                if spec.stage == "loss_interaction"
                else "reportable_development_cross_validation"
            )
        )
    )
    return {
        "schema_version": 1,
        "run_id": spec.run_id,
        "run_spec": asdict(spec),
        "run_spec_sha256": spec.sha256,
        "stage": spec.stage,
        "scientific_role": scientific_role,
        "git_commit": git_commit,
        "repository_dirty_at_start": False,
        "model_id": spec.model_id,
        "fold": spec.fold,
        "seed": spec.seed,
        "loss": spec.loss_name,
        "hardware": {
            "backend": "mps",
            "device": "Apple M1 Max",
            "memory_terminology": (
                "MPS framework-reported allocated unified memory"
            ),
        },
        "hashes": {
            "runner_config": file_digest(runner_config_path),
            "model_matrix": file_digest(matrix_path),
            "model_config": file_digest(model_config_path),
            "loss_catalog": file_digest(loss_catalog_path),
            "fold_manifest": file_digest(fold_path),
            "canonical_manifest": file_digest(canonical_manifest_path),
            "preprocessing": file_digest(preprocessing_path),
            "evaluation": file_digest(evaluation_path),
            **dict(extra_hashes or {}),
        },
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
        "status": "running",
    }


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


def _improved(
    *,
    metric: float,
    validation_loss: float,
    progress: dict[str, Any],
) -> bool:
    best_metric = progress["best_metric"]
    if best_metric is None or metric > float(best_metric):
        return True
    if not np.isclose(metric, float(best_metric), rtol=0.0, atol=1e-12):
        return False
    best_loss = progress["best_validation_loss"]
    return best_loss is None or validation_loss < float(best_loss)


def _context_offsets(model_id: str) -> tuple[int, ...]:
    return (-2, -1, 0, 1, 2) if model_id == "unet_2p5d_k5" else (0,)


def run_native_development(
    *,
    runner_config_path: Path,
    spec: NativeRunSpec,
    dataset_root: Path,
    allow_reportable_development_training: bool,
    resume: bool,
) -> Path:
    """Run or resume one frozen native development job."""
    if not allow_reportable_development_training:
        raise PermissionError(
            "Development training requires "
            "--allow-reportable-development-training"
        )
    config = _load_yaml(runner_config_path)
    if str(config["status"]) != "frozen_before_first_reportable_development_run":
        raise PermissionError("Native runner config is not frozen")
    selected_loss_path = Path("configs/q1q2_v2/selected_loss.yaml")
    allowed_specs = (
        loss_screen_specs(runner_config_path)
        if spec.stage == "loss_screen"
        else (
            main_compute_matched_specs(
                runner_config_path,
                selected_loss_path,
            )
            if spec.stage == "main_compute_matched"
            else (
                loss_interaction_specs(
                    runner_config_path,
                    selected_loss_path,
                )
                if spec.stage == "loss_interaction"
                else main_convergence_specs(
                    runner_config_path,
                    selected_loss_path,
                )
            )
        )
    )
    if spec.sha256 not in {allowed.sha256 for allowed in allowed_specs}:
        raise PermissionError("Run specification is outside the frozen stage matrix")
    guards = cast(dict[str, Any], config["guards"])
    if (
        bool(guards["allow_external_data"])
        or bool(guards["allow_legacy_internal_test"])
        or bool(guards["allow_silent_seed_replacement"])
    ):
        raise PermissionError("Prohibited native-runner access/conduct is enabled")
    if not (
        bool(guards["require_best_checkpoint"])
        and bool(guards["require_terminal_checkpoint"])
    ):
        raise PermissionError("Best and terminal checkpoints must both be required")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Native M1 development runs require available MPS")
    hardware = cast(dict[str, Any], config["hardware"])
    git_commit, git_dirty = _git_state()
    if bool(hardware["require_clean_git"]) and git_dirty:
        raise RuntimeError("Reportable development training requires a clean worktree")
    data = cast(dict[str, Any], config["data"])
    models = cast(dict[str, Any], config["models"])
    losses = cast(dict[str, Any], config["losses"])
    training = cast(dict[str, Any], config["training"])
    artifacts = cast(dict[str, Any], config["artifacts"])
    matrix_path = Path(str(models["matrix"]))
    model_entry = _model_entry(matrix_path, spec.model_id)
    if spec.seed not in [int(value) for value in model_entry["seeds"]]:
        raise PermissionError("Run seed is outside the model matrix")
    model_config_path = Path(str(model_entry["config"]))
    loss_catalog_path = Path(str(losses["catalog"]))
    fold_path = Path(str(data["fold_pattern"]).format(fold=spec.fold))
    canonical_manifest_path = Path(str(data["canonical_manifest"]))
    preprocessing_path = Path(str(data["preprocessing"]))
    evaluation_path = Path(str(data["evaluation"]))
    artifact_root = Path(str(artifacts["root"])).resolve()
    output_dir = artifact_root / spec.run_id
    assert_output_paths_safe([output_dir], [dataset_root.resolve()])
    if output_dir.exists() and not resume:
        raise FileExistsError(f"Run already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=resume)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    extra_hashes: dict[str, str] = {}
    if spec.stage in {
        "main_convergence",
        "main_compute_matched",
        "loss_interaction",
    }:
        selected = _load_yaml(selected_loss_path)
        selection_artifact = Path(str(selected["selection_artifact"]))
        extra_hashes = {
            "selected_loss_config": file_digest(selected_loss_path),
            "loss_selection_artifact": file_digest(selection_artifact),
        }
    metadata = _metadata(
        spec=spec,
        runner_config_path=runner_config_path,
        matrix_path=matrix_path,
        model_config_path=model_config_path,
        loss_catalog_path=loss_catalog_path,
        fold_path=fold_path,
        canonical_manifest_path=canonical_manifest_path,
        preprocessing_path=preprocessing_path,
        evaluation_path=evaluation_path,
        git_commit=git_commit,
        extra_hashes=extra_hashes,
    )
    metadata_path = output_dir / "metadata.json"
    progress_path = output_dir / "progress.json"
    if resume:
        stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if stored_metadata["run_spec_sha256"] != spec.sha256:
            raise ValueError("Resume run-spec hash differs")
        if stored_metadata["hashes"] != metadata["hashes"]:
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
    seed_everything(spec.seed)
    device = torch.device("mps")
    preprocessing = load_preprocessing_config(preprocessing_path)
    context_offsets = _context_offsets(spec.model_id)
    train_dataset = build_cv_fold_dataset(
        fold_path,
        canonical_manifest_path,
        "train",
        dataset_root,
        preprocessing,
        seed=spec.seed,
        context_offsets=context_offsets,
    )
    validation_dataset = build_cv_fold_dataset(
        fold_path,
        canonical_manifest_path,
        "validation",
        dataset_root,
        preprocessing,
        seed=spec.seed,
        context_offsets=context_offsets,
    )
    sampler = PatientGroupedSampler(
        patient_count=len(train_dataset.manifest),
        samples_per_patient=(
            preprocessing.training_sampling.samples_per_patient_per_epoch
        ),
        seed=spec.seed,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=int(data["training_batch_size"]),
        workers=int(data["training_workers"]),
        seed=spec.seed,
        sampler=sampler,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=int(data["validation_batch_size"]),
        workers=int(data["validation_workers"]),
        seed=spec.seed + 1,
        sampler=None,
    )
    model_config = load_model_config(model_config_path)
    model = model_from_config(model_config)
    loss_function = _loss(loss_catalog_path, spec.loss_name)
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
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        mixed_precision=bool(training["mixed_precision"]),
    )
    recovery_path = checkpoint_dir / "recovery.pt"
    if resume:
        if not recovery_path.is_file():
            raise FileNotFoundError(
                "Resume requested but recovery checkpoint is absent"
            )
        engine.state, checkpoint_metadata = load_checkpoint(
            recovery_path,
            model=engine.model,
            optimizer=engine.optimizer,
            scaler=engine.scaler,
            scheduler=scheduler,
            map_location=device,
        )
        if checkpoint_metadata["run_spec_sha256"] != spec.sha256:
            raise ValueError("Resume checkpoint run-spec hash differs")
    evaluator = CentralEvaluator(load_evaluation_config(evaluation_path))
    tracker = ResourceTracker(device)
    cumulative_before_session = float(progress["cumulative_elapsed_seconds"])
    validation_frequency = int(
        training["validation_frequency_optimizer_steps"]
    )
    minimum_delta = float(training["early_stopping_minimum_delta"])
    patience = int(training["early_stopping_patience_validation_checks"])
    stage_config = cast(
        dict[str, Any],
        cast(dict[str, Any], config["stages"])[spec.stage],
    )
    minimum_steps_before_early_stopping = int(
        stage_config.get("minimum_optimizer_steps_before_early_stopping", 0)
    )
    milestone_steps = {
        int(value)
        for value in cast(
            list[Any],
            stage_config.get("budget_sensitivity_checkpoint_steps", []),
        )
    }
    maximum_accelerator_hours = (
        float(stage_config["maximum_accelerator_hours"])
        if spec.stage == "main_compute_matched"
        else None
    )
    progress.setdefault("budget_sensitivity_checkpoints", {})
    stop = False
    try:
        while engine.state.global_step < spec.maximum_optimizer_steps and not stop:
            epoch = engine.state.epoch
            train_dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            consumed = engine.state.batches_consumed_in_epoch
            for batch_index, batch in _batches_from_resume(train_loader, consumed):
                training_loss = engine.train_step(batch["image"], batch["label"])
                scheduler.step()
                engine.state.batches_consumed_in_epoch = batch_index + 1
                if (
                    maximum_accelerator_hours is not None
                    and (
                        cumulative_before_session + tracker.elapsed_seconds()
                    )
                    / 3600.0
                    >= maximum_accelerator_hours
                ):
                    progress["stop_reason"] = "compute_budget_accelerator_hours"
                    progress["cumulative_elapsed_seconds"] = (
                        cumulative_before_session + tracker.elapsed_seconds()
                    )
                    stop = True
                    atomic_write_json(progress_path, progress)
                    break
                at_validation = (
                    engine.state.global_step % validation_frequency == 0
                    or engine.state.global_step >= spec.maximum_optimizer_steps
                )
                if not at_validation:
                    continue
                selection = validate_selection_dice(
                    engine.model,
                    validation_loader,
                    device=device,
                    config=evaluator.config,
                    loss_function=loss_function,
                )
                if selection.validation_loss is None:
                    raise RuntimeError("Validation loss was not computed")
                metric = selection.mean_regional_dice
                validation_loss = selection.validation_loss
                is_best = _improved(
                    metric=metric,
                    validation_loss=validation_loss,
                    progress=progress,
                )
                if is_best:
                    progress["best_metric"] = metric
                    progress["best_validation_loss"] = validation_loss
                    progress["best_step"] = engine.state.global_step
                    atomic_write_csv(
                        output_dir / "best_validation_per_patient.csv",
                        list(selection.patient_rows),
                    )
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model=engine.model,
                        optimizer=engine.optimizer,
                        scaler=engine.scaler,
                        scheduler=scheduler,
                        state=engine.state,
                        metadata=metadata,
                    )
                if engine.state.global_step in milestone_steps:
                    milestone_path = (
                        checkpoint_dir
                        / f"budget_step_{engine.state.global_step}.pt"
                    )
                    save_checkpoint(
                        milestone_path,
                        model=engine.model,
                        optimizer=engine.optimizer,
                        scaler=engine.scaler,
                        scheduler=scheduler,
                        state=engine.state,
                        metadata=metadata,
                    )
                    milestone_metrics = (
                        output_dir
                        / (
                            "validation_step_"
                            f"{engine.state.global_step}_per_patient.csv"
                        )
                    )
                    atomic_write_csv(
                        milestone_metrics,
                        list(selection.patient_rows),
                    )
                    progress["budget_sensitivity_checkpoints"][
                        str(engine.state.global_step)
                    ] = {
                        "checkpoint": milestone_path.as_posix(),
                        "checkpoint_sha256": file_digest(milestone_path),
                        "patient_metrics": milestone_metrics.as_posix(),
                        "patient_metrics_sha256": file_digest(
                            milestone_metrics
                        ),
                    }
                reference = progress["early_stopping_reference_metric"]
                if reference is None or metric > float(reference) + minimum_delta:
                    progress["early_stopping_reference_metric"] = metric
                    progress[
                        "validation_checks_without_minimum_improvement"
                    ] = 0
                else:
                    progress[
                        "validation_checks_without_minimum_improvement"
                    ] = (
                        int(
                            progress[
                                "validation_checks_without_minimum_improvement"
                            ]
                        )
                        + 1
                    )
                progress["completed_validation_checks"] = (
                    int(progress["completed_validation_checks"]) + 1
                )
                append_jsonl(
                    output_dir / "metrics_per_validation.jsonl",
                    {
                        "record_type": "validation_check",
                        "optimizer_step": engine.state.global_step,
                        "epoch": engine.state.epoch,
                        "batches_consumed_in_epoch": (
                            engine.state.batches_consumed_in_epoch
                        ),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "last_training_loss": training_loss,
                        "validation_patient_mean_regional_dice": metric,
                        "validation_loss": validation_loss,
                        "is_best": is_best,
                        "checks_without_minimum_improvement": progress[
                            "validation_checks_without_minimum_improvement"
                        ],
                    },
                )
                save_checkpoint(
                    recovery_path,
                    model=engine.model,
                    optimizer=engine.optimizer,
                    scaler=engine.scaler,
                    scheduler=scheduler,
                    state=engine.state,
                    metadata=metadata,
                )
                if (
                    spec.stage != "main_compute_matched"
                    and engine.state.global_step
                    >= minimum_steps_before_early_stopping
                    and (
                        int(
                            progress[
                                "validation_checks_without_minimum_improvement"
                            ]
                        )
                        >= patience
                    )
                ):
                    progress["stop_reason"] = "early_stopping_patience"
                    stop = True
                elif engine.state.global_step >= spec.maximum_optimizer_steps:
                    progress["stop_reason"] = "maximum_optimizer_steps"
                    stop = True
                progress["cumulative_elapsed_seconds"] = (
                    cumulative_before_session + tracker.elapsed_seconds()
                )
                atomic_write_json(progress_path, progress)
                if stop:
                    break
            if not stop:
                engine.state.epoch += 1
                engine.state.batches_consumed_in_epoch = 0
        save_checkpoint(
            checkpoint_dir / "terminal.pt",
            model=engine.model,
            optimizer=engine.optimizer,
            scaler=engine.scaler,
            scheduler=scheduler,
            state=engine.state,
            metadata=metadata,
        )
        acceptance = {
            "best_checkpoint": (checkpoint_dir / "best.pt").is_file(),
            "terminal_checkpoint": (checkpoint_dir / "terminal.pt").is_file(),
            "validation_checks": int(progress["completed_validation_checks"]) > 0,
            "stop_reason": progress["stop_reason"]
            in (
                {"compute_budget_accelerator_hours"}
                if spec.stage == "main_compute_matched"
                else {"early_stopping_patience", "maximum_optimizer_steps"}
            ),
            "budget_sensitivity_checkpoints": (
                set(progress["budget_sensitivity_checkpoints"])
                == {str(value) for value in milestone_steps}
                if spec.stage == "main_convergence"
                else True
            ),
        }
        progress["status"] = (
            "completed" if all(acceptance.values()) else "invalid"
        )
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
        resource["cumulative_elapsed_seconds_before_session"] = float(
            cumulative_before_session
        )
        resource["cumulative_elapsed_seconds_including_session"] = (
            cumulative_before_session + float(resource["elapsed_seconds"])
        )
        resource["completed_optimizer_steps"] = engine.state.global_step
        resource["completed_validation_checks"] = int(
            progress.get("completed_validation_checks", 0)
        )
        resource["checkpoint_size_bytes"] = (
            (checkpoint_dir / "terminal.pt").stat().st_size
            if (checkpoint_dir / "terminal.pt").is_file()
            else None
        )
        atomic_write_json(output_dir / "resource_profile.json", resource)
        atomic_write_json(progress_path, progress)
        atomic_write_json(metadata_path, metadata)
    return output_dir


__all__ = [
    "NativeRunSpec",
    "loss_interaction_specs",
    "loss_screen_specs",
    "main_compute_matched_specs",
    "main_convergence_specs",
    "resolve_loss_interaction_spec",
    "resolve_loss_screen_spec",
    "resolve_main_compute_matched_spec",
    "resolve_main_convergence_spec",
    "run_native_development",
]
