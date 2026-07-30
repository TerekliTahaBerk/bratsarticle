"""Ordered nonreportable diagnostics for the v2 M1 training path."""

from __future__ import annotations

import subprocess
import time
import traceback
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from bratsarticle.data.dataset import (
    BraTSSliceDataset,
    build_cv_fold_dataset,
)
from bratsarticle.data.preprocessing import (
    IntensityAugmentationConfig,
    SpatialAugmentationConfig,
    TrainingSamplingConfig,
    load_preprocessing_config,
)
from bratsarticle.experiments.pilot_runner import PatientGroupedSampler
from bratsarticle.experiments.registry import ResourceTracker
from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    model_from_config,
)
from bratsarticle.training.checkpoint import save_checkpoint
from bratsarticle.training.engine import TrainingEngine
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
from bratsarticle.training.validation import (
    validate_full_volumes,
    validate_selection_dice,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text
from evaluation import (
    CentralEvaluator,
    load_evaluation_config,
    summarize_patient_metrics,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], payload)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _loss(catalog_path: Path, name: str) -> ConfiguredSegmentationLoss:
    matches = [
        config for config in load_loss_catalog(catalog_path) if config.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one loss named {name}")
    return build_loss(matches[0])


def _metrics(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    *,
    device: torch.device,
    evaluator: CentralEvaluator,
    patient_id: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = cast(torch.Tensor, model(image.to(device, dtype=torch.float32)))
        prediction = class_indices_to_labels(torch.argmax(logits, dim=1)).cpu()
    prediction_volume = np.stack(
        [prediction[0].numpy(), prediction[0].numpy()],
        axis=2,
    )
    target_volume = np.stack(
        [label[0].cpu().numpy(), label[0].cpu().numpy()],
        axis=2,
    )
    row = evaluator.evaluate_batch(
        prediction_volume,
        target_volume,
        patient_ids=[patient_id],
        spacings_mm=[(1.0, 1.0, 1.0)],
    )[0]
    return {
        "mean_regional_dice": float(row["mean_regional_dice"]),
        "wt_dice": float(row["wt_dice"]),
        "tc_dice": float(row["tc_dice"]),
        "et_dice": float(row["et_dice"]),
    }


def _engine(
    *,
    model: nn.Module,
    loss_function: nn.Module,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
) -> TrainingEngine:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    return TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        mixed_precision=False,
    )


def _synthetic_batch(
    *,
    batch_size: int,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image_size < 32 or image_size % 16:
        raise ValueError("Synthetic image size must be >=32 and divisible by 16")
    label = torch.zeros((batch_size, image_size, image_size), dtype=torch.long)
    outer = slice(image_size // 8, 7 * image_size // 8)
    middle = slice(image_size // 4, 3 * image_size // 4)
    inner = slice(3 * image_size // 8, 5 * image_size // 8)
    label[:, outer, outer] = 2
    label[:, middle, middle] = 1
    label[:, inner, inner] = 4
    image = torch.zeros((batch_size, 4, image_size, image_size))
    image[:, 0] = (label > 0).float()
    image[:, 1] = ((label == 1) | (label == 4)).float()
    image[:, 2] = (label == 2).float()
    image[:, 3] = (label == 4).float()
    image = image + 0.01 * torch.randn_like(image)
    return image.to(device), label.to(device)


def _run_steps(
    engine: TrainingEngine,
    image: torch.Tensor,
    label: torch.Tensor,
    steps: int,
) -> list[float]:
    if steps < 1:
        raise ValueError("Diagnostic step count must be positive")
    return [engine.train_step(image, label) for _ in range(steps)]


def synthetic_overfit(
    config: dict[str, Any],
    *,
    device: torch.device,
    evaluator: CentralEvaluator,
) -> dict[str, Any]:
    """Overfit a deterministic nested-region phantom."""
    synthetic = cast(dict[str, Any], config["synthetic_overfit"])
    optimization = cast(dict[str, Any], config["optimization"])
    model_raw = cast(dict[str, Any], config["model"])
    loss_raw = cast(dict[str, Any], config["loss"])
    model = model_from_config(load_model_config(Path(str(model_raw["config"]))))
    loss_function = _loss(Path(str(loss_raw["catalog"])), str(loss_raw["name"]))
    engine = _engine(
        model=model,
        loss_function=loss_function,
        device=device,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    image, label = _synthetic_batch(
        batch_size=int(synthetic["batch_size"]),
        image_size=int(synthetic["image_size"]),
        device=device,
    )
    with torch.no_grad():
        initial_loss = float(loss_function(model(image), label))
    losses = _run_steps(engine, image, label, int(synthetic["steps"]))
    with torch.no_grad():
        final_loss = float(loss_function(model(image), label))
    final_metrics = _metrics(
        model,
        image[:1],
        label[:1],
        device=device,
        evaluator=evaluator,
        patient_id="synthetic_nested_regions",
    )
    acceptance_config = cast(dict[str, Any], synthetic["acceptance"])
    acceptance = {
        "loss_ratio": (
            final_loss / initial_loss
            <= float(acceptance_config["maximum_final_to_initial_loss_ratio"])
        ),
        "mean_regional_dice": (
            final_metrics["mean_regional_dice"]
            >= float(acceptance_config["minimum_mean_regional_dice"])
        ),
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "steps": int(synthetic["steps"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": final_loss / initial_loss,
        "loss_every_10_steps": losses[9::10],
        "final_metrics": final_metrics,
        "acceptance": acceptance,
    }


def one_patient_overfit(
    config: dict[str, Any],
    *,
    dataset_root: Path,
    device: torch.device,
    evaluator: CentralEvaluator,
) -> dict[str, Any]:
    """Overfit the maximum-ET slice of one authorized fold-training patient."""
    data = cast(dict[str, Any], config["data"])
    optimization = cast(dict[str, Any], config["optimization"])
    diagnostic = cast(dict[str, Any], config["one_patient_overfit"])
    model_raw = cast(dict[str, Any], config["model"])
    loss_raw = cast(dict[str, Any], config["loss"])
    preprocessing = load_preprocessing_config(Path(str(data["preprocessing"])))
    preprocessing = replace(
        preprocessing,
        training_sampling=TrainingSamplingConfig(
            tumor_probability=1.0,
            tumor_minimum_voxels_per_slice=1,
            samples_per_patient_per_epoch=1,
        ),
        spatial_augmentation=SpatialAugmentationConfig(enabled=False),
        intensity_augmentation=IntensityAugmentationConfig(enabled=False),
    )
    dataset = build_cv_fold_dataset(
        Path(str(data["fold_manifest"])),
        Path(str(data["canonical_manifest"])),
        "train",
        dataset_root,
        preprocessing,
        seed=int(optimization["seed"]),
    )
    volume = dataset.subject_volume(0)
    et_counts = np.count_nonzero(volume.label == 4, axis=(0, 1))
    slice_index = int(np.argmax(et_counts))
    image = torch.from_numpy(
        np.ascontiguousarray(volume.image[:, :, :, slice_index])
    ).unsqueeze(0)
    label = torch.from_numpy(
        np.ascontiguousarray(volume.label[:, :, slice_index], dtype=np.int64)
    ).unsqueeze(0)
    subject_id = str(dataset.manifest.iloc[0]["subject_id"])
    model = model_from_config(load_model_config(Path(str(model_raw["config"]))))
    loss_function = _loss(Path(str(loss_raw["catalog"])), str(loss_raw["name"]))
    engine = _engine(
        model=model,
        loss_function=loss_function,
        device=device,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    image_device = image.to(device)
    label_device = label.to(device)
    with torch.no_grad():
        initial_loss = float(loss_function(model(image_device), label_device))
    losses = _run_steps(
        engine,
        image_device,
        label_device,
        int(diagnostic["steps"]),
    )
    with torch.no_grad():
        final_loss = float(loss_function(model(image_device), label_device))
    final_metrics = _metrics(
        model,
        image,
        label,
        device=device,
        evaluator=evaluator,
        patient_id=subject_id,
    )
    acceptance_config = cast(dict[str, Any], diagnostic["acceptance"])
    acceptance = {
        "loss_ratio": (
            final_loss / initial_loss
            <= float(acceptance_config["maximum_final_to_initial_loss_ratio"])
        ),
        "mean_regional_dice": (
            final_metrics["mean_regional_dice"]
            >= float(acceptance_config["minimum_mean_regional_dice"])
        ),
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "subject_id": subject_id,
        "slice_index": slice_index,
        "steps": int(diagnostic["steps"]),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_to_initial_loss_ratio": final_loss / initial_loss,
        "loss_every_10_steps": losses[9::10],
        "final_metrics": final_metrics,
        "acceptance": acceptance,
    }


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


def _selection_metric(rows: list[dict[str, Any]]) -> float:
    matches = [
        row
        for row in summarize_patient_metrics(rows)
        if row["evaluation_stage"] == "raw"
        and row["metric"] == "mean_regional_dice"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one raw validation mean-regional-Dice summary")
    return float(matches[0]["mean_finite"])


def _take_steps(
    engine: TrainingEngine,
    batches: Iterable[dict[str, Any]],
    steps: int,
) -> list[float]:
    losses: list[float] = []
    for batch in batches:
        losses.append(engine.train_step(batch["image"], batch["label"]))
        if len(losses) >= steps:
            break
    if len(losses) != steps:
        raise RuntimeError("Training loader ended before the smoke step budget")
    return losses


def single_fold_smoke(
    config: dict[str, Any],
    *,
    dataset_root: Path,
    device: torch.device,
    evaluator: CentralEvaluator,
    artifact_root: Path,
) -> dict[str, Any]:
    """Exercise real fold training, full validation, and checkpoint writes."""
    data = cast(dict[str, Any], config["data"])
    optimization = cast(dict[str, Any], config["optimization"])
    diagnostic = cast(dict[str, Any], config["single_fold_smoke"])
    model_raw = cast(dict[str, Any], config["model"])
    loss_raw = cast(dict[str, Any], config["loss"])
    preprocessing = load_preprocessing_config(Path(str(data["preprocessing"])))
    seed = int(optimization["seed"])
    train_dataset = build_cv_fold_dataset(
        Path(str(data["fold_manifest"])),
        Path(str(data["canonical_manifest"])),
        "train",
        dataset_root,
        preprocessing,
        seed=seed,
    )
    validation_dataset = build_cv_fold_dataset(
        Path(str(data["fold_manifest"])),
        Path(str(data["canonical_manifest"])),
        "validation",
        dataset_root,
        preprocessing,
        seed=seed,
    )
    sampler = PatientGroupedSampler(
        patient_count=len(train_dataset.manifest),
        samples_per_patient=(
            preprocessing.training_sampling.samples_per_patient_per_epoch
        ),
        seed=seed,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=int(diagnostic["training_batch_size"]),
        workers=int(data["training_workers"]),
        seed=seed,
        sampler=sampler,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=int(diagnostic["validation_batch_size"]),
        workers=int(data["validation_workers"]),
        seed=seed + 1,
        sampler=None,
    )
    model = model_from_config(load_model_config(Path(str(model_raw["config"]))))
    loss_function = _loss(Path(str(loss_raw["catalog"])), str(loss_raw["name"]))
    engine = _engine(
        model=model,
        loss_function=loss_function,
        device=device,
        learning_rate=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    tracker = ResourceTracker(device)
    losses = _take_steps(
        engine,
        train_loader,
        int(diagnostic["optimizer_steps"]),
    )
    selection_started = time.perf_counter()
    selection_result = validate_selection_dice(
        engine.model,
        validation_loader,
        device=device,
        config=evaluator.config,
    )
    selection_seconds = time.perf_counter() - selection_started
    full_started = time.perf_counter()
    patient_rows = validate_full_volumes(
        engine.model,
        validation_loader,
        device=device,
        evaluator=evaluator,
    )
    full_evaluator_seconds = time.perf_counter() - full_started
    metric = _selection_metric(patient_rows)
    checkpoint_dir = artifact_root / "single_fold_smoke" / "checkpoints"
    metadata = {
        "run_kind": "q1q2_v2_nonreportable_single_fold_smoke",
        "scientific_use": "prohibited",
        "git_commit": _git_commit(),
        "fold": int(data["fold"]),
        "seed": seed,
        "model": str(model_raw["id"]),
        "model_config_sha256": file_digest(Path(str(model_raw["config"]))),
        "loss_catalog_sha256": file_digest(Path(str(loss_raw["catalog"]))),
        "fold_manifest_sha256": file_digest(Path(str(data["fold_manifest"]))),
        "canonical_manifest_sha256": file_digest(
            Path(str(data["canonical_manifest"]))
        ),
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
    }
    save_checkpoint(
        checkpoint_dir / "best.pt",
        model=engine.model,
        optimizer=engine.optimizer,
        scaler=engine.scaler,
        state=engine.state,
        metadata=metadata,
    )
    save_checkpoint(
        checkpoint_dir / "terminal.pt",
        model=engine.model,
        optimizer=engine.optimizer,
        scaler=engine.scaler,
        state=engine.state,
        metadata=metadata,
    )
    resource_profile = tracker.snapshot()
    acceptance = {
        "finite_training_loss": bool(np.isfinite(losses).all()),
        "all_validation_patients": (
            len(patient_rows) == len(validation_dataset.manifest)
        ),
        "selection_metric_parity": bool(
            np.isclose(
                selection_result.mean_regional_dice,
                metric,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "best_checkpoint": (checkpoint_dir / "best.pt").is_file(),
        "terminal_checkpoint": (checkpoint_dir / "terminal.pt").is_file(),
    }
    return {
        "status": "pass" if all(acceptance.values()) else "fail",
        "optimizer_steps": int(diagnostic["optimizer_steps"]),
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "validation_patient_count": len(patient_rows),
        "validation_patient_mean_regional_dice": metric,
        "fast_selection_patient_mean_regional_dice": (
            selection_result.mean_regional_dice
        ),
        "fast_selection_seconds": selection_seconds,
        "full_evaluator_seconds": full_evaluator_seconds,
        "parameter_count": count_trainable_parameters(model),
        "resource_profile": resource_profile,
        "metadata": metadata,
        "acceptance": acceptance,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Q1/Q2 v2 ordered M1 diagnostics",
        "",
        f"Overall status: **{payload['status'].upper()}**",
        "",
        (
            "These are pipeline diagnostics, not development estimates or "
            "manuscript results. External and legacy internal-test data were not used."
        ),
        "",
        "| Stage | Status |",
        "|---|---|",
    ]
    for name in ("synthetic_overfit", "one_patient_overfit", "single_fold_smoke"):
        stage = cast(dict[str, Any], payload["stages"].get(name, {}))
        lines.append(f"| {name} | {stage.get('status', 'not_run')} |")
    lines.extend(
        [
            "",
            (
                "A PASS permits implementation work to proceed. It does not pass "
                "Gate F, authorize external evaluation, or establish convergence."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_ordered_diagnostics(
    *,
    config_path: Path,
    dataset_root: Path,
    artifact_root: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    """Run diagnostics in the frozen order and persist every stage transition."""
    config = _load_yaml(config_path)
    if str(config["status"]) != "frozen_nonreportable_training_diagnostics":
        raise PermissionError("Diagnostic config is not frozen")
    guards = cast(dict[str, Any], config["guards"])
    if any(bool(value) for value in guards.values()):
        raise PermissionError("Every diagnostic result/data guard must be false")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Ordered M1 diagnostics require available MPS")
    device = torch.device("mps")
    optimization = cast(dict[str, Any], config["optimization"])
    seed_everything(int(optimization["seed"]))
    data = cast(dict[str, Any], config["data"])
    evaluator = CentralEvaluator(
        load_evaluation_config(Path(str(data["evaluation"])))
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "scientific_use": "prohibited",
        "config": config_path.as_posix(),
        "config_sha256": file_digest(config_path),
        "dataset_root_recorded": False,
        "raw_data_access": "authorized fold-training and fold-validation only",
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
        "stages": {},
    }
    atomic_write_json(output_json, payload)
    try:
        stages = cast(dict[str, Any], payload["stages"])
        stages["synthetic_overfit"] = synthetic_overfit(
            config,
            device=device,
            evaluator=evaluator,
        )
        atomic_write_json(output_json, payload)
        if stages["synthetic_overfit"]["status"] != "pass":
            raise RuntimeError("Synthetic overfit acceptance failed")
        stages["one_patient_overfit"] = one_patient_overfit(
            config,
            dataset_root=dataset_root,
            device=device,
            evaluator=evaluator,
        )
        atomic_write_json(output_json, payload)
        if stages["one_patient_overfit"]["status"] != "pass":
            raise RuntimeError("One-patient overfit acceptance failed")
        stages["single_fold_smoke"] = single_fold_smoke(
            config,
            dataset_root=dataset_root,
            device=device,
            evaluator=evaluator,
            artifact_root=artifact_root,
        )
        if stages["single_fold_smoke"]["status"] != "pass":
            raise RuntimeError("Single-fold smoke acceptance failed")
        payload["status"] = "pass"
    except Exception:
        payload["status"] = "fail"
        payload["error"] = traceback.format_exc()
        atomic_write_json(output_json, payload)
        atomic_write_text(output_markdown, _markdown(payload))
        raise
    atomic_write_json(output_json, payload)
    atomic_write_text(output_markdown, _markdown(payload))
    return payload


__all__ = [
    "one_patient_overfit",
    "run_ordered_diagnostics",
    "single_fold_smoke",
    "synthetic_overfit",
]
