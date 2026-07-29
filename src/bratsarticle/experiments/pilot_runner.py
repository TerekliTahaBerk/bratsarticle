"""Guarded single-arm Gate 8 pilot runner for an eligible CUDA host."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from bratsarticle.experiments.fairness import load_compute_matched_protocol
from bratsarticle.experiments.pilots import (
    PilotArm,
    PilotPlan,
    load_pilot_plan,
    pilot_preflight,
)
from bratsarticle.experiments.registry import (
    ExperimentRegistry,
    ResourceTracker,
    RunDescriptor,
)
from bratsarticle.models.configurable_unet import (
    count_trainable_parameters,
    load_model_config,
    model_from_config,
)
from bratsarticle.training.checkpoint import save_checkpoint
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.loss_catalog import build_loss, load_loss_catalog
from bratsarticle.training.reproducibility import (
    seed_dataloader_worker,
    seed_everything,
)
from bratsarticle.training.schedule import build_warmup_cosine_scheduler
from bratsarticle.utils.hashing import file_digest

if TYPE_CHECKING:
    from bratsarticle.data.dataset import BraTSSliceDataset


def _loader(
    dataset: BraTSSliceDataset,
    *,
    batch_size: int,
    workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _split_hashes(split_dir: Path) -> dict[str, str]:
    metadata = json.loads(
        (split_dir / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "train": str(metadata["manifest_sha256"]["train"]),
        "validation": str(metadata["manifest_sha256"]["validation"]),
    }


def _selection_metric(rows: list[dict[str, Any]]) -> float:
    from evaluation import summarize_patient_metrics

    matches = [
        row
        for row in summarize_patient_metrics(rows)
        if row["evaluation_stage"] == "raw" and row["metric"] == "mean_regional_dice"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one raw validation mean-regional-Dice summary")
    return float(matches[0]["mean_finite"])


def _arm(plan: PilotPlan, arm_id: str) -> PilotArm:
    matches = [candidate for candidate in plan.arms if candidate.arm_id == arm_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous pilot arm: {arm_id}")
    return matches[0]


def _run_id(arm: PilotArm, supplied: str | None) -> str:
    if supplied:
        return supplied
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"gate8_{arm.arm_id}_s{arm.seed}_{timestamp}"


def _loss(plan_arm: PilotArm) -> torch.nn.Module:
    matches = [
        config
        for config in load_loss_catalog(Path("configs/losses/catalog.yaml"))
        if config.name == plan_arm.loss_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one loss config for {plan_arm.loss_name}")
    return build_loss(matches[0])


def _budget_reached(
    *,
    step: int,
    plan: PilotPlan,
    tracker: ResourceTracker,
) -> str | None:
    if step >= plan.maximum_optimizer_steps:
        return "maximum_optimizer_steps"
    if tracker.elapsed_seconds() / 3600.0 >= plan.maximum_gpu_hours:
        return "maximum_gpu_hours"
    return None


def _train_until_validation(
    *,
    engine: TrainingEngine,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_loader: Iterable[dict[str, Any]],
    plan: PilotPlan,
    tracker: ResourceTracker,
) -> tuple[list[float], str | None]:
    losses: list[float] = []
    stop_reason: str | None = None
    for batch in train_loader:
        losses.append(engine.train_step(batch["image"], batch["label"]))
        scheduler.step()
        stop_reason = _budget_reached(
            step=engine.state.global_step,
            plan=plan,
            tracker=tracker,
        )
        if (
            stop_reason is not None
            or engine.state.global_step % plan.validation_frequency_optimizer_steps == 0
        ):
            break
    return losses, stop_reason


def run_pilot_arm(
    *,
    plan_path: Path,
    arm_id: str,
    allow_pilot_training: bool,
    run_id: str | None = None,
) -> Path:
    """Run one guarded development pilot and return its registry directory."""
    if not allow_pilot_training:
        raise PermissionError("Pilot training requires --allow-pilot-training")
    plan = load_pilot_plan(plan_path)
    preflight = pilot_preflight(plan)
    if not preflight["eligible"]:
        failed = [
            name for name, passed in preflight["checks"].items() if not bool(passed)
        ]
        raise RuntimeError(f"Gate 8 pilot preflight failed: {failed}")
    from bratsarticle.data.dataset import build_development_dataset
    from bratsarticle.data.preprocessing import load_preprocessing_config
    from bratsarticle.training.validation import validate_full_volumes
    from evaluation import CentralEvaluator, load_evaluation_config

    arm = _arm(plan, arm_id)
    fairness = load_compute_matched_protocol(plan.fairness_protocol_path)
    dataset_root = Path(os.environ["BRATS2020_ROOT"]).expanduser().resolve()
    seed_everything(arm.seed)
    preprocessing = load_preprocessing_config(plan.preprocessing_config_path)
    train_dataset = build_development_dataset(
        plan.split_dir,
        "train",
        dataset_root,
        preprocessing,
        seed=arm.seed,
    )
    validation_dataset = build_development_dataset(
        plan.split_dir,
        "validation",
        dataset_root,
        preprocessing,
        seed=arm.seed,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=fairness.batch_size,
        workers=plan.training_workers,
        seed=arm.seed,
        shuffle=True,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=fairness.batch_size,
        workers=plan.validation_workers,
        seed=arm.seed + 1,
        shuffle=False,
    )
    model_config = load_model_config(arm.model_config_path)
    model = model_from_config(model_config)
    parameter_count = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=plan.learning_rate,
        weight_decay=plan.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=plan.warmup_optimizer_steps,
        total_steps=plan.maximum_optimizer_steps,
        minimum_fraction=plan.minimum_learning_rate_fraction,
    )
    device = torch.device("cuda")
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=_loss(arm),
        device=device,
        mixed_precision=plan.mixed_precision,
    )
    registry = ExperimentRegistry(
        artifact_root=plan.artifact_root,
        descriptor=RunDescriptor(
            run_id=_run_id(arm, run_id),
            seed=arm.seed,
            model=model_config.name,
            loss=arm.loss_name,
            optimizer=plan.optimizer,
            scheduler=plan.scheduler,
            parameter_count=parameter_count,
            input_specification=(1, *fairness.input_shape),
            data_manifest_path=plan.canonical_manifest_path,
            split_hashes=_split_hashes(plan.split_dir),
            tags={
                "gate": 8,
                "pilot_arm_id": arm.arm_id,
                "pilot_screen": arm.screen,
                "pilot_config_sha256": file_digest(plan_path),
            },
        ),
        config_path=plan_path,
        raw_data_roots=[dataset_root],
    )
    evaluator = CentralEvaluator(load_evaluation_config(plan.evaluation_config_path))
    tracker = ResourceTracker(device)
    best_metric = -float("inf")
    validation_checks = 0
    stop_reason: str | None = None
    epoch = 0
    recent_losses: list[float] = []
    try:
        while stop_reason is None:
            train_dataset.set_epoch(epoch)
            losses, stop_reason = _train_until_validation(
                engine=engine,
                scheduler=scheduler,
                train_loader=train_loader,
                plan=plan,
                tracker=tracker,
            )
            recent_losses.extend(losses)
            if not losses:
                raise RuntimeError("Training DataLoader yielded no batches")
            at_validation_step = (
                engine.state.global_step % plan.validation_frequency_optimizer_steps
                == 0
            )
            if at_validation_step:
                patient_rows = validate_full_volumes(
                    engine.model,
                    validation_loader,
                    device=device,
                    evaluator=evaluator,
                )
                validation_checks += 1
                metric = _selection_metric(patient_rows)
                improved = metric > best_metric
                registry.log_epoch(
                    {
                        "record_type": "validation_check",
                        "epoch": epoch,
                        "optimizer_step": engine.state.global_step,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "train_loss_mean_since_validation": float(
                            np.mean(recent_losses)
                        ),
                        "validation_patient_mean_regional_dice": metric,
                        "is_best": improved,
                    }
                )
                recent_losses = []
                if improved:
                    best_metric = metric
                    registry.write_validation_cases(patient_rows)
                    save_checkpoint(
                        registry.checkpoint_directory / "best.pt",
                        model=engine.model,
                        optimizer=engine.optimizer,
                        scaler=engine.scaler,
                        scheduler=scheduler,
                        state=engine.state,
                        metadata=dict(registry.metadata),
                    )
            save_checkpoint(
                registry.checkpoint_directory / "last.pt",
                model=engine.model,
                optimizer=engine.optimizer,
                scaler=engine.scaler,
                scheduler=scheduler,
                state=engine.state,
                metadata=dict(registry.metadata),
            )
            epoch += 1
            engine.state.epoch = epoch
        resource_profile = tracker.snapshot()
        resource_profile.update(
            {
                "completed_optimizer_steps": engine.state.global_step,
                "completed_validation_checks": validation_checks,
                "budget_stop_reason": stop_reason,
                "model_checkpoint_size_bytes": (
                    registry.checkpoint_directory / "last.pt"
                )
                .stat()
                .st_size,
                "input_specification": [1, *fairness.input_shape],
                "mixed_precision": plan.mixed_precision,
            }
        )
        valid = validation_checks >= plan.minimum_completed_validation_checks
        registry.finalize(
            status="completed" if valid else "invalid",
            resource_profile=resource_profile,
            best_validation_checkpoint=(
                "checkpoints/best.pt" if np.isfinite(best_metric) else None
            ),
        )
    except Exception:
        resource_profile = tracker.snapshot()
        resource_profile.update(
            {
                "completed_optimizer_steps": engine.state.global_step,
                "completed_validation_checks": validation_checks,
                "budget_stop_reason": "exception",
                "mixed_precision": plan.mixed_precision,
            }
        )
        registry.finalize(
            status="failed",
            resource_profile=resource_profile,
            error_trace=traceback.format_exc(),
        )
        raise
    return registry.run_directory
