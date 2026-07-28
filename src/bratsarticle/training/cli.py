"""Guarded full-training CLI for the Standard 2D U-Net baseline."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bratsarticle.data.dataset import (
    BraTSSliceDataset,
    build_development_dataset,
)
from bratsarticle.data.preprocessing import load_preprocessing_config
from bratsarticle.models import StandardUNet2D
from bratsarticle.training.checkpoint import load_checkpoint, save_checkpoint
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.losses import DiceCrossEntropyLoss
from bratsarticle.training.reproducibility import (
    collect_run_metadata,
    seed_dataloader_worker,
    seed_everything,
)
from bratsarticle.training.validation import validate_full_volumes
from bratsarticle.utils.serialization import append_jsonl, atomic_write_json
from evaluation import (
    CentralEvaluator,
    load_evaluation_config,
    summarize_patient_metrics,
)


def resolve_device(preference: str) -> torch.device:
    """Resolve `auto`, `cpu`, or `cuda` with explicit availability checks."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if preference == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if preference not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported training device: {preference}")
    return torch.device(preference)


def assert_training_authorized(
    *,
    allow_full_training: bool,
    smoke_steps: int,
    device: torch.device,
    require_cuda_for_full_training: bool,
) -> None:
    """Guard expensive training separately from bounded smoke execution."""
    if smoke_steps < 0:
        raise ValueError("smoke_steps cannot be negative")
    if smoke_steps > 0:
        return
    if not allow_full_training:
        raise PermissionError("Full training requires --allow-full-training")
    if require_cuda_for_full_training and device.type != "cuda":
        raise RuntimeError(
            "Full training requires a CUDA host under the current protocol"
        )


def _split_hashes(split_dir: Path) -> dict[str, str]:
    metadata = json.loads(
        (split_dir / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "train": str(metadata["manifest_sha256"]["train"]),
        "validation": str(metadata["manifest_sha256"]["validation"]),
    }


def _run_directory(config: DictConfig, run_id: str | None) -> Path:
    resolved_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path(str(config.run.artifact_root)).resolve() / resolved_id


def _loader(
    dataset: BraTSSliceDataset,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _selection_metric(rows: list[dict[str, Any]]) -> float:
    summary = summarize_patient_metrics(rows)
    matches = [
        row
        for row in summary
        if row["evaluation_stage"] == "raw" and row["metric"] == "mean_regional_dice"
    ]
    if len(matches) != 1:
        raise RuntimeError("Expected one raw mean_regional_dice summary")
    return float(matches[0]["mean_finite"])


def run_training(
    *,
    config_path: Path,
    allow_full_training: bool,
    smoke_steps: int,
    run_id: str | None,
    resume: Path | None,
) -> Path:
    """Execute guarded baseline training and return the artifact directory."""
    config = cast(DictConfig, OmegaConf.load(config_path))
    OmegaConf.resolve(config)
    seed = int(config.run.seed)
    seed_everything(seed)
    device = resolve_device(str(config.run.device))
    assert_training_authorized(
        allow_full_training=allow_full_training,
        smoke_steps=smoke_steps,
        device=device,
        require_cuda_for_full_training=bool(config.run.require_cuda_for_full_training),
    )
    split_dir = Path(str(config.data.split_dir)).resolve()
    preprocessing = load_preprocessing_config(
        Path(str(config.data.preprocessing_config))
    )
    dataset_root = Path(str(config.data.brats2020_root))
    train_dataset = build_development_dataset(
        split_dir,
        "train",
        dataset_root,
        preprocessing,
        seed=seed,
    )
    validation_dataset = build_development_dataset(
        split_dir,
        "validation",
        dataset_root,
        preprocessing,
        seed=seed,
    )
    train_loader = _loader(
        train_dataset,
        batch_size=int(config.data.batch_size),
        num_workers=int(config.data.num_workers),
        seed=seed,
        shuffle=True,
    )
    validation_loader = _loader(
        validation_dataset,
        batch_size=int(config.data.validation_batch_size),
        num_workers=int(config.data.num_workers),
        seed=seed + 1,
        shuffle=False,
    )
    model = StandardUNet2D(
        input_channels=int(config.model.input_channels),
        output_channels=int(config.model.output_channels),
        base_channels=int(config.model.base_channels),
        depth=int(config.model.depth),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.optimization.learning_rate),
        weight_decay=float(config.optimization.weight_decay),
    )
    loss_function = DiceCrossEntropyLoss(
        cross_entropy_weight=float(config.optimization.loss.cross_entropy_weight),
        dice_weight=float(config.optimization.loss.dice_weight),
        smooth=float(config.optimization.loss.training_dice_smooth),
    )
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        mixed_precision=bool(config.optimization.mixed_precision),
    )
    output_dir = (
        resume.resolve().parent
        if resume is not None
        else _run_directory(config, run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = collect_run_metadata(
        config_path=config_path,
        split_hashes=_split_hashes(split_dir),
        seed=seed,
        device=device,
        mixed_precision=bool(config.optimization.mixed_precision),
        run_kind=("unet2d_baseline_smoke" if smoke_steps else "unet2d_baseline_full"),
    )
    if resume is not None:
        engine.state, checkpoint_metadata = load_checkpoint(
            resume,
            model=engine.model,
            optimizer=engine.optimizer,
            scaler=engine.scaler,
            map_location=device,
        )
        if checkpoint_metadata["config_sha256"] != metadata["config_sha256"]:
            raise ValueError("Resume checkpoint configuration hash differs")
        metadata["resumed_from"] = resume.as_posix()
    atomic_write_json(output_dir / "run_metadata.json", metadata)

    evaluator = CentralEvaluator(
        load_evaluation_config(Path(str(config.data.evaluation_config)))
    )
    best_metric = -float("inf")
    epochs_without_improvement = 0
    maximum_epochs = int(config.optimization.epochs)
    stopped_for_smoke = False
    for epoch in range(engine.state.epoch, maximum_epochs):
        train_dataset.set_epoch(epoch)
        losses: list[float] = []
        for batch in train_loader:
            losses.append(engine.train_step(batch["image"], batch["label"]))
            if smoke_steps and engine.state.global_step >= smoke_steps:
                stopped_for_smoke = True
                break
        engine.state.epoch = epoch + 1
        epoch_record: dict[str, Any] = {
            "epoch": epoch,
            "global_step": engine.state.global_step,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_last": losses[-1],
        }
        if not smoke_steps:
            patient_rows = validate_full_volumes(
                engine.model,
                validation_loader,
                device=device,
                evaluator=evaluator,
            )
            for row in patient_rows:
                append_jsonl(
                    output_dir / "validation_patient_metrics.jsonl",
                    {"epoch": epoch, **row},
                )
            metric = _selection_metric(patient_rows)
            epoch_record["validation_mean_regional_dice"] = metric
            if metric > best_metric:
                best_metric = metric
                epochs_without_improvement = 0
                save_checkpoint(
                    output_dir / "best.pt",
                    model=engine.model,
                    optimizer=engine.optimizer,
                    scaler=engine.scaler,
                    state=engine.state,
                    metadata=metadata,
                )
            else:
                epochs_without_improvement += 1
        append_jsonl(output_dir / "metrics_per_epoch.jsonl", epoch_record)
        save_checkpoint(
            output_dir / "last.pt",
            model=engine.model,
            optimizer=engine.optimizer,
            scaler=engine.scaler,
            state=engine.state,
            metadata=metadata,
        )
        if stopped_for_smoke or (
            not smoke_steps
            and epochs_without_improvement
            >= int(config.optimization.early_stopping_patience)
        ):
            break
    metadata["status"] = "completed"
    metadata["global_step"] = engine.state.global_step
    metadata["completed_epochs"] = engine.state.epoch
    metadata["best_validation_mean_regional_dice"] = (
        best_metric if np.isfinite(best_metric) else None
    )
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    return output_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the guarded baseline-training parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/training/unet2d_baseline.yaml"),
    )
    parser.add_argument("--allow-full-training", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> int:
    """Command-line entry point."""
    arguments = build_argument_parser().parse_args()
    try:
        output_dir = run_training(
            config_path=arguments.config,
            allow_full_training=arguments.allow_full_training,
            smoke_steps=arguments.smoke_steps,
            run_id=arguments.run_id,
            resume=arguments.resume,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "unet2d_training_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"artifact_directory": output_dir.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
