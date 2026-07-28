"""Run bounded synthetic overfit and checkpoint diagnostics for Gate 6."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from bratsarticle.models import ConfigurableUNet2D
from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.training.checkpoint import load_checkpoint, save_checkpoint
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.losses import DiceCrossEntropyLoss
from bratsarticle.training.reproducibility import (
    collect_run_metadata,
    seed_everything,
)
from bratsarticle.utils.serialization import atomic_write_json


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    image = torch.randn((1, 4, 24, 24), generator=generator)
    label = torch.zeros((1, 24, 24), dtype=torch.long)
    label[:, 3:21, 3:21] = 2
    label[:, 7:17, 7:17] = 1
    label[:, 10:14, 10:14] = 4
    return image, label


def _diagnostic_config(config_path: Path) -> Any:
    return replace(
        load_model_config(config_path),
        base_channels=2,
        depth=1,
        batch_normalization=False,
        dropout_probability=0.0,
        res_kernel_sizes=(3,),
        wc_kernel_size=3,
    )


def _run_model(
    config_path: Path,
    *,
    checkpoint_dir: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    seed_everything(seed)
    config = _diagnostic_config(config_path)
    model = ConfigurableUNet2D(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    image, label = _batch()
    losses = [engine.train_step(image, label) for _ in range(steps)]
    checkpoint_path = checkpoint_dir / f"{config_path.stem}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=engine.scaler,
        state=engine.state,
        metadata={"architecture": config_path.stem},
    )

    restored_model = ConfigurableUNet2D(config)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.02)
    restored_engine = TrainingEngine(
        model=restored_model,
        optimizer=restored_optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    restored_state, metadata = load_checkpoint(
        checkpoint_path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=restored_engine.scaler,
        map_location=torch.device("cpu"),
    )
    checkpoint_exact = all(
        torch.equal(value, restored_model.state_dict()[name])
        for name, value in model.state_dict().items()
    )
    acceptance = {
        "loss_decreased": losses[-1] < losses[0],
        "checkpoint_state_exact": checkpoint_exact,
        "checkpoint_counter_restored": restored_state.global_step == steps,
        "checkpoint_metadata_restored": metadata == {"architecture": config_path.stem},
    }
    return {
        "architecture": config_path.stem,
        "source_config": config_path.as_posix(),
        "diagnostic_overrides": {
            "base_channels": config.base_channels,
            "depth": config.depth,
            "batch_normalization": config.batch_normalization,
            "dropout_probability": config.dropout_probability,
            "res_kernel_sizes": list(config.res_kernel_sizes),
            "wc_kernel_size": config.wc_kernel_size,
        },
        "resolved_diagnostic_config": asdict(config),
        "steps": steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "acceptance": acceptance,
        "status": "pass" if all(acceptance.values()) else "fail",
    }


def run(
    *,
    config_dir: Path,
    output: Path,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run every bounded Gate 6 architecture diagnostic."""
    if steps < 1:
        raise ValueError("steps must be positive")
    config_paths = sorted(config_dir.glob("*.yaml"))
    with tempfile.TemporaryDirectory(prefix="gate6-checkpoints-") as raw_directory:
        checkpoint_dir = Path(raw_directory)
        models = [
            _run_model(
                config_path,
                checkpoint_dir=checkpoint_dir,
                steps=steps,
                seed=seed,
            )
            for config_path in config_paths
        ]
    metadata = collect_run_metadata(
        config_path=Path("configs/losses/catalog.yaml"),
        split_hashes={"train": "not_applicable_synthetic"},
        seed=seed,
        device=torch.device("cpu"),
        mixed_precision=False,
        run_kind="gate6_synthetic_model_family_diagnostic",
    )
    metadata["status"] = "completed"
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": (
            "pass"
            if models and all(row["status"] == "pass" for row in models)
            else "fail"
        ),
        "scope": "bounded synthetic diagnostic; not a performance comparison",
        "models": models,
        "metadata": metadata,
    }
    atomic_write_json(output, result)
    return result


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/models"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gate6_smoke_results.json"),
    )
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20260729)
    arguments = parser.parse_args()
    result = run(
        config_dir=arguments.config_dir,
        output=arguments.output,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
