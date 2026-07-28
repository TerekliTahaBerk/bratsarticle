"""Controlled one-slice real-data overfit diagnostic for Gate 5."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from bratsarticle.data.dataset import build_development_dataset
from bratsarticle.data.preprocessing import (
    IntensityAugmentationConfig,
    PreprocessingConfig,
    SpatialAugmentationConfig,
    TrainingSamplingConfig,
)
from bratsarticle.models import StandardUNet2D
from bratsarticle.training import DiceCrossEntropyLoss, TrainingEngine
from bratsarticle.training.losses import class_indices_to_labels
from bratsarticle.training.reproducibility import (
    collect_run_metadata,
    seed_everything,
)
from bratsarticle.utils.serialization import atomic_write_json
from evaluation import CentralEvaluator, EvaluationConfig


def _split_hashes(split_dir: Path) -> dict[str, str]:
    metadata = json.loads(
        (split_dir / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "train": str(metadata["manifest_sha256"]["train"]),
        "validation": str(metadata["manifest_sha256"]["validation"]),
    }


def _slice_metrics(
    model: StandardUNet2D,
    image: torch.Tensor,
    label: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = class_indices_to_labels(torch.argmax(model(image), dim=1)).cpu()
    prediction_volume = np.stack(
        [prediction[0].numpy(), prediction[0].numpy()],
        axis=2,
    )
    target_volume = np.stack(
        [label[0].numpy(), label[0].numpy()],
        axis=2,
    )
    row = CentralEvaluator(EvaluationConfig(output_mode="labels")).evaluate_batch(
        prediction_volume,
        target_volume,
        patient_ids=["gate5_real_overfit_slice"],
        spacings_mm=[(1.0, 1.0, 1.0)],
    )[0]
    return {
        "mean_regional_dice": float(row["mean_regional_dice"]),
        "wt_dice": float(row["wt_dice"]),
        "tc_dice": float(row["tc_dice"]),
        "et_dice": float(row["et_dice"]),
    }


def run(
    *,
    dataset_root: Path,
    split_dir: Path,
    output: Path,
    steps: int,
    seed: int,
) -> dict[str, object]:
    """Run the controlled diagnostic and write machine-readable results."""
    seed_everything(seed)
    preprocessing = replace(
        PreprocessingConfig(),
        training_sampling=TrainingSamplingConfig(
            tumor_probability=1.0,
            tumor_minimum_voxels_per_slice=1,
            samples_per_patient_per_epoch=1,
        ),
        spatial_augmentation=SpatialAugmentationConfig(enabled=False),
        intensity_augmentation=IntensityAugmentationConfig(enabled=False),
    )
    dataset = build_development_dataset(
        split_dir,
        "train",
        dataset_root,
        preprocessing,
        seed=seed,
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
    model = StandardUNet2D(base_channels=4, depth=2)
    loss_function = DiceCrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    with torch.no_grad():
        initial_loss = float(loss_function(model(image), label))
    initial_metrics = _slice_metrics(model, image, label)
    losses = [engine.train_step(image, label) for _ in range(steps)]
    with torch.no_grad():
        final_loss = float(loss_function(model(image), label))
    final_metrics = _slice_metrics(model, image, label)
    metadata = collect_run_metadata(
        config_path=Path("configs/training/unet2d_baseline.yaml"),
        split_hashes=_split_hashes(split_dir),
        seed=seed,
        device=torch.device("cpu"),
        mixed_precision=False,
        run_kind="gate5_real_single_slice_overfit_diagnostic",
    )
    metadata["status"] = "completed"
    acceptance = {
        "loss_reduced_by_at_least_50_percent": final_loss < initial_loss * 0.5,
        "mean_regional_dice_improved": (
            final_metrics["mean_regional_dice"] > initial_metrics["mean_regional_dice"]
        ),
        "final_wt_dice_at_least_0_80": final_metrics["wt_dice"] >= 0.80,
        "final_tc_dice_at_least_0_80": final_metrics["tc_dice"] >= 0.80,
        "final_et_dice_at_least_0_80": final_metrics["et_dice"] >= 0.80,
    }
    result: dict[str, object] = {
        "status": "pass" if all(acceptance.values()) else "fail",
        "subject_id": subject_id,
        "slice_index": slice_index,
        "target_voxels": {
            "wt": int(np.count_nonzero(label.numpy())),
            "tc": int(np.count_nonzero(np.isin(label.numpy(), (1, 4)))),
            "et": int(np.count_nonzero(label.numpy() == 4)),
        },
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_history_every_10_steps": [
            float(losses[index]) for index in range(9, len(losses), 10)
        ],
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "acceptance": acceptance,
        "metadata": metadata,
    }
    atomic_write_json(output, result)
    return result


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("splits/provisional"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gate5_real_overfit_metrics.json"),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    arguments = parser.parse_args()
    result = run(
        dataset_root=arguments.dataset_root,
        split_dir=arguments.split_dir,
        output=arguments.output,
        steps=arguments.steps,
        seed=arguments.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
