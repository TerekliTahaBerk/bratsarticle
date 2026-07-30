"""Full-volume validation routed through the central evaluator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from bratsarticle.training.losses import class_indices_to_labels
from evaluation import CentralEvaluator
from evaluation.config import EvaluationConfig


@dataclass(frozen=True)
class SelectionDiceResult:
    """Patient rows and their exact mean used only for checkpoint selection."""

    patient_rows: tuple[dict[str, Any], ...]
    mean_regional_dice: float
    validation_loss: float | None


def _dice_from_counts(
    true_positive: int,
    prediction_count: int,
    target_count: int,
    config: EvaluationConfig,
) -> float:
    if prediction_count == 0 and target_count == 0:
        return config.empty_masks.overlap_both_empty
    if (prediction_count == 0) != (target_count == 0):
        return config.empty_masks.overlap_one_empty
    return 2.0 * true_positive / (prediction_count + target_count)


def validate_selection_dice(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    config: EvaluationConfig,
    loss_function: nn.Module | None = None,
) -> SelectionDiceResult:
    """Compute exact patient WT/TC/ET Dice without surface or lesion metrics."""
    was_training = model.training
    model.eval()
    counts: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((3, 3), dtype=np.int64)
    )
    observed_slices: set[tuple[str, int]] = set()
    weighted_loss_sum = 0.0
    loss_sample_count = 0
    with torch.no_grad():
        for batch in batches:
            split_values = [str(value) for value in batch["split"]]
            if any(value != "validation" for value in split_values):
                raise ValueError(
                    "Selection-metric development evaluation requires validation"
                )
            images = batch["image"].to(device, dtype=torch.float32)
            targets = batch["label"].to(device, dtype=torch.long)
            logits = model(images)
            predicted_indices = torch.argmax(logits, dim=1)
            if loss_function is not None:
                batch_loss = loss_function(logits, targets)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError(
                        "Selection validation produced non-finite loss"
                    )
                batch_size = int(images.shape[0])
                weighted_loss_sum += float(batch_loss.detach().cpu()) * batch_size
                loss_sample_count += batch_size
            prediction_regions = torch.stack(
                (
                    predicted_indices > 0,
                    (predicted_indices == 1) | (predicted_indices == 3),
                    predicted_indices == 3,
                ),
                dim=1,
            )
            target_regions = torch.stack(
                (
                    targets > 0,
                    (targets == 1) | (targets == 4),
                    targets == 4,
                ),
                dim=1,
            )
            spatial_axes = tuple(range(2, prediction_regions.ndim))
            batch_true_positive = torch.sum(
                prediction_regions & target_regions,
                dim=spatial_axes,
            ).cpu()
            batch_prediction_count = torch.sum(
                prediction_regions,
                dim=spatial_axes,
            ).cpu()
            batch_target_count = torch.sum(
                target_regions,
                dim=spatial_axes,
            ).cpu()
            for index, patient_id in enumerate(batch["subject_id"]):
                subject = str(patient_id)
                slice_index = int(batch["slice_index"][index])
                key = (subject, slice_index)
                if key in observed_slices:
                    raise ValueError(
                        f"Duplicate validation slice {subject}:{slice_index}"
                    )
                observed_slices.add(key)
                counts[subject][:, 0] += batch_true_positive[index].numpy()
                counts[subject][:, 1] += batch_prediction_count[index].numpy()
                counts[subject][:, 2] += batch_target_count[index].numpy()
    if was_training:
        model.train()
    patient_rows: list[dict[str, Any]] = []
    for subject in sorted(counts):
        regional = [
            _dice_from_counts(
                int(counts[subject][region_index, 0]),
                int(counts[subject][region_index, 1]),
                int(counts[subject][region_index, 2]),
                config,
            )
            for region_index in range(3)
        ]
        patient_rows.append(
            {
                "patient_id": subject,
                "evaluation_stage": "raw",
                "wt_dice": regional[0],
                "tc_dice": regional[1],
                "et_dice": regional[2],
                "mean_regional_dice": float(np.mean(regional)),
            }
        )
    if not patient_rows:
        raise RuntimeError("Selection validation produced no patient rows")
    return SelectionDiceResult(
        patient_rows=tuple(patient_rows),
        mean_regional_dice=float(
            np.mean([float(row["mean_regional_dice"]) for row in patient_rows])
        ),
        validation_loss=(
            weighted_loss_sum / loss_sample_count
            if loss_sample_count
            else None
        ),
    )


def validate_full_volumes(
    model: nn.Module,
    batches: Iterable[dict[str, Any]],
    *,
    device: torch.device,
    evaluator: CentralEvaluator,
) -> list[dict[str, Any]]:
    """Reassemble every validation slice and call the central patient evaluator."""
    was_training = model.training
    model.eval()
    predictions: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    targets: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    spacings: dict[str, tuple[float, float, float]] = {}
    axes: dict[str, int] = {}
    with torch.no_grad():
        for batch in batches:
            split_values = [str(value) for value in batch["split"]]
            if any(value != "validation" for value in split_values):
                raise ValueError(
                    "Full-volume development evaluation requires validation"
                )
            images = batch["image"].to(device, dtype=torch.float32)
            logits = model(images)
            predicted_labels = class_indices_to_labels(torch.argmax(logits, dim=1))
            target_labels = batch["label"].long()
            for index, patient_id in enumerate(batch["subject_id"]):
                subject = str(patient_id)
                slice_index = int(batch["slice_index"][index])
                if slice_index in predictions[subject]:
                    raise ValueError(
                        f"Duplicate validation slice {subject}:{slice_index}"
                    )
                predictions[subject][slice_index] = (
                    predicted_labels[index].detach().cpu().numpy()
                )
                targets[subject][slice_index] = (
                    target_labels[index].detach().cpu().numpy()
                )
                spacing_values = batch["spacing_mm"][index]
                spacing = tuple(float(value) for value in spacing_values)
                if len(spacing) != 3:
                    raise ValueError("Validation spacing must have three values")
                spacings[subject] = (spacing[0], spacing[1], spacing[2])
                axes[subject] = int(batch["slice_axis"][index])
    rows: list[dict[str, Any]] = []
    for subject in sorted(predictions):
        indices = sorted(predictions[subject])
        if indices != list(range(len(indices))):
            raise ValueError(f"Incomplete validation volume for {subject}")
        axis = axes[subject]
        prediction_volume = np.stack(
            [predictions[subject][index] for index in indices],
            axis=axis,
        )
        target_volume = np.stack(
            [targets[subject][index] for index in indices],
            axis=axis,
        )
        rows.extend(
            evaluator.evaluate_batch(
                prediction_volume,
                target_volume,
                patient_ids=[subject],
                spacings_mm=[spacings[subject]],
            )
        )
    if was_training:
        model.train()
    return rows


__all__ = [
    "SelectionDiceResult",
    "validate_full_volumes",
    "validate_selection_dice",
]
