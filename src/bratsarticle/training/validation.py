"""Full-volume validation routed through the central evaluator."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn

from bratsarticle.training.losses import class_indices_to_labels
from evaluation import CentralEvaluator


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
