"""Connected-component lesion evaluation and one-to-one matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from evaluation.binary import compute_binary_metrics
from evaluation.config import EvaluationConfig, LesionEvaluationConfig


@dataclass(frozen=True)
class LesionMatch:
    """One accepted ground-truth/prediction component pairing."""

    target_index: int
    prediction_index: int
    iou: float


def _structure(connectivity: int) -> np.ndarray:
    rank = {6: 1, 18: 2, 26: 3}.get(connectivity)
    if rank is None:
        raise ValueError("3D connectivity must be one of 6, 18, or 26")
    return np.asarray(ndimage.generate_binary_structure(3, rank), dtype=bool)


def component_masks(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    connectivity: int,
    minimum_voxels: int,
    minimum_volume_mm3: float,
) -> list[np.ndarray]:
    """Extract deterministic, size-filtered 3D connected components."""
    if mask.ndim != 3:
        raise ValueError("Lesion components require a 3D mask")
    labeled, count = ndimage.label(
        mask.astype(bool), structure=_structure(connectivity)
    )
    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_mm, dtype=np.float64)))
    components: list[np.ndarray] = []
    for label_index in range(1, count + 1):
        component = labeled == label_index
        voxels = int(np.count_nonzero(component))
        if voxels < minimum_voxels:
            continue
        if voxels * voxel_volume_mm3 < minimum_volume_mm3:
            continue
        components.append(component)
    return components


def filter_small_components(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    *,
    connectivity: int,
    minimum_voxels: int,
    minimum_volume_mm3: float,
) -> np.ndarray:
    """Return the union of components meeting the declared size rules."""
    retained = component_masks(
        mask,
        spacing_mm,
        connectivity=connectivity,
        minimum_voxels=minimum_voxels,
        minimum_volume_mm3=minimum_volume_mm3,
    )
    output = np.zeros_like(mask, dtype=bool)
    for component in retained:
        output |= component
    return output


def match_components(
    targets: list[np.ndarray],
    predictions: list[np.ndarray],
    *,
    minimum_iou: float,
) -> list[LesionMatch]:
    """Find the maximum-total-IoU one-to-one assignment with positive overlap."""
    if not targets or not predictions:
        return []
    iou_matrix = np.zeros((len(targets), len(predictions)), dtype=np.float64)
    for target_index, target in enumerate(targets):
        for prediction_index, prediction in enumerate(predictions):
            intersection = int(np.count_nonzero(target & prediction))
            if intersection == 0:
                continue
            union = int(np.count_nonzero(target | prediction))
            iou_matrix[target_index, prediction_index] = intersection / union
    target_indices, prediction_indices = linear_sum_assignment(
        iou_matrix,
        maximize=True,
    )
    matches = []
    for target_index, prediction_index in zip(
        target_indices,
        prediction_indices,
        strict=True,
    ):
        iou = float(iou_matrix[target_index, prediction_index])
        if iou > 0.0 and iou >= minimum_iou:
            matches.append(
                LesionMatch(
                    target_index=int(target_index),
                    prediction_index=int(prediction_index),
                    iou=iou,
                )
            )
    return matches


def compute_lesion_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Compute patient-level lesion detection and matched-lesion metrics."""
    lesion_config: LesionEvaluationConfig = config.lesions
    targets = component_masks(
        target,
        spacing_mm,
        connectivity=lesion_config.connectivity,
        minimum_voxels=lesion_config.minimum_voxels,
        minimum_volume_mm3=lesion_config.minimum_volume_mm3,
    )
    predictions = component_masks(
        prediction,
        spacing_mm,
        connectivity=lesion_config.connectivity,
        minimum_voxels=lesion_config.minimum_voxels,
        minimum_volume_mm3=lesion_config.minimum_volume_mm3,
    )
    matches = match_components(
        targets,
        predictions,
        minimum_iou=lesion_config.minimum_match_iou,
    )
    true_positive_count = len(matches)
    false_negative_count = len(targets) - true_positive_count
    false_positive_count = len(predictions) - true_positive_count
    recall = true_positive_count / len(targets) if targets else float("nan")
    precision = true_positive_count / len(predictions) if predictions else float("nan")

    dice_values: list[float] = []
    hd95_values: list[float] = []
    matches_by_target = {match.target_index: match for match in matches}
    for target_index, target_component in enumerate(targets):
        match = matches_by_target.get(target_index)
        if match is None:
            dice_values.append(0.0)
            hd95_values.append(float("inf"))
            continue
        prediction_component = predictions[match.prediction_index]
        overlap = compute_binary_metrics(
            prediction_component,
            target_component,
            spacing_mm,
            config,
        )
        dice_values.append(float(overlap["dice"]))
        hd95_values.append(float(overlap["hd95_mm"]))

    return {
        "lesion_recall": float(recall),
        "lesion_precision": float(precision),
        "false_positive_lesion_count": false_positive_count,
        "false_negative_lesion_count": false_negative_count,
        "matched_lesion_count": true_positive_count,
        "target_lesion_count": len(targets),
        "prediction_lesion_count": len(predictions),
        "lesion_wise_dice": (
            float(np.mean(dice_values)) if dice_values else float("nan")
        ),
        "lesion_wise_hd95_mm": (
            float(np.mean(hd95_values)) if hd95_values else float("nan")
        ),
    }
