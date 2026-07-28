"""Voxel-wise regional segmentation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from evaluation.config import EvaluationConfig
from evaluation.surface import compute_surface_metrics


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_binary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: EvaluationConfig,
) -> dict[str, Any]:
    """Compute one patient's overlap, classification, surface, and volume metrics."""
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} differs from target {target.shape}"
        )
    true_positive = int(np.count_nonzero(prediction & target))
    false_positive = int(np.count_nonzero(prediction & ~target))
    false_negative = int(np.count_nonzero(~prediction & target))
    true_negative = int(np.count_nonzero(~prediction & ~target))
    prediction_count = true_positive + false_positive
    target_count = true_positive + false_negative
    both_empty = prediction_count == 0 and target_count == 0
    one_empty = (prediction_count == 0) != (target_count == 0)

    if both_empty:
        dice = config.empty_masks.overlap_both_empty
        iou = config.empty_masks.overlap_both_empty
    elif one_empty:
        dice = config.empty_masks.overlap_one_empty
        iou = config.empty_masks.overlap_one_empty
    else:
        dice = (
            2.0 * true_positive / (2 * true_positive + false_positive + false_negative)
        )
        iou = true_positive / (true_positive + false_positive + false_negative)

    voxel_volume_mm3 = float(np.prod(np.asarray(spacing_mm, dtype=np.float64)))
    signed_volume_error_mm3 = (prediction_count - target_count) * voxel_volume_mm3
    if target_count:
        relative_volume_error = (prediction_count - target_count) / target_count
    elif prediction_count == 0:
        relative_volume_error = 0.0
    else:
        relative_volume_error = float("inf")
    surface = compute_surface_metrics(prediction, target, spacing_mm, config)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": _safe_rate(true_positive, true_positive + false_negative),
        "precision": _safe_rate(true_positive, true_positive + false_positive),
        "specificity": _safe_rate(true_negative, true_negative + false_positive),
        "hd95_mm": surface.hd95_mm,
        "surface_dice": surface.surface_dice,
        "signed_volume_error_mm3": float(signed_volume_error_mm3),
        "absolute_volume_error_mm3": float(abs(signed_volume_error_mm3)),
        "relative_volume_error": float(relative_volume_error),
        "prediction_voxels": prediction_count,
        "target_voxels": target_count,
    }
