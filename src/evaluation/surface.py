"""Physical-space surface metrics with explicit empty-mask behavior."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from surface_distance import metrics as surface_metrics

from evaluation.config import EvaluationConfig


@dataclass(frozen=True)
class SurfaceMetrics:
    """HD percentile and area-weighted surface Dice."""

    hd95_mm: float
    surface_dice: float


def compute_surface_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: EvaluationConfig,
) -> SurfaceMetrics:
    """Compute physical-space surface metrics using DeepMind's implementation."""
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    prediction_empty = not bool(prediction.any())
    target_empty = not bool(target.any())
    if prediction_empty and target_empty:
        return SurfaceMetrics(
            hd95_mm=config.empty_masks.hd95_both_empty_mm,
            surface_dice=config.empty_masks.surface_dice_both_empty,
        )
    if prediction_empty or target_empty:
        return SurfaceMetrics(
            hd95_mm=float("inf"),
            surface_dice=config.empty_masks.surface_dice_one_empty,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        distances = surface_metrics.compute_surface_distances(
            target,
            prediction,
            spacing_mm,
        )
        return SurfaceMetrics(
            hd95_mm=float(
                surface_metrics.compute_robust_hausdorff(
                    distances,
                    config.hd_percentile,
                )
            ),
            surface_dice=float(
                surface_metrics.compute_surface_dice_at_tolerance(
                    distances,
                    config.surface_tolerance_mm,
                )
            ),
        )
