"""Single central evaluator for all reportable segmentation results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.binary import compute_binary_metrics
from evaluation.config import EvaluationConfig
from evaluation.lesions import compute_lesion_metrics, filter_small_components
from evaluation.regions import (
    REGION_NAMES,
    decode_prediction,
    enforce_outward_union,
    labels_to_regions,
)

MetricRow = dict[str, Any]


def _resolve_patient_ids(
    batch_size: int,
    patient_ids: Sequence[str] | None,
) -> list[str]:
    if patient_ids is None:
        return [f"patient_{index:04d}" for index in range(batch_size)]
    resolved = [str(value) for value in patient_ids]
    if len(resolved) != batch_size:
        raise ValueError("patient_ids length must equal the batch size")
    if len(set(resolved)) != len(resolved):
        raise ValueError("patient_ids must be unique within an evaluation batch")
    return resolved


def _resolve_spacings(
    batch_size: int,
    spacings_mm: Sequence[Sequence[float]] | None,
    default: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    if spacings_mm is None:
        return [default] * batch_size
    if len(spacings_mm) != batch_size:
        raise ValueError("spacings_mm length must equal the batch size")
    output: list[tuple[float, float, float]] = []
    for spacing in spacings_mm:
        values = tuple(float(value) for value in spacing)
        if len(values) != 3 or any(value <= 0 for value in values):
            raise ValueError("Each spacing must contain three positive values")
        output.append((values[0], values[1], values[2]))
    return output


def _filtered_regions(
    regions: dict[str, np.ndarray],
    spacing_mm: tuple[float, float, float],
    config: EvaluationConfig,
) -> dict[str, np.ndarray]:
    filtered = {
        region: filter_small_components(
            regions[region],
            spacing_mm,
            connectivity=config.lesions.connectivity,
            minimum_voxels=config.postprocessing.minimum_prediction_voxels,
            minimum_volume_mm3=(config.postprocessing.minimum_prediction_volume_mm3),
        )
        for region in REGION_NAMES
    }
    if config.enforce_nested_consistency:
        return enforce_outward_union(filtered)
    return filtered


class CentralEvaluator:
    """Evaluate full 3D patient volumes under one immutable metric policy."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def evaluate_batch(
        self,
        prediction: Any,
        target_labels: Any,
        *,
        patient_ids: Sequence[str] | None = None,
        spacings_mm: Sequence[Sequence[float]] | None = None,
    ) -> list[MetricRow]:
        """Return one deterministic wide metric row per patient and stage."""
        decoded = decode_prediction(prediction, self.config)
        targets = labels_to_regions(target_labels)
        batch_size = int(targets["wt"].shape[0])
        if any(
            decoded.regions[region].shape != targets[region].shape
            for region in REGION_NAMES
        ):
            raise ValueError("Decoded prediction and target shapes must match")
        resolved_ids = _resolve_patient_ids(batch_size, patient_ids)
        resolved_spacings = _resolve_spacings(
            batch_size,
            spacings_mm,
            self.config.default_spacing_mm,
        )

        rows: list[MetricRow] = []
        for patient_index, patient_id in enumerate(resolved_ids):
            spacing = resolved_spacings[patient_index]
            raw_regions = {
                region: decoded.regions[region][patient_index]
                for region in REGION_NAMES
            }
            target_regions = {
                region: targets[region][patient_index] for region in REGION_NAMES
            }
            for stage in self.config.postprocessing.stages:
                prediction_regions = (
                    raw_regions
                    if stage == "raw"
                    else _filtered_regions(raw_regions, spacing, self.config)
                )
                row: MetricRow = {
                    "patient_id": patient_id,
                    "evaluation_stage": stage,
                    "output_mode": self.config.output_mode,
                    "nested_consistency_enforced": (
                        self.config.enforce_nested_consistency
                    ),
                    "nested_violation_voxels_before_correction": (
                        decoded.nested_violation_voxels[patient_index]
                    ),
                    "spacing_axis0_mm": spacing[0],
                    "spacing_axis1_mm": spacing[1],
                    "spacing_axis2_mm": spacing[2],
                }
                regional_dice: list[float] = []
                for region in REGION_NAMES:
                    binary = compute_binary_metrics(
                        prediction_regions[region],
                        target_regions[region],
                        spacing,
                        self.config,
                    )
                    lesions = compute_lesion_metrics(
                        prediction_regions[region],
                        target_regions[region],
                        spacing,
                        self.config,
                    )
                    regional_dice.append(float(binary["dice"]))
                    for metric_name, value in {**binary, **lesions}.items():
                        row[f"{region}_{metric_name}"] = value
                row["mean_regional_dice"] = float(np.mean(regional_dice))
                rows.append(row)
        return rows


def summarize_patient_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize numeric patient metrics while exposing NaN and infinity counts."""
    if not rows:
        return []
    stages = sorted({str(row["evaluation_stage"]) for row in rows})
    excluded = {
        "patient_id",
        "evaluation_stage",
        "output_mode",
        "nested_consistency_enforced",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded
            and isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    summary: list[dict[str, Any]] = []
    for stage in stages:
        stage_rows = [row for row in rows if row["evaluation_stage"] == stage]
        for metric_name in metric_names:
            values = np.asarray(
                [float(row[metric_name]) for row in stage_rows],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            summary.append(
                {
                    "evaluation_stage": stage,
                    "metric": metric_name,
                    "patient_count": len(values),
                    "finite_count": len(finite),
                    "nan_count": int(np.count_nonzero(np.isnan(values))),
                    "infinite_count": int(np.count_nonzero(np.isinf(values))),
                    "mean_finite": (
                        float(np.mean(finite)) if len(finite) else float("nan")
                    ),
                    "median_finite": (
                        float(np.median(finite)) if len(finite) else float("nan")
                    ),
                    "standard_deviation_finite": (
                        float(np.std(finite, ddof=1))
                        if len(finite) > 1
                        else float("nan")
                    ),
                }
            )
    return summary
