"""BraTS label-to-region transformations and output decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from evaluation.config import EvaluationConfig

REGION_NAMES: tuple[str, ...] = ("wt", "tc", "et")
VALID_LABELS = frozenset({0, 1, 2, 4})


@dataclass(frozen=True)
class DecodedPrediction:
    """Decoded regional masks and pre-correction hierarchy violations."""

    regions: dict[str, np.ndarray]
    nested_violation_voxels: tuple[int, ...]


def as_numpy(value: Any) -> np.ndarray:
    """Detach torch tensors and return a CPU NumPy array."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def labels_to_regions(labels: Any) -> dict[str, np.ndarray]:
    """Transform BraTS labels into nested WT, TC, and ET boolean regions."""
    array = as_numpy(labels)
    if array.ndim == 3:
        array = array[np.newaxis, ...]
    if array.ndim != 4:
        raise ValueError("Label arrays must have shape [B, D, H, W] or [D, H, W]")
    unique = set(int(value) for value in np.unique(array))
    invalid = unique - VALID_LABELS
    if invalid:
        raise ValueError(f"Unexpected BraTS labels: {sorted(invalid)}")
    return {
        "wt": np.isin(array, (1, 2, 4)),
        "tc": np.isin(array, (1, 4)),
        "et": array == 4,
    }


def count_nested_violations(regions: dict[str, np.ndarray]) -> int:
    """Count voxels violating ET subset TC subset WT."""
    et_outside_tc = regions["et"] & ~regions["tc"]
    tc_outside_wt = regions["tc"] & ~regions["wt"]
    return int(np.count_nonzero(et_outside_tc) + np.count_nonzero(tc_outside_wt))


def count_nested_violations_per_patient(
    regions: dict[str, np.ndarray],
) -> tuple[int, ...]:
    """Count hierarchy violations separately for every batch member."""
    return tuple(
        count_nested_violations(
            {region: regions[region][patient_index] for region in REGION_NAMES}
        )
        for patient_index in range(regions["wt"].shape[0])
    )


def enforce_outward_union(
    regions: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Project nested masks outward while preserving positive inner regions."""
    et = regions["et"].astype(bool, copy=True)
    tc = regions["tc"].astype(bool, copy=True) | et
    wt = regions["wt"].astype(bool, copy=True) | tc
    return {"wt": wt, "tc": tc, "et": et}


def _sigmoid(array: np.ndarray) -> np.ndarray:
    clipped = np.clip(array.astype(np.float64, copy=False), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def decode_prediction(
    prediction: Any,
    config: EvaluationConfig,
) -> DecodedPrediction:
    """Decode label, four-class softmax, or nested-sigmoid predictions."""
    array = as_numpy(prediction)
    if config.output_mode == "labels":
        regions = labels_to_regions(array)
        return DecodedPrediction(
            regions=regions,
            nested_violation_voxels=(0,) * regions["wt"].shape[0],
        )

    if array.ndim == 4:
        array = array[np.newaxis, ...]
    if array.ndim != 5:
        raise ValueError(
            "Channel predictions must have shape [B, C, D, H, W] or [C, D, H, W]"
        )

    if config.output_mode == "softmax":
        if array.shape[1] != 4:
            raise ValueError("Four-class softmax output requires exactly 4 channels")
        channel_to_label = np.asarray((0, 1, 2, 4), dtype=np.int16)
        labels = channel_to_label[np.argmax(array, axis=1)]
        regions = labels_to_regions(labels)
        return DecodedPrediction(
            regions=regions,
            nested_violation_voxels=(0,) * regions["wt"].shape[0],
        )

    if array.shape[1] != 3:
        raise ValueError("Nested sigmoid output requires WT, TC, ET channels")
    values = _sigmoid(array) if config.from_logits else array
    regions = {
        "wt": values[:, 0] >= config.wt_threshold,
        "tc": values[:, 1] >= config.tc_threshold,
        "et": values[:, 2] >= config.et_threshold,
    }
    violations = count_nested_violations_per_patient(regions)
    if config.enforce_nested_consistency:
        regions = enforce_outward_union(regions)
    return DecodedPrediction(
        regions=regions,
        nested_violation_voxels=violations,
    )
