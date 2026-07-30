"""Deterministic qualitative selection and rendering after frozen evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy import ndimage

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
)

REGIONS = ("wt", "tc", "et")
SELECTION_RULES = (
    "highest_patient_mean_regional_dice",
    "median_patient_mean_regional_dice",
    "lowest_et_lesion_wise_dice",
    "largest_false_positive_lesion_burden",
    "largest_regional_hd95",
    "largest_pairwise_model_disagreement",
)


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _region_mask(labels: np.ndarray, region: str) -> np.ndarray:
    if region == "wt":
        return np.asarray(labels > 0, dtype=bool)
    if region == "tc":
        return np.asarray(np.isin(labels, (1, 4)), dtype=bool)
    if region == "et":
        return np.asarray(labels == 4, dtype=bool)
    raise ValueError(f"Unknown BraTS region: {region}")


def pairwise_model_disagreement(
    predictions: list[np.ndarray],
    target: np.ndarray,
) -> float:
    """Compute exact mean pairwise regional disagreement without pair loops."""
    if len(predictions) < 2:
        raise ValueError("Pairwise disagreement requires at least two models")
    shape = target.shape
    if target.ndim != 3 or any(prediction.shape != shape for prediction in predictions):
        raise ValueError("Pairwise disagreement shapes differ")
    union_wt = _region_mask(target, "wt").copy()
    for prediction in predictions:
        union_wt |= _region_mask(prediction, "wt")
    domain_voxels = int(np.count_nonzero(union_wt))
    if domain_voxels == 0:
        return 0.0
    model_count = len(predictions)
    pair_count = model_count * (model_count - 1) // 2
    regional: list[float] = []
    for region in REGIONS:
        positive_count = np.zeros(shape, dtype=np.uint8)
        for prediction in predictions:
            positive_count += _region_mask(prediction, region)
        disagreeing_pairs = positive_count.astype(np.int64) * (
            model_count - positive_count.astype(np.int64)
        )
        regional.append(
            float(disagreeing_pairs[union_wt].sum())
            / float(pair_count * domain_voxels)
        )
    return float(np.mean(regional))


def _prediction_index(gate_h: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stored = cast(
        dict[str, dict[str, str]],
        gate_h["model_prediction_manifests"],
    )
    output: dict[str, dict[str, Any]] = {}
    for model_id, entry in stored.items():
        manifest_path = Path(entry["path"])
        if (
            not manifest_path.is_file()
            or file_digest(manifest_path) != entry["sha256"]
        ):
            raise RuntimeError(f"Model prediction manifest differs: {model_id}")
        manifest = _load_json(manifest_path)
        if manifest.get("model_id") != model_id or manifest.get("status") != "complete":
            raise RuntimeError(f"Model prediction identity differs: {model_id}")
        output[model_id] = {
            "directory": manifest_path.parent,
            "patients": cast(dict[str, dict[str, str]], manifest["patients"]),
        }
    return output


def _load_model_prediction(
    index: dict[str, dict[str, Any]],
    model_id: str,
    patient_id: str,
) -> np.ndarray:
    model = index[model_id]
    entry = cast(dict[str, str], model["patients"][patient_id])
    path = cast(Path, model["directory"]) / f"{patient_id}.npz"
    if not path.is_file() or file_digest(path) != entry["sha256"]:
        raise RuntimeError(
            f"Retained model prediction differs: {model_id}:{patient_id}"
        )
    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction_label"], dtype=np.uint8)
    return prediction


def _cache_paths(gate_h: dict[str, Any], patient_id: str) -> tuple[Path, Path]:
    cache = cast(dict[str, Any], gate_h["cache"])
    directory = (
        Path(str(cache["cache_root"]))
        / f"{patient_id}-{str(cache['inventory_sha256'])[:16]}.npycache"
    )
    if not (directory / "COMPLETE").is_file():
        raise RuntimeError(f"External derived cache is incomplete: {patient_id}")
    return directory / "image.npy", directory / "label.npy"


def _patient_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {
        "patient_id",
        "model_id",
        "mean_regional_dice",
        "et_lesion_wise_dice",
        "wt_false_positive_lesion_count",
        "tc_false_positive_lesion_count",
        "et_false_positive_lesion_count",
        "wt_hd95_mm",
        "tc_hd95_mm",
        "et_hd95_mm",
        "wt_target_voxels",
        "spacing_axis0_mm",
        "spacing_axis1_mm",
        "spacing_axis2_mm",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Qualitative metrics miss columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    for patient_id, group in metrics.groupby("patient_id", sort=True):
        et_values = group["et_lesion_wise_dice"].to_numpy(dtype=np.float64)
        finite_et = et_values[np.isfinite(et_values)]
        hd95 = group[
            ["wt_hd95_mm", "tc_hd95_mm", "et_hd95_mm"]
        ].to_numpy(dtype=np.float64)
        voxel_volume = float(
            group.iloc[0]["spacing_axis0_mm"]
            * group.iloc[0]["spacing_axis1_mm"]
            * group.iloc[0]["spacing_axis2_mm"]
        )
        rows.append(
            {
                "patient_id": str(patient_id),
                "patient_mean_regional_dice": float(
                    group["mean_regional_dice"].mean()
                ),
                "patient_mean_finite_et_lesion_wise_dice": (
                    float(finite_et.mean()) if len(finite_et) else float("nan")
                ),
                "patient_mean_false_positive_lesion_burden": float(
                    group[
                        [
                            "wt_false_positive_lesion_count",
                            "tc_false_positive_lesion_count",
                            "et_false_positive_lesion_count",
                        ]
                    ]
                    .sum(axis=1)
                    .mean()
                ),
                "patient_largest_regional_hd95_mm": float(np.nanmax(hd95)),
                "whole_tumor_reference_volume_mm3": float(
                    group.iloc[0]["wt_target_voxels"] * voxel_volume
                ),
            }
        )
    return pd.DataFrame(rows)


def _choose(
    frame: pd.DataFrame,
    *,
    value: str,
    ascending: bool,
) -> dict[str, Any]:
    eligible = frame.loc[frame[value].notna()].copy()
    if not len(eligible):
        raise RuntimeError(f"No eligible qualitative patient for {value}")
    selected = eligible.sort_values(
        [value, "whole_tumor_reference_volume_mm3", "patient_id"],
        ascending=[ascending, False, True],
        kind="mergesort",
    ).iloc[0]
    return cast(dict[str, Any], selected.to_dict())


def select_qualitative_cases(
    summary: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Apply all frozen rules and tie-breakers to patient-level summaries."""
    median = float(summary["patient_mean_regional_dice"].median())
    median_frame = summary.assign(
        distance_to_cohort_median=(
            summary["patient_mean_regional_dice"] - median
        ).abs()
    )
    return {
        "highest_patient_mean_regional_dice": _choose(
            summary,
            value="patient_mean_regional_dice",
            ascending=False,
        ),
        "median_patient_mean_regional_dice": _choose(
            median_frame,
            value="distance_to_cohort_median",
            ascending=True,
        ),
        "lowest_et_lesion_wise_dice": _choose(
            summary,
            value="patient_mean_finite_et_lesion_wise_dice",
            ascending=True,
        ),
        "largest_false_positive_lesion_burden": _choose(
            summary,
            value="patient_mean_false_positive_lesion_burden",
            ascending=False,
        ),
        "largest_regional_hd95": _choose(
            summary,
            value="patient_largest_regional_hd95_mm",
            ascending=False,
        ),
        "largest_pairwise_model_disagreement": _choose(
            summary,
            value="pairwise_model_disagreement",
            ascending=False,
        ),
    }


def _normalized_slice(volume: np.ndarray, slice_index: int) -> np.ndarray:
    image = np.asarray(volume[:, :, slice_index], dtype=np.float32)
    nonzero = image[image != 0]
    if not len(nonzero):
        return image
    lower, upper = np.quantile(nonzero, [0.01, 0.99])
    if upper <= lower:
        return np.zeros_like(image)
    return np.asarray(
        np.clip((image - lower) / (upper - lower), 0.0, 1.0),
        dtype=np.float32,
    )


def _render_case(
    *,
    rule: str,
    patient_id: str,
    image: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    model_names: dict[str, str],
    destination: Path,
    dpi: int,
) -> dict[str, Any]:
    wt_per_slice = _region_mask(target, "wt").sum(axis=(0, 1))
    slice_index = int(np.flatnonzero(wt_per_slice == wt_per_slice.max())[0])
    flair = _normalized_slice(image[3], slice_index)
    figure, axes = plt.subplots(5, 6, figsize=(18, 14))
    modalities = ("T1", "T1ce", "T2", "FLAIR")
    for index, name in enumerate(modalities):
        axes[0, index].imshow(
            _normalized_slice(image[index], slice_index).T,
            cmap="gray",
            origin="lower",
        )
        axes[0, index].set_title(name)
    label_cmap = ListedColormap(["black", "#2E86DE", "#F1C40F", "#E74C3C"])
    label_norm = BoundaryNorm([-0.5, 0.5, 1.5, 3.0, 4.5], label_cmap.N)
    display_target = target[:, :, slice_index].copy()
    display_target[display_target == 4] = 3
    axes[0, 4].imshow(
        display_target.T,
        cmap=label_cmap,
        norm=label_norm,
        origin="lower",
    )
    axes[0, 4].set_title("Reference labels")
    components, _ = ndimage.label(
        _region_mask(target, "wt"),
        structure=np.ones((3, 3, 3), dtype=np.uint8),
    )
    axes[0, 5].imshow(
        components[:, :, slice_index].T,
        cmap="nipy_spectral",
        origin="lower",
    )
    axes[0, 5].set_title("Reference WT components")

    ordered_models = sorted(predictions)
    for index, model_id in enumerate(ordered_models):
        row = 1 + index // 6
        column = index % 6
        display = predictions[model_id][:, :, slice_index].copy()
        display[display == 4] = 3
        axes[row, column].imshow(
            display.T,
            cmap=label_cmap,
            norm=label_norm,
            origin="lower",
        )
        axes[row, column].set_title(model_names[model_id], fontsize=8)
        overlay_row = 3 + index // 6
        prediction_wt = _region_mask(predictions[model_id], "wt")[:, :, slice_index]
        target_wt = _region_mask(target, "wt")[:, :, slice_index]
        false_positive = prediction_wt & ~target_wt
        false_negative = target_wt & ~prediction_wt
        axes[overlay_row, column].imshow(flair.T, cmap="gray", origin="lower")
        overlay = np.zeros((*false_positive.shape, 4), dtype=np.float32)
        overlay[false_positive] = (1.0, 0.15, 0.15, 0.75)
        overlay[false_negative] = (0.0, 0.9, 0.95, 0.75)
        axes[overlay_row, column].imshow(overlay.transpose(1, 0, 2), origin="lower")
        axes[overlay_row, column].set_title(
            f"{model_names[model_id]} FP/FN",
            fontsize=7,
        )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{rule} | {patient_id} | axial slice {slice_index}",
        fontsize=13,
    )
    figure.tight_layout()
    destination.mkdir(parents=True, exist_ok=True)
    png = destination / f"{rule}.png"
    svg = destination / f"{rule}.svg"
    figure.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "patient_id": patient_id,
        "slice_index": slice_index,
        "png": png.as_posix(),
        "png_sha256": file_digest(png),
        "svg": svg.as_posix(),
        "svg_sha256": file_digest(svg),
    }


def analyze_q1q2_qualitative(
    *,
    config_path: Path = Path("configs/q1q2_v2/qualitative_execution.yaml"),
) -> dict[str, Any]:
    """Select and render cases without any new model inference."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_external_results":
        raise PermissionError("Qualitative execution protocol is not frozen")
    protocol_path = Path(str(config["qualitative_protocol"]))
    protocol = _load_yaml(protocol_path)
    if protocol.get("status") != "prespecified_rules_only_no_cases_selected":
        raise PermissionError("Qualitative selection rules are not prespecified")
    gate_h_path = Path(str(config["gate_h_completion"]))
    gate_h = _load_json(gate_h_path)
    if (
        gate_h.get("status") != "pass"
        or gate_h.get("gate_h_pass") is not True
        or gate_h.get("all_model_predictions_retained") is not True
    ):
        raise PermissionError("Passing Gate H with retained predictions is required")
    metrics_path = Path(str(gate_h["model_patient_metrics"]))
    if file_digest(metrics_path) != gate_h["model_patient_metrics_sha256"]:
        raise RuntimeError("Gate H model-patient metric hash differs")
    metrics = pd.read_csv(metrics_path)
    metrics = metrics.loc[
        metrics["cohort_role"].eq(str(config["cohort_role"]))
    ].copy()
    expected = cast(dict[str, Any], config["expected"])
    if (
        len(metrics) != int(expected["model_patient_rows"])
        or metrics["model_id"].nunique() != int(expected["models"])
        or metrics["patient_id"].nunique() != int(expected["patients"])
    ):
        raise RuntimeError("Qualitative model-patient matrix is incomplete")
    prediction_index = _prediction_index(gate_h)
    model_ids = sorted(prediction_index)
    if len(model_ids) != int(expected["models"]):
        raise RuntimeError("Qualitative retained model count differs")
    summary = _patient_summary(metrics)
    disagreement_rows: list[dict[str, Any]] = []
    for patient_id in sorted(str(value) for value in summary["patient_id"]):
        _, label_path = _cache_paths(gate_h, patient_id)
        target = np.load(label_path, mmap_mode="r", allow_pickle=False)
        predictions = [
            _load_model_prediction(prediction_index, model_id, patient_id)
            for model_id in model_ids
        ]
        disagreement_rows.append(
            {
                "patient_id": patient_id,
                "pairwise_model_disagreement": pairwise_model_disagreement(
                    predictions,
                    target,
                ),
            }
        )
    disagreement = pd.DataFrame(disagreement_rows)
    summary = summary.merge(
        disagreement,
        on="patient_id",
        validate="one_to_one",
    )
    selections = select_qualitative_cases(summary)
    if set(selections) != set(SELECTION_RULES):
        raise RuntimeError("Qualitative rule execution is incomplete")
    matrix = _load_yaml(Path(str(config["model_matrix"])))
    model_names = {
        str(entry["id"]): str(entry["display_name"])
        for entry in cast(list[dict[str, Any]], matrix["main_models"])
    }
    outputs = cast(dict[str, Any], config["outputs"])
    output_directory = Path(str(outputs["directory"]))
    panels: dict[str, dict[str, Any]] = {}
    for rule, selection in selections.items():
        patient_id = str(selection["patient_id"])
        image_path, label_path = _cache_paths(gate_h, patient_id)
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        target = np.load(label_path, mmap_mode="r", allow_pickle=False)
        panel_predictions = {
            model_id: _load_model_prediction(
                prediction_index,
                model_id,
                patient_id,
            )
            for model_id in model_ids
        }
        panels[rule] = _render_case(
            rule=rule,
            patient_id=patient_id,
            image=image,
            target=target,
            predictions=panel_predictions,
            model_names=model_names,
            destination=output_directory / "panels",
            dpi=int(cast(dict[str, Any], config["render"])["dpi"]),
        )
    selected_path = Path(str(outputs["selected_cases"]))
    disagreement_path = Path(str(outputs["patient_disagreement"]))
    panel_path = Path(str(outputs["panel_manifest"]))
    atomic_write_json(
        selected_path,
        {
            "schema_version": 1,
            "status": "selected_after_evaluation_by_prespecified_rules",
            "required_wording": protocol["required_wording"],
            "rules": selections,
        },
    )
    atomic_write_csv(
        disagreement_path,
        cast(list[dict[str, Any]], disagreement.to_dict(orient="records")),
    )
    atomic_write_json(panel_path, {"schema_version": 1, "panels": panels})
    completion = {
        "schema_version": 1,
        "status": "complete",
        "cases_selected_after_evaluation": True,
        "cherry_picking_performed": False,
        "new_model_inference_performed": False,
        "rule_count": len(selections),
        "gate_h_completion_sha256": file_digest(gate_h_path),
        "qualitative_protocol_sha256": file_digest(protocol_path),
        "selected_cases_sha256": file_digest(selected_path),
        "patient_disagreement_sha256": file_digest(disagreement_path),
        "panel_manifest_sha256": file_digest(panel_path),
    }
    atomic_write_json(Path(str(outputs["completion"])), completion)
    return completion


__all__ = [
    "analyze_q1q2_qualitative",
    "pairwise_model_disagreement",
    "select_qualitative_cases",
]
