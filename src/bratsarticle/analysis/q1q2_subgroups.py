"""Prespecified exploratory external subgroup analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from bratsarticle.analysis.q1q2_statistics import (
    Contrast,
    paired_bootstrap_interval,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def _clean_category(value: Any) -> str:
    if pd.isna(value) or not str(value).strip():
        return "unknown"
    return str(value).strip()


def _burden_category(
    volume_mm3: float,
    *,
    lower: float,
    upper: float,
) -> str:
    if volume_mm3 <= lower:
        return "small"
    if volume_mm3 <= upper:
        return "medium"
    return "large"


def assign_external_subgroups(
    model_patient_metrics: pd.DataFrame,
    external_manifest: pd.DataFrame,
    *,
    lower_burden_mm3: float,
    upper_burden_mm3: float,
) -> pd.DataFrame:
    """Attach frozen metadata-derived subgroup labels to model-patient rows."""
    required_metrics = {
        "model_id",
        "patient_id",
        "cohort_role",
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
        "spacing_axis0_mm",
        "spacing_axis1_mm",
        "spacing_axis2_mm",
    }
    required_manifest = {
        "patient_id",
        "disease_group",
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
        "grade",
        "wt_volume_mm3",
        "et_voxel_count",
    }
    missing_metrics = required_metrics.difference(model_patient_metrics.columns)
    missing_manifest = required_manifest.difference(external_manifest.columns)
    if missing_metrics or missing_manifest:
        raise ValueError(
            "Subgroup inputs are missing columns: "
            f"metrics={sorted(missing_metrics)}, manifest={sorted(missing_manifest)}"
        )
    manifest = external_manifest.loc[
        external_manifest["disease_group"].eq("glioma"),
        sorted(required_manifest),
    ].copy()
    if len(manifest) != 95 or not manifest["patient_id"].is_unique:
        raise ValueError("External manifest must contain 95 unique glioma patients")
    output = model_patient_metrics.merge(
        manifest,
        on="patient_id",
        how="inner",
        validate="many_to_one",
        suffixes=("", "_manifest"),
    )
    if len(output) != len(model_patient_metrics):
        raise ValueError("Model metrics do not map to every confirmatory patient")
    for column in (
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
    ):
        observed = output[column].map(_clean_category)
        frozen = output[f"{column}_manifest"].map(_clean_category)
        if not observed.eq(frozen).all():
            raise ValueError(f"Gate H metadata differs from the manifest: {column}")
        output[column] = frozen
        output = output.drop(columns=f"{column}_manifest")
    output["grade_if_available"] = output["grade"].map(_clean_category)
    output["et_present"] = np.where(
        output["et_voxel_count"].astype(float) > 0,
        "present",
        "absent",
    )
    output["development_derived_tumor_burden_tertile"] = output["wt_volume_mm3"].map(
        lambda value: _burden_category(
            float(value),
            lower=lower_burden_mm3,
            upper=upper_burden_mm3,
        )
    )
    output["resolution"] = output.apply(
        lambda row: (
            f"{float(row['spacing_axis0_mm']):.3f}x"
            f"{float(row['spacing_axis1_mm']):.3f}x"
            f"{float(row['spacing_axis2_mm']):.3f}_mm"
        ),
        axis=1,
    )
    return output.sort_values(["model_id", "patient_id"]).reset_index(drop=True)


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "observation_count": len(array),
        "finite_count": len(finite),
        "nan_count": int(np.isnan(array).sum()),
        "positive_infinity_count": int(np.isposinf(array).sum()),
        "negative_infinity_count": int(np.isneginf(array).sum()),
        "mean_finite": float(finite.mean()) if len(finite) else float("nan"),
        "median_finite": float(np.median(finite)) if len(finite) else float("nan"),
        "q1_finite": (
            float(np.quantile(finite, 0.25, method="linear"))
            if len(finite)
            else float("nan")
        ),
        "q3_finite": (
            float(np.quantile(finite, 0.75, method="linear"))
            if len(finite)
            else float("nan")
        ),
    }


def _metric_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    suffixes = (
        "_dice",
        "_hd95_mm",
        "_surface_dice",
        "_sensitivity",
        "_precision",
        "_specificity",
        "_relative_volume_error",
        "_lesion_recall",
        "_lesion_precision",
        "_lesion_wise_dice",
        "_lesion_wise_hd95_mm",
        "_false_positive_lesion_count",
    )
    return tuple(
        sorted(
            column
            for column in frame.columns
            if column == "mean_regional_dice" or column.endswith(suffixes)
        )
    )


def model_subgroup_summaries(
    frame: pd.DataFrame,
    *,
    dimensions: tuple[str, ...],
    minimum_reportable_patients: int,
) -> pd.DataFrame:
    """Summarize every endpoint without treating small cells as inference."""
    rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        if dimension not in frame:
            raise ValueError(f"Frozen subgroup dimension is absent: {dimension}")
        for category in sorted(str(value) for value in frame[dimension].unique()):
            category_frame = frame.loc[frame[dimension].astype(str).eq(category)]
            for model_id in sorted(str(value) for value in frame["model_id"].unique()):
                subset = category_frame.loc[category_frame["model_id"].eq(model_id)]
                patient_count = int(subset["patient_id"].nunique())
                for endpoint in _metric_columns(frame):
                    rows.append(
                        {
                            "analysis_type": "model_estimate",
                            "dimension": dimension,
                            "category": category,
                            "model_id": model_id,
                            "endpoint": endpoint,
                            "patient_count": patient_count,
                            "minimum_reportable_patients": (
                                minimum_reportable_patients
                            ),
                            "reporting_role": (
                                "exploratory_estimation"
                                if patient_count >= minimum_reportable_patients
                                else "descriptive_small_cell"
                            ),
                            **_finite_summary(
                                subset[endpoint].to_numpy(dtype=np.float64)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def contrast_subgroup_summaries(
    frame: pd.DataFrame,
    *,
    dimensions: tuple[str, ...],
    contrasts: tuple[Contrast, ...],
    endpoint: str,
    minimum_reportable_patients: int,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    """Estimate prespecified paired contrasts within exploratory subgroups."""
    rows: list[dict[str, Any]] = []
    analysis_index = 0
    for dimension in dimensions:
        for category in sorted(str(value) for value in frame[dimension].unique()):
            category_frame = frame.loc[frame[dimension].astype(str).eq(category)]
            for contrast in contrasts:
                first = category_frame.loc[
                    category_frame["model_id"].eq(contrast.first),
                    ["patient_id", endpoint],
                ].rename(columns={endpoint: "first"})
                second = category_frame.loc[
                    category_frame["model_id"].eq(contrast.second),
                    ["patient_id", endpoint],
                ].rename(columns={endpoint: "second"})
                paired = first.merge(
                    second,
                    on="patient_id",
                    how="inner",
                    validate="one_to_one",
                )
                if len(paired) != len(first) or len(paired) != len(second):
                    raise ValueError(
                        f"Incomplete subgroup pairing: {dimension}:{category}:"
                        f"{contrast.contrast_id}"
                    )
                differences = paired["first"].to_numpy(dtype=np.float64) - paired[
                    "second"
                ].to_numpy(dtype=np.float64)
                if not np.isfinite(differences).all():
                    raise ValueError("Primary subgroup endpoint must be finite")
                patient_count = len(differences)
                if patient_count >= minimum_reportable_patients:
                    lower, upper = paired_bootstrap_interval(
                        differences,
                        resamples=resamples,
                        confidence_level=confidence_level,
                        seed=seed + analysis_index,
                    )
                    role = "exploratory_estimation"
                else:
                    lower = upper = float("nan")
                    role = "descriptive_small_cell"
                rows.append(
                    {
                        "analysis_type": "paired_difference",
                        "dimension": dimension,
                        "category": category,
                        "contrast_id": contrast.contrast_id,
                        "first_model_id": contrast.first,
                        "second_model_id": contrast.second,
                        "endpoint": endpoint,
                        "patient_count": patient_count,
                        "minimum_reportable_patients": minimum_reportable_patients,
                        "reporting_role": role,
                        "mean_difference": float(differences.mean()),
                        "median_difference": float(np.median(differences)),
                        "bootstrap_lower_95": lower,
                        "bootstrap_upper_95": upper,
                        "probability_of_superiority": float(
                            np.mean(differences > 0) + 0.5 * np.mean(differences == 0)
                        ),
                        "multiplicity_adjusted": False,
                        "confirmatory_interpretation_permitted": False,
                    }
                )
                analysis_index += 1
    return pd.DataFrame(rows)


def _contrasts(
    plan: dict[str, Any],
    *,
    selected_2d_model: str,
) -> tuple[Contrast, ...]:
    primary = cast(dict[str, Any], plan["primary_contrast"])
    output = [
        Contrast(str(primary["id"]), str(primary["first"]), str(primary["second"]))
    ]
    for raw in cast(list[Any], plan["confirmatory_secondary_contrasts"]):
        entry = cast(dict[str, Any], raw)
        if "second" in entry:
            second = str(entry["second"])
        elif (
            entry.get("second_selection")
            == "highest_development_cv_mean_regional_dice_among_frozen_2d_models"
        ):
            second = selected_2d_model
        else:
            raise ValueError(f"Unknown frozen second-model rule: {entry}")
        output.append(
            Contrast(
                contrast_id=str(entry["id"]),
                first=str(entry["first"]),
                second=second,
            )
        )
    return tuple(output)


def analyze_q1q2_external_subgroups(
    config_path: Path = Path("configs/q1q2_v2/subgroup_execution.yaml"),
) -> dict[str, Any]:
    """Generate external subgroup artifacts after the frozen main analysis."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_external_results":
        raise PermissionError("Subgroup execution contract is not frozen")
    completion_path = Path(str(config["statistical_completion"]))
    statistical_completion = _load_json(completion_path)
    if statistical_completion.get("status") != "complete":
        raise PermissionError("Completed frozen statistical analysis is required")
    statistical_execution_path = Path(str(config["statistical_execution"]))
    statistical_execution = _load_yaml(statistical_execution_path)
    statistical_outputs = cast(dict[str, Any], statistical_execution["outputs"])
    model_metrics_path = Path(
        str(statistical_outputs["external_confirmatory_model_patient_metrics"])
    )
    stored_output = cast(
        dict[str, Any],
        cast(dict[str, Any], statistical_completion["outputs"])[
            "external_confirmatory_model_patient_metrics"
        ],
    )
    if stored_output.get("path") != model_metrics_path.as_posix() or stored_output.get(
        "sha256"
    ) != file_digest(model_metrics_path):
        raise ValueError("Statistical model-patient artifact hash differs")
    thresholds_path = Path(str(config["subgroup_thresholds"]))
    thresholds = _load_yaml(thresholds_path)
    if (
        thresholds.get("status") != "frozen_from_development_before_external_results"
        or thresholds.get("external_outcomes_used") is not False
    ):
        raise PermissionError("Development subgroup thresholds are not frozen")
    threshold_source = cast(dict[str, Any], thresholds["source"])
    threshold_source_path = Path(str(threshold_source["path"]))
    if file_digest(threshold_source_path) != str(threshold_source["sha256"]):
        raise ValueError("Development burden-threshold source hash differs")
    manifest_path = Path(str(config["external_test_manifest"]))
    model_metrics = pd.read_csv(model_metrics_path)
    model_metrics = model_metrics.loc[
        model_metrics["cohort_role"].eq(str(config["cohort_role"]))
    ].copy()
    bounds = cast(dict[str, Any], thresholds["thresholds_mm3"])
    enriched = assign_external_subgroups(
        model_metrics,
        pd.read_csv(manifest_path),
        lower_burden_mm3=float(bounds["small_to_medium"]),
        upper_burden_mm3=float(bounds["medium_to_large"]),
    )
    dimensions = tuple(str(value) for value in cast(list[Any], config["dimensions"]))
    minimum = int(config["minimum_reportable_patients"])
    model_summary = model_subgroup_summaries(
        enriched,
        dimensions=dimensions,
        minimum_reportable_patients=minimum,
    )
    plan = _load_yaml(Path(str(config["statistical_analysis_plan"])))
    bootstrap = cast(
        dict[str, Any],
        cast(dict[str, Any], plan["estimation"])["paired_patient_bootstrap"],
    )
    selection_path = Path(str(statistical_outputs["development_2d_selection"]))
    selection = _load_json(selection_path)
    contrast_summary = contrast_subgroup_summaries(
        enriched,
        dimensions=dimensions,
        contrasts=_contrasts(
            plan,
            selected_2d_model=str(selection["selected_model_id"]),
        ),
        endpoint=str(cast(dict[str, Any], plan["primary_endpoint"])["name"]),
        minimum_reportable_patients=minimum,
        resamples=int(bootstrap["resamples"]),
        confidence_level=float(bootstrap["confidence_level"]),
        seed=int(bootstrap["seed"]) + 10_000,
    )
    outputs = {
        key: Path(str(value))
        for key, value in cast(dict[str, Any], config["outputs"]).items()
    }
    atomic_write_csv(outputs["enriched_model_patient_metrics"], _records(enriched))
    atomic_write_csv(outputs["model_subgroup_summary"], _records(model_summary))
    atomic_write_csv(
        outputs["contrast_subgroup_summary"],
        _records(contrast_summary),
    )
    completion = {
        "schema_version": 1,
        "status": "complete",
        "inferential_role": "exploratory_estimation_only",
        "confirmatory_claims_permitted": False,
        "model_patient_count": len(enriched),
        "unique_patient_count": int(enriched["patient_id"].nunique()),
        "model_count": int(enriched["model_id"].nunique()),
        "dimensions": list(dimensions),
        "minimum_reportable_patients": minimum,
        "statistical_completion_sha256": file_digest(completion_path),
        "external_test_manifest_sha256": file_digest(manifest_path),
        "subgroup_thresholds_sha256": file_digest(thresholds_path),
        "outputs": {
            key: {"path": path.as_posix(), "sha256": file_digest(path)}
            for key, path in outputs.items()
            if key != "completion"
        },
    }
    atomic_write_json(outputs["completion"], completion)
    return completion


__all__ = [
    "analyze_q1q2_external_subgroups",
    "assign_external_subgroups",
    "contrast_subgroup_summaries",
    "model_subgroup_summaries",
]
