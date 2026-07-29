"""Artifact-only Gate 11 patient-level statistical analysis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from bratsarticle.experiments.gate11_runner import load_gate11_plan
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

_IDENTITY_COLUMNS = {
    "candidate_id",
    "seed",
    "run_id",
    "patient_id",
    "evaluation_stage",
    "output_mode",
    "nested_consistency_enforced",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _records(frame: pd.DataFrame) -> list[Mapping[str, Any]]:
    return cast(list[Mapping[str, Any]], frame.to_dict(orient="records"))


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for the paired mean."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(resamples, len(values)))
    bootstrap_means = np.mean(values[indices], axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail, 1.0 - tail],
        method="linear",
    )
    return float(lower), float(upper)


def sign_flip_permutation_p_value(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> float:
    """Return a two-sided Monte Carlo paired sign-flip p-value."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1 or not np.isfinite(values).all():
        return float("nan")
    observed = abs(float(np.mean(values)))
    generator = np.random.default_rng(seed)
    extreme = 0
    completed = 0
    chunk_size = 10_000
    while completed < resamples:
        current = min(chunk_size, resamples - completed)
        signs = generator.integers(
            0,
            2,
            size=(current, len(values)),
            dtype=np.int8,
        )
        signs = signs * 2 - 1
        permuted = np.mean(signs * values, axis=1)
        extreme += int(np.count_nonzero(np.abs(permuted) >= observed))
        completed += current
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply deterministic Holm family-wise error correction."""
    finite = {
        name: float(value)
        for name, value in p_values.items()
        if np.isfinite(float(value))
    }
    ordered = sorted(finite, key=lambda name: (finite[name], name))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        candidate = min(1.0, (count - index) * finite[name])
        running = max(running, candidate)
        adjusted[name] = running
    return {
        name: adjusted.get(name, float("nan")) for name in p_values
    }


def _finite_summary(
    values: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if len(finite):
        lower, upper = paired_bootstrap_interval(
            finite,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        mean = float(np.mean(finite))
        median = float(np.median(finite))
        standard_deviation = (
            float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        )
    else:
        lower = upper = mean = median = standard_deviation = float("nan")
    return {
        "patient_count": len(array),
        "finite_count": len(finite),
        "nan_count": int(np.count_nonzero(np.isnan(array))),
        "positive_infinity_count": int(np.count_nonzero(np.isposinf(array))),
        "negative_infinity_count": int(np.count_nonzero(np.isneginf(array))),
        "mean_finite": mean,
        "median_finite": median,
        "standard_deviation_finite": standard_deviation,
        "bootstrap_lower_finite": lower,
        "bootstrap_upper_finite": upper,
    }


def _comparison_summary(
    first: pd.Series[float],
    second: pd.Series[float],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    first_values = first.to_numpy(dtype=np.float64)
    second_values = second.to_numpy(dtype=np.float64)
    paired = np.isfinite(first_values) & np.isfinite(second_values)
    differences = first_values[paired] - second_values[paired]
    if len(differences):
        lower, upper = paired_bootstrap_interval(
            differences,
            resamples=resamples,
            confidence_level=confidence_level,
            seed=seed,
        )
        mean = float(np.mean(differences))
        median = float(np.median(differences))
        standard_deviation = (
            float(np.std(differences, ddof=1))
            if len(differences) > 1
            else float("nan")
        )
        dz = (
            mean / standard_deviation
            if np.isfinite(standard_deviation) and standard_deviation > 0
            else float("nan")
        )
    else:
        lower = upper = mean = median = dz = float("nan")
    return (
        {
            "paired_patient_count": len(differences),
            "excluded_nonfinite_pair_count": int(len(first_values) - len(differences)),
            "paired_mean_difference": mean,
            "paired_median_difference": median,
            "paired_bootstrap_lower": lower,
            "paired_bootstrap_upper": upper,
            "standardized_paired_effect_dz": dz,
        },
        differences,
    )


def _load_artifacts(
    plan: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    gate10 = _mapping(plan["gate10"], "gate10")
    inference = _mapping(plan["inference"], "inference")
    access = _mapping(plan["access"], "access")
    artifact_root = Path(str(inference["artifact_root"]))
    checkpoint_manifest_path = Path(str(gate10["checkpoint_manifest"]))
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    entries = cast(
        Sequence[Mapping[str, Any]],
        checkpoint_manifest["checkpoints"],
    )
    run_metadata = json.loads(
        (artifact_root / "run_metadata.json").read_text(encoding="utf-8")
    )
    invalid: dict[str, str] = {}
    metric_frames: list[pd.DataFrame] = []
    latency_frames: list[pd.DataFrame] = []
    valid_runs: list[str] = []
    seen_pairs: set[tuple[str, int]] = set()
    for entry in entries:
        candidate = str(entry["candidate_id"])
        seed = int(entry["seed"])
        key = (candidate, seed)
        if key in seen_pairs:
            invalid[f"{candidate}:{seed}"] = "duplicate candidate-seed"
            continue
        seen_pairs.add(key)
        directory = artifact_root / candidate / str(seed)
        try:
            metadata = json.loads(
                (directory / "metadata.json").read_text(encoding="utf-8")
            )
            metrics = pd.read_csv(directory / "patient_metrics.csv")
            latency = pd.read_csv(directory / "latency.csv")
            if metadata["status"] != "completed":
                raise RuntimeError("status is not completed")
            if metadata["checkpoint_sha256"] != entry["checkpoint_sha256"]:
                raise RuntimeError("checkpoint hash differs from Gate 10")
            if metadata["model_config_sha256"] != entry["model_config_sha256"]:
                raise RuntimeError("model-config hash differs from Gate 10")
            if len(metrics) != 74 or metrics["patient_id"].nunique() != 74:
                raise RuntimeError("patient metric rows are not 74 unique patients")
            if set(metrics["evaluation_stage"].astype(str)) != {"raw"}:
                raise RuntimeError("non-raw evaluation stage found")
            if len(latency) != 74 or latency["patient_id"].nunique() != 74:
                raise RuntimeError("latency rows are not 74 unique patients")
            if set(metrics["candidate_id"].astype(str)) != {candidate}:
                raise RuntimeError("candidate mismatch in patient metrics")
            if set(metrics["seed"].astype(int)) != {seed}:
                raise RuntimeError("seed mismatch in patient metrics")
            metric_frames.append(metrics)
            latency_frames.append(latency)
            valid_runs.append(f"{candidate}:{seed}")
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            invalid[f"{candidate}:{seed}"] = str(error)
    audit_log = Path(str(access["audit_log"]))
    access_events = [
        json.loads(line)
        for line in audit_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("event") == "internal_test_manifest_access"
    ]
    test_manifest = Path(str(gate10["frozen_split_dir"])) / "test.csv"
    access_valid = (
        len(access_events) == 1
        and access_events[0]["purpose"] == access["purpose"]
        and access_events[0]["manifest_sha256"] == file_digest(test_manifest)
    )
    audit = {
        "status": (
            "complete"
            if len(valid_runs) == int(inference["expected_checkpoints"])
            and not invalid
            and access_valid
            and run_metadata["status"] == "completed"
            else "invalid"
        ),
        "expected_checkpoint_count": int(inference["expected_checkpoints"]),
        "valid_checkpoint_count": len(valid_runs),
        "expected_patient_count_per_checkpoint": int(
            inference["expected_patients"]
        ),
        "invalid_runs": invalid,
        "valid_runs": valid_runs,
        "access_event_count": len(access_events),
        "access_event_valid": access_valid,
        "run_metadata_status": run_metadata["status"],
        "test_manifest_sha256": file_digest(test_manifest),
    }
    if audit["status"] != "complete":
        raise RuntimeError(f"Gate 11 artifact audit failed: {audit}")
    return (
        pd.concat(metric_frames, ignore_index=True),
        pd.concat(latency_frames, ignore_index=True),
        audit,
    )


def _endpoint_names(plan: Mapping[str, Any]) -> list[str]:
    statistical_path = Path(str(_mapping(plan["gate10"], "gate10")["plan"]))
    import yaml

    statistical_root = yaml.safe_load(statistical_path.read_text(encoding="utf-8"))
    statistical = _mapping(
        _mapping(statistical_root, "statistics")["gate10"],
        "gate10",
    )
    endpoints = _mapping(statistical["endpoints"], "endpoints")
    primary = str(_mapping(endpoints["primary"], "primary")["name"])
    secondary = [str(value) for value in cast(Sequence[Any], endpoints["secondary"])]
    return [primary, *secondary]


def _candidate_aggregate(
    seed_metrics: pd.DataFrame,
    endpoints: Sequence[str],
) -> pd.DataFrame:
    missing = set(endpoints) - set(seed_metrics.columns)
    if missing:
        raise RuntimeError(f"Frozen endpoints are missing from artifacts: {missing}")
    rows: list[dict[str, Any]] = []
    for (candidate, patient), group in seed_metrics.groupby(
        ["candidate_id", "patient_id"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "candidate_id": str(candidate),
            "patient_id": str(patient),
            "seed_count": int(group["seed"].nunique()),
        }
        for endpoint in endpoints:
            values = group[endpoint].to_numpy(dtype=np.float64)
            row[endpoint] = (
                float(np.mean(values))
                if not np.isnan(values).any()
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _metric_summaries(
    candidate_metrics: pd.DataFrame,
    endpoints: Sequence[str],
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(
        sorted(candidate_metrics["candidate_id"].unique())
    ):
        subset = candidate_metrics.loc[
            candidate_metrics["candidate_id"] == candidate
        ]
        for endpoint_index, endpoint in enumerate(endpoints):
            rows.append(
                {
                    "candidate_id": candidate,
                    "metric": endpoint,
                    **_finite_summary(
                        subset[endpoint].to_numpy(dtype=np.float64),
                        resamples=resamples,
                        confidence_level=confidence_level,
                        seed=seed + candidate_index * 1000 + endpoint_index,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _comparisons(
    plan: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
    endpoints: Sequence[str],
    *,
    resamples: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    gate10_analysis = json.loads(
        Path(
            str(_mapping(plan["gate10"], "gate10")["analysis_freeze"])
        ).read_text(encoding="utf-8")
    )
    hypothesis = _mapping(gate10_analysis["hypothesis_testing"], "hypothesis")
    planned = cast(Sequence[Mapping[str, Any]], hypothesis["comparisons"])
    primary = str(_mapping(gate10_analysis["primary_endpoint"], "primary")["name"])
    rows: list[dict[str, Any]] = []
    primary_p_values: dict[str, float] = {}
    for comparison_index, comparison in enumerate(planned):
        first_name = str(comparison["first"])
        second_name = str(comparison["second"])
        first = candidate_metrics.loc[
            candidate_metrics["candidate_id"] == first_name
        ].set_index("patient_id")
        second = candidate_metrics.loc[
            candidate_metrics["candidate_id"] == second_name
        ].set_index("patient_id")
        common = sorted(set(first.index) & set(second.index))
        if len(common) != 74:
            raise RuntimeError("Frozen candidate comparison is not fully paired")
        for endpoint_index, endpoint in enumerate(endpoints):
            summary, differences = _comparison_summary(
                first.loc[common, endpoint],
                second.loc[common, endpoint],
                resamples=resamples,
                confidence_level=confidence_level,
                seed=bootstrap_seed + comparison_index * 1000 + endpoint_index,
            )
            formal = endpoint == primary
            p_value = (
                sign_flip_permutation_p_value(
                    differences,
                    resamples=int(hypothesis["resamples"]),
                    seed=int(hypothesis["seed"]) + comparison_index,
                )
                if formal
                else float("nan")
            )
            row = {
                "comparison_id": str(comparison["id"]),
                "comparison_role": str(comparison["role"]),
                "first_candidate": first_name,
                "second_candidate": second_name,
                "metric": endpoint,
                "formal_hypothesis_test": formal,
                **summary,
                "permutation_p_value": p_value,
                "holm_adjusted_p_value": float("nan"),
                "reject_holm_alpha_0_05": False,
            }
            rows.append(row)
            if formal:
                primary_p_values[str(comparison["id"])] = p_value
    adjusted = holm_adjust(primary_p_values)
    for row in rows:
        if row["formal_hypothesis_test"]:
            adjusted_value = adjusted[str(row["comparison_id"])]
            row["holm_adjusted_p_value"] = adjusted_value
            row["reject_holm_alpha_0_05"] = bool(adjusted_value < 0.05)
    return pd.DataFrame(rows)


def _cohort_groups(
    cohort: pd.DataFrame,
    gate10_analysis: Mapping[str, Any],
) -> pd.DataFrame:
    output = cohort.copy()
    output["grade_group"] = output["grade"].astype(str)
    output["et_group"] = np.where(
        output["et_voxel_count"].astype(float) > 0,
        "present",
        "absent",
    )
    thresholds = _mapping(
        _mapping(gate10_analysis["resolved_subgroups"], "subgroups")[
            "whole_tumor_burden_train_only_tertiles"
        ],
        "thresholds",
    )
    q1 = float(thresholds["q1_mm3"])
    q2 = float(thresholds["q2_mm3"])
    volumes = output["wt_volume_mm3"].astype(float)
    output["wt_burden_group"] = np.select(
        [volumes <= q1, volumes <= q2],
        ["small", "medium"],
        default="large",
    )
    return output


def _subgroup_rows(
    plan: Mapping[str, Any],
    candidate_metrics: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    gate10 = _mapping(plan["gate10"], "gate10")
    gate10_analysis = json.loads(
        Path(str(gate10["analysis_freeze"])).read_text(encoding="utf-8")
    )
    groups = _cohort_groups(cohort, gate10_analysis)
    enriched = candidate_metrics.merge(
        groups[
            [
                "subject_id",
                "grade_group",
                "et_group",
                "wt_burden_group",
            ]
        ],
        left_on="patient_id",
        right_on="subject_id",
        validate="many_to_one",
    )
    statistical_path = Path(str(gate10["plan"]))
    import yaml

    root = yaml.safe_load(statistical_path.read_text(encoding="utf-8"))
    statistical = _mapping(_mapping(root, "root")["gate10"], "gate10")
    minimum = int(_mapping(statistical["subgroups"], "subgroups")[
        "minimum_reportable_patient_count"
    ])
    definitions = (
        ("grade", "grade_group", ("HGG", "LGG")),
        ("enhancing_tumor_reference", "et_group", ("present", "absent")),
        (
            "whole_tumor_burden",
            "wt_burden_group",
            ("small", "medium", "large"),
        ),
    )
    rows: list[dict[str, Any]] = []
    row_index = 0
    for subgroup, column, categories in definitions:
        for category in categories:
            subset = enriched.loc[enriched[column] == category]
            for candidate in sorted(subset["candidate_id"].unique()):
                values = subset.loc[
                    subset["candidate_id"] == candidate,
                    "mean_regional_dice",
                ].to_numpy(dtype=np.float64)
                summary = _finite_summary(
                    values,
                    resamples=resamples,
                    confidence_level=confidence_level,
                    seed=seed + row_index,
                )
                rows.append(
                    {
                        "record_type": "candidate_estimate",
                        "subgroup": subgroup,
                        "category": category,
                        "candidate_id": candidate,
                        "comparison_id": "",
                        "patient_count": len(values),
                        "reportability": (
                            "exploratory"
                            if len(values) >= minimum
                            else "descriptive_insufficient_n"
                        ),
                        "mean_regional_dice": summary["mean_finite"],
                        "paired_mean_difference": float("nan"),
                        "bootstrap_lower": summary["bootstrap_lower_finite"],
                        "bootstrap_upper": summary["bootstrap_upper_finite"],
                    }
                )
                row_index += 1
            for comparison in cast(
                Sequence[Mapping[str, Any]],
                gate10_analysis["hypothesis_testing"]["comparisons"],
            ):
                first_name = str(comparison["first"])
                second_name = str(comparison["second"])
                first = subset.loc[
                    subset["candidate_id"] == first_name,
                    ["patient_id", "mean_regional_dice"],
                ].set_index("patient_id")
                second = subset.loc[
                    subset["candidate_id"] == second_name,
                    ["patient_id", "mean_regional_dice"],
                ].set_index("patient_id")
                common = sorted(set(first.index) & set(second.index))
                summary, _ = _comparison_summary(
                    first.loc[common, "mean_regional_dice"],
                    second.loc[common, "mean_regional_dice"],
                    resamples=resamples,
                    confidence_level=confidence_level,
                    seed=seed + row_index,
                )
                rows.append(
                    {
                        "record_type": "paired_difference",
                        "subgroup": subgroup,
                        "category": category,
                        "candidate_id": "",
                        "comparison_id": str(comparison["id"]),
                        "patient_count": len(common),
                        "reportability": (
                            "exploratory"
                            if len(common) >= minimum
                            else "descriptive_insufficient_n"
                        ),
                        "mean_regional_dice": float("nan"),
                        "paired_mean_difference": summary[
                            "paired_mean_difference"
                        ],
                        "bootstrap_lower": summary["paired_bootstrap_lower"],
                        "bootstrap_upper": summary["paired_bootstrap_upper"],
                    }
                )
                row_index += 1
    return pd.DataFrame(rows)


def _resource_rows(
    plan: Mapping[str, Any],
    latencies: pd.DataFrame,
) -> pd.DataFrame:
    gate10 = _mapping(plan["gate10"], "gate10")
    inference = _mapping(plan["inference"], "inference")
    checkpoint_manifest = json.loads(
        Path(str(gate10["checkpoint_manifest"])).read_text(encoding="utf-8")
    )
    artifact_root = Path(str(inference["artifact_root"]))
    rows: list[dict[str, Any]] = []
    for entry in cast(
        Sequence[Mapping[str, Any]],
        checkpoint_manifest["checkpoints"],
    ):
        candidate = str(entry["candidate_id"])
        seed = int(entry["seed"])
        values = latencies.loc[
            (latencies["candidate_id"].astype(str) == candidate)
            & (latencies["seed"].astype(int) == seed),
            "latency_seconds",
        ].to_numpy(dtype=np.float64)
        metadata = json.loads(
            (artifact_root / candidate / str(seed) / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        rows.append(
            {
                "candidate_id": candidate,
                "seed": seed,
                "patient_count": len(values),
                "parameter_count": int(entry["parameter_count"]),
                "checkpoint_size_bytes": int(entry["checkpoint_size_bytes"]),
                "development_gpu_hours": float(entry["gpu_hours"]),
                "development_peak_allocated_vram_bytes": int(
                    entry["peak_allocated_vram_bytes"]
                ),
                "development_peak_reserved_vram_bytes": int(
                    entry["peak_reserved_vram_bytes"]
                ),
                "macs_per_slice": int(metadata["macs_per_slice"]),
                "flops_per_slice": int(metadata["flops_per_slice"]),
                "latency_mean_seconds": float(np.mean(values)),
                "latency_p50_seconds": float(np.quantile(values, 0.5)),
                "latency_p95_seconds": float(np.quantile(values, 0.95)),
                "throughput_volumes_per_hour": float(3600.0 / np.mean(values)),
            }
        )
    return pd.DataFrame(rows)


def _qualitative_selection(
    candidate_metrics: pd.DataFrame,
    cohort: pd.DataFrame,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    bunet = candidate_metrics.loc[
        candidate_metrics["candidate_id"] == "bunet"
    ].merge(
        cohort[["subject_id", "et_voxel_count"]],
        left_on="patient_id",
        right_on="subject_id",
        validate="one_to_one",
    )
    present = bunet.loc[bunet["et_voxel_count"].astype(float) > 0].copy()
    if present.empty:
        raise RuntimeError("Qualitative selection requires ET-present patients")
    success = present.sort_values(
        ["mean_regional_dice", "patient_id"],
        ascending=[False, True],
    ).iloc[0]
    median = float(np.median(present["mean_regional_dice"].astype(float)))
    hard = (
        present.assign(
            distance_to_median=(
                present["mean_regional_dice"].astype(float) - median
            ).abs()
        )
        .sort_values(["distance_to_median", "patient_id"])
        .iloc[0]
    )
    failure = present.sort_values(
        ["et_dice", "patient_id"],
        ascending=[True, True],
    ).iloc[0]
    output: dict[str, Any] = {}
    for role, row in (
        ("success", success),
        ("hard", hard),
        ("failure", failure),
    ):
        patient_id = str(row["patient_id"])
        context = artifact_root / "qualitative" / patient_id / "context.npz"
        if not context.is_file():
            raise FileNotFoundError(context)
        output[role] = {
            "patient_id": patient_id,
            "bunet_mean_regional_dice": float(row["mean_regional_dice"]),
            "bunet_et_dice": float(row["et_dice"]),
            "context_path": context.as_posix(),
        }
    return output


def _completion_markdown(
    audit: Mapping[str, Any],
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    analysis: Mapping[str, Any],
) -> str:
    selected_metrics = ("mean_regional_dice", "wt_dice", "tc_dice", "et_dice")
    lines = [
        "# Gate 11 Completion",
        "",
        "**Decision:** PASS",
        "",
        "## Artifact and access audit",
        "",
        f"- Valid frozen checkpoints: {audit['valid_checkpoint_count']}/"
        f"{audit['expected_checkpoint_count']}",
        f"- Patients per checkpoint: {audit['expected_patient_count_per_checkpoint']}",
        f"- Guarded internal-test manifest openings: {audit['access_event_count']}",
        f"- Invalid runs: {len(audit['invalid_runs'])}",
        "",
        "## Internal held-out test estimates",
        "",
        "| Candidate | Mean regional Dice | WT Dice | TC Dice | ET Dice |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in ("unet_reference", "bunet", "unet_res"):
        values: dict[str, float] = {}
        for metric in selected_metrics:
            selected = summaries.loc[
                (summaries["candidate_id"] == candidate)
                & (summaries["metric"] == metric)
            ]
            values[metric] = float(selected.iloc[0]["mean_finite"])
        lines.append(
            f"| {candidate} | {values['mean_regional_dice']:.6f} | "
            f"{values['wt_dice']:.6f} | {values['tc_dice']:.6f} | "
            f"{values['et_dice']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen primary-endpoint comparisons",
            "",
            "| Comparison | Mean difference | 95% paired bootstrap CI | "
            "Raw p | Holm p | Reject |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
    )
    primary = comparisons.loc[comparisons["formal_hypothesis_test"].astype(bool)]
    for comparison_row in primary.itertuples(index=False):
        lines.append(
            f"| {comparison_row.comparison_id} | "
            f"{comparison_row.paired_mean_difference:.6f} | "
            f"[{comparison_row.paired_bootstrap_lower:.6f}, "
            f"{comparison_row.paired_bootstrap_upper:.6f}] | "
            f"{comparison_row.permutation_p_value:.6f} | "
            f"{comparison_row.holm_adjusted_p_value:.6f} | "
            f"{'yes' if comparison_row.reject_holm_alpha_0_05 else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "All frozen candidates and seeds are reported. Secondary endpoints and "
            "subgroups are estimation-only. These results are from one internal "
            "held-out subset on a single dataset and do not establish external "
            "generalization or clinical applicability.",
            "",
            "Predeclared qualitative cases: "
            + ", ".join(
                f"{role}={details['patient_id']}"
                for role, details in cast(
                    Mapping[str, Mapping[str, Any]],
                    analysis["qualitative_cases"],
                ).items()
            ),
        ]
    )
    return "\n".join(lines)


def analyze_gate11(plan_path: Path) -> dict[str, Any]:
    """Audit Gate 11 and write every report from machine-readable artifacts."""
    plan = load_gate11_plan(plan_path)
    outputs = _mapping(plan["outputs"], "outputs")
    seed_metrics, latencies, audit = _load_artifacts(plan)
    endpoints = _endpoint_names(plan)
    candidate_metrics = _candidate_aggregate(seed_metrics, endpoints)
    artifact_root = Path(str(_mapping(plan["inference"], "inference")[
        "artifact_root"
    ]))
    cohort = pd.read_csv(artifact_root / "cohort_metadata.csv")
    gate10_plan_path = Path(str(_mapping(plan["gate10"], "gate10")["plan"]))
    import yaml

    statistical_root = yaml.safe_load(
        gate10_plan_path.read_text(encoding="utf-8")
    )
    statistical = _mapping(
        _mapping(statistical_root, "statistical root")["gate10"],
        "gate10",
    )
    estimation = _mapping(statistical["estimation"], "estimation")
    bootstrap = _mapping(estimation["paired_bootstrap"], "paired bootstrap")
    resamples = int(bootstrap["resamples"])
    confidence_level = float(estimation["confidence_level"])
    seed = int(bootstrap["seed"])
    summaries = _metric_summaries(
        candidate_metrics,
        endpoints,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    comparisons = _comparisons(
        plan,
        candidate_metrics,
        endpoints,
        resamples=resamples,
        confidence_level=confidence_level,
        bootstrap_seed=seed,
    )
    subgroups = _subgroup_rows(
        plan,
        candidate_metrics,
        cohort,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed + 50_000,
    )
    resources = _resource_rows(plan, latencies)
    qualitative = _qualitative_selection(
        candidate_metrics,
        cohort,
        artifact_root=artifact_root,
    )
    gate10_analysis_path = Path(
        str(_mapping(plan["gate10"], "gate10")["analysis_freeze"])
    )
    analysis: dict[str, Any] = {
        "status": "complete",
        "gate": 11,
        "audit": audit,
        "statistical_unit": "patient",
        "primary_endpoint": "mean_regional_dice",
        "seed_aggregation": (
            "per_patient_arithmetic_mean_before_patient_level_inference"
        ),
        "gate10_analysis_freeze_sha256": file_digest(gate10_analysis_path),
        "gate11_plan_sha256": file_digest(plan_path),
        "qualitative_cases": qualitative,
        "clinical_applicability_established": False,
        "external_generalization_established": False,
    }
    atomic_write_csv(Path(str(outputs["patient_seed_metrics"])), _records(seed_metrics))
    atomic_write_csv(
        Path(str(outputs["patient_candidate_metrics"])),
        _records(candidate_metrics),
    )
    atomic_write_csv(Path(str(outputs["metric_summary"])), _records(summaries))
    atomic_write_csv(Path(str(outputs["comparisons"])), _records(comparisons))
    atomic_write_csv(Path(str(outputs["subgroups"])), _records(subgroups))
    atomic_write_csv(Path(str(outputs["resources"])), _records(resources))
    atomic_write_json(Path(str(outputs["audit"])), audit)
    atomic_write_json(Path(str(outputs["analysis"])), analysis)
    atomic_write_text(
        Path(str(outputs["completion"])),
        _completion_markdown(audit, summaries, comparisons, analysis),
    )
    return analysis
