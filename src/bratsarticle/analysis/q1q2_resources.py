"""Fail-closed measured resource and accuracy-cost analysis for Q1/Q2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], frame.to_dict(orient="records"))


def pareto_flags(
    frame: pd.DataFrame,
    *,
    accuracy_column: str,
    cost_columns: list[str],
) -> np.ndarray:
    """Return non-dominated flags for higher accuracy and lower measured costs."""
    required = {accuracy_column, *cost_columns}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Pareto table misses columns: {sorted(missing)}")
    values = frame[[accuracy_column, *cost_columns]].to_numpy(dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Pareto analysis requires complete finite measurements")
    flags = np.ones(len(values), dtype=bool)
    for index, current in enumerate(values):
        for candidate_index, candidate in enumerate(values):
            if candidate_index == index:
                continue
            accuracy_not_worse = candidate[0] >= current[0]
            costs_not_worse = bool(np.all(candidate[1:] <= current[1:]))
            strictly_better = candidate[0] > current[0] or bool(
                np.any(candidate[1:] < current[1:])
            )
            if accuracy_not_worse and costs_not_worse and strictly_better:
                flags[index] = False
                break
    return flags


def _native_static_profile(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    static = cast(dict[str, Any], metadata.get("static_profile", {}))
    if "flops_per_slice" in static:
        return {
            "parameter_count": int(static["parameter_count"]),
            "flops_per_declared_input": int(static["flops_per_slice"]),
            "mac_equivalents_per_declared_input": int(static["macs_per_slice"]),
            "largest_single_activation_bytes": int(
                static["largest_single_activation_bytes"]
            ),
            "input_shape": json.dumps(static["input_shape"]),
            "output_shape": json.dumps(static["output_shape"]),
            "receptive_field_proxy": json.dumps(
                {
                    "pixels": static["receptive_field_proxy_pixels"],
                    "definition": "longest_path_axial_proxy",
                },
                sort_keys=True,
            ),
        }
    return {
        "parameter_count": int(static["parameter_count"]),
        "flops_per_declared_input": int(static["flops_per_input"]),
        "mac_equivalents_per_declared_input": int(
            static["mac_equivalents_per_input"]
        ),
        "largest_single_activation_bytes": int(
            static["largest_single_activation_bytes"]
        ),
        "input_shape": json.dumps(static["input_shape"]),
        "output_shape": json.dumps(static["output_shape"]),
        "receptive_field_proxy": json.dumps(
            {
                "voxels_per_axis": static[
                    "receptive_field_proxy_voxels_per_axis"
                ],
                "definition": static["receptive_field_proxy_definition"],
            },
            sort_keys=True,
        ),
    }


def _environment_hashes(metadata: dict[str, Any], adapter: str) -> tuple[str, str, str]:
    if adapter == "official_nnunetv2":
        return (
            str(metadata["environment_lock_sha256"]),
            str(metadata["requirements_lock_sha256"]),
            str(metadata["hardware_preflight_sha256"]),
        )
    hashes = cast(dict[str, Any], metadata["hashes"])
    return (
        str(hashes["environment_lock"]),
        str(hashes["requirements_lock"]),
        str(hashes["hardware_preflight"]),
    )


def _development_resource_row(
    run: dict[str, Any],
    *,
    expected_environment_hashes: tuple[str, str, str],
) -> dict[str, Any]:
    resource_path = Path(str(run["resource_profile_path"]))
    if (
        not resource_path.is_file()
        or file_digest(resource_path) != run["resource_profile_sha256"]
    ):
        raise RuntimeError(f"Frozen resource artifact differs: {run['run_id']}")
    resource = _load_json(resource_path)
    adapter = str(run["adapter"])
    if adapter == "official_nnunetv2":
        metadata = resource
        if resource.get("device") != "mps":
            raise RuntimeError(f"nnU-Net resource device differs: {run['run_id']}")
        peak_framework = int(
            resource["framework_peak_allocated_unified_memory_bytes"]
        )
        peak_driver = int(resource["driver_peak_allocated_unified_memory_bytes"])
    else:
        metadata = _load_json(resource_path.parent / "metadata.json")
        hardware = cast(dict[str, Any], metadata.get("hardware", {}))
        if hardware.get("backend") != "mps":
            raise RuntimeError(f"Native resource device differs: {run['run_id']}")
        peak_framework = int(resource["peak_memory_allocated_bytes"])
        peak_driver = int(resource["peak_memory_reserved_or_driver_bytes"])
    if _environment_hashes(metadata, adapter) != expected_environment_hashes:
        raise RuntimeError(f"Run environment contract differs: {run['run_id']}")
    static = _native_static_profile(metadata)
    checkpoint_path = Path(str(run["best_checkpoint_path"]))
    if (
        not checkpoint_path.is_file()
        or file_digest(checkpoint_path) != run["best_checkpoint_sha256"]
    ):
        raise RuntimeError(f"Frozen best checkpoint differs: {run['run_id']}")
    timing_count = int(resource["synchronized_training_step_measurement_count"])
    if (
        timing_count != 100
        or float(resource["synchronized_training_step_p50_seconds"]) <= 0.0
        or float(resource["synchronized_training_step_p95_seconds"]) <= 0.0
        or peak_framework <= 0
        or peak_driver <= 0
    ):
        raise RuntimeError(f"Training resource profile is incomplete: {run['run_id']}")
    return {
        "run_id": str(run["run_id"]),
        "model_id": str(run["model_id"]),
        "adapter": adapter,
        "stage": str(run["stage"]),
        "fold": int(run["fold"]),
        "seed": int(run["seed"]),
        "loss": str(run["loss"]),
        "git_commit": str(run["git_commit"]),
        **static,
        "best_checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "training_peak_framework_allocated_unified_memory_bytes": peak_framework,
        "training_peak_driver_allocated_unified_memory_bytes": peak_driver,
        "training_step_measurement_count": timing_count,
        "training_step_p50_seconds": float(
            resource["synchronized_training_step_p50_seconds"]
        ),
        "training_step_p95_seconds": float(
            resource["synchronized_training_step_p95_seconds"]
        ),
        "training_step_mean_seconds": float(
            resource["synchronized_training_step_mean_seconds"]
        ),
        "training_accelerator_hours": float(run["accelerator_hours"]),
        "completed_optimizer_steps": run["completed_optimizer_steps"],
        "completed_epochs": run["completed_epochs"],
        "memory_terminology": (
            "MPS framework-reported allocated unified memory; "
            "MPS driver-allocated unified memory"
        ),
    }


def _model_resource_summary(
    main: pd.DataFrame,
    *,
    expected_runs_per_model: int,
) -> pd.DataFrame:
    static_columns = [
        "adapter",
        "parameter_count",
        "flops_per_declared_input",
        "mac_equivalents_per_declared_input",
        "largest_single_activation_bytes",
        "input_shape",
        "output_shape",
        "receptive_field_proxy",
    ]
    rows: list[dict[str, Any]] = []
    for model_id, group in main.groupby("model_id", sort=True):
        if len(group) != expected_runs_per_model:
            raise RuntimeError(f"{model_id} does not have 25 main resource runs")
        for column in static_columns:
            if group[column].nunique(dropna=False) != 1:
                raise RuntimeError(
                    f"{model_id} static resource field differs: {column}"
                )
        rows.append(
            {
                "model_id": model_id,
                **{column: group.iloc[0][column] for column in static_columns},
                "main_run_count": len(group),
                "best_checkpoint_size_bytes_mean": float(
                    group["best_checkpoint_size_bytes"].mean()
                ),
                "training_peak_framework_allocated_unified_memory_bytes_max": int(
                    group[
                        "training_peak_framework_allocated_unified_memory_bytes"
                    ].max()
                ),
                "training_peak_driver_allocated_unified_memory_bytes_max": int(
                    group[
                        "training_peak_driver_allocated_unified_memory_bytes"
                    ].max()
                ),
                "training_step_p50_seconds_median_across_runs": float(
                    group["training_step_p50_seconds"].median()
                ),
                "training_step_p95_seconds_median_across_runs": float(
                    group["training_step_p95_seconds"].median()
                ),
                "training_accelerator_hours_mean": float(
                    group["training_accelerator_hours"].mean()
                ),
                "training_accelerator_hours_sum": float(
                    group["training_accelerator_hours"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def analyze_q1q2_resources(
    *,
    config_path: Path = Path("configs/q1q2_v2/resource_execution.yaml"),
) -> dict[str, Any]:
    """Join frozen development, inference, and accuracy artifacts."""
    config = _load_yaml(config_path)
    if config.get("status") != "frozen_before_external_results":
        raise PermissionError("Resource analysis protocol is not frozen")
    expected = cast(dict[str, Any], config["expected"])
    gate_g_path = Path(str(config["gate_g_analysis_freeze"]))
    manifest_path = Path(str(config["gate_g_checkpoint_manifest"]))
    gate_h_path = Path(str(config["gate_h_completion"]))
    gate_g = _load_json(gate_g_path)
    manifest = _load_json(manifest_path)
    gate_h = _load_json(gate_h_path)
    if (
        gate_g.get("status") != "frozen_external_inference_permitted"
        or gate_g.get("checkpoint_manifest_sha256") != file_digest(manifest_path)
        or manifest.get("status") != "frozen"
        or int(manifest.get("run_count", -1))
        != int(expected["all_development_runs"])
        or gate_h.get("status") != "pass"
        or gate_h.get("gate_h_pass") is not True
    ):
        raise PermissionError("Resource analysis gates are incomplete")
    environment_paths = (
        Path(str(config["environment_lock"])),
        Path(str(config["requirements_lock"])),
        Path(str(config["hardware_preflight"])),
    )
    expected_environment_hashes = (
        file_digest(environment_paths[0]),
        file_digest(environment_paths[1]),
        file_digest(environment_paths[2]),
    )
    frozen_hashes = cast(dict[str, str], gate_g["analysis_input_sha256"])
    frozen_inputs = (
        config_path,
        Path(str(config["resource_profile_protocol"])),
        *environment_paths,
    )
    for path in frozen_inputs:
        if frozen_hashes.get(path.as_posix()) != file_digest(path):
            raise PermissionError(f"Frozen resource-analysis input differs: {path}")
    for path, digest in zip(
        environment_paths,
        expected_environment_hashes,
        strict=True,
    ):
        if frozen_hashes.get(path.as_posix()) != digest:
            raise PermissionError(f"Frozen environment artifact differs: {path}")

    runs = cast(list[dict[str, Any]], manifest["runs"])
    rows = [
        _development_resource_row(
            run,
            expected_environment_hashes=expected_environment_hashes,
        )
        for run in runs
    ]
    development = pd.DataFrame(rows)
    main = development.loc[development["stage"].eq("main_convergence")].copy()
    if len(main) != int(expected["main_development_runs"]):
        raise RuntimeError("Main development resource matrix is incomplete")
    model_resources = _model_resource_summary(
        main,
        expected_runs_per_model=int(expected["main_runs_per_model"]),
    )
    if len(model_resources) != int(expected["models"]):
        raise RuntimeError("Model resource summary does not contain 12 models")

    inference_path = Path(str(gate_h["model_inference_resources"]))
    metrics_path = Path(str(gate_h["model_patient_metrics"]))
    if (
        file_digest(inference_path) != gate_h["model_inference_resources_sha256"]
        or file_digest(metrics_path) != gate_h["model_patient_metrics_sha256"]
    ):
        raise RuntimeError("Gate H resource or metric artifact differs")
    inference = pd.read_csv(inference_path)
    metrics = pd.read_csv(metrics_path)
    confirmatory = metrics.loc[
        metrics["cohort_role"].eq("external_confirmatory")
    ]
    accuracy = (
        confirmatory.groupby("model_id", sort=True)["mean_regional_dice"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "external_confirmatory_patient_mean_regional_dice",
                "count": "external_confirmatory_patient_count",
            }
        )
    )
    if (
        len(accuracy) != int(expected["models"])
        or not accuracy["external_confirmatory_patient_count"]
        .eq(int(expected["external_confirmatory_patients_per_model"]))
        .all()
    ):
        raise RuntimeError("External confirmatory accuracy matrix is incomplete")
    joined = model_resources.merge(
        inference,
        on="model_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        accuracy,
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    joined = joined.rename(
        columns={
            "end_to_end_p50_seconds": "inference_end_to_end_p50_seconds",
            "end_to_end_p95_seconds": "inference_end_to_end_p95_seconds",
        }
    )
    accuracy_column = "external_confirmatory_patient_mean_regional_dice"
    cost_columns = [
        str(value)
        for value in cast(dict[str, Any], config["pareto"])["costs"]
    ]
    for cost in cost_columns:
        joined[f"pareto_accuracy_vs_{cost}"] = pareto_flags(
            joined,
            accuracy_column=accuracy_column,
            cost_columns=[cost],
        )
    joined["pareto_all_measured_cost_dimensions"] = pareto_flags(
        joined,
        accuracy_column=accuracy_column,
        cost_columns=cost_columns,
    )

    control_ids = {
        "unet_res",
        "unet_parameter_matched_res",
        "unet_compute_matched_res",
    }
    controls = joined.loc[joined["model_id"].isin(control_ids)].copy()
    if set(controls["model_id"]) != control_ids:
        raise RuntimeError("Parameter/compute matching control table is incomplete")
    sensitivity = development.loc[
        development["stage"].isin(["main_compute_matched", "loss_interaction"])
    ].copy()
    sensitivity_counts = sensitivity["stage"].value_counts().to_dict()
    if sensitivity_counts != {
        "main_compute_matched": 200,
        "loss_interaction": 100,
    }:
        raise RuntimeError("Development sensitivity resource matrix is incomplete")
    outputs = cast(dict[str, Any], config["outputs"])
    output_frames = {
        "development_run_resources": development,
        "main_model_resources": model_resources,
        "sensitivity_run_resources": sensitivity,
        "matching_control_table": controls,
        "accuracy_cost_pareto": joined,
    }
    hashes: dict[str, str] = {}
    for key, frame in output_frames.items():
        path = Path(str(outputs[key]))
        atomic_write_csv(path, _records(frame))
        hashes[path.as_posix()] = file_digest(path)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "manual_resource_values_used": False,
        "subjective_cost_score_used": False,
        "same_device_contract_verified": True,
        "same_software_environment_contract_verified": True,
        "development_run_count": len(development),
        "main_model_count": len(model_resources),
        "external_model_count": len(joined),
        "memory_terminology": (
            "MPS framework-reported allocated unified memory; "
            "MPS driver-allocated unified memory"
        ),
        "artifacts": hashes,
    }
    completion_path = Path(str(outputs["completion"]))
    atomic_write_json(completion_path, completion)
    return completion


__all__ = ["analyze_q1q2_resources", "pareto_flags"]
