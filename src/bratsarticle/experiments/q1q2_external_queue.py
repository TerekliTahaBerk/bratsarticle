"""Single-session, restart-audited Gate H external inference queue."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import torch
import yaml

from bratsarticle.data.external_audit import resolve_brats_africa_data_root
from bratsarticle.data.external_dataset import (
    ExternalVolumeDataset,
    verify_external_files,
)
from bratsarticle.experiments.q1q2_external_inference import (
    native_model_config,
    predict_native_external_checkpoint,
    predict_nnunet_external_checkpoint,
    predict_swin_external_checkpoint,
    prepare_nnunet_external_input,
)
from bratsarticle.experiments.registry import ResourceTracker
from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
)
from evaluation import CentralEvaluator, load_evaluation_config


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return cast(dict[str, Any], loaded)


def _config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    config = cast(dict[str, Any], loaded)
    if config.get("status") != "frozen_before_external_results":
        raise PermissionError("Gate H protocol is not frozen before external results")
    return config


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def _validate_gate_g(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    freeze_path = Path(str(config["gate_g_analysis_freeze"]))
    manifest_path = Path(str(config["gate_g_checkpoint_manifest"]))
    if not freeze_path.is_file() or not manifest_path.is_file():
        raise PermissionError("Gate G analysis/checkpoint freeze is absent")
    freeze = _load_json(freeze_path)
    manifest = _load_json(manifest_path)
    if (
        freeze.get("status") != "frozen_external_inference_permitted"
        or freeze.get("external_inference_permitted") is not True
        or freeze.get("external_retuning_permitted") is not False
        or freeze.get("checkpoint_manifest_sha256") != file_digest(manifest_path)
        or manifest.get("status") != "frozen"
        or int(manifest.get("run_count", -1)) != 600
    ):
        raise PermissionError("Gate G freeze integrity is invalid")
    frozen_inputs = cast(dict[str, str], freeze["analysis_input_sha256"])
    for key in (
        "external_inventory",
        "external_confirmatory_manifest",
        "evaluation",
        "model_matrix",
    ):
        path = Path(str(config[key]))
        if frozen_inputs.get(path.as_posix()) != file_digest(path):
            raise PermissionError(f"Gate H input differs from Gate G: {path}")
    runs = [
        cast(dict[str, Any], run)
        for run in cast(list[Any], manifest["runs"])
        if cast(dict[str, Any], run).get("stage") == "main_convergence"
    ]
    expected = cast(dict[str, Any], config["expected"])
    if len(runs) != int(expected["main_checkpoint_count"]):
        raise ValueError("Gate G does not contain exactly 300 main checkpoints")
    identities = {
        (str(run["model_id"]), int(run["fold"]), int(run["seed"])) for run in runs
    }
    models = {identity[0] for identity in identities}
    if (
        len(identities) != len(runs)
        or len(models) != int(expected["model_count"])
        or any(
            sum(str(run["model_id"]) == model for run in runs)
            != int(expected["checkpoints_per_model"])
            for model in models
        )
    ):
        raise ValueError("Gate H main checkpoint identity matrix differs")
    return freeze, sorted(
        runs,
        key=lambda run: (
            str(run["model_id"]),
            int(run["fold"]),
            int(run["seed"]),
        ),
    )


def _prior_external_inference_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        cast(dict[str, Any], json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("model_inference") is True
    ]


def _open_session(
    *,
    config_path: Path,
    config: dict[str, Any],
    freeze: dict[str, Any],
    external_root: Path,
    allow_frozen_external_inference: bool,
) -> dict[str, Any]:
    if not allow_frozen_external_inference:
        raise PermissionError("Gate H requires --allow-frozen-external-inference")
    artifacts = cast(dict[str, Any], config["artifacts"])
    runtime_path = Path(str(artifacts["runtime"]))
    access_log = Path(str(artifacts["access_log"]))
    session_id = text_digest(
        "|".join(
            (
                file_digest(config_path),
                file_digest(Path(str(config["gate_g_analysis_freeze"]))),
                file_digest(Path(str(config["external_inventory"]))),
            )
        )
    )[:24]
    if runtime_path.is_file():
        runtime = _load_json(runtime_path)
        if runtime.get("session_id") != session_id:
            raise PermissionError("A different external session already exists")
        if runtime.get("status") in {"failed", "completed_with_failures"}:
            raise PermissionError("Failed external inference cannot be rerun")
        if runtime.get("external_root") != external_root.as_posix():
            raise PermissionError(
                "External session root differs from its first opening"
            )
        return runtime
    prior = _prior_external_inference_events(access_log)
    if prior:
        raise PermissionError("External model inference was already opened")
    runtime = {
        "schema_version": 1,
        "status": "running",
        "session_id": session_id,
        "started_at_utc": _timestamp(),
        "protocol_path": config_path.as_posix(),
        "protocol_sha256": file_digest(config_path),
        "gate_g_analysis_freeze_sha256": file_digest(
            Path(str(config["gate_g_analysis_freeze"]))
        ),
        "gate_g_checkpoint_manifest_sha256": str(freeze["checkpoint_manifest_sha256"]),
        "external_root": external_root.as_posix(),
        "external_root_not_persisted_in_public_reports": True,
        "external_data_accessed": True,
        "model_inference": True,
        "prediction_metrics_accessed": True,
        "retuning_permitted": False,
        "completed_checkpoint_count": 0,
        "failed_checkpoint_count": 0,
    }
    atomic_write_json(runtime_path, runtime)
    append_jsonl(
        access_log,
        {
            "event": "gate_h_frozen_external_session_started",
            "session_id": session_id,
            "cohort": "BraTS-Africa-TCIA-v1",
            "model_inference": True,
            "prediction_metrics_accessed": True,
            "retuning_permitted": False,
            "gate_g_freeze_sha256": runtime["gate_g_analysis_freeze_sha256"],
        },
    )
    return runtime


def _run_status(run_directory: Path) -> str:
    path = run_directory / "runtime.json"
    if not path.is_file():
        return "not_started"
    return str(_load_json(path).get("status", "unknown"))


@contextmanager
def _exclusive_gate_h_lock() -> Iterator[None]:
    runtime_root = Path("artifacts/q1q2_v2/queue_runtime")
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "gate_h_external.lock"
    active_locks = [
        path.name
        for path in runtime_root.glob("*.lock")
        if path.is_file() and path != lock_path
    ]
    if active_locks:
        raise RuntimeError(
            "Gate H cannot overlap another MPS queue: "
            + ", ".join(sorted(active_locks))
        )
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "Gate H external queue lock already exists; verify the existing process"
        ) from error
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
    finally:
        os.close(descriptor)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    run: dict[str, Any],
    expected_patient_count: int,
) -> None:
    if len(rows) != expected_patient_count:
        raise RuntimeError(
            f"External run {run['run_id']} returned {len(rows)} patient rows"
        )
    patient_ids = {str(row["patient_id"]) for row in rows}
    if len(patient_ids) != expected_patient_count:
        raise RuntimeError("External patient identities are missing or duplicated")
    if any(
        row["evaluation_stage"] != "raw"
        or row["model_id"] != run["model_id"]
        or int(row["training_fold"]) != int(run["fold"])
        or int(row["training_seed"]) != int(run["seed"])
        or row["checkpoint_sha256"] != run["best_checkpoint_sha256"]
        for row in rows
    ):
        raise RuntimeError("External metric identity differs from Gate G")
    roles = [str(row["cohort_role"]) for row in rows]
    if (
        roles.count("external_confirmatory") != 95
        or roles.count("external_supportive_other_neoplasm") != 51
    ):
        raise RuntimeError("External confirmatory/supportive roles differ")


def _run_one(
    *,
    run: dict[str, Any],
    dataset: ExternalVolumeDataset,
    evaluator: CentralEvaluator,
    config: dict[str, Any],
    run_directory: Path,
    nnunet_input: Path,
) -> None:
    run_directory.mkdir(parents=True, exist_ok=True)
    runtime_path = run_directory / "runtime.json"
    status = _run_status(run_directory)
    if status == "completed":
        stored = _load_json(runtime_path)
        metrics_path = Path(str(stored["patient_metrics"]))
        if (
            not metrics_path.is_file()
            or file_digest(metrics_path) != stored["patient_metrics_sha256"]
        ):
            raise RuntimeError("Completed external metric artifact differs")
        return
    if status == "failed":
        return
    attempt = 1
    if status in {"running", "unknown"}:
        attempt = int(_load_json(runtime_path).get("attempt", 1)) + 1
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "attempt": attempt,
        "attempt_policy": (
            "same_session_operational_resume_without_prior_metric_artifact"
        ),
        "run": run,
        "started_at_utc": _timestamp(),
        "external_data_accessed": True,
        "legacy_internal_test_accessed": False,
        "retuning_permitted": False,
    }
    atomic_write_json(runtime_path, report)
    tracker = ResourceTracker(torch.device("mps"))
    try:
        adapter = str(run["adapter"])
        if adapter == "native_configurable_unet":
            native_config = _load_yaml(Path(str(config["native_runner"])))
            rows = predict_native_external_checkpoint(
                run=run,
                dataset=dataset,
                evaluator=evaluator,
                model_config_path=native_model_config(
                    Path(str(config["model_matrix"])),
                    str(run["model_id"]),
                ),
                preprocessing_config_path=Path(
                    str(cast(dict[str, Any], native_config["data"])["preprocessing"])
                ),
                validation_batch_size=int(
                    cast(dict[str, Any], native_config["data"])["validation_batch_size"]
                ),
                device=torch.device("mps"),
            )
        elif adapter == "monai_swinunetr":
            swin_config = _load_yaml(Path(str(config["swin_runner"])))
            validation = cast(dict[str, Any], swin_config["validation"])
            rows = predict_swin_external_checkpoint(
                run=run,
                dataset=dataset,
                evaluator=evaluator,
                model_config_path=Path(
                    str(cast(dict[str, Any], swin_config["model"])["config"])
                ),
                overlap=float(validation["sliding_window_overlap"]),
                mode=str(validation["sliding_window_mode"]),
                sliding_window_batch_size=int(validation["sliding_window_batch_size"]),
                device=torch.device("mps"),
            )
        elif adapter == "official_nnunetv2":
            rows = predict_nnunet_external_checkpoint(
                run=run,
                dataset=dataset,
                evaluator=evaluator,
                input_directory=nnunet_input,
                queue_path=Path(str(config["nnunet_queue"])),
                environment=os.environ.copy(),
                log_directory=run_directory,
            )
        else:
            raise ValueError(f"Unknown Gate H adapter: {adapter}")
        expected = cast(dict[str, Any], config["expected"])
        _validate_rows(
            rows,
            run=run,
            expected_patient_count=int(expected["total_patients"]),
        )
        metrics_path = run_directory / "patient_metrics.csv"
        atomic_write_csv(metrics_path, rows)
        report.update(
            {
                "status": "completed",
                "finished_at_utc": _timestamp(),
                "patient_count": len(rows),
                "patient_metrics": metrics_path.as_posix(),
                "patient_metrics_sha256": file_digest(metrics_path),
                "resource_profile": tracker.snapshot(),
            }
        )
    except Exception:
        report.update(
            {
                "status": "failed",
                "finished_at_utc": _timestamp(),
                "error": traceback.format_exc(),
                "failure_policy": "report_without_retry_or_replacement",
                "resource_profile": tracker.snapshot(),
            }
        )
        atomic_write_json(runtime_path, report)
        return
    atomic_write_json(runtime_path, report)


def _aggregate(
    *,
    runs: list[dict[str, Any]],
    artifact_root: Path,
    checkpoint_output: Path,
    model_output: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    for run in runs:
        runtime_path = artifact_root / str(run["run_id"]) / "runtime.json"
        runtime = _load_json(runtime_path)
        if runtime.get("status") != "completed":
            failed.append(str(run["run_id"]))
            continue
        metrics_path = Path(str(runtime["patient_metrics"]))
        if file_digest(metrics_path) != runtime["patient_metrics_sha256"]:
            raise RuntimeError("External checkpoint metric hash differs at aggregation")
        frames.append(pd.read_csv(metrics_path))
    if not frames:
        raise RuntimeError("No external checkpoint completed")
    checkpoint_frame = pd.concat(frames, ignore_index=True)
    checkpoint_frame = checkpoint_frame.sort_values(
        ["model_id", "patient_id", "training_fold", "training_seed"]
    ).reset_index(drop=True)
    atomic_write_csv(
        checkpoint_output,
        cast(list[dict[str, Any]], checkpoint_frame.to_dict(orient="records")),
    )
    group_columns = [
        "model_id",
        "patient_id",
        "evaluation_stage",
        "cohort_role",
        "disease_group",
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
    ]
    excluded_numeric = {
        "training_fold",
        "training_seed",
        "spacing_axis0_mm",
        "spacing_axis1_mm",
        "spacing_axis2_mm",
    }
    numeric = [
        column
        for column in checkpoint_frame.select_dtypes(include="number").columns
        if column not in excluded_numeric and column not in group_columns
    ]
    grouped = checkpoint_frame.groupby(group_columns, dropna=False, sort=True)
    model_frame = grouped[numeric].mean().reset_index()
    model_frame["valid_checkpoint_count"] = grouped.size().to_numpy()
    spacing = grouped[
        ["spacing_axis0_mm", "spacing_axis1_mm", "spacing_axis2_mm"]
    ].first()
    model_frame = model_frame.merge(
        spacing.reset_index(),
        on=group_columns,
        how="left",
        validate="one_to_one",
    )
    model_frame = model_frame.sort_values(["model_id", "patient_id"]).reset_index(
        drop=True
    )
    atomic_write_csv(
        model_output,
        cast(list[dict[str, Any]], model_frame.to_dict(orient="records")),
    )
    required_replicates = int(expected["checkpoints_per_model"])
    complete_groups = bool(
        model_frame["valid_checkpoint_count"].eq(required_replicates).all()
    )
    return {
        "completed_checkpoint_count": len(frames),
        "failed_checkpoint_count": len(failed),
        "failed_run_ids": failed,
        "checkpoint_patient_row_count": len(checkpoint_frame),
        "model_patient_row_count": len(model_frame),
        "all_model_patient_groups_have_25_checkpoints": complete_groups,
        "checkpoint_patient_metrics": checkpoint_output.as_posix(),
        "checkpoint_patient_metrics_sha256": file_digest(checkpoint_output),
        "model_patient_metrics": model_output.as_posix(),
        "model_patient_metrics_sha256": file_digest(model_output),
    }


def _run_gate_h_external_queue_locked(
    *,
    external_root: Path,
    allow_frozen_external_inference: bool,
    config_path: Path = Path("configs/q1q2_v2/gate_h_external.yaml"),
) -> dict[str, Any]:
    """Run or resume the only frozen external inference session."""
    config = _config(config_path)
    freeze, runs = _validate_gate_g(config)
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("Gate H M1 execution requires available MPS")
    data_root = resolve_brats_africa_data_root(external_root)
    artifacts = cast(dict[str, Any], config["artifacts"])
    output_paths = [
        Path(str(artifacts[key]))
        for key in (
            "root",
            "runtime",
            "cache_root",
            "nnunet_input",
            "checkpoint_patient_metrics",
            "model_patient_metrics",
            "completion",
        )
    ]
    assert_output_paths_safe(output_paths, [data_root])
    runtime = _open_session(
        config_path=config_path,
        config=config,
        freeze=freeze,
        external_root=data_root,
        allow_frozen_external_inference=allow_frozen_external_inference,
    )
    if runtime.get("status") == "completed":
        return runtime
    verification = verify_external_files(
        data_root=data_root,
        inventory_path=Path(str(config["external_inventory"])),
    )
    dataset = ExternalVolumeDataset(
        data_root=data_root,
        inventory_path=Path(str(config["external_inventory"])),
        cache_root=Path(str(artifacts["cache_root"])),
    )
    cache = dataset.materialize()
    evaluator = CentralEvaluator(
        load_evaluation_config(Path(str(config["evaluation"])))
    )
    artifact_root = Path(str(artifacts["root"]))
    artifact_root.mkdir(parents=True, exist_ok=True)
    nnunet_runs = [run for run in runs if run["adapter"] == "official_nnunetv2"]
    if nnunet_runs:
        nnunet_preparation = prepare_nnunet_external_input(
            dataset=dataset,
            destination=Path(str(artifacts["nnunet_input"])),
        )
    else:
        nnunet_preparation = {"status": "not_required"}
    for run in runs:
        _run_one(
            run=run,
            dataset=dataset,
            evaluator=evaluator,
            config=config,
            run_directory=artifact_root / str(run["run_id"]),
            nnunet_input=Path(str(artifacts["nnunet_input"])),
        )
        statuses = [
            _run_status(artifact_root / str(candidate["run_id"])) for candidate in runs
        ]
        runtime.update(
            {
                "updated_at_utc": _timestamp(),
                "completed_checkpoint_count": statuses.count("completed"),
                "failed_checkpoint_count": statuses.count("failed"),
                "not_started_or_running_count": sum(
                    status not in {"completed", "failed"} for status in statuses
                ),
            }
        )
        atomic_write_json(Path(str(artifacts["runtime"])), runtime)
    aggregation = _aggregate(
        runs=runs,
        artifact_root=artifact_root,
        checkpoint_output=Path(str(artifacts["checkpoint_patient_metrics"])),
        model_output=Path(str(artifacts["model_patient_metrics"])),
        expected=cast(dict[str, Any], config["expected"]),
    )
    gate_h_pass = (
        int(aggregation["completed_checkpoint_count"]) == len(runs)
        and int(aggregation["failed_checkpoint_count"]) == 0
        and bool(aggregation["all_model_patient_groups_have_25_checkpoints"])
    )
    completion = {
        "schema_version": 1,
        "status": "pass" if gate_h_pass else "completed_with_failures",
        "gate": "H",
        "completed_at_utc": _timestamp(),
        "session_id": runtime["session_id"],
        "gate_h_pass": gate_h_pass,
        "external_retuning_performed": False,
        "external_threshold_selection_performed": False,
        "external_postprocessing_selection_performed": False,
        "source_verification": verification,
        "cache": cache,
        "nnunet_input_preparation": nnunet_preparation,
        **aggregation,
    }
    completion_path = Path(str(artifacts["completion"]))
    atomic_write_json(completion_path, completion)
    runtime.update(
        {
            "status": "completed" if gate_h_pass else "completed_with_failures",
            "completed_at_utc": _timestamp(),
            "completion": completion_path.as_posix(),
            "completion_sha256": file_digest(completion_path),
            **aggregation,
        }
    )
    atomic_write_json(Path(str(artifacts["runtime"])), runtime)
    append_jsonl(
        Path(str(artifacts["access_log"])),
        {
            "event": "gate_h_frozen_external_session_completed",
            "session_id": runtime["session_id"],
            "cohort": "BraTS-Africa-TCIA-v1",
            "model_inference": True,
            "prediction_metrics_accessed": True,
            "retuning_performed": False,
            "gate_h_pass": gate_h_pass,
            "completion_sha256": runtime["completion_sha256"],
        },
    )
    return completion


def run_gate_h_external_queue(
    *,
    external_root: Path,
    allow_frozen_external_inference: bool,
    config_path: Path = Path("configs/q1q2_v2/gate_h_external.yaml"),
) -> dict[str, Any]:
    """Run or resume the only frozen external inference session."""
    with _exclusive_gate_h_lock():
        return _run_gate_h_external_queue_locked(
            external_root=external_root,
            allow_frozen_external_inference=allow_frozen_external_inference,
            config_path=config_path,
        )


__all__ = ["run_gate_h_external_queue"]
