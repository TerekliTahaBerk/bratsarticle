"""Audited checkpoint and analysis freeze for Q1/Q2 Gate G."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml

from bratsarticle.experiments.q1q2_native_runner import (
    NativeRunSpec,
    loss_interaction_specs,
    main_compute_matched_specs,
    main_convergence_specs,
)
from bratsarticle.experiments.q1q2_nnunet_queue import load_nnunet_jobs
from bratsarticle.experiments.q1q2_swin_runner import (
    SwinRunSpec,
    swin_convergence_specs,
)
from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.serialization import atomic_write_json

_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class FrozenRun:
    """One verified run entry ready for the immutable checkpoint manifest."""

    adapter: str
    stage: str
    run_id: str
    model_id: str
    fold: int
    seed: int
    loss: str
    git_commit: str
    stop_reason: str
    best_checkpoint_path: str
    best_checkpoint_sha256: str
    terminal_checkpoint_path: str
    terminal_checkpoint_sha256: str
    patient_metrics_path: str
    patient_metrics_sha256: str
    metric_summary_path: str
    metric_summary_sha256: str
    resource_profile_path: str
    resource_profile_sha256: str
    patient_count: int
    completed_optimizer_steps: int | None
    completed_epochs: int | None
    accelerator_hours: float


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], loaded)


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return cast(dict[str, Any], loaded)


def _validation_subjects(fold: int) -> tuple[str, ...]:
    frame = pd.read_csv(Path(f"splits/q1q2_v2/cv_fold_{fold}.csv"))
    patients = tuple(
        sorted(
            str(value)
            for value in frame.loc[
                frame["role"].eq("validation"),
                "subject_id",
            ]
        )
    )
    if len(patients) not in {73, 74} or len(set(patients)) != len(patients):
        raise ValueError(f"Fold {fold} has an invalid validation cohort")
    return patients


def _verify_file(
    path: Path,
    *,
    problems: list[str],
    label: str,
    expected_sha256: str | None = None,
) -> str:
    if not path.is_file():
        problems.append(f"{label}: missing file {path}")
        return ""
    observed = file_digest(path)
    if expected_sha256 is not None and observed != expected_sha256:
        problems.append(f"{label}: SHA-256 mismatch for {path}")
    return observed


def _verify_patient_metrics(
    *,
    path: Path,
    expected_sha256: str | None,
    model_id: str,
    fold: int,
    seed: int,
    checkpoint_sha256: str,
    problems: list[str],
    label: str,
) -> int:
    observed_hash = _verify_file(
        path,
        problems=problems,
        label=label,
        expected_sha256=expected_sha256,
    )
    if not observed_hash:
        return 0
    frame = pd.read_csv(path)
    required = {
        "patient_id",
        "evaluation_stage",
        "model_id",
        "fold",
        "seed",
        "checkpoint_sha256",
        "mean_regional_dice",
        "wt_dice",
        "tc_dice",
        "et_dice",
        "wt_hd95_mm",
        "tc_hd95_mm",
        "et_hd95_mm",
        "wt_surface_dice",
        "tc_surface_dice",
        "et_surface_dice",
        "wt_lesion_recall",
        "tc_lesion_recall",
        "et_lesion_recall",
    }
    missing = required.difference(frame.columns)
    if missing:
        problems.append(f"{label}: missing metric columns {sorted(missing)}")
        return 0
    expected_patients = set(_validation_subjects(fold))
    raw = frame.loc[frame["evaluation_stage"].eq("raw")]
    observed_patients = set(str(value) for value in raw["patient_id"])
    if observed_patients != expected_patients or len(raw) != len(expected_patients):
        problems.append(f"{label}: patient identities differ from fold {fold}")
    identity_checks = {
        "model_id": {model_id},
        "fold": {fold},
        "seed": {seed},
        "checkpoint_sha256": {checkpoint_sha256},
    }
    for column, expected in identity_checks.items():
        if set(frame[column]) != expected:
            problems.append(f"{label}: {column} identity mismatch")
    return len(raw)


def _native_hash_inputs(
    metadata: dict[str, Any],
    *,
    spec: NativeRunSpec,
    runner_config_path: Path,
    model_config_path: Path,
    repeat_tolerance_path: Path | None,
    problems: list[str],
) -> None:
    hashes = cast(dict[str, Any], metadata.get("hashes", {}))
    runner = _load_yaml(runner_config_path)
    resource_profile_path = Path(
        str(cast(dict[str, Any], runner["resource_profiling"])["protocol"])
    )
    current = {
        "runner_config": runner_config_path,
        "model_matrix": Path("configs/q1q2_v2/model_matrix.yaml"),
        "model_config": model_config_path,
        "fold_manifest": Path(f"splits/q1q2_v2/cv_fold_{spec.fold}.csv"),
        "canonical_manifest": Path(
            "manifests/canonical/brats2020_canonical_manifest.csv"
        ),
        "preprocessing": Path("configs/data/preprocessing_pilot_cached.yaml"),
        "evaluation": Path("configs/q1q2_v2/evaluation.yaml"),
        "loss_catalog": Path("configs/losses/catalog.yaml"),
        "resource_profile_protocol": resource_profile_path,
        "environment_lock": Path("environment/q1q2_v2-environment.json"),
        "requirements_lock": Path(
            "environment/q1q2_v2-requirements-lock.txt"
        ),
        "hardware_preflight": Path(
            "reports/q1q2_v2/hardware_preflight.json"
        ),
    }
    if spec.stage != "loss_screen":
        current["selected_loss_config"] = Path("configs/q1q2_v2/selected_loss.yaml")
        selected = _load_yaml(current["selected_loss_config"])
        current["loss_selection_artifact"] = Path(str(selected["selection_artifact"]))
    if repeat_tolerance_path is not None:
        current["repeat_tolerance_audit"] = repeat_tolerance_path
    for key, path in current.items():
        if not path.is_file() or hashes.get(key) != file_digest(path):
            problems.append(f"{spec.run_id}: scientific input hash differs: {key}")


def _audit_native_run(
    spec: NativeRunSpec,
    *,
    artifact_root: Path,
    minimum_compute_hours: float,
    adapter: str = "native_configurable_unet",
    runner_config_path: Path = Path("configs/q1q2_v2/m1_native_runner.yaml"),
    model_config_path: Path | None = None,
    repeat_tolerance_path: Path | None = None,
    expected_spec_sha256: str | None = None,
) -> tuple[FrozenRun | None, list[str]]:
    problems: list[str] = []
    run_dir = artifact_root / spec.run_id
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    resource_path = run_dir / "resource_profile.json"
    required_json = (metadata_path, progress_path, resource_path)
    if not all(path.is_file() for path in required_json):
        return None, [f"{spec.run_id}: run artifacts are incomplete or absent"]
    metadata = _load_json(metadata_path)
    progress = _load_json(progress_path)
    resource = _load_json(resource_path)
    if metadata.get("status") != "completed" or progress.get("status") != "completed":
        problems.append(f"{spec.run_id}: run status is not completed")
    if (
        metadata.get("repository_dirty_at_start") is not False
        or metadata.get("external_data_accessed") is not False
        or metadata.get("legacy_internal_test_accessed") is not False
    ):
        problems.append(f"{spec.run_id}: provenance/data-access guard failed")
    expected_identity = {
        "run_id": spec.run_id,
        "model_id": spec.model_id,
        "fold": spec.fold,
        "seed": spec.seed,
        "loss": spec.loss_name,
        "stage": spec.stage,
    }
    for key, expected in expected_identity.items():
        if metadata.get(key) != expected:
            problems.append(f"{spec.run_id}: metadata identity mismatch: {key}")
    if metadata.get("run_spec_sha256") != (expected_spec_sha256 or spec.sha256):
        problems.append(f"{spec.run_id}: run-spec hash differs")
    commit = str(metadata.get("git_commit", ""))
    if not _COMMIT_PATTERN.fullmatch(commit):
        problems.append(f"{spec.run_id}: invalid git commit")
    if model_config_path is None:
        matrix = _load_yaml(Path("configs/q1q2_v2/model_matrix.yaml"))
        matches = [
            cast(dict[str, Any], entry)
            for entry in cast(list[Any], matrix["main_models"])
            if cast(dict[str, Any], entry).get("id") == spec.model_id
        ]
        if len(matches) != 1:
            problems.append(f"{spec.run_id}: model matrix entry is ambiguous")
            model_config_path = Path("missing")
        else:
            model_config_path = Path(str(matches[0]["config"]))
    _native_hash_inputs(
        metadata,
        spec=spec,
        runner_config_path=runner_config_path,
        model_config_path=model_config_path,
        repeat_tolerance_path=repeat_tolerance_path,
        problems=problems,
    )
    stop_reason = str(progress.get("stop_reason", ""))
    if spec.stage == "main_compute_matched":
        if stop_reason != "compute_budget_accelerator_hours":
            problems.append(f"{spec.run_id}: compute budget was not the stop reason")
        elapsed = float(progress.get("cumulative_elapsed_seconds", 0.0))
        if elapsed + 1.0 < minimum_compute_hours * 3600.0:
            problems.append(f"{spec.run_id}: four-hour compute budget not reached")
    elif stop_reason != "early_stopping_patience":
        problems.append(f"{spec.run_id}: convergence patience did not stop the run")
    best_path = run_dir / "checkpoints/best.pt"
    terminal_path = run_dir / "checkpoints/terminal.pt"
    best_hash = _verify_file(
        best_path,
        problems=problems,
        label=f"{spec.run_id}: best checkpoint",
    )
    terminal_hash = _verify_file(
        terminal_path,
        problems=problems,
        label=f"{spec.run_id}: terminal checkpoint",
    )
    full = cast(dict[str, Any], progress.get("full_metric_evaluation", {}))
    metric_path = Path(str(full.get("patient_metrics", "")))
    summary_path = Path(str(full.get("metric_summary", "")))
    patient_count = _verify_patient_metrics(
        path=metric_path,
        expected_sha256=str(full.get("patient_metrics_sha256", "")),
        model_id=spec.model_id,
        fold=spec.fold,
        seed=spec.seed,
        checkpoint_sha256=best_hash,
        problems=problems,
        label=f"{spec.run_id}: common metrics",
    )
    summary_hash = _verify_file(
        summary_path,
        problems=problems,
        label=f"{spec.run_id}: metric summary",
        expected_sha256=str(full.get("metric_summary_sha256", "")),
    )
    if spec.stage == "main_convergence":
        milestones = cast(
            dict[str, dict[str, Any]],
            progress.get("budget_sensitivity_checkpoints", {}),
        )
        for step in ("2000", "10000"):
            entry = milestones.get(step, {})
            _verify_file(
                Path(str(entry.get("checkpoint", ""))),
                problems=problems,
                label=f"{spec.run_id}: milestone {step}",
                expected_sha256=str(entry.get("checkpoint_sha256", "")),
            )
            milestone_metrics = Path(str(entry.get("patient_metrics", "")))
            _verify_file(
                milestone_metrics,
                problems=problems,
                label=f"{spec.run_id}: milestone metrics {step}",
                expected_sha256=str(entry.get("patient_metrics_sha256", "")),
            )
            if milestone_metrics.is_file():
                milestone_frame = pd.read_csv(milestone_metrics)
                if set(str(value) for value in milestone_frame["patient_id"]) != set(
                    _validation_subjects(spec.fold)
                ):
                    problems.append(
                        f"{spec.run_id}: milestone {step} patient identities differ"
                    )
    resource_hash = _verify_file(
        resource_path,
        problems=problems,
        label=f"{spec.run_id}: resource profile",
    )
    elapsed_hours = (
        float(resource.get("cumulative_elapsed_seconds_including_session", 0.0))
        / 3600.0
    )
    completed_steps = int(resource.get("completed_optimizer_steps", -1))
    resource_protocol_path = Path(
        str(resource.get("resource_profile_protocol", "missing"))
    )
    if resource_protocol_path.is_file():
        resource_protocol = _load_yaml(resource_protocol_path)
        required_timing_count = int(
            cast(dict[str, Any], resource_protocol["timing"])["measured_iterations"]
        )
        resource_protocol_hash_valid = resource.get(
            "resource_profile_protocol_sha256"
        ) == file_digest(resource_protocol_path)
    else:
        required_timing_count = -1
        resource_protocol_hash_valid = False
    if elapsed_hours <= 0.0 or completed_steps <= 0:
        problems.append(f"{spec.run_id}: resource profile is incomplete")
    if (
        not resource_protocol_hash_valid
        or int(resource.get("synchronized_training_step_measurement_count", -1))
        != required_timing_count
        or len(
            cast(
                list[Any],
                resource.get("synchronized_training_step_seconds", []),
            )
        )
        != required_timing_count
    ):
        problems.append(f"{spec.run_id}: synchronized step timing is incomplete")
    static_profile = cast(dict[str, Any], metadata.get("static_profile", {}))
    if adapter == "native_configurable_unet" and (
        int(static_profile.get("parameter_count", 0)) <= 0
        or int(static_profile.get("macs_per_slice", 0)) <= 0
        or int(static_profile.get("flops_per_slice", 0)) <= 0
    ):
        problems.append(f"{spec.run_id}: native static profile is incomplete")
    if adapter == "monai_swinunetr" and (
        int(static_profile.get("parameter_count", 0)) <= 0
        or int(static_profile.get("flops_per_input", 0)) <= 0
        or int(static_profile.get("mac_equivalents_per_input", 0)) <= 0
        or len(
            cast(
                list[Any],
                static_profile.get("receptive_field_proxy_voxels_per_axis", []),
            )
        )
        != 3
        or int(metadata.get("parameter_count", 0))
        != int(static_profile.get("parameter_count", -1))
    ):
        problems.append(f"{spec.run_id}: Swin static profile is incomplete")
    if problems:
        return None, problems
    return (
        FrozenRun(
            adapter=adapter,
            stage=spec.stage,
            run_id=spec.run_id,
            model_id=spec.model_id,
            fold=spec.fold,
            seed=spec.seed,
            loss=spec.loss_name,
            git_commit=commit,
            stop_reason=stop_reason,
            best_checkpoint_path=best_path.as_posix(),
            best_checkpoint_sha256=best_hash,
            terminal_checkpoint_path=terminal_path.as_posix(),
            terminal_checkpoint_sha256=terminal_hash,
            patient_metrics_path=metric_path.as_posix(),
            patient_metrics_sha256=str(full["patient_metrics_sha256"]),
            metric_summary_path=summary_path.as_posix(),
            metric_summary_sha256=summary_hash,
            resource_profile_path=resource_path.as_posix(),
            resource_profile_sha256=resource_hash,
            patient_count=patient_count,
            completed_optimizer_steps=completed_steps,
            completed_epochs=None,
            accelerator_hours=elapsed_hours,
        ),
        [],
    )


def _audit_swin_run(
    spec: SwinRunSpec,
    *,
    artifact_root: Path,
) -> tuple[FrozenRun | None, list[str]]:
    native_spec = NativeRunSpec(
        stage="main_convergence",
        model_id=spec.model_id,
        fold=spec.fold,
        seed=spec.seed,
        loss_name=spec.loss_name,
        maximum_optimizer_steps=spec.maximum_optimizer_steps,
        warmup_optimizer_steps=spec.warmup_optimizer_steps,
        full_metric_evaluation=True,
    )
    frozen, problems = _audit_native_run(
        native_spec,
        artifact_root=artifact_root,
        minimum_compute_hours=4.0,
        adapter="monai_swinunetr",
        runner_config_path=Path("configs/q1q2_v2/swin_m1_runner.yaml"),
        model_config_path=Path("configs/q1q2_v2/models/swin_unetr_monai.yaml"),
        repeat_tolerance_path=Path("reports/q1q2_v2/swin_mps_repeat_tolerance.json"),
        expected_spec_sha256=spec.sha256,
    )
    if frozen is None:
        return None, problems
    return frozen, []


def _audit_nnunet_run(
    job: dict[str, Any],
    *,
    artifact_root: Path,
    required_epochs: int,
    final_lr_fraction: float,
) -> tuple[FrozenRun | None, list[str]]:
    run_id = str(job["run_id"])
    problems: list[str] = []
    runtime_path = artifact_root / run_id / "runtime.json"
    if not runtime_path.is_file():
        return None, [f"{run_id}: run artifacts are incomplete or absent"]
    runtime = _load_json(runtime_path)
    if runtime.get("status") != "completed":
        problems.append(f"{run_id}: runtime status is not completed")
    if (
        runtime.get("external_data_accessed") is not False
        or runtime.get("legacy_internal_test_accessed") is not False
    ):
        problems.append(f"{run_id}: data-access guard failed")
    stored_job = cast(dict[str, Any], runtime.get("job", {}))
    if stored_job != job:
        problems.append(f"{run_id}: frozen job identity differs")
    commit = str(runtime.get("repository_commit_at_queue_start", ""))
    if not _COMMIT_PATTERN.fullmatch(commit):
        problems.append(f"{run_id}: invalid git commit")
    metadata_path = Path(str(runtime.get("official_metadata_path", "")))
    metadata_hash = _verify_file(
        metadata_path,
        problems=problems,
        label=f"{run_id}: official metadata",
        expected_sha256=str(runtime.get("official_metadata_sha256", "")),
    )
    metadata = _load_json(metadata_path) if metadata_hash else {}
    nnunet_runner = _load_yaml(Path("configs/q1q2_v2/nnunet_m1_runner.yaml"))
    resource_profile_path = Path(str(nnunet_runner["resource_profile_protocol"]))
    resource_profile = _load_yaml(resource_profile_path)
    resource_timing = cast(dict[str, Any], resource_profile["timing"])
    required_timing_count = int(resource_timing["measured_iterations"])
    if (
        metadata.get("repository_dirty_at_start") is not False
        or metadata.get("git_commit") != commit
        or metadata.get("status") != "completed"
        or metadata.get("seed") != int(job["seed"])
        or metadata.get("fold_zero_indexed") != int(job["fold_nnunet_zero_indexed"])
        or metadata.get("configuration") != str(job["configuration"])
        or int(metadata.get("completed_epochs", -1)) != required_epochs
    ):
        problems.append(f"{run_id}: official completion/conduct check failed")
    timing_samples = cast(
        list[Any],
        metadata.get("synchronized_training_step_seconds", []),
    )
    if (
        metadata.get("resource_profile_protocol") != resource_profile_path.as_posix()
        or metadata.get("resource_profile_protocol_sha256")
        != file_digest(resource_profile_path)
        or metadata.get("environment_lock_sha256")
        != file_digest(Path("environment/q1q2_v2-environment.json"))
        or metadata.get("requirements_lock_sha256")
        != file_digest(Path("environment/q1q2_v2-requirements-lock.txt"))
        or metadata.get("hardware_preflight_sha256")
        != file_digest(Path("reports/q1q2_v2/hardware_preflight.json"))
        or int(metadata.get("training_step_timing_target_count", -1))
        != required_timing_count
        or int(metadata.get("synchronized_training_step_measurement_count", -1))
        != required_timing_count
        or len(timing_samples) != required_timing_count
    ):
        problems.append(f"{run_id}: official synchronized step timing is incomplete")
    static_profile = cast(dict[str, Any], metadata.get("static_profile", {}))
    patch_dimensions = 2 if str(job["configuration"]) == "2d" else 3
    if (
        int(static_profile.get("parameter_count", 0)) <= 0
        or int(static_profile.get("flops_per_input", 0)) <= 0
        or int(static_profile.get("mac_equivalents_per_input", 0)) <= 0
        or len(
            cast(
                list[Any],
                static_profile.get("receptive_field_proxy_voxels_per_axis", []),
            )
        )
        != patch_dimensions
        or int(metadata.get("parameter_count", 0))
        != int(static_profile.get("parameter_count", -1))
    ):
        problems.append(f"{run_id}: official nnU-Net static profile is incomplete")
    defaults = cast(dict[str, Any], metadata.get("official_defaults", {}))
    initial_lr = float(defaults.get("initial_lr", 0.0))
    final_lr = float(metadata.get("final_learning_rate", float("inf")))
    if initial_lr <= 0.0 or final_lr < 0.0 or final_lr > initial_lr * final_lr_fraction:
        problems.append(f"{run_id}: official learning-rate schedule is incomplete")
    best_path = Path(str(runtime.get("best_checkpoint_path", "")))
    final_path = Path(str(runtime.get("final_checkpoint_path", "")))
    best_hash = _verify_file(
        best_path,
        problems=problems,
        label=f"{run_id}: best checkpoint",
        expected_sha256=str(runtime.get("best_checkpoint_sha256", "")),
    )
    final_hash = _verify_file(
        final_path,
        problems=problems,
        label=f"{run_id}: final checkpoint",
        expected_sha256=str(runtime.get("final_checkpoint_sha256", "")),
    )
    milestones = cast(
        dict[str, str],
        runtime.get("budget_sensitivity_checkpoint_sha256", {}),
    )
    for step in ("2000", "10000"):
        milestone_path = best_path.parent / f"checkpoint_q1q2_step_{step}.pth"
        _verify_file(
            milestone_path,
            problems=problems,
            label=f"{run_id}: milestone checkpoint {step}",
            expected_sha256=str(milestones.get(step, "")),
        )
    central = cast(dict[str, Any], runtime.get("central_evaluation", {}))
    if (
        central.get("external_data_accessed") is not False
        or central.get("legacy_internal_test_accessed") is not False
        or central.get("checkpoint_sha256") != best_hash
    ):
        problems.append(f"{run_id}: central-evaluation provenance differs")
    metric_path = Path(str(central.get("patient_metrics", "")))
    summary_path = Path(str(central.get("metric_summary", "")))
    patient_count = _verify_patient_metrics(
        path=metric_path,
        expected_sha256=str(central.get("patient_metrics_sha256", "")),
        model_id=str(job["model_id"]),
        fold=int(job["fold_one_indexed"]),
        seed=int(job["seed"]),
        checkpoint_sha256=best_hash,
        problems=problems,
        label=f"{run_id}: common metrics",
    )
    summary_hash = _verify_file(
        summary_path,
        problems=problems,
        label=f"{run_id}: metric summary",
        expected_sha256=str(central.get("metric_summary_sha256", "")),
    )
    if problems:
        return None, problems
    return (
        FrozenRun(
            adapter="official_nnunetv2",
            stage="main_convergence",
            run_id=run_id,
            model_id=str(job["model_id"]),
            fold=int(job["fold_one_indexed"]),
            seed=int(job["seed"]),
            loss="official_nnunet_compound_loss",
            git_commit=commit,
            stop_reason="official_1000_epoch_schedule_complete",
            best_checkpoint_path=best_path.as_posix(),
            best_checkpoint_sha256=best_hash,
            terminal_checkpoint_path=final_path.as_posix(),
            terminal_checkpoint_sha256=final_hash,
            patient_metrics_path=metric_path.as_posix(),
            patient_metrics_sha256=str(central["patient_metrics_sha256"]),
            metric_summary_path=summary_path.as_posix(),
            metric_summary_sha256=summary_hash,
            resource_profile_path=metadata_path.as_posix(),
            resource_profile_sha256=metadata_hash,
            patient_count=patient_count,
            completed_optimizer_steps=int(
                metadata.get("completed_optimizer_steps", -1)
            ),
            completed_epochs=int(metadata["completed_epochs"]),
            accelerator_hours=float(metadata["accelerator_hours"]),
        ),
        [],
    )


def _external_access_problems(path: Path) -> list[str]:
    if not path.is_file():
        return [f"External access log is missing: {path}"]
    problems: list[str] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        event = json.loads(line)
        if (
            event.get("model_inference") is not False
            or event.get("prediction_metrics_accessed") is not False
        ):
            problems.append(
                f"External result access occurred before Gate G at line {line_number}"
            )
    return problems


def _expected_counts(protocol: dict[str, Any]) -> dict[str, int]:
    counts = {
        str(key): int(value)
        for key, value in cast(
            dict[str, Any],
            protocol["expected_runs"],
        ).items()
    }
    if counts != {
        "native_main_convergence": 225,
        "swin_main_convergence": 25,
        "official_nnunet_main_convergence": 50,
        "native_compute_matched": 200,
        "native_loss_interaction": 100,
        "total": 600,
    }:
        raise ValueError("Gate G expected-run counts differ from the frozen design")
    return counts


def audit_gate_g(
    *,
    protocol_path: Path = Path("configs/q1q2_v2/gate_g_freeze.yaml"),
    write_report: bool = True,
) -> dict[str, Any]:
    """Audit every Gate G prerequisite without authorizing external inference."""
    protocol = _load_yaml(protocol_path)
    if (
        protocol.get("status") != "frozen_before_main_results"
        or protocol.get("external_inference_permitted") is not False
    ):
        raise PermissionError("Gate G protocol is not frozen before results")
    counts = _expected_counts(protocol)
    problems = _external_access_problems(
        Path("artifacts/q1q2_v2/external_access_log.jsonl")
    )
    required_inputs = [
        Path(str(value))
        for value in cast(list[Any], protocol["required_analysis_inputs"])
    ]
    for path in required_inputs:
        if not path.is_file():
            problems.append(f"Required analysis input is missing: {path}")
    frozen_runs: list[FrozenRun] = []
    selected_loss = Path("configs/q1q2_v2/selected_loss.yaml")
    selected_plan = Path("configs/q1q2_v2/selected_nnunet_3d_plan.yaml")
    native_specs: tuple[NativeRunSpec, ...] = ()
    swin_specs: tuple[SwinRunSpec, ...] = ()
    nnunet_jobs: tuple[dict[str, Any], ...] = ()
    if selected_loss.is_file():
        try:
            native_specs = (
                *main_convergence_specs(
                    Path("configs/q1q2_v2/m1_native_runner.yaml"),
                    selected_loss,
                ),
                *main_compute_matched_specs(
                    Path("configs/q1q2_v2/m1_native_runner.yaml"),
                    selected_loss,
                ),
                *loss_interaction_specs(
                    Path("configs/q1q2_v2/m1_native_runner.yaml"),
                    selected_loss,
                ),
            )
            swin_specs = swin_convergence_specs(
                Path("configs/q1q2_v2/swin_m1_runner.yaml"),
                selected_loss,
            )
        except Exception as error:
            problems.append(f"Frozen native/Swin matrix is invalid: {error}")
    else:
        problems.append("Loss selection is not complete and frozen")
    if selected_plan.is_file():
        try:
            nnunet_jobs = load_nnunet_jobs(
                Path("configs/q1q2_v2/nnunet_m1_runner.yaml")
            )
        except Exception as error:
            problems.append(f"Frozen nnU-Net matrix is invalid: {error}")
    else:
        problems.append("Outcome-blind nnU-Net 3D plan is not frozen")
    convergence = cast(dict[str, Any], protocol["convergence"])
    for native_spec in native_specs:
        frozen, run_problems = _audit_native_run(
            native_spec,
            artifact_root=Path("artifacts/q1q2_v2/native_runs"),
            minimum_compute_hours=float(
                convergence["native_compute_matched_minimum_accelerator_hours"]
            ),
        )
        problems.extend(run_problems)
        if frozen is not None:
            frozen_runs.append(frozen)
    for swin_spec in swin_specs:
        frozen, run_problems = _audit_swin_run(
            swin_spec,
            artifact_root=Path("artifacts/q1q2_v2/swin_runs"),
        )
        problems.extend(run_problems)
        if frozen is not None:
            frozen_runs.append(frozen)
    for job in nnunet_jobs:
        frozen, run_problems = _audit_nnunet_run(
            job,
            artifact_root=Path("artifacts/q1q2_v2/nnunet_runs"),
            required_epochs=int(convergence["nnunet_required_epochs"]),
            final_lr_fraction=float(
                convergence["nnunet_final_lr_maximum_fraction_of_initial"]
            ),
        )
        problems.extend(run_problems)
        if frozen is not None:
            frozen_runs.append(frozen)
    expected_identity_count = len(native_specs) + len(swin_specs) + len(nnunet_jobs)
    if expected_identity_count not in {0, counts["total"]}:
        problems.append(
            "Expanded run matrix has "
            f"{expected_identity_count} identities instead of {counts['total']}"
        )
    if (
        expected_identity_count == counts["total"]
        and len(frozen_runs) != counts["total"]
    ):
        problems.append(
            f"Only {len(frozen_runs)} of {counts['total']} runs passed audit"
        )
    identities = {
        (run.stage, run.model_id, run.fold, run.seed, run.loss) for run in frozen_runs
    }
    if len(identities) != len(frozen_runs):
        problems.append("Verified run manifest contains duplicate identities")
    analysis_hashes = {
        path.as_posix(): file_digest(path) for path in required_inputs if path.is_file()
    }
    report = {
        "schema_version": 1,
        "status": "pass" if not problems else "blocked",
        "audited_at_utc": _timestamp(),
        "external_inference_permitted": False,
        "expected_run_counts": counts,
        "expanded_expected_identity_count": expected_identity_count,
        "verified_run_count": len(frozen_runs),
        "problem_count": len(problems),
        "problems": problems,
        "analysis_input_sha256": analysis_hashes,
        "runs": [asdict(run) for run in frozen_runs],
    }
    if write_report:
        output = Path(str(cast(dict[str, Any], protocol["outputs"])["audit"]))
        atomic_write_json(output, report)
    return report


def freeze_gate_g(
    *,
    allow_analysis_freeze: bool,
    protocol_path: Path = Path("configs/q1q2_v2/gate_g_freeze.yaml"),
) -> dict[str, Any]:
    """Write immutable checkpoint/statistical manifests only after a passing audit."""
    if not allow_analysis_freeze:
        raise PermissionError("Gate G freeze requires --allow-analysis-freeze")
    protocol = _load_yaml(protocol_path)
    report = audit_gate_g(protocol_path=protocol_path, write_report=True)
    if report["status"] != "pass":
        raise RuntimeError(
            f"Gate G remains blocked by {report['problem_count']} audit findings"
        )
    outputs = cast(dict[str, Any], protocol["outputs"])
    checkpoint_path = Path(str(outputs["checkpoint_manifest"]))
    freeze_path = Path(str(outputs["analysis_freeze"]))
    manifest_core = {
        "schema_version": 1,
        "status": "frozen",
        "run_count": int(report["verified_run_count"]),
        "runs": report["runs"],
    }
    manifest_digest = text_digest(
        json.dumps(manifest_core, sort_keys=True, separators=(",", ":"))
    )
    manifest = {
        **manifest_core,
        "frozen_at_utc": _timestamp(),
        "content_sha256": manifest_digest,
    }
    if checkpoint_path.exists():
        existing = _load_json(checkpoint_path)
        if existing.get("content_sha256") != manifest_digest:
            raise FileExistsError("A different Gate G checkpoint freeze already exists")
    else:
        atomic_write_json(checkpoint_path, manifest)
    if freeze_path.exists():
        existing_freeze = _load_json(freeze_path)
        if (
            existing_freeze.get("status") != "frozen_external_inference_permitted"
            or existing_freeze.get("external_inference_permitted") is not True
            or existing_freeze.get("external_retuning_permitted") is not False
            or existing_freeze.get("gate_g_protocol_sha256")
            != file_digest(protocol_path)
            or existing_freeze.get("checkpoint_manifest_sha256")
            != file_digest(checkpoint_path)
            or existing_freeze.get("statistical_analysis_plan_sha256")
            != file_digest(Path("configs/q1q2_v2/statistical_analysis_plan.yaml"))
        ):
            raise FileExistsError("The existing Gate G analysis freeze differs")
        return existing_freeze
    freeze = {
        "schema_version": 1,
        "status": "frozen_external_inference_permitted",
        "frozen_at_utc": _timestamp(),
        "gate_g_protocol": protocol_path.as_posix(),
        "gate_g_protocol_sha256": file_digest(protocol_path),
        "gate_g_audit": str(outputs["audit"]),
        "gate_g_audit_sha256": file_digest(Path(str(outputs["audit"]))),
        "checkpoint_manifest": checkpoint_path.as_posix(),
        "checkpoint_manifest_sha256": file_digest(checkpoint_path),
        "statistical_analysis_plan": ("configs/q1q2_v2/statistical_analysis_plan.yaml"),
        "statistical_analysis_plan_sha256": file_digest(
            Path("configs/q1q2_v2/statistical_analysis_plan.yaml")
        ),
        "analysis_input_sha256": report["analysis_input_sha256"],
        "external_inference_permitted": True,
        "external_retuning_permitted": False,
    }
    atomic_write_json(freeze_path, freeze)
    return freeze


__all__ = ["FrozenRun", "audit_gate_g", "freeze_gate_g"]
