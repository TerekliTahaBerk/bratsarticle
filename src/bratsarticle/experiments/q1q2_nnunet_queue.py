"""Guarded sequential runner for the official nnU-Net fold-seed matrix."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml

from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import atomic_write_json


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    return commit, bool(status.strip())


def load_nnunet_runner_config(path: Path) -> dict[str, Any]:
    """Load and validate the official nnU-Net runner contract."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("nnU-Net runner configuration must be a mapping")
    config = cast(dict[str, Any], loaded)
    if config.get("status") not in {
        "blocked_until_hardware_plan_freeze",
        "frozen_before_first_reportable_development_run",
    }:
        raise PermissionError("nnU-Net runner status is invalid")
    hardware = cast(dict[str, Any], config["hardware"])
    if hardware.get("backend") != "mps":
        raise ValueError("The M1 nnU-Net runner requires MPS")
    guards = cast(dict[str, Any], config["guards"])
    if (
        bool(guards["allow_external_data"])
        or bool(guards["allow_legacy_internal_test"])
        or bool(guards["allow_silent_seed_replacement"])
    ):
        raise PermissionError("nnU-Net runner enables prohibited conduct")
    return config


def load_nnunet_jobs(
    runner_config_path: Path,
) -> tuple[dict[str, Any], ...]:
    """Validate the selected-plan-bound 50-job official matrix."""
    config = load_nnunet_runner_config(runner_config_path)
    matrix = cast(dict[str, Any], config["matrix"])
    selected_path = Path(str(matrix["selected_3d_plan"]))
    if not selected_path.is_file():
        raise PermissionError("nnU-Net 3D hardware plan is not frozen")
    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    if not isinstance(selected, dict):
        raise ValueError("Selected nnU-Net plan must be a mapping")
    if (
        selected.get("status")
        != "frozen_from_outcome_blind_hardware_feasibility"
        or selected.get("performance_outcomes_used") is not False
        or selected.get("external_data_accessed") is not False
    ):
        raise PermissionError("Selected nnU-Net 3D plan is not eligible")
    expected_selected_hash = matrix.get("selected_3d_plan_sha256")
    if (
        expected_selected_hash is not None
        and expected_selected_hash != file_digest(selected_path)
    ):
        raise PermissionError("nnU-Net runner selected-plan hash changed")
    queue_path = Path(str(matrix["queue"]))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    if queue.get("status") != "frozen_not_started":
        raise PermissionError("nnU-Net queue is not frozen")
    if (
        queue.get("selected_3d_plan_config_sha256")
        != file_digest(selected_path)
    ):
        raise PermissionError("nnU-Net selected-plan hash changed")
    expected_queue_hash = matrix.get("queue_sha256")
    if expected_queue_hash is not None and expected_queue_hash != file_digest(
        queue_path
    ):
        raise PermissionError("nnU-Net runner queue hash changed")
    jobs = cast(list[dict[str, Any]], queue["jobs"])
    if len(jobs) != int(matrix["expected_jobs"]):
        raise ValueError("nnU-Net queue does not contain exactly 50 jobs")
    identities = {
        (str(job["model_id"]), int(job["fold_one_indexed"]), int(job["seed"]))
        for job in jobs
    }
    if len(identities) != len(jobs):
        raise ValueError("nnU-Net queue contains duplicate job identities")
    expected_folds = {int(value) for value in matrix["expected_folds"]}
    expected_seeds = {int(value) for value in matrix["expected_seeds"]}
    if {identity[1] for identity in identities} != expected_folds:
        raise ValueError("nnU-Net queue has an invalid fold set")
    if {identity[2] for identity in identities} != expected_seeds:
        raise ValueError("nnU-Net queue has an invalid seed set")
    selected_3d = str(selected["selected_plans_identifier"])
    for job in jobs:
        if job["configuration"] == "3d_fullres":
            if job["plans_identifier"] != selected_3d:
                raise ValueError("nnU-Net 3D job differs from the selected plan")
        elif job["plans_identifier"] != "nnUNetPlans":
            raise ValueError("nnU-Net 2D jobs must use the standard plans")
        environment = cast(dict[str, str], job["environment"])
        if environment.get("nnUNet_n_proc_DA") != "0":
            raise ValueError("nnU-Net seeded jobs require single-process augmentation")
    return tuple(jobs)


def _required_environment_path(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return Path(raw).expanduser().resolve()


def official_output_directory(
    results_root: Path,
    *,
    dataset_name: str,
    job: Mapping[str, Any],
) -> Path:
    """Resolve the official nnU-Net result folder without searching by glob."""
    trainer = str(job["trainer"])
    plans = str(job["plans_identifier"])
    configuration = str(job["configuration"])
    fold = int(job["fold_nnunet_zero_indexed"])
    return (
        results_root
        / dataset_name
        / f"{trainer}__{plans}__{configuration}"
        / f"fold_{fold}"
    )


def _job_sha256(job: Mapping[str, Any]) -> str:
    return text_digest(
        json.dumps(dict(job), sort_keys=True, separators=(",", ":"))
    )


def _runtime_status(runtime_dir: Path) -> str:
    path = runtime_dir / "runtime.json"
    if not path.is_file():
        return "not_started"
    report = json.loads(path.read_text(encoding="utf-8"))
    return str(report.get("status", "unknown"))


def nnunet_queue_snapshot(
    jobs: tuple[dict[str, Any], ...],
    artifact_root: Path,
) -> dict[str, Any]:
    """Return current status for every immutable official job."""
    entries = [
        {
            "run_id": str(job["run_id"]),
            "job_sha256": _job_sha256(job),
            "status": _runtime_status(artifact_root / str(job["run_id"])),
        }
        for job in jobs
    ]
    return {
        "schema_version": 1,
        "updated_at_utc": _timestamp(),
        "job_count": len(entries),
        "completed_count": sum(row["status"] == "completed" for row in entries),
        "failed_count": sum(row["status"] == "failed" for row in entries),
        "running_or_resumable_count": sum(
            row["status"] in {"running", "unknown"} for row in entries
        ),
        "not_started_count": sum(
            row["status"] == "not_started" for row in entries
        ),
        "jobs": entries,
    }


def _validate_completed_output(
    official_output: Path,
    job: Mapping[str, Any],
    *,
    expected_git_commit: str,
) -> dict[str, Any]:
    metadata_path = official_output / "q1q2_run_metadata.json"
    best = official_output / "checkpoint_best.pth"
    final = official_output / "checkpoint_final.pth"
    milestones = {
        step: official_output / f"checkpoint_q1q2_step_{step}.pth"
        for step in (2_000, 10_000)
    }
    if (
        not metadata_path.is_file()
        or not best.is_file()
        or not final.is_file()
        or not all(path.is_file() for path in milestones.values())
    ):
        raise RuntimeError("Completed nnU-Net run is missing required artifacts")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise RuntimeError("Official nnU-Net metadata is not completed")
    if metadata.get("repository_dirty_at_start") is not False:
        raise RuntimeError("Official nnU-Net run started from a dirty repository")
    if metadata.get("git_commit") != expected_git_commit:
        raise RuntimeError("Official nnU-Net run commit differs from the queue")
    expected = {
        "trainer": str(job["trainer"]),
        "seed": int(job["seed"]),
        "fold_zero_indexed": int(job["fold_nnunet_zero_indexed"]),
        "configuration": str(job["configuration"]),
        "device": "mps",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Official nnU-Net metadata mismatch: {key}")
    official_defaults = cast(dict[str, Any], metadata["official_defaults"])
    if (
        int(metadata.get("completed_epochs", -1)) != 1_000
        or int(official_defaults.get("epochs", -1)) != 1_000
    ):
        raise RuntimeError("Official nnU-Net run did not complete 1,000 epochs")
    best_hash = file_digest(best)
    final_hash = file_digest(final)
    if metadata.get("checkpoint_best_sha256") != best_hash:
        raise RuntimeError("Official nnU-Net best checkpoint hash differs")
    if metadata.get("checkpoint_final_sha256") != final_hash:
        raise RuntimeError("Official nnU-Net final checkpoint hash differs")
    milestone_entries = cast(
        dict[str, dict[str, Any]],
        metadata.get("budget_sensitivity_checkpoints", {}),
    )
    milestone_hashes: dict[str, str] = {}
    for step, path in milestones.items():
        observed = file_digest(path)
        entry = milestone_entries.get(str(step), {})
        if entry.get("sha256") != observed:
            raise RuntimeError(
                f"Official nnU-Net milestone checkpoint hash differs: {step}"
            )
        milestone_hashes[str(step)] = observed
    return {
        "official_metadata_path": metadata_path.as_posix(),
        "official_metadata_sha256": file_digest(metadata_path),
        "best_checkpoint_path": best.as_posix(),
        "best_checkpoint_sha256": best_hash,
        "final_checkpoint_path": final.as_posix(),
        "final_checkpoint_sha256": final_hash,
        "budget_sensitivity_checkpoint_sha256": milestone_hashes,
        "parameter_count": int(metadata["parameter_count"]),
        "completed_epochs": int(metadata["completed_epochs"]),
        "accelerator_hours": float(metadata["accelerator_hours"]),
        "framework_peak_allocated_unified_memory_bytes": int(
            metadata["framework_peak_allocated_unified_memory_bytes"]
        ),
        "driver_peak_allocated_unified_memory_bytes": int(
            metadata["driver_peak_allocated_unified_memory_bytes"]
        ),
    }


def _run_one_job(
    *,
    job: dict[str, Any],
    artifact_root: Path,
    results_root: Path,
    dataset_name: str,
    repository_commit: str,
) -> None:
    runtime_dir = artifact_root / str(job["run_id"])
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / "runtime.json"
    official_output = official_output_directory(
        results_root,
        dataset_name=dataset_name,
        job=job,
    )
    assert_output_paths_safe([runtime_dir, official_output], [])
    status = _runtime_status(runtime_dir)
    if status == "completed":
        _validate_completed_output(
            official_output,
            job,
            expected_git_commit=repository_commit,
        )
        return
    if status == "failed":
        raise RuntimeError(
            f"Failed nnU-Net seed cannot be silently retried: {job['run_id']}"
        )
    latest = official_output / "checkpoint_latest.pth"
    if official_output.exists() and not latest.is_file():
        existing_files = list(official_output.iterdir())
        if existing_files:
            raise RuntimeError(
                "Partial nnU-Net output has no resumable checkpoint and will "
                f"not be overwritten: {official_output}"
            )
    continuation = official_output.exists() and latest.is_file()
    if continuation:
        prior_metadata_path = official_output / "q1q2_run_metadata.json"
        if not prior_metadata_path.is_file():
            raise RuntimeError("Resumable nnU-Net output lacks provenance metadata")
        prior_metadata = json.loads(
            prior_metadata_path.read_text(encoding="utf-8")
        )
        if prior_metadata.get("git_commit") != repository_commit:
            raise RuntimeError("nnU-Net continuation across commits is prohibited")
    command = [str(value) for value in cast(list[Any], job["command"])]
    if continuation:
        command.append("--c")
    environment = os.environ.copy()
    environment.update(
        {
            str(key): str(value)
            for key, value in cast(dict[str, Any], job["environment"]).items()
        }
    )
    environment["Q1Q2_CONTINUATION_REQUESTED"] = "1" if continuation else "0"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_id": str(job["run_id"]),
        "job": job,
        "job_sha256": _job_sha256(job),
        "repository_commit_at_queue_start": repository_commit,
        "continuation_requested": continuation,
        "continuation_bitwise_reproducibility_claimed": False,
        "command": command,
        "official_output_directory": official_output.as_posix(),
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
        "started_at_utc": _timestamp(),
    }
    atomic_write_json(runtime_path, report)
    stdout_path = runtime_dir / "stdout.log"
    stderr_path = runtime_dir / "stderr.log"
    with (
        stdout_path.open("a", encoding="utf-8") as stdout,
        stderr_path.open("a", encoding="utf-8") as stderr,
    ):
        result = subprocess.run(
            command,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    report["return_code"] = int(result.returncode)
    report["finished_at_utc"] = _timestamp()
    if result.returncode != 0:
        report["status"] = "failed"
        atomic_write_json(runtime_path, report)
        raise RuntimeError(
            f"Official nnU-Net job failed without seed replacement: {job['run_id']}"
        )
    report.update(
        _validate_completed_output(
            official_output,
            job,
            expected_git_commit=repository_commit,
        )
    )
    report["status"] = "completed"
    atomic_write_json(runtime_path, report)


def run_nnunet_main_queue(
    *,
    runner_config_path: Path,
    allow_reportable_development_training: bool,
) -> dict[str, Any]:
    """Run or resume all 50 official nnU-Net jobs sequentially."""
    if not allow_reportable_development_training:
        raise PermissionError(
            "nnU-Net main queue requires reportable development authorization"
        )
    config = load_nnunet_runner_config(runner_config_path)
    if config["status"] != "frozen_before_first_reportable_development_run":
        raise PermissionError("nnU-Net runner must be frozen after plan selection")
    commit, dirty = _git_state()
    if bool(config["hardware"]["require_clean_git"]) and dirty:
        raise RuntimeError("Reportable nnU-Net training requires a clean repository")
    data = cast(dict[str, Any], config["data"])
    artifacts = cast(dict[str, Any], config["artifacts"])
    raw_root = _required_environment_path(str(data["nnunet_raw_environment"]))
    preprocessed_root = _required_environment_path(
        str(data["nnunet_preprocessed_environment"])
    )
    results_root = _required_environment_path(
        str(data["nnunet_results_environment"])
    )
    dataset_name = str(data["dataset_name"])
    required_preprocessed = preprocessed_root / dataset_name
    if not (required_preprocessed / "splits_final.json").is_file():
        raise FileNotFoundError("nnU-Net frozen splits are missing")
    if not (raw_root / dataset_name / "derivation_manifest.json").is_file():
        raise FileNotFoundError("nnU-Net audited raw derivation is missing")
    jobs = load_nnunet_jobs(runner_config_path)
    runtime_root = Path(str(artifacts["runtime_root"]))
    runtime_root.mkdir(parents=True, exist_ok=True)
    conflicts = [
        str(name)
        for name in cast(list[Any], config["guards"]["mutually_exclusive_mps_queues"])
        if (runtime_root / str(name)).exists()
    ]
    if conflicts:
        raise RuntimeError(
            "nnU-Net queue conflicts with active MPS work: "
            + ", ".join(conflicts)
        )
    lock_path = runtime_root / "nnunetv2_main.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError("nnU-Net main queue lock already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    artifact_root = Path(str(artifacts["root"])).resolve()
    assert_output_paths_safe(
        [artifact_root, results_root],
        [raw_root, preprocessed_root],
    )
    state_path = runtime_root / "nnunetv2_main_runtime.json"
    try:
        snapshot = nnunet_queue_snapshot(jobs, artifact_root)
        atomic_write_json(state_path, snapshot)
        for job in jobs:
            _run_one_job(
                job=job,
                artifact_root=artifact_root,
                results_root=results_root,
                dataset_name=dataset_name,
                repository_commit=commit,
            )
            snapshot = nnunet_queue_snapshot(jobs, artifact_root)
            atomic_write_json(state_path, snapshot)
        snapshot = nnunet_queue_snapshot(jobs, artifact_root)
        snapshot["status"] = (
            "completed"
            if int(snapshot["completed_count"]) == len(jobs)
            else "incomplete"
        )
        atomic_write_json(state_path, snapshot)
        return snapshot
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = [
    "load_nnunet_jobs",
    "load_nnunet_runner_config",
    "nnunet_queue_snapshot",
    "official_output_directory",
    "run_nnunet_main_queue",
]
