"""Sequential, restart-safe queue control for bounded M1 development work."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bratsarticle.experiments.q1q2_native_runner import (
    NativeRunSpec,
    loss_screen_specs,
    run_native_development,
)
from bratsarticle.utils.serialization import atomic_write_json


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _run_status(artifact_root: Path, spec: NativeRunSpec) -> str:
    progress_path = artifact_root / spec.run_id / "progress.json"
    if not progress_path.is_file():
        return "not_started"
    progress = cast(
        dict[str, Any],
        json.loads(progress_path.read_text(encoding="utf-8")),
    )
    return str(progress.get("status", "unknown"))


def queue_snapshot(
    *,
    specs: tuple[NativeRunSpec, ...],
    artifact_root: Path,
) -> dict[str, Any]:
    """Return the current immutable-job/runtime-status mapping."""
    jobs = [
        {
            "run_id": spec.run_id,
            "spec_sha256": spec.sha256,
            "status": _run_status(artifact_root, spec),
        }
        for spec in specs
    ]
    return {
        "schema_version": 1,
        "updated_at_utc": _timestamp(),
        "job_count": len(jobs),
        "completed_count": sum(job["status"] == "completed" for job in jobs),
        "failed_count": sum(job["status"] == "failed" for job in jobs),
        "running_or_resumable_count": sum(
            job["status"] in {"running", "unknown"} for job in jobs
        ),
        "not_started_count": sum(job["status"] == "not_started" for job in jobs),
        "jobs": jobs,
    }


def run_loss_screen_queue(
    *,
    runner_config_path: Path,
    dataset_root: Path,
    runtime_root: Path,
    allow_reportable_development_training: bool,
) -> dict[str, Any]:
    """Run all frozen loss-screen jobs, resuming only identical partial jobs."""
    if not allow_reportable_development_training:
        raise PermissionError(
            "Loss-screen queue requires reportable development authorization"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_root / "loss_screen.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "Loss-screen queue lock already exists; verify the existing process"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    specs = loss_screen_specs(runner_config_path)
    artifact_root = Path("artifacts/q1q2_v2/native_runs").resolve()
    state_path = runtime_root / "loss_screen_runtime.json"
    try:
        snapshot = queue_snapshot(specs=specs, artifact_root=artifact_root)
        atomic_write_json(state_path, snapshot)
        for spec in specs:
            status = _run_status(artifact_root, spec)
            if status == "completed":
                continue
            output_dir = artifact_root / spec.run_id
            run_native_development(
                runner_config_path=runner_config_path,
                spec=spec,
                dataset_root=dataset_root,
                allow_reportable_development_training=True,
                resume=output_dir.exists(),
            )
            snapshot = queue_snapshot(specs=specs, artifact_root=artifact_root)
            atomic_write_json(state_path, snapshot)
        snapshot = queue_snapshot(specs=specs, artifact_root=artifact_root)
        snapshot["status"] = (
            "completed"
            if int(snapshot["completed_count"]) == len(specs)
            else "incomplete"
        )
        atomic_write_json(state_path, snapshot)
        return snapshot
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = ["queue_snapshot", "run_loss_screen_queue"]
