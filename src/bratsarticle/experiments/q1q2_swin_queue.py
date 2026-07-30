"""Sequential, restart-safe queue control for reportable Swin development."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from bratsarticle.experiments.q1q2_swin_runner import (
    SwinRunSpec,
    run_swin_development,
    swin_convergence_specs,
)
from bratsarticle.utils.serialization import atomic_write_json


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _run_status(artifact_root: Path, spec: SwinRunSpec) -> str:
    progress_path = artifact_root / spec.run_id / "progress.json"
    if not progress_path.is_file():
        return "not_started"
    progress = cast(
        dict[str, Any],
        json.loads(progress_path.read_text(encoding="utf-8")),
    )
    return str(progress.get("status", "unknown"))


def swin_queue_snapshot(
    specs: tuple[SwinRunSpec, ...],
    artifact_root: Path,
) -> dict[str, Any]:
    """Return immutable Swin jobs with their current runtime states."""
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


def run_swin_main_queue(
    *,
    runner_config_path: Path,
    selected_loss_path: Path,
    dataset_root: Path,
    runtime_root: Path,
    allow_reportable_development_training: bool,
) -> dict[str, Any]:
    """Run or resume all 25 frozen Swin jobs sequentially."""
    if not allow_reportable_development_training:
        raise PermissionError(
            "Swin main queue requires reportable development authorization"
        )
    runtime_root.mkdir(parents=True, exist_ok=True)
    conflicting_locks = (
        "loss_screen.lock",
        "native_main.lock",
        "nnunetv2_main.lock",
        "nnunetv2_preflight.lock",
    )
    present = [name for name in conflicting_locks if (runtime_root / name).exists()]
    if present:
        raise RuntimeError(
            "Swin queue conflicts with active MPS work: " + ", ".join(present)
        )
    lock_path = runtime_root / "swin_main.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            "Swin main queue lock already exists; verify the existing process"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()}\n")
    specs = swin_convergence_specs(runner_config_path, selected_loss_path)
    artifact_root = Path("artifacts/q1q2_v2/swin_runs").resolve()
    state_path = runtime_root / "swin_main_runtime.json"
    try:
        snapshot = swin_queue_snapshot(specs, artifact_root)
        atomic_write_json(state_path, snapshot)
        for spec in specs:
            status = _run_status(artifact_root, spec)
            if status == "completed":
                continue
            output_dir = artifact_root / spec.run_id
            run_swin_development(
                runner_config_path=runner_config_path,
                selected_loss_path=selected_loss_path,
                spec=spec,
                dataset_root=dataset_root,
                allow_reportable_development_training=True,
                resume=output_dir.exists(),
            )
            snapshot = swin_queue_snapshot(specs, artifact_root)
            atomic_write_json(state_path, snapshot)
        snapshot = swin_queue_snapshot(specs, artifact_root)
        snapshot["status"] = (
            "completed"
            if int(snapshot["completed_count"]) == len(specs)
            else "incomplete"
        )
        atomic_write_json(state_path, snapshot)
        return snapshot
    finally:
        lock_path.unlink(missing_ok=True)


__all__ = ["run_swin_main_queue", "swin_queue_snapshot"]
