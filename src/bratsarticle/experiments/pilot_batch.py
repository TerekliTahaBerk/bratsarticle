"""Sequential, restart-safe execution of all frozen Gate 8 pilot arms."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bratsarticle.experiments.pilot_runner import run_pilot_arm
from bratsarticle.experiments.pilots import load_pilot_plan, pilot_preflight
from bratsarticle.utils.hashing import file_digest


def pilot_run_id(arm_id: str, seed: int, config_hash: str) -> str:
    """Build a deterministic run ID from the frozen scientific identity."""
    return f"gate8_{arm_id}_s{seed}_{config_hash[:8]}"


def existing_run_is_reusable(
    run_directory: Path,
    *,
    arm_id: str,
    config_hash: str,
) -> bool:
    """Return whether a prior run is clean, complete, and identity-matched."""
    metadata_path = run_directory / "metadata.json"
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return bool(
        metadata.get("status") == "completed"
        and metadata.get("repository_dirty") is False
        and metadata.get("tags", {}).get("pilot_arm_id") == arm_id
        and metadata.get("tags", {}).get("pilot_config_sha256") == config_hash
        and metadata.get("test_access", {}).get("accessed") is False
    )


def run_all_pilot_arms(
    *,
    config_path: Path,
    allow_pilot_training: bool,
) -> list[dict[str, Any]]:
    """Run or safely reuse every frozen arm in config order."""
    if not allow_pilot_training:
        raise PermissionError("All-pilot execution requires --allow-pilot-training")
    plan = load_pilot_plan(config_path)
    preflight = pilot_preflight(plan)
    if not preflight["eligible"]:
        failed = [
            key for key, passed in preflight["checks"].items() if not bool(passed)
        ]
        raise RuntimeError(f"Gate 8 all-pilot preflight failed: {failed}")
    config_hash = file_digest(config_path)
    events: list[dict[str, Any]] = []
    for index, arm in enumerate(plan.arms, start=1):
        run_id = pilot_run_id(arm.arm_id, arm.seed, config_hash)
        run_directory = plan.artifact_root.resolve() / run_id
        if run_directory.exists():
            if not existing_run_is_reusable(
                run_directory,
                arm_id=arm.arm_id,
                config_hash=config_hash,
            ):
                raise RuntimeError(
                    f"Existing run cannot be reused safely: {run_directory}"
                )
            event = {
                "event": "pilot_arm_reused",
                "index": index,
                "total": len(plan.arms),
                "arm_id": arm.arm_id,
                "run_directory": run_directory.as_posix(),
            }
            print(json.dumps(event, sort_keys=True), flush=True)
            events.append(event)
            continue
        started = {
            "event": "pilot_arm_started",
            "index": index,
            "total": len(plan.arms),
            "arm_id": arm.arm_id,
            "run_id": run_id,
        }
        print(json.dumps(started, sort_keys=True), flush=True)
        completed_directory = run_pilot_arm(
            plan_path=config_path,
            arm_id=arm.arm_id,
            allow_pilot_training=True,
            run_id=run_id,
        )
        event = {
            "event": "pilot_arm_completed",
            "index": index,
            "total": len(plan.arms),
            "arm_id": arm.arm_id,
            "run_directory": completed_directory.as_posix(),
        }
        print(json.dumps(event, sort_keys=True), flush=True)
        events.append(event)
    return events
