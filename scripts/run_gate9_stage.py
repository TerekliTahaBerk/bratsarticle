"""Run one restart-safe Gate 9 execution stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.gate9 import (
    extension_arms_for_finalists,
    stage_arms,
)
from bratsarticle.experiments.pilot_batch import (
    existing_run_is_reusable,
    pilot_run_id,
)
from bratsarticle.experiments.pilot_runner import run_pilot_arm
from bratsarticle.experiments.pilots import load_pilot_plan, pilot_preflight
from bratsarticle.utils.hashing import file_digest


def main() -> int:
    """Run confirmation arms or only the selected finalist extensions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate9.yaml"),
    )
    parser.add_argument(
        "--stage",
        choices=("confirmation", "finalist_extension"),
        required=True,
    )
    parser.add_argument(
        "--confirmation-analysis",
        type=Path,
        default=Path("reports/gate9_confirmation_analysis.json"),
    )
    parser.add_argument("--allow-pilot-training", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_pilot_training:
        raise PermissionError("Gate 9 execution requires --allow-pilot-training")
    plan = load_pilot_plan(arguments.config)
    if plan.gate != 9:
        raise ValueError("Gate 9 execution requires a Gate 9 config")
    preflight = pilot_preflight(plan)
    if not preflight["eligible"]:
        failed = [
            key for key, passed in preflight["checks"].items() if not bool(passed)
        ]
        raise RuntimeError(f"Gate 9 preflight failed: {failed}")
    if arguments.stage == "confirmation":
        arms = stage_arms(plan, "confirmation")
    else:
        analysis = json.loads(
            arguments.confirmation_analysis.read_text(encoding="utf-8")
        )
        if analysis.get("status") != "confirmation_complete":
            raise RuntimeError("Finalist extension requires complete confirmation")
        arms = extension_arms_for_finalists(
            plan,
            [str(value) for value in analysis["finalists"]],
        )

    config_hash = file_digest(arguments.config)
    for index, arm in enumerate(arms, start=1):
        run_id = pilot_run_id(plan.gate, arm.arm_id, arm.seed, config_hash)
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
            event = "gate9_arm_reused"
        else:
            print(
                json.dumps(
                    {
                        "event": "gate9_arm_started",
                        "stage": arguments.stage,
                        "index": index,
                        "total": len(arms),
                        "arm_id": arm.arm_id,
                        "candidate_id": arm.candidate_id,
                        "seed": arm.seed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            run_directory = run_pilot_arm(
                plan_path=arguments.config,
                arm_id=arm.arm_id,
                allow_pilot_training=True,
                run_id=run_id,
            )
            event = "gate9_arm_completed"
        print(
            json.dumps(
                {
                    "event": event,
                    "stage": arguments.stage,
                    "index": index,
                    "total": len(arms),
                    "arm_id": arm.arm_id,
                    "candidate_id": arm.candidate_id,
                    "seed": arm.seed,
                    "run_directory": run_directory.as_posix(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
