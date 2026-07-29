"""Write the frozen Gate 9 plan and current-host preflight artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from bratsarticle.experiments.pilots import (
    load_pilot_plan,
    pilot_plan_record,
    pilot_preflight,
)
from bratsarticle.utils.serialization import atomic_write_json


def main() -> int:
    """Create plan/preflight reports without training or test access."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate9.yaml"),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("reports/gate9_plan.json"),
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("reports/gate9_preflight.json"),
    )
    parser.add_argument("--require-eligible-host", action="store_true")
    arguments = parser.parse_args()
    plan = load_pilot_plan(arguments.config)
    if plan.gate != 9:
        raise ValueError("Gate 9 preparation requires a Gate 9 plan")
    plan_record = pilot_plan_record(plan, arguments.config)
    preflight = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pilot_config_sha256": plan_record["source_config_sha256"],
        **pilot_preflight(plan),
    }
    atomic_write_json(arguments.plan_output, plan_record)
    atomic_write_json(arguments.preflight_output, preflight)
    if arguments.require_eligible_host and not preflight["eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
