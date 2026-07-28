"""Create the frozen Gate 8 pilot-plan and current-host preflight artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bratsarticle.experiments.pilots import (
    load_pilot_plan,
    pilot_plan_record,
    pilot_preflight,
)
from bratsarticle.utils.serialization import atomic_write_json


def run(
    *,
    config_path: Path,
    plan_output: Path,
    preflight_output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Write plan and preflight records without starting training."""
    plan = load_pilot_plan(config_path)
    plan_record = pilot_plan_record(plan, config_path)
    preflight = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "pilot_config_sha256": plan_record["source_config_sha256"],
        **pilot_preflight(plan),
    }
    atomic_write_json(plan_output, plan_record)
    atomic_write_json(preflight_output, preflight)
    return plan_record, preflight


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate8.yaml"),
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=Path("reports/gate8_pilot_plan.json"),
    )
    parser.add_argument(
        "--preflight-output",
        type=Path,
        default=Path("reports/gate8_preflight.json"),
    )
    parser.add_argument("--require-eligible-host", action="store_true")
    arguments = parser.parse_args()
    _, preflight = run(
        config_path=arguments.config,
        plan_output=arguments.plan_output,
        preflight_output=arguments.preflight_output,
    )
    if arguments.require_eligible_host and not preflight["eligible"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
