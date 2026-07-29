"""Run one non-reportable MPS/data/evaluator integration diagnostic."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bratsarticle.experiments.pilot_runner import run_pilot_arm
from bratsarticle.experiments.pilots import write_mps_diagnostic_config


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/pilots/gate8.yaml"),
    )
    parser.add_argument("--arm", default="architecture_unet")
    parser.add_argument("--allow-pilot-training", action="store_true")
    arguments = parser.parse_args()
    destination = Path(tempfile.gettempdir()) / "bratsarticle-gate8-mps-smoke.yaml"
    write_mps_diagnostic_config(arguments.source_config, destination)
    run_id = "gate8_mps_diagnostic_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    try:
        run_directory = run_pilot_arm(
            plan_path=destination,
            arm_id=arguments.arm,
            allow_pilot_training=arguments.allow_pilot_training,
            run_id=run_id,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "gate8_mps_diagnostic_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "scope": "diagnostic_only_not_for_selection",
                "run_directory": run_directory.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
