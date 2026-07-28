"""CLI for one explicitly authorized Gate 8 pilot arm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.pilot_runner import run_pilot_arm


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate8.yaml"),
    )
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--allow-pilot-training", action="store_true")
    arguments = parser.parse_args()
    try:
        run_directory = run_pilot_arm(
            plan_path=arguments.config,
            arm_id=arguments.arm,
            allow_pilot_training=arguments.allow_pilot_training,
            run_id=arguments.run_id,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "gate8_pilot_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"run_directory": run_directory.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
