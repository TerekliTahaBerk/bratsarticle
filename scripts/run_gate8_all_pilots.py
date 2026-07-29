"""Sequential, restart-safe execution of all frozen Gate 8 pilot arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.pilot_batch import run_all_pilot_arms


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate8.yaml"),
    )
    parser.add_argument("--allow-pilot-training", action="store_true")
    arguments = parser.parse_args()
    try:
        events = run_all_pilot_arms(
            config_path=arguments.config,
            allow_pilot_training=arguments.allow_pilot_training,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "gate8_all_pilots_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(
        json.dumps(
            {
                "event": "gate8_all_pilots_completed",
                "arm_count": len(events),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
