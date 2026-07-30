#!/usr/bin/env python3
"""Run or resume the complete frozen M1 loss-screen queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_m1_queue import run_loss_screen_queue


def main() -> int:
    """Run the guarded sequential queue."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-reportable-development-training", action="store_true")
    parser.add_argument("--dataset-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_loss_screen_queue(
        runner_config_path=Path("configs/q1q2_v2/m1_native_runner.yaml"),
        dataset_root=arguments.dataset_root,
        runtime_root=Path("artifacts/q1q2_v2/queue_runtime"),
        allow_reportable_development_training=(
            arguments.allow_reportable_development_training
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
