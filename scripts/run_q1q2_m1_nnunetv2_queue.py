#!/usr/bin/env python3
"""Run or resume the frozen 50-job official nnU-Net M1 queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_nnunet_queue import run_nnunet_main_queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-reportable-development-training", action="store_true")
    arguments = parser.parse_args()
    result = run_nnunet_main_queue(
        runner_config_path=Path("configs/q1q2_v2/nnunet_m1_runner.yaml"),
        allow_reportable_development_training=(
            arguments.allow_reportable_development_training
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
