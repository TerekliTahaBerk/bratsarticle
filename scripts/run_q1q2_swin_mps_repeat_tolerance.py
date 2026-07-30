#!/usr/bin/env python3
"""Run the real-data Swin MPS repeat-tolerance audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_swin_tolerance import (
    run_swin_repeat_tolerance,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-training-diagnostics", action="store_true")
    parser.add_argument("--dataset-root", type=Path, required=True)
    arguments = parser.parse_args()
    report = run_swin_repeat_tolerance(
        runner_config_path=Path("configs/q1q2_v2/swin_m1_runner.yaml"),
        selected_loss_path=Path("configs/q1q2_v2/selected_loss.yaml"),
        dataset_root=arguments.dataset_root,
        output_path=Path("reports/q1q2_v2/swin_mps_repeat_tolerance.json"),
        allow_training_diagnostics=arguments.allow_training_diagnostics,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
