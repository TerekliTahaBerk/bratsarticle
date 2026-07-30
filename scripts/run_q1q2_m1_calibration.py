#!/usr/bin/env python3
"""Run the nonreportable Q1/Q2 M1 Max calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.m1_calibration import run_calibration


def main() -> int:
    """Run guarded calibration and return nonzero if any workload fails."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-nonreportable-calibration", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_nonreportable_calibration:
        raise PermissionError(
            "M1 calibration requires --allow-nonreportable-calibration"
        )
    result = run_calibration(
        calibration_path=Path("configs/q1q2_v2/m1_calibration.yaml"),
        matrix_path=Path("configs/q1q2_v2/model_matrix.yaml"),
        loss_catalog_path=Path("configs/losses/catalog.yaml"),
        output_json=Path("reports/q1q2_v2/m1_calibration.json"),
        output_markdown=Path("reports/q1q2_v2/m1_calibration.md"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
