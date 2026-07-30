#!/usr/bin/env python3
"""Run ordered Q1/Q2 training diagnostics on the selected M1 Max."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_diagnostics import run_ordered_diagnostics


def main() -> int:
    """Run guarded diagnostics and return nonzero on a failed stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-training-diagnostics", action="store_true")
    parser.add_argument("--dataset-root", type=Path, required=True)
    arguments = parser.parse_args()
    if not arguments.allow_training_diagnostics:
        raise PermissionError(
            "M1 training diagnostics require --allow-training-diagnostics"
        )
    result = run_ordered_diagnostics(
        config_path=Path("configs/q1q2_v2/diagnostics.yaml"),
        dataset_root=arguments.dataset_root,
        artifact_root=Path("artifacts/q1q2_v2/diagnostics"),
        output_json=Path("reports/q1q2_v2/m1_diagnostics.json"),
        output_markdown=Path("reports/q1q2_v2/m1_diagnostics.md"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
