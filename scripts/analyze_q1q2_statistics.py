"""Run the frozen Q1/Q2 patient-level statistical analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.analysis.q1q2_statistics import analyze_q1q2_statistics


def main() -> int:
    """Analyze passing Gate G/H artifacts without reopening external images."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1q2_v2/statistical_execution.yaml"),
    )
    arguments = parser.parse_args()
    analyze_q1q2_statistics(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
