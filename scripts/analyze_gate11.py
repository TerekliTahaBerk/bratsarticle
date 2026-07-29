"""Analyze the frozen Gate 11 internal-test artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.gate11_analysis import analyze_gate11


def main() -> int:
    """Generate Gate 11 audit, statistics, resources, and completion report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/internal_test/gate11.yaml"),
    )
    arguments = parser.parse_args()
    analyze_gate11(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
