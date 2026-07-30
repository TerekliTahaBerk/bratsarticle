"""Generate the prespecified external subgroup analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.analysis.q1q2_subgroups import (
    analyze_q1q2_external_subgroups,
)


def main() -> int:
    """Run exploratory subgroup estimation after the frozen main analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/q1q2_v2/subgroup_execution.yaml"),
    )
    arguments = parser.parse_args()
    analyze_q1q2_external_subgroups(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
