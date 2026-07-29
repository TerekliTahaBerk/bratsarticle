"""Freeze the Gate 10 split, checkpoints, subgroups, and statistical plan."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.gate10 import (
    execute_gate10_freeze,
    load_gate10_plan,
    resolve_gate10_paths,
)


def main() -> int:
    """Create deterministic Gate 10 artifacts from a clean protocol commit."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/statistics/gate10.yaml"),
    )
    arguments = parser.parse_args()
    plan = load_gate10_plan(arguments.config)
    paths = resolve_gate10_paths(plan, arguments.config)
    execute_gate10_freeze(paths, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
