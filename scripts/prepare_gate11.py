"""Run the no-test-access Gate 11 preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.gate11_runner import (
    load_gate11_plan,
    write_gate11_preflight,
)


def main() -> int:
    """Write a host-specific Gate 11 eligibility report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/internal_test/gate11.yaml"),
    )
    arguments = parser.parse_args()
    plan = load_gate11_plan(arguments.config)
    output = Path(str(plan["outputs"]["preflight"]))
    report = write_gate11_preflight(arguments.config, output)
    if not report["eligible"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
