"""Audit and analyze Gate 9 confirmation or finalist-extension artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.gate9 import (
    analyze_confirmation,
    analyze_finalists,
    write_gate9_analysis,
)
from bratsarticle.experiments.pilots import load_pilot_plan


def main() -> int:
    """Write confirmation or final multi-seed analysis artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate9.yaml"),
    )
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/gate9_confirmation_analysis.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("reports/gate9_confirmation_summary.csv"),
    )
    arguments = parser.parse_args()
    plan = load_pilot_plan(arguments.config)
    if arguments.finalize:
        result, rows = analyze_finalists(plan=plan, plan_path=arguments.config)
    else:
        result, rows = analyze_confirmation(
            plan=plan,
            plan_path=arguments.config,
        )
    write_gate9_analysis(
        result=result,
        rows=rows,
        json_output=arguments.json_output,
        csv_output=arguments.csv_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
