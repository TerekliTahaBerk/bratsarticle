"""Audit Gate 8 registry artifacts and generate a shortlist when complete."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.pilot_analysis import (
    PilotArtifactsIncompleteError,
    analyze_pilot_artifacts,
    audit_pilot_artifacts,
    write_pilot_analysis,
)
from bratsarticle.experiments.pilots import load_pilot_plan
from bratsarticle.utils.serialization import atomic_write_json


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilots/gate8.yaml"),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("reports/gate8_artifact_audit.json"),
    )
    parser.add_argument(
        "--analysis-output",
        type=Path,
        default=Path("reports/gate8_pilot_analysis.json"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/gate8_arm_summary.csv"),
    )
    arguments = parser.parse_args()
    plan = load_pilot_plan(arguments.config)
    audit, _ = audit_pilot_artifacts(
        plan=plan,
        plan_path=arguments.config,
        artifact_root=arguments.artifact_root,
    )
    atomic_write_json(arguments.audit_output, audit)
    try:
        result, rows = analyze_pilot_artifacts(
            plan=plan,
            plan_path=arguments.config,
            artifact_root=arguments.artifact_root,
        )
    except PilotArtifactsIncompleteError:
        return 2
    write_pilot_analysis(
        result=result,
        rows=rows,
        json_output=arguments.analysis_output,
        csv_output=arguments.summary_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
