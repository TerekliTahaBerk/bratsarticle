"""Generate the human-readable Gate 8 completion report from artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from bratsarticle.utils.serialization import atomic_write_text


def _resource_row(artifact_root: Path, run_id: str) -> dict[str, Any]:
    run_directory = artifact_root / run_id
    metadata = json.loads(
        (run_directory / "metadata.json").read_text(encoding="utf-8")
    )
    resources = json.loads(
        (run_directory / "resource_profile.json").read_text(encoding="utf-8")
    )
    return {
        "status": str(metadata["status"]),
        "repository_dirty": bool(metadata["repository_dirty"]),
        "test_accessed": bool(metadata["test_access"]["accessed"]),
        "steps": int(resources["completed_optimizer_steps"]),
        "validation_checks": int(resources["completed_validation_checks"]),
        "gpu_hours": float(resources["gpu_hours"]),
    }


def main() -> int:
    """Write a deterministic report containing no hand-entered result values."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis",
        type=Path,
        default=Path("reports/gate8_pilot_analysis.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("reports/gate8_artifact_audit.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/gate8_arm_summary.csv"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gate8_completion.md"),
    )
    arguments = parser.parse_args()
    analysis = json.loads(arguments.analysis.read_text(encoding="utf-8"))
    audit = json.loads(arguments.audit.read_text(encoding="utf-8"))
    summary = pd.read_csv(arguments.summary)
    resources = {
        str(run_id): _resource_row(arguments.artifact_root, str(run_id))
        for run_id in summary["run_id"].astype(str).unique()
    }
    unique_resource_rows = list(resources.values())
    all_valid = all(
        row["status"] == "completed"
        and not row["repository_dirty"]
        and not row["test_accessed"]
        and row["steps"] == 2000
        and row["validation_checks"] >= 1
        and row["gpu_hours"] <= 0.5
        for row in unique_resource_rows
    )
    if audit["status"] != "complete" or not all_valid:
        raise RuntimeError("Gate 8 completion report requires a complete valid audit")

    lines = [
        "# Gate 8 Completion",
        "",
        "**Decision:** PASS",
        "",
        "## Artifact audit",
        "",
        f"- Valid reportable arms: {audit['valid_arm_count']}/"
        f"{audit['expected_arm_count']}",
        f"- Validation patients per arm: {audit['expected_validation_patients']}",
        f"- Invalid runs: {len(audit['invalid_runs'])}",
        f"- Duplicate arms: {len(audit['duplicate_arm_runs'])}",
        f"- Internal-test access used: {str(audit['test_access_used']).lower()}",
        f"- GPU-hour range: {min(row['gpu_hours'] for row in unique_resource_rows):.6f}"
        f"-{max(row['gpu_hours'] for row in unique_resource_rows):.6f}",
        "",
        "Diagnostic and prior-protocol runs listed as foreign by the audit were "
        "excluded from selection.",
        "",
        "## Architecture screen",
        "",
        "| Arm | Mean regional Dice | Paired mean difference | 95% bootstrap CI | "
        "Eliminated |",
        "|---|---:|---:|---:|:---:|",
    ]
    architecture = summary.loc[summary["screen"] == "architecture"]
    for row in architecture.itertuples(index=False):
        lines.append(
            f"| {row.arm_id} | {row.mean_regional_dice:.6f} | "
            f"{row.paired_mean_difference:.6f} | "
            f"[{row.paired_bootstrap_lower:.6f}, "
            f"{row.paired_bootstrap_upper:.6f}] | "
            f"{'yes' if row.eliminated else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Shortlist: "
            + ", ".join(analysis["architecture_screen"]["shortlist"]),
            "",
            "## Loss screen",
            "",
            "| Arm | Mean regional Dice | Paired mean difference | 95% bootstrap CI | "
            "Eliminated |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    losses = summary.loc[summary["screen"] == "loss"]
    for row in losses.itertuples(index=False):
        lines.append(
            f"| {row.arm_id} | {row.mean_regional_dice:.6f} | "
            f"{row.paired_mean_difference:.6f} | "
            f"[{row.paired_bootstrap_lower:.6f}, "
            f"{row.paired_bootstrap_upper:.6f}] | "
            f"{'yes' if row.eliminated else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Shortlist: " + ", ".join(analysis["loss_screen"]["shortlist"]),
            "",
            "## Scope",
            "",
            "These are single-seed development-screen results. They support "
            "shortlisting only; they are not internal held-out test results and "
            "do not establish generalization, clinical applicability, or final "
            "model superiority.",
        ]
    )
    atomic_write_text(arguments.output, "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
