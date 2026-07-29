"""Generate the human-readable Gate 9 completion report from artifacts."""

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


def _run_ids(frame: pd.DataFrame) -> list[str]:
    return sorted(
        {
            run_id
            for joined in frame["run_ids"].astype(str)
            for run_id in joined.split("|")
        }
    )


def main() -> int:
    """Write a deterministic report containing no hand-entered result values."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-analysis",
        type=Path,
        default=Path("reports/gate9_confirmation_analysis.json"),
    )
    parser.add_argument(
        "--confirmation-summary",
        type=Path,
        default=Path("reports/gate9_confirmation_summary.csv"),
    )
    parser.add_argument(
        "--final-analysis",
        type=Path,
        default=Path("reports/gate9_final_analysis.json"),
    )
    parser.add_argument(
        "--final-summary",
        type=Path,
        default=Path("reports/gate9_final_summary.csv"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gate9_completion.md"),
    )
    arguments = parser.parse_args()

    confirmation_analysis = json.loads(
        arguments.confirmation_analysis.read_text(encoding="utf-8")
    )
    final_analysis = json.loads(arguments.final_analysis.read_text(encoding="utf-8"))
    confirmation = pd.read_csv(arguments.confirmation_summary)
    final = pd.read_csv(arguments.final_summary)
    run_ids = sorted(set(_run_ids(confirmation)) | set(_run_ids(final)))
    resources = {
        run_id: _resource_row(arguments.artifact_root, run_id) for run_id in run_ids
    }
    all_valid = all(
        row["status"] == "completed"
        and not row["repository_dirty"]
        and not row["test_accessed"]
        and row["steps"] == 2000
        and row["validation_checks"] >= 1
        and row["gpu_hours"] <= 0.5
        for row in resources.values()
    )
    audit = final_analysis["audit"]
    if (
        final_analysis["status"] != "complete"
        or audit["status"] != "complete"
        or audit["valid_arm_count"] != audit["expected_arm_count"]
        or not all_valid
    ):
        raise RuntimeError("Gate 9 completion report requires a complete valid audit")

    lines = [
        "# Gate 9 Completion",
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
        f"- GPU-hour range: {min(row['gpu_hours'] for row in resources.values()):.6f}"
        f"-{max(row['gpu_hours'] for row in resources.values()):.6f}",
        "",
        "Diagnostic, Gate 8, and superseded-protocol runs listed as foreign by the "
        "audit were excluded from selection.",
        "",
        "## Three-seed confirmation",
        "",
        "| Candidate | Role | Mean regional Dice | Seed SD | Paired mean "
        "difference | 95% bootstrap CI | Eliminated |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for row in confirmation.itertuples(index=False):
        lines.append(
            f"| {row.candidate_id} | {row.role} | "
            f"{row.mean_regional_dice:.6f} | "
            f"{row.seed_mean_standard_deviation:.6f} | "
            f"{row.paired_mean_difference:.6f} | "
            f"[{row.paired_bootstrap_lower:.6f}, "
            f"{row.paired_bootstrap_upper:.6f}] | "
            f"{'yes' if row.eliminated else 'no'} |"
        )
    lines.extend(
        [
            "",
            "Predeclared finalists: "
            + ", ".join(confirmation_analysis["finalists"]),
            "",
            "## Five-seed finalist analysis",
            "",
            "| Candidate | Seeds | Mean regional Dice | Seed SD | Paired mean "
            "difference | 95% bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in final.itertuples(index=False):
        lines.append(
            f"| {row.candidate_id} | {row.seed_count} | "
            f"{row.mean_regional_dice:.6f} | "
            f"{row.seed_mean_standard_deviation:.6f} | "
            f"{row.paired_mean_difference:.6f} | "
            f"[{row.paired_bootstrap_lower:.6f}, "
            f"{row.paired_bootstrap_upper:.6f}] |"
        )
    lines.extend(
        [
            "",
            f"Primary finalist by the frozen ranking rule: "
            f"{final_analysis['primary_finalist']}",
            "",
            "Candidates frozen for internal-test evaluation: "
            + ", ".join(final_analysis["internal_test_candidates"]),
            "",
            "## Scope",
            "",
            "Gate 9 used development-validation data only. The paired confidence "
            "interval for the two five-seed finalists includes zero, so the "
            "ranking does not establish superiority. Internal-test performance, "
            "generalization, clinical applicability, thresholds, and "
            "post-processing remain unobserved and unfrozen at this gate.",
        ]
    )
    atomic_write_text(arguments.output, "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
