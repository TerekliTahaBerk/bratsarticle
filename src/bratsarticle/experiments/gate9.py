"""Multi-seed Gate 9 confirmation, finalist extension, and analysis."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bratsarticle.experiments.pilot_analysis import (
    PilotArtifactsIncompleteError,
    ValidPilotArtifact,
    audit_pilot_artifacts,
    paired_bootstrap_mean_interval,
)
from bratsarticle.experiments.pilots import PilotArm, PilotPlan
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json


def stage_arms(plan: PilotPlan, execution_stage: str) -> tuple[PilotArm, ...]:
    """Return the predeclared arms for one Gate 9 execution stage."""
    arms = tuple(
        arm for arm in plan.arms if arm.execution_stage == execution_stage
    )
    if not arms:
        raise ValueError(f"No arms declared for stage {execution_stage}")
    return arms


def extension_arms_for_finalists(
    plan: PilotPlan,
    finalists: list[str],
) -> tuple[PilotArm, ...]:
    """Return only predeclared extension arms for selected finalists."""
    allowed = {
        str(value)
        for value in plan.elimination["finalist_eligible_candidates"]
    }
    requested = set(finalists)
    if not requested or not requested.issubset(allowed):
        raise ValueError("Finalist extension requested unknown/ineligible candidates")
    arms = tuple(
        arm
        for arm in stage_arms(plan, "finalist_extension")
        if arm.candidate_id in requested
    )
    expected = len(requested) * len(
        plan.elimination["finalist_extension_seeds"]
    )
    if len(arms) != expected:
        raise ValueError("Finalist extension arm count is inconsistent")
    return arms


def _candidate_values(
    *,
    candidate_id: str,
    arms: tuple[PilotArm, ...],
    artifacts: Mapping[str, ValidPilotArtifact],
    expected_seeds: set[int],
) -> tuple[pd.Series, dict[int, float], list[str]]:
    candidate_arms = [arm for arm in arms if arm.candidate_id == candidate_id]
    observed_seeds = {arm.seed for arm in candidate_arms}
    if observed_seeds != expected_seeds:
        raise ValueError(
            f"Candidate {candidate_id} seed mismatch: "
            f"{sorted(observed_seeds)} != {sorted(expected_seeds)}"
        )
    seed_series = {
        arm.seed: artifacts[arm.arm_id].values.rename(arm.seed)
        for arm in candidate_arms
    }
    frame = pd.concat(seed_series.values(), axis=1).sort_index()
    if frame.isna().any().any():
        raise ValueError(f"Candidate {candidate_id} has unpaired patient values")
    patient_mean = frame.mean(axis=1)
    seed_means = {
        int(seed): float(series.mean()) for seed, series in seed_series.items()
    }
    run_ids = [artifacts[arm.arm_id].run_id for arm in candidate_arms]
    return patient_mean, seed_means, run_ids


def _confirmation_rows(
    *,
    plan: PilotPlan,
    artifacts: Mapping[str, ValidPilotArtifact],
    arms: tuple[PilotArm, ...],
) -> tuple[list[dict[str, Any]], list[str], str, bool]:
    expected_seeds = {
        int(value) for value in plan.elimination["confirmation_seeds"]
    }
    candidate_ids = sorted({arm.candidate_id for arm in arms})
    aggregates: dict[str, pd.Series] = {}
    seed_means_by_candidate: dict[str, dict[int, float]] = {}
    run_ids_by_candidate: dict[str, list[str]] = {}
    for candidate_id in candidate_ids:
        aggregate, seed_means, run_ids = _candidate_values(
            candidate_id=candidate_id,
            arms=arms,
            artifacts=artifacts,
            expected_seeds=expected_seeds,
        )
        aggregates[candidate_id] = aggregate
        seed_means_by_candidate[candidate_id] = seed_means
        run_ids_by_candidate[candidate_id] = run_ids

    eligible = [
        str(value) for value in plan.elimination["finalist_eligible_candidates"]
    ]
    best_id = max(
        eligible,
        key=lambda candidate: (float(aggregates[candidate].mean()), candidate),
    )
    best_values = aggregates[best_id]
    margin = float(plan.elimination["practical_noninferiority_margin"])
    resamples = int(plan.elimination["paired_bootstrap_resamples"])
    confidence = float(plan.elimination["confidence_level"])
    rows: list[dict[str, Any]] = []
    retained: list[str] = []
    for index, candidate_id in enumerate(candidate_ids):
        paired = aggregates[candidate_id].reindex(best_values.index) - best_values
        lower, upper = paired_bootstrap_mean_interval(
            paired.to_numpy(dtype=np.float64),
            resamples=resamples,
            confidence_level=confidence,
            seed=plan.seed + index,
        )
        mean_difference = float(paired.mean())
        is_eligible = candidate_id in eligible
        eliminated = bool(
            is_eligible and mean_difference < -margin and upper < 0.0
        )
        if is_eligible and not eliminated:
            retained.append(candidate_id)
        seed_means = seed_means_by_candidate[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "role": "finalist_eligible" if is_eligible else "reference",
                "seed_count": len(seed_means),
                "patient_count": int(aggregates[candidate_id].size),
                "mean_regional_dice": float(aggregates[candidate_id].mean()),
                "median_regional_dice": float(aggregates[candidate_id].median()),
                "seed_mean_standard_deviation": float(
                    np.std(list(seed_means.values()), ddof=1)
                ),
                "paired_reference_candidate": best_id,
                "paired_mean_difference": mean_difference,
                "paired_bootstrap_lower": lower,
                "paired_bootstrap_upper": upper,
                "practical_margin": margin,
                "eliminated": eliminated,
                "run_ids": "|".join(sorted(run_ids_by_candidate[candidate_id])),
            }
        )

    top_k = int(plan.elimination["finalist_top_k"])
    minimum = int(plan.elimination["minimum_finalists"])
    ranked_eligible = sorted(
        eligible,
        key=lambda candidate: (-float(aggregates[candidate].mean()), candidate),
    )
    finalists = sorted(
        retained,
        key=lambda candidate: (-float(aggregates[candidate].mean()), candidate),
    )[:top_k]
    fallback_applied = False
    if len(finalists) < minimum:
        fallback_applied = True
        for candidate in ranked_eligible:
            if candidate not in finalists:
                finalists.append(candidate)
            if len(finalists) == minimum:
                break
    return rows, finalists, best_id, fallback_applied


def analyze_confirmation(
    *,
    plan: PilotPlan,
    plan_path: Path,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit 12 confirmation runs and select two extension finalists."""
    if plan.gate != 9:
        raise ValueError("Gate 9 analysis requires a Gate 9 plan")
    arms = stage_arms(plan, "confirmation")
    audit, artifacts = audit_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
        expected_arms=arms,
    )
    if audit["status"] != "complete":
        raise PilotArtifactsIncompleteError(
            "Gate 9 confirmation artifacts are incomplete"
        )
    rows, finalists, best_id, fallback_applied = _confirmation_rows(
        plan=plan,
        artifacts=artifacts,
        arms=arms,
    )
    result = {
        "status": "confirmation_complete",
        "audit": audit,
        "best_confirmation_candidate": best_id,
        "finalists": finalists,
        "minimum_finalist_fallback_applied": fallback_applied,
        "mandatory_reference_candidate": plan.elimination[
            "mandatory_reference_candidate"
        ],
        "confirmation_seeds": plan.elimination["confirmation_seeds"],
        "finalist_extension_seeds": plan.elimination[
            "finalist_extension_seeds"
        ],
        "internal_test_access": False,
    }
    return result, rows


def analyze_finalists(
    *,
    plan: PilotPlan,
    plan_path: Path,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Audit selected extensions and rank five-seed finalists."""
    confirmation, _ = analyze_confirmation(
        plan=plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
    )
    finalists = [str(value) for value in confirmation["finalists"]]
    confirmation_arms = stage_arms(plan, "confirmation")
    extension_arms = extension_arms_for_finalists(plan, finalists)
    expected_arms = (*confirmation_arms, *extension_arms)
    audit, artifacts = audit_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
        expected_arms=expected_arms,
    )
    if audit["status"] != "complete":
        raise PilotArtifactsIncompleteError(
            "Gate 9 finalist-extension artifacts are incomplete"
        )

    all_seeds = {
        *[int(value) for value in plan.elimination["confirmation_seeds"]],
        *[
            int(value)
            for value in plan.elimination["finalist_extension_seeds"]
        ],
    }
    finalist_arms = tuple(
        arm for arm in expected_arms if arm.candidate_id in finalists
    )
    aggregates: dict[str, pd.Series] = {}
    seed_means: dict[str, dict[int, float]] = {}
    run_ids: dict[str, list[str]] = {}
    for candidate in finalists:
        aggregate, candidate_seed_means, candidate_run_ids = _candidate_values(
            candidate_id=candidate,
            arms=finalist_arms,
            artifacts=artifacts,
            expected_seeds=all_seeds,
        )
        aggregates[candidate] = aggregate
        seed_means[candidate] = candidate_seed_means
        run_ids[candidate] = candidate_run_ids
    primary = max(
        finalists,
        key=lambda candidate: (float(aggregates[candidate].mean()), candidate),
    )
    primary_values = aggregates[primary]
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(sorted(finalists)):
        paired = aggregates[candidate].reindex(primary_values.index) - primary_values
        lower, upper = paired_bootstrap_mean_interval(
            paired.to_numpy(dtype=np.float64),
            resamples=int(plan.elimination["paired_bootstrap_resamples"]),
            confidence_level=float(plan.elimination["confidence_level"]),
            seed=plan.seed + 100 + index,
        )
        rows.append(
            {
                "candidate_id": candidate,
                "seed_count": len(seed_means[candidate]),
                "patient_count": int(aggregates[candidate].size),
                "mean_regional_dice": float(aggregates[candidate].mean()),
                "median_regional_dice": float(aggregates[candidate].median()),
                "seed_mean_standard_deviation": float(
                    np.std(list(seed_means[candidate].values()), ddof=1)
                ),
                "paired_reference_candidate": primary,
                "paired_mean_difference": float(paired.mean()),
                "paired_bootstrap_lower": lower,
                "paired_bootstrap_upper": upper,
                "run_ids": "|".join(sorted(run_ids[candidate])),
            }
        )
    result = {
        "status": "complete",
        "audit": audit,
        "confirmation_analysis": confirmation,
        "five_seed_finalists": finalists,
        "primary_finalist": primary,
        "mandatory_reference_candidate": plan.elimination[
            "mandatory_reference_candidate"
        ],
        "internal_test_candidates": [
            plan.elimination["mandatory_reference_candidate"],
            *finalists,
        ],
        "all_finalist_seeds": sorted(all_seeds),
        "internal_test_access": False,
    }
    return result, rows


def write_gate9_analysis(
    *,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    json_output: Path,
    csv_output: Path,
) -> None:
    """Write Gate 9 analysis artifacts atomically."""
    atomic_write_json(json_output, result)
    atomic_write_csv(csv_output, rows)
