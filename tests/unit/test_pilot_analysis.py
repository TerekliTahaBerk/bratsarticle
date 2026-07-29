from pathlib import Path

import numpy as np
import pytest

from bratsarticle.experiments.pilot_analysis import (
    PilotArtifactsIncompleteError,
    analyze_pilot_artifacts,
    audit_pilot_artifacts,
    paired_bootstrap_mean_interval,
)
from bratsarticle.experiments.pilots import load_pilot_plan


def test_paired_bootstrap_interval_is_deterministic_and_paired() -> None:
    differences = np.linspace(-0.12, -0.08, 37)

    first = paired_bootstrap_mean_interval(
        differences,
        resamples=10000,
        confidence_level=0.95,
        seed=20260729,
    )
    second = paired_bootstrap_mean_interval(
        differences,
        resamples=10000,
        confidence_level=0.95,
        seed=20260729,
    )

    assert first == second
    assert first[1] < 0.0
    assert np.mean(differences) < -0.02


def test_artifact_audit_blocks_shortlist_when_all_runs_are_missing(
    tmp_path: Path,
) -> None:
    plan_path = Path("configs/pilots/gate8.yaml")
    plan = load_pilot_plan(plan_path)

    audit, valid = audit_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=tmp_path / "empty-runs",
    )

    assert audit["status"] == "incomplete"
    assert audit["valid_arm_count"] == 0
    assert len(audit["missing_or_invalid_arms"]) == 12
    assert not audit["shortlist_permitted"]
    assert not valid
    with pytest.raises(PilotArtifactsIncompleteError, match="forbidden"):
        analyze_pilot_artifacts(
            plan=plan,
            plan_path=plan_path,
            artifact_root=tmp_path / "empty-runs",
        )
