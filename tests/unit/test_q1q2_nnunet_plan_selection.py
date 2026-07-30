from __future__ import annotations

import json
from pathlib import Path

import pytest

from bratsarticle.adapters.nnunetv2 import (
    NNUNET_3D_FALLBACK_PLANS,
    NNUNET_3D_PRIMARY_PLANS,
)
from bratsarticle.experiments.q1q2_nnunet_plan_selection import (
    select_nnunet_3d_plan,
)


def _preflight(path: Path, plan: str, status: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "purpose": "hardware_feasibility_not_model_evaluation",
                "status": status,
                "configuration": "3d_fullres",
                "plans_identifier": plan,
                "external_data_accessed": False,
                "repository_dirty_at_start": False,
                "diagnostic_training_loss": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_primary_resenc_l_is_selected_when_feasible(tmp_path: Path) -> None:
    selection = select_nnunet_3d_plan(
        primary_preflight_path=_preflight(
            tmp_path / "primary.json",
            NNUNET_3D_PRIMARY_PLANS,
            "pass",
        ),
        fallback_preflight_path=None,
    )

    assert selection["selected_plans_identifier"] == NNUNET_3D_PRIMARY_PLANS
    assert selection["performance_outcomes_used"] is False
    assert len(selection["evidence"]) == 1


def test_fallback_resenc_m_requires_primary_failure_and_own_pass(
    tmp_path: Path,
) -> None:
    selection = select_nnunet_3d_plan(
        primary_preflight_path=_preflight(
            tmp_path / "primary.json",
            NNUNET_3D_PRIMARY_PLANS,
            "fail",
        ),
        fallback_preflight_path=_preflight(
            tmp_path / "fallback.json",
            NNUNET_3D_FALLBACK_PLANS,
            "pass",
        ),
    )

    assert selection["selected_plans_identifier"] == NNUNET_3D_FALLBACK_PLANS
    assert len(selection["evidence"]) == 2


def test_failed_primary_without_fallback_blocks_selection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="fallback"):
        select_nnunet_3d_plan(
            primary_preflight_path=_preflight(
                tmp_path / "primary.json",
                NNUNET_3D_PRIMARY_PLANS,
                "fail",
            ),
            fallback_preflight_path=None,
        )


def test_performance_outcome_in_preflight_is_rejected(tmp_path: Path) -> None:
    primary = _preflight(
        tmp_path / "primary.json",
        NNUNET_3D_PRIMARY_PLANS,
        "pass",
    )
    report = json.loads(primary.read_text(encoding="utf-8"))
    report["mean_regional_dice"] = 0.9
    primary.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(PermissionError, match="Performance"):
        select_nnunet_3d_plan(
            primary_preflight_path=primary,
            fallback_preflight_path=None,
        )
