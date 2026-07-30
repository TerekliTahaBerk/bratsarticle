"""Outcome-blind MPS feasibility selection for the frozen nnU-Net 3D plan."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from bratsarticle.adapters.nnunetv2 import (
    NNUNET_3D_FALLBACK_PLANS,
    NNUNET_3D_PRIMARY_PLANS,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_text


def _load_preflight(
    path: Path,
    *,
    expected_plan: str,
) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    report = cast(dict[str, Any], loaded)
    if report.get("purpose") != "hardware_feasibility_not_model_evaluation":
        raise ValueError("nnU-Net preflight has an invalid scientific purpose")
    if report.get("configuration") != "3d_fullres":
        raise ValueError("nnU-Net 3D plan selection requires a 3D preflight")
    if report.get("plans_identifier") != expected_plan:
        raise ValueError("nnU-Net preflight plan identifier differs")
    if report.get("external_data_accessed") is not False:
        raise PermissionError("nnU-Net plan preflight accessed external data")
    prohibited_outcome_keys = {
        "dice",
        "mean_regional_dice",
        "validation_metric",
        "external_metric",
    }
    if prohibited_outcome_keys.intersection(report):
        raise PermissionError("Performance outcomes cannot select the 3D plan")
    if report.get("repository_dirty_at_start") is not False:
        raise ValueError("nnU-Net preflight did not start from a clean repository")
    return report


def select_nnunet_3d_plan(
    *,
    primary_preflight_path: Path,
    fallback_preflight_path: Path | None,
) -> dict[str, Any]:
    """Select ResEnc-L if feasible, otherwise require a passing ResEnc-M test."""
    primary = _load_preflight(
        primary_preflight_path,
        expected_plan=NNUNET_3D_PRIMARY_PLANS,
    )
    evidence = [primary_preflight_path]
    if primary.get("status") == "pass":
        selected = NNUNET_3D_PRIMARY_PLANS
        reason = "primary_untouched_plan_passed_mps_feasibility"
    else:
        if fallback_preflight_path is None:
            raise RuntimeError(
                "ResEnc-L failed; a ResEnc-M fallback hardware preflight is required"
            )
        fallback = _load_preflight(
            fallback_preflight_path,
            expected_plan=NNUNET_3D_FALLBACK_PLANS,
        )
        evidence.append(fallback_preflight_path)
        if fallback.get("status") != "pass":
            raise RuntimeError(
                "Neither predeclared nnU-Net 3D plan passed MPS feasibility"
            )
        selected = NNUNET_3D_FALLBACK_PLANS
        reason = "primary_failed_and_predeclared_fallback_passed_mps_feasibility"
    return {
        "schema_version": 1,
        "status": "frozen_from_outcome_blind_hardware_feasibility",
        "model_id": "nnunetv2_3d_fullres",
        "selected_plans_identifier": selected,
        "selection_reason": reason,
        "selection_criterion": "untouched_plan_mps_feasibility_only",
        "performance_outcomes_used": False,
        "external_data_accessed": False,
        "primary_plans_identifier": NNUNET_3D_PRIMARY_PLANS,
        "fallback_plans_identifier": NNUNET_3D_FALLBACK_PLANS,
        "evidence": [
            {
                "path": path.as_posix(),
                "sha256": file_digest(path),
            }
            for path in evidence
        ],
    }


def write_nnunet_3d_plan_selection(
    selection: dict[str, Any],
    output_path: Path,
) -> None:
    """Atomically write the immutable plan selection as YAML."""
    import yaml

    atomic_write_text(
        output_path,
        yaml.safe_dump(selection, sort_keys=False),
    )


__all__ = ["select_nnunet_3d_plan", "write_nnunet_3d_plan_selection"]
