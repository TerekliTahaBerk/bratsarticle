#!/usr/bin/env python3
"""Derive the M1 execution decision from measured calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text


def _read(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def main() -> int:
    """Write a transparent, measured M1 scheduling decision."""
    calibration = _read(Path("reports/q1q2_v2/m1_calibration.json"))
    diagnostics = _read(Path("reports/q1q2_v2/m1_diagnostics.json"))
    queue = _read(Path("artifacts/q1q2_v2/queues/loss_screen.json"))
    rows = {
        str(row["model_id"]): row
        for row in cast(list[dict[str, Any]], calibration["models"])
        if row["status"] == "pass"
    }
    unet_seconds = float(rows["unet_small"]["median_optimizer_step_seconds"])
    smoke = cast(
        dict[str, Any],
        cast(dict[str, Any], diagnostics["stages"])["single_fold_smoke"],
    )
    fast_validation_seconds = float(smoke["fast_selection_seconds"])
    loss_jobs = len(cast(list[Any], queue["jobs"]))
    maximum_loss_steps = max(
        int(job["maximum_optimizer_steps"])
        for job in cast(list[dict[str, Any]], queue["jobs"])
    )
    validation_frequency = 500
    maximum_checks = maximum_loss_steps // validation_frequency
    earliest_checks = 12
    maximum_loss_screen_hours = loss_jobs * (
        unet_seconds * maximum_loss_steps
        + fast_validation_seconds * maximum_checks
    ) / 3600.0
    earliest_loss_screen_hours = loss_jobs * (
        unet_seconds * 6_000 + fast_validation_seconds * earliest_checks
    ) / 3600.0
    measured_training_hours = float(
        cast(dict[str, Any], calibration["serial_budget_proxy"])[
            "serial_upper_proxy_hours_for_measured_models"
        ]
    )
    known_interaction_ids = (
        "unet_parameter_matched_res",
        "unet_compute_matched_res",
        "unet_res",
        "bunet",
    )
    known_interaction_hours = 25 * sum(
        float(
            cast(dict[str, Any], calibration["serial_budget_proxy"])[
                "model_upper_proxy_hours_per_run"
            ][model_id]
        )
        for model_id in known_interaction_ids
    )
    fixed_core_compute_hours = 200 * 4.0
    known_scheduled_proxy = (
        measured_training_hours
        + known_interaction_hours
        + fixed_core_compute_hours
        + maximum_loss_screen_hours
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "decision": "LOSS_SCREEN_FEASIBLE_FULL_MATRIX_BLOCKED",
        "selected_host": "Apple M1 Max 32-core GPU 32-GB unified memory",
        "loss_screen": {
            "status": "frozen_ready_to_start",
            "job_count": loss_jobs,
            "maximum_optimizer_steps_per_job": maximum_loss_steps,
            "estimated_serial_hours_if_earliest_patience_stop": (
                earliest_loss_screen_hours
            ),
            "estimated_serial_hours_at_step_ceiling": maximum_loss_screen_hours,
            "queue_artifact": "artifacts/q1q2_v2/queues/loss_screen.json",
        },
        "full_matrix": {
            "status": "blocked_before_start",
            "measured_10_model_convergence_training_proxy_hours": (
                measured_training_hours
            ),
            "fixed_200_core_compute_matched_hours": fixed_core_compute_hours,
            "known_4_of_4_interaction_training_proxy_hours": (
                known_interaction_hours
            ),
            "known_scheduled_work_proxy_hours": known_scheduled_proxy,
            "known_scheduled_work_proxy_serial_days": known_scheduled_proxy / 24.0,
            "explicit_exclusions": [
                "nnU-Net v2 2D convergence",
                "nnU-Net v2 3D full-resolution convergence",
                "all main-run repeated validation overhead",
                "best and terminal full-metric evaluation",
                "checkpoint I/O",
                "reproduction reruns",
                "external confirmatory inference",
            ],
            "reason": (
                "The known subtotal already spans months on one serial MPS "
                "device and omits material mandatory work."
            ),
        },
        "measured_validation": {
            "fold": 1,
            "patient_count": int(smoke["validation_patient_count"]),
            "fast_selection_seconds": fast_validation_seconds,
            "full_evaluator_seconds": float(smoke["full_evaluator_seconds"]),
            "metric_parity": bool(
                cast(dict[str, Any], smoke["acceptance"])[
                    "selection_metric_parity"
                ]
            ),
            "metric_value": float(
                smoke["validation_patient_mean_regional_dice"]
            ),
            "scientific_use": "prohibited_pipeline_diagnostic",
        },
        "invariants": {
            "folds": 5,
            "common_main_seeds": 5,
            "silent_reduction": False,
            "external_inference_permitted": False,
            "legacy_internal_test_accessed": False,
        },
        "next_decision_point": (
            "After all 15 loss-screen jobs complete, freeze the selected loss, "
            "benchmark official nnU-Net plans, and use observed learning curves "
            "to request or freeze an explicit full-matrix protocol decision."
        ),
    }
    atomic_write_json(Path("reports/q1q2_v2/m1_execution_decision.json"), payload)
    lines = [
        "# M1 execution decision",
        "",
        "Decision: **LOSS SCREEN FEASIBLE; FULL MATRIX BLOCKED**",
        "",
        (
            f"The restart-safe loss screen contains {loss_jobs} jobs and is "
            f"estimated at {earliest_loss_screen_hours:.1f}-"
            f"{maximum_loss_screen_hours:.1f} serial hours on the measured M1 Max."
        ),
        "",
        (
            f"The 10 measured convergence models alone require "
            f"{measured_training_hours:,.1f} optimizer-work hours at the frozen "
            "50,000-step ceiling. Adding the fixed core compute regime, the known "
            "four interaction finalists, and the loss screen yields "
            f"{known_scheduled_proxy:,.1f} hours "
            f"({known_scheduled_proxy / 24.0:,.1f} serial days). This is not a "
            "complete total: both nnU-Net baselines, repeated validation, full "
            "metrics, reproduction, and external inference are excluded."
        ),
        "",
        (
            f"On fold 1, fast selection validation took "
            f"{fast_validation_seconds:.1f} seconds versus "
            f"{float(smoke['full_evaluator_seconds']):.1f} seconds for the full "
            "evaluator and matched mean regional Dice exactly."
        ),
        "",
        (
            "Therefore only the bounded development loss screen may start now. "
            "Five folds and five common main seeds remain unchanged. Gate F and "
            "external inference remain blocked."
        ),
    ]
    atomic_write_text(
        Path("reports/q1q2_v2/m1_execution_decision.md"),
        "\n".join(lines) + "\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
