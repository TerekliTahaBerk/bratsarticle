from __future__ import annotations

import pytest

from bratsarticle.experiments.m1_calibration import estimate_serial_budget


def test_estimate_serial_budget_uses_successful_rows_only() -> None:
    result = estimate_serial_budget(
        [
            {
                "status": "pass",
                "model_id": "a",
                "median_optimizer_step_seconds": 0.5,
            },
            {
                "status": "fail",
                "model_id": "b",
            },
        ],
        runs_per_model=25,
        maximum_optimizer_steps=50_000,
    )

    assert result["measured_model_count"] == 1
    assert result["model_upper_proxy_hours_per_run"]["a"] == pytest.approx(
        6.9444444444
    )
    assert result["serial_upper_proxy_hours_for_measured_models"] == pytest.approx(
        173.6111111111
    )
