from pathlib import Path

import pytest

from bratsarticle.experiments.gate11_runner import (
    load_gate11_plan,
    run_gate11,
)


def test_gate11_plan_requires_one_opening_and_thirteen_checkpoints() -> None:
    plan = load_gate11_plan(Path("configs/internal_test/gate11.yaml"))
    assert plan["inference"]["expected_patients"] == 74
    assert plan["inference"]["expected_checkpoints"] == 13
    assert plan["access"]["maximum_manifest_open_events"] == 1
    assert plan["qualitative"]["frozen_seed"] == 20260729


def test_gate11_runner_requires_explicit_test_permission() -> None:
    with pytest.raises(PermissionError, match="allow-test-evaluation"):
        run_gate11(
            Path("configs/internal_test/gate11.yaml"),
            allow_test_evaluation=False,
        )
