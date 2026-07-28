from pathlib import Path

import pytest

from bratsarticle.experiments.pilot_runner import run_pilot_arm


def test_pilot_runner_requires_explicit_permission() -> None:
    with pytest.raises(PermissionError, match="allow-pilot-training"):
        run_pilot_arm(
            plan_path=Path("configs/pilots/gate8.yaml"),
            arm_id="architecture_unet",
            allow_pilot_training=False,
        )


def test_pilot_runner_rejects_ineligible_host_before_data_access() -> None:
    with pytest.raises(RuntimeError, match="preflight failed"):
        run_pilot_arm(
            plan_path=Path("configs/pilots/gate8.yaml"),
            arm_id="architecture_unet",
            allow_pilot_training=True,
        )
