from __future__ import annotations

from pathlib import Path

import pytest

from bratsarticle.experiments.q1q2_native_runner import (
    loss_screen_specs,
    resolve_loss_screen_spec,
)

RUNNER_CONFIG = Path("configs/q1q2_v2/m1_native_runner.yaml")


def test_loss_screen_expands_to_frozen_unique_matrix() -> None:
    specs = loss_screen_specs(RUNNER_CONFIG)

    assert len(specs) == 15
    assert len({spec.run_id for spec in specs}) == 15
    assert {spec.fold for spec in specs} == {1, 2, 3, 4, 5}
    assert {spec.seed for spec in specs} == {20260730}
    assert all(spec.maximum_optimizer_steps == 10_000 for spec in specs)


def test_resolver_rejects_unfrozen_seed() -> None:
    with pytest.raises(PermissionError, match="outside the frozen"):
        resolve_loss_screen_spec(
            RUNNER_CONFIG,
            model_id="unet_small",
            fold=1,
            seed=999,
            loss_name="cross_entropy_plus_soft_dice",
        )
