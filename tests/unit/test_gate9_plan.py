from pathlib import Path

from bratsarticle.experiments.gate9 import (
    extension_arms_for_finalists,
    stage_arms,
)
from bratsarticle.experiments.pilots import load_pilot_plan


def test_gate9_plan_predeclares_confirmation_and_extensions() -> None:
    plan = load_pilot_plan(Path("configs/pilots/gate9.yaml"))
    confirmation = stage_arms(plan, "confirmation")
    extensions = stage_arms(plan, "finalist_extension")

    assert plan.gate == 9
    assert len(plan.arms) == 20
    assert len(confirmation) == 12
    assert len(extensions) == 8
    assert {arm.seed for arm in confirmation} == {
        20260729,
        20260730,
        20260731,
    }
    assert {arm.candidate_id for arm in confirmation} == {
        "unet_reference",
        "bunet",
        "unet_res",
        "unet_wc",
    }
    selected = extension_arms_for_finalists(plan, ["bunet", "unet_res"])
    assert len(selected) == 4
    assert {arm.seed for arm in selected} == {20260732, 20260733}
    assert {arm.candidate_id for arm in selected} == {"bunet", "unet_res"}
