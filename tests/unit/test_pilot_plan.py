from pathlib import Path

from bratsarticle.experiments.pilots import load_pilot_plan, pilot_preflight


def test_gate8_plan_is_single_seed_and_non_factorial() -> None:
    plan = load_pilot_plan(Path("configs/pilots/gate8.yaml"))
    pairs = {(arm.model_config_path, arm.loss_name) for arm in plan.arms}

    assert len(plan.arms) == 12
    assert len(pairs) == 12
    assert {arm.seed for arm in plan.arms} == {20260729}
    assert {arm.screen for arm in plan.arms} == {"architecture", "loss"}
    assert plan.maximum_optimizer_steps == 2000
    assert plan.maximum_gpu_hours == 0.5
    assert plan.elimination["statistical_unit"] == "patient"
    assert not plan.elimination["internal_test_permitted"]


def test_gate8_preflight_refuses_current_non_cuda_host() -> None:
    plan = load_pilot_plan(Path("configs/pilots/gate8.yaml"))

    preflight = pilot_preflight(plan)

    assert not preflight["eligible"]
    assert preflight["checks"]["test_manifest_not_referenced_by_plan"]
    assert preflight["checks"]["pilot_budget_within_compute_protocol"]
