from __future__ import annotations

from pathlib import Path

import yaml

from bratsarticle.experiments.q1q2_nnunet_queue import (
    load_nnunet_runner_config,
    official_output_directory,
)


def test_nnunet_runner_prohibits_external_and_legacy_access() -> None:
    config = load_nnunet_runner_config(
        Path("configs/q1q2_v2/nnunet_m1_runner.yaml")
    )

    assert config["hardware"]["backend"] == "mps"
    assert config["guards"]["allow_external_data"] is False
    assert config["guards"]["allow_legacy_internal_test"] is False
    assert config["matrix"]["expected_jobs"] == 50
    assert config["matrix"]["expected_seeds"] == [
        20260730,
        20260731,
        20260732,
        20260733,
        20260734,
    ]
    assert config["resource_profile_protocol"] == (
        "configs/q1q2_v2/resource_profile_protocol.yaml"
    )


def test_official_output_directory_is_exact_and_does_not_glob() -> None:
    job = {
        "trainer": "nnUNetTrainerSeed20260730",
        "plans_identifier": "nnUNetPlans",
        "configuration": "2d",
        "fold_nnunet_zero_indexed": 3,
    }

    output = official_output_directory(
        Path("/tmp/results"),
        dataset_name="Dataset501_BraTS2020Q1Q2",
        job=job,
    )

    assert output == Path(
        "/tmp/results/Dataset501_BraTS2020Q1Q2/"
        "nnUNetTrainerSeed20260730__nnUNetPlans__2d/fold_3"
    )


def test_nnunet_runner_requires_plan_freeze_before_status_change() -> None:
    config_path = Path("configs/q1q2_v2/nnunet_m1_runner.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert config["status"] == "blocked_until_hardware_plan_freeze"
    assert config["matrix"]["selected_3d_plan"].endswith(
        "selected_nnunet_3d_plan.yaml"
    )
