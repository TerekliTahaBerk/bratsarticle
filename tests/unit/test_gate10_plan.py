from pathlib import Path

import pandas as pd
import pytest
import yaml

from bratsarticle.experiments.gate10 import (
    development_tumor_thresholds,
    load_gate10_plan,
)


def test_gate10_plan_freezes_patient_level_multiseed_analysis() -> None:
    plan = load_gate10_plan(Path("configs/statistics/gate10.yaml"))
    candidates = plan["candidates"]
    assert candidates["primary"] == "bunet"
    assert candidates["mandatory_reference"] == "unet_reference"
    assert candidates["ordered_internal_test_candidates"] == [
        "unet_reference",
        "bunet",
        "unet_res",
    ]
    assert candidates["seed_ensemble"] is False
    assert candidates["evaluate_every_predeclared_seed"] is True
    assert plan["endpoints"]["statistical_unit"] == "patient"
    assert plan["hypothesis_testing"]["multiplicity"]["correction"] == "holm"
    assert plan["conduct"]["no_model_selection_after_test_access"] is True
    assert plan["conduct"]["clinical_applicability_claims_permitted"] is False


def test_gate10_plan_rejects_test_permission(tmp_path: Path) -> None:
    source = yaml.safe_load(
        Path("configs/statistics/gate10.yaml").read_text(encoding="utf-8")
    )
    source["gate10"]["internal_test_permitted"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot permit"):
        load_gate10_plan(path)


def test_tumor_thresholds_require_train_only(tmp_path: Path) -> None:
    valid = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "split": ["train"] * 6,
            "wt_volume_mm3": [1.0, 2.0, 3.0, 6.0, 7.0, 8.0],
        }
    ).to_csv(valid, index=False)
    thresholds = development_tumor_thresholds(
        valid,
        [1.0 / 3.0, 2.0 / 3.0],
        column="wt_volume_mm3",
        method="linear",
    )
    assert thresholds == {"q1_mm3": 2.6666666666666665, "q2_mm3": 6.333333333333333}

    invalid = tmp_path / "mixed.csv"
    pd.DataFrame(
        {
            "split": ["train", "test"],
            "wt_volume_mm3": [1.0, 100.0],
        }
    ).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="train only"):
        development_tumor_thresholds(
            invalid,
            [1.0 / 3.0, 2.0 / 3.0],
            column="wt_volume_mm3",
            method="linear",
        )
