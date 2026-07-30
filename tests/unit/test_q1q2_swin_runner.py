from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from bratsarticle.data.preprocessing import PreprocessingConfig
from bratsarticle.experiments.q1q2_swin_runner import (
    SWIN_MODEL_ID,
    _sample_patch,
    _selection_dice,
    load_swin_runner_config,
    swin_convergence_specs,
)
from bratsarticle.utils.hashing import file_digest


def _selected_loss(tmp_path: Path) -> Path:
    evidence = tmp_path / "selection.json"
    evidence.write_text(
        json.dumps(
            {
                "status": "selected_from_complete_development_cv",
                "selected_loss": "cross_entropy_plus_soft_dice",
                "external_data_accessed": False,
                "legacy_internal_test_accessed": False,
            }
        ),
        encoding="utf-8",
    )
    selected = tmp_path / "selected_loss.yaml"
    selected.write_text(
        yaml.safe_dump(
            {
                "status": "frozen_from_complete_development_cv",
                "selected_loss": "cross_entropy_plus_soft_dice",
                "selection_artifact": evidence.as_posix(),
                "selection_artifact_sha256": file_digest(evidence),
                "external_data_used_for_selection": False,
                "legacy_internal_test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )
    return selected


def test_swin_matrix_has_all_five_folds_and_equal_five_seeds(
    tmp_path: Path,
) -> None:
    config = Path("configs/q1q2_v2/swin_m1_runner.yaml")
    specs = swin_convergence_specs(config, _selected_loss(tmp_path))

    assert len(specs) == 25
    assert {spec.model_id for spec in specs} == {SWIN_MODEL_ID}
    assert {spec.fold for spec in specs} == {1, 2, 3, 4, 5}
    assert {spec.seed for spec in specs} == {
        20260730,
        20260731,
        20260732,
        20260733,
        20260734,
    }
    assert len({spec.run_id for spec in specs}) == 25
    assert len({spec.sha256 for spec in specs}) == 25


def test_swin_config_freezes_mps_and_prohibits_test_access() -> None:
    config = load_swin_runner_config(Path("configs/q1q2_v2/swin_m1_runner.yaml"))

    assert config["hardware"]["backend"] == "mps"
    assert config["hardware"]["deterministic_algorithms"] == (
        "warn_only_with_repeat_tolerance_audit"
    )
    assert config["guards"]["allow_external_data"] is False
    assert config["guards"]["allow_legacy_internal_test"] is False
    assert config["training"]["effective_batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 2
    assert config["repeat_tolerance"]["comparison"] == {
        "absolute_tolerance": 0.00001,
        "relative_tolerance": 0.00001,
        "loss_absolute_tolerance": 0.000001,
    }


def test_swin_specs_reject_selection_that_used_external_data(
    tmp_path: Path,
) -> None:
    selected = _selected_loss(tmp_path)
    payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
    payload["external_data_used_for_selection"] = True
    selected.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(PermissionError, match="external"):
        swin_convergence_specs(
            Path("configs/q1q2_v2/swin_m1_runner.yaml"),
            selected,
        )


def test_swin_patch_sampling_is_seed_deterministic_and_preserves_labels() -> None:
    image = np.zeros((4, 40, 40, 40), dtype=np.float32)
    label = np.zeros((40, 40, 40), dtype=np.int16)
    label[15:25, 15:25, 15:25] = 2
    label[17:23, 17:23, 17:23] = 1
    label[19:21, 19:21, 19:21] = 4
    image[0] = label != 0
    image[1] = (label == 1) | (label == 4)
    image[2] = label == 2
    image[3] = label == 4
    config = PreprocessingConfig()

    first = _sample_patch(
        image,
        label,
        patch_size=(32, 32, 32),
        generator=np.random.default_rng(20260730),
        tumor_probability=1.0,
        preprocessing=config,
    )
    second = _sample_patch(
        image,
        label,
        patch_size=(32, 32, 32),
        generator=np.random.default_rng(20260730),
        tumor_probability=1.0,
        preprocessing=config,
    )

    assert first[0].shape == (4, 32, 32, 32)
    assert first[1].shape == (32, 32, 32)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert set(np.unique(first[1])) == {0, 1, 2, 4}


def test_swin_selection_dice_obeys_brats_regions_and_empty_rule() -> None:
    target = np.zeros((8, 8, 8), dtype=np.int16)
    target[1:7, 1:7, 1:7] = 2
    target[2:6, 2:6, 2:6] = 1
    prediction = target.copy()

    wt, tc, et = _selection_dice(
        prediction,
        target,
        both_empty=1.0,
        one_empty=0.0,
    )

    assert wt == 1.0
    assert tc == 1.0
    assert et == 1.0

    target[3:5, 3:5, 3:5] = 4
    prediction = target.copy()
    prediction[target == 4] = 0
    _, _, et_missing = _selection_dice(
        prediction,
        target,
        both_empty=1.0,
        one_empty=0.0,
    )
    assert et_missing == 0.0
