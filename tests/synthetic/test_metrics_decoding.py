from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from evaluation import CentralEvaluator, EvaluationConfig, load_evaluation_config
from evaluation.config import PostprocessingConfig
from evaluation.regions import decode_prediction


def test_nested_region_violation_is_visible_and_optional() -> None:
    probabilities = np.zeros((1, 3, 5, 5, 5), dtype=np.float32)
    probabilities[:, 2, 2, 2, 2] = 1.0
    raw_config = EvaluationConfig(
        output_mode="nested_sigmoid",
        from_logits=False,
        enforce_nested_consistency=False,
    )
    raw = decode_prediction(probabilities, raw_config)

    assert raw.nested_violation_voxels == (1,)
    assert raw.regions["et"][0, 2, 2, 2]
    assert not raw.regions["tc"][0, 2, 2, 2]
    assert not raw.regions["wt"][0, 2, 2, 2]

    corrected = decode_prediction(
        probabilities,
        replace(raw_config, enforce_nested_consistency=True),
    )
    assert corrected.nested_violation_voxels == (1,)
    assert corrected.regions["et"][0, 2, 2, 2]
    assert corrected.regions["tc"][0, 2, 2, 2]
    assert corrected.regions["wt"][0, 2, 2, 2]


def test_four_class_softmax_channel_mapping() -> None:
    logits = np.zeros((1, 4, 4, 4, 4), dtype=np.float32)
    logits[:, 0] = 1.0
    logits[:, 3, 1, 1, 1] = 5.0
    decoded = decode_prediction(
        logits,
        EvaluationConfig(output_mode="softmax"),
    )

    assert decoded.regions["et"][0, 1, 1, 1]
    assert decoded.regions["tc"][0, 1, 1, 1]
    assert decoded.regions["wt"][0, 1, 1, 1]


def test_raw_and_filtered_postprocessing_are_separate_rows() -> None:
    target = np.zeros((1, 6, 6, 6), dtype=np.int16)
    prediction = target.copy()
    prediction[:, 2, 2, 2] = 4
    config = EvaluationConfig(
        postprocessing=PostprocessingConfig(
            stages=("raw", "filtered"),
            minimum_prediction_voxels=2,
            minimum_prediction_volume_mm3=0.0,
        )
    )
    rows = CentralEvaluator(config).evaluate_batch(prediction, target)

    assert [row["evaluation_stage"] for row in rows] == ["raw", "filtered"]
    assert rows[0]["et_dice"] == pytest.approx(0.0)
    assert rows[0]["et_false_positive_lesion_count"] == 1
    assert rows[1]["et_dice"] == pytest.approx(1.0)
    assert rows[1]["et_false_positive_lesion_count"] == 0


def test_default_evaluator_config_is_valid_and_explicit() -> None:
    config = load_evaluation_config(Path("configs/evaluation/default.yaml"))

    assert config.output_mode == "labels"
    assert config.lesions.connectivity == 26
    assert config.lesions.minimum_voxels == 1
    assert config.lesions.matching_method == "maximum_total_iou"
    assert config.postprocessing.stages == ("raw",)
