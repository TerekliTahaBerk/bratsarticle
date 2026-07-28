import math

import numpy as np
import pytest

from evaluation import CentralEvaluator, EvaluationConfig


def _perfect_labels() -> np.ndarray:
    labels = np.zeros((1, 9, 10, 11), dtype=np.int16)
    labels[:, 2:7, 2:8, 2:9] = 2
    labels[:, 3:6, 3:7, 3:8] = 1
    labels[:, 4:6, 4:6, 4:7] = 4
    return labels


def test_perfect_prediction() -> None:
    labels = _perfect_labels()
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        labels,
        labels,
        patient_ids=["perfect"],
    )[0]

    assert row["mean_regional_dice"] == pytest.approx(1.0)
    for region in ("wt", "tc", "et"):
        assert row[f"{region}_dice"] == pytest.approx(1.0)
        assert row[f"{region}_iou"] == pytest.approx(1.0)
        assert row[f"{region}_hd95_mm"] == pytest.approx(0.0)
        assert row[f"{region}_surface_dice"] == pytest.approx(1.0)
        assert row[f"{region}_lesion_recall"] == pytest.approx(1.0)
        assert row[f"{region}_lesion_precision"] == pytest.approx(1.0)


def test_completely_empty_prediction_has_infinite_hd95() -> None:
    target = _perfect_labels()
    prediction = np.zeros_like(target)
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]

    assert row["mean_regional_dice"] == pytest.approx(0.0)
    assert row["wt_dice"] == pytest.approx(0.0)
    assert math.isinf(row["wt_hd95_mm"])
    assert row["wt_lesion_recall"] == pytest.approx(0.0)
    assert row["wt_false_negative_lesion_count"] == 1


def test_empty_et_ground_truth_rule_is_explicit() -> None:
    target = np.zeros((1, 7, 7, 7), dtype=np.int16)
    target[:, 2:5, 2:5, 2:5] = 1
    prediction = target.copy()
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]

    assert row["et_dice"] == pytest.approx(1.0)
    assert row["et_iou"] == pytest.approx(1.0)
    assert row["et_hd95_mm"] == pytest.approx(0.0)
    assert row["et_surface_dice"] == pytest.approx(1.0)
    assert math.isnan(row["et_sensitivity"])
    assert math.isnan(row["et_lesion_recall"])
    assert math.isnan(row["et_lesion_precision"])


def test_false_positive_et_lesion_is_counted() -> None:
    target = np.zeros((1, 7, 7, 7), dtype=np.int16)
    prediction = target.copy()
    prediction[:, 2:4, 2:4, 2:4] = 4
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]

    assert row["et_dice"] == pytest.approx(0.0)
    assert math.isinf(row["et_hd95_mm"])
    assert math.isnan(row["et_lesion_recall"])
    assert row["et_lesion_precision"] == pytest.approx(0.0)
    assert row["et_false_positive_lesion_count"] == 1


def test_dice_iou_mathematical_consistency() -> None:
    target = np.zeros((1, 8, 8, 8), dtype=np.int16)
    prediction = target.copy()
    target[:, 1:5, 1:5, 1:5] = 2
    prediction[:, 3:7, 1:5, 1:5] = 2
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]
    iou = row["wt_iou"]

    assert row["wt_dice"] == pytest.approx(2.0 * iou / (1.0 + iou))


def test_axis_order_shape_error_is_rejected() -> None:
    target = np.zeros((1, 4, 5, 6), dtype=np.int16)
    prediction = np.transpose(target, (0, 3, 2, 1))

    with pytest.raises(ValueError, match="shapes must match"):
        CentralEvaluator(EvaluationConfig()).evaluate_batch(prediction, target)
