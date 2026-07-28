import math

import numpy as np
import pytest

from evaluation import CentralEvaluator, EvaluationConfig


def test_one_to_many_lesion_split_uses_one_to_one_matching() -> None:
    target = np.zeros((1, 10, 10, 10), dtype=np.int16)
    target[:, 2:8, 2:8, 2:8] = 2
    prediction = np.zeros_like(target)
    prediction[:, 2:4, 2:8, 2:8] = 2
    prediction[:, 6:8, 2:8, 2:8] = 2
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]

    assert row["wt_target_lesion_count"] == 1
    assert row["wt_prediction_lesion_count"] == 2
    assert row["wt_matched_lesion_count"] == 1
    assert row["wt_lesion_recall"] == pytest.approx(1.0)
    assert row["wt_lesion_precision"] == pytest.approx(0.5)
    assert row["wt_false_positive_lesion_count"] == 1


def test_many_to_one_lesion_merge_leaves_one_false_negative() -> None:
    target = np.zeros((1, 12, 12, 12), dtype=np.int16)
    target[:, 2:4, 3:6, 3:6] = 2
    target[:, 8:10, 3:6, 3:6] = 2
    prediction = np.zeros_like(target)
    prediction[:, 2:10, 3:6, 3:6] = 2
    row = CentralEvaluator(EvaluationConfig()).evaluate_batch(
        prediction,
        target,
    )[0]

    assert row["wt_target_lesion_count"] == 2
    assert row["wt_prediction_lesion_count"] == 1
    assert row["wt_matched_lesion_count"] == 1
    assert row["wt_lesion_recall"] == pytest.approx(0.5)
    assert row["wt_lesion_precision"] == pytest.approx(1.0)
    assert row["wt_false_negative_lesion_count"] == 1
    assert math.isinf(row["wt_lesion_wise_hd95_mm"])
