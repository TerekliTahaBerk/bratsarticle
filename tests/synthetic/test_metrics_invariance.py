import json
import math
from typing import Any

import numpy as np
import pytest
import torch

from evaluation import CentralEvaluator, EvaluationConfig


def _assert_rows_equal(left: dict[str, Any], right: dict[str, Any]) -> None:
    assert left.keys() == right.keys()
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if isinstance(left_value, float) and isinstance(right_value, float):
            if math.isnan(left_value) and math.isnan(right_value):
                continue
            assert left_value == pytest.approx(right_value)
        else:
            assert left_value == right_value


def _batch() -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros((2, 8, 8, 8), dtype=np.int16)
    target[0, 2:6, 2:6, 2:6] = 2
    target[1, 3:5, 3:5, 3:5] = 4
    prediction = target.copy()
    prediction[1, 5, 5, 5] = 4
    return prediction, target


def test_batch_size_invariance() -> None:
    prediction, target = _batch()
    evaluator = CentralEvaluator(EvaluationConfig())
    together = evaluator.evaluate_batch(
        prediction,
        target,
        patient_ids=["first", "second"],
    )
    separately = [
        evaluator.evaluate_batch(
            prediction[index : index + 1],
            target[index : index + 1],
            patient_ids=[patient_id],
        )[0]
        for index, patient_id in enumerate(("first", "second"))
    ]

    for combined_row, separate_row in zip(together, separately, strict=True):
        _assert_rows_equal(combined_row, separate_row)


def test_numpy_and_cpu_tensor_consistency() -> None:
    prediction, target = _batch()
    evaluator = CentralEvaluator(EvaluationConfig())
    numpy_rows = evaluator.evaluate_batch(prediction, target)
    tensor_rows = evaluator.evaluate_batch(
        torch.as_tensor(prediction),
        torch.as_tensor(target),
    )

    for numpy_row, tensor_row in zip(numpy_rows, tensor_rows, strict=True):
        _assert_rows_equal(numpy_row, tensor_row)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_gpu_consistency() -> None:
    prediction, target = _batch()
    evaluator = CentralEvaluator(EvaluationConfig())
    cpu_rows = evaluator.evaluate_batch(prediction, target)
    gpu_rows = evaluator.evaluate_batch(
        torch.as_tensor(prediction, device="cuda"),
        torch.as_tensor(target, device="cuda"),
    )

    for cpu_row, gpu_row in zip(cpu_rows, gpu_rows, strict=True):
        _assert_rows_equal(cpu_row, gpu_row)


def test_deterministic_output() -> None:
    prediction, target = _batch()
    evaluator = CentralEvaluator(EvaluationConfig())

    first = evaluator.evaluate_batch(
        prediction,
        target,
        patient_ids=["first", "second"],
    )
    second = evaluator.evaluate_batch(
        prediction,
        target,
        patient_ids=["first", "second"],
    )

    assert json.dumps(first, sort_keys=True, allow_nan=True) == json.dumps(
        second,
        sort_keys=True,
        allow_nan=True,
    )
