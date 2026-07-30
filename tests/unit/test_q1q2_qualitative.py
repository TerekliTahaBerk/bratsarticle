import numpy as np
import pandas as pd

from bratsarticle.analysis.q1q2_qualitative import (
    pairwise_model_disagreement,
    select_qualitative_cases,
)


def test_pairwise_disagreement_is_zero_for_identical_predictions() -> None:
    target = np.zeros((3, 3, 3), dtype=np.uint8)
    target[1, 1, 1] = 4
    prediction = target.copy()

    value = pairwise_model_disagreement(
        [prediction, prediction.copy(), prediction.copy()],
        target,
    )

    assert value == 0.0


def test_pairwise_disagreement_retains_region_specific_differences() -> None:
    target = np.zeros((2, 2, 2), dtype=np.uint8)
    target[0, 0, 0] = 4
    first = target.copy()
    second = np.zeros_like(target)

    value = pairwise_model_disagreement([first, second], target)

    assert value == 1.0


def test_qualitative_selection_applies_frozen_tie_breakers() -> None:
    summary = pd.DataFrame(
        {
            "patient_id": ["b", "a", "c"],
            "patient_mean_regional_dice": [0.8, 0.8, 0.2],
            "patient_mean_finite_et_lesion_wise_dice": [0.4, 0.4, 0.1],
            "patient_mean_false_positive_lesion_burden": [1.0, 1.0, 5.0],
            "patient_largest_regional_hd95_mm": [2.0, 2.0, np.inf],
            "whole_tumor_reference_volume_mm3": [10.0, 20.0, 30.0],
            "pairwise_model_disagreement": [0.1, 0.1, 0.9],
        }
    )

    selected = select_qualitative_cases(summary)

    assert selected["highest_patient_mean_regional_dice"]["patient_id"] == "a"
    assert selected["lowest_et_lesion_wise_dice"]["patient_id"] == "c"
    assert selected["largest_pairwise_model_disagreement"]["patient_id"] == "c"
