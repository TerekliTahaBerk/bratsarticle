from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from bratsarticle.training.validation import validate_selection_dice
from evaluation import CentralEvaluator, EvaluationConfig


class EncodedPredictionModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = inputs[:, 0].long()
        return (
            torch.nn.functional.one_hot(indices, num_classes=4)
            .movedim(-1, 1)
            .float()
        )


def test_fast_selection_dice_matches_central_evaluator() -> None:
    prediction_indices = torch.tensor(
        [
            [[0, 1, 2], [3, 0, 1]],
            [[0, 0, 2], [3, 3, 1]],
        ]
    )
    target_labels = torch.tensor(
        [
            [[0, 1, 2], [4, 0, 2]],
            [[0, 0, 2], [4, 1, 1]],
        ]
    )
    images = torch.zeros((2, 4, 2, 3))
    images[:, 0] = prediction_indices
    batch = {
        "image": images,
        "label": target_labels,
        "subject_id": ["patient", "patient"],
        "slice_index": torch.tensor([0, 1]),
        "split": ["validation", "validation"],
    }
    config = EvaluationConfig(output_mode="labels")

    fast = validate_selection_dice(
        EncodedPredictionModel(),
        [batch],
        device=torch.device("cpu"),
        config=config,
    )

    channel_to_label = np.asarray((0, 1, 2, 4), dtype=np.int16)
    prediction_volume = channel_to_label[
        np.moveaxis(prediction_indices.numpy(), 0, 2)
    ]
    target_volume = np.moveaxis(target_labels.numpy(), 0, 2)
    central = CentralEvaluator(config).evaluate_batch(
        prediction_volume,
        target_volume,
        patient_ids=["patient"],
        spacings_mm=[(1.0, 1.0, 1.0)],
    )[0]
    fast_row = fast.patient_rows[0]

    assert fast_row["wt_dice"] == pytest.approx(central["wt_dice"])
    assert fast_row["tc_dice"] == pytest.approx(central["tc_dice"])
    assert fast_row["et_dice"] == pytest.approx(central["et_dice"])
    assert fast.mean_regional_dice == pytest.approx(central["mean_regional_dice"])
    assert fast.validation_loss is None
