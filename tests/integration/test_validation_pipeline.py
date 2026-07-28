import numpy as np
import pytest
import torch
from torch import nn

from bratsarticle.training.validation import validate_full_volumes
from evaluation import CentralEvaluator, EvaluationConfig


class EncodedLabelModel(nn.Module):
    """Decode contiguous class IDs stored in the first image channel."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        indices = inputs[:, 0].long()
        logits = torch.full(
            (inputs.shape[0], 4, *inputs.shape[-2:]),
            -10.0,
            device=inputs.device,
        )
        return logits.scatter_(1, indices.unsqueeze(1), 10.0)


def _validation_batch(split: str = "validation") -> dict[str, object]:
    indices = torch.zeros((2, 4, 5), dtype=torch.long)
    indices[0, 1:3, 1:4] = 2
    indices[1, 1:3, 1:4] = 1
    indices[1, 2, 2] = 3
    labels = torch.where(indices == 3, torch.full_like(indices, 4), indices)
    images = torch.zeros((2, 4, 4, 5), dtype=torch.float32)
    images[:, 0] = indices.float()
    return {
        "image": images,
        "label": labels,
        "subject_id": ["synthetic_patient", "synthetic_patient"],
        "slice_index": torch.tensor([0, 1]),
        "slice_axis": torch.tensor([2, 2]),
        "spacing_mm": torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        "split": [split, split],
    }


def test_validation_predictions_use_central_evaluator() -> None:
    rows = validate_full_volumes(
        EncodedLabelModel(),
        [_validation_batch()],
        device=torch.device("cpu"),
        evaluator=CentralEvaluator(EvaluationConfig(output_mode="labels")),
    )

    assert len(rows) == 1
    assert rows[0]["patient_id"] == "synthetic_patient"
    assert rows[0]["mean_regional_dice"] == pytest.approx(1.0)
    assert rows[0]["wt_hd95_mm"] == pytest.approx(0.0)
    assert np.isfinite(rows[0]["et_surface_dice"])


def test_validation_pipeline_rejects_test_batches() -> None:
    with pytest.raises(ValueError, match="requires validation"):
        validate_full_volumes(
            EncodedLabelModel(),
            [_validation_batch(split="test")],
            device=torch.device("cpu"),
            evaluator=CentralEvaluator(EvaluationConfig(output_mode="labels")),
        )
