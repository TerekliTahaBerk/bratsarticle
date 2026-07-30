from __future__ import annotations

import torch

from bratsarticle.experiments.q1q2_diagnostics import _synthetic_batch


def test_synthetic_batch_contains_all_brats_labels_and_modalities() -> None:
    image, label = _synthetic_batch(
        batch_size=2,
        image_size=64,
        device=torch.device("cpu"),
    )

    assert image.shape == (2, 4, 64, 64)
    assert label.shape == (2, 64, 64)
    assert set(torch.unique(label).tolist()) == {0, 1, 2, 4}
    assert torch.isfinite(image).all()
