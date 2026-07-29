from __future__ import annotations

import numpy as np

from bratsarticle.data.dataset import extract_context_slices
from bratsarticle.data.preprocessing import (
    IntensityTransformPlan,
    apply_intensity_transform,
)


def test_context_slice_order_and_boundary_replication() -> None:
    image = np.zeros((4, 2, 2, 3), dtype=np.float32)
    for modality in range(4):
        for slice_index in range(3):
            image[modality, :, :, slice_index] = 100 * modality + slice_index

    context = extract_context_slices(
        image,
        0,
        slice_axis=2,
        context_offsets=(-2, -1, 0, 1, 2),
    )

    assert context.shape == (20, 2, 2)
    assert context[:5, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]
    assert context[5:10, 0, 0].tolist() == [
        100.0,
        100.0,
        100.0,
        101.0,
        102.0,
    ]


def test_intensity_transform_reuses_one_plan_across_context_slices() -> None:
    image = np.ones((20, 3, 3), dtype=np.float32)
    plan = IntensityTransformPlan(
        scales=(2.0, 3.0, 4.0, 5.0),
        shifts=(0.1, 0.2, 0.3, 0.4),
    )

    transformed = apply_intensity_transform(image, plan)

    assert np.allclose(transformed[0:5], 2.1)
    assert np.allclose(transformed[5:10], 3.2)
    assert np.allclose(transformed[10:15], 4.3)
    assert np.allclose(transformed[15:20], 5.4)
