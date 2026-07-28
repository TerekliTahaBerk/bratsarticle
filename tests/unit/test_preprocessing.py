from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bratsarticle.data.preprocessing import (
    IntensityTransformPlan,
    PreprocessingConfig,
    SpatialTransformPlan,
    TrainingSamplingConfig,
    apply_intensity_transform,
    apply_spatial_transform,
    load_preprocessing_config,
    preprocess_modalities,
    select_training_slice,
)


def _modalities() -> dict[str, np.ndarray]:
    base = np.zeros((4, 5, 6), dtype=np.float32)
    base[1:4, 1:5, 1:6] = np.arange(60, dtype=np.float32).reshape(3, 4, 5) + 1
    return {
        "t1": base,
        "t1ce": np.flip(base, axis=0).copy(),
        "t2": np.flip(base, axis=1).copy(),
        "flair": np.flip(base, axis=2).copy(),
    }


def test_modality_order_and_nonzero_patient_zscore() -> None:
    modalities = _modalities()
    output = preprocess_modalities(modalities, PreprocessingConfig())

    assert output.shape == (4, 4, 5, 6)
    for channel, modality in enumerate(("t1", "t1ce", "t2", "flair")):
        original_foreground = modalities[modality] != 0
        assert float(output[channel][original_foreground].mean()) == pytest.approx(
            0.0,
            abs=1e-6,
        )
        assert float(output[channel][original_foreground].std()) == pytest.approx(
            1.0,
            abs=1e-6,
        )
        assert np.all(output[channel][~original_foreground] == 0)

    with pytest.raises(ValueError, match="Expected modalities"):
        preprocess_modalities(
            {key: value for key, value in modalities.items() if key != "flair"},
            PreprocessingConfig(),
        )


def test_spatial_augmentation_is_synchronized() -> None:
    label = np.arange(30, dtype=np.int16).reshape(5, 6)
    image = np.stack([label.astype(np.float32)] * 4)
    plan = SpatialTransformPlan(
        flip_axis0=True,
        flip_axis1=False,
        rotation_quarters=1,
    )
    transformed_image, transformed_label = apply_spatial_transform(
        image,
        label,
        plan,
    )

    assert np.array_equal(transformed_image[0], transformed_label)
    assert np.array_equal(transformed_image[3], transformed_label)


def test_intensity_augmentation_is_modality_specific_and_mask_safe() -> None:
    image = np.ones((4, 3, 3), dtype=np.float32)
    image[:, 0, 0] = 0
    transformed = apply_intensity_transform(
        image,
        IntensityTransformPlan(
            scales=(1.0, 2.0, 3.0, 4.0),
            shifts=(0.0, 0.0, 0.0, 0.0),
        ),
    )

    assert [float(transformed[index, 1, 1]) for index in range(4)] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert np.all(transformed[:, 0, 0] == 0)


def test_tumor_non_tumor_sampling_ratio_is_configurable() -> None:
    label = np.zeros((5, 6, 7), dtype=np.int16)
    label[:, :, 3] = 2
    tumor_only = replace(
        PreprocessingConfig(),
        training_sampling=TrainingSamplingConfig(tumor_probability=1.0),
    )
    non_tumor_only = replace(
        PreprocessingConfig(),
        training_sampling=TrainingSamplingConfig(tumor_probability=0.0),
    )

    assert select_training_slice(label, np.random.default_rng(1), tumor_only) == 3
    assert select_training_slice(label, np.random.default_rng(1), non_tumor_only) != 3


def test_preprocessing_config_retains_all_evaluation_slices() -> None:
    config = load_preprocessing_config(Path("configs/data/preprocessing.yaml"))

    assert config.modality_order == ("t1", "t1ce", "t2", "flair")
    assert config.validation_include_all_slices
    assert config.validation_include_empty_slices
    assert config.validation_deterministic
    assert config.test_include_all_slices
    assert config.test_include_empty_slices
    assert config.test_deterministic
    assert not config.clipping.enabled
