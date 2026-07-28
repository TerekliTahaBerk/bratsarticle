"""Leakage-safe multimodal BraTS preprocessing and augmentation primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

MODALITY_ORDER: tuple[str, ...] = ("t1", "t1ce", "t2", "flair")


@dataclass(frozen=True)
class ClippingConfig:
    """Fixed intensity clipping rules derived from development data only."""

    enabled: bool = False
    provenance: Literal["development_train_only_fixed_bounds"] = (
        "development_train_only_fixed_bounds"
    )
    fixed_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingSamplingConfig:
    """Tumor/non-tumor slice sampling policy."""

    tumor_probability: float = 0.67
    tumor_minimum_voxels_per_slice: int = 1
    samples_per_patient_per_epoch: int = 16


@dataclass(frozen=True)
class SpatialAugmentationConfig:
    """Synchronized discrete spatial augmentation settings."""

    enabled: bool = True
    flip_probability: float = 0.5
    rotate_90: bool = True


@dataclass(frozen=True)
class IntensityAugmentationConfig:
    """Independent per-modality affine intensity augmentation."""

    enabled: bool = True
    apply_probability_per_modality: float = 0.5
    scale_range: tuple[float, float] = (0.9, 1.1)
    shift_range: tuple[float, float] = (-0.1, 0.1)


@dataclass(frozen=True)
class CacheConfig:
    """Optional normalized-volume cache stored outside raw-data roots."""

    enabled: bool = False
    root: Path | None = None
    memory_subjects: int = 1


@dataclass(frozen=True)
class PreprocessingConfig:
    """Complete preprocessing policy for 2D slice models."""

    modality_order: tuple[str, ...] = MODALITY_ORDER
    slice_axis: Literal[0, 1, 2] = 2
    normalization: Literal["nonzero_patient_modality_zscore"] = (
        "nonzero_patient_modality_zscore"
    )
    clipping: ClippingConfig = ClippingConfig()
    training_sampling: TrainingSamplingConfig = TrainingSamplingConfig()
    spatial_augmentation: SpatialAugmentationConfig = SpatialAugmentationConfig()
    intensity_augmentation: IntensityAugmentationConfig = IntensityAugmentationConfig()
    validation_include_all_slices: bool = True
    validation_include_empty_slices: bool = True
    validation_deterministic: bool = True
    test_include_all_slices: bool = True
    test_include_empty_slices: bool = True
    test_deterministic: bool = True
    cache: CacheConfig = CacheConfig()

    def __post_init__(self) -> None:
        """Reject policies that could change channels or drop evaluation slices."""
        if self.modality_order != MODALITY_ORDER:
            raise ValueError(f"Modality order must be exactly {MODALITY_ORDER}")
        if self.slice_axis not in {0, 1, 2}:
            raise ValueError("slice_axis must be 0, 1, or 2")
        if self.normalization != "nonzero_patient_modality_zscore":
            raise ValueError("Unsupported normalization policy")
        if not 0.0 <= self.training_sampling.tumor_probability <= 1.0:
            raise ValueError("Tumor sampling probability must be in [0, 1]")
        if self.training_sampling.tumor_minimum_voxels_per_slice < 1:
            raise ValueError("Tumor slice minimum must be positive")
        if self.training_sampling.samples_per_patient_per_epoch < 1:
            raise ValueError("Training samples per patient must be positive")
        if not 0.0 <= self.spatial_augmentation.flip_probability <= 1.0:
            raise ValueError("Flip probability must be in [0, 1]")
        intensity = self.intensity_augmentation
        if not 0.0 <= intensity.apply_probability_per_modality <= 1.0:
            raise ValueError("Intensity augmentation probability must be in [0, 1]")
        if intensity.scale_range[0] <= 0 or (
            intensity.scale_range[0] > intensity.scale_range[1]
        ):
            raise ValueError("Invalid intensity scale range")
        if intensity.shift_range[0] > intensity.shift_range[1]:
            raise ValueError("Invalid intensity shift range")
        if self.clipping.provenance != "development_train_only_fixed_bounds":
            raise ValueError("Clipping provenance must be development-only")
        if self.clipping.enabled:
            if set(self.clipping.fixed_bounds) != set(MODALITY_ORDER):
                raise ValueError(
                    "Enabled clipping requires fixed bounds for all modalities"
                )
            for modality, bounds in self.clipping.fixed_bounds.items():
                if bounds[0] >= bounds[1]:
                    raise ValueError(f"Invalid clipping bounds for {modality}")
        if not (
            self.validation_include_all_slices
            and self.validation_include_empty_slices
            and self.validation_deterministic
        ):
            raise ValueError("Validation must retain all slices deterministically")
        if not (
            self.test_include_all_slices
            and self.test_include_empty_slices
            and self.test_deterministic
        ):
            raise ValueError("Test must retain all slices deterministically")
        if self.cache.memory_subjects < 0:
            raise ValueError("memory_subjects cannot be negative")
        if self.cache.enabled and self.cache.root is None:
            raise ValueError("Enabled disk cache requires a separate cache root")


@dataclass(frozen=True)
class SpatialTransformPlan:
    """One synchronized spatial transform for images and label."""

    flip_axis0: bool
    flip_axis1: bool
    rotation_quarters: int


@dataclass(frozen=True)
class IntensityTransformPlan:
    """Per-modality intensity scales and shifts."""

    scales: tuple[float, float, float, float]
    shifts: tuple[float, float, float, float]


def zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score one modality using only its nonzero voxels; preserve background."""
    values = np.asarray(volume, dtype=np.float32)
    mask = values != 0
    output = np.zeros_like(values, dtype=np.float32)
    if not bool(mask.any()):
        return output
    foreground = values[mask].astype(np.float64)
    mean = float(np.mean(foreground))
    standard_deviation = float(np.std(foreground, ddof=0))
    if standard_deviation == 0.0:
        return output
    output[mask] = ((foreground - mean) / standard_deviation).astype(np.float32)
    return output


def preprocess_modalities(
    modalities: dict[str, np.ndarray],
    config: PreprocessingConfig,
) -> np.ndarray:
    """Clip with fixed rules, normalize per patient/modality, and stack channels."""
    if set(modalities) != set(MODALITY_ORDER):
        raise ValueError(f"Expected modalities {MODALITY_ORDER}")
    shapes = {np.asarray(modalities[name]).shape for name in MODALITY_ORDER}
    if len(shapes) != 1:
        raise ValueError(f"All modalities must share one shape, got {sorted(shapes)}")
    channels: list[np.ndarray] = []
    for modality in config.modality_order:
        volume = np.asarray(modalities[modality], dtype=np.float32)
        if config.clipping.enabled:
            lower, upper = config.clipping.fixed_bounds[modality]
            volume = np.clip(volume, lower, upper)
        channels.append(zscore_nonzero(volume))
    return np.stack(channels, axis=0).astype(np.float32, copy=False)


def plan_spatial_transform(
    generator: np.random.Generator,
    config: SpatialAugmentationConfig,
) -> SpatialTransformPlan:
    """Draw one spatial plan shared by all channels and the mask."""
    if not config.enabled:
        return SpatialTransformPlan(False, False, 0)
    return SpatialTransformPlan(
        flip_axis0=bool(generator.random() < config.flip_probability),
        flip_axis1=bool(generator.random() < config.flip_probability),
        rotation_quarters=int(generator.integers(0, 4)) if config.rotate_90 else 0,
    )


def apply_spatial_transform(
    image: np.ndarray,
    label: np.ndarray,
    plan: SpatialTransformPlan,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an identical discrete transform to `[C,H,W]` image and `[H,W]` mask."""
    if image.ndim != 3 or label.ndim != 2:
        raise ValueError("Spatial transform expects image [C,H,W] and label [H,W]")
    if image.shape[1:] != label.shape:
        raise ValueError("Image and label spatial shapes must match")
    transformed_image = image
    transformed_label = label
    if plan.flip_axis0:
        transformed_image = np.flip(transformed_image, axis=1)
        transformed_label = np.flip(transformed_label, axis=0)
    if plan.flip_axis1:
        transformed_image = np.flip(transformed_image, axis=2)
        transformed_label = np.flip(transformed_label, axis=1)
    rotation = plan.rotation_quarters % 4
    if rotation:
        transformed_image = np.rot90(
            transformed_image,
            k=rotation,
            axes=(1, 2),
        )
        transformed_label = np.rot90(
            transformed_label,
            k=rotation,
            axes=(0, 1),
        )
    return (
        np.ascontiguousarray(transformed_image),
        np.ascontiguousarray(transformed_label),
    )


def plan_intensity_transform(
    generator: np.random.Generator,
    config: IntensityAugmentationConfig,
) -> IntensityTransformPlan:
    """Draw independent affine intensity parameters for each modality."""
    scales: list[float] = []
    shifts: list[float] = []
    for _ in MODALITY_ORDER:
        apply = config.enabled and (
            generator.random() < config.apply_probability_per_modality
        )
        if apply:
            scales.append(float(generator.uniform(*config.scale_range)))
            shifts.append(float(generator.uniform(*config.shift_range)))
        else:
            scales.append(1.0)
            shifts.append(0.0)
    return IntensityTransformPlan(
        scales=cast(tuple[float, float, float, float], tuple(scales)),
        shifts=cast(tuple[float, float, float, float], tuple(shifts)),
    )


def apply_intensity_transform(
    image: np.ndarray,
    plan: IntensityTransformPlan,
) -> np.ndarray:
    """Apply modality-specific intensity changes without altering zero background."""
    if image.ndim != 3 or image.shape[0] != len(MODALITY_ORDER):
        raise ValueError("Intensity transform expects image [4,H,W]")
    output = image.astype(np.float32, copy=True)
    for channel, (scale, shift) in enumerate(
        zip(plan.scales, plan.shifts, strict=True)
    ):
        foreground = output[channel] != 0
        output[channel, foreground] = output[channel, foreground] * scale + shift
    return output


def slice_has_tumor(
    label_volume: np.ndarray,
    slice_index: int,
    *,
    axis: int,
    minimum_voxels: int,
) -> bool:
    """Return whether one slice reaches the configured tumor-voxel threshold."""
    label_slice = np.take(label_volume, slice_index, axis=axis)
    return int(np.count_nonzero(label_slice)) >= minimum_voxels


def select_training_slice(
    label_volume: np.ndarray,
    generator: np.random.Generator,
    config: PreprocessingConfig,
) -> int:
    """Sample a tumor or non-tumor slice under the declared mixture ratio."""
    axis = config.slice_axis
    candidates = np.arange(label_volume.shape[axis])
    tumor = np.asarray(
        [
            index
            for index in candidates
            if slice_has_tumor(
                label_volume,
                int(index),
                axis=axis,
                minimum_voxels=(
                    config.training_sampling.tumor_minimum_voxels_per_slice
                ),
            )
        ],
        dtype=np.int64,
    )
    non_tumor = np.setdiff1d(candidates, tumor, assume_unique=True)
    choose_tumor = generator.random() < config.training_sampling.tumor_probability
    pool = tumor if choose_tumor else non_tumor
    if len(pool) == 0:
        pool = non_tumor if choose_tumor else tumor
    if len(pool) == 0:
        raise ValueError("Label volume has no selectable slices")
    return int(generator.choice(pool))


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    """Load and validate the preprocessing YAML without fitting cohort statistics."""
    raw = cast(DictConfig, OmegaConf.load(path)).preprocessing
    OmegaConf.resolve(raw)
    clipping_bounds = {
        str(modality): (float(bounds[0]), float(bounds[1]))
        for modality, bounds in raw.clipping.fixed_bounds.items()
    }
    cache_root_raw = raw.cache.root
    cache_root = (
        None
        if cache_root_raw is None or str(cache_root_raw).lower() == "null"
        else Path(str(cache_root_raw)).expanduser().resolve()
    )
    scale_range = tuple(
        float(value) for value in raw.augmentation.intensity.scale_range
    )
    shift_range = tuple(
        float(value) for value in raw.augmentation.intensity.shift_range
    )
    return PreprocessingConfig(
        modality_order=tuple(str(value) for value in raw.modality_order),
        slice_axis=cast(Literal[0, 1, 2], int(raw.slice_axis)),
        normalization=cast(
            Literal["nonzero_patient_modality_zscore"],
            str(raw.normalization),
        ),
        clipping=ClippingConfig(
            enabled=bool(raw.clipping.enabled),
            provenance=cast(
                Literal["development_train_only_fixed_bounds"],
                str(raw.clipping.provenance),
            ),
            fixed_bounds=clipping_bounds,
        ),
        training_sampling=TrainingSamplingConfig(
            tumor_probability=float(raw.training_sampling.tumor_probability),
            tumor_minimum_voxels_per_slice=int(
                raw.training_sampling.tumor_minimum_voxels_per_slice
            ),
            samples_per_patient_per_epoch=int(
                raw.training_sampling.samples_per_patient_per_epoch
            ),
        ),
        spatial_augmentation=SpatialAugmentationConfig(
            enabled=bool(raw.augmentation.spatial.enabled),
            flip_probability=float(raw.augmentation.spatial.flip_probability),
            rotate_90=bool(raw.augmentation.spatial.rotate_90),
        ),
        intensity_augmentation=IntensityAugmentationConfig(
            enabled=bool(raw.augmentation.intensity.enabled),
            apply_probability_per_modality=float(
                raw.augmentation.intensity.apply_probability_per_modality
            ),
            scale_range=cast(tuple[float, float], scale_range),
            shift_range=cast(tuple[float, float], shift_range),
        ),
        validation_include_all_slices=bool(raw.validation.include_all_slices),
        validation_include_empty_slices=bool(raw.validation.include_empty_slices),
        validation_deterministic=bool(raw.validation.deterministic),
        test_include_all_slices=bool(raw.test.include_all_slices),
        test_include_empty_slices=bool(raw.test.include_empty_slices),
        test_deterministic=bool(raw.test.deterministic),
        cache=CacheConfig(
            enabled=bool(raw.cache.enabled),
            root=cache_root,
            memory_subjects=int(raw.cache.memory_subjects),
        ),
    )
