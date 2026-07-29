import json
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from bratsarticle.data.dataset import BraTSSliceDataset, NormalizedVolumeCache
from bratsarticle.data.preprocessing import (
    CacheConfig,
    IntensityAugmentationConfig,
    PreprocessingConfig,
    SpatialAugmentationConfig,
    preprocess_modalities,
)
from bratsarticle.utils.paths import PathSafetyError


def _write_subject(root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    subject_id = "BraTS20_Training_001"
    subject_root = root / subject_id
    subject_root.mkdir(parents=True)
    shape = (5, 6, 7)
    coordinate = np.indices(shape).astype(np.float32)
    modalities = {
        "t1": coordinate[0] + 1.0,
        "t1ce": coordinate[1] ** 2 + 1.0,
        "t2": coordinate[2] ** 2 + 1.0,
        "flair": coordinate.sum(axis=0) ** 2 + 1.0,
    }
    label = np.zeros(shape, dtype=np.int16)
    label[1:4, 2:5, 3] = 2
    affine = np.eye(4, dtype=np.float64)
    row: dict[str, object] = {
        "subject_id": subject_id,
        "t1_shape": json.dumps(shape),
    }
    for role, array in {**modalities, "seg": label}.items():
        filename = f"{subject_id}_{role}.nii"
        path = subject_root / filename
        nib.save(nib.Nifti1Image(array, affine), path)
        row[f"{role}_relative_path"] = f"{subject_id}/{filename}"
        row[f"{role}_sha256"] = f"synthetic-{role}"
    return pd.DataFrame([row]), modalities


def _deterministic_config(**changes: object) -> PreprocessingConfig:
    base = PreprocessingConfig(
        spatial_augmentation=SpatialAugmentationConfig(enabled=False),
        intensity_augmentation=IntensityAugmentationConfig(enabled=False),
    )
    return replace(base, **changes)


def test_validation_dataset_preserves_empty_slices_and_conventions(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest, modalities = _write_subject(raw_root)
    config = _deterministic_config()
    dataset = BraTSSliceDataset(
        manifest,
        raw_root,
        config,
        split="validation",
        seed=17,
    )

    assert len(dataset) == 7
    empty_sample = dataset[0]
    tumor_sample = dataset[3]
    assert empty_sample["is_empty_slice"]
    assert not tumor_sample["is_empty_slice"]
    assert empty_sample["image"].shape == (4, 5, 6)
    assert empty_sample["image"].dtype == torch.float32
    assert empty_sample["label"].shape == (5, 6)
    assert empty_sample["label"].dtype == torch.int64
    assert empty_sample["subject_id"] == "BraTS20_Training_001"
    assert empty_sample["slice_axis"] == 2

    expected = preprocess_modalities(modalities, config)
    assert torch.equal(
        tumor_sample["image"],
        torch.from_numpy(np.ascontiguousarray(expected[:, :, :, 3])),
    )


def test_validation_dataset_is_deterministic(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest, _ = _write_subject(raw_root)
    dataset = BraTSSliceDataset(
        manifest,
        raw_root,
        _deterministic_config(),
        split="validation",
        seed=91,
    )

    first = dataset[3]
    second = dataset[3]
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["label"], second["label"])
    assert first["slice_index"] == second["slice_index"]


def test_training_sampling_and_augmentation_are_epoch_deterministic(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest, _ = _write_subject(raw_root)
    dataset = BraTSSliceDataset(
        manifest,
        raw_root,
        PreprocessingConfig(),
        split="train",
        seed=73,
    )

    first = dataset[0]
    repeated = dataset[0]
    assert torch.equal(first["image"], repeated["image"])
    assert torch.equal(first["label"], repeated["label"])
    assert first["slice_index"] == repeated["slice_index"]

    dataset.set_epoch(1)
    next_epoch = dataset[0]
    assert not torch.equal(first["image"], next_epoch["image"])


def test_disk_cache_is_outside_and_does_not_modify_raw_tree(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    raw_root.mkdir()
    manifest, _ = _write_subject(raw_root)
    before = sorted(path.relative_to(raw_root) for path in raw_root.rglob("*"))
    config = _deterministic_config(
        cache=CacheConfig(enabled=True, root=cache_root, memory_subjects=0)
    )
    dataset = BraTSSliceDataset(
        manifest,
        raw_root,
        config,
        split="validation",
        seed=4,
    )

    _ = dataset[0]
    after = sorted(path.relative_to(raw_root) for path in raw_root.rglob("*"))
    assert before == after
    assert len(list(cache_root.glob("*.npz"))) == 1
    assert not cache_root.is_relative_to(raw_root)


def test_memory_mapped_cache_reads_arrays_without_full_volume_copy(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    cache_root = tmp_path / "cache"
    raw_root.mkdir()
    manifest, modalities = _write_subject(raw_root)
    config = _deterministic_config(
        cache=CacheConfig(
            enabled=True,
            root=cache_root,
            memory_subjects=0,
            storage_format="memory_mapped_npy",
        )
    )
    dataset = BraTSSliceDataset(
        manifest,
        raw_root,
        config,
        split="validation",
        seed=4,
    )

    expected_image = preprocess_modalities(modalities, config)
    _ = dataset[0]
    cache_directories = list(cache_root.glob("*.npycache"))
    assert len(cache_directories) == 1
    assert (cache_directories[0] / "COMPLETE").is_file()

    row = {str(key): value for key, value in manifest.iloc[0].items()}
    cache = NormalizedVolumeCache(cache_root, raw_root, enabled=True)
    loaded = cache.load(row, config)
    assert loaded is not None
    assert isinstance(loaded.image, np.memmap)
    assert isinstance(loaded.label, np.memmap)
    assert np.array_equal(loaded.image, expected_image)
    assert np.array_equal(
        loaded.label,
        nib.load(raw_root / str(row["seg_relative_path"])).get_fdata().astype(
            np.int16
        ),
    )


def test_cache_below_raw_root_is_rejected(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest, _ = _write_subject(raw_root)
    config = _deterministic_config(
        cache=CacheConfig(
            enabled=True,
            root=raw_root / "forbidden-cache",
            memory_subjects=0,
        )
    )

    with pytest.raises(PathSafetyError, match="raw-data root"):
        BraTSSliceDataset(
            manifest,
            raw_root,
            config,
            split="validation",
            seed=4,
        )


def test_test_dataset_cannot_be_constructed_directly(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest, _ = _write_subject(raw_root)

    with pytest.raises(PermissionError, match="guarded test builder"):
        BraTSSliceDataset(
            manifest,
            raw_root,
            _deterministic_config(),
            split="test",
            seed=1,
        )
