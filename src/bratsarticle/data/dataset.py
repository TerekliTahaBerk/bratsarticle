"""Patient-safe BraTS 2D slice dataset with guarded internal-test construction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from bratsarticle.data.discovery import resolve_brats2020_training_root
from bratsarticle.data.preprocessing import (
    MODALITY_ORDER,
    PreprocessingConfig,
    apply_intensity_transform,
    apply_spatial_transform,
    plan_intensity_transform,
    plan_spatial_transform,
    preprocess_modalities,
    select_training_slice,
)
from bratsarticle.data.splits import (
    CrossValidationRole,
    DevelopmentSplitName,
    load_cv_fold_manifest,
    load_development_manifest,
    load_internal_test_manifest,
)
from bratsarticle.utils.paths import assert_output_paths_safe, is_relative_to

DatasetSplit = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SubjectVolume:
    """Normalized multimodal image, integer label, and array-axis spacing."""

    image: np.ndarray
    label: np.ndarray
    spacing_mm: tuple[float, float, float]


def _safe_raw_path(root: Path, relative_path: str) -> Path:
    path = (root / relative_path).resolve()
    if not is_relative_to(path, root):
        raise ValueError(f"Manifest path escapes the dataset root: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file does not exist: {path}")
    return path


def _load_subject_from_raw(
    row: Mapping[str, Any],
    root: Path,
    config: PreprocessingConfig,
) -> SubjectVolume:
    modality_arrays: dict[str, np.ndarray] = {}
    affines: list[np.ndarray] = []
    spacing: tuple[float, float, float] | None = None
    for modality in MODALITY_ORDER:
        path = _safe_raw_path(root, str(row[f"{modality}_relative_path"]))
        nifti_image = cast(nib.Nifti1Image, nib.load(path))
        modality_arrays[modality] = np.asarray(
            nifti_image.dataobj,
            dtype=np.float32,
        )
        affines.append(np.asarray(nifti_image.affine, dtype=np.float64))
        header: Any = nifti_image.header
        zooms = tuple(float(value) for value in header.get_zooms()[:3])
        if len(zooms) != 3:
            raise ValueError(f"Expected 3D spacing for {path}")
        if spacing is None:
            spacing = (zooms[0], zooms[1], zooms[2])
        elif not np.allclose(spacing, zooms, rtol=0.0, atol=1e-6):
            raise ValueError(f"Modality spacing mismatch for {row['subject_id']}")

    segmentation_path = _safe_raw_path(root, str(row["seg_relative_path"]))
    segmentation_image = cast(nib.Nifti1Image, nib.load(segmentation_path))
    label = np.asarray(segmentation_image.dataobj).astype(np.int16, copy=False)
    affines.append(np.asarray(segmentation_image.affine, dtype=np.float64))
    if any(
        not np.allclose(affines[0], affine, rtol=0.0, atol=1e-5)
        for affine in affines[1:]
    ):
        raise ValueError(f"Image/label affine mismatch for {row['subject_id']}")
    stacked_image = preprocess_modalities(modality_arrays, config)
    if stacked_image.shape[1:] != label.shape:
        raise ValueError(f"Image/label shape mismatch for {row['subject_id']}")
    invalid_labels = set(int(value) for value in np.unique(label)) - {0, 1, 2, 4}
    if invalid_labels:
        raise ValueError(
            f"Invalid labels for {row['subject_id']}: {sorted(invalid_labels)}"
        )
    if spacing is None:
        raise RuntimeError("No MRI modalities were loaded")
    return SubjectVolume(
        image=np.ascontiguousarray(stacked_image, dtype=np.float32),
        label=np.ascontiguousarray(label, dtype=np.int16),
        spacing_mm=spacing,
    )


def _cache_fingerprint(
    row: Mapping[str, Any],
    config: PreprocessingConfig,
) -> str:
    payload = {
        "subject_id": str(row["subject_id"]),
        "modality_order": config.modality_order,
        "normalization": config.normalization,
        "clipping": {
            "enabled": config.clipping.enabled,
            "provenance": config.clipping.provenance,
            "fixed_bounds": config.clipping.fixed_bounds,
        },
        "files": {
            role: {
                "path": str(row[f"{role}_relative_path"]),
                "sha256": str(row.get(f"{role}_sha256", "")),
            }
            for role in (*MODALITY_ORDER, "seg")
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class NormalizedVolumeCache:
    """Optional atomic disk cache located strictly outside the raw-data root."""

    def __init__(
        self,
        root: Path | None,
        raw_root: Path,
        *,
        enabled: bool,
    ) -> None:
        self.enabled = enabled
        self.root = root.resolve() if root is not None else None
        if enabled:
            if self.root is None:
                raise ValueError("Enabled cache requires a root")
            assert_output_paths_safe([self.root], [raw_root])
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(
        self,
        row: Mapping[str, Any],
        config: PreprocessingConfig,
    ) -> Path:
        if self.root is None:
            raise RuntimeError("Cache is disabled")
        subject_id = str(row["subject_id"])
        fingerprint = _cache_fingerprint(row, config)
        suffix = (
            ".npz"
            if config.cache.storage_format == "compressed_npz"
            else ".npycache"
        )
        return self.root / f"{subject_id}-{fingerprint[:20]}{suffix}"

    def load(
        self,
        row: Mapping[str, Any],
        config: PreprocessingConfig,
    ) -> SubjectVolume | None:
        """Return a cached normalized subject, or `None` on a cache miss."""
        if not self.enabled:
            return None
        path = self._path(row, config)
        if config.cache.storage_format == "compressed_npz":
            if not path.is_file():
                return None
            with np.load(path, allow_pickle=False) as cached:
                spacing_array = np.asarray(cached["spacing_mm"], dtype=np.float64)
                return SubjectVolume(
                    image=np.asarray(cached["image"], dtype=np.float32),
                    label=np.asarray(cached["label"], dtype=np.int16),
                    spacing_mm=(
                        float(spacing_array[0]),
                        float(spacing_array[1]),
                        float(spacing_array[2]),
                    ),
                )
        completion_marker = path / "COMPLETE"
        image_path = path / "image.npy"
        label_path = path / "label.npy"
        spacing_path = path / "spacing_mm.npy"
        if not (
            completion_marker.is_file()
            and image_path.is_file()
            and label_path.is_file()
            and spacing_path.is_file()
        ):
            return None
        image = np.load(image_path, mmap_mode="r", allow_pickle=False)
        label = np.load(label_path, mmap_mode="r", allow_pickle=False)
        spacing_array = np.asarray(
            np.load(spacing_path, allow_pickle=False),
            dtype=np.float64,
        )
        if image.dtype != np.float32 or label.dtype != np.int16:
            raise ValueError(f"Invalid memory-mapped cache dtypes at {path}")
        if image.ndim != 4 or image.shape[0] != len(MODALITY_ORDER):
            raise ValueError(f"Invalid memory-mapped image shape at {path}")
        if label.shape != image.shape[1:]:
            raise ValueError(f"Memory-mapped image/label shape mismatch at {path}")
        if spacing_array.shape != (3,):
            raise ValueError(f"Invalid memory-mapped spacing at {path}")
        return SubjectVolume(
            image=image,
            label=label,
            spacing_mm=(
                float(spacing_array[0]),
                float(spacing_array[1]),
                float(spacing_array[2]),
            ),
        )

    def store(
        self,
        row: Mapping[str, Any],
        config: PreprocessingConfig,
        volume: SubjectVolume,
    ) -> None:
        """Atomically store a normalized volume without touching raw data."""
        if not self.enabled:
            return
        destination = self._path(row, config)
        if config.cache.storage_format == "memory_mapped_npy":
            temporary = Path(
                tempfile.mkdtemp(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                )
            )
            try:
                np.save(
                    temporary / "image.npy",
                    np.asarray(volume.image, dtype=np.float32),
                    allow_pickle=False,
                )
                np.save(
                    temporary / "label.npy",
                    np.asarray(volume.label, dtype=np.int16),
                    allow_pickle=False,
                )
                np.save(
                    temporary / "spacing_mm.npy",
                    np.asarray(volume.spacing_mm, dtype=np.float32),
                    allow_pickle=False,
                )
                (temporary / "COMPLETE").touch()
                try:
                    temporary.replace(destination)
                except OSError:
                    if not destination.is_dir():
                        raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            return
        descriptor, raw_temporary = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    image=volume.image,
                    label=volume.label,
                    spacing_mm=np.asarray(volume.spacing_mm, dtype=np.float32),
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def _manifest_shape(row: Mapping[str, Any]) -> tuple[int, int, int]:
    raw_shape = row.get("t1_shape")
    if raw_shape is None:
        raise ValueError("Manifest requires t1_shape for exhaustive slice indexing")
    values = json.loads(str(raw_shape))
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"Invalid manifest shape: {raw_shape}")
    shape = tuple(int(value) for value in values)
    if any(value <= 0 for value in shape):
        raise ValueError(f"Invalid non-positive manifest shape: {raw_shape}")
    return (shape[0], shape[1], shape[2])


def _row_mapping(row: pd.Series[Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items()}


def extract_context_slices(
    image: np.ndarray,
    center_slice: int,
    *,
    slice_axis: int,
    context_offsets: tuple[int, ...],
) -> np.ndarray:
    """Stack modality-major neighboring slices with replicated boundaries."""
    if image.ndim != 4 or image.shape[0] != len(MODALITY_ORDER):
        raise ValueError("Context extraction expects image [4,X,Y,Z]")
    if slice_axis not in {0, 1, 2}:
        raise ValueError("slice_axis must be 0, 1, or 2")
    if not context_offsets or 0 not in context_offsets:
        raise ValueError("Context offsets must be nonempty and contain zero")
    image_axis = slice_axis + 1
    last_index = image.shape[image_axis] - 1
    indices = [
        min(max(center_slice + offset, 0), last_index)
        for offset in context_offsets
    ]
    extracted = np.take(image, indices, axis=image_axis)
    modality_context_first = np.moveaxis(extracted, image_axis, 1)
    spatial_shape = modality_context_first.shape[2:]
    return np.ascontiguousarray(
        modality_context_first.reshape(
            len(MODALITY_ORDER) * len(context_offsets),
            *spatial_shape,
        ),
        dtype=np.float32,
    )


class BraTSSliceDataset(Dataset[dict[str, Any]]):
    """2D dataset that preserves complete validation/test patient volumes."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        dataset_root: Path,
        config: PreprocessingConfig,
        *,
        split: DatasetSplit,
        seed: int,
        test_access_authorized: bool = False,
        context_offsets: tuple[int, ...] = (0,),
    ) -> None:
        if manifest.empty:
            raise ValueError("Dataset manifest cannot be empty")
        if split == "test" and not test_access_authorized:
            raise PermissionError(
                "Test dataset construction requires the guarded test builder"
            )
        self.manifest = manifest.reset_index(drop=True).copy()
        self.dataset_root = dataset_root.resolve()
        self.config = config
        self.split = split
        self.seed = int(seed)
        if not context_offsets or 0 not in context_offsets:
            raise ValueError("context_offsets must be nonempty and contain zero")
        self.context_offsets = tuple(int(offset) for offset in context_offsets)
        self.epoch = 0
        self.cache = NormalizedVolumeCache(
            config.cache.root,
            self.dataset_root,
            enabled=config.cache.enabled,
        )
        self._memory_cache: OrderedDict[int, SubjectVolume] = OrderedDict()
        self._slice_records: list[tuple[int, int]] = []
        if split != "train":
            for patient_index, (_, row) in enumerate(self.manifest.iterrows()):
                shape = _manifest_shape(_row_mapping(row))
                for slice_index in range(shape[config.slice_axis]):
                    self._slice_records.append((patient_index, slice_index))

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic training augmentation/sampling epoch."""
        if epoch < 0:
            raise ValueError("Epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.manifest) * (
                self.config.training_sampling.samples_per_patient_per_epoch
            )
        return len(self._slice_records)

    def _subject(self, patient_index: int) -> SubjectVolume:
        cached_memory = self._memory_cache.get(patient_index)
        if cached_memory is not None:
            self._memory_cache.move_to_end(patient_index)
            return cached_memory
        row = self.manifest.iloc[patient_index]
        row_mapping = _row_mapping(row)
        volume = self.cache.load(row_mapping, self.config)
        if volume is None:
            volume = _load_subject_from_raw(
                row_mapping,
                self.dataset_root,
                self.config,
            )
            self.cache.store(row_mapping, self.config, volume)
        if self.config.cache.memory_subjects > 0:
            self._memory_cache[patient_index] = volume
            self._memory_cache.move_to_end(patient_index)
            while len(self._memory_cache) > self.config.cache.memory_subjects:
                self._memory_cache.popitem(last=False)
        return volume

    def subject_volume(self, patient_index: int) -> SubjectVolume:
        """Load one authorized manifest subject for volume-level diagnostics."""
        if not 0 <= patient_index < len(self.manifest):
            raise IndexError(patient_index)
        return self._subject(patient_index)

    def _training_record(self, index: int) -> tuple[int, int, np.random.Generator]:
        samples_per_patient = (
            self.config.training_sampling.samples_per_patient_per_epoch
        )
        patient_index = (index // samples_per_patient) % len(self.manifest)
        generator = np.random.default_rng(self.seed + self.epoch * 1_000_003 + index)
        volume = self._subject(patient_index)
        slice_index = select_training_slice(volume.label, generator, self.config)
        return patient_index, slice_index, generator

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self.split == "train":
            patient_index, slice_index, generator = self._training_record(index)
        else:
            patient_index, slice_index = self._slice_records[index]
            generator = np.random.default_rng(self.seed + index)
        volume = self._subject(patient_index)
        image_slice = extract_context_slices(
            volume.image,
            slice_index,
            slice_axis=self.config.slice_axis,
            context_offsets=self.context_offsets,
        )
        label_slice = np.take(
            volume.label,
            slice_index,
            axis=self.config.slice_axis,
        )
        if self.split == "train":
            spatial_plan = plan_spatial_transform(
                generator,
                self.config.spatial_augmentation,
            )
            image_slice, label_slice = apply_spatial_transform(
                image_slice,
                label_slice,
                spatial_plan,
            )
            intensity_plan = plan_intensity_transform(
                generator,
                self.config.intensity_augmentation,
            )
            image_slice = apply_intensity_transform(image_slice, intensity_plan)
        row = self.manifest.iloc[patient_index]
        return {
            "image": torch.from_numpy(
                np.ascontiguousarray(image_slice, dtype=np.float32)
            ),
            "label": torch.from_numpy(
                np.ascontiguousarray(label_slice, dtype=np.int64)
            ),
            "subject_id": str(row["subject_id"]),
            "slice_index": slice_index,
            "slice_axis": self.config.slice_axis,
            "spacing_mm": np.asarray(volume.spacing_mm, dtype=np.float32),
            "is_empty_slice": not bool(np.any(label_slice)),
            "split": self.split,
        }


def build_development_dataset(
    split_dir: Path,
    split: DevelopmentSplitName,
    dataset_root: Path,
    config: PreprocessingConfig,
    *,
    seed: int,
    context_offsets: tuple[int, ...] = (0,),
) -> BraTSSliceDataset:
    """Build train/validation data without exposing the test manifest."""
    manifest = load_development_manifest(split_dir, split)
    resolved_root = resolve_brats2020_training_root(dataset_root)
    return BraTSSliceDataset(
        manifest,
        resolved_root,
        config,
        split=split,
        seed=seed,
        context_offsets=context_offsets,
    )


def build_cv_fold_dataset(
    fold_path: Path,
    canonical_manifest_path: Path,
    role: CrossValidationRole,
    dataset_root: Path,
    config: PreprocessingConfig,
    *,
    seed: int,
    context_offsets: tuple[int, ...] = (0,),
) -> BraTSSliceDataset:
    """Build one frozen v2 fold without exposing any test manifest."""
    manifest = load_cv_fold_manifest(
        fold_path,
        canonical_manifest_path,
        role,
    )
    resolved_root = resolve_brats2020_training_root(dataset_root)
    return BraTSSliceDataset(
        manifest,
        resolved_root,
        config,
        split=role,
        seed=seed,
        context_offsets=context_offsets,
    )


def build_internal_test_dataset(
    split_dir: Path,
    dataset_root: Path,
    config: PreprocessingConfig,
    *,
    seed: int,
    allow_test_evaluation: bool,
    purpose: str,
    audit_log: Path = Path("artifacts/test_access_log.jsonl"),
    context_offsets: tuple[int, ...] = (0,),
) -> BraTSSliceDataset:
    """Build exhaustive test data only after the guarded manifest access."""
    manifest = load_internal_test_manifest(
        split_dir,
        allow_test_evaluation=allow_test_evaluation,
        purpose=purpose,
        audit_log=audit_log,
    )
    resolved_root = resolve_brats2020_training_root(dataset_root)
    return BraTSSliceDataset(
        manifest,
        resolved_root,
        config,
        split="test",
        seed=seed,
        test_access_authorized=True,
        context_offsets=context_offsets,
    )
