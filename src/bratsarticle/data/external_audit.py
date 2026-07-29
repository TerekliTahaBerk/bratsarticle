"""Read-only inventory and overlap audit for the BraTS-Africa external cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes

from bratsarticle.data.preprocessing import zscore_nonzero
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.paths import assert_existing_directory, assert_output_paths_safe
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
)

EXTERNAL_COHORT_VERSION: Final[str] = "BraTS-Africa-TCIA-v1"
EXTERNAL_COLLECTION_DOI: Final[str] = "10.7937/V8H6-8X67"
EXTERNAL_LICENSE: Final[str] = "CC BY 4.0"
MODALITY_SUFFIXES: Final[Mapping[str, str]] = {
    "t1": "-t1n.nii.gz",
    "t1ce": "-t1c.nii.gz",
    "t2": "-t2w.nii.gz",
    "flair": "-t2f.nii.gz",
    "seg": "-seg.nii.gz",
}
MODALITY_ORDER: Final[tuple[str, ...]] = ("t1", "t1ce", "t2", "flair")
EXPECTED_EXTERNAL_LABELS: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
NORMALIZED_NEAR_MATCH_THRESHOLD: Final[float] = 1e-5
_PATIENT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^BraTS-SSA-\d{5}-\d{3}$"
)


@dataclass(frozen=True)
class ExternalSubject:
    """One discovered external patient and its five required NIfTI files."""

    patient_id: str
    disease_group: str
    source_sheet: str
    center: str
    scanner_information: str
    label_count_reported: str
    imaging_comments: str
    neoplasm_category: str
    subject_dir: Path
    files: Mapping[str, Path]


@dataclass(frozen=True)
class VolumeFingerprint:
    """Exact and robust signatures derived from a normalized four-channel volume."""

    normalized_sha256: tuple[str, str, str, str]
    sampled_sha256: tuple[str, str, str, str]
    descriptor: np.ndarray
    descriptor_sha256: str


@dataclass(frozen=True)
class DevelopmentFingerprint:
    """BraTS 2020 development signature backed by the immutable normalized cache."""

    patient_id: str
    fingerprint: VolumeFingerprint


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _metadata_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "patient_id",
            "disease_group",
            "source_sheet",
            "center",
            "scanner_information",
            "number_of_tumor_subregion_labels",
            "imaging_finding_comments",
            "neoplasm_category",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Metadata CSV is missing required fields: {sorted(required)}"
            )
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            patient_id = str(row["patient_id"])
            if patient_id in rows:
                raise ValueError(f"Duplicate metadata patient: {patient_id}")
            rows[patient_id] = {key: str(value or "") for key, value in row.items()}
    return rows


def resolve_brats_africa_data_root(candidate: Path) -> Path:
    """Resolve the directory containing the two BraTS-Africa disease groups."""
    root = assert_existing_directory(candidate, "BraTS-Africa external root")
    direct = root / "95_Glioma"
    if direct.is_dir() and (root / "51_OtherNeoplasms").is_dir():
        return root
    matches = sorted(
        path
        for path in root.rglob("BraTS-Africa")
        if (path / "95_Glioma").is_dir()
        and (path / "51_OtherNeoplasms").is_dir()
    )
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one BraTS-Africa data root containing "
            f"95_Glioma and 51_OtherNeoplasms, found {len(matches)}: {matches}"
        )
    return matches[0].resolve()


def discover_external_subjects(
    external_root: Path,
    metadata_csv: Path,
) -> list[ExternalSubject]:
    """Discover all public-release subjects without writing below the data root."""
    data_root = resolve_brats_africa_data_root(external_root)
    metadata = _metadata_rows(metadata_csv)
    discoveries: list[ExternalSubject] = []
    for folder_name, disease_group, source_sheet in (
        ("95_Glioma", "glioma", "95 Glioma"),
        ("51_OtherNeoplasms", "other_neoplasm", "51 OtherNeoplasms"),
    ):
        group_root = data_root / folder_name
        for subject_dir in sorted(
            path for path in group_root.iterdir() if path.is_dir()
        ):
            patient_id = subject_dir.name
            if not _PATIENT_PATTERN.fullmatch(patient_id):
                raise ValueError(
                    f"Unexpected external patient identifier: {patient_id}"
                )
            if patient_id not in metadata:
                raise ValueError(f"No official metadata row for {patient_id}")
            row = metadata[patient_id]
            if row["disease_group"] != disease_group:
                raise ValueError(f"Disease-group mismatch for {patient_id}")
            files: dict[str, Path] = {}
            for role, suffix in MODALITY_SUFFIXES.items():
                expected = subject_dir / f"{patient_id}{suffix}"
                if not expected.is_file():
                    raise FileNotFoundError(
                        f"Missing required {role} file for {patient_id}: {expected}"
                    )
                files[role] = expected
            extra_nifti = sorted(
                path.name
                for path in subject_dir.glob("*.nii*")
                if path not in files.values()
            )
            if extra_nifti:
                raise ValueError(
                    f"Unexpected NIfTI files for {patient_id}: {extra_nifti}"
                )
            discoveries.append(
                ExternalSubject(
                    patient_id=patient_id,
                    disease_group=disease_group,
                    source_sheet=source_sheet,
                    center=row["center"],
                    scanner_information=row["scanner_information"],
                    label_count_reported=row[
                        "number_of_tumor_subregion_labels"
                    ],
                    imaging_comments=row["imaging_finding_comments"],
                    neoplasm_category=row["neoplasm_category"],
                    subject_dir=subject_dir,
                    files=files,
                )
            )
    missing_on_disk = sorted(set(metadata) - {item.patient_id for item in discoveries})
    if missing_on_disk:
        raise ValueError(f"Metadata patients missing on disk: {missing_on_disk}")
    return sorted(discoveries, key=lambda item: item.patient_id)


def _hash_float32_array(array: np.ndarray) -> str:
    values = np.asarray(array, dtype="<f4", order="C")
    digest = hashlib.sha256()
    if values.ndim < 1:
        digest.update(values.tobytes(order="C"))
        return digest.hexdigest()
    for start in range(0, values.shape[0], 16):
        digest.update(
            np.ascontiguousarray(values[start : start + 16]).tobytes(order="C")
        )
    return digest.hexdigest()


def _sampled_hash(array: np.ndarray, target: int = 32) -> str:
    indices = [
        np.rint(np.linspace(0, size - 1, min(target, size))).astype(np.int64)
        for size in array.shape
    ]
    sampled = np.asarray(array[np.ix_(*indices)], dtype=np.float32)
    quantized = np.rint(np.clip(sampled, -8.0, 8.0) * 4096.0).astype("<i2")
    return hashlib.sha256(quantized.tobytes(order="C")).hexdigest()


def _resample_profile(values: np.ndarray, size: int = 32) -> np.ndarray:
    if values.size == 0:
        return np.zeros(size, dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, size)
    result = np.interp(target, source, np.asarray(values, dtype=np.float64))
    reverse = result[::-1]
    if tuple(reverse) < tuple(result):
        result = reverse
    return result


def _channel_descriptor(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    mask = values != 0
    nonzero = values[mask].astype(np.float64)
    if nonzero.size == 0:
        return np.zeros(64 + 9 + 3 * 32 + 4, dtype=np.float64)
    histogram, _ = np.histogram(nonzero, bins=64, range=(-6.0, 6.0))
    histogram_values = histogram.astype(np.float64) / float(nonzero.size)
    quantiles = np.quantile(
        nonzero,
        (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    )
    projection_profiles = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        projection = np.asarray(
            np.mean(mask, axis=other_axes, dtype=np.float64),
            dtype=np.float64,
        )
        projection_profiles.append(_resample_profile(projection))
    shape_features = np.asarray(
        [
            float(nonzero.size) / float(values.size),
            float(np.mean(nonzero)),
            float(np.std(nonzero, ddof=0)),
            float(np.mean(np.abs(nonzero))),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [histogram_values, quantiles, *projection_profiles, shape_features]
    )


def fingerprint_normalized_channels(
    channels: Sequence[np.ndarray],
) -> VolumeFingerprint:
    """Create exact, sampled, and distance-comparable signatures."""
    if len(channels) != 4:
        raise ValueError("Expected exactly four normalized modalities")
    shapes = {tuple(np.asarray(channel).shape) for channel in channels}
    if len(shapes) != 1:
        raise ValueError(f"All fingerprint channels must share a shape: {shapes}")
    normalized_hashes = tuple(_hash_float32_array(channel) for channel in channels)
    sampled_hashes = tuple(_sampled_hash(channel) for channel in channels)
    descriptor = np.concatenate(
        [_channel_descriptor(channel) for channel in channels]
    ).astype(np.float64, copy=False)
    descriptor_sha256 = hashlib.sha256(
        np.asarray(descriptor, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    return VolumeFingerprint(
        normalized_sha256=cast(
            tuple[str, str, str, str],
            normalized_hashes,
        ),
        sampled_sha256=cast(
            tuple[str, str, str, str],
            sampled_hashes,
        ),
        descriptor=descriptor,
        descriptor_sha256=descriptor_sha256,
    )


def _load_external_channel(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    image = cast(nib.Nifti1Image, nib.load(str(path), mmap="r"))
    raw = np.asarray(image.dataobj)
    normalized = zscore_nonzero(raw)
    affine = np.asarray(image.affine, dtype=np.float64)
    zooms = tuple(
        float(value)
        for value in image.header.get_zooms()[:3]  # type: ignore[no-untyped-call]
    )
    information = {
        "shape": list(raw.shape),
        "spacing": [round(value, 8) for value in zooms],
        "orientation": "".join(
            value if value is not None else "?"
            for value in aff2axcodes(affine)  # type: ignore[no-untyped-call]
        ),
        "dtype": str(image.get_data_dtype()),  # type: ignore[no-untyped-call]
        "nan_count": int(np.isnan(raw).sum())
        if np.issubdtype(raw.dtype, np.floating)
        else 0,
        "inf_count": int(np.isinf(raw).sum())
        if np.issubdtype(raw.dtype, np.floating)
        else 0,
        "all_zero": bool(np.count_nonzero(raw) == 0),
    }
    return normalized, information


def inspect_external_subject(
    subject: ExternalSubject,
    data_root: Path,
) -> tuple[dict[str, Any], VolumeFingerprint]:
    """Inspect one external patient and return its inventory row and signatures."""
    normalized_channels: list[np.ndarray] = []
    role_information: dict[str, dict[str, Any]] = {}
    file_hashes: dict[str, str] = {}
    file_sizes: dict[str, int] = {}
    for role in MODALITY_ORDER:
        path = subject.files[role]
        normalized, information = _load_external_channel(path)
        normalized_channels.append(normalized)
        role_information[role] = information
        file_hashes[role] = file_digest(path)
        file_sizes[role] = path.stat().st_size

    segmentation_path = subject.files["seg"]
    segmentation_image = cast(
        nib.Nifti1Image,
        nib.load(str(segmentation_path), mmap="r"),
    )
    segmentation = np.asarray(segmentation_image.dataobj)
    labels = frozenset(int(value) for value in np.unique(segmentation))
    unexpected_labels = sorted(labels - EXPECTED_EXTERNAL_LABELS)
    mapped = np.asarray(segmentation, dtype=np.int16)
    mapped = np.where(mapped == 3, 4, mapped).astype(np.int16, copy=False)
    spacing = tuple(
        float(value)
        for value in segmentation_image.header.get_zooms()[:3]  # type: ignore[no-untyped-call]
    )
    shapes = {tuple(info["shape"]) for info in role_information.values()}
    shapes.add(tuple(segmentation.shape))
    spacings = {
        tuple(float(value) for value in info["spacing"])
        for info in role_information.values()
    }
    spacings.add(tuple(round(value, 8) for value in spacing))
    orientations = {
        str(info["orientation"]) for info in role_information.values()
    }
    orientations.add(
        "".join(
            value if value is not None else "?"
            for value in aff2axcodes(segmentation_image.affine)  # type: ignore[no-untyped-call]
        )
    )
    integrity_errors: list[str] = []
    if len(shapes) != 1:
        integrity_errors.append(f"shape_mismatch:{sorted(shapes)}")
    if len(spacings) != 1:
        integrity_errors.append(f"spacing_mismatch:{sorted(spacings)}")
    if len(orientations) != 1:
        integrity_errors.append(f"orientation_mismatch:{sorted(orientations)}")
    if unexpected_labels:
        integrity_errors.append(f"unexpected_labels:{unexpected_labels}")
    if not bool(np.any(mapped != 0)):
        integrity_errors.append("empty_segmentation")
    for role, information in role_information.items():
        if information["nan_count"]:
            integrity_errors.append(f"{role}_nan")
        if information["inf_count"]:
            integrity_errors.append(f"{role}_inf")
        if information["all_zero"]:
            integrity_errors.append(f"{role}_all_zero")

    fingerprint = fingerprint_normalized_channels(normalized_channels)
    data_root_resolved = data_root.resolve()
    relative_paths = {
        role: path.resolve().relative_to(data_root_resolved).as_posix()
        for role, path in subject.files.items()
    }
    voxel_volume = float(math.prod(spacing))
    row: dict[str, Any] = {
        "cohort_version": EXTERNAL_COHORT_VERSION,
        "collection_doi": EXTERNAL_COLLECTION_DOI,
        "patient_id": subject.patient_id,
        "study_id": subject.patient_id,
        "session_id": subject.patient_id.rsplit("-", maxsplit=1)[-1],
        "disease_group": subject.disease_group,
        "source_sheet": subject.source_sheet,
        "t1_path": relative_paths["t1"],
        "t1ce_path": relative_paths["t1ce"],
        "t2_path": relative_paths["t2"],
        "flair_path": relative_paths["flair"],
        "label_path": relative_paths["seg"],
        "diagnosis": (
            "glioma"
            if subject.disease_group == "glioma"
            else subject.neoplasm_category or "other_neoplasm_unspecified"
        ),
        "grade": "",
        "institution": subject.center,
        "scanner_vendor": subject.scanner_information.split(maxsplit=1)[0],
        "scanner_model": subject.scanner_information,
        "field_strength_t": 1.5,
        "voxel_spacing": _compact_json(sorted(spacings)),
        "shape": _compact_json(sorted(shapes)),
        "orientation": _compact_json(sorted(orientations)),
        "preprocessing_status": (
            "TCIA processed BraTS-Africa release; co-registered, "
            "skull-stripped, standardized NIfTI"
        ),
        "label_source": "expert consensus BraTS-Africa tumor subregions",
        "source_label_set": _compact_json(sorted(labels)),
        "mapped_brats2020_label_set": _compact_json(
            sorted(int(value) for value in np.unique(mapped))
        ),
        "label_mapping": "0->0;1->1(NCR/NET);2->2(ED);3->4(ET)",
        "wt_voxel_count": int(np.isin(mapped, (1, 2, 4)).sum()),
        "tc_voxel_count": int(np.isin(mapped, (1, 4)).sum()),
        "et_voxel_count": int(np.equal(mapped, 4).sum()),
        "wt_volume_mm3": float(np.isin(mapped, (1, 2, 4)).sum()) * voxel_volume,
        "tc_volume_mm3": float(np.isin(mapped, (1, 4)).sum()) * voxel_volume,
        "et_volume_mm3": float(np.equal(mapped, 4).sum()) * voxel_volume,
        "reported_tumor_subregion_label_count": subject.label_count_reported,
        "imaging_finding_comments": subject.imaging_comments,
        "neoplasm_category": subject.neoplasm_category,
        "license": EXTERNAL_LICENSE,
        "primary_confirmatory_eligibility": (
            "eligible" if subject.disease_group == "glioma" else "supportive_only"
        ),
        "eligibility_status": "eligible" if not integrity_errors else "excluded",
        "exclusion_reason": "|".join(integrity_errors),
        "t1_file_size_bytes": file_sizes["t1"],
        "t1ce_file_size_bytes": file_sizes["t1ce"],
        "t2_file_size_bytes": file_sizes["t2"],
        "flair_file_size_bytes": file_sizes["flair"],
        "label_file_size_bytes": segmentation_path.stat().st_size,
        "t1_sha256": file_hashes["t1"],
        "t1ce_sha256": file_hashes["t1ce"],
        "t2_sha256": file_hashes["t2"],
        "flair_sha256": file_hashes["flair"],
        "label_sha256": file_digest(segmentation_path),
        "t1_normalized_sha256": fingerprint.normalized_sha256[0],
        "t1ce_normalized_sha256": fingerprint.normalized_sha256[1],
        "t2_normalized_sha256": fingerprint.normalized_sha256[2],
        "flair_normalized_sha256": fingerprint.normalized_sha256[3],
        "t1_sampled_sha256": fingerprint.sampled_sha256[0],
        "t1ce_sampled_sha256": fingerprint.sampled_sha256[1],
        "t2_sampled_sha256": fingerprint.sampled_sha256[2],
        "flair_sampled_sha256": fingerprint.sampled_sha256[3],
        "robust_descriptor_sha256": fingerprint.descriptor_sha256,
    }
    return row, fingerprint


def _development_fingerprint(cache_path: Path) -> DevelopmentFingerprint:
    patient_id = cache_path.name.split("-", maxsplit=1)[0]
    image_path = cache_path / "image.npy"
    label_path = cache_path / "label.npy"
    spacing_path = cache_path / "spacing_mm.npy"
    completion_path = cache_path / "COMPLETE"
    if not all(
        path.is_file()
        for path in (image_path, label_path, spacing_path, completion_path)
    ):
        raise ValueError(f"Incomplete development cache: {cache_path}")
    image = np.load(image_path, mmap_mode="r", allow_pickle=False)
    if image.shape != (4, 240, 240, 155) or image.dtype != np.float32:
        raise ValueError(
            f"Unexpected development cache image at {cache_path}: "
            f"shape={image.shape}, dtype={image.dtype}"
        )
    fingerprint = fingerprint_normalized_channels(
        [np.asarray(image[index]) for index in range(4)]
    )
    return DevelopmentFingerprint(patient_id=patient_id, fingerprint=fingerprint)


def load_development_fingerprints(
    cache_root: Path,
    workers: int,
) -> list[DevelopmentFingerprint]:
    """Load all 369 immutable BraTS 2020 cache signatures."""
    root = assert_existing_directory(cache_root, "BraTS 2020 normalized cache root")
    cache_paths = sorted(root.glob("BraTS20_Training_*.npycache"))
    if len(cache_paths) != 369:
        raise ValueError(
            f"Expected 369 development caches, found {len(cache_paths)} at {root}"
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(_development_fingerprint, cache_paths))
    patient_ids = [item.patient_id for item in results]
    if len(set(patient_ids)) != 369:
        raise ValueError("Development cache contains duplicate patient identifiers")
    return sorted(results, key=lambda item: item.patient_id)


def _canonical_hashes(
    canonical_manifest: Path,
) -> tuple[set[str], set[str]]:
    frame = pd.read_csv(canonical_manifest, dtype=str, keep_default_na=False)
    if len(frame) != 369:
        raise ValueError(f"Expected 369 canonical rows, found {len(frame)}")
    identifiers = set(frame["subject_id"].astype(str))
    hashes: set[str] = set()
    for role in (*MODALITY_ORDER, "seg"):
        column = f"{role}_sha256"
        hashes.update(value for value in frame[column].astype(str) if value)
    return identifiers, hashes


def _root_mean_square_distance(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(difference))))


def audit_overlap(
    inventory_rows: Sequence[Mapping[str, Any]],
    external_fingerprints: Mapping[str, VolumeFingerprint],
    development_fingerprints: Sequence[DevelopmentFingerprint],
    canonical_manifest: Path,
    evidence_artifact: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare every external patient against every development patient."""
    canonical_ids, canonical_hashes = _canonical_hashes(canonical_manifest)
    development_matrix = np.stack(
        [item.fingerprint.descriptor for item in development_fingerprints],
        axis=0,
    )
    exact_index = {
        item.fingerprint.normalized_sha256: item.patient_id
        for item in development_fingerprints
    }
    sampled_index = {
        item.fingerprint.sampled_sha256: item.patient_id
        for item in development_fingerprints
    }
    rows: list[dict[str, Any]] = []
    nearest_distances: list[float] = []
    for inventory in inventory_rows:
        patient_id = str(inventory["patient_id"])
        fingerprint = external_fingerprints[patient_id]
        distances = np.sqrt(
            np.mean(
                np.square(development_matrix - fingerprint.descriptor[None, :]),
                axis=1,
            )
        )
        nearest_index = int(np.argmin(distances))
        nearest = development_fingerprints[nearest_index]
        nearest_distance = float(distances[nearest_index])
        direct_distance = _root_mean_square_distance(
            fingerprint.descriptor,
            nearest.fingerprint.descriptor,
        )
        if not np.isclose(nearest_distance, direct_distance, rtol=0.0, atol=1e-15):
            raise RuntimeError("Nearest-signature distance calculation mismatch")
        nearest_distances.append(nearest_distance)
        normalized_match = exact_index.get(fingerprint.normalized_sha256)
        sampled_match = sampled_index.get(fingerprint.sampled_sha256)
        external_hashes = {
            str(inventory[f"{role}_sha256"])
            for role in (*MODALITY_ORDER, "label")
        }
        file_match = bool(external_hashes & canonical_hashes)
        identifier_match = patient_id in canonical_ids
        near_match = nearest_distance <= NORMALIZED_NEAR_MATCH_THRESHOLD
        if (
            identifier_match
            or file_match
            or normalized_match is not None
            or sampled_match is not None
            or near_match
        ):
            decision = "manual_review_required_possible_overlap"
        else:
            decision = "no_overlap_detected"
        rows.append(
            {
                "external_cohort_version": EXTERNAL_COHORT_VERSION,
                "external_patient_id": patient_id,
                "canonical_patient_id": (
                    normalized_match
                    or sampled_match
                    or nearest.patient_id
                ),
                "identifier_match": identifier_match,
                "file_sha256_match": file_match,
                "image_content_signature_match": normalized_match is not None,
                "sampled_content_signature_match": sampled_match is not None,
                "normalized_volume_signature_match": near_match,
                "normalized_signature_rms_distance": nearest_distance,
                "normalized_near_match_threshold": NORMALIZED_NEAR_MATCH_THRESHOLD,
                "metadata_match": False,
                "institution_mapping_signal": (
                    "independent Sub-Saharan African centers; no canonical "
                    "BraTS 2020 institution identifier mapping"
                ),
                "decision": decision,
                "evidence_artifact": evidence_artifact,
            }
        )
    decisions = {str(row["decision"]) for row in rows}
    summary = {
        "external_patient_count": len(rows),
        "development_patient_count": len(development_fingerprints),
        "pairwise_comparison_count": len(rows) * len(development_fingerprints),
        "identifier_match_count": sum(bool(row["identifier_match"]) for row in rows),
        "file_sha256_match_count": sum(
            bool(row["file_sha256_match"]) for row in rows
        ),
        "image_content_signature_match_count": sum(
            bool(row["image_content_signature_match"]) for row in rows
        ),
        "sampled_content_signature_match_count": sum(
            bool(row["sampled_content_signature_match"]) for row in rows
        ),
        "normalized_volume_signature_match_count": sum(
            bool(row["normalized_volume_signature_match"]) for row in rows
        ),
        "minimum_normalized_signature_rms_distance": min(nearest_distances),
        "median_normalized_signature_rms_distance": float(
            np.median(nearest_distances)
        ),
        "maximum_normalized_signature_rms_distance": max(nearest_distances),
        "near_match_threshold": NORMALIZED_NEAR_MATCH_THRESHOLD,
        "decisions": sorted(decisions),
        "zero_overlap_established": decisions == {"no_overlap_detected"},
    }
    return rows, summary


def run_external_audit(
    *,
    external_root: Path,
    metadata_csv: Path,
    development_cache_root: Path,
    canonical_manifest: Path,
    inventory_output: Path,
    overlap_output: Path,
    signature_output: Path,
    summary_output: Path,
    access_log: Path,
    workers: int,
) -> dict[str, Any]:
    """Run Gate C audit without model inference or external-result access."""
    if workers < 1:
        raise ValueError("workers must be positive")
    data_root = resolve_brats_africa_data_root(external_root)
    outputs = (
        inventory_output,
        overlap_output,
        signature_output,
        summary_output,
        access_log,
    )
    assert_output_paths_safe(outputs, [data_root])
    subjects = discover_external_subjects(data_root, metadata_csv)
    if len(subjects) != 146:
        raise ValueError(f"Expected 146 external patients, found {len(subjects)}")
    development = load_development_fingerprints(development_cache_root, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(
            executor.map(
                lambda subject: inspect_external_subject(subject, data_root),
                subjects,
            )
        )
    inventory_rows = [item[0] for item in inspected]
    external_fingerprints = {
        str(row["patient_id"]): fingerprint
        for row, fingerprint in inspected
    }
    overlap_rows, overlap_summary = audit_overlap(
        inventory_rows=inventory_rows,
        external_fingerprints=external_fingerprints,
        development_fingerprints=development,
        canonical_manifest=canonical_manifest,
        evidence_artifact=signature_output.as_posix(),
    )
    signature_payload: dict[str, Any] = {
        "schema_version": 1,
        "external_cohort_version": EXTERNAL_COHORT_VERSION,
        "external_collection_doi": EXTERNAL_COLLECTION_DOI,
        "normalization": "nonzero_patient_modality_zscore",
        "descriptor": (
            "per-modality normalized-intensity histogram, quantiles, "
            "orientation-canonicalized nonzero-mask projections, and moments"
        ),
        "near_match_threshold_rms": NORMALIZED_NEAR_MATCH_THRESHOLD,
        "development": {
            item.patient_id: {
                "normalized_sha256": list(item.fingerprint.normalized_sha256),
                "sampled_sha256": list(item.fingerprint.sampled_sha256),
                "robust_descriptor_sha256": item.fingerprint.descriptor_sha256,
            }
            for item in development
        },
        "external": {
            patient_id: {
                "normalized_sha256": list(fingerprint.normalized_sha256),
                "sampled_sha256": list(fingerprint.sampled_sha256),
                "robust_descriptor_sha256": fingerprint.descriptor_sha256,
            }
            for patient_id, fingerprint in sorted(external_fingerprints.items())
        },
    }
    group_counts = {
        group: sum(str(row["disease_group"]) == group for row in inventory_rows)
        for group in ("glioma", "other_neoplasm")
    }
    label_sets = sorted(
        {str(row["source_label_set"]) for row in inventory_rows}
    )
    integrity_failures = [
        str(row["patient_id"])
        for row in inventory_rows
        if str(row["eligibility_status"]) != "eligible"
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "gate": "C",
        "external_cohort_version": EXTERNAL_COHORT_VERSION,
        "collection_doi": EXTERNAL_COLLECTION_DOI,
        "license": EXTERNAL_LICENSE,
        "patient_count": len(inventory_rows),
        "group_counts": group_counts,
        "complete_four_modality_and_label_count": len(inventory_rows)
        - len(integrity_failures),
        "integrity_failure_patient_ids": integrity_failures,
        "source_label_sets": label_sets,
        "label_mapping": "0->0;1->1(NCR/NET);2->2(ED);3->4(ET)",
        "primary_confirmatory_patient_count": group_counts["glioma"],
        "supportive_other_neoplasm_patient_count": group_counts[
            "other_neoplasm"
        ],
        "overlap": overlap_summary,
        "external_results_accessed": False,
        "model_inference_run": False,
        "gate_c_pass": (
            not integrity_failures
            and overlap_summary["zero_overlap_established"] is True
            and group_counts["glioma"] > 0
        ),
    }
    atomic_write_csv(inventory_output, inventory_rows)
    atomic_write_csv(overlap_output, overlap_rows)
    atomic_write_json(signature_output, signature_payload)
    atomic_write_json(summary_output, summary)
    append_jsonl(
        access_log,
        {
            "event": "external_identity_integrity_label_audit",
            "cohort": EXTERNAL_COHORT_VERSION,
            "patient_count": len(inventory_rows),
            "model_inference": False,
            "prediction_metrics_accessed": False,
            "outputs": [path.as_posix() for path in outputs[:-1]],
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--development-cache-root", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    parser.add_argument("--overlap-output", type=Path, required=True)
    parser.add_argument("--signature-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--access-log", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the external identity/integrity/label audit."""
    arguments = build_parser().parse_args(argv)
    summary = run_external_audit(
        external_root=arguments.external_root,
        metadata_csv=arguments.metadata_csv,
        development_cache_root=arguments.development_cache_root,
        canonical_manifest=arguments.canonical_manifest,
        inventory_output=arguments.inventory_output,
        overlap_output=arguments.overlap_output,
        signature_output=arguments.signature_output,
        summary_output=arguments.summary_output,
        access_log=arguments.access_log,
        workers=arguments.workers,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_c_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
