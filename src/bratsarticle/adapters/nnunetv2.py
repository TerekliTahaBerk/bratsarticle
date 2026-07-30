"""Audited BraTS 2020 data and split adapter for the official nnU-Net v2 CLI."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import (
    assert_existing_directory,
    assert_output_paths_safe,
)
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

DATASET_ID: Final[int] = 501
DATASET_NAME: Final[str] = "Dataset501_BraTS2020Q1Q2"
EXPECTED_SUBJECT_COUNT: Final[int] = 369
MODALITY_COLUMNS: Final[tuple[tuple[str, str], ...]] = (
    ("t1_relative_path", "0000"),
    ("t1ce_relative_path", "0001"),
    ("t2_relative_path", "0002"),
    ("flair_relative_path", "0003"),
)
RAW_TO_NNUNET: Final[Mapping[int, int]] = {0: 0, 1: 2, 2: 1, 4: 3}
NNUNET_TO_RAW: Final[Mapping[int, int]] = {
    value: key for key, value in RAW_TO_NNUNET.items()
}
MAIN_SEEDS: Final[tuple[int, ...]] = (
    20260730,
    20260731,
    20260732,
    20260733,
    20260734,
)


class NNUNetAdapterError(RuntimeError):
    """Raised when an nnU-Net conversion would violate a frozen contract."""


@dataclass(frozen=True)
class PreparedNNUNetDataset:
    """Paths and hashes for one idempotently prepared nnU-Net dataset."""

    dataset_directory: Path
    split_file: Path
    source_manifest_sha256: str
    split_sha256: str
    case_count: int
    reused_existing_dataset: bool


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _repository_state() -> tuple[str, bool]:
    repository_root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def dataset_json_payload(case_count: int = EXPECTED_SUBJECT_COUNT) -> dict[str, Any]:
    """Return the official BraTS region definition used by nnU-Net v2."""
    return {
        "channel_names": {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR",
        },
        "labels": {
            "background": 0,
            "whole tumor": [1, 2, 3],
            "tumor core": [2, 3],
            "enhancing tumor": [3],
        },
        "numTraining": case_count,
        "file_ending": ".nii",
        "regions_class_order": [1, 2, 3],
    }


def remap_label_array(
    array: np.ndarray[Any, np.dtype[np.generic]],
    mapping: Mapping[int, int],
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Map a label image exactly and reject undeclared source labels."""
    observed = {int(value) for value in np.unique(array)}
    undeclared = observed.difference(mapping)
    if undeclared:
        raise NNUNetAdapterError(
            f"Segmentation contains undeclared labels: {sorted(undeclared)}"
        )
    output = np.zeros(array.shape, dtype=np.uint8)
    for source, destination in mapping.items():
        output[array == source] = destination
    return output


def brats_to_nnunet_labels(
    array: np.ndarray[Any, np.dtype[np.generic]],
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Convert BraTS labels 0/1/2/4 to nnU-Net labels 0/2/1/3."""
    return remap_label_array(array, RAW_TO_NNUNET)


def nnunet_to_brats_labels(
    array: np.ndarray[Any, np.dtype[np.generic]],
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Convert exported nnU-Net labels 0/1/2/3 to BraTS labels 0/2/1/4."""
    return remap_label_array(array, NNUNET_TO_RAW)


def _load_canonical_manifest(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(manifest_path)
    required = {
        "dataset",
        "subject_id",
        "complete",
        "eligible",
        "seg_valid_label_set",
        "seg_relative_path",
        "seg_sha256",
        *(column for column, _ in MODALITY_COLUMNS),
        *(
            column.replace("_relative_path", "_sha256")
            for column, _ in MODALITY_COLUMNS
        ),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise NNUNetAdapterError(
            f"Canonical manifest is missing columns: {sorted(missing)}"
        )
    eligible = frame.loc[
        frame["dataset"].eq("brats2020")
        & frame["complete"].eq(True)
        & frame["eligible"].eq(True)
        & frame["seg_valid_label_set"].eq(True)
    ].copy()
    if len(eligible) != EXPECTED_SUBJECT_COUNT:
        raise NNUNetAdapterError(
            "Expected exactly "
            f"{EXPECTED_SUBJECT_COUNT} eligible BraTS 2020 subjects, "
            f"found {len(eligible)}"
        )
    if not eligible["subject_id"].is_unique:
        raise NNUNetAdapterError("Canonical manifest contains duplicate subject IDs")
    return eligible.sort_values("subject_id").reset_index(drop=True)


def _source_path(raw_root: Path, relative_path: str) -> Path:
    candidate = (raw_root / relative_path).resolve()
    if not candidate.is_relative_to(raw_root):
        raise NNUNetAdapterError(
            f"Manifest path escapes the authorized raw root: {relative_path}"
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Manifest source file is missing: {relative_path}")
    return candidate


def _validate_source_hash(
    source: Path,
    expected_sha256: str,
    *,
    verify_source_hashes: bool,
) -> None:
    if verify_source_hashes:
        observed = file_digest(source)
        if observed != expected_sha256:
            raise NNUNetAdapterError(
                "Source hash does not match the canonical manifest: "
                f"{source.name}, expected={expected_sha256}, observed={observed}"
            )


def _relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, start=destination.parent))


def _save_remapped_segmentation(
    source: Path,
    destination: Path,
) -> str:
    image = cast(nib.Nifti1Image, nib.load(str(source), mmap="r"))
    raw = np.asanyarray(image.dataobj)
    remapped = brats_to_nnunet_labels(raw)
    header: Any = image.header.copy()  # type: ignore[no-untyped-call]
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        remapped,
        image.affine,
        header=header,
    )
    qform, qcode = image.get_qform(coded=True)  # type: ignore[no-untyped-call]
    sform, scode = image.get_sform(coded=True)  # type: ignore[no-untyped-call]
    if qform is not None:
        output.set_qform(qform, int(qcode or 0))  # type: ignore[no-untyped-call]
    if sform is not None:
        output.set_sform(sform, int(scode or 0))  # type: ignore[no-untyped-call]

    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=".nii",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        nib.save(output, str(temporary))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_digest(destination)


def _write_dataset_staging(
    *,
    staging: Path,
    raw_root: Path,
    manifest: pd.DataFrame,
    manifest_sha256: str,
    verify_source_hashes: bool,
) -> dict[str, Any]:
    images = staging / "imagesTr"
    labels = staging / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir()
    cases: list[dict[str, Any]] = []

    for row in manifest.to_dict(orient="records"):
        subject_id = str(row["subject_id"])
        image_records: list[dict[str, str]] = []
        for relative_column, channel in MODALITY_COLUMNS:
            relative_path = str(row[relative_column])
            source = _source_path(raw_root, relative_path)
            expected_sha256 = str(
                row[relative_column.replace("_relative_path", "_sha256")]
            )
            _validate_source_hash(
                source,
                expected_sha256,
                verify_source_hashes=verify_source_hashes,
            )
            destination = images / f"{subject_id}_{channel}.nii"
            _relative_symlink(source, destination)
            image_records.append(
                {
                    "channel": channel,
                    "source_relative_path": relative_path,
                    "source_sha256": expected_sha256,
                }
            )

        segmentation_relative = str(row["seg_relative_path"])
        segmentation_source = _source_path(raw_root, segmentation_relative)
        segmentation_source_sha256 = str(row["seg_sha256"])
        _validate_source_hash(
            segmentation_source,
            segmentation_source_sha256,
            verify_source_hashes=verify_source_hashes,
        )
        segmentation_destination = labels / f"{subject_id}.nii"
        derived_sha256 = _save_remapped_segmentation(
            segmentation_source,
            segmentation_destination,
        )
        cases.append(
            {
                "subject_id": subject_id,
                "images": image_records,
                "segmentation": {
                    "source_relative_path": segmentation_relative,
                    "source_sha256": segmentation_source_sha256,
                    "derived_relative_path": (
                        f"labelsTr/{segmentation_destination.name}"
                    ),
                    "derived_sha256": derived_sha256,
                },
            }
        )

    dataset_payload = dataset_json_payload(len(manifest))
    atomic_write_json(staging / "dataset.json", dataset_payload)
    dataset_json_sha256 = file_digest(staging / "dataset.json")
    git_commit, repository_dirty = _repository_state()
    derivation = {
        "schema_version": 1,
        "adapter": "bratsarticle.adapters.nnunetv2",
        "dataset_name": DATASET_NAME,
        "case_count": len(manifest),
        "source_manifest_sha256": manifest_sha256,
        "source_hashes_recomputed": verify_source_hashes,
        "git_commit": git_commit,
        "repository_dirty_at_preparation": repository_dirty,
        "adapter_source_sha256": file_digest(Path(__file__).resolve()),
        "raw_to_nnunet_label_mapping": {
            str(key): value for key, value in RAW_TO_NNUNET.items()
        },
        "dataset_json_sha256": dataset_json_sha256,
        "cases": cases,
    }
    atomic_write_json(staging / "derivation_manifest.json", derivation)
    return derivation


def _validate_existing_dataset(
    *,
    dataset_directory: Path,
    raw_root: Path,
    manifest: pd.DataFrame,
    manifest_sha256: str,
    verify_source_hashes: bool,
) -> None:
    derivation_path = dataset_directory / "derivation_manifest.json"
    dataset_json_path = dataset_directory / "dataset.json"
    if not derivation_path.is_file() or not dataset_json_path.is_file():
        raise NNUNetAdapterError(
            f"Existing dataset is incomplete and will not be overwritten: "
            f"{dataset_directory}"
        )
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    if derivation.get("source_manifest_sha256") != manifest_sha256:
        raise NNUNetAdapterError(
            "Existing nnU-Net dataset was derived from a different manifest"
        )
    if derivation.get("case_count") != len(manifest):
        raise NNUNetAdapterError("Existing nnU-Net dataset has a wrong case count")
    observed_dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    if observed_dataset_json != dataset_json_payload(len(manifest)):
        raise NNUNetAdapterError("Existing nnU-Net dataset.json is not canonical")
    if derivation.get("dataset_json_sha256") != file_digest(dataset_json_path):
        raise NNUNetAdapterError("Existing nnU-Net dataset.json hash changed")

    expected_subjects = set(manifest["subject_id"].astype(str))
    observed_subjects = {
        path.name.removesuffix(".nii")
        for path in (dataset_directory / "labelsTr").glob("*.nii")
    }
    if observed_subjects != expected_subjects:
        raise NNUNetAdapterError("Existing nnU-Net label cohort is not exact")
    derived_case_records = {
        str(case["subject_id"]): case
        for case in cast(list[dict[str, Any]], derivation.get("cases", []))
    }
    if set(derived_case_records) != expected_subjects:
        raise NNUNetAdapterError("Existing derivation manifest cohort is not exact")
    for row in manifest.to_dict(orient="records"):
        subject_id = str(row["subject_id"])
        for relative_column, channel in MODALITY_COLUMNS:
            link = dataset_directory / "imagesTr" / f"{subject_id}_{channel}.nii"
            if not link.is_symlink():
                raise NNUNetAdapterError(f"Expected a source symlink: {link}")
            expected = _source_path(raw_root, str(row[relative_column]))
            if link.resolve() != expected:
                raise NNUNetAdapterError(f"Source symlink target changed: {link}")
            _validate_source_hash(
                expected,
                str(row[relative_column.replace("_relative_path", "_sha256")]),
                verify_source_hashes=verify_source_hashes,
            )
        segmentation_source = _source_path(
            raw_root,
            str(row["seg_relative_path"]),
        )
        _validate_source_hash(
            segmentation_source,
            str(row["seg_sha256"]),
            verify_source_hashes=verify_source_hashes,
        )
        derived_label = (
            dataset_directory / "labelsTr" / f"{subject_id}.nii"
        )
        expected_derived_hash = str(
            derived_case_records[subject_id]["segmentation"]["derived_sha256"]
        )
        observed_derived_hash = file_digest(derived_label)
        if observed_derived_hash != expected_derived_hash:
            raise NNUNetAdapterError(
                f"Derived label hash changed: {derived_label}"
            )


def build_splits_final(
    split_paths: Sequence[Path],
    expected_subject_ids: set[str],
) -> list[dict[str, list[str]]]:
    """Convert the five frozen one-indexed fold CSVs to nnU-Net fold order."""
    if len(split_paths) != 5:
        raise NNUNetAdapterError("Exactly five frozen fold manifests are required")
    result: list[dict[str, list[str]]] = []
    validation_appearances: dict[str, int] = {
        subject_id: 0 for subject_id in expected_subject_ids
    }
    canonical_hash: str | None = None

    for expected_fold, split_path in enumerate(split_paths, start=1):
        frame = pd.read_csv(split_path)
        required = {
            "subject_id",
            "fold",
            "role",
            "canonical_manifest_sha256",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise NNUNetAdapterError(
                f"{split_path} is missing columns: {sorted(missing)}"
            )
        if set(frame["subject_id"].astype(str)) != expected_subject_ids:
            raise NNUNetAdapterError(
                f"{split_path} does not contain the exact development cohort"
            )
        if not frame["subject_id"].is_unique:
            raise NNUNetAdapterError(f"{split_path} repeats subject IDs")
        if set(frame["fold"].astype(int)) != {expected_fold}:
            raise NNUNetAdapterError(
                f"{split_path} is not frozen fold {expected_fold}"
            )
        roles = set(frame["role"].astype(str))
        if roles != {"train", "validation"}:
            raise NNUNetAdapterError(
                f"{split_path} has invalid train/validation roles: {sorted(roles)}"
            )
        hashes = set(frame["canonical_manifest_sha256"].astype(str))
        if len(hashes) != 1:
            raise NNUNetAdapterError(
                f"{split_path} has inconsistent canonical manifest hashes"
            )
        fold_hash = next(iter(hashes))
        if canonical_hash is None:
            canonical_hash = fold_hash
        elif fold_hash != canonical_hash:
            raise NNUNetAdapterError("Frozen folds reference different manifests")

        train = sorted(
            frame.loc[frame["role"].eq("train"), "subject_id"].astype(str)
        )
        validation = sorted(
            frame.loc[
                frame["role"].eq("validation"), "subject_id"
            ].astype(str)
        )
        if set(train).intersection(validation):
            raise NNUNetAdapterError(f"{split_path} leaks patients across roles")
        for subject_id in validation:
            validation_appearances[subject_id] += 1
        result.append({"train": train, "val": validation})

    wrong_appearances = {
        subject_id: count
        for subject_id, count in validation_appearances.items()
        if count != 1
    }
    if wrong_appearances:
        raise NNUNetAdapterError(
            "Every patient must appear in exactly one validation fold; "
            f"violations={wrong_appearances}"
        )
    return result


def write_splits_final(
    destination: Path,
    splits: Sequence[Mapping[str, Sequence[str]]],
) -> str:
    """Write deterministic nnU-Net splits without overwriting a mismatch."""
    payload = [
        {"train": list(fold["train"]), "val": list(fold["val"])}
        for fold in splits
    ]
    canonical = _canonical_json(payload) + "\n"
    expected_hash = text_digest(canonical)
    if destination.exists():
        observed = json.loads(destination.read_text(encoding="utf-8"))
        if observed != payload:
            raise NNUNetAdapterError(
                f"Existing split file differs and will not be overwritten: "
                f"{destination}"
            )
        return file_digest(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, canonical)
    observed_hash = file_digest(destination)
    if observed_hash != expected_hash:
        raise NNUNetAdapterError("Deterministic split serialization hash mismatch")
    return observed_hash


def build_main_job_matrix() -> list[dict[str, Any]]:
    """Return all 50 official nnU-Net model/fold/seed training jobs."""
    jobs: list[dict[str, Any]] = []
    for configuration in ("2d", "3d_fullres"):
        model_id = (
            "nnunetv2_2d"
            if configuration == "2d"
            else "nnunetv2_3d_fullres"
        )
        for fold_one_indexed in range(1, 6):
            for seed in MAIN_SEEDS:
                trainer = f"nnUNetTrainerSeed{seed}"
                run_id = (
                    f"{model_id}__f{fold_one_indexed}__s{seed}"
                    "__convergence"
                )
                jobs.append(
                    {
                        "run_id": run_id,
                        "model_id": model_id,
                        "configuration": configuration,
                        "fold_one_indexed": fold_one_indexed,
                        "fold_nnunet_zero_indexed": fold_one_indexed - 1,
                        "seed": seed,
                        "trainer": trainer,
                        "device": "mps",
                        "environment": {
                            "PYTHONHASHSEED": str(seed),
                            "nnUNet_compile": "false",
                            "nnUNet_n_proc_DA": "0",
                            "nnUNet_extTrainer": "nnunet_ext",
                        },
                        "command": [
                            "nnUNetv2_train",
                            str(DATASET_ID),
                            configuration,
                            str(fold_one_indexed - 1),
                            "-tr",
                            trainer,
                            "-device",
                            "mps",
                        ],
                        "status": "not_started",
                    }
                )
    return jobs


def prepare_nnunet_dataset(
    *,
    raw_root: Path,
    canonical_manifest_path: Path,
    split_paths: Sequence[Path],
    nnunet_raw_root: Path,
    nnunet_preprocessed_root: Path,
    verify_source_hashes: bool = True,
) -> PreparedNNUNetDataset:
    """Prepare the official nnU-Net raw layout and exact frozen CV split."""
    git_commit, repository_dirty = _repository_state()
    if verify_source_hashes and repository_dirty:
        raise NNUNetAdapterError(
            "Reportable nnU-Net preparation requires a clean repository; "
            f"resolved commit={git_commit}"
        )
    raw_root = assert_existing_directory(raw_root, "BraTS 2020 raw root")
    canonical_manifest_path = canonical_manifest_path.resolve()
    manifest = _load_canonical_manifest(canonical_manifest_path)
    manifest_sha256 = file_digest(canonical_manifest_path)
    split_hashes = {
        str(value)
        for path in split_paths
        for value in pd.read_csv(path, usecols=["canonical_manifest_sha256"])[
            "canonical_manifest_sha256"
        ].astype(str)
    }
    if split_hashes != {manifest_sha256}:
        raise NNUNetAdapterError(
            "Frozen folds do not reference the selected canonical manifest"
        )

    dataset_directory = nnunet_raw_root.resolve() / DATASET_NAME
    split_file = nnunet_preprocessed_root.resolve() / DATASET_NAME / (
        "splits_final.json"
    )
    assert_output_paths_safe(
        [dataset_directory, split_file],
        [raw_root],
    )
    nnunet_raw_root.mkdir(parents=True, exist_ok=True)

    reused = dataset_directory.exists()
    if reused:
        _validate_existing_dataset(
            dataset_directory=dataset_directory,
            raw_root=raw_root,
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            verify_source_hashes=verify_source_hashes,
        )
    else:
        staging = Path(
            tempfile.mkdtemp(
                dir=nnunet_raw_root,
                prefix=f".{DATASET_NAME}.",
            )
        )
        try:
            _write_dataset_staging(
                staging=staging,
                raw_root=raw_root,
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                verify_source_hashes=verify_source_hashes,
            )
            staging.replace(dataset_directory)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    splits = build_splits_final(
        split_paths,
        set(manifest["subject_id"].astype(str)),
    )
    split_sha256 = write_splits_final(split_file, splits)
    return PreparedNNUNetDataset(
        dataset_directory=dataset_directory,
        split_file=split_file,
        source_manifest_sha256=manifest_sha256,
        split_sha256=split_sha256,
        case_count=len(manifest),
        reused_existing_dataset=reused,
    )


def convert_prediction_to_brats(
    source: Path,
    destination: Path,
    *,
    raw_roots: Sequence[Path],
) -> str:
    """Convert one nnU-Net exported segmentation back to BraTS label semantics."""
    if not raw_roots:
        raise NNUNetAdapterError(
            "At least one protected raw-data root must be declared"
        )
    if destination.exists():
        raise NNUNetAdapterError(
            f"Prediction destination already exists: {destination}"
        )
    assert_output_paths_safe([destination], raw_roots)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = cast(nib.Nifti1Image, nib.load(str(source), mmap="r"))
    converted = nnunet_to_brats_labels(np.asanyarray(image.dataobj))
    header: Any = image.header.copy()  # type: ignore[no-untyped-call]
    header.set_data_dtype(np.uint8)
    output = nib.Nifti1Image(  # type: ignore[no-untyped-call]
        converted,
        image.affine,
        header=header,
    )
    nib.save(output, str(destination))
    return file_digest(destination)
