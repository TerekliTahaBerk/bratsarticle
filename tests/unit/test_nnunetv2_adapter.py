from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

import bratsarticle.adapters.nnunetv2 as adapter
from bratsarticle.adapters.nnunetv2 import (
    NNUNetAdapterError,
    brats_to_nnunet_labels,
    build_main_job_matrix,
    build_splits_final,
    convert_prediction_to_brats,
    nnunet_to_brats_labels,
    prepare_nnunet_dataset,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.paths import PathSafetyError


@pytest.fixture(autouse=True)
def _clean_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "_repository_state",
        lambda: ("test-commit", False),
    )


def _save_nifti(path: Path, values: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(values, np.eye(4)), path)


def _tiny_cohort(tmp_path: Path, case_count: int = 5) -> tuple[Path, Path]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    rows: list[dict[str, object]] = []
    for index in range(case_count):
        subject_id = f"BraTS20_Training_{index + 1:03d}"
        subject_root = raw_root / subject_id
        subject_root.mkdir()
        row: dict[str, object] = {
            "dataset": "brats2020",
            "subject_id": subject_id,
            "complete": True,
            "eligible": True,
            "seg_valid_label_set": True,
        }
        for modality in ("t1", "t1ce", "t2", "flair"):
            path = subject_root / f"{subject_id}_{modality}.nii"
            _save_nifti(path, np.full((2, 2, 2), index, dtype=np.int16))
            relative = path.relative_to(raw_root).as_posix()
            row[f"{modality}_relative_path"] = relative
            row[f"{modality}_sha256"] = file_digest(path)
        segmentation = np.array(
            [[[0, 1], [2, 4]], [[4, 2], [1, 0]]],
            dtype=np.uint8,
        )
        segmentation_path = subject_root / f"{subject_id}_seg.nii"
        _save_nifti(segmentation_path, segmentation)
        row["seg_relative_path"] = segmentation_path.relative_to(
            raw_root
        ).as_posix()
        row["seg_sha256"] = file_digest(segmentation_path)
        rows.append(row)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return raw_root, manifest_path


def _five_fold_files(tmp_path: Path, manifest_path: Path) -> list[Path]:
    subject_ids = [f"BraTS20_Training_{index + 1:03d}" for index in range(5)]
    manifest_hash = file_digest(manifest_path)
    split_paths: list[Path] = []
    for fold in range(1, 6):
        rows = [
            {
                "subject_id": subject_id,
                "fold": fold,
                "role": "validation" if index == fold else "train",
                "canonical_manifest_sha256": manifest_hash,
            }
            for index, subject_id in enumerate(subject_ids, start=1)
        ]
        path = tmp_path / f"cv_fold_{fold}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        split_paths.append(path)
    return split_paths


def test_brats_label_mapping_is_exactly_reversible() -> None:
    raw = np.array([0, 1, 2, 4], dtype=np.uint8)
    encoded = brats_to_nnunet_labels(raw)

    assert encoded.tolist() == [0, 2, 1, 3]
    assert np.array_equal(nnunet_to_brats_labels(encoded), raw)


def test_label_mapping_rejects_an_undeclared_class() -> None:
    with pytest.raises(NNUNetAdapterError, match="undeclared labels"):
        brats_to_nnunet_labels(np.array([0, 3], dtype=np.uint8))


def test_frozen_split_conversion_preserves_patient_exclusivity(
    tmp_path: Path,
) -> None:
    _, manifest_path = _tiny_cohort(tmp_path)
    split_paths = _five_fold_files(tmp_path, manifest_path)
    subject_ids = {
        f"BraTS20_Training_{index + 1:03d}" for index in range(5)
    }

    splits = build_splits_final(split_paths, subject_ids)

    assert len(splits) == 5
    assert all(len(fold["train"]) == 4 for fold in splits)
    assert all(len(fold["val"]) == 1 for fold in splits)
    assert {
        subject_id for fold in splits for subject_id in fold["val"]
    } == subject_ids


def test_dataset_preparation_is_safe_audited_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "EXPECTED_SUBJECT_COUNT", 5)
    raw_root, manifest_path = _tiny_cohort(tmp_path)
    split_paths = _five_fold_files(tmp_path, manifest_path)
    nnunet_raw = tmp_path / "nnunet_raw"
    nnunet_preprocessed = tmp_path / "nnunet_preprocessed"

    first = prepare_nnunet_dataset(
        raw_root=raw_root,
        canonical_manifest_path=manifest_path,
        split_paths=split_paths,
        nnunet_raw_root=nnunet_raw,
        nnunet_preprocessed_root=nnunet_preprocessed,
    )
    second = prepare_nnunet_dataset(
        raw_root=raw_root,
        canonical_manifest_path=manifest_path,
        split_paths=split_paths,
        nnunet_raw_root=nnunet_raw,
        nnunet_preprocessed_root=nnunet_preprocessed,
    )

    assert first.case_count == 5
    assert first.reused_existing_dataset is False
    assert second.reused_existing_dataset is True
    dataset_json = json.loads(
        (first.dataset_directory / "dataset.json").read_text(encoding="utf-8")
    )
    assert dataset_json["numTraining"] == 5
    assert dataset_json["labels"]["whole tumor"] == [1, 2, 3]
    image_link = (
        first.dataset_directory
        / "imagesTr"
        / "BraTS20_Training_001_0000.nii"
    )
    assert image_link.is_symlink()
    derived = np.asanyarray(
        nib.load(
            first.dataset_directory
            / "labelsTr"
            / "BraTS20_Training_001.nii"
        ).dataobj
    )
    assert set(int(value) for value in np.unique(derived)) == {0, 1, 2, 3}
    assert json.loads(first.split_file.read_text(encoding="utf-8"))[0]["val"] == [
        "BraTS20_Training_001"
    ]


def test_dataset_preparation_rejects_output_below_raw_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "EXPECTED_SUBJECT_COUNT", 5)
    raw_root, manifest_path = _tiny_cohort(tmp_path)
    split_paths = _five_fold_files(tmp_path, manifest_path)

    with pytest.raises(PathSafetyError):
        prepare_nnunet_dataset(
            raw_root=raw_root,
            canonical_manifest_path=manifest_path,
            split_paths=split_paths,
            nnunet_raw_root=raw_root / "generated",
            nnunet_preprocessed_root=tmp_path / "nnunet_preprocessed",
            verify_source_hashes=False,
        )


def test_reportable_preparation_rejects_a_dirty_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setattr(
        adapter,
        "_repository_state",
        lambda: ("dirty-test-commit", True),
    )

    with pytest.raises(NNUNetAdapterError, match="clean repository"):
        prepare_nnunet_dataset(
            raw_root=raw_root,
            canonical_manifest_path=tmp_path / "unused.csv",
            split_paths=[],
            nnunet_raw_root=tmp_path / "nnunet_raw",
            nnunet_preprocessed_root=tmp_path / "nnunet_preprocessed",
        )


def test_dataset_reuse_rejects_a_modified_derived_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapter, "EXPECTED_SUBJECT_COUNT", 5)
    raw_root, manifest_path = _tiny_cohort(tmp_path)
    split_paths = _five_fold_files(tmp_path, manifest_path)
    nnunet_raw = tmp_path / "nnunet_raw"
    nnunet_preprocessed = tmp_path / "nnunet_preprocessed"
    prepared = prepare_nnunet_dataset(
        raw_root=raw_root,
        canonical_manifest_path=manifest_path,
        split_paths=split_paths,
        nnunet_raw_root=nnunet_raw,
        nnunet_preprocessed_root=nnunet_preprocessed,
    )
    modified = (
        prepared.dataset_directory
        / "labelsTr"
        / "BraTS20_Training_001.nii"
    )
    _save_nifti(modified, np.zeros((2, 2, 2), dtype=np.uint8))

    with pytest.raises(NNUNetAdapterError, match="Derived label hash changed"):
        prepare_nnunet_dataset(
            raw_root=raw_root,
            canonical_manifest_path=manifest_path,
            split_paths=split_paths,
            nnunet_raw_root=nnunet_raw,
            nnunet_preprocessed_root=nnunet_preprocessed,
        )


def test_prediction_conversion_restores_brats_semantics(tmp_path: Path) -> None:
    source_root = tmp_path / "nnunet_predictions"
    source_root.mkdir()
    source = source_root / "case.nii"
    _save_nifti(source, np.array([[[0, 1, 2, 3]]], dtype=np.uint8))
    destination = tmp_path / "brats_predictions" / "case.nii"
    protected_raw = tmp_path / "raw"
    protected_raw.mkdir()

    convert_prediction_to_brats(
        source,
        destination,
        raw_roots=[protected_raw],
    )

    converted = np.asanyarray(nib.load(destination).dataobj)
    assert converted.tolist() == [[[0, 2, 1, 4]]]


def test_prediction_conversion_requires_a_protected_raw_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "case.nii"
    _save_nifti(source, np.zeros((1, 1, 1), dtype=np.uint8))

    with pytest.raises(NNUNetAdapterError, match="raw-data root"):
        convert_prediction_to_brats(
            source,
            tmp_path / "converted" / "case.nii",
            raw_roots=[],
        )


def test_nnunet_job_matrix_uses_every_frozen_fold_and_seed() -> None:
    jobs = build_main_job_matrix()

    assert len(jobs) == 50
    assert len(
        {
            (job["model_id"], job["fold_one_indexed"], job["seed"])
            for job in jobs
        }
    ) == 50
    assert {job["configuration"] for job in jobs} == {"2d", "3d_fullres"}
    assert {job["fold_nnunet_zero_indexed"] for job in jobs} == set(range(5))
    assert all(job["environment"]["nnUNet_n_proc_DA"] == "0" for job in jobs)
    assert all(job["status"] == "not_started" for job in jobs)
