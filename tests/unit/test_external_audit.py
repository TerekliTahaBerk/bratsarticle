from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np

from bratsarticle.data.external_audit import (
    ExternalSubject,
    audit_overlap,
    discover_external_subjects,
    fingerprint_normalized_channels,
    inspect_external_subject,
)


def _write_metadata(path: Path, patient_ids: list[tuple[str, str]]) -> None:
    fieldnames = [
        "patient_id",
        "disease_group",
        "source_sheet",
        "center",
        "scanner_information",
        "number_of_tumor_subregion_labels",
        "imaging_finding_comments",
        "neoplasm_category",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for patient_id, group in patient_ids:
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "disease_group": group,
                    "source_sheet": (
                        "95 Glioma" if group == "glioma" else "51 OtherNeoplasms"
                    ),
                    "center": "TEST",
                    "scanner_information": "Test Scanner",
                    "number_of_tumor_subregion_labels": "3",
                    "imaging_finding_comments": "",
                    "neoplasm_category": "",
                }
            )


def _touch_external_files(subject_dir: Path, patient_id: str) -> None:
    for suffix in ("t1n", "t1c", "t2w", "t2f", "seg"):
        (subject_dir / f"{patient_id}-{suffix}.nii.gz").touch()


def test_discovers_both_external_disease_groups(tmp_path: Path) -> None:
    data_root = tmp_path / "BraTS-Africa"
    patient_groups = [
        ("BraTS-SSA-00001-000", "glioma"),
        ("BraTS-SSA-00002-000", "other_neoplasm"),
    ]
    for patient_id, group in patient_groups:
        folder = "95_Glioma" if group == "glioma" else "51_OtherNeoplasms"
        subject_dir = data_root / folder / patient_id
        subject_dir.mkdir(parents=True)
        _touch_external_files(subject_dir, patient_id)
    metadata_csv = tmp_path / "metadata.csv"
    _write_metadata(metadata_csv, patient_groups)

    subjects = discover_external_subjects(data_root, metadata_csv)

    assert [subject.patient_id for subject in subjects] == [
        "BraTS-SSA-00001-000",
        "BraTS-SSA-00002-000",
    ]
    assert subjects[0].disease_group == "glioma"
    assert subjects[1].disease_group == "other_neoplasm"
    assert set(subjects[0].files) == {"t1", "t1ce", "t2", "flair", "seg"}


def test_fingerprint_is_deterministic_and_content_sensitive() -> None:
    base = np.zeros((8, 8, 4), dtype=np.float32)
    base[2:6, 2:6, 1:3] = np.linspace(1.0, 4.0, 32).reshape(4, 4, 2)
    channels = [base + index * (base != 0) for index in range(4)]

    first = fingerprint_normalized_channels(channels)
    second = fingerprint_normalized_channels([array.copy() for array in channels])
    changed_channels = [array.copy() for array in channels]
    changed_channels[0][3, 3, 2] += 0.5
    changed = fingerprint_normalized_channels(changed_channels)

    assert first.normalized_sha256 == second.normalized_sha256
    assert first.sampled_sha256 == second.sampled_sha256
    assert np.array_equal(first.descriptor, second.descriptor)
    assert first.normalized_sha256 != changed.normalized_sha256
    assert first.descriptor_sha256 != changed.descriptor_sha256


def test_external_label_three_maps_to_brats2020_et(tmp_path: Path) -> None:
    patient_id = "BraTS-SSA-00001-000"
    subject_dir = tmp_path / "95_Glioma" / patient_id
    subject_dir.mkdir(parents=True)
    affine = np.eye(4, dtype=np.float64)
    files: dict[str, Path] = {}
    for role, suffix in (
        ("t1", "t1n"),
        ("t1ce", "t1c"),
        ("t2", "t2w"),
        ("flair", "t2f"),
    ):
        values = np.zeros((8, 8, 4), dtype=np.float32)
        values[1:7, 1:7, :] = np.arange(144, dtype=np.float32).reshape(6, 6, 4)
        path = subject_dir / f"{patient_id}-{suffix}.nii.gz"
        nib.save(nib.Nifti1Image(values, affine), path)
        files[role] = path
    segmentation = np.zeros((8, 8, 4), dtype=np.uint8)
    segmentation[1:3, 1:3, 1] = 1
    segmentation[3:5, 3:5, 1] = 2
    segmentation[5:7, 5:7, 1] = 3
    segmentation_path = subject_dir / f"{patient_id}-seg.nii.gz"
    nib.save(nib.Nifti1Image(segmentation, affine), segmentation_path)
    files["seg"] = segmentation_path
    subject = ExternalSubject(
        patient_id=patient_id,
        disease_group="glioma",
        source_sheet="95 Glioma",
        center="TEST",
        scanner_information="Test Scanner",
        label_count_reported="3",
        imaging_comments="",
        neoplasm_category="",
        subject_dir=subject_dir,
        files=files,
    )

    row, _ = inspect_external_subject(subject, tmp_path)

    assert row["source_label_set"] == "[0,1,2,3]"
    assert row["mapped_brats2020_label_set"] == "[0,1,2,4]"
    assert row["wt_voxel_count"] == 12
    assert row["tc_voxel_count"] == 8
    assert row["et_voxel_count"] == 4
    assert row["eligibility_status"] == "eligible"


def test_overlap_audit_flags_exact_normalized_match(tmp_path: Path) -> None:
    canonical_manifest = tmp_path / "canonical.csv"
    fieldnames = [
        "subject_id",
        "t1_sha256",
        "t1ce_sha256",
        "t2_sha256",
        "flair_sha256",
        "seg_sha256",
    ]
    with canonical_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(369):
            writer.writerow(
                {
                    "subject_id": f"BraTS20_Training_{index + 1:03d}",
                    "t1_sha256": f"t1-{index}",
                    "t1ce_sha256": f"t1ce-{index}",
                    "t2_sha256": f"t2-{index}",
                    "flair_sha256": f"flair-{index}",
                    "seg_sha256": f"seg-{index}",
                }
            )
    base = np.zeros((8, 8, 4), dtype=np.float32)
    base[1:7, 1:7, :] = np.arange(144, dtype=np.float32).reshape(6, 6, 4)
    fingerprint = fingerprint_normalized_channels([base] * 4)
    development = [
        type(
            "Development",
            (),
            {
                "patient_id": "BraTS20_Training_001",
                "fingerprint": fingerprint,
            },
        )()
    ]
    inventory = [
        {
            "patient_id": "BraTS-SSA-00001-000",
            "t1_sha256": "external-t1",
            "t1ce_sha256": "external-t1ce",
            "t2_sha256": "external-t2",
            "flair_sha256": "external-flair",
            "label_sha256": "external-seg",
        }
    ]

    rows, summary = audit_overlap(
        inventory_rows=inventory,
        external_fingerprints={"BraTS-SSA-00001-000": fingerprint},
        development_fingerprints=development,
        canonical_manifest=canonical_manifest,
        evidence_artifact="evidence.json",
    )

    assert rows[0]["image_content_signature_match"] is True
    assert rows[0]["decision"] == "manual_review_required_possible_overlap"
    assert summary["zero_overlap_established"] is False
