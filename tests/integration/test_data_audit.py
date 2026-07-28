from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.data.audit import AuditSettings, run_audit


def _write_subject(subject_dir: Path, subject_id: str) -> None:
    subject_dir.mkdir(parents=True)
    affine = np.eye(4, dtype=np.float64)
    image = np.zeros((8, 8, 4), dtype=np.float32)
    image[2:6, 2:6, 1:3] = 1.0
    segmentation = np.zeros((8, 8, 4), dtype=np.uint8)
    segmentation[2:6, 2:6, 1:3] = 2
    segmentation[3:5, 3:5, 1:3] = 1
    segmentation[4, 4, 2] = 4
    for role in ("t1", "t1ce", "t2", "flair"):
        nib.save(
            nib.Nifti1Image(image, affine),
            subject_dir / f"{subject_id}_{role}.nii",
        )
    nib.save(
        nib.Nifti1Image(segmentation, affine),
        subject_dir / f"{subject_id}_seg.nii",
    )


def test_synthetic_end_to_end_audit_does_not_modify_raw_data(
    tmp_path: Path,
) -> None:
    brats2020 = tmp_path / "raw2020"
    brats2019 = tmp_path / "raw2019"
    output_root = tmp_path / "generated"
    brats2020.mkdir()
    (brats2019 / "HGG").mkdir(parents=True)
    (brats2019 / "LGG").mkdir()

    subject2020 = "BraTS20_Training_001"
    subject2019 = "BraTS19_SYNTHETIC_001"
    _write_subject(brats2020 / subject2020, subject2020)
    _write_subject(brats2019 / "HGG" / subject2019, subject2019)
    pd.DataFrame(
        [
            {
                "Grade": "HGG",
                "BraTS_2019_subject_ID": subject2019,
                "BraTS_2020_subject_ID": subject2020,
            }
        ]
    ).to_csv(brats2020 / "name_mapping.csv", index=False)

    raw_files = sorted([*brats2020.rglob("*"), *brats2019.rglob("*")])
    raw_mtimes_before = {
        path: path.stat().st_mtime_ns for path in raw_files if path.is_file()
    }

    summary = run_audit(
        AuditSettings(
            brats2020_root=brats2020,
            brats2019_root=brats2019,
            output_root=output_root,
            workers=1,
            hash_algorithm="sha256",
            limit_subjects=None,
            compare_content_on_hash_mismatch=True,
            fail_on_invalid_label_set=True,
            expected_segmentation_labels=frozenset({0, 1, 2, 4}),
        )
    )

    raw_mtimes_after = {
        path: path.stat().st_mtime_ns for path in raw_files if path.is_file()
    }
    assert raw_mtimes_after == raw_mtimes_before
    assert summary["brats2020"]["subject_count"] == 1
    assert summary["brats2019"]["subject_count"] == 1
    assert summary["integrity"]["file_error_count"] == 0
    assert (output_root / "reports/data_audit_summary.json").is_file()

    canonical = pd.read_csv(
        output_root / "manifests/canonical/brats2020_canonical_manifest.csv"
    )
    assert canonical.loc[0, "subject_id"] == subject2020
    assert bool(canonical.loc[0, "eligible"])
    assert not str(canonical.loc[0, "t1_relative_path"]).startswith("/")
