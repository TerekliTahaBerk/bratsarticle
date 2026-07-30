from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bratsarticle.data.splits import SplitIntegrityError, load_cv_fold_manifest
from bratsarticle.utils.hashing import file_digest


def _write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    canonical_path = tmp_path / "canonical.csv"
    pd.DataFrame(
        [
            {"subject_id": "A", "t1_relative_path": "A/A_t1.nii.gz"},
            {"subject_id": "B", "t1_relative_path": "B/B_t1.nii.gz"},
            {"subject_id": "C", "t1_relative_path": "C/C_t1.nii.gz"},
        ]
    ).to_csv(canonical_path, index=False)
    canonical_hash = file_digest(canonical_path)
    fold_path = tmp_path / "cv_fold_1.csv"
    pd.DataFrame(
        [
            {
                "subject_id": "A",
                "fold": 1,
                "role": "train",
                "canonical_manifest_sha256": canonical_hash,
            },
            {
                "subject_id": "B",
                "fold": 1,
                "role": "train",
                "canonical_manifest_sha256": canonical_hash,
            },
            {
                "subject_id": "C",
                "fold": 1,
                "role": "validation",
                "canonical_manifest_sha256": canonical_hash,
            },
        ]
    ).to_csv(fold_path, index=False)
    return canonical_path, fold_path


def test_load_cv_fold_manifest_joins_only_requested_role(tmp_path: Path) -> None:
    canonical_path, fold_path = _write_manifests(tmp_path)

    validation = load_cv_fold_manifest(
        fold_path,
        canonical_path,
        "validation",
    )

    assert validation["subject_id"].tolist() == ["C"]
    assert validation["cv_role"].tolist() == ["validation"]
    assert validation["cv_fold"].tolist() == [1]
    assert validation["t1_relative_path"].tolist() == ["C/C_t1.nii.gz"]


def test_load_cv_fold_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    canonical_path, fold_path = _write_manifests(tmp_path)
    fold = pd.read_csv(fold_path)
    fold["canonical_manifest_sha256"] = "0" * 64
    fold.to_csv(fold_path, index=False)

    with pytest.raises(SplitIntegrityError, match="hash does not match"):
        load_cv_fold_manifest(fold_path, canonical_path, "train")
