from pathlib import Path

import pandas as pd
import pytest

from bratsarticle.data.splits import (
    SplitIntegrityError,
    add_stratification_features,
    assert_no_duplicate_file_hashes,
    assert_no_duplicate_image_signatures,
    load_development_manifest,
    load_internal_test_manifest,
)


def _canonical_frame(count: int = 8) -> pd.DataFrame:
    rows = []
    for index in range(count):
        rows.append(
            {
                "subject_id": f"subject_{index:03d}",
                "grade": "HGG" if index % 2 == 0 else "LGG",
                "eligible": True,
                "wt_voxel_count": 100 + index * 10,
                "tc_voxel_count": 50 + index * 5,
                "et_voxel_count": 0 if index % 3 == 0 else 10 + index,
                "voxel_volume_mm3": 1.0,
                "t1_sha256": f"t1_{index}",
                "t1ce_sha256": f"t1ce_{index}",
                "t2_sha256": f"t2_{index}",
                "flair_sha256": f"flair_{index}",
            }
        )
    return pd.DataFrame(rows)


def test_stratification_features_are_patient_level() -> None:
    enriched = add_stratification_features(_canonical_frame())

    assert enriched["subject_id"].is_unique
    assert set(enriched["wt_volume_quartile"].astype(str)) == {
        "Q1",
        "Q2",
        "Q3",
        "Q4",
    }
    assert enriched.loc[0, "et_volume_quartile"] == "absent"


def test_duplicate_image_signatures_are_rejected() -> None:
    frame = _canonical_frame()
    for column in ("t1_sha256", "t1ce_sha256", "t2_sha256", "flair_sha256"):
        frame.loc[1, column] = frame.loc[0, column]

    with pytest.raises(SplitIntegrityError, match="Exact image signatures"):
        assert_no_duplicate_image_signatures(frame)


def test_duplicate_same_role_file_hashes_are_rejected() -> None:
    frame = _canonical_frame()
    frame.loc[1, "t2_sha256"] = frame.loc[0, "t2_sha256"]

    with pytest.raises(SplitIntegrityError, match="Exact file hashes"):
        assert_no_duplicate_file_hashes(frame)


def test_development_loader_cannot_open_test(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame([{"subject_id": "train"}]).to_csv(split_dir / "train.csv", index=False)

    assert load_development_manifest(split_dir, "train").iloc[0]["subject_id"] == (
        "train"
    )
    with pytest.raises(ValueError, match="train/validation"):
        load_development_manifest(split_dir, "test")  # type: ignore[arg-type]


def test_internal_test_access_requires_flag_and_is_logged(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    pd.DataFrame([{"subject_id": "test"}]).to_csv(split_dir / "test.csv", index=False)
    audit_log = tmp_path / "audit" / "test_access.jsonl"

    with pytest.raises(PermissionError, match="allow-test-evaluation"):
        load_internal_test_manifest(
            split_dir,
            allow_test_evaluation=False,
            purpose="unit test",
            audit_log=audit_log,
        )
    result = load_internal_test_manifest(
        split_dir,
        allow_test_evaluation=True,
        purpose="guard behavior unit test",
        audit_log=audit_log,
    )

    assert result.iloc[0]["subject_id"] == "test"
    log_text = audit_log.read_text(encoding="utf-8")
    assert "internal_test_manifest_access" in log_text
    assert "guard behavior unit test" in log_text
