from pathlib import Path

import pytest

from bratsarticle.data.discovery import DiscoveryError, discover_subject


def _touch_roles(subject_dir: Path, subject_id: str) -> None:
    for role in ("t1", "t1ce", "t2", "flair"):
        (subject_dir / f"{subject_id}_{role}.nii").touch()


def test_discovers_controlled_segmentation_fallback(tmp_path: Path) -> None:
    subject_id = "BraTS20_Training_355"
    subject_dir = tmp_path / subject_id
    subject_dir.mkdir()
    _touch_roles(subject_dir, subject_id)
    fallback = subject_dir / "W39_1998.09.19_Segm.nii"
    fallback.touch()

    result = discover_subject("brats2020", subject_dir, grade=None)

    assert result.complete
    assert result.files["seg"] == fallback
    assert result.warnings == (
        "segmentation_filename_fallback:W39_1998.09.19_Segm.nii",
    )


def test_rejects_ambiguous_segmentation_fallback(tmp_path: Path) -> None:
    subject_id = "BraTS20_Training_999"
    subject_dir = tmp_path / subject_id
    subject_dir.mkdir()
    _touch_roles(subject_dir, subject_id)
    (subject_dir / "first_Segm.nii").touch()
    (subject_dir / "second_segmentation.nii").touch()

    with pytest.raises(DiscoveryError, match="Ambiguous segmentation fallback"):
        discover_subject("brats2020", subject_dir, grade=None)


def test_missing_role_is_explicit_warning(tmp_path: Path) -> None:
    subject_id = "BraTS20_Training_001"
    subject_dir = tmp_path / subject_id
    subject_dir.mkdir()
    _touch_roles(subject_dir, subject_id)

    result = discover_subject("brats2020", subject_dir, grade=None)

    assert not result.complete
    assert "missing_role:seg" in result.warnings
