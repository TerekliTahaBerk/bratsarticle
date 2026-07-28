from pathlib import Path

import pytest

from bratsarticle.utils.paths import PathSafetyError, assert_output_paths_safe


def test_rejects_output_inside_raw_root(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()

    with pytest.raises(PathSafetyError):
        assert_output_paths_safe(
            [raw_root / "generated.csv"],
            [raw_root],
        )


def test_allows_sibling_output(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    output_root = tmp_path / "reports"
    raw_root.mkdir()

    assert_output_paths_safe(
        [output_root / "generated.csv"],
        [raw_root],
    )
