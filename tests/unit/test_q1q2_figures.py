from pathlib import Path

import yaml

from bratsarticle.analysis.q1q2_figures import build_study_design_figure
from bratsarticle.utils.hashing import file_digest


def test_study_design_figure_uses_frozen_counts(tmp_path: Path) -> None:
    source = yaml.safe_load(
        Path("configs/q1q2_v2/figure_execution.yaml").read_text(encoding="utf-8")
    )
    source["outputs"]["directory"] = (tmp_path / "figures").as_posix()
    config = tmp_path / "figures.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    report = build_study_design_figure(config_path=config)

    assert len(report) == 2
    assert all(Path(path).is_file() for path in report)


def test_study_design_figure_is_byte_reproducible(tmp_path: Path) -> None:
    source = yaml.safe_load(
        Path("configs/q1q2_v2/figure_execution.yaml").read_text(encoding="utf-8")
    )
    source["outputs"]["directory"] = (tmp_path / "figures").as_posix()
    config = tmp_path / "figures.yaml"
    config.write_text(yaml.safe_dump(source), encoding="utf-8")

    first = build_study_design_figure(config_path=config)
    first_hashes = {path: file_digest(Path(path)) for path in first}
    second = build_study_design_figure(config_path=config)

    assert second == first
    assert {path: file_digest(Path(path)) for path in second} == first_hashes
