from pathlib import Path

import yaml

from bratsarticle.analysis.q1q2_figures import build_study_design_figure


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
