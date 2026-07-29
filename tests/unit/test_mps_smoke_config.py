from pathlib import Path

from omegaconf import OmegaConf

from bratsarticle.experiments.pilots import write_mps_diagnostic_config


def test_mps_smoke_config_is_bounded_and_nonreportable(tmp_path: Path) -> None:
    output = write_mps_diagnostic_config(
        Path("configs/pilots/gate8.yaml"),
        tmp_path / "smoke.yaml",
    )
    root = OmegaConf.load(output)

    assert root.pilot.status == "diagnostic_only_not_for_selection"
    assert root.pilot.budget.maximum_optimizer_steps == 10
    assert root.pilot.budget.validation_frequency_optimizer_steps == 10
    assert root.pilot.budget.minimum_completed_validation_checks == 1
    assert root.pilot.data.training_workers == 0
    assert root.pilot.data.validation_workers == 0
    assert root.pilot.data.training_memory_subjects == 1
    assert root.pilot.data.validation_memory_subjects == 1
