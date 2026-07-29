import json
from pathlib import Path

from bratsarticle.experiments.pilot_batch import (
    existing_run_is_reusable,
    pilot_run_id,
)


def test_all_pilot_run_id_is_deterministic() -> None:
    assert (
        pilot_run_id("architecture_unet", 20260729, "abcdef012345")
        == "gate8_architecture_unet_s20260729_abcdef01"
    )


def test_completed_run_reuse_requires_clean_matching_metadata(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    metadata = {
        "status": "completed",
        "repository_dirty": False,
        "tags": {
            "pilot_arm_id": "architecture_unet",
            "pilot_config_sha256": "hash",
        },
        "test_access": {"accessed": False},
    }
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    assert existing_run_is_reusable(
        run_directory,
        arm_id="architecture_unet",
        config_hash="hash",
    )
    metadata["repository_dirty"] = True
    (run_directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
    assert not existing_run_is_reusable(
        run_directory,
        arm_id="architecture_unet",
        config_hash="hash",
    )
