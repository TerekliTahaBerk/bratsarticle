import json
from collections import Counter
from pathlib import Path

import pytest

from bratsarticle.utils.hashing import file_digest


def test_gate10_freeze_pins_splits_and_all_checkpoint_seeds() -> None:
    checkpoint_manifest_path = Path("reports/gate10_checkpoint_manifest.json")
    if not checkpoint_manifest_path.is_file():
        pytest.skip("Gate 10 freeze artifacts are unavailable")

    split_metadata_path = Path("splits/frozen/split_metadata.json")
    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    assert split_metadata["status"] == "pass"
    assert split_metadata["frozen"] is True
    assert split_metadata["counts"] == {"train": 258, "validation": 37, "test": 74}
    for split in ("train", "validation", "test"):
        provisional = Path("splits/provisional") / f"{split}.csv"
        frozen = Path("splits/frozen") / f"{split}.csv"
        assert file_digest(provisional) == file_digest(frozen)
        assert file_digest(frozen) == split_metadata["manifest_sha256"][split]

    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["status"] == "frozen"
    assert checkpoint_manifest["checkpoint_count"] == 13
    assert checkpoint_manifest["seed_ensemble"] is False
    checkpoints = checkpoint_manifest["checkpoints"]
    counts = Counter(entry["candidate_id"] for entry in checkpoints)
    assert counts == {
        "unet_reference": 3,
        "bunet": 5,
        "unet_res": 5,
    }
    assert len({entry["checkpoint_sha256"] for entry in checkpoints}) == 13
    for entry in checkpoints:
        checkpoint = Path(entry["checkpoint_path"])
        if not checkpoint.is_file():
            pytest.skip("Local Gate 9 checkpoints are unavailable")
        assert file_digest(checkpoint) == entry["checkpoint_sha256"]
        assert checkpoint.stat().st_size == entry["checkpoint_size_bytes"]
        assert (
            file_digest(Path(entry["model_config_path"]))
            == entry["model_config_sha256"]
        )

    analysis = json.loads(
        Path("reports/gate10_analysis_freeze.json").read_text(encoding="utf-8")
    )
    assert analysis["status"] == "frozen"
    assert analysis["internal_test_accessed"] is False
    assert analysis["checkpoint_count"] == 13
    assert (
        file_digest(checkpoint_manifest_path)
        == analysis["checkpoint_manifest_sha256"]
    )
    assert file_digest(split_metadata_path) == analysis["frozen_split_metadata_sha256"]
