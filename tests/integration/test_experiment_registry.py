import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from bratsarticle.experiments.registry import (
    ExperimentRegistry,
    ResourceTracker,
    RunDescriptor,
)


def _descriptor(manifest: Path, run_id: str = "unit-run") -> RunDescriptor:
    return RunDescriptor(
        run_id=run_id,
        seed=20260729,
        model="unet",
        loss="cross_entropy_plus_soft_dice",
        optimizer="adamw",
        scheduler="linear_warmup_cosine_decay",
        parameter_count=1234,
        input_specification=(1, 4, 240, 240),
        data_manifest_path=manifest,
        split_hashes={"train": "train-hash", "validation": "validation-hash"},
    )


def test_registry_creates_required_artifact_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id\nsubject-1\n", encoding="utf-8")
    registry = ExperimentRegistry(
        artifact_root=tmp_path / "runs",
        descriptor=_descriptor(manifest),
        config_path=Path("configs/protocols/compute_matched.yaml"),
    )
    registry.log_epoch(
        {"optimizer_step": 10, "validation_patient_mean_regional_dice": 0.5}
    )
    registry.write_validation_cases(
        [{"patient_id": "subject-1", "mean_regional_dice": 0.5}]
    )
    tracker = ResourceTracker(torch.device("cpu"))
    profile = tracker.snapshot()
    registry.finalize(
        status="completed",
        resource_profile=profile,
        best_validation_checkpoint="checkpoints/best.pt",
    )

    expected = {
        "config.yaml",
        "metadata.json",
        "metrics_per_epoch.jsonl",
        "validation_per_case.csv",
        "checkpoints",
        "resource_profile.json",
        "logs",
    }
    assert {path.name for path in registry.run_directory.iterdir()} == expected
    metadata = json.loads(
        (registry.run_directory / "metadata.json").read_text(encoding="utf-8")
    )
    cases = pd.read_csv(registry.run_directory / "validation_per_case.csv")
    assert metadata["status"] == "completed"
    assert metadata["run_id"] == "unit-run"
    assert metadata["config_sha256"]
    assert metadata["data_manifest_sha256"]
    assert metadata["repository_dirty"] in {True, False}
    assert metadata["gpu_hours"] == 0.0
    assert metadata["test_access"]["accessed"] is False
    assert cases.loc[0, "patient_id"] == "subject-1"


def test_registry_rejects_unsafe_or_duplicate_run_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id\nsubject-1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsafe run_id"):
        _descriptor(manifest, "../escape")
    descriptor = _descriptor(manifest)
    ExperimentRegistry(
        artifact_root=tmp_path / "runs",
        descriptor=descriptor,
        config_path=Path("configs/protocols/compute_matched.yaml"),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        ExperimentRegistry(
            artifact_root=tmp_path / "runs",
            descriptor=descriptor,
            config_path=Path("configs/protocols/compute_matched.yaml"),
        )


def test_failed_run_requires_and_records_error_trace(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("subject_id\nsubject-1\n", encoding="utf-8")
    registry = ExperimentRegistry(
        artifact_root=tmp_path / "runs",
        descriptor=_descriptor(manifest, "failed-run"),
        config_path=Path("configs/protocols/compute_matched.yaml"),
    )
    profile = ResourceTracker(torch.device("cpu")).snapshot()

    with pytest.raises(ValueError, match="requires an error trace"):
        registry.finalize(status="failed", resource_profile=profile)
    registry.finalize(
        status="failed",
        resource_profile=profile,
        error_trace="RuntimeError: controlled test failure",
    )

    metadata = json.loads(
        (registry.run_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "failed"
    assert metadata["error_trace"] == "RuntimeError: controlled test failure"
