from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from bratsarticle.experiments.q1q2_external_queue import (
    _aggregate,
    _config,
    _exclusive_gate_h_lock,
    _open_session,
)
from bratsarticle.utils.hashing import file_digest


def test_gate_h_protocol_keeps_external_single_opening_frozen() -> None:
    config = _config(Path("configs/q1q2_v2/gate_h_external.yaml"))

    assert config["status"] == "frozen_before_external_results"
    assert config["expected"]["main_checkpoint_count"] == 300
    assert config["expected"]["checkpoints_per_model"] == 25
    assert config["expected"]["confirmatory_patients"] == 95
    assert config["expected"]["supportive_patients"] == 51
    assert config["guards"]["single_external_session"] is True
    assert config["guards"]["retuning"] == "prohibited"


def test_external_session_requires_explicit_authorization(tmp_path: Path) -> None:
    config_path = tmp_path / "gate_h.yaml"
    config_path.write_text("status: frozen_before_external_results\n")
    config: dict[str, Any] = {
        "gate_g_analysis_freeze": tmp_path / "freeze.json",
        "external_inventory": tmp_path / "inventory.csv",
        "artifacts": {
            "runtime": tmp_path / "runtime.json",
            "access_log": tmp_path / "access.jsonl",
        },
    }

    with pytest.raises(
        PermissionError,
        match="allow-frozen-external-inference",
    ):
        _open_session(
            config_path=config_path,
            config=config,
            freeze={},
            external_root=tmp_path,
            allow_frozen_external_inference=False,
        )


def test_gate_h_lock_is_atomic_and_mutually_exclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with _exclusive_gate_h_lock():
        lock = Path("artifacts/q1q2_v2/queue_runtime/gate_h_external.lock")
        assert lock.is_file()
        with pytest.raises(RuntimeError, match="lock already exists"):
            with _exclusive_gate_h_lock():
                pass

    assert not lock.exists()


def test_external_aggregation_averages_25_frozen_checkpoint_replicates(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "runs"
    runs: list[dict[str, Any]] = []
    for replicate in range(25):
        run_id = f"run_{replicate:02d}"
        runs.append({"run_id": run_id})
        run_directory = artifact_root / run_id
        run_directory.mkdir(parents=True)
        metrics = pd.DataFrame(
            [
                {
                    "model_id": "unet_small",
                    "patient_id": patient,
                    "evaluation_stage": "raw",
                    "cohort_role": "external_confirmatory",
                    "disease_group": "glioma",
                    "institution": "site",
                    "scanner_vendor": "vendor",
                    "scanner_model": "model",
                    "field_strength_t": 1.5,
                    "training_fold": replicate % 5 + 1,
                    "training_seed": 20260730 + replicate % 5,
                    "spacing_axis0_mm": 1.0,
                    "spacing_axis1_mm": 1.0,
                    "spacing_axis2_mm": 1.0,
                    "mean_regional_dice": replicate / 24.0,
                    "wt_dice": replicate / 24.0,
                }
                for patient in ("patient_a", "patient_b")
            ]
        )
        metrics_path = run_directory / "patient_metrics.csv"
        metrics.to_csv(metrics_path, index=False)
        (run_directory / "runtime.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "patient_metrics": metrics_path.as_posix(),
                    "patient_metrics_sha256": file_digest(metrics_path),
                }
            ),
            encoding="utf-8",
        )

    report = _aggregate(
        runs=runs,
        artifact_root=artifact_root,
        checkpoint_output=tmp_path / "checkpoint.csv",
        model_output=tmp_path / "model.csv",
        expected={"checkpoints_per_model": 25},
    )

    model = pd.read_csv(report["model_patient_metrics"])
    assert report["failed_checkpoint_count"] == 0
    assert report["all_model_patient_groups_have_25_checkpoints"] is True
    assert set(model["valid_checkpoint_count"]) == {25}
    assert set(model["mean_regional_dice"].round(12)) == {0.5}
