from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bratsarticle.experiments.q1q2_prediction_retention import (
    finalize_model_predictions,
    prepare_checkpoint_stager,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def test_model_prediction_finalization_uses_nested_strict_majority(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs"
    runs = [
        {
            "run_id": f"run_{index}",
            "model_id": "model",
            "fold": index + 1,
            "seed": 20260730 + index,
            "best_checkpoint_sha256": f"checkpoint-{index}",
        }
        for index in range(3)
    ]
    predictions = [
        np.asarray([[[0, 4], [2, 1]]], dtype=np.uint8),
        np.asarray([[[2, 4], [2, 0]]], dtype=np.uint8),
        np.asarray([[[2, 1], [0, 1]]], dtype=np.uint8),
    ]
    for run, prediction in zip(runs, predictions, strict=True):
        run_directory = run_root / str(run["run_id"])
        stager = prepare_checkpoint_stager(
            run_directory / "checkpoint_predictions"
        )
        stager.add("patient", prediction)
        manifest = stager.seal(run=run, expected_patient_count=1)
        atomic_write_json(
            run_directory / "runtime.json",
            {
                "status": "completed",
                "checkpoint_prediction_manifest": manifest.as_posix(),
                "checkpoint_prediction_manifest_sha256": file_digest(manifest),
            },
        )

    manifest = finalize_model_predictions(
        model_id="model",
        runs=runs,
        run_artifact_root=run_root,
        output_root=tmp_path / "model_predictions",
        expected_patient_count=1,
        required_replicates=3,
    )

    assert manifest is not None
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert stored["majority_threshold"] == 2
    with np.load(manifest.parent / "patient.npz", allow_pickle=False) as payload:
        ensemble = payload["prediction_label"]
    assert np.array_equal(
        ensemble,
        np.asarray([[[2, 4], [2, 1]]], dtype=np.uint8),
    )
    assert not any(
        (run_root / str(run["run_id"]) / "checkpoint_predictions").exists()
        for run in runs
    )
