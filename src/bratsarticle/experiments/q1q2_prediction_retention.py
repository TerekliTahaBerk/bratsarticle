"""Transactional prediction retention for the one-time Gate H session."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json

_ALLOWED_LABELS = {0, 1, 2, 4}


def _prediction_path(directory: Path, patient_id: str) -> Path:
    if not patient_id or "/" in patient_id or "\\" in patient_id:
        raise ValueError("Invalid anonymized patient identifier")
    return directory / f"{patient_id}.npz"


def _load_prediction(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction_label"], dtype=np.uint8)
    if prediction.ndim != 3:
        raise ValueError(f"Retained prediction is not three-dimensional: {path}")
    if not set(int(value) for value in np.unique(prediction)).issubset(
        _ALLOWED_LABELS
    ):
        raise ValueError(f"Retained prediction has invalid BraTS labels: {path}")
    return prediction


class CheckpointPredictionStager:
    """Write one checkpoint's predictions before atomically sealing a manifest."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self._patients: dict[str, dict[str, Any]] = {}

    def add(self, patient_id: str, prediction: np.ndarray) -> None:
        array = np.asarray(prediction, dtype=np.uint8)
        if array.ndim != 3:
            raise ValueError("Checkpoint prediction must be three-dimensional")
        if not set(int(value) for value in np.unique(array)).issubset(
            _ALLOWED_LABELS
        ):
            raise ValueError("Checkpoint prediction has invalid BraTS labels")
        destination = _prediction_path(self.directory, patient_id)
        if destination.exists() or patient_id in self._patients:
            raise ValueError(f"Duplicate checkpoint prediction: {patient_id}")
        descriptor, raw_temporary = tempfile.mkstemp(
            dir=self.directory,
            prefix=f".{patient_id}.",
            suffix=".npz.tmp",
        )
        os.close(descriptor)
        temporary = Path(raw_temporary)
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, prediction_label=array)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        self._patients[patient_id] = {
            "path": destination.as_posix(),
            "sha256": file_digest(destination),
            "shape": list(array.shape),
        }

    def seal(
        self,
        *,
        run: Mapping[str, Any],
        expected_patient_count: int,
    ) -> Path:
        if len(self._patients) != expected_patient_count:
            raise RuntimeError("Checkpoint prediction staging is incomplete")
        manifest_path = self.directory / "manifest.json"
        atomic_write_json(
            manifest_path,
            {
                "schema_version": 1,
                "status": "complete",
                "run_id": str(run["run_id"]),
                "model_id": str(run["model_id"]),
                "fold": int(run["fold"]),
                "seed": int(run["seed"]),
                "checkpoint_sha256": str(run["best_checkpoint_sha256"]),
                "patient_count": len(self._patients),
                "patients": dict(sorted(self._patients.items())),
            },
        )
        return manifest_path


def prepare_checkpoint_stager(directory: Path) -> CheckpointPredictionStager:
    """Start or operationally restart an unsealed checkpoint staging directory."""
    if directory.exists():
        manifest = directory / "manifest.json"
        if manifest.is_file():
            raise FileExistsError(
                "A sealed prediction staging directory already exists"
            )
        shutil.rmtree(directory)
    return CheckpointPredictionStager(directory)


def validate_checkpoint_prediction_manifest(
    manifest_path: Path,
    *,
    run: Mapping[str, Any],
    expected_patient_count: int,
) -> dict[str, Any]:
    """Verify every staged prediction and its frozen checkpoint identity."""
    manifest = cast(
        dict[str, Any],
        json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    if (
        manifest.get("status") != "complete"
        or manifest.get("run_id") != run["run_id"]
        or manifest.get("model_id") != run["model_id"]
        or int(manifest.get("fold", -1)) != int(run["fold"])
        or int(manifest.get("seed", -1)) != int(run["seed"])
        or manifest.get("checkpoint_sha256") != run["best_checkpoint_sha256"]
        or int(manifest.get("patient_count", -1)) != expected_patient_count
    ):
        raise RuntimeError("Checkpoint prediction manifest identity differs")
    patients = cast(dict[str, dict[str, Any]], manifest["patients"])
    if len(patients) != expected_patient_count:
        raise RuntimeError("Checkpoint prediction manifest patient count differs")
    for patient_id, entry in patients.items():
        path = Path(str(entry["path"]))
        if (
            path != _prediction_path(manifest_path.parent, patient_id)
            or not path.is_file()
            or file_digest(path) != entry["sha256"]
        ):
            raise RuntimeError("Checkpoint retained prediction hash differs")
        if list(_load_prediction(path).shape) != list(entry["shape"]):
            raise RuntimeError("Checkpoint retained prediction shape differs")
    return manifest


def finalize_model_predictions(
    *,
    model_id: str,
    runs: Sequence[Mapping[str, Any]],
    run_artifact_root: Path,
    output_root: Path,
    expected_patient_count: int,
    required_replicates: int,
) -> Path | None:
    """Create 25-checkpoint nested-region majority predictions for one model."""
    if len(runs) != required_replicates:
        raise ValueError("Model prediction finalization requires all frozen replicates")
    output_directory = output_root / model_id
    manifest_path = output_directory / "manifest.json"
    if manifest_path.is_file():
        stored = cast(
            dict[str, Any],
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
        if (
            stored.get("status") != "complete"
            or stored.get("model_id") != model_id
            or int(stored.get("replicate_count", -1)) != required_replicates
            or int(stored.get("patient_count", -1)) != expected_patient_count
        ):
            raise RuntimeError("Stored model-prediction manifest differs")
        for patient_id, entry in cast(
            dict[str, dict[str, str]], stored["patients"]
        ).items():
            path = _prediction_path(output_directory, patient_id)
            if not path.is_file() or file_digest(path) != entry["sha256"]:
                raise RuntimeError("Stored model prediction hash differs")
        return manifest_path

    manifests: list[dict[str, Any]] = []
    for run in runs:
        runtime_path = run_artifact_root / str(run["run_id"]) / "runtime.json"
        if not runtime_path.is_file():
            return None
        runtime = cast(
            dict[str, Any],
            json.loads(runtime_path.read_text(encoding="utf-8")),
        )
        if runtime.get("status") != "completed":
            return None
        staged_manifest = Path(str(runtime["checkpoint_prediction_manifest"]))
        manifests.append(
            validate_checkpoint_prediction_manifest(
                staged_manifest,
                run=run,
                expected_patient_count=expected_patient_count,
            )
        )

    patient_sets = [set(cast(dict[str, Any], item["patients"])) for item in manifests]
    if not patient_sets or any(
        patients != patient_sets[0] for patients in patient_sets
    ):
        raise RuntimeError("Checkpoint prediction patient identities differ")
    if output_directory.exists():
        shutil.rmtree(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    output_patients: dict[str, dict[str, str]] = {}
    threshold = required_replicates // 2 + 1
    for patient_id in sorted(patient_sets[0]):
        predictions = [
            _load_prediction(
                Path(
                    str(
                        cast(dict[str, Any], manifest["patients"])[patient_id][
                            "path"
                        ]
                    )
                )
            )
            for manifest in manifests
        ]
        shapes = {prediction.shape for prediction in predictions}
        if len(shapes) != 1:
            raise RuntimeError("Checkpoint prediction shapes differ within a patient")
        wt_votes = np.zeros(predictions[0].shape, dtype=np.uint8)
        tc_votes = np.zeros_like(wt_votes)
        et_votes = np.zeros_like(wt_votes)
        for prediction in predictions:
            wt_votes += prediction > 0
            tc_votes += np.isin(prediction, (1, 4))
            et_votes += prediction == 4
        ensemble = np.zeros(predictions[0].shape, dtype=np.uint8)
        ensemble[wt_votes >= threshold] = 2
        ensemble[tc_votes >= threshold] = 1
        ensemble[et_votes >= threshold] = 4
        destination = _prediction_path(output_directory, patient_id)
        with destination.open("wb") as handle:
            np.savez_compressed(handle, prediction_label=ensemble)
        output_patients[patient_id] = {"sha256": file_digest(destination)}

    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "model_id": model_id,
            "aggregation": "nested_region_strict_majority_vote",
            "majority_threshold": threshold,
            "replicate_count": required_replicates,
            "patient_count": len(output_patients),
            "patients": output_patients,
        },
    )
    manifest_sha256 = file_digest(manifest_path)
    for run, staged in zip(runs, manifests, strict=True):
        runtime_path = run_artifact_root / str(run["run_id"]) / "runtime.json"
        runtime = cast(
            dict[str, Any],
            json.loads(runtime_path.read_text(encoding="utf-8")),
        )
        runtime["predictions_disposition"] = "finalized_model_ensemble_then_removed"
        runtime["model_prediction_manifest"] = manifest_path.as_posix()
        runtime["model_prediction_manifest_sha256"] = manifest_sha256
        atomic_write_json(runtime_path, runtime)
        shutil.rmtree(Path(str(staged["patients"][next(iter(staged["patients"]))]["path"])).parent)
    return manifest_path


__all__ = [
    "CheckpointPredictionStager",
    "finalize_model_predictions",
    "prepare_checkpoint_stager",
    "validate_checkpoint_prediction_manifest",
]
