"""Artifact-only loss selection after all frozen development folds complete."""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text


class LossSelectionError(RuntimeError):
    """Raised when the loss screen is incomplete or scientifically invalid."""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LossSelectionError(f"Required loss-selection artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LossSelectionError(f"Expected a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise LossSelectionError(f"Required loss protocol is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LossSelectionError(f"Expected a YAML object: {path}")
    return cast(dict[str, Any], payload)


def _repository_state(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _validation_ids(split_path: Path) -> set[str]:
    frame = pd.read_csv(split_path)
    if not {"subject_id", "role"}.issubset(frame.columns):
        raise LossSelectionError(f"Invalid frozen fold manifest: {split_path}")
    validation = set(
        frame.loc[frame["role"].eq("validation"), "subject_id"].astype(str)
    )
    if not validation:
        raise LossSelectionError(f"Fold has no validation patients: {split_path}")
    return validation


def _validate_job_spec(
    queue_job: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> None:
    expected_keys = {
        "fold",
        "full_metric_evaluation",
        "loss_name",
        "maximum_optimizer_steps",
        "model_id",
        "seed",
        "stage",
        "warmup_optimizer_steps",
    }
    expected = {key: queue_job[key] for key in expected_keys}
    observed = {key: run_spec.get(key) for key in expected_keys}
    if observed != expected:
        raise LossSelectionError(
            f"Run specification differs from frozen queue: {queue_job['run_id']}"
        )


def _weighted_mean(values: Sequence[float], weights: Sequence[int]) -> float:
    return float(np.average(np.asarray(values), weights=np.asarray(weights)))


def collect_loss_selection(
    *,
    queue_path: Path,
    artifact_root: Path,
    fold_directory: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    """Validate all 15 runs and select by the prespecified patient endpoint."""
    queue = _load_json(queue_path)
    protocol = _load_yaml(protocol_path)
    selection = cast(dict[str, Any], protocol["selection"])
    candidates = tuple(str(value) for value in selection["candidates"])
    folds = tuple(int(value) for value in selection["folds"])
    expected_combinations = {
        (fold, candidate) for fold in folds for candidate in candidates
    }
    jobs = cast(list[dict[str, Any]], queue.get("jobs", []))
    if len(jobs) != len(expected_combinations):
        raise LossSelectionError(
            f"Expected {len(expected_combinations)} frozen jobs, found {len(jobs)}"
        )
    observed_combinations = {
        (int(job["fold"]), str(job["loss_name"])) for job in jobs
    }
    if observed_combinations != expected_combinations:
        raise LossSelectionError("Frozen loss-screen fold/candidate matrix changed")

    validation_ids = {
        fold: _validation_ids(fold_directory / f"cv_fold_{fold}.csv")
        for fold in folds
    }
    validation_union: set[str] = set()
    for fold in folds:
        overlap = validation_union.intersection(validation_ids[fold])
        if overlap:
            raise LossSelectionError(
                "Frozen validation folds overlap: "
                f"fold={fold}, patients={sorted(overlap)}"
            )
        validation_union.update(validation_ids[fold])
    candidate_rows: dict[str, list[dict[str, Any]]] = {
        candidate: [] for candidate in candidates
    }
    all_run_commits: set[str] = set()
    required_patient_columns = {
        "patient_id",
        "evaluation_stage",
        "wt_dice",
        "tc_dice",
        "et_dice",
        "mean_regional_dice",
    }

    for job in jobs:
        run_id = str(job["run_id"])
        run_root = artifact_root / run_id
        progress_path = run_root / "progress.json"
        metadata_path = run_root / "metadata.json"
        run_spec_path = run_root / "run_spec.json"
        patient_path = run_root / "best_validation_per_patient.csv"
        progress = _load_json(progress_path)
        metadata = _load_json(metadata_path)
        run_spec = _load_json(run_spec_path)
        _validate_job_spec(job, run_spec)
        if progress.get("status") != "completed":
            raise LossSelectionError(f"Loss-screen run is incomplete: {run_id}")
        if metadata.get("status") != "completed":
            raise LossSelectionError(f"Run metadata is not complete: {run_id}")
        if metadata.get("repository_dirty_at_start") is not False:
            raise LossSelectionError(f"Run started from a dirty repository: {run_id}")
        if metadata.get("external_data_accessed") is not False:
            raise LossSelectionError(
                f"External data was accessed by loss screen: {run_id}"
            )
        if metadata.get("legacy_internal_test_accessed") is not False:
            raise LossSelectionError(
                f"Legacy internal test was accessed by loss screen: {run_id}"
            )
        all_run_commits.add(str(metadata["git_commit"]))

        patient_frame = pd.read_csv(patient_path)
        missing_columns = required_patient_columns.difference(patient_frame.columns)
        if missing_columns:
            raise LossSelectionError(
                f"{run_id} patient metrics miss columns: {sorted(missing_columns)}"
            )
        if not patient_frame["patient_id"].is_unique:
            raise LossSelectionError(f"{run_id} repeats validation patients")
        fold = int(job["fold"])
        observed_ids = set(patient_frame["patient_id"].astype(str))
        if observed_ids != validation_ids[fold]:
            raise LossSelectionError(
                f"{run_id} does not evaluate the exact frozen validation fold"
            )
        if not patient_frame["evaluation_stage"].eq("raw").all():
            raise LossSelectionError(f"{run_id} uses a non-raw evaluation stage")
        numeric = patient_frame[
            ["wt_dice", "tc_dice", "et_dice", "mean_regional_dice"]
        ].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise LossSelectionError(f"{run_id} contains nonfinite Dice values")
        row_recomputed = numeric[:, :3].mean(axis=1)
        if not np.allclose(row_recomputed, numeric[:, 3], rtol=0.0, atol=1e-12):
            raise LossSelectionError(f"{run_id} patient endpoint is inconsistent")
        observed_metric = float(numeric[:, 3].mean())
        recorded_metric = float(progress["best_metric"])
        if not math.isclose(
            observed_metric,
            recorded_metric,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise LossSelectionError(
                f"{run_id} best metric does not match its patient artifact"
            )
        validation_loss = float(progress["best_validation_loss"])
        if not math.isfinite(validation_loss):
            raise LossSelectionError(f"{run_id} has nonfinite validation loss")

        candidate_rows[str(job["loss_name"])].append(
            {
                "run_id": run_id,
                "fold": fold,
                "patient_count": len(patient_frame),
                "best_step": int(progress["best_step"]),
                "patient_mean_regional_dice": observed_metric,
                "validation_loss": validation_loss,
                "progress_sha256": file_digest(progress_path),
                "patient_metrics_sha256": file_digest(patient_path),
                "metadata_sha256": file_digest(metadata_path),
            }
        )

    if len(all_run_commits) != 1:
        raise LossSelectionError(
            f"Loss-screen runs span multiple code commits: {sorted(all_run_commits)}"
        )

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = sorted(candidate_rows[candidate], key=lambda row: int(row["fold"]))
        if [int(row["fold"]) for row in rows] != list(folds):
            raise LossSelectionError(f"Candidate is missing a frozen fold: {candidate}")
        weights = [int(row["patient_count"]) for row in rows]
        fold_metrics = [float(row["patient_mean_regional_dice"]) for row in rows]
        validation_losses = [float(row["validation_loss"]) for row in rows]
        summaries.append(
            {
                "loss_name": candidate,
                "patient_count": sum(weights),
                "pooled_patient_mean_regional_dice": _weighted_mean(
                    fold_metrics,
                    weights,
                ),
                "unweighted_fold_mean_regional_dice": float(np.mean(fold_metrics)),
                "fold_standard_deviation_regional_dice": float(
                    np.std(fold_metrics, ddof=1)
                ),
                "patient_weighted_mean_validation_loss": _weighted_mean(
                    validation_losses,
                    weights,
                ),
                "fold_runs": rows,
            }
        )
    summaries.sort(
        key=lambda row: (
            -float(row["pooled_patient_mean_regional_dice"]),
            float(row["patient_weighted_mean_validation_loss"]),
            str(row["loss_name"]),
        )
    )
    selected = str(summaries[0]["loss_name"])
    return {
        "schema_version": 1,
        "status": "selected_from_complete_development_cv",
        "selection_endpoint": "pooled_369_patient_mean_regional_dice",
        "tie_breakers": [
            "lower_patient_weighted_mean_validation_loss",
            "lower_loss_name_lexicographic",
        ],
        "selected_loss": selected,
        "candidate_ranking": summaries,
        "run_git_commit": next(iter(all_run_commits)),
        "run_count": len(jobs),
        "fold_count": len(folds),
        "patient_count": len(validation_union),
        "queue_sha256": file_digest(queue_path),
        "loss_protocol_sha256": file_digest(protocol_path),
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
    }


def write_loss_freeze(
    *,
    queue_path: Path,
    artifact_root: Path,
    fold_directory: Path,
    protocol_path: Path,
    output_path: Path,
    selected_config_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Write the audited result and a hash-linked selected-loss config."""
    commit, dirty = _repository_state(repository_root)
    if dirty:
        raise LossSelectionError(
            f"Loss freeze requires a clean repository; resolved commit={commit}"
        )
    payload = collect_loss_selection(
        queue_path=queue_path,
        artifact_root=artifact_root,
        fold_directory=fold_directory,
        protocol_path=protocol_path,
    )
    payload["selection_git_commit"] = commit
    payload["repository_dirty_at_selection"] = False
    atomic_write_json(output_path, payload)
    selected_config = {
        "schema_version": 1,
        "status": "frozen_from_complete_development_cv",
        "selected_loss": payload["selected_loss"],
        "selection_artifact": output_path.as_posix(),
        "selection_artifact_sha256": file_digest(output_path),
        "external_data_used_for_selection": False,
        "legacy_internal_test_used_for_selection": False,
    }
    atomic_write_text(
        selected_config_path,
        yaml.safe_dump(selected_config, sort_keys=False),
    )
    return payload


__all__ = [
    "LossSelectionError",
    "collect_loss_selection",
    "write_loss_freeze",
]
