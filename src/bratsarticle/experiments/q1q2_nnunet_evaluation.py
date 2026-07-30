"""Common-metric evaluation of official nnU-Net development predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.adapters.nnunetv2 import nnunet_to_brats_labels
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json
from evaluation import (
    CentralEvaluator,
    load_evaluation_config,
    summarize_patient_metrics,
)


def _validation_subjects(fold_path: Path) -> tuple[str, ...]:
    frame = pd.read_csv(fold_path)
    required = {"subject_id", "role"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Fold manifest is missing columns: {sorted(missing)}")
    validation = frame.loc[frame["role"].eq("validation"), "subject_id"]
    subjects = tuple(sorted(str(value) for value in validation))
    if len(subjects) not in {73, 74} or len(set(subjects)) != len(subjects):
        raise ValueError("Fold must contain 73 or 74 unique validation patients")
    return subjects


def evaluate_nnunet_best_validation(
    *,
    prediction_directory: Path,
    label_directory: Path,
    fold_path: Path,
    evaluation_config_path: Path,
    output_directory: Path,
    run_id: str,
    model_id: str,
    fold: int,
    seed: int,
    best_checkpoint_path: Path,
) -> dict[str, Any]:
    """Evaluate exactly one fold's best-checkpoint predictions centrally."""
    subjects = _validation_subjects(fold_path)
    expected_files = {f"{subject}.nii" for subject in subjects}
    observed_files = {
        path.name for path in prediction_directory.glob("*.nii") if path.is_file()
    }
    if observed_files != expected_files:
        missing = sorted(expected_files - observed_files)
        unexpected = sorted(observed_files - expected_files)
        raise RuntimeError(
            "nnU-Net validation predictions differ from the frozen fold: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    evaluator = CentralEvaluator(load_evaluation_config(evaluation_config_path))
    checkpoint_sha256 = file_digest(best_checkpoint_path)
    rows: list[dict[str, Any]] = []
    prediction_hashes: dict[str, str] = {}
    for subject in subjects:
        prediction_path = prediction_directory / f"{subject}.nii"
        label_path = label_directory / f"{subject}.nii"
        if not label_path.is_file():
            raise FileNotFoundError(f"nnU-Net derived label is missing: {label_path}")
        prediction_image = cast(
            nib.Nifti1Image, nib.load(str(prediction_path), mmap="r")
        )
        label_image = cast(nib.Nifti1Image, nib.load(str(label_path), mmap="r"))
        prediction_nnunet = np.asanyarray(prediction_image.dataobj)
        label_nnunet = np.asanyarray(label_image.dataobj)
        if prediction_nnunet.shape != label_nnunet.shape:
            raise ValueError(f"Prediction/label shape mismatch for {subject}")
        if not np.allclose(
            prediction_image.affine,
            label_image.affine,
            rtol=0.0,
            atol=1e-5,
        ):
            raise ValueError(f"Prediction/label affine mismatch for {subject}")
        spacing = tuple(
            float(value)
            for value in prediction_image.header.get_zooms()[:3]  # type: ignore[no-untyped-call]
        )
        metric_rows = evaluator.evaluate_batch(
            nnunet_to_brats_labels(prediction_nnunet),
            nnunet_to_brats_labels(label_nnunet),
            patient_ids=[subject],
            spacings_mm=[spacing],
        )
        rows.extend(
            {
                "run_id": run_id,
                "model_id": model_id,
                "fold": fold,
                "seed": seed,
                "checkpoint_role": "best_development",
                "checkpoint_sha256": checkpoint_sha256,
                **row,
            }
            for row in metric_rows
        )
        prediction_hashes[subject] = file_digest(prediction_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    patient_path = output_directory / "best_checkpoint_full_metrics.csv"
    summary_path = output_directory / "best_checkpoint_full_metric_summary.csv"
    atomic_write_csv(patient_path, rows)
    metric_only_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            not in {
                "run_id",
                "model_id",
                "fold",
                "seed",
                "checkpoint_role",
                "checkpoint_sha256",
            }
        }
        for row in rows
    ]
    atomic_write_csv(summary_path, summarize_patient_metrics(metric_only_rows))
    report = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "model_id": model_id,
        "fold": fold,
        "seed": seed,
        "checkpoint_role": "best_development",
        "checkpoint_path": best_checkpoint_path.as_posix(),
        "checkpoint_sha256": checkpoint_sha256,
        "fold_manifest": fold_path.as_posix(),
        "fold_manifest_sha256": file_digest(fold_path),
        "evaluation_config": evaluation_config_path.as_posix(),
        "evaluation_config_sha256": file_digest(evaluation_config_path),
        "patient_count": len(subjects),
        "metric_row_count": len(rows),
        "patient_metrics": patient_path.as_posix(),
        "patient_metrics_sha256": file_digest(patient_path),
        "metric_summary": summary_path.as_posix(),
        "metric_summary_sha256": file_digest(summary_path),
        "prediction_sha256_by_patient": prediction_hashes,
        "external_data_accessed": False,
        "legacy_internal_test_accessed": False,
    }
    report_path = output_directory / "central_evaluation.json"
    atomic_write_json(report_path, report)
    report["report_path"] = report_path.as_posix()
    report["report_sha256"] = file_digest(report_path)
    return report


__all__ = ["evaluate_nnunet_best_validation"]
