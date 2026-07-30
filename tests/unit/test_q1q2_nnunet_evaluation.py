from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from bratsarticle.experiments.q1q2_nnunet_evaluation import (
    evaluate_nnunet_best_validation,
)


def _save_label(path: Path, values: np.ndarray) -> None:
    image = nib.Nifti1Image(values.astype(np.uint8), np.eye(4))
    nib.save(image, str(path))


def test_nnunet_best_validation_uses_exact_fold_and_common_evaluator(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions"
    labels = tmp_path / "labels"
    predictions.mkdir()
    labels.mkdir()
    subjects = [f"BraTS20_Training_{index:03d}" for index in range(1, 74)]
    fold_path = tmp_path / "fold.csv"
    pd.DataFrame(
        {
            "subject_id": subjects,
            "role": ["validation"] * len(subjects),
        }
    ).to_csv(fold_path, index=False)
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    label[1:3, 1:3, 1:3] = 1
    label[2, 2, 2] = 3
    for subject in subjects:
        _save_label(predictions / f"{subject}.nii", label)
        _save_label(labels / f"{subject}.nii", label)
    checkpoint = tmp_path / "checkpoint_best.pth"
    checkpoint.write_bytes(b"checkpoint")

    report = evaluate_nnunet_best_validation(
        prediction_directory=predictions,
        label_directory=labels,
        fold_path=fold_path,
        evaluation_config_path=Path("configs/q1q2_v2/evaluation.yaml"),
        output_directory=tmp_path / "evaluation",
        run_id="nnunetv2_2d__f1__s20260730__convergence",
        model_id="nnunetv2_2d",
        fold=1,
        seed=20260730,
        best_checkpoint_path=checkpoint,
    )

    metrics = pd.read_csv(report["patient_metrics"])
    assert report["status"] == "completed"
    assert report["patient_count"] == 73
    assert set(metrics["patient_id"]) == set(subjects)
    assert set(metrics["mean_regional_dice"]) == {1.0}
    assert set(metrics["model_id"]) == {"nnunetv2_2d"}
