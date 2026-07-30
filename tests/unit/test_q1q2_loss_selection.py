from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bratsarticle.experiments.q1q2_loss_selection import (
    LossSelectionError,
    collect_loss_selection,
)

CANDIDATES = (
    "cross_entropy_plus_soft_dice",
    "binary_cross_entropy_plus_focal_tversky",
    "cross_entropy_plus_focal_tversky",
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _selection_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    fold_directory = tmp_path / "folds"
    fold_directory.mkdir()
    for fold in range(1, 6):
        pd.DataFrame(
            [
                {
                    "subject_id": f"patient-{fold}",
                    "role": "validation",
                }
            ]
        ).to_csv(fold_directory / f"cv_fold_{fold}.csv", index=False)

    protocol_path = tmp_path / "loss_protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "selection": {
                    "folds": [1, 2, 3, 4, 5],
                    "candidates": list(CANDIDATES),
                }
            }
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue.json"
    artifact_root = tmp_path / "runs"
    jobs: list[dict[str, object]] = []
    metrics = {
        CANDIDATES[0]: (0.8, 0.5),
        CANDIDATES[1]: (0.8, 0.4),
        CANDIDATES[2]: (0.7, 0.3),
    }
    for fold in range(1, 6):
        for candidate in CANDIDATES:
            run_id = f"loss__f{fold}__{candidate}"
            spec: dict[str, object] = {
                "fold": fold,
                "full_metric_evaluation": False,
                "loss_name": candidate,
                "maximum_optimizer_steps": 10000,
                "model_id": "unet_small",
                "seed": 20260730,
                "stage": "loss_screen",
                "warmup_optimizer_steps": 1000,
            }
            jobs.append({"run_id": run_id, **spec})
            run_root = artifact_root / run_id
            metric, validation_loss = metrics[candidate]
            _write_json(run_root / "run_spec.json", spec)
            _write_json(
                run_root / "progress.json",
                {
                    "status": "completed",
                    "best_metric": metric,
                    "best_validation_loss": validation_loss,
                    "best_step": 1000,
                },
            )
            _write_json(
                run_root / "metadata.json",
                {
                    "status": "completed",
                    "repository_dirty_at_start": False,
                    "external_data_accessed": False,
                    "legacy_internal_test_accessed": False,
                    "git_commit": "frozen-run-commit",
                },
            )
            pd.DataFrame(
                [
                    {
                        "patient_id": f"patient-{fold}",
                        "evaluation_stage": "raw",
                        "wt_dice": metric,
                        "tc_dice": metric,
                        "et_dice": metric,
                        "mean_regional_dice": metric,
                    }
                ]
            ).to_csv(run_root / "best_validation_per_patient.csv", index=False)
    _write_json(queue_path, {"jobs": jobs})
    return queue_path, artifact_root, fold_directory, protocol_path


def test_loss_selection_uses_prespecified_tie_breakers(tmp_path: Path) -> None:
    queue, artifacts, folds, protocol = _selection_fixture(tmp_path)

    result = collect_loss_selection(
        queue_path=queue,
        artifact_root=artifacts,
        fold_directory=folds,
        protocol_path=protocol,
    )

    assert result["selected_loss"] == CANDIDATES[1]
    assert result["run_count"] == 15
    assert result["patient_count"] == 5
    assert [
        row["loss_name"] for row in result["candidate_ranking"]
    ] == [CANDIDATES[1], CANDIDATES[0], CANDIDATES[2]]


def test_loss_selection_rejects_an_incomplete_run(tmp_path: Path) -> None:
    queue, artifacts, folds, protocol = _selection_fixture(tmp_path)
    progress = artifacts / f"loss__f1__{CANDIDATES[0]}" / "progress.json"
    payload = json.loads(progress.read_text(encoding="utf-8"))
    payload["status"] = "running"
    _write_json(progress, payload)

    with pytest.raises(LossSelectionError, match="incomplete"):
        collect_loss_selection(
            queue_path=queue,
            artifact_root=artifacts,
            fold_directory=folds,
            protocol_path=protocol,
        )


def test_loss_selection_rejects_wrong_validation_patient(tmp_path: Path) -> None:
    queue, artifacts, folds, protocol = _selection_fixture(tmp_path)
    patient_path = (
        artifacts
        / f"loss__f1__{CANDIDATES[0]}"
        / "best_validation_per_patient.csv"
    )
    frame = pd.read_csv(patient_path)
    frame.loc[0, "patient_id"] = "patient-from-another-fold"
    frame.to_csv(patient_path, index=False)

    with pytest.raises(LossSelectionError, match="exact frozen validation fold"):
        collect_loss_selection(
            queue_path=queue,
            artifact_root=artifacts,
            fold_directory=folds,
            protocol_path=protocol,
        )
