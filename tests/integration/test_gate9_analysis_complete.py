import json
from pathlib import Path

import numpy as np
import pandas as pd

from bratsarticle.experiments.gate9 import (
    analyze_confirmation,
    stage_arms,
)
from bratsarticle.experiments.pilots import load_pilot_plan
from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.utils.hashing import file_digest


def _split_hashes(split_dir: Path) -> dict[str, str]:
    payload = json.loads(
        (split_dir / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "train": str(payload["manifest_sha256"]["train"]),
        "validation": str(payload["manifest_sha256"]["validation"]),
    }


def test_gate9_confirmation_selects_two_patient_paired_finalists(
    tmp_path: Path,
) -> None:
    plan_path = Path("configs/pilots/gate9.yaml")
    plan = load_pilot_plan(plan_path)
    patients = pd.read_csv(plan.split_dir / "validation.csv")[
        "subject_id"
    ].astype(str)
    base_means = {
        "unet_reference": 0.70,
        "bunet": 0.80,
        "unet_res": 0.795,
        "unet_wc": 0.74,
    }
    split_hashes = _split_hashes(plan.split_dir)
    manifest_hash = file_digest(plan.canonical_manifest_path)
    plan_hash = file_digest(plan_path)
    offsets = np.linspace(-0.01, 0.01, len(patients))

    for arm in stage_arms(plan, "confirmation"):
        run_directory = tmp_path / arm.arm_id
        (run_directory / "checkpoints").mkdir(parents=True)
        (run_directory / "checkpoints" / "best.pt").write_bytes(b"test")
        metadata = {
            "run_id": f"test_{arm.arm_id}",
            "status": "completed",
            "repository_dirty": False,
            "seed": arm.seed,
            "model": load_model_config(arm.model_config_path).name,
            "loss": arm.loss_name,
            "split_sha256": split_hashes,
            "data_manifest_sha256": manifest_hash,
            "best_validation_checkpoint": "checkpoints/best.pt",
            "test_access": {"allowed": False, "accessed": False},
            "hardware": {
                "accelerator_backend": "mps",
                "accelerator_device_names": ["Apple M1 Max"],
            },
            "tags": {
                "gate": 9,
                "pilot_protocol_revision": plan.protocol_revision,
                "pilot_arm_id": arm.arm_id,
                "candidate_id": arm.candidate_id,
                "execution_stage": arm.execution_stage,
                "pilot_config_sha256": plan_hash,
                "fairness_protocol_sha256": file_digest(
                    plan.fairness_protocol_path
                ),
                "preprocessing_config_sha256": file_digest(
                    plan.preprocessing_config_path
                ),
                "evaluation_config_sha256": file_digest(
                    plan.evaluation_config_path
                ),
            },
        }
        (run_directory / "metadata.json").write_text(
            json.dumps(metadata),
            encoding="utf-8",
        )
        (run_directory / "resource_profile.json").write_text(
            json.dumps(
                {
                    "completed_validation_checks": 1,
                    "completed_optimizer_steps": 2000,
                    "gpu_hours": 0.4,
                }
            ),
            encoding="utf-8",
        )
        seed_offset = (arm.seed - 20260730) * 0.001
        pd.DataFrame(
            {
                "patient_id": patients,
                "evaluation_stage": "raw",
                "mean_regional_dice": (
                    base_means[arm.candidate_id] + seed_offset + offsets
                ),
            }
        ).to_csv(run_directory / "validation_per_case.csv", index=False)

    result, rows = analyze_confirmation(
        plan=plan,
        plan_path=plan_path,
        artifact_root=tmp_path,
    )

    assert result["status"] == "confirmation_complete"
    assert result["audit"]["valid_arm_count"] == 12
    assert result["best_confirmation_candidate"] == "bunet"
    assert result["finalists"] == ["bunet", "unet_res"]
    assert len(rows) == 4
