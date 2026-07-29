import json
from pathlib import Path

import numpy as np
import pandas as pd

from bratsarticle.experiments.pilot_analysis import (
    analyze_pilot_artifacts,
    audit_pilot_artifacts,
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


def test_complete_artifacts_produce_paired_shortlists(tmp_path: Path) -> None:
    plan_path = Path("configs/pilots/gate8.yaml")
    plan = load_pilot_plan(plan_path)
    patients = pd.read_csv(plan.split_dir / "validation.csv")["subject_id"].astype(str)
    means = {
        "architecture_unet": 0.800,
        "architecture_unet_res": 0.790,
        "architecture_unet_wc": 0.810,
        "architecture_bunet": 0.840,
        "architecture_resunet": 0.830,
        "architecture_resunet_wc": 0.825,
        "loss_cross_entropy": 0.805,
        "loss_binary_cross_entropy": 0.755,
        "loss_soft_dice": 0.795,
        "loss_binary_cross_entropy_plus_soft_dice": 0.760,
        "loss_focal_tversky": 0.770,
        "loss_binary_cross_entropy_plus_focal_tversky": 0.765,
    }
    split_hashes = _split_hashes(plan.split_dir)
    manifest_hash = file_digest(plan.canonical_manifest_path)
    plan_hash = file_digest(plan_path)
    patient_offsets = np.linspace(-0.01, 0.01, len(patients))

    for arm in plan.arms:
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
                "gate": 8,
                "pilot_protocol_revision": plan.protocol_revision,
                "pilot_arm_id": arm.arm_id,
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
        resources = {
            "completed_validation_checks": 4,
            "completed_optimizer_steps": 2000,
            "gpu_hours": 0.4,
        }
        (run_directory / "resource_profile.json").write_text(
            json.dumps(resources),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "patient_id": patients,
                "evaluation_stage": "raw",
                "mean_regional_dice": means[arm.arm_id] + patient_offsets,
            }
        ).to_csv(run_directory / "validation_per_case.csv", index=False)

    audit, valid = audit_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=tmp_path,
    )
    result, rows = analyze_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=tmp_path,
    )

    assert audit["status"] == "complete"
    assert len(valid) == 12
    assert result["architecture_screen"]["best_arm"] == "architecture_bunet"
    assert len(result["architecture_screen"]["shortlist"]) == 3
    assert result["loss_screen"]["best_arm"] == "loss_cross_entropy"
    assert len(result["loss_screen"]["shortlist"]) == 2
    assert len(rows) == 13
    assert not result["internal_test_access"]
