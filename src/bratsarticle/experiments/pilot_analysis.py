"""Artifact audit and paired patient-level elimination for Gate 8 pilots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bratsarticle.experiments.fairness import load_compute_matched_protocol
from bratsarticle.experiments.pilots import PilotArm, PilotPlan
from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_csv, atomic_write_json


class PilotArtifactsIncompleteError(RuntimeError):
    """Raised when a shortlist would rely on missing or invalid pilot runs."""


@dataclass(frozen=True)
class ValidPilotArtifact:
    """Validated run metadata and paired primary-endpoint values."""

    arm: PilotArm
    run_directory: Path
    run_id: str
    values: pd.Series
    metadata: dict[str, Any]
    resources: dict[str, Any]


def paired_bootstrap_mean_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Return a percentile CI for the paired patient-wise mean difference."""
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a finite vector of size >= 2")
    if resamples < 1000:
        raise ValueError("At least 1000 bootstrap resamples are required")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(
        0,
        values.size,
        size=(resamples, values.size),
    )
    bootstrap_means = np.mean(values[sampled_indices], axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(
        bootstrap_means,
        (alpha / 2.0, 1.0 - alpha / 2.0),
    )
    return float(lower), float(upper)


def _expected_split_hashes(split_dir: Path) -> dict[str, str]:
    metadata = json.loads(
        (split_dir / "split_metadata.json").read_text(encoding="utf-8")
    )
    return {
        "train": str(metadata["manifest_sha256"]["train"]),
        "validation": str(metadata["manifest_sha256"]["validation"]),
    }


def _expected_patients(plan: PilotPlan) -> set[str]:
    frame = pd.read_csv(plan.split_dir / "validation.csv")
    if "subject_id" not in frame.columns:
        raise ValueError("Validation manifest requires subject_id")
    patients = set(frame["subject_id"].astype(str))
    if len(patients) != len(frame):
        raise ValueError("Validation manifest patient IDs must be unique")
    return patients


def _run_directories(artifact_root: Path) -> list[Path]:
    if not artifact_root.is_dir():
        return []
    return sorted(
        path
        for path in artifact_root.iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )


def _primary_values(
    path: Path,
    expected_patients: set[str],
) -> tuple[pd.Series | None, list[str]]:
    reasons: list[str] = []
    if not path.is_file():
        return None, ["validation_per_case.csv missing"]
    frame = pd.read_csv(path)
    if "evaluation_stage" in frame.columns:
        frame = frame.loc[frame["evaluation_stage"].astype(str) == "raw"]
    required = {"patient_id", "mean_regional_dice"}
    if not required.issubset(frame.columns):
        return None, ["validation cases missing patient_id/mean_regional_dice"]
    if frame["patient_id"].astype(str).duplicated().any():
        reasons.append("raw validation patient IDs are duplicated")
    observed = set(frame["patient_id"].astype(str))
    if observed != expected_patients:
        reasons.append("validation patient membership differs from frozen split")
    values = pd.to_numeric(frame["mean_regional_dice"], errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        reasons.append("primary endpoint contains non-finite values")
    indexed = pd.Series(
        values.to_numpy(dtype=np.float64),
        index=frame["patient_id"].astype(str),
        name="mean_regional_dice",
    ).sort_index()
    return (None if reasons else indexed), reasons


def _validate_candidate(
    *,
    arm: PilotArm,
    run_directory: Path,
    metadata: dict[str, Any],
    plan: PilotPlan,
    plan_path: Path,
    expected_patients: set[str],
) -> tuple[ValidPilotArtifact | None, list[str]]:
    reasons: list[str] = []
    model_name = load_model_config(arm.model_config_path).name
    fairness = load_compute_matched_protocol(plan.fairness_protocol_path)
    expected_split_hashes = _expected_split_hashes(plan.split_dir)
    expected_manifest_hash = file_digest(plan.canonical_manifest_path)
    resources_path = run_directory / "resource_profile.json"
    resources = (
        json.loads(resources_path.read_text(encoding="utf-8"))
        if resources_path.is_file()
        else {}
    )
    if metadata.get("status") != "completed":
        reasons.append("run status is not completed")
    if metadata.get("repository_dirty") is not False:
        reasons.append("run repository was dirty or unreported")
    if metadata.get("seed") != arm.seed:
        reasons.append("seed mismatch")
    if metadata.get("model") != model_name:
        reasons.append("model mismatch")
    if metadata.get("loss") != arm.loss_name:
        reasons.append("loss mismatch")
    if metadata.get("split_sha256") != expected_split_hashes:
        reasons.append("split hash mismatch")
    if metadata.get("data_manifest_sha256") != expected_manifest_hash:
        reasons.append("data manifest hash mismatch")
    if metadata.get("best_validation_checkpoint") != "checkpoints/best.pt":
        reasons.append("best validation checkpoint is missing")
    if not (run_directory / "checkpoints" / "best.pt").is_file():
        reasons.append("best checkpoint file is missing")
    test_access = metadata.get("test_access", {})
    if test_access.get("allowed") or test_access.get("accessed"):
        reasons.append("test access was allowed or recorded")
    hardware = metadata.get("hardware", {})
    if hardware.get("accelerator_backend") != fairness.accelerator_backend:
        reasons.append("accelerator backend differs from frozen protocol")
    if hardware.get("accelerator_device_names") != [fairness.gpu_model]:
        reasons.append("GPU model differs from frozen protocol")
    tags = metadata.get("tags", {})
    if tags.get("gate") != plan.gate or tags.get("pilot_arm_id") != arm.arm_id:
        reasons.append("pilot arm tags mismatch")
    if plan.gate != 8:
        if tags.get("candidate_id") != arm.candidate_id:
            reasons.append("candidate tag mismatch")
        if tags.get("execution_stage") != arm.execution_stage:
            reasons.append("execution-stage tag mismatch")
    if tags.get("pilot_protocol_revision") != plan.protocol_revision:
        reasons.append("pilot protocol revision mismatch")
    if tags.get("pilot_config_sha256") != file_digest(plan_path):
        reasons.append("pilot config hash tag mismatch")
    expected_config_hashes = {
        "fairness_protocol_sha256": file_digest(plan.fairness_protocol_path),
        "preprocessing_config_sha256": file_digest(
            plan.preprocessing_config_path
        ),
        "evaluation_config_sha256": file_digest(plan.evaluation_config_path),
    }
    for tag, expected_hash in expected_config_hashes.items():
        if tags.get(tag) != expected_hash:
            reasons.append(f"{tag} mismatch")
    validation_checks = resources.get("completed_validation_checks")
    if (
        not isinstance(validation_checks, int)
        or validation_checks < plan.minimum_completed_validation_checks
    ):
        reasons.append("insufficient validation checks")
    optimizer_steps = resources.get("completed_optimizer_steps")
    if (
        not isinstance(optimizer_steps, int)
        or optimizer_steps > plan.maximum_optimizer_steps
    ):
        reasons.append("optimizer-step budget missing or exceeded")
    gpu_hours = resources.get("gpu_hours")
    if (
        not isinstance(gpu_hours, (int, float))
        or float(gpu_hours) > plan.maximum_gpu_hours + 1e-9
    ):
        reasons.append("GPU-hour budget missing or exceeded")
    values, case_reasons = _primary_values(
        run_directory / "validation_per_case.csv",
        expected_patients,
    )
    reasons.extend(case_reasons)
    if reasons or values is None:
        return None, reasons
    return (
        ValidPilotArtifact(
            arm=arm,
            run_directory=run_directory,
            run_id=str(metadata["run_id"]),
            values=values,
            metadata=metadata,
            resources=resources,
        ),
        [],
    )


def audit_pilot_artifacts(
    *,
    plan: PilotPlan,
    plan_path: Path,
    artifact_root: Path | None = None,
    expected_arms: tuple[PilotArm, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, ValidPilotArtifact]]:
    """Audit all expected pilot runs without producing a shortlist."""
    root = artifact_root or plan.artifact_root
    arms = expected_arms or plan.arms
    expected_patients = _expected_patients(plan)
    candidates_by_arm: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    untagged_run_ids: list[str] = []
    foreign_config_run_ids: list[str] = []
    expected_plan_hash = file_digest(plan_path)
    for directory in _run_directories(root):
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        arm_id = str(metadata.get("tags", {}).get("pilot_arm_id", ""))
        if not arm_id:
            untagged_run_ids.append(str(metadata.get("run_id", directory.name)))
            continue
        if metadata.get("tags", {}).get("pilot_config_sha256") != expected_plan_hash:
            foreign_config_run_ids.append(str(metadata.get("run_id", directory.name)))
            continue
        candidates_by_arm.setdefault(arm_id, []).append((directory, metadata))

    valid: dict[str, ValidPilotArtifact] = {}
    invalid: dict[str, list[dict[str, Any]]] = {}
    duplicates: dict[str, int] = {}
    for arm in arms:
        candidates = candidates_by_arm.get(arm.arm_id, [])
        if len(candidates) > 1:
            duplicates[arm.arm_id] = len(candidates)
        for directory, metadata in candidates:
            artifact, reasons = _validate_candidate(
                arm=arm,
                run_directory=directory,
                metadata=metadata,
                plan=plan,
                plan_path=plan_path,
                expected_patients=expected_patients,
            )
            if artifact is not None:
                if arm.arm_id in valid:
                    duplicates[arm.arm_id] = len(candidates)
                else:
                    valid[arm.arm_id] = artifact
            else:
                invalid.setdefault(arm.arm_id, []).append(
                    {
                        "run_id": str(metadata.get("run_id", directory.name)),
                        "reasons": reasons,
                    }
                )
    missing = sorted(arm.arm_id for arm in arms if arm.arm_id not in valid)
    complete = not missing and not duplicates and len(valid) == len(arms)
    audit = {
        "status": "complete" if complete else "incomplete",
        "artifact_root": root.as_posix(),
        "expected_arm_count": len(arms),
        "valid_arm_count": len(valid),
        "missing_or_invalid_arms": missing,
        "duplicate_arm_runs": duplicates,
        "invalid_runs": invalid,
        "untagged_run_ids_ignored": sorted(untagged_run_ids),
        "foreign_config_run_ids_ignored": sorted(foreign_config_run_ids),
        "expected_validation_patients": len(expected_patients),
        "test_access_used": False,
        "shortlist_permitted": complete,
    }
    return audit, valid


def _screen_result(
    *,
    screen: str,
    arm_ids: list[str],
    artifacts: dict[str, ValidPilotArtifact],
    plan: PilotPlan,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    means = {arm_id: float(artifacts[arm_id].values.mean()) for arm_id in arm_ids}
    best_id = max(arm_ids, key=lambda value: (means[value], value))
    best_values = artifacts[best_id].values
    margin = float(plan.elimination["practical_noninferiority_margin"])
    resamples = int(plan.elimination["paired_bootstrap_resamples"])
    confidence = float(plan.elimination["confidence_level"])
    rows: list[dict[str, Any]] = []
    retained: list[str] = []
    for index, arm_id in enumerate(sorted(arm_ids)):
        candidate = artifacts[arm_id]
        paired = candidate.values.reindex(best_values.index) - best_values
        lower, upper = paired_bootstrap_mean_interval(
            paired.to_numpy(dtype=np.float64),
            resamples=resamples,
            confidence_level=confidence,
            seed=plan.seed + index,
        )
        mean_difference = float(paired.mean())
        eliminated = mean_difference < -margin and upper < 0.0
        if not eliminated:
            retained.append(arm_id)
        rows.append(
            {
                "screen": screen,
                "arm_id": arm_id,
                "run_id": candidate.run_id,
                "patient_count": int(candidate.values.size),
                "mean_regional_dice": means[arm_id],
                "median_regional_dice": float(candidate.values.median()),
                "paired_reference_arm": best_id,
                "paired_mean_difference": mean_difference,
                "paired_bootstrap_lower": lower,
                "paired_bootstrap_upper": upper,
                "practical_margin": margin,
                "eliminated": eliminated,
            }
        )
    top_key = f"{screen}_top_k"
    top_k = int(plan.elimination["fallback_shortlist_if_too_few_eliminated"][top_key])
    fallback_applied = len(retained) > top_k
    shortlist = (
        sorted(retained, key=lambda value: (-means[value], value))[:top_k]
        if fallback_applied
        else sorted(retained, key=lambda value: (-means[value], value))
    )
    return (
        {
            "screen": screen,
            "best_arm": best_id,
            "retained_by_statistical_rule": retained,
            "fallback_top_k": top_k,
            "fallback_applied": fallback_applied,
            "shortlist": shortlist,
        },
        rows,
    )


def analyze_pilot_artifacts(
    *,
    plan: PilotPlan,
    plan_path: Path,
    artifact_root: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce architecture/loss shortlists only after a complete audit."""
    audit, artifacts = audit_pilot_artifacts(
        plan=plan,
        plan_path=plan_path,
        artifact_root=artifact_root,
    )
    if audit["status"] != "complete":
        raise PilotArtifactsIncompleteError(
            "Gate 8 artifacts are incomplete; shortlist generation is forbidden"
        )
    architecture_ids = [arm.arm_id for arm in plan.arms if arm.screen == "architecture"]
    loss_ids = [arm.arm_id for arm in plan.arms if arm.screen == "loss"]
    loss_ids.extend(plan.loss_reuse.values())
    architecture_result, architecture_rows = _screen_result(
        screen="architecture",
        arm_ids=architecture_ids,
        artifacts=artifacts,
        plan=plan,
    )
    loss_result, loss_rows = _screen_result(
        screen="loss",
        arm_ids=sorted(set(loss_ids)),
        artifacts=artifacts,
        plan=plan,
    )
    result = {
        "status": "complete",
        "audit": audit,
        "architecture_screen": architecture_result,
        "loss_screen": loss_result,
        "next_stage_seeds": plan.elimination["next_stage_seeds"],
        "finalist_seed_count": plan.elimination["finalist_seed_count"],
        "internal_test_access": False,
    }
    return result, architecture_rows + loss_rows


def write_pilot_analysis(
    *,
    result: dict[str, Any],
    rows: list[dict[str, Any]],
    json_output: Path,
    csv_output: Path,
) -> None:
    """Atomically serialize Gate 8 analysis artifacts."""
    atomic_write_json(json_output, result)
    atomic_write_csv(csv_output, rows)
