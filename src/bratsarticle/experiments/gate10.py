"""Gate 10 split, checkpoint, and statistical-analysis freeze."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class Gate10Paths:
    """Resolved files required to construct the Gate 10 freeze."""

    config_path: Path
    gate9_analysis: Path
    confirmation_summary: Path
    final_summary: Path
    gate9_pilot_config: Path
    artifact_root: Path
    source_split_dir: Path
    frozen_split_dir: Path
    evaluation_config: Path
    preprocessing_config: Path
    checkpoint_manifest: Path
    analysis_freeze: Path
    completion_report: Path
    test_access_audit: Path


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return cast(Mapping[str, Any], value)


def _strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    return [str(item) for item in value]


def load_gate10_plan(path: Path) -> dict[str, Any]:
    """Load and validate the immutable Gate 10 analysis plan."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(loaded, "configuration")
    plan = dict(_mapping(root.get("gate10"), "gate10"))
    if int(plan.get("gate", -1)) != 10:
        raise ValueError("Gate 10 plan must declare gate: 10")
    if plan.get("status") != "frozen_pending_execution":
        raise ValueError("Gate 10 plan must be frozen before execution")
    if bool(plan.get("internal_test_permitted")):
        raise ValueError("Gate 10 cannot permit internal-test access")

    candidates = _mapping(plan.get("candidates"), "candidates")
    ordered = _strings(
        candidates.get("ordered_internal_test_candidates"),
        "ordered_internal_test_candidates",
    )
    if len(ordered) != 3 or len(set(ordered)) != 3:
        raise ValueError("Exactly three unique internal-test candidates are required")
    if str(candidates.get("primary")) not in ordered:
        raise ValueError("Primary candidate must be in the frozen candidate set")
    if str(candidates.get("mandatory_reference")) not in ordered:
        raise ValueError("Mandatory reference must be in the candidate set")
    if bool(candidates.get("seed_ensemble")):
        raise ValueError("Gate 10 forbids seed ensembling")
    if not bool(candidates.get("evaluate_every_predeclared_seed")):
        raise ValueError("Every predeclared seed must be evaluated")

    endpoints = _mapping(plan.get("endpoints"), "endpoints")
    if endpoints.get("statistical_unit") != "patient":
        raise ValueError("The statistical unit must be the patient")
    primary = _mapping(endpoints.get("primary"), "primary endpoint")
    if primary.get("name") != "mean_regional_dice":
        raise ValueError("The primary endpoint must be mean regional Dice")

    tests = _mapping(plan.get("hypothesis_testing"), "hypothesis_testing")
    comparisons = tests.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != 3:
        raise ValueError("Exactly three primary-endpoint comparisons are required")
    comparison_ids = {
        str(_mapping(item, "comparison").get("id")) for item in comparisons
    }
    if len(comparison_ids) != 3:
        raise ValueError("Hypothesis comparison IDs must be unique")
    multiplicity = _mapping(tests.get("multiplicity"), "multiplicity")
    if multiplicity.get("correction") != "holm":
        raise ValueError("Primary-endpoint comparisons require Holm correction")

    conduct = _mapping(plan.get("conduct"), "conduct")
    required_prohibitions = (
        "no_model_selection_after_test_access",
        "no_checkpoint_replacement_after_test_access",
        "no_threshold_or_postprocessing_tuning_after_test_access",
        "report_all_frozen_candidates_and_seeds",
    )
    if not all(bool(conduct.get(key)) for key in required_prohibitions):
        raise ValueError("All post-test conduct safeguards must be enabled")
    return plan


def resolve_gate10_paths(
    plan: Mapping[str, Any],
    config_path: Path,
    *,
    checkpoint_manifest: Path = Path("reports/gate10_checkpoint_manifest.json"),
    analysis_freeze: Path = Path("reports/gate10_analysis_freeze.json"),
    completion_report: Path = Path("reports/gate10_completion.md"),
) -> Gate10Paths:
    """Resolve plan paths relative to the repository working directory."""
    source = _mapping(plan["source"], "source")
    split = _mapping(plan["split"], "split")
    evaluation = _mapping(plan["evaluation"], "evaluation")
    conduct = _mapping(plan["conduct"], "conduct")
    return Gate10Paths(
        config_path=config_path,
        gate9_analysis=Path(str(source["gate9_analysis"])),
        confirmation_summary=Path(str(source["gate9_confirmation_summary"])),
        final_summary=Path(str(source["gate9_final_summary"])),
        gate9_pilot_config=Path(str(source["gate9_pilot_config"])),
        artifact_root=Path(str(source["artifact_root"])),
        source_split_dir=Path(str(split["source_dir"])),
        frozen_split_dir=Path(str(split["frozen_dir"])),
        evaluation_config=Path(str(evaluation["evaluation_config"])),
        preprocessing_config=Path(str(evaluation["preprocessing_config"])),
        checkpoint_manifest=checkpoint_manifest,
        analysis_freeze=analysis_freeze,
        completion_report=completion_report,
        test_access_audit=Path(str(conduct["append_only_test_access_audit"])),
    )


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def assert_clean_repository() -> None:
    """Refuse to freeze results from an uncommitted protocol state."""
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    if status.strip():
        raise RuntimeError("Gate 10 freeze requires a clean repository")


def assert_no_internal_test_access(audit_path: Path) -> None:
    """Require an empty/nonexistent access log before the first test opening."""
    if not audit_path.exists():
        return
    events = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(event.get("event") == "internal_test_manifest_access" for event in events):
        raise RuntimeError("Internal test was accessed before Gate 10 freeze")


def development_tumor_thresholds(
    train_manifest: Path,
    quantiles: Sequence[float],
    *,
    column: str,
    method: str,
) -> dict[str, float]:
    """Derive subgroup thresholds from training patients only."""
    frame = pd.read_csv(train_manifest, usecols=["split", column])
    if set(frame["split"].astype(str)) != {"train"}:
        raise ValueError("Subgroup thresholds must be derived from train only")
    values = frame[column].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Training tumor volumes must be finite and non-negative")
    if method != "linear":
        raise ValueError("Gate 10 freezes linear quantile interpolation")
    resolved = np.quantile(
        values,
        [float(value) for value in quantiles],
        method="linear",
    )
    if len(resolved) != 2 or not float(resolved[0]) < float(resolved[1]):
        raise ValueError("Tumor-burden tertile thresholds must be strictly ordered")
    return {"q1_mm3": float(resolved[0]), "q2_mm3": float(resolved[1])}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    try:
        shutil.copyfile(source, raw_path)
        Path(raw_path).replace(destination)
    finally:
        Path(raw_path).unlink(missing_ok=True)


def freeze_split_membership(
    paths: Gate10Paths,
    plan: Mapping[str, Any],
    *,
    git_commit: str,
) -> dict[str, Any]:
    """Copy the split manifests byte-for-byte and emit immutable metadata."""
    source_metadata_path = paths.source_split_dir / "split_metadata.json"
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if source_metadata.get("status") != "pass":
        raise RuntimeError("Only a passing provisional split can be frozen")
    expected_counts = _mapping(
        _mapping(plan["split"], "split")["expected_counts"],
        "expected_counts",
    )
    if source_metadata.get("counts") != {
        name: int(expected_counts[name]) for name in _SPLITS
    }:
        raise RuntimeError("Provisional split counts differ from the Gate 10 plan")

    paths.frozen_split_dir.mkdir(parents=True, exist_ok=True)
    copied_hashes: dict[str, str] = {}
    for split_name in _SPLITS:
        source = paths.source_split_dir / f"{split_name}.csv"
        expected_hash = str(source_metadata["manifest_sha256"][split_name])
        if file_digest(source) != expected_hash:
            raise RuntimeError(f"Provisional {split_name} manifest hash changed")
        destination = paths.frozen_split_dir / f"{split_name}.csv"
        _atomic_copy(source, destination)
        copied_hashes[split_name] = file_digest(destination)
        if copied_hashes[split_name] != expected_hash:
            raise RuntimeError(f"Frozen {split_name} manifest is not byte-identical")

    for filename in ("categorical_balance.csv", "continuous_balance.csv"):
        source = paths.source_split_dir / filename
        if source.is_file():
            _atomic_copy(source, paths.frozen_split_dir / filename)

    frozen_metadata = {
        **source_metadata,
        "frozen": True,
        "frozen_at_git_commit": git_commit,
        "freeze_config_path": paths.config_path.as_posix(),
        "freeze_config_sha256": file_digest(paths.config_path),
        "source_split_dir": paths.source_split_dir.as_posix(),
        "source_split_metadata_sha256": file_digest(source_metadata_path),
        "manifest_sha256": copied_hashes,
    }
    atomic_write_json(
        paths.frozen_split_dir / "split_metadata.json",
        frozen_metadata,
    )
    return frozen_metadata


def _candidate_run_ids(
    candidate: str,
    confirmation: pd.DataFrame,
    final: pd.DataFrame,
) -> list[str]:
    source = confirmation if candidate == "unet_reference" else final
    matches = source.loc[source["candidate_id"].astype(str) == candidate]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one summary row for {candidate}")
    return str(matches.iloc[0]["run_ids"]).split("|")


def build_checkpoint_manifest(
    paths: Gate10Paths,
    plan: Mapping[str, Any],
    *,
    git_commit: str,
) -> dict[str, Any]:
    """Pin every selected seed to an exact model config and checkpoint hash."""
    gate9 = json.loads(paths.gate9_analysis.read_text(encoding="utf-8"))
    if gate9.get("status") != "complete" or bool(gate9["internal_test_access"]):
        raise RuntimeError("Gate 9 must be complete with no internal-test access")
    candidates = _mapping(plan["candidates"], "candidates")
    ordered = _strings(
        candidates["ordered_internal_test_candidates"],
        "ordered_internal_test_candidates",
    )
    if gate9["internal_test_candidates"] != ordered:
        raise RuntimeError("Gate 9 candidate set differs from the Gate 10 freeze")
    if gate9["primary_finalist"] != candidates["primary"]:
        raise RuntimeError("Gate 9 primary finalist differs from the Gate 10 plan")

    confirmation = pd.read_csv(paths.confirmation_summary)
    final = pd.read_csv(paths.final_summary)
    expected_seeds = _mapping(candidates["expected_seeds"], "expected_seeds")
    model_configs = _mapping(candidates["model_configs"], "model_configs")
    entries: list[dict[str, Any]] = []
    common_pilot_hash: str | None = None
    common_evaluation_hash: str | None = None
    common_preprocessing_hash: str | None = None
    for candidate in ordered:
        model_config = Path(str(model_configs[candidate]))
        candidate_entries: list[dict[str, Any]] = []
        for run_id in _candidate_run_ids(candidate, confirmation, final):
            run_directory = paths.artifact_root / run_id
            metadata = json.loads(
                (run_directory / "metadata.json").read_text(encoding="utf-8")
            )
            if metadata["status"] != "completed":
                raise RuntimeError(f"Incomplete selected run: {run_id}")
            if bool(metadata["repository_dirty"]):
                raise RuntimeError(f"Selected run used a dirty repository: {run_id}")
            if bool(metadata["test_access"]["accessed"]):
                raise RuntimeError(f"Selected run accessed internal test: {run_id}")
            if metadata["tags"]["candidate_id"] != candidate:
                raise RuntimeError(f"Candidate tag mismatch: {run_id}")
            checkpoint = run_directory / str(metadata["best_validation_checkpoint"])
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            entry = {
                "candidate_id": candidate,
                "seed": int(metadata["seed"]),
                "run_id": run_id,
                "run_git_commit": str(metadata["git_commit"]),
                "run_config_sha256": str(metadata["config_sha256"]),
                "model_config_path": model_config.as_posix(),
                "model_config_sha256": file_digest(model_config),
                "checkpoint_path": checkpoint.as_posix(),
                "checkpoint_sha256": file_digest(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "parameter_count": int(metadata["parameter_count"]),
                "gpu_hours": float(metadata["gpu_hours"]),
                "peak_allocated_vram_bytes": int(
                    metadata["peak_allocated_vram_bytes"]
                ),
                "peak_reserved_vram_bytes": int(metadata["peak_reserved_vram_bytes"]),
            }
            candidate_entries.append(entry)
            pilot_hash = str(metadata["tags"]["pilot_config_sha256"])
            evaluation_hash = str(metadata["tags"]["evaluation_config_sha256"])
            preprocessing_hash = str(
                metadata["tags"]["preprocessing_config_sha256"]
            )
            if common_pilot_hash is None:
                common_pilot_hash = pilot_hash
                common_evaluation_hash = evaluation_hash
                common_preprocessing_hash = preprocessing_hash
            elif (
                pilot_hash != common_pilot_hash
                or evaluation_hash != common_evaluation_hash
                or preprocessing_hash != common_preprocessing_hash
            ):
                raise RuntimeError("Selected checkpoints do not share frozen configs")
        candidate_entries.sort(key=lambda entry: int(entry["seed"]))
        actual_seeds = [int(entry["seed"]) for entry in candidate_entries]
        planned_seeds = [
            int(seed) for seed in cast(Sequence[Any], expected_seeds[candidate])
        ]
        if actual_seeds != planned_seeds:
            raise RuntimeError(
                f"Selected seeds for {candidate} are {actual_seeds}, "
                f"expected {planned_seeds}"
            )
        entries.extend(candidate_entries)

    if common_pilot_hash != file_digest(paths.gate9_pilot_config):
        raise RuntimeError("Gate 9 pilot config hash changed after training")
    if common_evaluation_hash != file_digest(paths.evaluation_config):
        raise RuntimeError("Evaluation config hash changed after training")
    if common_preprocessing_hash != file_digest(paths.preprocessing_config):
        raise RuntimeError("Preprocessing config hash changed after training")
    return {
        "status": "frozen",
        "gate": 10,
        "frozen_at_git_commit": git_commit,
        "selection_rule": str(candidates["checkpoint_selection"]),
        "seed_ensemble": False,
        "seed_aggregation": str(candidates["seed_aggregation"]),
        "ordered_candidates": ordered,
        "primary_candidate": str(candidates["primary"]),
        "mandatory_reference": str(candidates["mandatory_reference"]),
        "gate9_analysis_path": paths.gate9_analysis.as_posix(),
        "gate9_analysis_sha256": file_digest(paths.gate9_analysis),
        "gate9_pilot_config_sha256": common_pilot_hash,
        "evaluation_config_sha256": common_evaluation_hash,
        "preprocessing_config_sha256": common_preprocessing_hash,
        "checkpoint_count": len(entries),
        "checkpoints": entries,
    }


def build_analysis_freeze(
    paths: Gate10Paths,
    plan: Mapping[str, Any],
    frozen_split: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    *,
    git_commit: str,
) -> dict[str, Any]:
    """Resolve data-derived development thresholds and pin all freeze hashes."""
    subgroup_definitions = _mapping(
        _mapping(plan["subgroups"], "subgroups")["definitions"],
        "subgroup definitions",
    )
    burden = _mapping(subgroup_definitions["whole_tumor_burden"], "burden")
    thresholds = development_tumor_thresholds(
        paths.frozen_split_dir / "train.csv",
        [float(value) for value in cast(Sequence[Any], burden["quantiles"])],
        column=str(burden["source_column"]),
        method=str(burden["quantile_method"]),
    )
    return {
        "status": "frozen",
        "gate": 10,
        "internal_test_accessed": False,
        "frozen_at_git_commit": git_commit,
        "plan_path": paths.config_path.as_posix(),
        "plan_sha256": file_digest(paths.config_path),
        "checkpoint_manifest_path": paths.checkpoint_manifest.as_posix(),
        "checkpoint_manifest_sha256": file_digest(paths.checkpoint_manifest),
        "frozen_split_metadata_path": (
            paths.frozen_split_dir / "split_metadata.json"
        ).as_posix(),
        "frozen_split_metadata_sha256": file_digest(
            paths.frozen_split_dir / "split_metadata.json"
        ),
        "manifest_sha256": dict(frozen_split["manifest_sha256"]),
        "checkpoint_count": int(checkpoint_manifest["checkpoint_count"]),
        "ordered_candidates": list(checkpoint_manifest["ordered_candidates"]),
        "resolved_subgroups": {
            "whole_tumor_burden_train_only_tertiles": thresholds,
            "boundaries": str(burden["boundaries"]),
        },
        "primary_endpoint": _mapping(
            _mapping(plan["endpoints"], "endpoints")["primary"],
            "primary",
        ),
        "hypothesis_testing": _mapping(
            plan["hypothesis_testing"],
            "hypothesis_testing",
        ),
        "conduct": _mapping(plan["conduct"], "conduct"),
    }


def _completion_markdown(
    plan: Mapping[str, Any],
    frozen_split: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    analysis: Mapping[str, Any],
) -> str:
    candidates = _mapping(plan["candidates"], "candidates")
    expected_seeds = _mapping(candidates["expected_seeds"], "expected_seeds")
    thresholds = _mapping(
        _mapping(analysis["resolved_subgroups"], "resolved_subgroups")[
            "whole_tumor_burden_train_only_tertiles"
        ],
        "thresholds",
    )
    lines = [
        "# Gate 10 Completion",
        "",
        "**Decision:** PASS — internal-test evaluation is now protocol-eligible.",
        "",
        "## Frozen split",
        "",
        f"- Train patients: {frozen_split['counts']['train']}",
        f"- Validation patients: {frozen_split['counts']['validation']}",
        f"- Internal held-out test patients: {frozen_split['counts']['test']}",
        "- Membership copied byte-for-byte from the passing provisional split.",
        "- Internal-test outcomes were not evaluated during the freeze.",
        "",
        "## Frozen candidates",
        "",
        "| Candidate | Seeds | Checkpoints |",
        "|---|---|---:|",
    ]
    checkpoints = cast(
        Sequence[Mapping[str, Any]],
        checkpoint_manifest["checkpoints"],
    )
    for candidate in cast(Sequence[str], checkpoint_manifest["ordered_candidates"]):
        candidate_rows = [
            entry for entry in checkpoints if entry["candidate_id"] == candidate
        ]
        seeds = ", ".join(
            str(seed) for seed in cast(Sequence[Any], expected_seeds[candidate])
        )
        lines.append(f"| {candidate} | {seeds} | {len(candidate_rows)} |")
    lines.extend(
        [
            "",
            f"Primary candidate: `{candidates['primary']}`. Every checkpoint is "
            "evaluated separately; no seed ensemble is permitted.",
            "",
            "## Frozen inference and statistics",
            "",
            "- Statistical unit: patient.",
            "- Primary endpoint: patient mean of WT, TC, and ET Dice.",
            "- Candidate value: per-patient arithmetic mean across frozen seeds.",
            "- Confidence intervals: 10,000 paired patient bootstrap resamples.",
            "- Hypothesis tests: 100,000 two-sided paired sign-flip permutations.",
            "- Multiplicity: Holm correction over three primary-endpoint comparisons.",
            "- Secondary endpoints and subgroups: estimation only.",
            "- Post-processing, threshold tuning, checkpoint replacement, and model "
            "selection after test access are prohibited.",
            "",
            "## Development-derived subgroup thresholds",
            "",
            f"- Small WT burden: ≤ {float(thresholds['q1_mm3']):.6f} mm³",
            f"- Medium WT burden: > {float(thresholds['q1_mm3']):.6f} and "
            f"≤ {float(thresholds['q2_mm3']):.6f} mm³",
            f"- Large WT burden: > {float(thresholds['q2_mm3']):.6f} mm³",
            "",
            "These tertiles were derived from the 258 training patients only.",
            "",
            "## Guard",
            "",
            "The append-only internal-test audit contained no prior test-manifest "
            "access event. Gate 11 must use the exact split, checkpoint, evaluator, "
            "preprocessing, and statistical-plan hashes frozen here.",
        ]
    )
    return "\n".join(lines)


def execute_gate10_freeze(paths: Gate10Paths, plan: Mapping[str, Any]) -> None:
    """Create all Gate 10 artifacts without opening the internal-test manifest."""
    assert_clean_repository()
    assert_no_internal_test_access(paths.test_access_audit)
    git_commit = _git_commit()
    frozen_split = freeze_split_membership(
        paths,
        plan,
        git_commit=git_commit,
    )
    checkpoint_manifest = build_checkpoint_manifest(
        paths,
        plan,
        git_commit=git_commit,
    )
    atomic_write_json(paths.checkpoint_manifest, checkpoint_manifest)
    analysis = build_analysis_freeze(
        paths,
        plan,
        frozen_split,
        checkpoint_manifest,
        git_commit=git_commit,
    )
    atomic_write_json(paths.analysis_freeze, analysis)
    atomic_write_text(
        paths.completion_report,
        _completion_markdown(plan, frozen_split, checkpoint_manifest, analysis),
    )
