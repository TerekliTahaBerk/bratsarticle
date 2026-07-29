"""Deterministic five-fold development and locked external-test design."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.stats import nct, t
from sklearn.model_selection import StratifiedKFold

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)


class StudyDesignError(RuntimeError):
    """Raised when the frozen cohort design would violate a study invariant."""


@dataclass(frozen=True)
class FoldDesign:
    """Selected deterministic cross-validation assignment."""

    assignment: np.ndarray
    seed: int
    candidate_index: int
    balance_objective: float


def _quartile(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=4, labels=("Q1", "Q2", "Q3", "Q4"))


def _development_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "subject_id",
        "dataset",
        "grade",
        "eligible",
        "wt_voxel_count",
        "tc_voxel_count",
        "et_voxel_count",
        "voxel_volume_mm3",
        "t1_sha256",
        "t1ce_sha256",
        "t2_sha256",
        "flair_sha256",
        "seg_sha256",
    }
    missing = required - set(frame.columns)
    if missing:
        raise StudyDesignError(f"Canonical manifest missing fields: {sorted(missing)}")
    if len(frame) != 369:
        raise StudyDesignError(f"Expected 369 development patients, found {len(frame)}")
    if not frame["subject_id"].is_unique:
        raise StudyDesignError("Development subject identifiers are not unique")
    if not frame["eligible"].astype(bool).all():
        raise StudyDesignError("Development manifest contains ineligible patients")

    enriched = frame.copy()
    voxel_volume = enriched["voxel_volume_mm3"].astype(float)
    for region in ("wt", "tc", "et"):
        enriched[f"{region}_volume_mm3"] = (
            enriched[f"{region}_voxel_count"].astype(float) * voxel_volume
        )
    enriched["et_present"] = enriched["et_voxel_count"].astype(int).gt(0)
    enriched["wt_volume_quartile"] = _quartile(enriched["wt_volume_mm3"])
    enriched["primary_stratum"] = (
        enriched["grade"].astype(str)
        + "|"
        + enriched["et_present"].astype(int).astype(str)
        + "|"
        + enriched["wt_volume_quartile"].astype(str)
    )
    if int(enriched["primary_stratum"].value_counts().min()) < 5:
        raise StudyDesignError("A primary stratum has fewer patients than folds")
    return enriched


def _fold_balance_objective(frame: pd.DataFrame, assignment: np.ndarray) -> float:
    categorical = ("grade", "et_present", "wt_volume_quartile")
    continuous = ("wt_volume_mm3", "tc_volume_mm3", "et_volume_mm3")
    score = 0.0
    for feature in categorical:
        global_distribution = frame[feature].astype(str).value_counts(normalize=True)
        for fold in range(1, 6):
            distribution = (
                frame.loc[assignment == fold, feature]
                .astype(str)
                .value_counts(normalize=True)
            )
            for category, prevalence in global_distribution.items():
                difference = float(distribution.get(category, 0.0)) - float(prevalence)
                score += difference * difference
    for feature in continuous:
        values = np.log1p(frame[feature].astype(float).to_numpy())
        global_mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1))
        if standard_deviation == 0:
            continue
        for fold in range(1, 6):
            fold_mean = float(np.mean(values[assignment == fold]))
            score += 0.25 * ((fold_mean - global_mean) / standard_deviation) ** 2
    return score


def select_five_fold_design(
    frame: pd.DataFrame,
    *,
    seed: int,
    candidate_count: int = 1000,
) -> FoldDesign:
    """Select the best deterministic stratified assignment from fixed seeds."""
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive")
    strata = frame["primary_stratum"].astype(str).to_numpy()
    placeholder = np.zeros(len(frame), dtype=np.int8)
    best: FoldDesign | None = None
    for candidate_index in range(candidate_count):
        candidate_seed = seed + candidate_index
        splitter = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=candidate_seed,
        )
        assignment = np.zeros(len(frame), dtype=np.int8)
        for fold_index, (_, validation_indices) in enumerate(
            splitter.split(placeholder, strata),
            start=1,
        ):
            assignment[validation_indices] = fold_index
        objective = _fold_balance_objective(frame, assignment)
        candidate = FoldDesign(
            assignment=assignment,
            seed=candidate_seed,
            candidate_index=candidate_index,
            balance_objective=objective,
        )
        if best is None or (
            candidate.balance_objective,
            candidate.candidate_index,
        ) < (best.balance_objective, best.candidate_index):
            best = candidate
    if best is None:
        raise StudyDesignError("No cross-validation assignment was generated")
    return best


def _fold_frame(
    frame: pd.DataFrame,
    assignment: np.ndarray,
    fold: int,
    manifest_sha256: str,
) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "dataset": frame["dataset"].astype(str),
            "subject_id": frame["subject_id"].astype(str),
            "fold": fold,
            "role": np.where(assignment == fold, "validation", "train"),
            "grade": frame["grade"].astype(str),
            "et_present": frame["et_present"].astype(bool),
            "wt_volume_mm3": frame["wt_volume_mm3"].astype(float),
            "tc_volume_mm3": frame["tc_volume_mm3"].astype(float),
            "et_volume_mm3": frame["et_volume_mm3"].astype(float),
            "wt_volume_quartile": frame["wt_volume_quartile"].astype(str),
            "primary_stratum": frame["primary_stratum"].astype(str),
            "canonical_manifest_sha256": manifest_sha256,
        }
    )
    return output.sort_values(["role", "subject_id"], kind="stable").reset_index(
        drop=True
    )


def _validate_assignment(frame: pd.DataFrame, assignment: np.ndarray) -> None:
    if assignment.shape != (369,):
        raise StudyDesignError("Fold assignment does not cover 369 patients")
    if set(assignment.tolist()) != {1, 2, 3, 4, 5}:
        raise StudyDesignError("Fold assignment must contain folds 1 through 5")
    fold_sizes = pd.Series(assignment).value_counts().sort_index().tolist()
    if fold_sizes != [74, 74, 74, 74, 73]:
        raise StudyDesignError(f"Unexpected validation-fold sizes: {fold_sizes}")
    if not frame["subject_id"].is_unique:
        raise StudyDesignError("A development patient occurs more than once")


def _external_test_frame(frame: pd.DataFrame, inventory_sha256: str) -> pd.DataFrame:
    required = {
        "patient_id",
        "disease_group",
        "primary_confirmatory_eligibility",
        "eligibility_status",
        "t1_path",
        "t1ce_path",
        "t2_path",
        "flair_path",
        "label_path",
        "institution",
        "scanner_vendor",
        "scanner_model",
        "field_strength_t",
        "wt_voxel_count",
        "tc_voxel_count",
        "et_voxel_count",
        "label_mapping",
    }
    missing = required - set(frame.columns)
    if missing:
        raise StudyDesignError(f"External inventory missing fields: {sorted(missing)}")
    eligible = frame.loc[
        frame["primary_confirmatory_eligibility"].astype(str).eq("eligible")
        & frame["eligibility_status"].astype(str).eq("eligible")
        & frame["disease_group"].astype(str).eq("glioma")
    ].copy()
    if len(eligible) != 95:
        raise StudyDesignError(
            f"Expected 95 eligible external glioma patients, found {len(eligible)}"
        )
    if not eligible["patient_id"].is_unique:
        raise StudyDesignError("External patient identifiers are not unique")
    output = eligible[
        [
            "patient_id",
            "disease_group",
            "institution",
            "scanner_vendor",
            "scanner_model",
            "field_strength_t",
            "t1_path",
            "t1ce_path",
            "t2_path",
            "flair_path",
            "label_path",
            "label_mapping",
            "wt_voxel_count",
            "tc_voxel_count",
            "et_voxel_count",
        ]
    ].copy()
    output.insert(0, "dataset", "BraTS-Africa-TCIA-v1")
    output.insert(2, "role", "external_confirmatory_test")
    output["external_inventory_sha256"] = inventory_sha256
    output["result_access_policy"] = "single_frozen_inference_after_gate_g"
    return output.sort_values("patient_id", kind="stable").reset_index(drop=True)


def _planning_precision(
    *,
    sample_size: int,
    historical_difference_sd: float,
    alpha: float = 0.05,
) -> dict[str, Any]:
    degrees_of_freedom = sample_size - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
    half_width = critical * historical_difference_sd / np.sqrt(sample_size)
    detectable_effects = (0.01, 0.015, 0.02, 0.025, 0.03)
    power: dict[str, float] = {}
    for effect in detectable_effects:
        noncentrality = effect * np.sqrt(sample_size) / historical_difference_sd
        probability = float(
            nct.cdf(-critical, degrees_of_freedom, noncentrality)
            + 1.0
            - nct.cdf(critical, degrees_of_freedom, noncentrality)
        )
        power[f"{effect:.3f}"] = probability
    return {
        "sample_size": sample_size,
        "alpha_two_sided": alpha,
        "historical_paired_difference_sd": historical_difference_sd,
        "historical_source": (
            "legacy internal patient-level U-Net+RES minus U-Net mean-regional "
            "Dice after averaging declared seeds; planning only"
        ),
        "expected_95_percent_ci_half_width": float(half_width),
        "power_by_true_mean_difference": power,
        "practical_effect_threshold": 0.02,
        "threshold_role": (
            "interpretive SESOI for superiority, not an equivalence or "
            "non-inferiority margin"
        ),
    }


def prepare_design(
    *,
    canonical_manifest: Path,
    external_inventory: Path,
    gate_c_summary: Path,
    fold_output_dir: Path,
    external_test_output: Path,
    metadata_output: Path,
    protocol_report_output: Path,
    precision_json_output: Path,
    precision_report_output: Path,
    seed: int = 20260730,
    candidate_count: int = 1000,
    historical_difference_sd: float = 0.05887957821997264,
) -> dict[str, Any]:
    """Create folds, a locked external manifest, and pre-result precision evidence."""
    gate_c = json.loads(gate_c_summary.read_text(encoding="utf-8"))
    if not gate_c.get("gate_c_pass"):
        raise StudyDesignError("Gate C must pass before cohort design")
    if gate_c.get("external_results_accessed") or gate_c.get("model_inference_run"):
        raise StudyDesignError("External results were accessed before design freeze")

    canonical_sha256 = file_digest(canonical_manifest)
    inventory_sha256 = file_digest(external_inventory)
    development = _development_features(pd.read_csv(canonical_manifest))
    external = pd.read_csv(external_inventory)
    selected = select_five_fold_design(
        development,
        seed=seed,
        candidate_count=candidate_count,
    )
    _validate_assignment(development, selected.assignment)

    fold_output_dir.mkdir(parents=True, exist_ok=True)
    fold_hashes: dict[str, str] = {}
    fold_counts: dict[str, dict[str, int]] = {}
    for fold in range(1, 6):
        fold_path = fold_output_dir / f"cv_fold_{fold}.csv"
        fold_frame = _fold_frame(
            development,
            selected.assignment,
            fold,
            canonical_sha256,
        )
        atomic_write_csv(
            fold_path,
            cast(list[dict[str, Any]], fold_frame.to_dict(orient="records")),
        )
        fold_hashes[f"cv_fold_{fold}"] = file_digest(fold_path)
        fold_counts[f"cv_fold_{fold}"] = {
            "train": int(fold_frame["role"].eq("train").sum()),
            "validation": int(fold_frame["role"].eq("validation").sum()),
        }

    external_frame = _external_test_frame(external, inventory_sha256)
    atomic_write_csv(
        external_test_output,
        cast(list[dict[str, Any]], external_frame.to_dict(orient="records")),
    )
    external_test_sha256 = file_digest(external_test_output)

    precision = _planning_precision(
        sample_size=len(external_frame),
        historical_difference_sd=historical_difference_sd,
    )
    atomic_write_json(precision_json_output, precision)
    precision_lines = [
        "# External confirmatory precision and power plan",
        "",
        "This calculation was completed before model inference on the external cohort.",
        "It is a planning analysis, not a result.",
        "",
        f"- Eligible primary external glioma patients: {len(external_frame)}",
        (
            "- Historical planning SD of paired patient differences: "
            f"{historical_difference_sd:.6f}"
        ),
        (
            "- Expected two-sided 95% CI half-width under that SD: "
            f"{precision['expected_95_percent_ci_half_width']:.4f} Dice"
        ),
        "- Prespecified practical-effect threshold: 0.020 mean-regional Dice.",
        (
            "- Rationale: a two-percentage-point average across WT, TC and ET is "
            "large enough to require a distributed regional gain rather than a "
            "rounding-level change. It is an interpretation threshold, not a "
            "clinical MCID or an equivalence/non-inferiority margin."
        ),
        "",
        "| True paired mean difference | Approximate two-sided power |",
        "|---:|---:|",
    ]
    for effect, probability in precision["power_by_true_mean_difference"].items():
        precision_lines.append(f"| {float(effect):.3f} | {probability:.3f} |")
    precision_lines.extend(
        [
            "",
            (
                "The planning SD is artifact-derived from the legacy internal "
                "patient-level U-Net+RES versus U-Net contrast after seed averaging. "
                "Because the new parameter-matched comparator and African cohort "
                "may have a different variance, final inference will emphasize "
                "the observed paired confidence interval."
            ),
        ]
    )
    atomic_write_text(precision_report_output, "\n".join(precision_lines) + "\n")

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "cohort_membership_frozen_before_external_inference",
        "development": {
            "dataset": "BraTS 2020 training",
            "patient_count": len(development),
            "canonical_manifest": canonical_manifest.as_posix(),
            "canonical_manifest_sha256": canonical_sha256,
            "fold_count": 5,
            "assignment_seed": selected.seed,
            "candidate_index": selected.candidate_index,
            "candidate_count": candidate_count,
            "balance_objective": selected.balance_objective,
            "fold_counts": fold_counts,
            "fold_sha256": fold_hashes,
            "stratification": ["grade", "ET presence", "WT volume quartile"],
        },
        "external": {
            "dataset": "BraTS-Africa-TCIA-v1",
            "primary_patient_count": len(external_frame),
            "supportive_other_neoplasm_count": 51,
            "inventory_sha256": inventory_sha256,
            "external_test_sha256": external_test_sha256,
            "results_accessed": False,
            "model_inference_run": False,
        },
        "legacy_internal_test": {
            "patient_count": 74,
            "permitted_for_new_models": False,
            "role": "legacy replication appendix only",
        },
        "preprocessing": {
            "cohort_level_parameters": "none",
            "per_volume_nonzero_voxel_normalization": True,
            "fold_training_only_rule": (
                "Any learned normalization, sampling, augmentation, threshold, "
                "or post-processing parameter must be derived from that fold's "
                "training rows only."
            ),
        },
        "precision_plan_sha256": file_digest(precision_json_output),
    }
    atomic_write_json(metadata_output, metadata)

    report_lines = [
        "# Split and cohort protocol",
        "",
        "Status: **frozen before external model inference**",
        "",
        "## Development cohort",
        "",
        (
            "All 369 unique BraTS 2020 training patients form the development "
            "cohort. A deterministic candidate search selected five "
            "patient-level stratified folds using grade, ET presence, and WT "
            "volume quartile."
        ),
        "",
        "| Fold | Training patients | Validation patients |",
        "|---:|---:|---:|",
    ]
    for fold in range(1, 6):
        counts = fold_counts[f"cv_fold_{fold}"]
        report_lines.append(
            f"| {fold} | {counts['train']} | {counts['validation']} |"
        )
    report_lines.extend(
        [
            "",
            (
                "Every patient is a validation case in exactly one fold. No slices "
                "cross patient or fold boundaries. The legacy 258/37/74 partition "
                "is not used for v2 development, and its 74-patient internal subset "
                "is prohibited for all new-model inference."
            ),
            "",
            "## External confirmatory cohort",
            "",
            (
                "The primary external manifest contains 95 eligible glioma patients "
                "from the processed BraTS-Africa TCIA v1 release. The 51 other-"
                "neoplasm cases are excluded from confirmatory inference and may "
                "only be used in a separately labelled supportive analysis."
            ),
            "",
            (
                "No external model prediction, metric, threshold selection, "
                "post-processing choice, or adaptation was performed during cohort "
                "design. External inference is allowed once only after the complete "
                "model/checkpoint and statistical freeze passes Gate G."
            ),
            "",
            "## Preprocessing isolation",
            "",
            (
                "The current normalization is per volume over nonzero voxels and "
                "does not estimate cohort-wide parameters. Any future learned "
                "normalization, sampler, augmentation, threshold, calibration, or "
                "post-processing value is fitted using training rows of the current "
                "fold only. External labels are never used for adaptation."
            ),
            "",
            "## Machine-readable anchors",
            "",
            f"- Canonical development manifest SHA-256: `{canonical_sha256}`",
            f"- External inventory SHA-256: `{inventory_sha256}`",
            f"- External test manifest SHA-256: `{external_test_sha256}`",
            f"- Selected assignment seed: `{selected.seed}`",
            f"- Candidate index: `{selected.candidate_index}` of `{candidate_count}`",
        ]
    )
    atomic_write_text(protocol_report_output, "\n".join(report_lines) + "\n")
    return metadata


def design_digest(paths: list[Path]) -> str:
    """Return a stable digest over named study-design files."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.as_posix()):
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()
