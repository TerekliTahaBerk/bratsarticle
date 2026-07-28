"""Deterministic patient-level split generation and guarded test access."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import StratifiedShuffleSplit

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SplitName = Literal["train", "validation", "test"]
DevelopmentSplitName = Literal["train", "validation"]

_SPLIT_ORDER: tuple[SplitName, ...] = ("train", "validation", "test")
_IMAGE_HASH_COLUMNS = tuple(f"{role}_sha256" for role in ("t1", "t1ce", "t2", "flair"))
_FILE_HASH_COLUMNS = (*_IMAGE_HASH_COLUMNS, "seg_sha256")


class SplitIntegrityError(RuntimeError):
    """Raised when a proposed split violates patient-level safeguards."""


@dataclass(frozen=True)
class SplitSettings:
    """Resolved patient-level split configuration."""

    canonical_manifest: Path
    output_dir: Path
    figure_dir: Path
    report_path: Path
    metadata_path: Path
    seed: int
    candidate_seeds: int
    counts: Mapping[SplitName, int]
    max_categorical_prevalence_deviation: float
    max_absolute_standardized_mean_difference: float


@dataclass(frozen=True)
class CandidateSplit:
    """Membership and balance objective for one deterministic candidate."""

    membership: Mapping[SplitName, np.ndarray]
    objective: float
    candidate_index: int


def _quartile(series: pd.Series) -> pd.Series:
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=4, labels=("Q1", "Q2", "Q3", "Q4"))


def add_stratification_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive patient-level tumor-burden features without using test outcomes."""
    enriched = frame.copy()
    required = {
        "subject_id",
        "grade",
        "eligible",
        "wt_voxel_count",
        "tc_voxel_count",
        "et_voxel_count",
        "voxel_volume_mm3",
        *_IMAGE_HASH_COLUMNS,
    }
    missing = required - set(enriched.columns)
    if missing:
        raise SplitIntegrityError(
            f"Canonical manifest is missing split fields: {sorted(missing)}"
        )
    if not enriched["subject_id"].is_unique:
        raise SplitIntegrityError("Canonical subject IDs are not unique")
    if not enriched["eligible"].astype(bool).all():
        ineligible = enriched.loc[
            ~enriched["eligible"].astype(bool), "subject_id"
        ].tolist()
        raise SplitIntegrityError(
            f"Canonical manifest contains ineligible subjects: {ineligible}"
        )

    voxel_volume = enriched["voxel_volume_mm3"].astype(float)
    for region in ("wt", "tc", "et"):
        enriched[f"{region}_volume_mm3"] = (
            enriched[f"{region}_voxel_count"].astype(float) * voxel_volume
        )
    enriched["total_tumor_volume_mm3"] = enriched["wt_volume_mm3"]
    enriched["et_present"] = enriched["et_voxel_count"].astype(float).gt(0)
    enriched["wt_volume_quartile"] = _quartile(enriched["wt_volume_mm3"])
    enriched["tc_volume_quartile"] = _quartile(enriched["tc_volume_mm3"])
    enriched["total_tumor_volume_quartile"] = enriched["wt_volume_quartile"]
    enriched["et_volume_quartile"] = "absent"
    et_present = enriched["et_present"]
    if int(et_present.sum()) >= 4:
        enriched.loc[et_present, "et_volume_quartile"] = (
            _quartile(enriched.loc[et_present, "et_volume_mm3"])
            .astype("string")
            .to_numpy()
        )
    enriched["primary_stratum"] = (
        enriched["grade"].astype(str)
        + "|"
        + enriched["et_present"].astype(int).astype(str)
        + "|"
        + enriched["wt_volume_quartile"].astype(str)
    )
    return enriched


def assert_no_duplicate_image_signatures(frame: pd.DataFrame) -> None:
    """Reject exact within-cohort image duplicates before partitioning."""
    signature = frame[list(_IMAGE_HASH_COLUMNS)].astype(str).agg("|".join, axis=1)
    duplicate_mask = signature.duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_subjects = frame.loc[duplicate_mask, "subject_id"].tolist()
        raise SplitIntegrityError(
            "Exact image signatures occur in multiple canonical subjects: "
            f"{duplicate_subjects}"
        )


def assert_no_duplicate_file_hashes(frame: pd.DataFrame) -> None:
    """Reject exact same-role files assigned to distinct canonical subjects."""
    duplicate_records: list[str] = []
    for column in _FILE_HASH_COLUMNS:
        if column not in frame.columns:
            continue
        nonempty = frame.loc[frame[column].astype(str).ne(""), ["subject_id", column]]
        duplicate_mask = nonempty[column].duplicated(keep=False)
        if duplicate_mask.any():
            subjects = ",".join(
                sorted(nonempty.loc[duplicate_mask, "subject_id"].astype(str))
            )
            duplicate_records.append(f"{column}:{subjects}")
    if duplicate_records:
        raise SplitIntegrityError(
            "Exact file hashes occur in multiple canonical subjects: "
            + "; ".join(duplicate_records)
        )


def _candidate_membership(
    frame: pd.DataFrame,
    counts: Mapping[SplitName, int],
    seed: int,
) -> dict[SplitName, np.ndarray]:
    indices = np.arange(len(frame))
    strata = frame["primary_stratum"].astype(str).to_numpy()
    test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=counts["test"],
        random_state=seed,
    )
    development_indices, test_indices = next(test_splitter.split(indices, strata))
    development_strata = strata[development_indices]
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=counts["validation"],
        random_state=seed + 1,
    )
    train_relative, validation_relative = next(
        validation_splitter.split(development_indices, development_strata)
    )
    return {
        "train": np.sort(development_indices[train_relative]),
        "validation": np.sort(development_indices[validation_relative]),
        "test": np.sort(test_indices),
    }


def _balance_objective(
    frame: pd.DataFrame,
    membership: Mapping[SplitName, np.ndarray],
) -> float:
    categorical = (
        "grade",
        "et_present",
        "wt_volume_quartile",
        "tc_volume_quartile",
        "et_volume_quartile",
    )
    continuous = ("wt_volume_mm3", "tc_volume_mm3", "et_volume_mm3")
    score = 0.0
    for feature in categorical:
        global_distribution = frame[feature].astype(str).value_counts(normalize=True)
        for split in _SPLIT_ORDER:
            split_distribution = (
                frame.iloc[membership[split]][feature]
                .astype(str)
                .value_counts(normalize=True)
            )
            for category, global_prevalence in global_distribution.items():
                deviation = float(split_distribution.get(category, 0.0)) - float(
                    global_prevalence
                )
                score += deviation * deviation
    for feature in continuous:
        values = np.log1p(frame[feature].astype(float).to_numpy())
        standard_deviation = float(np.std(values, ddof=1))
        if standard_deviation == 0:
            continue
        global_mean = float(np.mean(values))
        for split in _SPLIT_ORDER:
            split_mean = float(np.mean(values[membership[split]]))
            score += 0.25 * ((split_mean - global_mean) / standard_deviation) ** 2
    return score


def select_candidate(
    frame: pd.DataFrame,
    settings: SplitSettings,
) -> CandidateSplit:
    """Select the best deterministic stratified candidate without test labels."""
    candidates: list[CandidateSplit] = []
    for candidate_index in range(settings.candidate_seeds):
        membership = _candidate_membership(
            frame,
            settings.counts,
            seed=settings.seed + 2 * candidate_index,
        )
        candidates.append(
            CandidateSplit(
                membership=membership,
                objective=_balance_objective(frame, membership),
                candidate_index=candidate_index,
            )
        )
    return min(
        candidates,
        key=lambda candidate: (candidate.objective, candidate.candidate_index),
    )


def validate_membership(
    frame: pd.DataFrame,
    membership: Mapping[SplitName, np.ndarray],
    counts: Mapping[SplitName, int],
) -> None:
    """Enforce exact patient-level coverage and non-overlap."""
    subject_sets: dict[SplitName, set[str]] = {}
    for split in _SPLIT_ORDER:
        if len(membership[split]) != counts[split]:
            raise SplitIntegrityError(
                f"{split} has {len(membership[split])} subjects; "
                f"expected {counts[split]}"
            )
        subject_sets[split] = set(
            frame.iloc[membership[split]]["subject_id"].astype(str)
        )
    for index, left in enumerate(_SPLIT_ORDER):
        for right in _SPLIT_ORDER[index + 1 :]:
            overlap = subject_sets[left] & subject_sets[right]
            if overlap:
                raise SplitIntegrityError(
                    f"Subject overlap between {left} and {right}: {sorted(overlap)}"
                )
    union = set().union(*subject_sets.values())
    expected = set(frame["subject_id"].astype(str))
    if union != expected:
        raise SplitIntegrityError(
            "Split membership does not cover the canonical cohort exactly"
        )


def balance_tables(
    frame: pd.DataFrame,
    membership: Mapping[SplitName, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return categorical prevalence and continuous SMD balance tables."""
    categorical_rows: list[dict[str, Any]] = []
    categorical = (
        "grade",
        "et_present",
        "wt_volume_quartile",
        "tc_volume_quartile",
        "et_volume_quartile",
    )
    for feature in categorical:
        global_distribution = frame[feature].astype(str).value_counts(normalize=True)
        categories = sorted(global_distribution.index)
        for split in _SPLIT_ORDER:
            split_values = frame.iloc[membership[split]][feature].astype(str)
            split_distribution = split_values.value_counts(normalize=True)
            for category in categories:
                prevalence = float(split_distribution.get(category, 0.0))
                global_prevalence = float(global_distribution[category])
                categorical_rows.append(
                    {
                        "split": split,
                        "feature": feature,
                        "category": category,
                        "count": int((split_values == category).sum()),
                        "prevalence": prevalence,
                        "global_prevalence": global_prevalence,
                        "absolute_deviation": abs(prevalence - global_prevalence),
                    }
                )

    continuous_rows: list[dict[str, Any]] = []
    for feature in ("wt_volume_mm3", "tc_volume_mm3", "et_volume_mm3"):
        values = np.log1p(frame[feature].astype(float).to_numpy())
        global_mean = float(np.mean(values))
        global_standard_deviation = float(np.std(values, ddof=1))
        for split in _SPLIT_ORDER:
            split_values = values[membership[split]]
            smd = (
                (float(np.mean(split_values)) - global_mean) / global_standard_deviation
                if global_standard_deviation
                else 0.0
            )
            continuous_rows.append(
                {
                    "split": split,
                    "feature": feature,
                    "log1p_mean": float(np.mean(split_values)),
                    "global_log1p_mean": global_mean,
                    "standardized_mean_difference": smd,
                }
            )
    return pd.DataFrame(categorical_rows), pd.DataFrame(continuous_rows)


def _plot_categorical(
    table: pd.DataFrame,
    feature: str,
    destination: Path,
    title: str,
) -> None:
    subset = table.loc[table["feature"] == feature].copy()
    categories = sorted(subset["category"].unique())
    x_positions = np.arange(len(categories), dtype=float)
    width = 0.24
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for offset_index, split in enumerate(_SPLIT_ORDER):
        values = [
            float(
                subset.loc[
                    (subset["split"] == split) & (subset["category"] == category),
                    "prevalence",
                ].iloc[0]
            )
            for category in categories
        ]
        axis.bar(
            x_positions + (offset_index - 1) * width,
            values,
            width=width,
            label=split,
        )
    axis.set_xticks(x_positions, categories)
    axis.set_ylabel("Patient proportion")
    axis.set_title(title)
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _plot_smd(table: pd.DataFrame, destination: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {"train": "#35618f", "validation": "#db7c26", "test": "#3a8f5c"}
    for row in table.to_dict(orient="records"):
        labels.append(f"{row['split']} · {row['feature'].replace('_mm3', '')}")
        values.append(float(row["standardized_mean_difference"]))
        colors.append(palette[str(row["split"])])
    positions = np.arange(len(labels))
    axis.barh(positions, values, color=colors)
    axis.set_yticks(positions, labels)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.1, color="grey", linestyle="--", linewidth=0.8)
    axis.axvline(-0.1, color="grey", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Standardized mean difference (log1p volume)")
    axis.set_title("Continuous tumor-burden balance")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=200)
    plt.close(figure)


def _render_report(
    settings: SplitSettings,
    candidate: CandidateSplit,
    categorical: pd.DataFrame,
    continuous: pd.DataFrame,
    hashes: Mapping[SplitName, str],
    balance_pass: bool,
) -> str:
    maximum_categorical = float(categorical["absolute_deviation"].max())
    maximum_smd = float(continuous["standardized_mean_difference"].abs().max())
    lines = [
        "# Provisional Patient-Level Split Balance",
        "",
        "**Status:** " + ("PASS" if balance_pass else "FAIL"),
        "",
        "## Split definition",
        "",
        f"- Seed: `{settings.seed}`",
        f"- Candidate search size: {settings.candidate_seeds}",
        f"- Selected candidate index: {candidate.candidate_index}",
        f"- Balance objective: {candidate.objective:.8f}",
        f"- Train subjects: {settings.counts['train']}",
        f"- Validation subjects: {settings.counts['validation']}",
        f"- Internal held-out test subjects: {settings.counts['test']}",
        "",
        "All partitions are patient-level. The internal held-out test subset is "
        "not available through the development loader.",
        "Exact same-role file hashes were checked globally before partitioning; "
        "no cross-patient duplicate was found.",
        "",
        "## Balance acceptance",
        "",
        f"- Maximum categorical prevalence deviation: {maximum_categorical:.4f} "
        f"(limit {settings.max_categorical_prevalence_deviation:.4f})",
        f"- Maximum absolute SMD: {maximum_smd:.4f} "
        f"(limit {settings.max_absolute_standardized_mean_difference:.4f})",
        "",
        "## Manifest hashes",
        "",
        "| Manifest | SHA-256 |",
        "|---|---|",
    ]
    for name in _SPLIT_ORDER:
        lines.append(f"| {name} | `{hashes[name]}` |")
    lines.extend(
        [
            "",
            "## Continuous balance",
            "",
            "| Split | Feature | SMD |",
            "|---|---|---:|",
        ]
    )
    for row in continuous.to_dict(orient="records"):
        lines.append(
            f"| {row['split']} | {row['feature']} | "
            f"{float(row['standardized_mean_difference']):.4f} |"
        )
    return "\n".join(lines)


def settings_from_config(config: DictConfig) -> SplitSettings:
    """Resolve the split configuration."""
    counts: dict[SplitName, int] = {
        name: int(config.split.counts[name]) for name in _SPLIT_ORDER
    }
    return SplitSettings(
        canonical_manifest=Path(str(config.split.canonical_manifest)).resolve(),
        output_dir=Path(str(config.split.output_dir)).resolve(),
        figure_dir=Path(str(config.split.figure_dir)).resolve(),
        report_path=Path(str(config.split.report_path)).resolve(),
        metadata_path=Path(str(config.split.metadata_path)).resolve(),
        seed=int(config.split.seed),
        candidate_seeds=int(config.split.candidate_seeds),
        counts=counts,
        max_categorical_prevalence_deviation=float(
            config.split.tolerances.max_categorical_prevalence_deviation
        ),
        max_absolute_standardized_mean_difference=float(
            config.split.tolerances.max_absolute_standardized_mean_difference
        ),
    )


def _records(frame: pd.DataFrame) -> list[Mapping[str, Any]]:
    """Return CSV-compatible row mappings with a precise static type."""
    return cast(list[Mapping[str, Any]], frame.to_dict(orient="records"))


def generate_split(settings: SplitSettings) -> dict[str, Any]:
    """Generate and validate the provisional patient-level split."""
    frame = pd.read_csv(settings.canonical_manifest)
    frame = add_stratification_features(frame)
    if len(frame) != sum(settings.counts.values()):
        raise SplitIntegrityError(
            f"Canonical cohort has {len(frame)} rows but split counts total "
            f"{sum(settings.counts.values())}"
        )
    assert_no_duplicate_image_signatures(frame)
    assert_no_duplicate_file_hashes(frame)
    candidate = select_candidate(frame, settings)
    validate_membership(frame, candidate.membership, settings.counts)
    categorical, continuous = balance_tables(frame, candidate.membership)
    maximum_categorical = float(categorical["absolute_deviation"].max())
    maximum_smd = float(continuous["standardized_mean_difference"].abs().max())
    balance_pass = (
        maximum_categorical <= settings.max_categorical_prevalence_deviation
        and maximum_smd <= settings.max_absolute_standardized_mean_difference
    )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[SplitName, Path] = {}
    for split in _SPLIT_ORDER:
        split_frame = frame.iloc[candidate.membership[split]].copy()
        split_frame.insert(0, "split", split)
        split_frame = split_frame.sort_values("subject_id").reset_index(drop=True)
        destination = settings.output_dir / f"{split}.csv"
        atomic_write_csv(destination, _records(split_frame))
        manifest_paths[split] = destination

    manifest_hashes = {
        split: file_digest(path) for split, path in manifest_paths.items()
    }
    canonical_hash = file_digest(settings.canonical_manifest)
    metadata: dict[str, Any] = {
        "status": "pass" if balance_pass else "fail",
        "algorithm": (
            "two-stage stratified shuffle candidate search; primary stratum="
            "grade|ET presence|WT quartile"
        ),
        "seed": settings.seed,
        "candidate_seeds": settings.candidate_seeds,
        "selected_candidate_index": candidate.candidate_index,
        "objective": candidate.objective,
        "counts": dict(settings.counts),
        "canonical_manifest_sha256": canonical_hash,
        "manifest_sha256": manifest_hashes,
        "maximum_categorical_prevalence_deviation": maximum_categorical,
        "maximum_absolute_standardized_mean_difference": maximum_smd,
        "tolerances": {
            "max_categorical_prevalence_deviation": (
                settings.max_categorical_prevalence_deviation
            ),
            "max_absolute_standardized_mean_difference": (
                settings.max_absolute_standardized_mean_difference
            ),
        },
        "patient_id_overlap_count": 0,
        "duplicate_image_signature_count": 0,
        "duplicate_file_hash_count": 0,
        "frozen": False,
    }
    atomic_write_json(settings.metadata_path, metadata)
    atomic_write_csv(
        settings.output_dir / "categorical_balance.csv",
        _records(categorical),
    )
    atomic_write_csv(
        settings.output_dir / "continuous_balance.csv",
        _records(continuous),
    )

    _plot_categorical(
        categorical,
        feature="grade",
        destination=settings.figure_dir / "split_balance_grade.png",
        title="Grade balance across patient-level splits",
    )
    _plot_categorical(
        categorical,
        feature="et_present",
        destination=settings.figure_dir / "split_balance_et_presence.png",
        title="Enhancing-tumor presence across patient-level splits",
    )
    _plot_categorical(
        categorical,
        feature="wt_volume_quartile",
        destination=settings.figure_dir / "split_balance_total_tumor_quartiles.png",
        title="Whole-tumor volume quartiles across patient-level splits",
    )
    _plot_smd(
        continuous,
        destination=settings.figure_dir / "split_balance_volume_smd.png",
    )
    atomic_write_text(
        settings.report_path,
        _render_report(
            settings,
            candidate,
            categorical,
            continuous,
            manifest_hashes,
            balance_pass,
        ),
    )
    if not balance_pass:
        raise SplitIntegrityError(
            "Provisional split violates configured balance tolerances"
        )
    return metadata


def load_development_manifest(
    split_dir: Path,
    split: DevelopmentSplitName,
) -> pd.DataFrame:
    """Load train/validation data while making test access impossible."""
    if split not in {"train", "validation"}:
        raise ValueError(f"Development split must be train/validation, got {split}")
    return pd.read_csv(split_dir / f"{split}.csv")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def load_internal_test_manifest(
    split_dir: Path,
    *,
    allow_test_evaluation: bool,
    purpose: str,
    audit_log: Path = Path("artifacts/test_access_log.jsonl"),
) -> pd.DataFrame:
    """Load the internal test manifest only with explicit, logged authorization."""
    if not allow_test_evaluation:
        raise PermissionError(
            "Internal held-out test access requires --allow-test-evaluation"
        )
    if not purpose.strip():
        raise ValueError("A non-empty test-access purpose is required")
    manifest_path = split_dir / "test.csv"
    append_jsonl(
        audit_log,
        {
            "event": "internal_test_manifest_access",
            "purpose": purpose,
            "manifest_path": manifest_path.as_posix(),
            "manifest_sha256": file_digest(manifest_path),
            "git_commit": _git_commit(),
            "command": sys.argv,
        },
    )
    return pd.read_csv(manifest_path)


def load_config(path: Path, overrides: Sequence[str]) -> DictConfig:
    """Load split config and dot-list overrides."""
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return cast(DictConfig, config)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the split-generation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic patient-level provisional splits"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/split.yaml"),
    )
    parser.add_argument("overrides", nargs="*")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    arguments = build_argument_parser().parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.overrides)
        metadata = generate_split(settings_from_config(config))
        print(json.dumps(metadata, indent=2, sort_keys=True))
    except Exception as error:
        print(
            json.dumps(
                {
                    "event": "split_generation_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
