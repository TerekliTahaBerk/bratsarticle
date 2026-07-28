"""Read-only BraTS inventory, integrity, and cross-year duplicate audit."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.orientations import aff2axcodes
from omegaconf import DictConfig, OmegaConf

from bratsarticle.data.discovery import (
    FILE_ROLES,
    SubjectDiscovery,
    discover_brats2019_subjects,
    discover_brats2020_subjects,
    resolve_brats2019_root,
    resolve_brats2020_training_root,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)


@dataclass(frozen=True)
class AuditSettings:
    """Resolved configuration for a data-audit run."""

    brats2020_root: Path
    brats2019_root: Path
    output_root: Path
    workers: int
    hash_algorithm: str
    limit_subjects: int | None
    compare_content_on_hash_mismatch: bool
    fail_on_invalid_label_set: bool
    expected_segmentation_labels: frozenset[int]


@dataclass(frozen=True)
class SubjectAudit:
    """File-level and subject-level records produced by an audit."""

    subject_row: Mapping[str, Any]
    file_rows: tuple[Mapping[str, Any], ...]


def _json_compact(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _relative_path(path: Path, dataset_root: Path) -> str:
    return path.resolve().relative_to(dataset_root.resolve()).as_posix()


def _finite_statistics(data: np.ndarray) -> tuple[int, int, float, float, float]:
    if np.issubdtype(data.dtype, np.floating):
        nan_count = int(np.isnan(data).sum())
        inf_count = int(np.isinf(data).sum())
        finite = data[np.isfinite(data)]
    else:
        nan_count = 0
        inf_count = 0
        finite = data
    if finite.size == 0:
        return nan_count, inf_count, math.nan, math.nan, math.nan
    return (
        nan_count,
        inf_count,
        float(np.min(finite)),
        float(np.max(finite)),
        float(np.mean(finite, dtype=np.float64)),
    )


def inspect_nifti(
    path: Path,
    dataset: str,
    subject_id: str,
    role: str,
    dataset_root: Path,
    hash_algorithm: str,
    expected_segmentation_labels: frozenset[int],
) -> dict[str, Any]:
    """Read and summarize one NIfTI file without writing to the source tree."""
    image = cast(nib.Nifti1Image, nib.load(str(path), mmap="r"))
    data = np.asanyarray(image.dataobj)
    nan_count, inf_count, minimum, maximum, mean = _finite_statistics(data)
    spacing = tuple(
        float(value)
        for value in image.header.get_zooms()[: data.ndim]  # type: ignore[no-untyped-call]
    )
    affine = np.asarray(image.affine, dtype=np.float64)

    row: dict[str, Any] = {
        "dataset": dataset,
        "subject_id": subject_id,
        "role": role,
        "relative_path": _relative_path(path, dataset_root),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size,
        f"{hash_algorithm}": file_digest(path, algorithm=hash_algorithm),
        "shape": _json_compact(list(data.shape)),
        "voxel_spacing": _json_compact([round(value, 8) for value in spacing]),
        "affine": _json_compact(np.round(affine, decimals=8).tolist()),
        "orientation": "".join(
            value if value is not None else "?"
            for value in aff2axcodes(affine)  # type: ignore[no-untyped-call]
        ),
        "dtype": str(image.get_data_dtype()),  # type: ignore[no-untyped-call]
        "nan_count": nan_count,
        "inf_count": inf_count,
        "all_zero": bool(np.count_nonzero(data) == 0),
        "nonzero_voxel_count": int(np.count_nonzero(data)),
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "label_set": "",
        "unexpected_labels": "",
        "valid_label_set": "",
        "wt_voxel_count": "",
        "tc_voxel_count": "",
        "et_voxel_count": "",
        "voxel_volume_mm3": float(np.prod(spacing[:3])),
        "integrity_status": "ok",
        "error": "",
    }

    if role == "seg":
        labels = frozenset(int(value) for value in np.unique(data))
        unexpected = sorted(labels - expected_segmentation_labels)
        row.update(
            {
                "label_set": _json_compact(sorted(labels)),
                "unexpected_labels": _json_compact(unexpected),
                "valid_label_set": len(unexpected) == 0,
                "wt_voxel_count": int(np.isin(data, (1, 2, 4)).sum()),
                "tc_voxel_count": int(np.isin(data, (1, 4)).sum()),
                "et_voxel_count": int(np.equal(data, 4).sum()),
            }
        )
    return row


def _error_file_row(
    discovery: SubjectDiscovery,
    role: str,
    path: Path,
    dataset_root: Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "dataset": discovery.dataset,
        "subject_id": discovery.subject_id,
        "role": role,
        "relative_path": _relative_path(path, dataset_root),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.exists() else "",
        "integrity_status": "error",
        "error": f"{type(error).__name__}: {error}",
    }


def audit_subject(
    discovery: SubjectDiscovery,
    dataset_root: Path,
    hash_algorithm: str,
    expected_segmentation_labels: frozenset[int],
) -> SubjectAudit:
    """Audit one subject and retain errors as explicit integrity records."""
    file_rows: list[dict[str, Any]] = []
    by_role: dict[str, dict[str, Any]] = {}
    for role in FILE_ROLES:
        path = discovery.files.get(role)
        if path is None:
            continue
        try:
            row = inspect_nifti(
                path=path,
                dataset=discovery.dataset,
                subject_id=discovery.subject_id,
                role=role,
                dataset_root=dataset_root,
                hash_algorithm=hash_algorithm,
                expected_segmentation_labels=expected_segmentation_labels,
            )
        except Exception as error:  # retained in the audit rather than hidden
            row = _error_file_row(discovery, role, path, dataset_root, error)
        file_rows.append(row)
        by_role[role] = row

    has_file_errors = any(row.get("integrity_status") != "ok" for row in file_rows)
    segmentation_row = by_role.get("seg", {})
    valid_label_set = segmentation_row.get("valid_label_set") is True
    eligible = discovery.complete and not has_file_errors and valid_label_set

    subject_row: dict[str, Any] = {
        "dataset": discovery.dataset,
        "subject_id": discovery.subject_id,
        "grade": discovery.grade or "",
        "complete": discovery.complete,
        "eligible": eligible,
        "discovery_warnings": "|".join(discovery.warnings),
        "file_error_count": sum(
            row.get("integrity_status") != "ok" for row in file_rows
        ),
        "seg_label_set": segmentation_row.get("label_set", ""),
        "seg_valid_label_set": segmentation_row.get("valid_label_set", ""),
        "wt_voxel_count": segmentation_row.get("wt_voxel_count", ""),
        "tc_voxel_count": segmentation_row.get("tc_voxel_count", ""),
        "et_voxel_count": segmentation_row.get("et_voxel_count", ""),
        "voxel_volume_mm3": segmentation_row.get("voxel_volume_mm3", ""),
    }
    for role in FILE_ROLES:
        role_row = by_role.get(role, {})
        subject_row[f"{role}_relative_path"] = role_row.get("relative_path", "")
        subject_row[f"{role}_file_size_bytes"] = role_row.get("file_size_bytes", "")
        subject_row[f"{role}_{hash_algorithm}"] = role_row.get(hash_algorithm, "")
        subject_row[f"{role}_shape"] = role_row.get("shape", "")
        subject_row[f"{role}_voxel_spacing"] = role_row.get("voxel_spacing", "")
        subject_row[f"{role}_orientation"] = role_row.get("orientation", "")
        subject_row[f"{role}_dtype"] = role_row.get("dtype", "")
        subject_row[f"{role}_nan_count"] = role_row.get("nan_count", "")
        subject_row[f"{role}_inf_count"] = role_row.get("inf_count", "")
        subject_row[f"{role}_all_zero"] = role_row.get("all_zero", "")

    return SubjectAudit(subject_row=subject_row, file_rows=tuple(file_rows))


def audit_subjects(
    discoveries: Sequence[SubjectDiscovery],
    dataset_root: Path,
    settings: AuditSettings,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Audit subjects concurrently while returning deterministic row ordering."""
    total = len(discoveries)
    results: dict[str, SubjectAudit] = {}
    with ThreadPoolExecutor(max_workers=settings.workers) as executor:
        futures = {
            executor.submit(
                audit_subject,
                discovery,
                dataset_root,
                settings.hash_algorithm,
                settings.expected_segmentation_labels,
            ): discovery
            for discovery in discoveries
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            discovery = futures[future]
            results[discovery.subject_id] = future.result()
            if completed == 1 or completed % 25 == 0 or completed == total:
                print(
                    _json_compact(
                        {
                            "event": "audit_progress",
                            "dataset": discovery.dataset,
                            "completed": completed,
                            "total": total,
                        }
                    ),
                    flush=True,
                )

    subject_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    for subject_id in sorted(results):
        result = results[subject_id]
        subject_rows.append(dict(result.subject_row))
        file_rows.extend(dict(row) for row in result.file_rows)
    file_rows.sort(key=lambda row: (row["dataset"], row["subject_id"], row["role"]))
    return subject_rows, file_rows


def _load_brats2020_mapping(root: Path) -> pd.DataFrame:
    mapping_path = root / "name_mapping.csv"
    frame = pd.read_csv(mapping_path, dtype=str).fillna("NA")
    required = {
        "Grade",
        "BraTS_2019_subject_ID",
        "BraTS_2020_subject_ID",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"BraTS 2020 name_mapping.csv is missing columns: {sorted(missing)}"
        )
    if frame["BraTS_2020_subject_ID"].duplicated().any():
        duplicates = frame.loc[
            frame["BraTS_2020_subject_ID"].duplicated(keep=False),
            "BraTS_2020_subject_ID",
        ].tolist()
        raise ValueError(f"Duplicate BraTS 2020 mapping IDs: {duplicates}")
    return frame


def _apply_brats2020_grades(
    rows: list[dict[str, Any]],
    mapping: pd.DataFrame,
) -> None:
    grade_by_id = dict(
        zip(
            mapping["BraTS_2020_subject_ID"],
            mapping["Grade"],
            strict=True,
        )
    )
    for row in rows:
        row["grade"] = grade_by_id.get(row["subject_id"], "")


def compare_nifti_content(
    left: Path,
    right: Path,
    include_value_pairs: bool = False,
) -> tuple[bool, float | None, int | None, str]:
    """Compare voxel arrays and summarize any content revision."""
    left_image = cast(nib.Nifti1Image, nib.load(str(left), mmap="r"))
    right_image = cast(nib.Nifti1Image, nib.load(str(right), mmap="r"))
    left_data = np.asanyarray(left_image.dataobj)
    right_data = np.asanyarray(right_image.dataobj)
    if left_data.shape != right_data.shape:
        return False, None, None, ""
    equal = bool(np.array_equal(left_data, right_data, equal_nan=True))
    if equal:
        return True, 0.0, 0, ""
    equal_mask = np.equal(left_data, right_data)
    if np.issubdtype(left_data.dtype, np.floating) or np.issubdtype(
        right_data.dtype, np.floating
    ):
        equal_mask = np.logical_or(
            equal_mask,
            np.logical_and(np.isnan(left_data), np.isnan(right_data)),
        )
    differing_voxel_count = int(np.count_nonzero(~equal_mask))
    difference = np.abs(
        left_data.astype(np.float64, copy=False)
        - right_data.astype(np.float64, copy=False)
    )
    finite = difference[np.isfinite(difference)]
    value_pairs = ""
    if include_value_pairs:
        left_values = left_data[~equal_mask].astype(np.int64, copy=False)
        right_values = right_data[~equal_mask].astype(np.int64, copy=False)
        pairs, counts = np.unique(
            np.column_stack((left_values, right_values)),
            axis=0,
            return_counts=True,
        )
        value_pairs = _json_compact(
            {
                f"{int(pair[0])}->{int(pair[1])}": int(count)
                for pair, count in zip(pairs, counts, strict=True)
            }
        )
    return (
        False,
        float(np.max(finite)) if finite.size else math.nan,
        differing_voxel_count,
        value_pairs,
    )


def build_duplicate_mapping(
    mapping: pd.DataFrame,
    brats2020_rows: Sequence[Mapping[str, Any]],
    brats2019_rows: Sequence[Mapping[str, Any]],
    brats2020_discoveries: Sequence[SubjectDiscovery],
    brats2019_discoveries: Sequence[SubjectDiscovery],
    settings: AuditSettings,
) -> list[dict[str, Any]]:
    """Compare mapped BraTS 2019/2020 subjects by identity and file content."""
    row2020 = {str(row["subject_id"]): row for row in brats2020_rows}
    row2019 = {str(row["subject_id"]): row for row in brats2019_rows}
    discovery2020 = {item.subject_id: item for item in brats2020_discoveries}
    discovery2019 = {item.subject_id: item for item in brats2019_discoveries}
    records: list[dict[str, Any]] = []

    for mapping_row in mapping.to_dict(orient="records"):
        subject2020 = str(mapping_row["BraTS_2020_subject_ID"])
        raw_subject2019 = str(mapping_row["BraTS_2019_subject_ID"])
        subject2019 = "" if raw_subject2019 in {"", "NA", "nan"} else raw_subject2019
        status = "mapped_overlap" if subject2019 else "new_in_brats2020"
        record: dict[str, Any] = {
            "grade": mapping_row["Grade"],
            "brats2020_subject_id": subject2020,
            "brats2019_subject_id": subject2019,
            "mapping_status": status,
            "brats2020_inventory_present": subject2020 in row2020,
            "brats2019_inventory_present": (
                subject2019 in row2019 if subject2019 else ""
            ),
        }
        role_matches: list[bool] = []
        role_equivalences: list[bool] = []
        image_role_equivalences: list[bool] = []
        for role in FILE_ROLES:
            match_value: bool | str = ""
            content_equal: bool | str = ""
            maximum_difference: float | str | None = ""
            differing_voxel_count: int | str | None = ""
            value_pair_counts = ""
            content_or_sha_equivalent: bool | str = ""
            if subject2019 and subject2020 in row2020 and subject2019 in row2019:
                digest2020 = row2020[subject2020].get(
                    f"{role}_{settings.hash_algorithm}", ""
                )
                digest2019 = row2019[subject2019].get(
                    f"{role}_{settings.hash_algorithm}", ""
                )
                match_value = bool(digest2020 and digest2020 == digest2019)
                role_matches.append(match_value)
                if (
                    not match_value
                    and settings.compare_content_on_hash_mismatch
                    and role in discovery2020[subject2020].files
                    and role in discovery2019[subject2019].files
                ):
                    (
                        content_equal,
                        maximum_difference,
                        differing_voxel_count,
                        value_pair_counts,
                    ) = compare_nifti_content(
                        discovery2020[subject2020].files[role],
                        discovery2019[subject2019].files[role],
                        include_value_pairs=role == "seg",
                    )
                content_or_sha_equivalent = bool(match_value or content_equal is True)
                role_equivalences.append(content_or_sha_equivalent)
                if role != "seg":
                    image_role_equivalences.append(content_or_sha_equivalent)
            record[f"{role}_sha256_match"] = match_value
            record[f"{role}_voxel_content_equal"] = content_equal
            record[f"{role}_max_abs_difference"] = maximum_difference
            record[f"{role}_differing_voxel_count"] = differing_voxel_count
            record[f"{role}_value_pair_counts"] = value_pair_counts
            record[f"{role}_content_or_sha_equivalent"] = content_or_sha_equivalent
        record["all_role_sha256_match"] = (
            all(role_matches) if len(role_matches) == len(FILE_ROLES) else ""
        )
        record["all_role_content_or_sha_equivalent"] = (
            all(role_equivalences) if len(role_equivalences) == len(FILE_ROLES) else ""
        )
        record["all_image_modalities_content_or_sha_equivalent"] = (
            all(image_role_equivalences)
            if len(image_role_equivalences) == len(FILE_ROLES) - 1
            else ""
        )
        records.append(record)
    return records


def _count_truthy(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(row.get(key) is True for row in rows)


def _grade_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = (
        pd.Series([str(row.get("grade", "")) for row in rows], dtype="string")
        .value_counts()
        .sort_index()
    )
    return {str(grade): int(count) for grade, count in counts.items()}


def build_summary(
    brats2020_rows: Sequence[Mapping[str, Any]],
    brats2019_rows: Sequence[Mapping[str, Any]],
    file_rows: Sequence[Mapping[str, Any]],
    duplicate_rows: Sequence[Mapping[str, Any]],
    full_run: bool,
) -> dict[str, Any]:
    """Create a machine-readable audit summary and explicit hypothesis checks."""
    overlap_rows = [
        row for row in duplicate_rows if row["mapping_status"] == "mapped_overlap"
    ]
    new_rows = [
        row for row in duplicate_rows if row["mapping_status"] == "new_in_brats2020"
    ]
    segmentation_revision_rows = [
        row for row in overlap_rows if row.get("seg_content_or_sha_equivalent") is False
    ]
    naming_exceptions = [
        row
        for row in (*brats2020_rows, *brats2019_rows)
        if row.get("discovery_warnings")
    ]
    invalid_label_subjects = [
        str(row["subject_id"])
        for row in (*brats2020_rows, *brats2019_rows)
        if row.get("seg_valid_label_set") is not True
    ]
    file_errors = [row for row in file_rows if row.get("integrity_status") != "ok"]

    checks = {
        "brats2020_subject_count_is_369": len(brats2020_rows) == 369,
        "brats2019_subject_count_is_335": len(brats2019_rows) == 335,
        "mapped_overlap_count_is_335": len(overlap_rows) == 335,
        "new_brats2020_subject_count_is_34": len(new_rows) == 34,
        "all_brats2020_subjects_complete": _count_truthy(brats2020_rows, "complete")
        == len(brats2020_rows),
        "all_brats2019_subjects_complete": _count_truthy(brats2019_rows, "complete")
        == len(brats2019_rows),
        "no_file_integrity_errors": len(file_errors) == 0,
        "all_segmentation_label_sets_valid": len(invalid_label_subjects) == 0,
        "all_mapped_image_modalities_content_equivalent": sum(
            row.get("all_image_modalities_content_or_sha_equivalent") is True
            for row in overlap_rows
        )
        == len(overlap_rows),
    }
    return {
        "run_scope": "full" if full_run else "limited_smoke_test",
        "brats2020": {
            "subject_count": len(brats2020_rows),
            "complete_subject_count": _count_truthy(brats2020_rows, "complete"),
            "eligible_subject_count": _count_truthy(brats2020_rows, "eligible"),
            "grade_counts": _grade_counts(brats2020_rows),
        },
        "brats2019": {
            "subject_count": len(brats2019_rows),
            "complete_subject_count": _count_truthy(brats2019_rows, "complete"),
            "eligible_subject_count": _count_truthy(brats2019_rows, "eligible"),
            "grade_counts": _grade_counts(brats2019_rows),
        },
        "mapping": {
            "row_count": len(duplicate_rows),
            "mapped_overlap_count": len(overlap_rows),
            "new_in_brats2020_count": len(new_rows),
            "all_role_sha256_match_count": sum(
                row.get("all_role_sha256_match") is True for row in overlap_rows
            ),
            "all_role_content_or_sha_equivalent_count": sum(
                row.get("all_role_content_or_sha_equivalent") is True
                for row in overlap_rows
            ),
            "all_image_modalities_content_or_sha_equivalent_count": sum(
                row.get("all_image_modalities_content_or_sha_equivalent") is True
                for row in overlap_rows
            ),
            "segmentation_revision_count": len(segmentation_revision_rows),
            "segmentation_revision_subjects": [
                {
                    "brats2020_subject_id": row["brats2020_subject_id"],
                    "brats2019_subject_id": row["brats2019_subject_id"],
                    "differing_voxel_count": row["seg_differing_voxel_count"],
                    "value_pair_counts": row["seg_value_pair_counts"],
                }
                for row in segmentation_revision_rows
            ],
        },
        "integrity": {
            "audited_file_count": len(file_rows),
            "file_error_count": len(file_errors),
            "invalid_label_subjects": invalid_label_subjects,
            "naming_exception_count": len(naming_exceptions),
            "naming_exceptions": [
                {
                    "dataset": row["dataset"],
                    "subject_id": row["subject_id"],
                    "warnings": row["discovery_warnings"],
                }
                for row in naming_exceptions
            ],
        },
        "hypothesis_checks": checks,
        "gate1_integrity_pass": full_run and all(checks.values()),
    }


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    """Render the audit summary without manually entered scientific values."""
    checks = summary["hypothesis_checks"]
    lines = [
        "# BraTS Data Audit Summary",
        "",
        f"**Run scope:** `{summary['run_scope']}`",
        "",
        "## Cohort inventory",
        "",
        "| Dataset | Subjects | Complete | Eligible | Grade counts |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in ("brats2020", "brats2019"):
        block = summary[dataset]
        lines.append(
            f"| {dataset} | {block['subject_count']} | "
            f"{block['complete_subject_count']} | "
            f"{block['eligible_subject_count']} | "
            f"`{_json_compact(block['grade_counts'])}` |"
        )
    lines.extend(
        [
            "",
            "## Cross-year mapping",
            "",
            f"- Mapping rows: {summary['mapping']['row_count']}",
            f"- Mapped overlaps: {summary['mapping']['mapped_overlap_count']}",
            f"- New BraTS 2020 subjects: "
            f"{summary['mapping']['new_in_brats2020_count']}",
            f"- Mapped pairs with all five exact file hashes equal: "
            f"{summary['mapping']['all_role_sha256_match_count']}",
            f"- Mapped pairs with all five voxel contents equivalent: "
            f"{summary['mapping']['all_role_content_or_sha_equivalent_count']}",
            f"- Mapped pairs with all four MRI modalities equivalent: "
            f"{summary['mapping']['all_image_modalities_content_or_sha_equivalent_count']}",
            f"- Mapped pairs with a segmentation annotation revision: "
            f"{summary['mapping']['segmentation_revision_count']}",
            "",
            "## File integrity",
            "",
            f"- Audited NIfTI files: {summary['integrity']['audited_file_count']}",
            f"- File errors: {summary['integrity']['file_error_count']}",
            f"- Naming exceptions: {summary['integrity']['naming_exception_count']}",
            f"- Subjects with invalid label sets: "
            f"{len(summary['integrity']['invalid_label_subjects'])}",
            "",
            "## Gate checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"**Gate 1 integrity status:** "
            f"{'PASS' if summary['gate1_integrity_pass'] else 'NOT PASSED'}",
        ]
    )
    return "\n".join(lines)


def _output_paths(output_root: Path) -> dict[str, Path]:
    return {
        "brats2020_inventory": output_root / "manifests/raw/brats2020_inventory.csv",
        "brats2019_inventory": output_root / "manifests/raw/brats2019_inventory.csv",
        "canonical_manifest": output_root
        / "manifests/canonical/brats2020_canonical_manifest.csv",
        "duplicate_mapping": output_root / "manifests/audit/duplicate_mapping.csv",
        "file_integrity": output_root / "manifests/audit/file_integrity_report.csv",
        "summary_markdown": output_root / "reports/data_audit_summary.md",
        "summary_json": output_root / "reports/data_audit_summary.json",
    }


def settings_from_config(config: DictConfig) -> AuditSettings:
    """Resolve and validate the OmegaConf audit configuration."""
    brats2020_root = resolve_brats2020_training_root(
        Path(str(config.data.brats2020_root))
    )
    brats2019_root = resolve_brats2019_root(Path(str(config.data.brats2019_root)))
    limit = config.audit.limit_subjects
    return AuditSettings(
        brats2020_root=brats2020_root,
        brats2019_root=brats2019_root,
        output_root=Path(str(config.audit.output_root)).expanduser().resolve(),
        workers=max(1, int(config.audit.workers)),
        hash_algorithm=str(config.audit.hash_algorithm),
        limit_subjects=None if limit is None else int(limit),
        compare_content_on_hash_mismatch=bool(
            config.audit.compare_content_on_hash_mismatch
        ),
        fail_on_invalid_label_set=bool(config.audit.fail_on_invalid_label_set),
        expected_segmentation_labels=frozenset(
            int(value) for value in config.audit.expected_segmentation_labels
        ),
    )


def run_audit(settings: AuditSettings) -> dict[str, Any]:
    """Execute the complete read-only audit and write controlled artifacts."""
    output_paths = _output_paths(settings.output_root)
    assert_output_paths_safe(
        output_paths.values(),
        (settings.brats2020_root, settings.brats2019_root),
    )

    discoveries2020 = discover_brats2020_subjects(settings.brats2020_root)
    discoveries2019 = discover_brats2019_subjects(settings.brats2019_root)
    full_run = settings.limit_subjects is None
    if settings.limit_subjects is not None:
        discoveries2020 = discoveries2020[: settings.limit_subjects]
        discoveries2019 = discoveries2019[: settings.limit_subjects]

    print(
        _json_compact(
            {
                "event": "discovery_complete",
                "brats2020_subjects": len(discoveries2020),
                "brats2019_subjects": len(discoveries2019),
                "scope": "full" if full_run else "limited",
            }
        ),
        flush=True,
    )

    rows2020, files2020 = audit_subjects(
        discoveries2020, settings.brats2020_root, settings
    )
    rows2019, files2019 = audit_subjects(
        discoveries2019, settings.brats2019_root, settings
    )
    mapping = _load_brats2020_mapping(settings.brats2020_root)
    _apply_brats2020_grades(rows2020, mapping)
    duplicate_rows = build_duplicate_mapping(
        mapping=mapping,
        brats2020_rows=rows2020,
        brats2019_rows=rows2019,
        brats2020_discoveries=discoveries2020,
        brats2019_discoveries=discoveries2019,
        settings=settings,
    )
    file_rows = sorted(
        [*files2020, *files2019],
        key=lambda row: (row["dataset"], row["subject_id"], row["role"]),
    )
    summary = build_summary(
        brats2020_rows=rows2020,
        brats2019_rows=rows2019,
        file_rows=file_rows,
        duplicate_rows=duplicate_rows,
        full_run=full_run,
    )

    canonical_rows = sorted(rows2020, key=lambda row: str(row["subject_id"]))
    atomic_write_csv(output_paths["brats2020_inventory"], rows2020)
    atomic_write_csv(output_paths["brats2019_inventory"], rows2019)
    atomic_write_csv(output_paths["canonical_manifest"], canonical_rows)
    atomic_write_csv(output_paths["duplicate_mapping"], duplicate_rows)
    atomic_write_csv(output_paths["file_integrity"], file_rows)
    atomic_write_json(output_paths["summary_json"], summary)
    atomic_write_text(
        output_paths["summary_markdown"],
        render_summary_markdown(summary),
    )
    print(
        _json_compact(
            {
                "event": "audit_complete",
                "gate1_integrity_pass": summary["gate1_integrity_pass"],
                "summary": str(output_paths["summary_json"]),
            }
        ),
        flush=True,
    )

    if (
        settings.fail_on_invalid_label_set
        and full_run
        and summary["integrity"]["invalid_label_subjects"]
    ):
        raise RuntimeError(
            "Invalid segmentation label sets were detected; inspect the audit "
            "artifacts before continuing"
        )
    return summary


def load_config(path: Path, overrides: Sequence[str]) -> DictConfig:
    """Load configuration and apply command-line dot-list overrides."""
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(config)
    return cast(DictConfig, config)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the audit CLI parser."""
    parser = argparse.ArgumentParser(
        description="Read-only BraTS data integrity and duplicate audit"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/audit.yaml"),
        help="OmegaConf YAML configuration path",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dot-list overrides, e.g. audit.workers=4",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config, arguments.overrides)
        settings = settings_from_config(config)
        run_audit(settings)
    except Exception as error:
        print(
            _json_compact(
                {
                    "event": "audit_failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
