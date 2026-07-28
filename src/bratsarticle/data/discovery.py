"""Dataset-root resolution and controlled subject-file discovery."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Mapping

from bratsarticle.utils.paths import assert_existing_directory

FILE_ROLES: Final[tuple[str, ...]] = ("t1", "t1ce", "t2", "flair", "seg")
_ROLE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"_(t1ce|flair|t1|t2|seg)\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)
_NIFTI_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\.nii(?:\.gz)?$",
    flags=re.IGNORECASE,
)
_SEGMENTATION_FALLBACK: Final[re.Pattern[str]] = re.compile(
    r"(?:^|[_\-.])seg(?:m|mentation)?(?:[_\-.]|$)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SubjectDiscovery:
    """Files and discovery warnings for one dataset subject."""

    dataset: str
    subject_id: str
    grade: str | None
    subject_dir: Path
    files: Mapping[str, Path]
    warnings: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether all required MRI/segmentation roles were found."""
        return all(role in self.files for role in FILE_ROLES)


class DiscoveryError(RuntimeError):
    """Raised for ambiguous or structurally invalid dataset discovery."""


def _contains_training_subjects(path: Path) -> bool:
    return any(
        child.is_dir() and child.name.startswith("BraTS20_Training_")
        for child in path.iterdir()
    )


def resolve_brats2020_training_root(candidate: Path) -> Path:
    """Resolve the directory containing BraTS 2020 training subject folders."""
    root = assert_existing_directory(candidate, "BraTS 2020 root")
    if (root / "name_mapping.csv").is_file() and _contains_training_subjects(root):
        return root

    matches = sorted(
        mapping.parent
        for mapping in root.rglob("name_mapping.csv")
        if _contains_training_subjects(mapping.parent)
    )
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) != 1:
        raise DiscoveryError(
            "Expected exactly one BraTS 2020 training root below "
            f"{root}, found {len(unique_matches)}: {unique_matches}"
        )
    return unique_matches[0]


def resolve_brats2019_root(candidate: Path) -> Path:
    """Resolve the directory containing the BraTS 2019 HGG/LGG folders."""
    root = assert_existing_directory(candidate, "BraTS 2019 root")
    if (root / "HGG").is_dir() and (root / "LGG").is_dir():
        return root

    matches = sorted(
        path
        for path in root.rglob("*")
        if path.is_dir() and (path / "HGG").is_dir() and (path / "LGG").is_dir()
    )
    unique_matches = list(dict.fromkeys(matches))
    if len(unique_matches) != 1:
        raise DiscoveryError(
            "Expected exactly one BraTS 2019 root below "
            f"{root}, found {len(unique_matches)}: {unique_matches}"
        )
    return unique_matches[0]


def _nifti_files(subject_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in subject_dir.iterdir()
        if path.is_file() and _NIFTI_PATTERN.search(path.name)
    )


def _role_from_name(path: Path) -> str | None:
    match = _ROLE_PATTERN.search(path.name)
    return match.group(1).lower() if match else None


def discover_subject(
    dataset: str,
    subject_dir: Path,
    grade: str | None,
) -> SubjectDiscovery:
    """Discover required files with a generic, audited segmentation fallback."""
    role_candidates: dict[str, list[Path]] = {role: [] for role in FILE_ROLES}
    unmatched: list[Path] = []
    warnings: list[str] = []

    for path in _nifti_files(subject_dir):
        role = _role_from_name(path)
        if role is None:
            unmatched.append(path)
        else:
            role_candidates[role].append(path)

    if not role_candidates["seg"]:
        fallback_candidates = [
            path
            for path in unmatched
            if _SEGMENTATION_FALLBACK.search(
                _NIFTI_PATTERN.sub("", path.name).lower()
            )
        ]
        if len(fallback_candidates) == 1:
            role_candidates["seg"] = fallback_candidates
            warnings.append(
                "segmentation_filename_fallback:"
                f"{fallback_candidates[0].name}"
            )
        elif len(fallback_candidates) > 1:
            raise DiscoveryError(
                f"Ambiguous segmentation fallback for {subject_dir.name}: "
                f"{[path.name for path in fallback_candidates]}"
            )

    selected: dict[str, Path] = {}
    for role, candidates in role_candidates.items():
        if len(candidates) > 1:
            raise DiscoveryError(
                f"Multiple files for role {role!r} in {subject_dir}: "
                f"{[path.name for path in candidates]}"
            )
        if candidates:
            selected[role] = candidates[0]
        else:
            warnings.append(f"missing_role:{role}")

    return SubjectDiscovery(
        dataset=dataset,
        subject_id=subject_dir.name,
        grade=grade,
        subject_dir=subject_dir,
        files=selected,
        warnings=tuple(warnings),
    )


def discover_brats2020_subjects(root: Path) -> list[SubjectDiscovery]:
    """Discover all BraTS 2020 training subjects in deterministic order."""
    subject_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.startswith("BraTS20_Training_")
    )
    return [
        discover_subject("brats2020", subject_dir, grade=None)
        for subject_dir in subject_dirs
    ]


def discover_brats2019_subjects(root: Path) -> list[SubjectDiscovery]:
    """Discover BraTS 2019 HGG and LGG subjects in deterministic order."""
    discoveries: list[SubjectDiscovery] = []
    for grade in ("HGG", "LGG"):
        grade_root = root / grade
        subject_dirs: Iterable[Path] = sorted(
            path for path in grade_root.iterdir() if path.is_dir()
        )
        discoveries.extend(
            discover_subject("brats2019", subject_dir, grade=grade)
            for subject_dir in subject_dirs
        )
    return sorted(discoveries, key=lambda item: item.subject_id)
