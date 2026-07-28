"""Path resolution and raw-data write-safety guards."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path


class PathSafetyError(RuntimeError):
    """Raised when an output path could modify or contaminate raw data."""


def path_from_environment(name: str) -> Path:
    """Return a resolved path from a required environment variable."""
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Required environment variable {name!r} is not set")
    return Path(value).expanduser().resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    """Return whether *path* equals or is below *parent* after resolution."""
    resolved_path = path.expanduser().resolve()
    resolved_parent = parent.expanduser().resolve()
    return resolved_path == resolved_parent or resolved_path.is_relative_to(
        resolved_parent
    )


def assert_existing_directory(path: Path, description: str) -> Path:
    """Validate and resolve an existing directory."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {resolved}")
    return resolved


def assert_output_paths_safe(
    output_paths: Iterable[Path],
    raw_roots: Iterable[Path],
) -> None:
    """Reject any generated path located at or below a raw-data root."""
    resolved_roots = [root.expanduser().resolve() for root in raw_roots]
    for output_path in output_paths:
        resolved_output = output_path.expanduser().resolve()
        for raw_root in resolved_roots:
            if is_relative_to(resolved_output, raw_root):
                raise PathSafetyError(
                    "Generated output must not be located inside a raw-data root: "
                    f"output={resolved_output}, raw_root={raw_root}"
                )
