"""Freeze the realized q1q2 v2 Python environment without editing legacy locks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
from pathlib import Path
from typing import Any

import torch
from packaging.utils import canonicalize_name

from bratsarticle.utils.serialization import atomic_write_json, atomic_write_text

EXCLUDED_PROJECTS = {"bratsarticle"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_requirements() -> list[str]:
    resolved: dict[str, tuple[str, str]] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata["Name"]
        if not raw_name:
            raise RuntimeError("Installed distribution has no Name metadata")
        canonical_name = canonicalize_name(raw_name)
        if canonical_name in EXCLUDED_PROJECTS:
            continue
        value = (raw_name, distribution.version)
        previous = resolved.get(canonical_name)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"Conflicting installed versions for {canonical_name}: "
                f"{previous!r} and {value!r}"
            )
        resolved[canonical_name] = value
    return [
        f"{name}=={version}"
        for _, (name, version) in sorted(resolved.items())
    ]


def _metadata(
    lock_path: Path,
    pyproject_path: Path,
    generator_path: Path,
) -> dict[str, Any]:
    repository_root = generator_path.resolve().parents[1]

    def repository_label(path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(repository_root))
        except ValueError:
            return str(path)

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    return {
        "schema_version": 1,
        "scope": "realized_local_q1q2_v2_environment",
        "status": "immutable_exact_version_snapshot",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "mps_built_in_lock_process": mps_built,
            "mps_available_in_lock_process": mps_available,
            "availability_note": (
                "Availability may be false inside a sandbox; the reportable "
                "outside-sandbox preflight is reports/q1q2_v2/"
                "hardware_preflight.json."
            ),
        },
        "artifacts": {
            repository_label(lock_path): _sha256(lock_path),
            repository_label(pyproject_path): _sha256(pyproject_path),
            repository_label(generator_path): _sha256(generator_path),
        },
        "installation": [
            "python -m pip install -r environment/q1q2_v2-requirements-lock.txt",
            "python -m pip install --no-deps .",
        ],
        "boundary": (
            "This lock captures the realized Apple/Python environment. "
            "A cluster-specific CUDA image and lock must be frozen after the "
            "blocked hardware allocation is supplied and before training."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lock-output",
        type=Path,
        default=Path("environment/q1q2_v2-requirements-lock.txt"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("environment/q1q2_v2-environment.json"),
    )
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    requirements = _installed_requirements()
    header = [
        "# q1q2 v2 exact-version environment snapshot.",
        "# Generated from the realized Python 3.11 environment.",
        "# The local project is installed separately from the checked-out source.",
        "# Do not substitute this Apple lock for the pending frozen CUDA lock.",
        "",
    ]
    atomic_write_text(
        arguments.lock_output,
        "\n".join([*header, *requirements, ""]),
    )
    generator_path = Path(__file__)
    atomic_write_json(
        arguments.metadata_output,
        _metadata(
            arguments.lock_output,
            arguments.pyproject,
            generator_path,
        ),
    )


if __name__ == "__main__":
    main()
