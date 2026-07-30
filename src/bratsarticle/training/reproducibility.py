"""Deterministic seeding and environment metadata."""

from __future__ import annotations

import importlib.metadata
import platform
import random
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from bratsarticle.utils.hashing import file_digest


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    deterministic_warn_only: bool = False,
) -> None:
    """Seed Python, NumPy, and PyTorch and configure deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=deterministic_warn_only,
    )


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from PyTorch's deterministic worker seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "torch", "monai", "nibabel", "scipy")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def collect_run_metadata(
    *,
    config_path: Path,
    split_hashes: Mapping[str, str],
    seed: int,
    device: torch.device,
    mixed_precision: bool,
    run_kind: str,
) -> dict[str, Any]:
    """Collect required provenance for a reportable or diagnostic run."""
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_kind": run_kind,
        "config_path": config_path.as_posix(),
        "config_sha256": file_digest(config_path),
        "split_sha256": dict(split_hashes),
        "git_commit": _git_commit(),
        "seed": seed,
        "device": str(device),
        "mixed_precision_requested": mixed_precision,
        "mixed_precision_effective": bool(
            mixed_precision and device.type in {"cpu", "cuda"}
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "packages": _package_versions(),
        "status": "initialized",
    }
