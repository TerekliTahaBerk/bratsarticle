"""Machine-readable, append-oriented experiment artifact registry."""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import psutil
import torch
from omegaconf import DictConfig, OmegaConf

from bratsarticle.utils.hashing import file_digest, text_digest
from bratsarticle.utils.paths import assert_output_paths_safe
from bratsarticle.utils.serialization import (
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
)

RunStatus = Literal["running", "completed", "failed", "invalid"]
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _git_state() -> tuple[str, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", None
    return commit, bool(status.strip())


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _hardware() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_bytes": psutil.virtual_memory().total,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "package_versions": {
            "monai": _version("monai"),
            "numpy": _version("numpy"),
            "torch": _version("torch"),
        },
    }


@dataclass(frozen=True)
class RunDescriptor:
    """Immutable scientific identity supplied when opening a run."""

    run_id: str
    seed: int
    model: str
    loss: str
    optimizer: str
    scheduler: str
    parameter_count: int
    input_specification: tuple[int, ...]
    data_manifest_path: Path
    split_hashes: Mapping[str, str]
    test_access_allowed: bool = False
    test_accessed: bool = False
    test_access_audit_log: str = "artifacts/test_access_log.jsonl"

    def __post_init__(self) -> None:
        """Reject ambiguous identities or invalid scientific metadata."""
        if not _RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ValueError(
                "Unsafe run_id; use letters, numbers, dot, dash, underscore"
            )
        if self.seed < 0:
            raise ValueError("Seed cannot be negative")
        if not all((self.model, self.loss, self.optimizer, self.scheduler)):
            raise ValueError("Model, loss, optimizer, and scheduler are required")
        if self.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        if not self.input_specification or any(
            value < 1 for value in self.input_specification
        ):
            raise ValueError("A positive FLOPs/MACs input specification is required")
        if self.test_accessed and not self.test_access_allowed:
            raise ValueError("Test access cannot occur without authorization")


class ResourceTracker:
    """Measure elapsed accelerator time and peak CUDA memory for one run."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.started_at = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

    def snapshot(self) -> dict[str, Any]:
        """Return the current resource measurement."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_seconds = time.perf_counter() - self.started_at
        cuda_active = self.device.type == "cuda"
        return {
            "device": str(self.device),
            "elapsed_seconds": elapsed_seconds,
            "gpu_hours": elapsed_seconds / 3600.0 if cuda_active else 0.0,
            "peak_allocated_vram_bytes": (
                torch.cuda.max_memory_allocated(self.device) if cuda_active else None
            ),
            "peak_reserved_vram_bytes": (
                torch.cuda.max_memory_reserved(self.device) if cuda_active else None
            ),
        }

    def elapsed_seconds(self) -> float:
        """Return elapsed wall time without forcing a device synchronization."""
        return time.perf_counter() - self.started_at


class ExperimentRegistry:
    """Create and finalize one immutable run directory contract."""

    def __init__(
        self,
        *,
        artifact_root: Path,
        descriptor: RunDescriptor,
        config_path: Path,
        raw_data_roots: Sequence[Path] = (),
        macs: int | None = None,
        flops: int | None = None,
        allow_existing: bool = False,
    ) -> None:
        self.descriptor = descriptor
        self.run_directory = artifact_root.resolve() / descriptor.run_id
        assert_output_paths_safe([self.run_directory], raw_data_roots)
        if self.run_directory.exists() and not allow_existing:
            raise FileExistsError(f"Run directory already exists: {self.run_directory}")
        self.run_directory.mkdir(parents=True, exist_ok=allow_existing)
        self.checkpoint_directory = self.run_directory / "checkpoints"
        self.log_directory = self.run_directory / "logs"
        self.checkpoint_directory.mkdir(exist_ok=allow_existing)
        self.log_directory.mkdir(exist_ok=allow_existing)

        source = cast(DictConfig, OmegaConf.load(config_path))
        OmegaConf.resolve(source)
        resolved_config = OmegaConf.to_yaml(source, resolve=True)
        atomic_write_text(self.run_directory / "config.yaml", resolved_config)
        commit, dirty = _git_state()
        metadata: dict[str, Any] = {
            "run_id": descriptor.run_id,
            "git_commit": commit,
            "repository_dirty": dirty,
            "config_source_path": config_path.as_posix(),
            "config_sha256": text_digest(resolved_config.rstrip() + "\n"),
            "data_manifest_path": descriptor.data_manifest_path.as_posix(),
            "data_manifest_sha256": file_digest(descriptor.data_manifest_path),
            "split_sha256": dict(descriptor.split_hashes),
            "seed": descriptor.seed,
            "model": descriptor.model,
            "loss": descriptor.loss,
            "optimizer": descriptor.optimizer,
            "scheduler": descriptor.scheduler,
            "hardware": _hardware(),
            "start_timestamp_utc": _timestamp(),
            "end_timestamp_utc": None,
            "gpu_hours": None,
            "peak_allocated_vram_bytes": None,
            "peak_reserved_vram_bytes": None,
            "parameter_count": descriptor.parameter_count,
            "complexity": {
                "macs": macs,
                "flops": flops,
                "input_specification": list(descriptor.input_specification),
            },
            "best_validation_checkpoint": None,
            "status": "running",
            "error_trace": None,
            "test_access": {
                "allowed": descriptor.test_access_allowed,
                "accessed": descriptor.test_accessed,
                "audit_log": descriptor.test_access_audit_log,
            },
        }
        self._metadata = metadata
        atomic_write_json(self.run_directory / "metadata.json", metadata)

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Expose the current metadata as a read-only mapping contract."""
        return self._metadata

    def log_epoch(self, metrics: Mapping[str, Any]) -> None:
        """Append one epoch/validation-check metric record."""
        append_jsonl(
            self.run_directory / "metrics_per_epoch.jsonl",
            dict(metrics),
        )

    def write_validation_cases(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        """Atomically write patient-level validation results."""
        if not rows:
            raise ValueError("Validation case rows cannot be empty")
        atomic_write_csv(self.run_directory / "validation_per_case.csv", rows)

    def write_resource_profile(self, profile: Mapping[str, Any]) -> None:
        """Write the machine-readable resource profile."""
        atomic_write_json(self.run_directory / "resource_profile.json", profile)

    def finalize(
        self,
        *,
        status: Literal["completed", "failed", "invalid"],
        resource_profile: Mapping[str, Any],
        best_validation_checkpoint: str | None = None,
        error_trace: str | None = None,
    ) -> None:
        """Close the run with resource, checkpoint, and failure metadata."""
        if status == "completed" and error_trace:
            raise ValueError("A completed run cannot have an error trace")
        if status == "failed" and not error_trace:
            raise ValueError("A failed run requires an error trace")
        required_resources = {
            "gpu_hours",
            "peak_allocated_vram_bytes",
            "peak_reserved_vram_bytes",
        }
        if not required_resources.issubset(resource_profile):
            raise ValueError("Resource profile is missing required fields")
        self.write_resource_profile(resource_profile)
        self._metadata.update(
            {
                "end_timestamp_utc": _timestamp(),
                "gpu_hours": resource_profile["gpu_hours"],
                "peak_allocated_vram_bytes": resource_profile[
                    "peak_allocated_vram_bytes"
                ],
                "peak_reserved_vram_bytes": resource_profile[
                    "peak_reserved_vram_bytes"
                ],
                "best_validation_checkpoint": best_validation_checkpoint,
                "status": status,
                "error_trace": error_trace,
            }
        )
        atomic_write_json(
            self.run_directory / "metadata.json",
            self._metadata,
        )
