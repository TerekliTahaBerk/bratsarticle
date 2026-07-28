"""Validate and serialize the frozen Gate 7 protocol contracts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf

from bratsarticle.experiments.fairness import (
    load_compute_matched_protocol,
    load_convergence_matched_protocol,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def _registry_contract(path: Path) -> dict[str, Any]:
    root = cast(DictConfig, OmegaConf.load(path))
    OmegaConf.resolve(root)
    registry = root.registry
    files = {str(key): str(value) for key, value in registry.files.items()}
    directories = {str(key): str(value) for key, value in registry.directories.items()}
    expected_files = {
        "resolved_config": "config.yaml",
        "metadata": "metadata.json",
        "epoch_metrics": "metrics_per_epoch.jsonl",
        "validation_cases": "validation_per_case.csv",
        "resource_profile": "resource_profile.json",
    }
    expected_directories = {"checkpoints": "checkpoints", "logs": "logs"}
    if files != expected_files or directories != expected_directories:
        raise ValueError("Registry layout does not match the frozen artifact contract")
    return {
        "config_path": path.as_posix(),
        "config_sha256": file_digest(path),
        "artifact_root": str(registry.artifact_root),
        "allow_existing_run_directory": bool(registry.allow_existing_run_directory),
        "files": files,
        "directories": directories,
        "status_values": [str(value) for value in registry.status_values],
        "test_access_default": {
            "allowed": bool(registry.test_access_default.allowed),
            "accessed": bool(registry.test_access_default.accessed),
            "audit_log": str(registry.test_access_default.audit_log),
        },
    }


def run(
    *,
    compute_path: Path,
    convergence_path: Path,
    registry_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Validate the Gate 7 configs and write their hashes and host eligibility."""
    compute = load_compute_matched_protocol(compute_path)
    convergence = load_convergence_matched_protocol(convergence_path)
    if compute.gpu_model != convergence.gpu_model:
        raise ValueError("Both fairness regimes must target the same GPU")
    cuda_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
    host_eligible = (
        torch.cuda.is_available()
        and len(cuda_names) == 1
        and cuda_names[0] == compute.gpu_model
    )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass",
        "compute_matched": {
            "config_path": compute_path.as_posix(),
            "config_sha256": file_digest(compute_path),
            "resolved": asdict(compute),
        },
        "convergence_matched": {
            "config_path": convergence_path.as_posix(),
            "config_sha256": file_digest(convergence_path),
            "resolved": asdict(convergence),
        },
        "registry": _registry_contract(registry_path),
        "host_eligibility": {
            "required_gpu_model": compute.gpu_model,
            "cuda_available": torch.cuda.is_available(),
            "visible_cuda_devices": cuda_names,
            "eligible_for_reportable_pilots": host_eligible,
            "action_if_false": (
                "Do not start reportable pilot or full training on this host"
            ),
        },
    }
    atomic_write_json(output, payload)
    return payload


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compute-config",
        type=Path,
        default=Path("configs/protocols/compute_matched.yaml"),
    )
    parser.add_argument(
        "--convergence-config",
        type=Path,
        default=Path("configs/protocols/convergence_matched.yaml"),
    )
    parser.add_argument(
        "--registry-config",
        type=Path,
        default=Path("configs/experiments/registry.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gate7_protocol_validation.json"),
    )
    arguments = parser.parse_args()
    run(
        compute_path=arguments.compute_config,
        convergence_path=arguments.convergence_config,
        registry_path=arguments.registry_config,
        output=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
