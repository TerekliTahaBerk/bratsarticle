#!/usr/bin/env python3
"""Run one real nnU-Net training step for MPS hardware feasibility only."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnunetv2.run.run_training import get_trainer_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        choices=("2d", "3d_fullres"),
        required=True,
    )
    parser.add_argument("--plans-identifier", required=True)
    parser.add_argument(
        "--trainer",
        default="nnUNetTrainerSeed20260730",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--assert-no-active-mps-lock",
        required=True,
        type=Path,
        help="Abort if this training-queue lock exists.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    repository = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _required_environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return Path(value).expanduser().resolve()


def _mps_memory() -> dict[str, int | None]:
    current = int(torch.mps.current_allocated_memory())
    driver_function = getattr(torch.mps, "driver_allocated_memory", None)
    recommended_function = getattr(torch.mps, "recommended_max_memory", None)
    return {
        "framework_current_allocated_unified_memory_bytes": current,
        "driver_allocated_unified_memory_bytes": (
            int(driver_function()) if driver_function is not None else None
        ),
        "framework_recommended_max_unified_memory_bytes": (
            int(recommended_function())
            if recommended_function is not None
            else None
        ),
    }


class _MemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self.framework_peak_bytes = 0
        self.driver_peak_bytes = 0

    def _sample(self) -> None:
        while not self._stop.wait(0.01):
            memory = _mps_memory()
            self.framework_peak_bytes = max(
                self.framework_peak_bytes,
                int(
                    memory[
                        "framework_current_allocated_unified_memory_bytes"
                    ]
                    or 0
                ),
            )
            self.driver_peak_bytes = max(
                self.driver_peak_bytes,
                int(memory["driver_allocated_unified_memory_bytes"] or 0),
            )

    def __enter__(self) -> _MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join()
        memory = _mps_memory()
        self.framework_peak_bytes = max(
            self.framework_peak_bytes,
            int(
                memory["framework_current_allocated_unified_memory_bytes"] or 0
            ),
        )
        self.driver_peak_bytes = max(
            self.driver_peak_bytes,
            int(memory["driver_allocated_unified_memory_bytes"] or 0),
        )


def _atomic_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    commit, dirty = _git_state()
    if dirty:
        raise RuntimeError(
            f"MPS preflight requires a clean repository; commit={commit}"
        )
    if args.assert_no_active_mps_lock.exists():
        raise RuntimeError(
            "Another guarded MPS queue is active: "
            f"{args.assert_no_active_mps_lock}"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable")
    if os.environ.get("nnUNet_n_proc_DA") != "0":
        raise RuntimeError("Set nnUNet_n_proc_DA=0 for seeded preflight")

    preprocessed_root = _required_environment_path("nnUNet_preprocessed")
    plans_path = (
        preprocessed_root
        / "Dataset501_BraTS2020Q1Q2"
        / f"{args.plans_identifier}.json"
    )
    split_path = (
        preprocessed_root
        / "Dataset501_BraTS2020Q1Q2"
        / "splits_final.json"
    )
    if not plans_path.is_file() or not split_path.is_file():
        raise FileNotFoundError(
            "Required nnU-Net plans or frozen splits are missing"
        )

    plans = json.loads(plans_path.read_text(encoding="utf-8"))
    configuration = plans["configurations"][args.configuration]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "hardware_feasibility_not_model_evaluation",
        "status": "running",
        "dataset_id": 501,
        "external_data_accessed": False,
        "configuration": args.configuration,
        "plans_identifier": args.plans_identifier,
        "plans_sha256": _file_sha256(plans_path),
        "splits_final_sha256": _file_sha256(split_path),
        "trainer": args.trainer,
        "fold_zero_indexed": args.fold,
        "seed": int(args.trainer.removeprefix("nnUNetTrainerSeed")),
        "planned_batch_size": int(configuration["batch_size"]),
        "planned_patch_size": [
            int(value) for value in configuration["patch_size"]
        ],
        "architecture": configuration["architecture"]["network_class_name"],
        "git_commit": commit,
        "repository_dirty_at_start": False,
        "nnunetv2_version": importlib.metadata.version("nnunetv2"),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "device": "mps",
    }
    _atomic_json(args.output, payload)

    trainer: Any = None
    batch: dict[str, Any] | None = None
    started = time.perf_counter()
    sampler = _MemorySampler()
    try:
        torch.mps.empty_cache()
        with sampler:
            trainer = get_trainer_from_args(
                "501",
                args.configuration,
                args.fold,
                trainer_name=args.trainer,
                plans_identifier=args.plans_identifier,
                continue_training=False,
                device=torch.device("mps"),
            )
            trainer.on_train_start()
            actual_batch_size = int(trainer.batch_size)
            batch = next(trainer.dataloader_train)
            torch.mps.synchronize()
            step_started = time.perf_counter()
            result = trainer.train_step(batch)
            torch.mps.synchronize()
            step_seconds = time.perf_counter() - step_started

        loss_value = float(np.asarray(result["loss"]).item())
        payload.update(
            {
                "status": "pass",
                "actual_batch_size": actual_batch_size,
                "loss_finite": bool(np.isfinite(loss_value)),
                "diagnostic_training_loss": loss_value,
                "one_step_seconds": step_seconds,
                "total_seconds": time.perf_counter() - started,
                "framework_peak_allocated_unified_memory_bytes": (
                    sampler.framework_peak_bytes
                ),
                "driver_peak_allocated_unified_memory_bytes": (
                    sampler.driver_peak_bytes
                ),
                "memory_after_step": _mps_memory(),
            }
        )
        if not payload["loss_finite"]:
            payload["status"] = "fail"
            payload["failure_reason"] = "nonfinite_training_loss"
    except Exception as error:
        payload.update(
            {
                "status": "fail",
                "failure_type": type(error).__name__,
                "failure_reason": str(error),
                "traceback": traceback.format_exc(),
                "total_seconds": time.perf_counter() - started,
                "framework_peak_allocated_unified_memory_bytes": (
                    sampler.framework_peak_bytes
                ),
                "driver_peak_allocated_unified_memory_bytes": (
                    sampler.driver_peak_bytes
                ),
            }
        )
    finally:
        if trainer is not None and trainer.optimizer is not None:
            trainer.optimizer.zero_grad(set_to_none=True)
        del batch
        del trainer
        torch.mps.empty_cache()
        _atomic_json(args.output, payload)
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
