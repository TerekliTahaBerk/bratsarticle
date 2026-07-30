"""Seed-locked trainers for the official nnU-Net v2 implementation.

These subclasses intentionally preserve nnU-Net's architecture, optimizer,
schedule, augmentations, loss, and default training duration. They only make
the five frozen training seeds explicit and record their provenance.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import (  # type: ignore[import-untyped]
    nnUNetTrainer,
)

from bratsarticle.models.resource_profile import profile_torch_module

DEFAULT_DEVICE = torch.device("cuda")
Q1Q2_BUDGET_SENSITIVITY_STEPS = frozenset({2_000, 10_000})


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_state() -> tuple[str, bool]:
    repository_root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(repository_root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class Q1Q2SeededNNUNetTrainer(nnUNetTrainer):  # type: ignore[misc]
    """Preserve official training defaults while enforcing a declared seed."""

    Q1Q2_SEED: int | None = None

    def __init__(
        self,
        plans: dict[str, Any],
        configuration: str,
        fold: int,
        dataset_json: dict[str, Any],
        device: torch.device = DEFAULT_DEVICE,
    ) -> None:
        if self.Q1Q2_SEED is None:
            raise RuntimeError("Concrete seeded trainer class is required")
        seed = self.Q1Q2_SEED
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            raise RuntimeError(
                f"Set PYTHONHASHSEED={seed} before launching this trainer"
            )
        if os.environ.get("nnUNet_n_proc_DA") != "0":
            raise RuntimeError(
                "Reportable seeded training requires nnUNet_n_proc_DA=0; "
                "multi-process NonDetMultiThreadedAugmenter is prohibited"
            )
        git_commit, repository_dirty = _git_state()
        if repository_dirty:
            raise RuntimeError(
                "Reportable nnU-Net training requires a clean repository; "
                f"resolved commit={git_commit}"
            )
        self.q1q2_git_commit = git_commit
        self.q1q2_continuation_requested = (
            os.environ.get("Q1Q2_CONTINUATION_REQUESTED") == "1"
        )
        self.q1q2_started_at = 0.0
        self.q1q2_elapsed_before_session = 0.0
        self.q1q2_framework_peak_bytes = 0
        self.q1q2_driver_peak_bytes = 0
        self.q1q2_timing_warmup_steps = int(
            os.environ.get("Q1Q2_TIMING_WARMUP_STEPS", "20")
        )
        self.q1q2_timing_measurement_count = int(
            os.environ.get("Q1Q2_TIMING_MEASUREMENT_COUNT", "100")
        )
        if self.q1q2_timing_warmup_steps < 0 or self.q1q2_timing_measurement_count < 1:
            raise RuntimeError("Q1/Q2 resource timing contract is invalid")
        self.q1q2_train_step_calls = 0
        self.q1q2_training_step_seconds: list[float] = []
        self._seed_all(device)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.logger.update_config(
            {
                "q1q2_seed_contract": {
                    "seed": seed,
                    "python_hash_seed": os.environ["PYTHONHASHSEED"],
                    "data_augmentation_processes": 0,
                    "deterministic_algorithms": True,
                    "deterministic_warn_only": device.type == "mps",
                    "official_training_defaults_preserved": True,
                }
            }
        )

    def _seed_all(self, device: torch.device) -> None:
        seed = self.Q1Q2_SEED
        if seed is None:
            raise RuntimeError("Seed is undefined")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(
            True,
            warn_only=device.type == "mps",
        )

    def _sample_mps_memory(self) -> None:
        if self.device.type != "mps":
            return
        self.q1q2_framework_peak_bytes = max(
            self.q1q2_framework_peak_bytes,
            int(torch.mps.current_allocated_memory()),
        )
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        if driver is not None:
            self.q1q2_driver_peak_bytes = max(
                self.q1q2_driver_peak_bytes,
                int(driver()),
            )

    def _metadata_path(self) -> Path:
        return Path(self.output_folder) / "q1q2_run_metadata.json"

    def _synchronize(self) -> None:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _timing_payload(self) -> dict[str, Any]:
        values = np.asarray(self.q1q2_training_step_seconds, dtype=np.float64)
        return {
            "resource_profile_protocol": os.environ.get(
                "Q1Q2_RESOURCE_PROFILE_PROTOCOL",
                "unavailable",
            ),
            "resource_profile_protocol_sha256": os.environ.get(
                "Q1Q2_RESOURCE_PROFILE_PROTOCOL_SHA256",
                "unavailable",
            ),
            "training_step_timing_warmup_iterations": (self.q1q2_timing_warmup_steps),
            "training_step_timing_target_count": (self.q1q2_timing_measurement_count),
            "synchronized_training_step_seconds": values.tolist(),
            "synchronized_training_step_measurement_count": len(values),
            "synchronized_training_step_mean_seconds": (
                float(values.mean()) if len(values) else None
            ),
            "synchronized_training_step_p50_seconds": (
                float(np.quantile(values, 0.5)) if len(values) else None
            ),
            "synchronized_training_step_p95_seconds": (
                float(np.quantile(values, 0.95)) if len(values) else None
            ),
        }

    def _update_metadata(self, **updates: Any) -> None:
        path = self._metadata_path()
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.is_file()
            else {"schema_version": 1}
        )
        payload.update(updates)
        _atomic_json(path, payload)

    def on_train_start(self) -> None:
        self._seed_all(self.device)
        super().on_train_start()
        existing_path = self._metadata_path()
        if self.q1q2_continuation_requested and existing_path.is_file():
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            self.q1q2_elapsed_before_session = float(
                existing.get("cumulative_elapsed_seconds", 0.0)
            )
            self.q1q2_framework_peak_bytes = int(
                existing.get(
                    "framework_peak_allocated_unified_memory_bytes",
                    0,
                )
            )
            self.q1q2_driver_peak_bytes = int(
                existing.get(
                    "driver_peak_allocated_unified_memory_bytes",
                    0,
                )
            )
            self.q1q2_training_step_seconds = [
                float(value)
                for value in existing.get(
                    "synchronized_training_step_seconds",
                    [],
                )
            ]
            self.q1q2_train_step_calls = int(
                existing.get("completed_optimizer_steps", 0)
            )
        self.q1q2_started_at = time.perf_counter()
        self._sample_mps_memory()
        split_path = Path(self.preprocessed_dataset_folder_base) / ("splits_final.json")
        patch_size = tuple(
            int(value) for value in self.configuration_manager.patch_size
        )
        static_profile = profile_torch_module(
            self.network,
            input_shape=(1, int(self.num_input_channels), *patch_size),
        )
        static_profile.update(
            {
                "receptive_field_proxy_definition": (
                    "full_declared_nnunet_patch_for_self_configuring_encoder_decoder"
                ),
                "receptive_field_proxy_voxels_per_axis": list(patch_size),
            }
        )
        payload = {
            "schema_version": 1,
            "trainer": self.__class__.__name__,
            "seed": self.Q1Q2_SEED,
            "fold_zero_indexed": self.fold,
            "configuration": self.configuration_name,
            "device": str(self.device),
            "git_commit": self.q1q2_git_commit,
            "repository_dirty_at_start": False,
            "trainer_source_sha256": _file_sha256(Path(__file__).resolve()),
            "environment_lock_sha256": os.environ.get(
                "Q1Q2_ENVIRONMENT_LOCK_SHA256",
                "missing",
            ),
            "requirements_lock_sha256": os.environ.get(
                "Q1Q2_REQUIREMENTS_LOCK_SHA256",
                "missing",
            ),
            "hardware_preflight_sha256": os.environ.get(
                "Q1Q2_HARDWARE_PREFLIGHT_SHA256",
                "missing",
            ),
            "nnunetv2_version": importlib.metadata.version("nnunetv2"),
            "plans_sha256": _canonical_sha256(self.plans_manager.plans),
            "dataset_json_sha256": _canonical_sha256(self.dataset_json),
            "splits_final_sha256": (
                _file_sha256(split_path) if split_path.is_file() else "missing"
            ),
            "python_hash_seed": os.environ["PYTHONHASHSEED"],
            "data_augmentation_processes": 0,
            "deterministic_algorithms": True,
            "deterministic_warn_only": self.device.type == "mps",
            "continuation_requested": self.q1q2_continuation_requested,
            "cumulative_elapsed_seconds_before_session": (
                self.q1q2_elapsed_before_session
            ),
            "parameter_count": sum(
                parameter.numel()
                for parameter in self.network.parameters()
                if parameter.requires_grad
            ),
            "static_profile": static_profile,
            "official_defaults": {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": (self.oversample_foreground_percent),
                "iterations_per_epoch": self.num_iterations_per_epoch,
                "validation_iterations_per_epoch": (self.num_val_iterations_per_epoch),
                "epochs": self.num_epochs,
                "deep_supervision": self.enable_deep_supervision,
            },
            "restart_reproducibility_note": (
                "Fresh reruns are seed-locked. Interrupted continuation is "
                "not claimed bitwise-identical because upstream nnU-Net "
                "checkpoints do not serialize every augmentation RNG state."
            ),
            "status": "running",
            **self._timing_payload(),
        }
        _atomic_json(self._metadata_path(), payload)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        measure = (
            self.q1q2_train_step_calls >= self.q1q2_timing_warmup_steps
            and len(self.q1q2_training_step_seconds)
            < self.q1q2_timing_measurement_count
        )
        if measure:
            self._synchronize()
            started = time.perf_counter()
        result = cast(dict[str, Any], super().train_step(batch))
        if measure:
            self._synchronize()
            self.q1q2_training_step_seconds.append(time.perf_counter() - started)
        self.q1q2_train_step_calls += 1
        self._sample_mps_memory()
        return result

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        result = cast(dict[str, Any], super().validation_step(batch))
        self._sample_mps_memory()
        return result

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        self._sample_mps_memory()
        completed_optimizer_steps = int(self.current_epoch) * int(
            self.num_iterations_per_epoch
        )
        if completed_optimizer_steps in Q1Q2_BUDGET_SENSITIVITY_STEPS:
            self.save_checkpoint(
                str(
                    Path(self.output_folder)
                    / (f"checkpoint_q1q2_step_{completed_optimizer_steps}.pth")
                )
            )
        cumulative = self.q1q2_elapsed_before_session + (
            time.perf_counter() - self.q1q2_started_at
        )
        self._update_metadata(
            last_completed_epoch=int(self.current_epoch),
            completed_optimizer_steps=completed_optimizer_steps,
            cumulative_elapsed_seconds=cumulative,
            framework_peak_allocated_unified_memory_bytes=(
                self.q1q2_framework_peak_bytes
            ),
            driver_peak_allocated_unified_memory_bytes=(self.q1q2_driver_peak_bytes),
            **self._timing_payload(),
        )

    def on_train_end(self) -> None:
        super().on_train_end()
        self._sample_mps_memory()
        checkpoint_best = Path(self.output_folder) / "checkpoint_best.pth"
        checkpoint_final = Path(self.output_folder) / "checkpoint_final.pth"
        milestone_paths = {
            str(step): (Path(self.output_folder) / f"checkpoint_q1q2_step_{step}.pth")
            for step in sorted(Q1Q2_BUDGET_SENSITIVITY_STEPS)
        }
        session_elapsed_seconds = time.perf_counter() - self.q1q2_started_at
        cumulative_elapsed_seconds = (
            self.q1q2_elapsed_before_session + session_elapsed_seconds
        )
        self._update_metadata(
            status="completed",
            completed_epochs=int(self.current_epoch),
            final_learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            elapsed_seconds_this_session=session_elapsed_seconds,
            cumulative_elapsed_seconds=cumulative_elapsed_seconds,
            accelerator_hours=cumulative_elapsed_seconds / 3600.0,
            framework_peak_allocated_unified_memory_bytes=(
                self.q1q2_framework_peak_bytes
            ),
            driver_peak_allocated_unified_memory_bytes=(self.q1q2_driver_peak_bytes),
            **self._timing_payload(),
            checkpoint_best_sha256=(
                _file_sha256(checkpoint_best)
                if checkpoint_best.is_file()
                else "missing"
            ),
            checkpoint_final_sha256=(
                _file_sha256(checkpoint_final)
                if checkpoint_final.is_file()
                else "missing"
            ),
            budget_sensitivity_checkpoints={
                step: {
                    "path": path.as_posix(),
                    "sha256": (_file_sha256(path) if path.is_file() else "missing"),
                }
                for step, path in milestone_paths.items()
            },
        )


class nnUNetTrainerSeed20260730(Q1Q2SeededNNUNetTrainer):
    Q1Q2_SEED = 20260730


class nnUNetTrainerSeed20260731(Q1Q2SeededNNUNetTrainer):
    Q1Q2_SEED = 20260731


class nnUNetTrainerSeed20260732(Q1Q2SeededNNUNetTrainer):
    Q1Q2_SEED = 20260732


class nnUNetTrainerSeed20260733(Q1Q2SeededNNUNetTrainer):
    Q1Q2_SEED = 20260733


class nnUNetTrainerSeed20260734(Q1Q2SeededNNUNetTrainer):
    Q1Q2_SEED = 20260734
