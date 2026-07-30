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
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

DEFAULT_DEVICE = torch.device("cuda")


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


class Q1Q2SeededNNUNetTrainer(nnUNetTrainer):
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
        self.q1q2_continuation_requested = bool(
            plans.get("continue_training", False)
        )
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

    def on_train_start(self) -> None:
        self._seed_all(self.device)
        super().on_train_start()
        split_path = Path(self.preprocessed_dataset_folder_base) / (
            "splits_final.json"
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
            "official_defaults": {
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "oversample_foreground_percent": (
                    self.oversample_foreground_percent
                ),
                "iterations_per_epoch": self.num_iterations_per_epoch,
                "validation_iterations_per_epoch": (
                    self.num_val_iterations_per_epoch
                ),
                "epochs": self.num_epochs,
                "deep_supervision": self.enable_deep_supervision,
            },
            "restart_reproducibility_note": (
                "Fresh reruns are seed-locked. Interrupted continuation is "
                "not claimed bitwise-identical because upstream nnU-Net "
                "checkpoints do not serialize every augmentation RNG state."
            ),
        }
        destination = Path(self.output_folder) / "q1q2_run_metadata.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)


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
