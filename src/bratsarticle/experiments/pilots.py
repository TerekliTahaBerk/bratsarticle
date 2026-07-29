"""Gate 8 pilot-plan parsing, validation, and hardware preflight."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from bratsarticle.experiments.fairness import load_compute_matched_protocol
from bratsarticle.experiments.hardware import (
    accelerator_available,
    accelerator_device_names,
)
from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.training.loss_catalog import load_loss_catalog
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.paths import is_relative_to
from bratsarticle.utils.serialization import atomic_write_text

_ARM_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True)
class PilotArm:
    """One predeclared architecture/loss screening run."""

    arm_id: str
    screen: str
    model_config_path: Path
    loss_name: str
    seed: int


@dataclass(frozen=True)
class PilotPlan:
    """Resolved single-seed Gate 8 pilot plan."""

    name: str
    status: str
    seed: int
    fairness_protocol_path: Path
    registry_config_path: Path
    artifact_root: Path
    canonical_manifest_path: Path
    split_dir: Path
    preprocessing_config_path: Path
    evaluation_config_path: Path
    maximum_optimizer_steps: int
    maximum_gpu_hours: float
    validation_frequency_optimizer_steps: int
    minimum_completed_validation_checks: int
    training_workers: int
    validation_workers: int
    training_memory_subjects: int
    validation_memory_subjects: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_optimizer_steps: int
    minimum_learning_rate_fraction: float
    mixed_precision: bool
    arms: tuple[PilotArm, ...]
    loss_reuse: dict[str, str]
    elimination: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate budgets, unique arms, and screen scope."""
        if self.maximum_optimizer_steps < 1 or self.maximum_gpu_hours <= 0:
            raise ValueError("Pilot budgets must be positive")
        if self.validation_frequency_optimizer_steps < 1:
            raise ValueError("Validation frequency must be positive")
        possible_checks = (
            self.maximum_optimizer_steps // self.validation_frequency_optimizer_steps
        )
        if not 1 <= self.minimum_completed_validation_checks <= possible_checks:
            raise ValueError("Minimum validation checks do not fit the step budget")
        if self.warmup_optimizer_steps >= self.maximum_optimizer_steps:
            raise ValueError("Pilot warm-up must end before the step budget")
        if self.training_workers < 0 or self.validation_workers < 0:
            raise ValueError("DataLoader worker counts cannot be negative")
        if self.training_memory_subjects < 1 or self.validation_memory_subjects < 1:
            raise ValueError("Pilot memory-cache subject counts must be positive")
        identifiers = [arm.arm_id for arm in self.arms]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Pilot arm identifiers must be unique")
        if not self.arms:
            raise ValueError("At least one pilot arm is required")
        if any(not _ARM_ID_PATTERN.fullmatch(value) for value in identifiers):
            raise ValueError("Pilot arm identifier is unsafe")
        if any(value not in identifiers for value in self.loss_reuse.values()):
            raise ValueError("Reused loss-screen arm must reference a declared arm")
        if bool(self.elimination["internal_test_permitted"]):
            raise ValueError("Pilot elimination cannot use the internal test")


def _architecture_arms(config: DictConfig, seed: int) -> list[PilotArm]:
    fixed_loss = str(config.fixed_loss)
    return [
        PilotArm(
            arm_id=str(raw.id),
            screen="architecture",
            model_config_path=Path(str(raw.model_config)),
            loss_name=fixed_loss,
            seed=seed,
        )
        for raw in config.arms
    ]


def _loss_arms(config: DictConfig, seed: int) -> list[PilotArm]:
    fixed_model = Path(str(config.fixed_model_config))
    return [
        PilotArm(
            arm_id=str(raw.id),
            screen="loss",
            model_config_path=fixed_model,
            loss_name=str(raw.loss),
            seed=seed,
        )
        for raw in config.arms
    ]


def load_pilot_plan(path: Path) -> PilotPlan:
    """Load the Gate 8 pilot plan without opening any test manifest."""
    root = cast(DictConfig, OmegaConf.load(path))
    pilot = root.pilot
    seed = int(pilot.seed)
    arms = (
        *_architecture_arms(pilot.screen.architecture, seed),
        *_loss_arms(pilot.screen.loss, seed),
    )
    plan = PilotPlan(
        name=str(pilot.name),
        status=str(pilot.status),
        seed=seed,
        fairness_protocol_path=Path(str(pilot.fairness_protocol)),
        registry_config_path=Path(str(pilot.registry_config)),
        artifact_root=Path(str(pilot.artifact_root)),
        canonical_manifest_path=Path(str(pilot.data.canonical_manifest)),
        split_dir=Path(str(pilot.data.split_dir)),
        preprocessing_config_path=Path(str(pilot.data.preprocessing_config)),
        evaluation_config_path=Path(str(pilot.data.evaluation_config)),
        maximum_optimizer_steps=int(pilot.budget.maximum_optimizer_steps),
        maximum_gpu_hours=float(pilot.budget.maximum_gpu_hours),
        validation_frequency_optimizer_steps=int(
            pilot.budget.validation_frequency_optimizer_steps
        ),
        minimum_completed_validation_checks=int(
            pilot.budget.minimum_completed_validation_checks
        ),
        training_workers=int(pilot.data.training_workers),
        validation_workers=int(pilot.data.validation_workers),
        training_memory_subjects=int(pilot.data.training_memory_subjects),
        validation_memory_subjects=int(pilot.data.validation_memory_subjects),
        optimizer=str(pilot.optimization.optimizer),
        learning_rate=float(pilot.optimization.learning_rate),
        weight_decay=float(pilot.optimization.weight_decay),
        scheduler=str(pilot.optimization.scheduler),
        warmup_optimizer_steps=int(pilot.optimization.warmup_optimizer_steps),
        minimum_learning_rate_fraction=float(
            pilot.optimization.minimum_learning_rate_fraction
        ),
        mixed_precision=bool(pilot.optimization.mixed_precision),
        arms=arms,
        loss_reuse={
            str(loss): str(arm_id)
            for loss, arm_id in pilot.screen.loss.reuse_architecture_arm.items()
        },
        elimination=cast(
            dict[str, Any],
            OmegaConf.to_container(
                pilot.elimination,
                resolve=True,
            ),
        ),
    )
    _validate_references(plan)
    return plan


def write_mps_diagnostic_config(source: Path, destination: Path) -> Path:
    """Create a short non-selection diagnostic from the frozen pilot config."""
    root = cast(DictConfig, OmegaConf.load(source))
    root.pilot.name = "gate8_mps_integration_diagnostic"
    root.pilot.status = "diagnostic_only_not_for_selection"
    root.pilot.budget.maximum_optimizer_steps = 10
    root.pilot.budget.maximum_gpu_hours = 0.25
    root.pilot.budget.validation_frequency_optimizer_steps = 10
    root.pilot.budget.minimum_completed_validation_checks = 1
    root.pilot.optimization.warmup_optimizer_steps = 2
    root.pilot.data.training_workers = 0
    root.pilot.data.validation_workers = 0
    root.pilot.data.training_memory_subjects = 1
    root.pilot.data.validation_memory_subjects = 1
    atomic_write_text(
        destination,
        OmegaConf.to_yaml(root, resolve=False),
    )
    return destination


def _validate_references(plan: PilotPlan) -> None:
    required_paths = (
        plan.fairness_protocol_path,
        plan.registry_config_path,
        plan.canonical_manifest_path,
        plan.preprocessing_config_path,
        plan.evaluation_config_path,
        plan.split_dir / "train.csv",
        plan.split_dir / "validation.csv",
        plan.split_dir / "split_metadata.json",
    )
    missing = [path.as_posix() for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing pilot plan references: {missing}")
    catalog = {
        config.name for config in load_loss_catalog(Path("configs/losses/catalog.yaml"))
    }
    for arm in plan.arms:
        load_model_config(arm.model_config_path)
        if arm.loss_name not in catalog:
            raise ValueError(f"Unknown pilot loss: {arm.loss_name}")
    if len(plan.arms) != 12:
        raise ValueError("Gate 8 plan must contain 12 unique non-factorial arms")
    pairs = {(arm.model_config_path, arm.loss_name) for arm in plan.arms}
    if len(pairs) != len(plan.arms):
        raise ValueError("Duplicate model/loss pilot combination")


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def pilot_plan_record(plan: PilotPlan, source_path: Path) -> dict[str, Any]:
    """Return a machine-readable, hash-linked representation of the plan."""
    return {
        "name": plan.name,
        "status": plan.status,
        "source_config": source_path.as_posix(),
        "source_config_sha256": file_digest(source_path),
        "git": _git_state(),
        "seed": plan.seed,
        "budget": {
            "maximum_optimizer_steps": plan.maximum_optimizer_steps,
            "maximum_gpu_hours": plan.maximum_gpu_hours,
            "validation_frequency_optimizer_steps": (
                plan.validation_frequency_optimizer_steps
            ),
            "minimum_completed_validation_checks": (
                plan.minimum_completed_validation_checks
            ),
        },
        "data_loader_workers": {
            "training": plan.training_workers,
            "validation": plan.validation_workers,
        },
        "memory_cache_subjects": {
            "training": plan.training_memory_subjects,
            "validation": plan.validation_memory_subjects,
        },
        "optimization": {
            "optimizer": plan.optimizer,
            "learning_rate": plan.learning_rate,
            "weight_decay": plan.weight_decay,
            "scheduler": plan.scheduler,
            "warmup_optimizer_steps": plan.warmup_optimizer_steps,
            "minimum_learning_rate_fraction": (plan.minimum_learning_rate_fraction),
            "mixed_precision": plan.mixed_precision,
        },
        "arms": [
            {
                "arm_id": arm.arm_id,
                "screen": arm.screen,
                "model_config": arm.model_config_path.as_posix(),
                "model_config_sha256": file_digest(arm.model_config_path),
                "loss": arm.loss_name,
                "seed": arm.seed,
            }
            for arm in plan.arms
        ],
        "loss_screen_reused_arms": plan.loss_reuse,
        "elimination": plan.elimination,
        "factorial_combinations_not_scheduled": True,
        "internal_test_access": False,
    }


def pilot_preflight(plan: PilotPlan) -> dict[str, Any]:
    """Check hardware and development-data prerequisites without training."""
    fairness = load_compute_matched_protocol(plan.fairness_protocol_path)
    visible_devices = accelerator_device_names(fairness.accelerator_backend)
    backend_available = accelerator_available(fairness.accelerator_backend)
    data_root_value = os.environ.get("BRATS2020_ROOT")
    data_root = (
        None if not data_root_value else Path(data_root_value).expanduser().resolve()
    )
    cache_root_value = os.environ.get("BRATS_CACHE_ROOT")
    cache_root = (
        None if not cache_root_value else Path(cache_root_value).expanduser().resolve()
    )
    checks = {
        "accelerator_backend_available": backend_available,
        "exactly_one_visible_accelerator": len(visible_devices) == 1,
        "gpu_model_matches_frozen_protocol": (
            len(visible_devices) == 1 and visible_devices[0] == fairness.gpu_model
        ),
        "brats2020_root_set": data_root is not None,
        "brats2020_root_exists": bool(data_root is not None and data_root.is_dir()),
        "brats_cache_root_set": cache_root is not None,
        "brats_cache_root_exists": bool(cache_root is not None and cache_root.is_dir()),
        "cache_root_outside_raw_root": bool(
            cache_root is not None
            and data_root is not None
            and not is_relative_to(cache_root, data_root)
        ),
        "canonical_manifest_exists": plan.canonical_manifest_path.is_file(),
        "train_manifest_exists": (plan.split_dir / "train.csv").is_file(),
        "validation_manifest_exists": (plan.split_dir / "validation.csv").is_file(),
        "test_manifest_not_referenced_by_plan": True,
        "pilot_budget_within_compute_protocol": (
            plan.maximum_optimizer_steps <= fairness.maximum_optimizer_steps
            and plan.maximum_gpu_hours <= fairness.maximum_gpu_hours_per_run
        ),
        "precision_matches_compute_protocol": (
            plan.mixed_precision == fairness.mixed_precision
        ),
        "scheduler_matches_compute_protocol": (
            plan.scheduler == fairness.scheduler.name
        ),
    }
    return {
        "eligible": all(checks.values()),
        "required_accelerator_backend": fairness.accelerator_backend,
        "required_gpu_model": fairness.gpu_model,
        "visible_accelerators": visible_devices,
        "checks": checks,
        "action_if_ineligible": "Do not start Gate 8 reportable pilot training",
    }
