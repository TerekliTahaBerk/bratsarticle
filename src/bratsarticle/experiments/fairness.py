"""Typed, validated fairness contracts for model training comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from omegaconf import DictConfig, OmegaConf


@dataclass(frozen=True)
class SchedulerProtocol:
    """Single warm-up plus cosine-decay scheduler declaration."""

    name: str
    warmup_optimizer_steps: int
    minimum_learning_rate_fraction: float
    decay_until_optimizer_step: int | None = None

    def __post_init__(self) -> None:
        """Validate scheduler bounds and disallow ambiguous stacking."""
        if self.name != "linear_warmup_cosine_decay":
            raise ValueError("The frozen protocol permits one integrated scheduler")
        if self.warmup_optimizer_steps < 0:
            raise ValueError("Warm-up steps cannot be negative")
        if not 0.0 <= self.minimum_learning_rate_fraction <= 1.0:
            raise ValueError("Minimum learning-rate fraction must be in [0, 1]")
        if (
            self.decay_until_optimizer_step is not None
            and self.decay_until_optimizer_step <= self.warmup_optimizer_steps
        ):
            raise ValueError("Cosine decay must end after warm-up")


@dataclass(frozen=True)
class ComputeMatchedProtocol:
    """Fixed compute budget shared by each compared training run."""

    name: str
    status: str
    gpu_model: str
    identical_gpu_required: bool
    maximum_gpu_hours_per_run: float
    maximum_optimizer_steps: int
    maximum_tuning_trials_per_family: int
    maximum_tuning_gpu_hours_per_family: float
    mixed_precision: bool
    autocast_dtype: str
    input_shape: tuple[int, int, int]
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    scheduler: SchedulerProtocol
    pretraining_status: str
    pretraining_source: str | None
    fine_tuning_gpu_hours: float
    external_pretraining_cost: Literal["included", "excluded"]
    stop_when_first_budget_is_reached: bool
    epoch_count_is_fairness_criterion: bool
    test_subset_permitted: bool

    def __post_init__(self) -> None:
        """Validate the compute-matched comparison contract."""
        if not self.gpu_model.strip():
            raise ValueError("An exact target GPU model is required")
        if self.maximum_gpu_hours_per_run <= 0 or self.maximum_optimizer_steps < 1:
            raise ValueError("Compute budgets must be positive")
        if (
            self.maximum_tuning_trials_per_family < 1
            or self.maximum_tuning_gpu_hours_per_family <= 0
        ):
            raise ValueError("Tuning budgets must be positive")
        if len(self.input_shape) != 3 or any(value < 1 for value in self.input_shape):
            raise ValueError("Input shape must be positive [C,H,W]")
        if self.batch_size < 1 or self.gradient_accumulation_steps < 1:
            raise ValueError("Batch and accumulation must be positive")
        if (
            self.effective_batch_size
            != self.batch_size * self.gradient_accumulation_steps
        ):
            raise ValueError("Effective batch size is inconsistent")
        if self.fine_tuning_gpu_hours < 0:
            raise ValueError("Fine-tuning time cannot be negative")
        if not self.stop_when_first_budget_is_reached:
            raise ValueError("Compute runs must stop at the first exhausted budget")
        if self.epoch_count_is_fairness_criterion:
            raise ValueError("Epoch count is not an accepted fairness criterion")
        if self.test_subset_permitted:
            raise ValueError("The internal test subset is forbidden during tuning")


@dataclass(frozen=True)
class ConvergenceMatchedProtocol:
    """Model-appropriate training under one predeclared stopping rule."""

    name: str
    status: str
    gpu_model: str
    identical_gpu_required: bool
    maximum_optimizer_steps: int
    validation_frequency_optimizer_steps: int
    early_stopping_patience_validation_checks: int
    minimum_improvement: float
    monitored_metric: str
    mode: Literal["maximize", "minimize"]
    mixed_precision: bool
    autocast_dtype: str
    input_shape: tuple[int, int, int]
    batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    scheduler: SchedulerProtocol
    best_checkpoint_rule: str
    best_checkpoint_tie_breaker: str
    convergence_rule: str
    rolling_validation_checks: int
    epoch_count_is_fairness_criterion: bool
    scheduler_count: int
    test_subset_permitted: bool

    def __post_init__(self) -> None:
        """Validate the convergence-matched comparison contract."""
        if not self.gpu_model.strip():
            raise ValueError("An exact target GPU model is required")
        if self.maximum_optimizer_steps < 1:
            raise ValueError("Maximum optimizer steps must be positive")
        if self.validation_frequency_optimizer_steps < 1:
            raise ValueError("Validation frequency must be positive")
        if self.early_stopping_patience_validation_checks < 1:
            raise ValueError("Early-stopping patience must be positive")
        if self.minimum_improvement < 0:
            raise ValueError("Minimum improvement cannot be negative")
        if not self.monitored_metric:
            raise ValueError("A monitored validation metric is required")
        if len(self.input_shape) != 3 or any(value < 1 for value in self.input_shape):
            raise ValueError("Input shape must be positive [C,H,W]")
        if (
            self.effective_batch_size
            != self.batch_size * self.gradient_accumulation_steps
        ):
            raise ValueError("Effective batch size is inconsistent")
        if self.rolling_validation_checks != (
            self.early_stopping_patience_validation_checks
        ):
            raise ValueError("Convergence window must equal stopping patience")
        if self.epoch_count_is_fairness_criterion:
            raise ValueError("Epoch count is not an accepted fairness criterion")
        if self.scheduler_count != 1:
            raise ValueError("Exactly one scheduler is permitted")
        if self.test_subset_permitted:
            raise ValueError("The internal test subset is forbidden during tuning")


def _scheduler(config: DictConfig) -> SchedulerProtocol:
    decay = config.get("decay_until_optimizer_step")
    return SchedulerProtocol(
        name=str(config.name),
        warmup_optimizer_steps=int(config.warmup_optimizer_steps),
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        decay_until_optimizer_step=None if decay is None else int(decay),
    )


def _input_shape(config: DictConfig) -> tuple[int, int, int]:
    values = [int(value) for value in config.input_shape_channels_height_width]
    if len(values) != 3:
        raise ValueError("Input shape must contain exactly C, H, and W")
    return values[0], values[1], values[2]


def load_compute_matched_protocol(path: Path) -> ComputeMatchedProtocol:
    """Load and validate the fixed-compute protocol."""
    root = cast(DictConfig, OmegaConf.load(path))
    OmegaConf.resolve(root)
    protocol = root.protocol
    if str(protocol.regime) != "compute_matched":
        raise ValueError("Expected a compute_matched protocol")
    return ComputeMatchedProtocol(
        name=str(protocol.name),
        status=str(protocol.status),
        gpu_model=str(protocol.hardware.gpu_model),
        identical_gpu_required=bool(protocol.hardware.identical_gpu_required),
        maximum_gpu_hours_per_run=float(protocol.hardware.maximum_gpu_hours_per_run),
        maximum_optimizer_steps=int(protocol.budget.maximum_optimizer_steps),
        maximum_tuning_trials_per_family=int(
            protocol.budget.tuning.maximum_trials_per_model_loss_family
        ),
        maximum_tuning_gpu_hours_per_family=float(
            protocol.budget.tuning.maximum_gpu_hours_per_model_loss_family
        ),
        mixed_precision=bool(protocol.training.mixed_precision),
        autocast_dtype=str(protocol.training.autocast_dtype),
        input_shape=_input_shape(protocol.training),
        batch_size=int(protocol.training.batch_size),
        gradient_accumulation_steps=int(protocol.training.gradient_accumulation_steps),
        effective_batch_size=int(protocol.training.effective_batch_size),
        scheduler=_scheduler(protocol.training.scheduler),
        pretraining_status=str(protocol.pretraining.status),
        pretraining_source=(
            None
            if protocol.pretraining.source is None
            else str(protocol.pretraining.source)
        ),
        fine_tuning_gpu_hours=float(protocol.pretraining.fine_tuning_gpu_hours),
        external_pretraining_cost=cast(
            Literal["included", "excluded"],
            str(protocol.pretraining.external_pretraining_cost),
        ),
        stop_when_first_budget_is_reached=bool(
            protocol.fairness.stop_when_first_budget_is_reached
        ),
        epoch_count_is_fairness_criterion=bool(
            protocol.fairness.epoch_count_is_fairness_criterion
        ),
        test_subset_permitted=bool(protocol.fairness.test_subset_permitted),
    )


def load_convergence_matched_protocol(path: Path) -> ConvergenceMatchedProtocol:
    """Load and validate the convergence-controlled protocol."""
    root = cast(DictConfig, OmegaConf.load(path))
    OmegaConf.resolve(root)
    protocol = root.protocol
    if str(protocol.regime) != "convergence_matched":
        raise ValueError("Expected a convergence_matched protocol")
    return ConvergenceMatchedProtocol(
        name=str(protocol.name),
        status=str(protocol.status),
        gpu_model=str(protocol.hardware.gpu_model),
        identical_gpu_required=bool(protocol.hardware.identical_gpu_required),
        maximum_optimizer_steps=int(protocol.stopping.maximum_optimizer_steps),
        validation_frequency_optimizer_steps=int(
            protocol.stopping.validation_frequency_optimizer_steps
        ),
        early_stopping_patience_validation_checks=int(
            protocol.stopping.early_stopping_patience_validation_checks
        ),
        minimum_improvement=float(protocol.stopping.minimum_improvement),
        monitored_metric=str(protocol.stopping.monitored_metric),
        mode=cast(Literal["maximize", "minimize"], str(protocol.stopping.mode)),
        mixed_precision=bool(protocol.training.mixed_precision),
        autocast_dtype=str(protocol.training.autocast_dtype),
        input_shape=_input_shape(protocol.training),
        batch_size=int(protocol.training.batch_size),
        gradient_accumulation_steps=int(protocol.training.gradient_accumulation_steps),
        effective_batch_size=int(protocol.training.effective_batch_size),
        scheduler=_scheduler(protocol.training.scheduler),
        best_checkpoint_rule=str(protocol.checkpoint.rule),
        best_checkpoint_tie_breaker=str(protocol.checkpoint.tie_breaker),
        convergence_rule=str(protocol.convergence_diagnostic.rule),
        rolling_validation_checks=int(
            protocol.convergence_diagnostic.rolling_validation_checks
        ),
        epoch_count_is_fairness_criterion=bool(
            protocol.fairness.epoch_count_is_fairness_criterion
        ),
        scheduler_count=int(protocol.fairness.scheduler_count),
        test_subset_permitted=bool(protocol.fairness.test_subset_permitted),
    )
