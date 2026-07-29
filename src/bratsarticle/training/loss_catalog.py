"""Config-driven segmentation loss catalog for controlled ablations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import functional as functional

from bratsarticle.training.losses import labels_to_class_indices

LossName = Literal[
    "cross_entropy",
    "binary_cross_entropy",
    "soft_dice",
    "cross_entropy_plus_soft_dice",
    "binary_cross_entropy_plus_soft_dice",
    "focal_tversky",
    "cross_entropy_plus_focal_tversky",
    "binary_cross_entropy_plus_focal_tversky",
]
Reduction = Literal["mean", "sum"]

LOSS_FORMULAS: dict[str, str] = {
    "cross_entropy": "-sum_c w_c y_c log softmax(z)_c",
    "binary_cross_entropy": (
        "-sum_c w_c [y_c log sigmoid(z_c) + (1-y_c) log(1-sigmoid(z_c))]"
    ),
    "soft_dice": "1 - mean_c [(2 sum p_c y_c + s)/(sum p_c + sum y_c + s)]",
    "cross_entropy_plus_soft_dice": "0.5 CE + 0.5 SoftDice",
    "binary_cross_entropy_plus_soft_dice": "0.5 BCE + 0.5 SoftDice",
    "focal_tversky": "mean_c (1 - (TP+s)/(TP+alpha FP+beta FN+s))^gamma",
    "cross_entropy_plus_focal_tversky": "0.5 CE + 0.5 FocalTversky",
    "binary_cross_entropy_plus_focal_tversky": "0.5 BCE + 0.5 FocalTversky",
}


@dataclass(frozen=True)
class LossConfig:
    """All mathematical and reduction settings for one loss."""

    name: LossName
    alpha: float
    beta: float
    gamma: float
    smoothing: float
    class_weights: tuple[float, ...] | None
    include_background: bool
    reduction: Reduction
    expects_logits: bool
    bce_include_background: bool | None = None
    overlap_include_background: bool | None = None

    def __post_init__(self) -> None:
        """Validate loss hyperparameters."""
        if self.name not in LOSS_FORMULAS:
            raise ValueError(f"Unsupported loss: {self.name}")
        if self.alpha < 0 or self.beta < 0 or self.gamma <= 0:
            raise ValueError("Alpha/beta must be nonnegative and gamma positive")
        if self.smoothing <= 0:
            raise ValueError("Loss smoothing must be positive")
        if self.reduction not in {"mean", "sum"}:
            raise ValueError("Loss reduction must be mean or sum")
        if not self.expects_logits:
            raise ValueError("The catalog requires raw logits")
        if self.class_weights is not None and any(
            weight < 0 for weight in self.class_weights
        ):
            raise ValueError("Class weights cannot be negative")


def _one_hot(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = labels_to_class_indices(labels)
    targets = functional.one_hot(indices, num_classes=logits.shape[1])
    targets = targets.permute(0, 3, 1, 2).to(dtype=logits.dtype)
    return indices, targets


def _select_channels(
    tensor: torch.Tensor,
    include_background: bool,
) -> torch.Tensor:
    return tensor if include_background else tensor[:, 1:]


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    return torch.mean(values) if reduction == "mean" else torch.sum(values)


class ConfiguredSegmentationLoss(nn.Module):
    """Evaluate one declared loss without changing evaluator definitions."""

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def _weights(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.config.class_weights is None:
            return None
        if len(self.config.class_weights) != logits.shape[1]:
            raise ValueError("Class-weight count must equal output channels")
        return torch.as_tensor(
            self.config.class_weights,
            dtype=logits.dtype,
            device=logits.device,
        )

    @property
    def _bce_include_background(self) -> bool:
        return (
            self.config.include_background
            if self.config.bce_include_background is None
            else self.config.bce_include_background
        )

    @property
    def _overlap_include_background(self) -> bool:
        return (
            self.config.include_background
            if self.config.overlap_include_background is None
            else self.config.overlap_include_background
        )

    def _cross_entropy(
        self,
        logits: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        return functional.cross_entropy(
            logits,
            indices,
            weight=self._weights(logits),
            reduction=self.config.reduction,
        )

    def _binary_cross_entropy(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        losses = functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        weights = self._weights(logits)
        if weights is not None:
            losses = losses * weights.view(1, -1, 1, 1)
        losses = _select_channels(losses, self._bce_include_background)
        return _reduce(losses, self.config.reduction)

    def _soft_dice(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = torch.softmax(logits.float(), dim=1)
        probabilities = _select_channels(
            probabilities,
            self._overlap_include_background,
        )
        selected_targets = _select_channels(
            targets.float(),
            self._overlap_include_background,
        )
        axes = (0, 2, 3)
        intersection = torch.sum(probabilities * selected_targets, dim=axes)
        denominator = torch.sum(probabilities + selected_targets, dim=axes)
        dice = (2.0 * intersection + self.config.smoothing) / (
            denominator + self.config.smoothing
        )
        return _reduce(1.0 - dice, self.config.reduction)

    def _focal_tversky(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        probabilities = _select_channels(
            torch.softmax(logits.float(), dim=1),
            self._overlap_include_background,
        )
        selected_targets = _select_channels(
            targets.float(),
            self._overlap_include_background,
        )
        axes = (0, 2, 3)
        true_positive = torch.sum(probabilities * selected_targets, dim=axes)
        false_positive = torch.sum(
            probabilities * (1.0 - selected_targets),
            dim=axes,
        )
        false_negative = torch.sum(
            (1.0 - probabilities) * selected_targets,
            dim=axes,
        )
        tversky = (true_positive + self.config.smoothing) / (
            true_positive
            + self.config.alpha * false_positive
            + self.config.beta * false_negative
            + self.config.smoothing
        )
        return _reduce(
            torch.pow(1.0 - tversky, self.config.gamma),
            self.config.reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the configured loss from raw logits and BraTS labels."""
        indices, targets = _one_hot(logits, labels)
        name = self.config.name
        if name == "cross_entropy":
            return self._cross_entropy(logits, indices)
        if name == "binary_cross_entropy":
            return self._binary_cross_entropy(logits, targets)
        if name == "soft_dice":
            return self._soft_dice(logits, targets)
        if name == "cross_entropy_plus_soft_dice":
            return 0.5 * self._cross_entropy(logits, indices) + 0.5 * self._soft_dice(
                logits, targets
            )
        if name == "binary_cross_entropy_plus_soft_dice":
            return 0.5 * self._binary_cross_entropy(
                logits, targets
            ) + 0.5 * self._soft_dice(logits, targets)
        if name == "focal_tversky":
            return self._focal_tversky(logits, targets)
        if name == "cross_entropy_plus_focal_tversky":
            return 0.5 * self._cross_entropy(
                logits, indices
            ) + 0.5 * self._focal_tversky(logits, targets)
        return 0.5 * self._binary_cross_entropy(
            logits, targets
        ) + 0.5 * self._focal_tversky(logits, targets)


def load_loss_catalog(path: Path) -> list[LossConfig]:
    """Load every declared loss configuration."""
    root = cast(DictConfig, OmegaConf.load(path))
    OmegaConf.resolve(root)
    output: list[LossConfig] = []
    for raw in root.losses:
        class_weights = (
            None
            if raw.class_weights is None
            else tuple(float(value) for value in raw.class_weights)
        )
        output.append(
            LossConfig(
                name=cast(LossName, str(raw.name)),
                alpha=float(raw.alpha),
                beta=float(raw.beta),
                gamma=float(raw.gamma),
                smoothing=float(raw.smoothing),
                class_weights=class_weights,
                include_background=bool(raw.include_background),
                reduction=cast(Reduction, str(raw.reduction)),
                expects_logits=bool(raw.expects_logits),
                bce_include_background=(
                    None
                    if raw.get("bce_include_background") is None
                    else bool(raw.bce_include_background)
                ),
                overlap_include_background=(
                    None
                    if raw.get("overlap_include_background") is None
                    else bool(raw.overlap_include_background)
                ),
            )
        )
    return output


def build_loss(config: LossConfig) -> ConfiguredSegmentationLoss:
    """Construct one configured segmentation loss."""
    return ConfiguredSegmentationLoss(config)
