"""Training-only segmentation losses for BraTS integer labels."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as functional


def labels_to_class_indices(labels: torch.Tensor) -> torch.Tensor:
    """Map BraTS labels `{0,1,2,4}` to contiguous classes `{0,1,2,3}`."""
    invalid = ~((labels == 0) | (labels == 1) | (labels == 2) | (labels == 4))
    if bool(invalid.any()):
        invalid_values = torch.unique(labels[invalid]).detach().cpu().tolist()
        raise ValueError(f"Unexpected BraTS labels: {invalid_values}")
    return torch.where(labels == 4, torch.full_like(labels, 3), labels).long()


def class_indices_to_labels(indices: torch.Tensor) -> torch.Tensor:
    """Map contiguous four-class predictions back to BraTS labels."""
    if bool(((indices < 0) | (indices > 3)).any()):
        raise ValueError("Class indices must be in [0, 3]")
    return torch.where(indices == 3, torch.full_like(indices, 4), indices).long()


class DiceCrossEntropyLoss(nn.Module):
    """Weighted cross-entropy plus foreground soft Dice loss."""

    def __init__(
        self,
        *,
        cross_entropy_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if cross_entropy_weight + dice_weight <= 0:
            raise ValueError("At least one loss term must be active")
        if smooth <= 0:
            raise ValueError("Training Dice smooth must be positive")
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the combined training loss."""
        targets = labels_to_class_indices(labels)
        if logits.ndim != 4 or targets.ndim != 3:
            raise ValueError("Expected logits [B,4,H,W] and labels [B,H,W]")
        if (
            logits.shape[0] != targets.shape[0]
            or logits.shape[-2:] != targets.shape[-2:]
        ):
            raise ValueError("Logit and label batch/spatial shapes must match")
        cross_entropy = functional.cross_entropy(logits, targets)
        probabilities = torch.softmax(logits.float(), dim=1)
        one_hot = functional.one_hot(targets, num_classes=logits.shape[1])
        one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)
        spatial_axes = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=spatial_axes)
        denominator = torch.sum(probabilities + one_hot, dim=spatial_axes)
        foreground_dice = (2.0 * intersection[1:] + self.smooth) / (
            denominator[1:] + self.smooth
        )
        dice_loss = 1.0 - torch.mean(foreground_dice)
        return self.cross_entropy_weight * cross_entropy + self.dice_weight * dice_loss
