"""Single integrated warm-up plus cosine-decay learning-rate schedule."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine_factor(
    optimizer_step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_fraction: float,
) -> float:
    """Return the deterministic LR multiplier for one optimizer step."""
    if optimizer_step < 0:
        raise ValueError("optimizer_step cannot be negative")
    if warmup_steps < 0 or total_steps <= warmup_steps:
        raise ValueError("total_steps must be greater than nonnegative warmup_steps")
    if not 0.0 <= minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in [0, 1]")
    if optimizer_step < warmup_steps:
        return (optimizer_step + 1) / max(1, warmup_steps)
    progress = min(
        1.0,
        (optimizer_step - warmup_steps) / (total_steps - warmup_steps),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_fraction + (1.0 - minimum_fraction) * cosine


def build_warmup_cosine_scheduler(
    optimizer: Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_fraction: float,
) -> LambdaLR:
    """Build the only scheduler permitted by the frozen fairness protocols."""

    def multiplier(step: int) -> float:
        return warmup_cosine_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            minimum_fraction=minimum_fraction,
        )

    return LambdaLR(optimizer, lr_lambda=multiplier)
