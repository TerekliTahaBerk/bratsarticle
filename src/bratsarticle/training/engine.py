"""Minimal deterministic training engine with AMP and exact resume."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext

import torch
from torch import nn
from torch.optim import Optimizer

from bratsarticle.training.checkpoint import TrainingState


def _autocast_context(
    device: torch.device,
    enabled: bool,
) -> AbstractContextManager[None]:
    if not enabled:
        return nullcontext()
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
    return torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=True,
    )


class TrainingEngine:
    """Run deterministic optimization steps without owning data selection."""

    def __init__(
        self,
        *,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        device: torch.device,
        mixed_precision: bool,
        state: TrainingState | None = None,
    ) -> None:
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("Training engine currently supports CPU or CUDA")
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = device
        self.mixed_precision = mixed_precision
        self.state = state or TrainingState()
        self.scaler = torch.amp.GradScaler(
            device.type,
            enabled=mixed_precision and device.type == "cuda",
        )

    def train_step(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
    ) -> float:
        """Run one optimizer step and return the detached scalar loss."""
        self.model.train()
        image = image.to(self.device, dtype=torch.float32)
        label = label.to(self.device, dtype=torch.long)
        self.optimizer.zero_grad(set_to_none=True)
        with _autocast_context(self.device, self.mixed_precision):
            logits = self.model(image)
            loss = self.loss_function(logits, label)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.state.global_step += 1
        return float(loss.detach().cpu())


__all__ = ["TrainingEngine", "TrainingState"]
