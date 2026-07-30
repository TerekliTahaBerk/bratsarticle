"""Atomic, RNG-complete training checkpoint save and resume."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@dataclass
class TrainingState:
    """Serializable training progress counters."""

    epoch: int = 0
    global_step: int = 0
    batches_consumed_in_epoch: int = 0


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    destination: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LRScheduler | None = None,
    state: TrainingState,
    metadata: dict[str, Any],
) -> None:
    """Atomically save model, optimizer, scaler, counters, metadata, and RNG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(raw_temporary)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "state": asdict(state),
        "metadata": metadata,
        "rng": _rng_state(),
    }
    try:
        torch.save(payload, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    source: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: LRScheduler | None = None,
    map_location: torch.device,
) -> tuple[TrainingState, dict[str, Any]]:
    """Restore all training and random state required for deterministic resume."""
    payload = cast(
        dict[str, Any],
        torch.load(source, map_location=map_location, weights_only=False),
    )
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    scheduler_state = payload.get("scheduler")
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("Checkpoint has no scheduler state")
        scheduler.load_state_dict(scheduler_state)
    elif scheduler_state is not None:
        raise ValueError("Checkpoint contains scheduler state but none was supplied")
    state = TrainingState(**payload["state"])
    _restore_rng_state(payload["rng"])
    return state, cast(dict[str, Any], payload["metadata"])
