"""Accelerator discovery shared by protocol, registry, and pilot guards."""

from __future__ import annotations

import json
import subprocess
from typing import Literal

import torch

AcceleratorBackend = Literal["cuda", "mps"]


def accelerator_available(backend: AcceleratorBackend) -> bool:
    """Return whether the requested GPU backend is usable."""
    if backend == "cuda":
        return torch.cuda.is_available()
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _mps_device_names() -> list[str]:
    try:
        output = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return []
    entries = payload.get("SPDisplaysDataType", [])
    names = [
        str(entry.get("sppci_model", "")).strip()
        for entry in entries
        if str(entry.get("sppci_model", "")).strip().startswith("Apple ")
    ]
    return names


def accelerator_device_names(backend: AcceleratorBackend) -> list[str]:
    """Return visible device names for CUDA or Apple Metal."""
    if backend == "cuda":
        return [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    return _mps_device_names()


def accelerator_device(backend: AcceleratorBackend) -> torch.device:
    """Return an available accelerator device or raise."""
    if not accelerator_available(backend):
        raise RuntimeError(f"Requested accelerator is unavailable: {backend}")
    return torch.device(backend)
