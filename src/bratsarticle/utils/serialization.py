"""Deterministic and atomic serialization helpers."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    return Path(raw_path)


def atomic_write_json(destination: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write sorted, indented JSON."""
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(destination: Path, text: str) -> None:
    """Atomically write UTF-8 text with exactly one trailing newline."""
    temporary = _temporary_path(destination)
    try:
        temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_csv(
    destination: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Atomically write deterministic CSV rows."""
    temporary = _temporary_path(destination)
    try:
        frame = pd.DataFrame(rows)
        frame.to_csv(temporary, index=False, lineterminator="\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
