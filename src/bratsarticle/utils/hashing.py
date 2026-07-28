"""Streaming cryptographic hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_digest(
    path: Path,
    algorithm: str = "sha256",
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    """Return a streaming digest without loading the complete file into memory."""
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def text_digest(text: str, algorithm: str = "sha256") -> str:
    """Return the digest of UTF-8 text."""
    digest = hashlib.new(algorithm)
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()
