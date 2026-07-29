"""Convert development-only Gate 8 cache volumes to memory-mapped NPY arrays."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bratsarticle.data.dataset import NormalizedVolumeCache
from bratsarticle.data.discovery import resolve_brats2020_training_root
from bratsarticle.data.preprocessing import CacheConfig, load_preprocessing_config
from bratsarticle.utils.serialization import atomic_write_json


def _row_mapping(row: pd.Series[Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items()}


def main() -> int:
    """Convert only train/validation patients; never open the test manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("splits/provisional"),
    )
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default=Path("configs/data/preprocessing_pilot_cached.yaml"),
    )
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--destination-cache-root", type=Path, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/gate8_cache_conversion.json"),
    )
    arguments = parser.parse_args()

    raw_root = resolve_brats2020_training_root(arguments.dataset_root)
    source_root = arguments.source_cache_root.expanduser().resolve()
    destination_root = arguments.destination_cache_root.expanduser().resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    base = load_preprocessing_config(arguments.preprocessing_config)
    source_config = replace(
        base,
        cache=CacheConfig(
            enabled=True,
            root=source_root,
            memory_subjects=0,
            storage_format="compressed_npz",
        ),
    )
    destination_config = replace(
        base,
        cache=CacheConfig(
            enabled=True,
            root=destination_root,
            memory_subjects=0,
            storage_format="memory_mapped_npy",
        ),
    )
    source_cache = NormalizedVolumeCache(source_root, raw_root, enabled=True)
    destination_cache = NormalizedVolumeCache(
        destination_root,
        raw_root,
        enabled=True,
    )

    split_frames = []
    for split in ("train", "validation"):
        manifest_path = arguments.split_dir / f"{split}.csv"
        split_frame = pd.read_csv(manifest_path)
        split_frame.insert(0, "development_split", split)
        split_frames.append(split_frame)
    development = pd.concat(split_frames, ignore_index=True)
    if not development["subject_id"].is_unique:
        raise RuntimeError("Development manifests contain duplicate subject IDs")

    converted = 0
    reused = 0
    for index, (_, row) in enumerate(development.iterrows(), start=1):
        row_mapping = _row_mapping(row)
        existing = destination_cache.load(row_mapping, destination_config)
        if existing is not None:
            reused += 1
        else:
            source = source_cache.load(row_mapping, source_config)
            if source is None:
                raise FileNotFoundError(
                    "Compressed source cache miss for development subject "
                    f"{row_mapping['subject_id']}"
                )
            destination_cache.store(row_mapping, destination_config, source)
            verified = destination_cache.load(row_mapping, destination_config)
            if verified is None:
                raise RuntimeError(
                    f"Failed to verify converted cache for {row_mapping['subject_id']}"
                )
            if not (
                isinstance(verified.image, np.memmap)
                and isinstance(verified.label, np.memmap)
                and verified.image.shape == source.image.shape
                and verified.label.shape == source.label.shape
            ):
                raise RuntimeError(
                    f"Invalid converted cache for {row_mapping['subject_id']}"
                )
            converted += 1
        print(
            json.dumps(
                {
                    "event": "gate8_cache_conversion_progress",
                    "completed": index,
                    "total": len(development),
                    "subject_id": str(row_mapping["subject_id"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    cache_directories = list(destination_root.glob("*.npycache"))
    complete_directories = [
        path for path in cache_directories if (path / "COMPLETE").is_file()
    ]
    if len(complete_directories) != len(development):
        raise RuntimeError(
            "Converted cache subject count does not match development manifests"
        )
    payload = {
        "status": "complete",
        "test_manifest_accessed": False,
        "source_storage_format": "compressed_npz",
        "destination_storage_format": "memory_mapped_npy",
        "source_cache_root": source_root.as_posix(),
        "destination_cache_root": destination_root.as_posix(),
        "development_subject_count": len(development),
        "train_subject_count": int(
            (development["development_split"] == "train").sum()
        ),
        "validation_subject_count": int(
            (development["development_split"] == "validation").sum()
        ),
        "converted_subject_count": converted,
        "reused_subject_count": reused,
        "complete_cache_directory_count": len(complete_directories),
    }
    atomic_write_json(arguments.report, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
