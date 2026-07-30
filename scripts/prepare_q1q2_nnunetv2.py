#!/usr/bin/env python3
"""Prepare the audited BraTS 2020 layout consumed by official nnU-Net v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.adapters.nnunetv2 import prepare_nnunet_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--canonical-manifest",
        type=Path,
        default=Path("manifests/canonical/brats2020_canonical_manifest.csv"),
    )
    parser.add_argument(
        "--fold-directory",
        type=Path,
        default=Path("splits/q1q2_v2"),
    )
    parser.add_argument("--nnunet-raw-root", required=True, type=Path)
    parser.add_argument("--nnunet-preprocessed-root", required=True, type=Path)
    parser.add_argument(
        "--skip-source-rehash",
        action="store_true",
        help=(
            "Trust hashes already frozen in the canonical manifest. "
            "Reportable preparation should not use this option."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = prepare_nnunet_dataset(
        raw_root=args.dataset_root,
        canonical_manifest_path=args.canonical_manifest,
        split_paths=[
            args.fold_directory / f"cv_fold_{fold}.csv"
            for fold in range(1, 6)
        ],
        nnunet_raw_root=args.nnunet_raw_root,
        nnunet_preprocessed_root=args.nnunet_preprocessed_root,
        verify_source_hashes=not args.skip_source_rehash,
    )
    print(
        json.dumps(
            {
                "dataset_directory": str(prepared.dataset_directory),
                "split_file": str(prepared.split_file),
                "source_manifest_sha256": prepared.source_manifest_sha256,
                "split_sha256": prepared.split_sha256,
                "case_count": prepared.case_count,
                "reused_existing_dataset": prepared.reused_existing_dataset,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
