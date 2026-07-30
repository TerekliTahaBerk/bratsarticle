#!/usr/bin/env python3
"""Generate, but do not execute, the frozen nnU-Net v2 fold/seed job matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.adapters.nnunetv2 import (
    DATASET_ID,
    MAIN_SEEDS,
    NNUNET_3D_FALLBACK_PLANS,
    NNUNET_3D_PRIMARY_PLANS,
    build_main_job_matrix,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/q1q2_v2/queues/nnunetv2_main.json"),
    )
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fold_hashes = {
        f"cv_fold_{fold}.csv": file_digest(
            args.fold_directory / f"cv_fold_{fold}.csv"
        )
        for fold in range(1, 6)
    }
    jobs = build_main_job_matrix()
    payload = {
        "schema_version": 1,
        "status": "generated_hardware_preflight_pending",
        "dataset_id": DATASET_ID,
        "canonical_manifest_sha256": file_digest(args.canonical_manifest),
        "fold_sha256": fold_hashes,
        "seed_list": list(MAIN_SEEDS),
        "job_count": len(jobs),
        "job_count_by_configuration": {"2d": 25, "3d_fullres": 25},
        "three_d_plan_selection": {
            "primary": NNUNET_3D_PRIMARY_PLANS,
            "hardware_fallback": NNUNET_3D_FALLBACK_PLANS,
            "selection_criterion": (
                "untouched official-plan one-batch MPS feasibility only; "
                "never validation performance"
            ),
            "selected": None,
        },
        "external_data_permitted": False,
        "jobs": jobs,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
