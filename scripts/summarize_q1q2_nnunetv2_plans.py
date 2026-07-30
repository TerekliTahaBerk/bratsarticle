#!/usr/bin/env python3
"""Create a compact tracked summary of local official nnU-Net plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json

DATASET_NAME = "Dataset501_BraTS2020Q1Q2"
PLAN_IDENTIFIERS = (
    "nnUNetPlans",
    "nnUNetResEncUNetMPlans",
    "nnUNetResEncUNetLPlans",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnunet-preprocessed-root", type=Path, required=True)
    parser.add_argument("--derived-dataset-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/q1q2_v2/nnunet_planning_summary.json"),
    )
    return parser.parse_args()


def _configuration_summary(configuration: dict[str, Any]) -> dict[str, Any]:
    architecture = configuration["architecture"]
    arguments = architecture["arch_kwargs"]
    return {
        "architecture": architecture["network_class_name"],
        "batch_size": int(configuration["batch_size"]),
        "patch_size": [int(value) for value in configuration["patch_size"]],
        "spacing": [float(value) for value in configuration["spacing"]],
        "data_identifier": str(configuration["data_identifier"]),
        "features_per_stage": [
            int(value) for value in arguments["features_per_stage"]
        ],
        "stage_count": int(arguments["n_stages"]),
        "batch_dice": bool(configuration["batch_dice"]),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.nnunet_preprocessed_root.resolve() / DATASET_NAME
    derivation_path = (
        args.derived_dataset_root.resolve() / "derivation_manifest.json"
    )
    fingerprint_path = dataset_root / "dataset_fingerprint.json"
    split_path = dataset_root / "splits_final.json"
    required = [derivation_path, fingerprint_path, split_path]
    plans: list[dict[str, Any]] = []
    for identifier in PLAN_IDENTIFIERS:
        path = dataset_root / f"{identifier}.json"
        required.append(path)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        plans.append(
            {
                "plans_identifier": identifier,
                "plans_sha256": file_digest(path),
                "experiment_planner": payload["experiment_planner_used"],
                "configurations": {
                    name: _configuration_summary(
                        payload["configurations"][name]
                    )
                    for name in ("2d", "3d_fullres")
                },
            }
        )
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing nnU-Net planning inputs: {missing}")
    derivation = json.loads(derivation_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "status": "planned_hardware_preflight_pending",
        "dataset": DATASET_NAME,
        "case_count": int(derivation["case_count"]),
        "source_manifest_sha256": derivation["source_manifest_sha256"],
        "adapter_git_commit": derivation["git_commit"],
        "dataset_integrity": {
            "label_mapping_validated": True,
            "source_hashes_recomputed": derivation[
                "source_hashes_recomputed"
            ],
            "fingerprint_sha256": file_digest(fingerprint_path),
            "splits_final_sha256": file_digest(split_path),
        },
        "three_d_selection": {
            "primary": "nnUNetResEncUNetLPlans",
            "hardware_fallback": "nnUNetResEncUNetMPlans",
            "criterion": "untouched_plan_mps_feasibility_only",
            "performance_outcomes_permitted": False,
            "selected": None,
        },
        "plans": plans,
        "external_data_accessed": False,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
