#!/usr/bin/env python3
"""Freeze the predeclared nnU-Net 3D plan using MPS feasibility only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from bratsarticle.adapters.nnunetv2 import DATASET_ID, build_main_job_matrix
from bratsarticle.experiments.q1q2_nnunet_plan_selection import (
    select_nnunet_3d_plan,
    write_nnunet_3d_plan_selection,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import (
    atomic_write_json,
    atomic_write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-preflight", type=Path, required=True)
    parser.add_argument("--fallback-preflight", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/q1q2_v2/selected_nnunet_3d_plan.yaml"),
    )
    parser.add_argument(
        "--queue-output",
        type=Path,
        default=Path("artifacts/q1q2_v2/queues/nnunetv2_main.json"),
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.output.exists():
        raise FileExistsError("nnU-Net 3D plan selection is already frozen")
    select_nnunet_3d_plan(
        primary_preflight_path=arguments.primary_preflight,
        fallback_preflight_path=arguments.fallback_preflight,
    )
    report_root = Path("reports/q1q2_v2")
    report_root.mkdir(parents=True, exist_ok=True)
    primary_copy = report_root / "nnunet_mps_preflight_resenc_l.json"
    if primary_copy.exists():
        raise FileExistsError("Tracked ResEnc-L preflight already exists")
    fallback_copy: Path | None = None
    if arguments.fallback_preflight is not None:
        fallback_copy = report_root / "nnunet_mps_preflight_resenc_m.json"
        if fallback_copy.exists():
            raise FileExistsError("Tracked ResEnc-M preflight already exists")
    shutil.copyfile(arguments.primary_preflight, primary_copy)
    if fallback_copy is not None and arguments.fallback_preflight is not None:
        shutil.copyfile(arguments.fallback_preflight, fallback_copy)
    selection = select_nnunet_3d_plan(
        primary_preflight_path=primary_copy,
        fallback_preflight_path=fallback_copy,
    )
    write_nnunet_3d_plan_selection(selection, arguments.output)
    jobs = build_main_job_matrix(selection["selected_plans_identifier"])
    payload = {
        "schema_version": 1,
        "status": "frozen_not_started",
        "dataset_id": DATASET_ID,
        "selected_3d_plan_config": arguments.output.as_posix(),
        "selected_3d_plan_config_sha256": file_digest(arguments.output),
        "external_data_permitted": False,
        "legacy_internal_test_permitted": False,
        "job_count": len(jobs),
        "job_count_by_configuration": {"2d": 25, "3d_fullres": 25},
        "jobs": [{**job, "status": "not_started"} for job in jobs],
    }
    atomic_write_json(arguments.queue_output, payload)
    runner_path = Path("configs/q1q2_v2/nnunet_m1_runner.yaml")
    runner = yaml.safe_load(runner_path.read_text(encoding="utf-8"))
    if runner.get("status") != "blocked_until_hardware_plan_freeze":
        raise RuntimeError("nnU-Net runner is not awaiting its first plan freeze")
    runner["status"] = "frozen_before_first_reportable_development_run"
    runner["matrix"]["selected_3d_plan_sha256"] = file_digest(arguments.output)
    runner["matrix"]["queue_sha256"] = file_digest(arguments.queue_output)
    atomic_write_text(runner_path, yaml.safe_dump(runner, sort_keys=False))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
