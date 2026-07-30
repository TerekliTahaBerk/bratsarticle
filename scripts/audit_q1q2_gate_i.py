#!/usr/bin/env python3
"""Build or verify the Q1/Q2 Gate I artifact manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.reporting.q1q2_reproducibility import (
    build_gate_i_manifest,
    run_gate_i_clean_clone_audit,
    verify_gate_i_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--verify", action="store_true")
    action.add_argument("--clean-clone", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/q1q2_v2/reproducibility/artifact_manifest.json"),
    )
    arguments = parser.parse_args()
    if arguments.build:
        report = build_gate_i_manifest()
    elif arguments.clean_clone:
        report = run_gate_i_clean_clone_audit(
            output_path=Path("artifacts/q1q2_v2/reproducibility/clean_clone_audit.json")
        )
    else:
        report = verify_gate_i_manifest(arguments.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    if arguments.verify and not report["valid"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
