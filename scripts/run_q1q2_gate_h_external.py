#!/usr/bin/env python3
"""Run the single frozen Gate H external confirmatory session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_external_queue import (
    run_gate_h_external_queue,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", required=True, type=Path)
    parser.add_argument(
        "--allow-frozen-external-inference",
        action="store_true",
    )
    arguments = parser.parse_args()
    report = run_gate_h_external_queue(
        external_root=arguments.external_root,
        allow_frozen_external_inference=(arguments.allow_frozen_external_inference),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
