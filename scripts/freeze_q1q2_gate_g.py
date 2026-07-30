#!/usr/bin/env python3
"""Freeze all Q1/Q2 checkpoints and analyses before external inference."""

from __future__ import annotations

import argparse
import json

from bratsarticle.experiments.q1q2_gate_g import freeze_gate_g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-analysis-freeze", action="store_true")
    arguments = parser.parse_args()
    report = freeze_gate_g(allow_analysis_freeze=arguments.allow_analysis_freeze)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
