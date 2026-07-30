#!/usr/bin/env python3
"""Generate frozen, artifact-derived Q1/Q2 figures."""

from __future__ import annotations

import argparse
import json

from bratsarticle.analysis.q1q2_figures import (
    build_q1q2_result_figures,
    build_study_design_figure,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-only", action="store_true")
    arguments = parser.parse_args()
    report = (
        {"status": "complete", "figures": build_study_design_figure()}
        if arguments.design_only
        else build_q1q2_result_figures()
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
