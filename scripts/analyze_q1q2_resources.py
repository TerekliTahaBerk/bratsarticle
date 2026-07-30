#!/usr/bin/env python3
"""Run the frozen Q1/Q2 measured resource analysis."""

from __future__ import annotations

import json

from bratsarticle.analysis.q1q2_resources import analyze_q1q2_resources


def main() -> int:
    report = analyze_q1q2_resources()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
