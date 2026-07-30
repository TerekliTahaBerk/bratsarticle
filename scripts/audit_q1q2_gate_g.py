#!/usr/bin/env python3
"""Audit the Q1/Q2 Gate G checkpoint and analysis-freeze prerequisites."""

from __future__ import annotations

import json

from bratsarticle.experiments.q1q2_gate_g import audit_gate_g


def main() -> None:
    report = audit_gate_g()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
