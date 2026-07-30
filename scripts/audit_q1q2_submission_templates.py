#!/usr/bin/env python3
"""Audit result-independent Q1/Q2 submission-template coverage."""

from __future__ import annotations

import json
from pathlib import Path

from bratsarticle.reporting.q1q2_submission import (
    audit_claim_2024_template,
    audit_radiology_ai_contract,
)


def main() -> int:
    """Run the CLAIM 2024 template audit."""
    report = {
        "claim_2024": audit_claim_2024_template(
            Path("submission/q1q2_v2_claim_2024_checklist.template.md")
        ),
        "radiology_ai": audit_radiology_ai_contract(),
    }
    report["valid"] = all(
        section["valid"]
        for section in report.values()
        if isinstance(section, dict)
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
