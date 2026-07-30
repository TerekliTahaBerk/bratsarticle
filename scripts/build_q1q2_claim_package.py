#!/usr/bin/env python3
"""Build the Gate J registry and optionally render/audit a manuscript template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.reporting.q1q2_claims import (
    audit_claim_package,
    build_claim_registry,
    complete_gate_j,
    render_claim_template,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/q1q2_v2/claims/claim_registry.json"),
    )
    parser.add_argument(
        "--rendered",
        type=Path,
        default=Path("artifacts/q1q2_v2/claims/rendered_manuscript.md"),
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=Path("artifacts/q1q2_v2/claims/render_trace.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.audit_only:
        report = audit_claim_package(
            registry_path=arguments.registry,
            rendered_path=arguments.rendered,
            trace_path=arguments.trace,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] else 1
    registry = build_claim_registry()
    report: dict[str, object] = {
        "registry_status": registry["status"],
        "claim_count": registry["claim_count"],
    }
    if arguments.template is not None:
        report["render"] = render_claim_template(
            template_path=arguments.template,
            registry_path=arguments.registry,
            output_path=arguments.rendered,
            trace_path=arguments.trace,
        )
        report["completion"] = complete_gate_j()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
