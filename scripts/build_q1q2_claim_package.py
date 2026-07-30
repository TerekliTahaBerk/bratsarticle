#!/usr/bin/env python3
"""Build the Gate J registry and optionally render/audit a manuscript template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.reporting.q1q2_claims import (
    audit_claim_package,
    audit_reviewer_response_template,
    build_claim_registry,
    complete_gate_j,
    render_claim_template,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path)
    parser.add_argument("--reviewer-response-template", type=Path)
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
    parser.add_argument(
        "--rendered-reviewer-response",
        type=Path,
        default=Path(
            "artifacts/q1q2_v2/claims/rendered_response_to_reviewer.md"
        ),
    )
    parser.add_argument(
        "--reviewer-response-trace",
        type=Path,
        default=Path("artifacts/q1q2_v2/claims/reviewer_response_trace.json"),
    )
    parser.add_argument("--audit-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.audit_only:
        manuscript_report = audit_claim_package(
            registry_path=arguments.registry,
            rendered_path=arguments.rendered,
            trace_path=arguments.trace,
        )
        response_report = audit_claim_package(
            registry_path=arguments.registry,
            rendered_path=arguments.rendered_reviewer_response,
            trace_path=arguments.reviewer_response_trace,
        )
        report = {
            "valid": manuscript_report["valid"] and response_report["valid"],
            "manuscript": manuscript_report,
            "reviewer_response": response_report,
        }
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
    if arguments.reviewer_response_template is not None:
        report["reviewer_response_template_audit"] = (
            audit_reviewer_response_template(arguments.reviewer_response_template)
        )
        report["reviewer_response_render"] = render_claim_template(
            template_path=arguments.reviewer_response_template,
            registry_path=arguments.registry,
            output_path=arguments.rendered_reviewer_response,
            trace_path=arguments.reviewer_response_trace,
        )
    if (
        arguments.template is not None
        and arguments.reviewer_response_template is not None
    ):
        report["completion"] = complete_gate_j()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
