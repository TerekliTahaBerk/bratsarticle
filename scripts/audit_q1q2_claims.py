#!/usr/bin/env python3
"""Audit risky v2 wording against the machine-readable claim ledger."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from bratsarticle.utils.serialization import atomic_write_json

KEYWORDS = (
    "superior",
    "efficient",
    "lightweight",
    "robust",
    "generalizable",
    "clinical",
    "state of the art",
    "novel",
    "significant",
    "reproducible",
    "leakage-safe",
    "q1/q2-ready",
)
NEGATION_OR_LIMITATION = re.compile(
    r"\b(no|not|cannot|unknown|pending|blocked|prohibited|without|never|false|"
    r"unreleased|incomplete|unsupported|does not|must not)\b",
    re.IGNORECASE,
)


def _ledger_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit_claims(ledger_path: Path, output_path: Path) -> dict[str, Any]:
    """Fail only on untracked affirmative risky performance language."""
    rows = _ledger_rows(ledger_path)
    unsupported = [row for row in rows if row["status"] == "unsupported"]
    if not unsupported:
        raise RuntimeError("The ledger must retain explicit unsupported claims")
    roots = [
        Path("reports/q1q2_v2"),
        Path("submission"),
    ]
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.suffix.lower() in {".md", ".txt"}
    )
    occurrences: list[dict[str, Any]] = []
    untracked_affirmative: list[dict[str, Any]] = []
    technical_whitelist = (
        "robust normalized",
        "robust signature",
        "clinical mcid",
        "clinical cutoff",
        "clinical utility",
        "not a clinical",
        "does not authorize",
        "probability of superiority",
    )
    for path in files:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            lowered = line.lower()
            for keyword in KEYWORDS:
                if keyword not in lowered:
                    continue
                classification = "limitation_or_negated"
                if any(phrase in lowered for phrase in technical_whitelist):
                    classification = "technical_or_limitation_context"
                elif not NEGATION_OR_LIMITATION.search(line):
                    classification = "affirmative_manual_review"
                    untracked_affirmative.append(
                        {
                            "path": path.as_posix(),
                            "line": line_number,
                            "keyword": keyword,
                            "text": line.strip(),
                        }
                    )
                occurrences.append(
                    {
                        "path": path.as_posix(),
                        "line": line_number,
                        "keyword": keyword,
                        "classification": classification,
                        "text": line.strip(),
                    }
                )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "ledger_claim_count": len(rows),
        "supported_count": sum(row["status"] == "supported" for row in rows),
        "partially_supported_count": sum(
            row["status"] == "partially supported" for row in rows
        ),
        "unsupported_count": len(unsupported),
        "risky_word_occurrences": occurrences,
        "untracked_affirmative_high_risk_count": len(untracked_affirmative),
        "untracked_affirmative_high_risk": untracked_affirmative,
        "pass": not untracked_affirmative,
        "interpretation": (
            "Keyword review prevents unsupported affirmative high-risk wording; "
            "it is not an authorship, originality, or scientific-validity detector."
        ),
    }
    atomic_write_json(output_path, payload)
    if untracked_affirmative:
        raise RuntimeError(
            "Untracked affirmative high-risk claims were found: "
            f"{len(untracked_affirmative)}"
        )
    return payload


def main() -> None:
    audit_claims(
        Path("claims/q1q2_v2_claim_ledger.csv"),
        Path("reports/q1q2_v2/claim_audit.json"),
    )


if __name__ == "__main__":
    main()
