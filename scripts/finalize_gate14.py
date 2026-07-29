#!/usr/bin/env python3
"""Validate and hash the final Gate 14 manuscript package."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = [
    "manuscript/final_manuscript.md",
    "manuscript/final_manuscript.docx",
    "manuscript/final_manuscript.tex",
    "manuscript/final_manuscript.pdf",
    "manuscript/response_to_reviewer.md",
    "manuscript/response_to_reviewer.docx",
    "manuscript/response_to_reviewer.tex",
    "manuscript/response_to_reviewer.pdf",
    "manuscript/claim_2024_checklist.md",
    "literature/verified_sources.yaml",
    "reports/gate14_generation_manifest.json",
    "reports/gate14_originality_audit.json",
    "reports/gate14_originality_audit.md",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pdf_pages(path: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"^Pages:\s+(\d+)$", result.stdout, flags=re.MULTILINE)
    if match is None:
        raise RuntimeError(f"Could not read PDF page count: {path}")
    return int(match.group(1))


def _pdf_text(path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _count_test_access_events() -> int:
    path = ROOT / "artifacts/test_access_log.jsonl"
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytest-summary", default="not recorded")
    parser.add_argument("--quality-summary", default="not recorded")
    args = parser.parse_args()

    paths = [ROOT / relative_path for relative_path in DELIVERABLES]
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in paths
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"Missing Gate 14 deliverables: {missing}")

    manuscript = (ROOT / "manuscript/final_manuscript.md").read_text(encoding="utf-8")
    originality = json.loads(
        (ROOT / "reports/gate14_originality_audit.json").read_text(encoding="utf-8")
    )
    word_docx = ROOT / "manuscript/final_manuscript.docx"
    response_docx = ROOT / "manuscript/response_to_reviewer.docx"
    for path in (word_docx, response_docx):
        if not zipfile.is_zipfile(path):
            raise RuntimeError(f"Invalid DOCX package: {path}")
        with zipfile.ZipFile(path) as archive:
            if "word/document.xml" not in archive.namelist():
                raise RuntimeError(f"DOCX has no document body: {path}")

    manuscript_pdf = ROOT / "manuscript/final_manuscript.pdf"
    response_pdf = ROOT / "manuscript/response_to_reviewer.pdf"
    manuscript_pdf_text = _pdf_text(manuscript_pdf)
    checks = {
        "no_unresolved_template_tokens": "{{" not in manuscript,
        "ten_figure_references": manuscript.count("](figures/final/") == 10,
        "eight_manuscript_tables": manuscript.count("**Table ") == 8,
        "fifteen_references": (
            len(
                re.findall(
                    r"^\d+\. ",
                    manuscript.split("## References", 1)[1],
                    re.MULTILINE,
                )
            )
            == 15
        ),
        "originality_audit_pass": originality["status"] == "pass",
        "no_long_phrase_overlap": originality["exact_ngram_match_count"] == 0,
        "no_repeated_full_sentence": originality["repeated_sentences"] == [],
        "one_test_access_event_preserved": _count_test_access_events() == 1,
        "manuscript_docx_valid": zipfile.is_zipfile(word_docx),
        "response_docx_valid": zipfile.is_zipfile(response_docx),
        "manuscript_pdf_valid": manuscript_pdf.read_bytes().startswith(b"%PDF-"),
        "response_pdf_valid": response_pdf.read_bytes().startswith(b"%PDF-"),
        "latex_figure_captions_not_duplicated": (
            "Figure 1: Figure 1" not in manuscript_pdf_text
        ),
        "author_confirmation_boundaries_present": (
            "Declarations requiring author confirmation" in manuscript
        ),
    }
    status = "pass" if all(checks.values()) else "fail"
    report = {
        "gate": 14,
        "status": status,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "scientific_values": (
            "Generated from tracked machine-readable artifacts; no manual "
            "result transcription."
        ),
        "checks": checks,
        "manuscript": {
            "word_count": len(re.findall(r"\b[\w'-]+\b", manuscript)),
            "figure_count": manuscript.count("](figures/final/"),
            "table_count": manuscript.count("**Table "),
            "reference_count": 15,
            "pdf_pages": _pdf_pages(manuscript_pdf),
            "docx_visual_qa_pages": 18,
        },
        "reviewer_response": {
            "pdf_pages": _pdf_pages(response_pdf),
            "docx_visual_qa_pages": 4,
        },
        "validation": {
            "pytest": args.pytest_summary,
            "visual_quality": args.quality_summary,
        },
        "deliverables": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in paths
        },
        "scope_boundary": (
            "The study remains an internal, bounded 2D component evaluation. "
            "It does not establish external generalization, clinical utility, "
            "or superiority over untested 3D, transformer, or self-configuring "
            "systems. Author declarations still require confirmation."
        ),
    }
    (ROOT / "reports/gate14_completion.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Gate 14 Manuscript Reconstruction",
        "",
        f"**Decision:** {status.upper()}",
        "",
        "## Package",
        "",
        f"- Manuscript words: {report['manuscript']['word_count']}",
        f"- Manuscript figures: {report['manuscript']['figure_count']}",
        f"- Manuscript tables: {report['manuscript']['table_count']}",
        f"- References: {report['manuscript']['reference_count']}",
        f"- LaTeX PDF pages: {report['manuscript']['pdf_pages']}",
        (f"- Word-render QA pages: {report['manuscript']['docx_visual_qa_pages']}"),
        (f"- Reviewer-response PDF pages: {report['reviewer_response']['pdf_pages']}"),
        "",
        "## Checks",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'}: {name.replace('_', ' ')}"
            for name, value in checks.items()
        ],
        "",
        "## Validation",
        "",
        f"- Pytest: {args.pytest_summary}",
        f"- Visual QA: {args.quality_summary}",
        "",
        "## Scope boundary",
        "",
        report["scope_boundary"],
        "",
    ]
    (ROOT / "reports/gate14_completion.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    if status != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
