"""Submission-template audits that do not require result generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return cast(dict[str, Any], loaded)


def audit_claim_2024_template(
    template_path: Path,
    *,
    config_path: Path = Path("configs/q1q2_v2/claim_2024_execution.yaml"),
) -> dict[str, Any]:
    """Verify complete and uniquely mapped CLAIM 2024 checklist coverage."""
    config = _load_yaml(config_path)
    expected = [str(value) for value in cast(list[str], config["expected_item_ids"])]
    source = template_path.read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| (C\d{2}) \| .*? \| "
        r"(yes_pending_final_page_lines|partial_pending_final_results|no|"
        r"not_applicable) \|",
        source,
        flags=re.MULTILINE,
    )
    observed = [item_id for item_id, _ in rows]
    statuses = [status for _, status in rows]
    if observed != expected:
        raise RuntimeError(
            "CLAIM 2024 checklist item order or coverage differs: "
            f"expected={expected}, observed={observed}"
        )
    if len(observed) != len(set(observed)):
        raise RuntimeError("CLAIM 2024 checklist contains duplicate items")
    allowed = set(cast(list[str], config["allowed_statuses"]))
    if not set(statuses).issubset(allowed):
        raise RuntimeError("CLAIM 2024 checklist contains an invalid status")
    missing_evidence = [
        raw_path
        for raw_path in cast(list[str], config["required_evidence"])
        if not Path(raw_path).exists()
    ]
    if missing_evidence:
        raise RuntimeError(f"CLAIM 2024 evidence is missing: {missing_evidence}")
    return {
        "valid": True,
        "item_count": len(observed),
        "yes_count": statuses.count("yes_pending_final_page_lines"),
        "partial_count": statuses.count("partial_pending_final_results"),
        "no_count": statuses.count("no"),
        "not_applicable_count": statuses.count("not_applicable"),
        "all_required_evidence_exists": True,
        "final_layout_pending": "[FINAL PAGE/LINES" in source,
    }


def audit_radiology_ai_contract(
    config_path: Path = Path(
        "submission/radiology_ai_original_research_contract.yaml"
    ),
) -> dict[str, Any]:
    """Verify that the journal-format contract is complete and internally valid."""
    config = _load_yaml(config_path)
    if config.get("source") != "https://pubs.rsna.org/page/ai/author-instructions":
        raise RuntimeError("Radiology: AI contract must cite the official instructions")
    limits = cast(dict[str, int], config["limits"])
    required_limit_keys = {
        "main_text_words_introduction_through_discussion",
        "structured_abstract_words",
        "references",
        "figures",
        "tables",
    }
    invalid_limit = any(value <= 0 for value in limits.values())
    if set(limits) != required_limit_keys or invalid_limit:
        raise RuntimeError("Radiology: AI contract limits are incomplete or invalid")
    required = cast(dict[str, Any], config["required"])
    boolean_requirements = {
        "claim_checklist",
        "cover_letter",
        "full_title_page",
        "anonymized_manuscript",
        "data_sharing_statement",
    }
    if any(required.get(key) is not True for key in boolean_requirements):
        raise RuntimeError("Radiology: AI required submission elements are incomplete")
    final_rechecks = cast(list[str], config["final_recheck_required"])
    if not final_rechecks:
        raise RuntimeError("Radiology: AI final policy recheck cannot be empty")
    return {
        "valid": True,
        "verified_on": str(config["verified_on"]),
        "article_type": str(config["article_type"]),
        "limits": limits,
        "final_recheck_count": len(final_rechecks),
    }


__all__ = ["audit_claim_2024_template", "audit_radiology_ai_contract"]
