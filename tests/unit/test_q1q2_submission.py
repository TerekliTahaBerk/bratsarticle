from pathlib import Path

from bratsarticle.reporting.q1q2_submission import (
    audit_claim_2024_template,
    audit_radiology_ai_contract,
)


def test_claim_2024_template_covers_all_items_once() -> None:
    report = audit_claim_2024_template(
        Path("submission/q1q2_v2_claim_2024_checklist.template.md")
    )

    assert report["valid"] is True
    assert report["item_count"] == 44
    assert report["final_layout_pending"] is True


def test_radiology_ai_contract_has_verified_original_research_limits() -> None:
    report = audit_radiology_ai_contract()

    assert report["valid"] is True
    assert report["article_type"] == "Original Research"
    assert report["limits"]["main_text_words_introduction_through_discussion"] == 3000
    assert report["limits"]["structured_abstract_words"] == 250
    assert report["limits"]["references"] == 35
    assert report["limits"]["figures"] == 6
    assert report["limits"]["tables"] == 4
