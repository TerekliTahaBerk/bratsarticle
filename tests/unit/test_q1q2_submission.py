from pathlib import Path

from bratsarticle.reporting.q1q2_submission import audit_claim_2024_template


def test_claim_2024_template_covers_all_items_once() -> None:
    report = audit_claim_2024_template(
        Path("submission/q1q2_v2_claim_2024_checklist.template.md")
    )

    assert report["valid"] is True
    assert report["item_count"] == 44
    assert report["final_layout_pending"] is True
