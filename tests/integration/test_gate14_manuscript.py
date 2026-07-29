import csv
import json
import zipfile
from pathlib import Path


def _rows(path: str) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_gate14_manuscript_is_artifact_derived_and_bounded() -> None:
    manuscript = Path("manuscript/final_manuscript.md").read_text(encoding="utf-8")
    metric_rows = _rows("reports/gate11_metric_summary.csv")
    expected = {
        row["candidate_id"]: f"{float(row['mean_finite']):.3f}"
        for row in metric_rows
        if row["metric"] == "mean_regional_dice"
    }
    assert all(value in manuscript for value in expected.values())
    assert "RES and WC originated in BU-Net" in manuscript
    assert "No contribution is framed as a new network block" in manuscript
    assert "do not establish clinical utility" in manuscript
    assert "state-of-the-art result" in manuscript
    assert "nnU-Net" in manuscript
    assert "{{" not in manuscript
    assert manuscript.count("](figures/final/") == 10
    assert manuscript.count("**Table ") == 8


def test_gate14_outputs_and_originality_audit_are_valid() -> None:
    required = [
        Path("manuscript/final_manuscript.docx"),
        Path("manuscript/final_manuscript.pdf"),
        Path("manuscript/final_manuscript.tex"),
        Path("manuscript/response_to_reviewer.docx"),
        Path("manuscript/response_to_reviewer.pdf"),
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    assert required[1].read_bytes().startswith(b"%PDF-")
    assert required[4].read_bytes().startswith(b"%PDF-")
    for docx_path in (required[0], required[3]):
        assert zipfile.is_zipfile(docx_path)
        with zipfile.ZipFile(docx_path) as archive:
            assert "word/document.xml" in archive.namelist()

    audit = json.loads(
        Path("reports/gate14_originality_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "pass"
    assert audit["exact_ngram_match_count"] == 0
    assert audit["repeated_sentences"] == []
