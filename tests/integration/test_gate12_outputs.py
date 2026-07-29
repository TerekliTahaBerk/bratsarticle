import json
from pathlib import Path

from PIL import Image

from bratsarticle.utils.hashing import file_digest


def test_gate12_outputs_are_complete_and_traceable() -> None:
    manifest_path = Path("reports/gate12_output_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["gate"] == 12
    assert manifest["figure_count"] == 10
    assert manifest["table_count"] == 6

    outputs = manifest["outputs"]
    assert len(outputs) == 16
    assert len({entry["id"] for entry in outputs}) == 16

    figures = [entry for entry in outputs if entry["kind"] == "figure"]
    tables = [entry for entry in outputs if entry["kind"] == "table"]
    assert len(figures) == 10
    assert len(tables) == 6

    for entry in figures:
        png = Path(entry["png_path"])
        pdf = Path(entry["pdf_path"])
        assert file_digest(png) == entry["png_sha256"]
        assert file_digest(pdf) == entry["pdf_sha256"]
        with Image.open(png) as image:
            assert image.size == (
                entry["pixel_width"],
                entry["pixel_height"],
            )
            assert image.width >= 2000
            assert image.height >= 850
        for source, expected_digest in entry["source_sha256"].items():
            assert file_digest(Path(source)) == expected_digest

    for entry in tables:
        csv = Path(entry["csv_path"])
        latex = Path(entry["tex_path"])
        assert file_digest(csv) == entry["csv_sha256"]
        assert file_digest(latex) == entry["tex_sha256"]
        assert entry["row_count"] > 0
        assert entry["column_count"] > 0
        for source, expected_digest in entry["source_sha256"].items():
            assert file_digest(Path(source)) == expected_digest


def test_gate12_report_records_no_test_or_raw_data_reopening() -> None:
    report = Path("reports/gate12_completion.md").read_text(encoding="utf-8")
    assert "**Decision:** PASS" in report
    assert "- Hand-entered scientific result values: 0" in report
    assert "- Internal-test manifest reopened: no" in report
    assert "- Raw-data files accessed: no" in report
