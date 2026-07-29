from __future__ import annotations

import csv
from pathlib import Path

from bratsarticle.utils.serialization import atomic_write_json


def test_claim_ledger_keeps_unsupported_result_claims_visible() -> None:
    with Path("claims/q1q2_v2_claim_ledger.csv").open(
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) >= 10
    unsupported = {row["claim_id"] for row in rows if row["status"] == "unsupported"}
    assert {"Q2C-009", "Q2C-010"} <= unsupported
    assert all(row["artifact"] for row in rows)


def test_atomic_json_helper_used_by_claim_audit(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    atomic_write_json(output, {"pass": True})

    assert output.read_text(encoding="utf-8").endswith("\n")
