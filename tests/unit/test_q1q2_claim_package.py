from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bratsarticle.reporting.q1q2_claims import (
    ClaimRegistry,
    audit_claim_package,
    audit_claim_template,
    render_claim_template,
)
from bratsarticle.utils.hashing import file_digest


def _contract(tmp_path: Path) -> Path:
    path = tmp_path / "claim_execution.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "template_contract": {
                    "artifact_bound_start": ("<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->"),
                    "artifact_bound_end": "<!-- END_ARTIFACT_BOUND_RESULTS -->",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _registry(tmp_path: Path) -> Path:
    source = tmp_path / "primary.csv"
    source.write_text("contrast_id,mean_difference\nprimary,0.0234\n", encoding="utf-8")
    registry = ClaimRegistry()
    registry.add(
        "CONTRAST.PRIMARY.MEAN_DIFFERENCE",
        0.0234,
        source_path=source,
        selector={"contrast_id": "primary"},
        column="mean_difference",
        inferential_role="confirmatory_prespecified_contrast",
    )
    claims = registry.claims()
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "claim_count": len(claims),
                "claims": claims,
                "source_hashes": {source.as_posix(): file_digest(source)},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_claim_template_renders_only_registry_values_and_audits(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    template = tmp_path / "manuscript.template.md"
    template.write_text(
        "<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->\n"
        "The paired difference was "
        "{{claim:CONTRAST.PRIMARY.MEAN_DIFFERENCE|3f}}.\n"
        "<!-- END_ARTIFACT_BOUND_RESULTS -->\n",
        encoding="utf-8",
    )
    rendered = tmp_path / "manuscript.md"
    trace = tmp_path / "trace.json"

    report = render_claim_template(
        template_path=template,
        registry_path=registry,
        output_path=rendered,
        trace_path=trace,
        config_path=_contract(tmp_path),
    )
    audit = audit_claim_package(
        registry_path=registry,
        rendered_path=rendered,
        trace_path=trace,
    )

    assert "0.023" in rendered.read_text(encoding="utf-8")
    assert report["resolved_token_count"] == 1
    assert audit["valid"] is True


def test_claim_template_rejects_manual_result_numbers(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    template = tmp_path / "manuscript.template.md"
    template.write_text(
        "<!-- BEGIN_ARTIFACT_BOUND_RESULTS -->\n"
        "The manually entered difference was 0.023.\n"
        "{{claim:CONTRAST.PRIMARY.MEAN_DIFFERENCE|3f}}\n"
        "<!-- END_ARTIFACT_BOUND_RESULTS -->\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Manual numeric literals"):
        render_claim_template(
            template_path=template,
            registry_path=registry,
            output_path=tmp_path / "rendered.md",
            trace_path=tmp_path / "trace.json",
            config_path=_contract(tmp_path),
        )


def test_claim_registry_rejects_duplicate_identifiers(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    source.write_text("value\n1\n", encoding="utf-8")
    registry = ClaimRegistry()
    registry.add(
        "CLAIM.ID",
        1,
        source_path=source,
        selector={},
        column="value",
        inferential_role="design_fact",
    )

    with pytest.raises(RuntimeError, match="Duplicate"):
        registry.add(
            "CLAIM.ID",
            2,
            source_path=source,
            selector={},
            column="value",
            inferential_role="design_fact",
        )


def test_repository_manuscript_template_has_only_bound_result_tokens() -> None:
    report = audit_claim_template(Path("manuscript/q1q2_v2_manuscript.template.md"))

    assert report["valid"] is True
    assert report["artifact_bound_section_count"] == 2
    assert report["claim_token_count"] > 20
