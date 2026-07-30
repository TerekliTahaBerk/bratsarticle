from __future__ import annotations

import json
from pathlib import Path

import pytest

from bratsarticle.reporting.q1q2_reproducibility import (
    ArtifactIndex,
    verify_gate_i_manifest,
)
from bratsarticle.utils.hashing import file_digest


def test_artifact_index_merges_roles_without_losing_hash_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    index = ArtifactIndex()

    index.add(artifact, role="statistics")
    index.add(
        artifact,
        role="figure_source",
        expected_sha256=file_digest(artifact),
    )

    assert index.entries() == [
        {
            "path": artifact.as_posix(),
            "sha256": file_digest(artifact),
            "size_bytes": artifact.stat().st_size,
            "roles": ["figure_source", "statistics"],
        }
    ]


def test_artifact_index_rejects_hash_mismatch(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash differs"):
        ArtifactIndex().add(
            artifact,
            role="test",
            expected_sha256="0" * 64,
        )


def test_gate_i_manifest_verification_is_data_free(tmp_path: Path) -> None:
    artifact = tmp_path / "result.csv"
    artifact.write_text("value\n1\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "verified",
                "entry_count": 1,
                "entries": [
                    {
                        "path": artifact.as_posix(),
                        "sha256": file_digest(artifact),
                        "size_bytes": artifact.stat().st_size,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = verify_gate_i_manifest(manifest)

    assert report["valid"] is True
    assert report["raw_data_opened"] is False
    assert report["external_inference_performed"] is False
