from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from bratsarticle.experiments.q1q2_gate_g import (
    _expected_counts,
    _external_access_problems,
    freeze_gate_g,
)


def test_gate_g_protocol_requires_all_600_prespecified_runs() -> None:
    protocol = yaml.safe_load(
        Path("configs/q1q2_v2/gate_g_freeze.yaml").read_text(encoding="utf-8")
    )

    counts = _expected_counts(protocol)

    assert counts["native_main_convergence"] == 225
    assert counts["swin_main_convergence"] == 25
    assert counts["official_nnunet_main_convergence"] == 50
    assert counts["native_compute_matched"] == 200
    assert counts["native_loss_interaction"] == 100
    assert counts["total"] == 600
    assert protocol["external_inference_permitted"] is False


def test_gate_g_external_guard_rejects_any_prediction_access(
    tmp_path: Path,
) -> None:
    access_log = tmp_path / "external.jsonl"
    access_log.write_text(
        json.dumps(
            {
                "event": "external_model_inference",
                "model_inference": True,
                "prediction_metrics_accessed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    problems = _external_access_problems(access_log)

    assert len(problems) == 1
    assert "before Gate G" in problems[0]


def test_gate_g_freeze_requires_explicit_authorization() -> None:
    with pytest.raises(PermissionError, match="allow-analysis-freeze"):
        freeze_gate_g(allow_analysis_freeze=False)
