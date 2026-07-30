from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bratsarticle.experiments.q1q2_protocol import (
    ProtocolMatrixError,
    validate_model_matrix,
)


def test_repository_matrix_has_equal_seed_design() -> None:
    result = validate_model_matrix(
        Path("configs/q1q2_v2/model_matrix.yaml"),
        Path("configs/q1q2_v2/seeds.yaml"),
    )

    assert len(result.model_ids) == 12
    assert len(result.main_seeds) == 5
    assert result.convergence_run_count == 300
    assert result.core_compute_matched_run_count == 200


def test_seed_inequality_is_rejected(tmp_path: Path) -> None:
    matrix_path = Path("configs/q1q2_v2/model_matrix.yaml")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    payload["main_models"][0]["seeds"] = [1, 2, 3, 4, 5]
    changed = tmp_path / "matrix.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ProtocolMatrixError, match="common ordered seed"):
        validate_model_matrix(changed, Path("configs/q1q2_v2/seeds.yaml"))


def test_wrong_component_attribution_is_rejected(tmp_path: Path) -> None:
    matrix_path = Path("configs/q1q2_v2/model_matrix.yaml")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    payload["main_models"][3]["primary_source"] = "10.1007/incorrect"
    changed = tmp_path / "matrix.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ProtocolMatrixError, match="wrong primary-source"):
        validate_model_matrix(changed, Path("configs/q1q2_v2/seeds.yaml"))


def test_outcome_selected_nnunet_3d_plan_is_rejected(tmp_path: Path) -> None:
    matrix_path = Path("configs/q1q2_v2/model_matrix.yaml")
    payload = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    payload["main_models"][9]["config"] = "results_selected_plan"
    changed = tmp_path / "matrix.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ProtocolMatrixError, match="ResEnc-L primary"):
        validate_model_matrix(changed, Path("configs/q1q2_v2/seeds.yaml"))
