from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bratsarticle.experiments.q1q2_native_runner import (
    loss_screen_specs,
    main_convergence_specs,
    resolve_loss_screen_spec,
    resolve_main_convergence_spec,
)
from bratsarticle.utils.hashing import file_digest

RUNNER_CONFIG = Path("configs/q1q2_v2/m1_native_runner.yaml")


def test_loss_screen_expands_to_frozen_unique_matrix() -> None:
    specs = loss_screen_specs(RUNNER_CONFIG)

    assert len(specs) == 15
    assert len({spec.run_id for spec in specs}) == 15
    assert {spec.fold for spec in specs} == {1, 2, 3, 4, 5}
    assert {spec.seed for spec in specs} == {20260730}
    assert all(spec.maximum_optimizer_steps == 10_000 for spec in specs)


def test_resolver_rejects_unfrozen_seed() -> None:
    with pytest.raises(PermissionError, match="outside the frozen"):
        resolve_loss_screen_spec(
            RUNNER_CONFIG,
            model_id="unet_small",
            fold=1,
            seed=999,
            loss_name="cross_entropy_plus_soft_dice",
        )


def _selected_loss(tmp_path: Path, status: str) -> Path:
    evidence = tmp_path / "loss_selection.json"
    evidence.write_text(
        (
            '{"status":"selected_from_complete_development_cv",'
            '"selected_loss":"cross_entropy_plus_soft_dice",'
            '"external_data_accessed":false,'
            '"legacy_internal_test_accessed":false}'
        ),
        encoding="utf-8",
    )
    path = tmp_path / "selected_loss.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "status": status,
                "selected_loss": "cross_entropy_plus_soft_dice",
                "external_data_used_for_selection": False,
                "legacy_internal_test_used_for_selection": False,
                "selection_artifact": evidence.as_posix(),
                "selection_artifact_sha256": file_digest(evidence),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_main_expands_to_equal_seed_matrix(tmp_path: Path) -> None:
    selected = _selected_loss(
        tmp_path,
        "frozen_from_complete_development_cv",
    )

    specs = main_convergence_specs(RUNNER_CONFIG, selected)

    assert len(specs) == 225
    assert len({spec.run_id for spec in specs}) == 225
    assert len({spec.model_id for spec in specs}) == 9
    assert {spec.fold for spec in specs} == {1, 2, 3, 4, 5}
    assert {spec.seed for spec in specs} == {
        20260730,
        20260731,
        20260732,
        20260733,
        20260734,
    }
    assert {spec.loss_name for spec in specs} == {
        "cross_entropy_plus_soft_dice"
    }
    assert all(spec.maximum_optimizer_steps == 50_000 for spec in specs)
    assert all(spec.full_metric_evaluation for spec in specs)


def test_native_main_rejects_unfrozen_loss(tmp_path: Path) -> None:
    selected = _selected_loss(tmp_path, "pending")

    with pytest.raises(PermissionError, match="not frozen"):
        main_convergence_specs(RUNNER_CONFIG, selected)


def test_native_main_resolver_rejects_unfrozen_model(tmp_path: Path) -> None:
    selected = _selected_loss(
        tmp_path,
        "frozen_from_complete_development_cv",
    )

    with pytest.raises(PermissionError, match="outside the frozen"):
        resolve_main_convergence_spec(
            RUNNER_CONFIG,
            selected,
            model_id="nnunetv2_2d",
            fold=1,
            seed=20260730,
        )
