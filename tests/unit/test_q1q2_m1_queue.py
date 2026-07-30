from __future__ import annotations

from pathlib import Path

import pytest

from bratsarticle.experiments.q1q2_m1_queue import (
    queue_snapshot,
    run_native_main_queue,
)
from bratsarticle.experiments.q1q2_native_runner import NativeRunSpec


def _spec(fold: int) -> NativeRunSpec:
    return NativeRunSpec(
        stage="loss_screen",
        model_id="unet_small",
        fold=fold,
        seed=20260730,
        loss_name="cross_entropy_plus_soft_dice",
        maximum_optimizer_steps=10_000,
        warmup_optimizer_steps=1_000,
        full_metric_evaluation=False,
    )


def test_queue_snapshot_detects_completed_and_unstarted_jobs(
    tmp_path: Path,
) -> None:
    specs = (_spec(1), _spec(2))
    completed = tmp_path / specs[0].run_id
    completed.mkdir()
    (completed / "progress.json").write_text(
        '{"status": "completed"}\n',
        encoding="utf-8",
    )

    snapshot = queue_snapshot(specs=specs, artifact_root=tmp_path)

    assert snapshot["job_count"] == 2
    assert snapshot["completed_count"] == 1
    assert snapshot["not_started_count"] == 1


def test_native_main_queue_rejects_active_loss_screen(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "loss_screen.lock").write_text("123\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="still active"):
        run_native_main_queue(
            runner_config_path=tmp_path / "unused_runner.yaml",
            selected_loss_path=tmp_path / "unused_loss.yaml",
            dataset_root=tmp_path / "unused_data",
            runtime_root=runtime,
            allow_reportable_development_training=True,
        )
