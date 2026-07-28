import pytest
import torch

from bratsarticle.training.checkpoint import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from bratsarticle.training.schedule import (
    build_warmup_cosine_scheduler,
    warmup_cosine_factor,
)


def test_warmup_cosine_schedule_has_declared_boundaries() -> None:
    assert warmup_cosine_factor(
        0,
        warmup_steps=10,
        total_steps=100,
        minimum_fraction=0.01,
    ) == pytest.approx(0.1)
    assert warmup_cosine_factor(
        10,
        warmup_steps=10,
        total_steps=100,
        minimum_fraction=0.01,
    ) == pytest.approx(1.0)
    assert warmup_cosine_factor(
        100,
        warmup_steps=10,
        total_steps=100,
        minimum_fraction=0.01,
    ) == pytest.approx(0.01)


def test_checkpoint_roundtrip_restores_scheduler(tmp_path) -> None:
    model = torch.nn.Conv2d(4, 4, kernel_size=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=2,
        total_steps=10,
        minimum_fraction=0.01,
    )
    optimizer.step()
    scheduler.step()
    path = tmp_path / "scheduled.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        scheduler=scheduler,
        state=TrainingState(global_step=1),
        metadata={"kind": "scheduler-test"},
    )

    restored_model = torch.nn.Conv2d(4, 4, kernel_size=1)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.001)
    restored_scheduler = build_warmup_cosine_scheduler(
        restored_optimizer,
        warmup_steps=2,
        total_steps=10,
        minimum_fraction=0.01,
    )
    state, metadata = load_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        scheduler=restored_scheduler,
        map_location=torch.device("cpu"),
    )

    assert state.global_step == 1
    assert metadata == {"kind": "scheduler-test"}
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
        optimizer.param_groups[0]["lr"]
    )
