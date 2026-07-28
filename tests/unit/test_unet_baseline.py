from pathlib import Path

import numpy as np
import pytest
import torch

from bratsarticle.models import StandardUNet2D
from bratsarticle.training.checkpoint import load_checkpoint, save_checkpoint
from bratsarticle.training.cli import assert_training_authorized
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.losses import DiceCrossEntropyLoss
from bratsarticle.training.reproducibility import (
    collect_run_metadata,
    seed_everything,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    image = torch.randn((1, 4, 32, 32), generator=generator)
    label = torch.zeros((1, 32, 32), dtype=torch.long)
    label[:, 4:28, 4:28] = 2
    label[:, 10:22, 10:22] = 1
    label[:, 14:18, 14:18] = 4
    return image, label


def _model() -> StandardUNet2D:
    return StandardUNet2D(base_channels=4, depth=2)


def test_standard_unet_forward_backward() -> None:
    seed_everything(11)
    model = _model()
    image, label = _batch()
    logits = model(image)
    loss = DiceCrossEntropyLoss()(logits, label)
    loss.backward()

    assert logits.shape == (1, 4, 32, 32)
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )


def test_single_batch_loss_decreases() -> None:
    seed_everything(13)
    model = _model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    image, label = _batch()
    losses = [engine.train_step(image, label) for _ in range(25)]

    assert losses[-1] < losses[0] * 0.75
    assert min(losses[-5:]) < min(losses[:5])


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    seed_everything(19)
    initial = _model().state_dict()
    image, label = _batch()

    continuous_model = _model()
    continuous_model.load_state_dict(initial)
    continuous_optimizer = torch.optim.Adam(continuous_model.parameters(), lr=0.005)
    continuous = TrainingEngine(
        model=continuous_model,
        optimizer=continuous_optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    continuous.train_step(image, label)
    continuous_loss = continuous.train_step(image, label)

    interrupted_model = _model()
    interrupted_model.load_state_dict(initial)
    interrupted_optimizer = torch.optim.Adam(
        interrupted_model.parameters(),
        lr=0.005,
    )
    interrupted = TrainingEngine(
        model=interrupted_model,
        optimizer=interrupted_optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    interrupted.train_step(image, label)
    checkpoint = tmp_path / "resume.pt"
    save_checkpoint(
        checkpoint,
        model=interrupted.model,
        optimizer=interrupted.optimizer,
        scaler=interrupted.scaler,
        state=interrupted.state,
        metadata={"kind": "unit-test"},
    )

    resumed_model = _model()
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.005)
    resumed = TrainingEngine(
        model=resumed_model,
        optimizer=resumed_optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    resumed.state, metadata = load_checkpoint(
        checkpoint,
        model=resumed.model,
        optimizer=resumed.optimizer,
        scaler=resumed.scaler,
        map_location=torch.device("cpu"),
    )
    resumed_loss = resumed.train_step(image, label)

    assert metadata == {"kind": "unit-test"}
    assert resumed.state.global_step == 2
    assert resumed_loss == pytest.approx(continuous_loss, abs=0.0, rel=0.0)
    for name, continuous_parameter in continuous.model.state_dict().items():
        assert torch.equal(continuous_parameter, resumed.model.state_dict()[name])


def test_cpu_mixed_precision_toggle_runs() -> None:
    seed_everything(23)
    model = StandardUNet2D(base_channels=2, depth=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=True,
    )
    image, label = _batch()

    assert np.isfinite(engine.train_step(image, label))
    assert not engine.scaler.is_enabled()


def test_run_metadata_records_required_provenance() -> None:
    config_path = Path("configs/training/unet2d_baseline.yaml")
    metadata = collect_run_metadata(
        config_path=config_path,
        split_hashes={"train": "train-hash", "validation": "validation-hash"},
        seed=20260729,
        device=torch.device("cpu"),
        mixed_precision=False,
        run_kind="unit_test",
    )

    assert metadata["config_sha256"]
    assert metadata["split_sha256"]["train"] == "train-hash"
    assert metadata["git_commit"]
    assert metadata["seed"] == 20260729
    assert metadata["device"] == "cpu"
    assert metadata["memory_total_bytes"] > 0
    assert metadata["packages"]["torch"]


def test_full_training_has_flag_and_cuda_guards() -> None:
    with pytest.raises(PermissionError, match="allow-full-training"):
        assert_training_authorized(
            allow_full_training=False,
            smoke_steps=0,
            device=torch.device("cpu"),
            require_cuda_for_full_training=True,
        )
    with pytest.raises(RuntimeError, match="CUDA host"):
        assert_training_authorized(
            allow_full_training=True,
            smoke_steps=0,
            device=torch.device("cpu"),
            require_cuda_for_full_training=True,
        )
    assert_training_authorized(
        allow_full_training=False,
        smoke_steps=1,
        device=torch.device("cpu"),
        require_cuda_for_full_training=True,
    )
