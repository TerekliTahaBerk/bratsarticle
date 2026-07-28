from dataclasses import replace
from pathlib import Path

import pytest
import torch

from bratsarticle.models import ConfigurableUNet2D
from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.training.checkpoint import load_checkpoint, save_checkpoint
from bratsarticle.training.engine import TrainingEngine
from bratsarticle.training.losses import DiceCrossEntropyLoss
from bratsarticle.training.reproducibility import seed_everything

MODEL_CONFIGS = sorted(Path("configs/models").glob("*.yaml"))


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    image = torch.randn((1, 4, 24, 24), generator=generator)
    label = torch.zeros((1, 24, 24), dtype=torch.long)
    label[:, 3:21, 3:21] = 2
    label[:, 7:17, 7:17] = 1
    label[:, 10:14, 10:14] = 4
    return image, label


def _small_model(config_path: Path) -> ConfigurableUNet2D:
    config = replace(
        load_model_config(config_path),
        base_channels=2,
        depth=1,
        batch_normalization=False,
        dropout_probability=0.0,
        res_kernel_sizes=(3,),
        wc_kernel_size=3,
    )
    return ConfigurableUNet2D(config)


@pytest.mark.parametrize("config_path", MODEL_CONFIGS, ids=lambda path: path.stem)
def test_model_family_overfit_and_checkpoint_roundtrip(
    config_path: Path,
    tmp_path: Path,
) -> None:
    seed_everything(43)
    model = _small_model(config_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    image, label = _batch()
    losses = [engine.train_step(image, label) for _ in range(15)]
    checkpoint_path = tmp_path / f"{config_path.stem}.pt"
    save_checkpoint(
        checkpoint_path,
        model=engine.model,
        optimizer=engine.optimizer,
        scaler=engine.scaler,
        state=engine.state,
        metadata={"architecture": config_path.stem},
    )

    restored_model = _small_model(config_path)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.02)
    restored_engine = TrainingEngine(
        model=restored_model,
        optimizer=restored_optimizer,
        loss_function=DiceCrossEntropyLoss(),
        device=torch.device("cpu"),
        mixed_precision=False,
    )
    restored_state, metadata = load_checkpoint(
        checkpoint_path,
        model=restored_engine.model,
        optimizer=restored_engine.optimizer,
        scaler=restored_engine.scaler,
        map_location=torch.device("cpu"),
    )

    assert losses[-1] < losses[0]
    assert restored_state.global_step == 15
    assert metadata == {"architecture": config_path.stem}
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[name])
