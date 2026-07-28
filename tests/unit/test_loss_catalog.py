from pathlib import Path

import pytest
import torch

from bratsarticle.training.loss_catalog import (
    LOSS_FORMULAS,
    LossConfig,
    build_loss,
    load_loss_catalog,
)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn((2, 4, 16, 16), requires_grad=True)
    labels = torch.zeros((2, 16, 16), dtype=torch.long)
    labels[:, 2:14, 2:14] = 2
    labels[:, 5:11, 5:11] = 1
    labels[:, 7:9, 7:9] = 4
    return logits, labels


def test_loss_catalog_is_complete_and_explicit() -> None:
    catalog = load_loss_catalog(Path("configs/losses/catalog.yaml"))

    assert {config.name for config in catalog} == set(LOSS_FORMULAS)
    assert all(config.expects_logits for config in catalog)
    assert all(config.smoothing > 0 for config in catalog)
    assert all(config.reduction in {"mean", "sum"} for config in catalog)


@pytest.mark.parametrize(
    "config",
    load_loss_catalog(Path("configs/losses/catalog.yaml")),
    ids=lambda config: config.name,
)
def test_every_configured_loss_is_finite_and_differentiable(
    config: LossConfig,
) -> None:
    logits, labels = _batch()

    loss = build_loss(config)(logits, labels)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_catalog_rejects_invalid_labels() -> None:
    config = load_loss_catalog(Path("configs/losses/catalog.yaml"))[0]
    logits, labels = _batch()
    labels[0, 0, 0] = 3

    with pytest.raises(ValueError, match="Unexpected BraTS labels"):
        build_loss(config)(logits, labels)
