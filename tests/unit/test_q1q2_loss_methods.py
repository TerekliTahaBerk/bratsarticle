from __future__ import annotations

from pathlib import Path

from bratsarticle.training.loss_catalog import build_loss, load_loss_catalog
from bratsarticle.training.loss_methods import MANDATORY_Q1Q2_LOSSES, _row


def test_mandatory_loss_methods_match_executable_configs() -> None:
    catalog = {
        config.name: config
        for config in load_loss_catalog(Path("configs/losses/catalog.yaml"))
    }

    rows = [_row(catalog[name]) for name in MANDATORY_Q1Q2_LOSSES]

    assert len(rows) == 3
    assert rows[0]["ce_probability_transform"] == "softmax"
    assert rows[1]["bce_probability_transform"] == "independent sigmoid"
    assert rows[1]["bce_background_included"] is False
    assert rows[2]["overlap_probability_transform"] == "softmax"
    assert all(build_loss(catalog[name]) is not None for name in MANDATORY_Q1Q2_LOSSES)
