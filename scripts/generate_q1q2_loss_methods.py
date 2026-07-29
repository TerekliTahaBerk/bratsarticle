#!/usr/bin/env python3
"""Generate v2 loss Methods metadata from the executable catalog."""

from pathlib import Path

from bratsarticle.training.loss_methods import write_loss_methods


def main() -> None:
    write_loss_methods(
        Path("configs/losses/catalog.yaml"),
        Path("reports/q1q2_v2/loss_methods.json"),
        Path("reports/q1q2_v2/loss_methods.csv"),
        Path("reports/q1q2_v2/loss_methods.md"),
    )


if __name__ == "__main__":
    main()
