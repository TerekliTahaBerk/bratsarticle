#!/usr/bin/env python3
"""Validate and record the v2 model/seed matrix."""

from pathlib import Path

from bratsarticle.experiments.q1q2_protocol import write_matrix_validation


def main() -> None:
    write_matrix_validation(
        Path("configs/q1q2_v2/model_matrix.yaml"),
        Path("configs/q1q2_v2/seeds.yaml"),
        Path("reports/q1q2_v2/model_matrix_validation.json"),
    )


if __name__ == "__main__":
    main()
