"""Execute the one-opening Gate 11 internal-test evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from bratsarticle.experiments.gate11_runner import run_gate11


def main() -> int:
    """Run all frozen internal-test checkpoints with an explicit guard flag."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/internal_test/gate11.yaml"),
    )
    parser.add_argument("--allow-test-evaluation", action="store_true")
    arguments = parser.parse_args()
    run_gate11(
        arguments.config,
        allow_test_evaluation=arguments.allow_test_evaluation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
