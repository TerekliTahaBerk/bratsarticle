#!/usr/bin/env python3
"""Generate the frozen M1 loss-screen queue without starting training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from bratsarticle.experiments.q1q2_native_runner import loss_screen_specs
from bratsarticle.utils.serialization import atomic_write_json


def main() -> int:
    """Write all 15 frozen job specifications."""
    path = Path("configs/q1q2_v2/m1_native_runner.yaml")
    specs = loss_screen_specs(path)
    atomic_write_json(
        Path("artifacts/q1q2_v2/queues/loss_screen.json"),
        {
            "schema_version": 1,
            "status": "frozen_not_started",
            "runner_config": path.as_posix(),
            "jobs": [asdict(spec) | {"run_id": spec.run_id} for spec in specs],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
