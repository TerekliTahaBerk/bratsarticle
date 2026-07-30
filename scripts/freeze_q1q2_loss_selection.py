#!/usr/bin/env python3
"""Freeze the architecture-attribution loss after the 15-job screen."""

from __future__ import annotations

import json
from pathlib import Path

from bratsarticle.experiments.q1q2_loss_selection import write_loss_freeze


def main() -> None:
    payload = write_loss_freeze(
        queue_path=Path("artifacts/q1q2_v2/queues/loss_screen.json"),
        artifact_root=Path("artifacts/q1q2_v2/native_runs"),
        fold_directory=Path("splits/q1q2_v2"),
        protocol_path=Path("configs/q1q2_v2/loss_protocol.yaml"),
        output_path=Path("reports/q1q2_v2/loss_selection.json"),
        selected_config_path=Path("configs/q1q2_v2/selected_loss.yaml"),
        repository_root=Path.cwd(),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
