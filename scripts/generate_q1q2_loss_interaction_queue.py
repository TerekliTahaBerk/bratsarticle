#!/usr/bin/env python3
"""Generate the 100-job native architecture-by-loss sensitivity queue."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from bratsarticle.experiments.q1q2_native_runner import (
    loss_interaction_specs,
)
from bratsarticle.utils.hashing import file_digest
from bratsarticle.utils.serialization import atomic_write_json


def main() -> None:
    runner_config = Path("configs/q1q2_v2/m1_native_runner.yaml")
    selected_loss = Path("configs/q1q2_v2/selected_loss.yaml")
    specs = loss_interaction_specs(runner_config, selected_loss)
    payload = {
        "schema_version": 1,
        "status": "frozen_not_started",
        "scientific_role": "architecture_by_loss_interaction_sensitivity",
        "job_count": len(specs),
        "runner_config": runner_config.as_posix(),
        "runner_config_sha256": file_digest(runner_config),
        "selected_loss_config": selected_loss.as_posix(),
        "selected_loss_config_sha256": file_digest(selected_loss),
        "external_data_permitted": False,
        "legacy_internal_test_permitted": False,
        "jobs": [
            {
                **asdict(spec),
                "run_id": spec.run_id,
                "run_spec_sha256": spec.sha256,
                "status": "not_started",
            }
            for spec in specs
        ],
    }
    output = Path("artifacts/q1q2_v2/queues/loss_interaction.json")
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
