#!/usr/bin/env python3
"""Run or resume one frozen native Q1/Q2 development job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bratsarticle.experiments.q1q2_native_runner import (
    resolve_loss_screen_spec,
    resolve_main_convergence_spec,
    run_native_development,
)


def main() -> int:
    """Resolve a frozen job and execute it on M1/MPS."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-reportable-development-training", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("loss_screen", "main_convergence"),
        default="loss_screen",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--loss")
    arguments = parser.parse_args()
    runner_config = Path("configs/q1q2_v2/m1_native_runner.yaml")
    if arguments.stage == "loss_screen":
        if not arguments.loss:
            parser.error("--loss is required for loss_screen")
        spec = resolve_loss_screen_spec(
            runner_config,
            model_id=arguments.model,
            fold=arguments.fold,
            seed=arguments.seed,
            loss_name=arguments.loss,
        )
    else:
        if arguments.loss:
            parser.error(
                "--loss cannot be supplied for main_convergence; "
                "the frozen selection is mandatory"
            )
        spec = resolve_main_convergence_spec(
            runner_config,
            Path("configs/q1q2_v2/selected_loss.yaml"),
            model_id=arguments.model,
            fold=arguments.fold,
            seed=arguments.seed,
        )
    output = run_native_development(
        runner_config_path=runner_config,
        spec=spec,
        dataset_root=arguments.dataset_root,
        allow_reportable_development_training=(
            arguments.allow_reportable_development_training
        ),
        resume=arguments.resume,
    )
    print(json.dumps({"artifact_directory": output.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
