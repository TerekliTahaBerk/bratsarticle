#!/usr/bin/env python3
"""Search plain U-Net capacity controls for the v2 RES comparison."""

from pathlib import Path

from bratsarticle.models.configurable_unet import load_model_config
from bratsarticle.models.matching import write_matching_report


def main() -> None:
    write_matching_report(
        target_config=load_model_config(Path("configs/models/unet_res.yaml")),
        search_output=Path("reports/q1q2_v2/model_matching_search.csv"),
        summary_output=Path("reports/q1q2_v2/model_matching_summary.json"),
        report_output=Path("reports/q1q2_v2/model_matching_report.md"),
        tolerance_fraction=0.02,
    )


if __name__ == "__main__":
    main()
