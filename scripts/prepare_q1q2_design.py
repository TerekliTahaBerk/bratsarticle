#!/usr/bin/env python3
"""Create the frozen v2 cross-validation and external cohort manifests."""

from pathlib import Path

from bratsarticle.data.cv_design import prepare_design


def main() -> None:
    prepare_design(
        canonical_manifest=Path(
            "manifests/canonical/brats2020_canonical_manifest.csv"
        ),
        external_inventory=Path("manifests/q1q2_v2/external_inventory.csv"),
        gate_c_summary=Path("reports/q1q2_v2/external_gate_c_summary.json"),
        fold_output_dir=Path("splits/q1q2_v2"),
        external_test_output=Path("splits/q1q2_v2/external_test.csv"),
        metadata_output=Path("splits/q1q2_v2/split_metadata.json"),
        protocol_report_output=Path(
            "reports/q1q2_v2/split_and_cohort_protocol.md"
        ),
        precision_json_output=Path(
            "reports/q1q2_v2/external_precision_analysis.json"
        ),
        precision_report_output=Path(
            "reports/q1q2_v2/external_precision_analysis.md"
        ),
    )


if __name__ == "__main__":
    main()
