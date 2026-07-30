from __future__ import annotations

import pandas as pd
import pytest

from bratsarticle.analysis.q1q2_subgroups import assign_external_subgroups


def test_external_subgroups_use_frozen_development_burden_thresholds() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_id": model,
                "patient_id": patient,
                "cohort_role": "external_confirmatory",
                "institution": "site",
                "scanner_vendor": "vendor",
                "scanner_model": "scanner",
                "field_strength_t": 1.5,
                "spacing_axis0_mm": 1.0,
                "spacing_axis1_mm": 1.0,
                "spacing_axis2_mm": 1.0,
                "mean_regional_dice": 0.8,
            }
            for model in ("a", "b")
            for patient in (f"p{index:02d}" for index in range(95))
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "patient_id": f"p{index:02d}",
                "disease_group": "glioma",
                "institution": "site",
                "scanner_vendor": "vendor",
                "scanner_model": "scanner",
                "field_strength_t": 1.5,
                "grade": "",
                "wt_volume_mm3": float(index),
                "et_voxel_count": 0 if index == 0 else 1,
            }
            for index in range(95)
        ]
    )

    enriched = assign_external_subgroups(
        metrics,
        manifest,
        lower_burden_mm3=30.0,
        upper_burden_mm3=60.0,
    )

    assert len(enriched) == 190
    assert set(enriched["grade_if_available"]) == {"unknown"}
    assert set(enriched.loc[enriched["patient_id"].eq("p00"), "et_present"]) == {
        "absent"
    }
    assert set(
        enriched.loc[
            enriched["patient_id"].eq("p30"),
            "development_derived_tumor_burden_tertile",
        ]
    ) == {"small"}
    assert set(
        enriched.loc[
            enriched["patient_id"].eq("p31"),
            "development_derived_tumor_burden_tertile",
        ]
    ) == {"medium"}
    assert set(
        enriched.loc[
            enriched["patient_id"].eq("p61"),
            "development_derived_tumor_burden_tertile",
        ]
    ) == {"large"}
    assert set(enriched["resolution"]) == {"1.000x1.000x1.000_mm"}


def test_external_subgroups_reject_metadata_drift() -> None:
    metrics = pd.DataFrame(
        [
            {
                "model_id": "a",
                "patient_id": f"p{index:02d}",
                "cohort_role": "external_confirmatory",
                "institution": "wrong" if index == 0 else "site",
                "scanner_vendor": "vendor",
                "scanner_model": "scanner",
                "field_strength_t": 1.5,
                "spacing_axis0_mm": 1.0,
                "spacing_axis1_mm": 1.0,
                "spacing_axis2_mm": 1.0,
            }
            for index in range(95)
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "patient_id": f"p{index:02d}",
                "disease_group": "glioma",
                "institution": "site",
                "scanner_vendor": "vendor",
                "scanner_model": "scanner",
                "field_strength_t": 1.5,
                "grade": "",
                "wt_volume_mm3": 100.0,
                "et_voxel_count": 1,
            }
            for index in range(95)
        ]
    )

    with pytest.raises(ValueError, match="metadata differs"):
        assign_external_subgroups(
            metrics,
            manifest,
            lower_burden_mm3=50.0,
            upper_burden_mm3=150.0,
        )
