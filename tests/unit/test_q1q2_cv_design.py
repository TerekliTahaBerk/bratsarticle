from __future__ import annotations

import numpy as np
import pandas as pd

from bratsarticle.data.cv_design import (
    _external_test_frame,
    select_five_fold_design,
)


def _development_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(100):
        rows.append(
            {
                "subject_id": f"patient-{index:03d}",
                "grade": "HGG" if index < 70 else "LGG",
                "et_present": bool(index % 2),
                "wt_volume_quartile": f"Q{index % 4 + 1}",
                "primary_stratum": f"S{index % 10}",
                "wt_volume_mm3": float(index + 1),
                "tc_volume_mm3": float(index + 2),
                "et_volume_mm3": float(index + 3),
            }
        )
    return pd.DataFrame(rows)


def test_five_fold_assignment_is_deterministic_and_exclusive() -> None:
    frame = _development_frame()

    first = select_five_fold_design(frame, seed=17, candidate_count=8)
    second = select_five_fold_design(frame, seed=17, candidate_count=8)

    assert np.array_equal(first.assignment, second.assignment)
    assert set(first.assignment.tolist()) == {1, 2, 3, 4, 5}
    assert all(int(np.sum(first.assignment == fold)) == 20 for fold in range(1, 6))


def test_external_manifest_keeps_only_primary_glioma() -> None:
    rows: list[dict[str, object]] = []
    for index in range(96):
        rows.append(
            {
                "patient_id": f"ext-{index:03d}",
                "disease_group": "glioma" if index < 95 else "other_neoplasm",
                "primary_confirmatory_eligibility": (
                    "eligible" if index < 95 else "supportive_only"
                ),
                "eligibility_status": "eligible",
                "t1_path": "t1",
                "t1ce_path": "t1ce",
                "t2_path": "t2",
                "flair_path": "flair",
                "label_path": "seg",
                "institution": "site",
                "scanner_vendor": "vendor",
                "scanner_model": "model",
                "field_strength_t": 1.5,
                "wt_voxel_count": 100,
                "tc_voxel_count": 50,
                "et_voxel_count": 10,
                "label_mapping": "0->0;1->1;2->2;3->4",
            }
        )

    external = _external_test_frame(pd.DataFrame(rows), "inventory-hash")

    assert len(external) == 95
    assert external["disease_group"].eq("glioma").all()
    assert external["role"].eq("external_confirmatory_test").all()
    assert external["patient_id"].is_unique
