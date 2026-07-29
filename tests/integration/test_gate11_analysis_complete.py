import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_gate11_reports_all_frozen_seeds_with_patient_level_inference() -> None:
    audit = json.loads(
        Path("reports/gate11_artifact_audit.json").read_text(encoding="utf-8")
    )
    assert audit["status"] == "complete"
    assert audit["valid_checkpoint_count"] == 13
    assert audit["expected_checkpoint_count"] == 13
    assert audit["expected_patient_count_per_checkpoint"] == 74
    assert audit["access_event_count"] == 1
    assert audit["access_event_valid"] is True
    assert audit["invalid_runs"] == {}

    seed = pd.read_csv("reports/gate11_patient_seed_metrics.csv")
    candidate = pd.read_csv("reports/gate11_patient_candidate_metrics.csv")
    assert len(seed) == 13 * 74
    assert len(candidate) == 3 * 74
    assert seed.groupby(["candidate_id", "seed"])["patient_id"].nunique().eq(74).all()
    expected = (
        seed.groupby(["candidate_id", "patient_id"])["mean_regional_dice"]
        .mean()
        .sort_index()
    )
    actual = candidate.set_index(["candidate_id", "patient_id"])[
        "mean_regional_dice"
    ].sort_index()
    np.testing.assert_allclose(
        expected.to_numpy(),
        actual.to_numpy(),
        rtol=0.0,
        atol=1e-15,
    )

    means = actual.groupby(level=0).mean()
    np.testing.assert_allclose(
        means.loc[["unet_reference", "bunet", "unet_res"]].to_numpy(),
        [0.7355293941591515, 0.7522553258849438, 0.7559864312915746],
        rtol=0.0,
        atol=1e-15,
    )

    comparisons = pd.read_csv("reports/gate11_comparisons.csv")
    formal = comparisons.loc[comparisons["formal_hypothesis_test"].astype(bool)]
    assert len(formal) == 3
    assert formal["paired_patient_count"].eq(74).all()
    assert formal["holm_adjusted_p_value"].le(0.05).all()


def test_gate11_access_log_contains_exactly_one_frozen_opening() -> None:
    events = [
        json.loads(line)
        for line in Path("artifacts/test_access_log.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(events) == 1
    assert events[0]["event"] == "internal_test_manifest_access"
    assert (
        events[0]["manifest_sha256"]
        == "455b3b661be73a84fc99458798ee9a5cbbf9c70deac0b425397220fbbab7a525"
    )
