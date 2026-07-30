from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bratsarticle.analysis.q1q2_statistics import (
    Contrast,
    hierarchical_bootstrap_intervals,
    holm_adjust,
    paired_contrast_summary,
)


def test_holm_adjustment_is_monotone_in_sorted_order() -> None:
    adjusted = holm_adjust({"c": 0.04, "a": 0.01, "b": 0.03})

    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_paired_summary_uses_patients_not_replicates() -> None:
    frame = pd.DataFrame(
        [
            {"model_id": model, "patient_id": patient, "score": score}
            for patient, first, second in (
                ("p1", 0.9, 0.7),
                ("p2", 0.8, 0.7),
                ("p3", 0.7, 0.7),
            )
            for model, score in (("first", first), ("second", second))
        ]
    )
    summary, paired = paired_contrast_summary(
        frame,
        contrast=Contrast("first_vs_second", "first", "second"),
        endpoint="score",
        bootstrap_resamples=1000,
        confidence_level=0.95,
        bootstrap_seed=12,
        permutation_resamples=1000,
        permutation_seed=13,
        smallest_effect_size_of_interest=0.02,
    )

    assert summary["paired_patient_count"] == 3
    assert summary["mean_difference"] == pytest.approx(0.1)
    assert summary["probability_of_superiority"] == pytest.approx(5 / 6)
    assert len(paired) == 3


def test_hierarchical_bootstrap_resamples_seed_then_patient_deterministically() -> None:
    rows: list[dict[str, object]] = []
    for model, sign in (("first", 1.0), ("second", 0.0)):
        for patient_index, patient in enumerate(("p1", "p2", "p3")):
            for seed in (10, 11):
                for fold in (1, 2):
                    rows.append(
                        {
                            "model_id": model,
                            "patient_id": patient,
                            "training_seed": seed,
                            "training_fold": fold,
                            "score": (
                                0.5
                                + sign * 0.1
                                + patient_index * 0.01
                                + seed * 0.0001
                                + fold * 0.00001
                            ),
                        }
                    )
    frame = pd.DataFrame(rows)
    first = hierarchical_bootstrap_intervals(
        frame,
        contrast=Contrast("first_vs_second", "first", "second"),
        endpoint="score",
        resamples=500,
        confidence_level=0.95,
        seed=27,
    )
    second = hierarchical_bootstrap_intervals(
        frame,
        contrast=Contrast("first_vs_second", "first", "second"),
        endpoint="score",
        resamples=500,
        confidence_level=0.95,
        seed=27,
    )

    assert first == second
    assert first["patient_count"] == 3
    assert first["training_seed_count"] == 2
    assert first["fold_count"] == 2
    assert first["hierarchical_mean_difference"] == pytest.approx(0.1)
    assert np.isfinite(first["fold_resampling_sensitivity_lower_95"])


def test_hierarchical_bootstrap_rejects_incomplete_pairing() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_id": "first",
                "patient_id": "p1",
                "training_seed": 1,
                "training_fold": 1,
                "score": 0.8,
            },
            {
                "model_id": "second",
                "patient_id": "p2",
                "training_seed": 1,
                "training_fold": 1,
                "score": 0.7,
            },
        ]
    )

    with pytest.raises(ValueError, match="incomplete replicate pairing"):
        hierarchical_bootstrap_intervals(
            frame,
            contrast=Contrast("first_vs_second", "first", "second"),
            endpoint="score",
            resamples=10,
            confidence_level=0.95,
            seed=1,
        )
