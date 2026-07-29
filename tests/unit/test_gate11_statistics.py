import numpy as np
import pytest

from bratsarticle.experiments.gate11_analysis import (
    holm_adjust,
    paired_bootstrap_interval,
    sign_flip_permutation_p_value,
)


def test_paired_bootstrap_is_deterministic_and_patient_level() -> None:
    differences = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    first = paired_bootstrap_interval(
        differences,
        resamples=1000,
        confidence_level=0.95,
        seed=7,
    )
    second = paired_bootstrap_interval(
        differences,
        resamples=1000,
        confidence_level=0.95,
        seed=7,
    )
    assert first == second
    assert first[0] < np.mean(differences) < first[1]


def test_sign_flip_detects_large_consistent_paired_effect() -> None:
    differences = np.ones(20, dtype=np.float64)
    p_value = sign_flip_permutation_p_value(
        differences,
        resamples=20_000,
        seed=11,
    )
    assert p_value < 0.001


def test_holm_adjustment_controls_family_and_is_monotone() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["b"] == pytest.approx(0.06)
    assert adjusted["c"] == pytest.approx(0.2)
