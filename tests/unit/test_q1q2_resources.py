import numpy as np
import pandas as pd

from bratsarticle.analysis.q1q2_resources import pareto_flags


def test_pareto_flags_require_no_worse_accuracy_and_cost() -> None:
    frame = pd.DataFrame(
        {
            "accuracy": [0.80, 0.82, 0.81, 0.79],
            "latency": [1.0, 2.0, 1.5, 3.0],
        }
    )

    flags = pareto_flags(
        frame,
        accuracy_column="accuracy",
        cost_columns=["latency"],
    )

    assert np.array_equal(flags, np.asarray([True, True, True, False]))


def test_multicost_pareto_does_not_create_a_subjective_score() -> None:
    frame = pd.DataFrame(
        {
            "accuracy": [0.80, 0.81, 0.82],
            "latency": [1.0, 2.0, 3.0],
            "memory": [3.0, 2.0, 1.0],
        }
    )

    flags = pareto_flags(
        frame,
        accuracy_column="accuracy",
        cost_columns=["latency", "memory"],
    )

    assert flags.all()
