from __future__ import annotations

import pandas as pd

from analysis.audit_seed_variation import (
    _display_optional_query,
    classify_checkpoint_variation,
    earliest_seed_divergence,
)


def test_earliest_seed_divergence_finds_first_nonconstant_query() -> None:
    frame = pd.DataFrame(
        {
            "seed": [5, 5, 5, 6, 6, 6],
            "oracle_evaluations": [1, 2, 3, 1, 2, 3],
            "cumulative_target_count": [1, 2, 2, 1, 2, 3],
        }
    )
    assert earliest_seed_divergence(frame) == 3


def test_earliest_seed_divergence_returns_none_for_identical_trajectories() -> None:
    frame = pd.DataFrame(
        {
            "seed": [5, 5, 6, 6],
            "oracle_evaluations": [1, 2, 1, 2],
            "cumulative_target_count": [1, 2, 1, 2],
        }
    )
    assert earliest_seed_divergence(frame) is None


def test_classification_separates_checkpoint_equality_from_prefix_equality() -> None:
    assert classify_checkpoint_variation([2, 3], prefix_divergence_query=3) == (
        "checkpoint_varies_across_seeds"
    )
    assert classify_checkpoint_variation([3, 3], prefix_divergence_query=2) == (
        "checkpoint_equal_but_round_prefix_differs"
    )
    assert classify_checkpoint_variation([3, 3], prefix_divergence_query=None) == (
        "checkpoint_and_round_prefix_identical"
    )


def test_optional_query_display_handles_pandas_nan_and_integer_floats() -> None:
    assert _display_optional_query(float("nan")) == "none"
    assert _display_optional_query(None) == "none"
    assert _display_optional_query(288.0) == "288"
