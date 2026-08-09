import pandas as pd
import pytest

from analysis.recompute_partial_budget_metrics import partial_metrics_from_history


def test_partial_autc_uses_left_continuous_completed_batch_values():
    history = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "iteration": [1, 1, 2, 2],
            "target_label": [1, 0, 1, 0],
        }
    )

    metrics = partial_metrics_from_history(
        history, batch_size=2, total_targets=2, full_budget=4, horizons=(2, 3, 4)
    )

    assert metrics["AUTC_at_2"] == 0.0
    assert metrics["AUTC_at_3"] == pytest.approx(1 / 6)
    assert metrics["AUTC_at_4"] == 0.25
    assert metrics["Recovery_at_2"] == 1
    assert metrics["Recovery_at_3"] == 1
    assert metrics["Recovery_at_4"] == 2
    assert metrics["full_budget_AUTC"] == 0.25


def test_partial_metrics_reject_horizon_beyond_full_budget():
    history = pd.DataFrame(
        {"id": ["a", "b"], "iteration": [1, 1], "target_label": [0, 1]}
    )

    with pytest.raises(ValueError, match="horizon"):
        partial_metrics_from_history(
            history, batch_size=2, total_targets=1, full_budget=2, horizons=(3,)
        )


def test_partial_metrics_require_complete_unique_history():
    history = pd.DataFrame(
        {"id": ["a", "a"], "iteration": [1, 1], "target_label": [0, 1]}
    )

    with pytest.raises(ValueError, match="duplicate"):
        partial_metrics_from_history(
            history, batch_size=2, total_targets=1, full_budget=2, horizons=(2,)
        )
