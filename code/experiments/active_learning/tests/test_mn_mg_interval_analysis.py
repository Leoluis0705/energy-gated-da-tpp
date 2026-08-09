from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.mn_mg_interval_analysis import (
    expected_evaluability_at,
    paired_statistics,
    reconstruct_trajectory,
    summarize_target_set,
)


def test_reconstruct_trajectory_includes_frozen_initial_batch() -> None:
    summary = pd.DataFrame(
        {
            "round": [0, 1, 2],
            "oracle_evaluations": [4, 8, 12],
            "cumulative_target_count": [1, 3, 5],
        }
    )

    trajectory = reconstruct_trajectory(summary, budget=12, total_targets=6)

    assert trajectory["recovery_at_8"] == 3
    assert trajectory["final_recovery"] == 5
    assert trajectory["autc"] == pytest.approx(2 / 9)


def test_summarize_target_set_reports_composition_and_cluster_concentration() -> None:
    pool = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d", "e"],
            "m_element": ["Mn", "Mn", "Cr", "Mg", "Mg"],
            "alignn_formation_energy_eV_atom": [-2.05, -2.00, -2.03, -2.20, -1.50],
            "structure_matcher_cluster": ["x", "x", "y", "z", "w"],
        }
    )

    summary, composition = summarize_target_set(pool, low=-2.1, high=-1.9)

    assert summary["target_count"] == 3
    assert summary["target_fraction"] == pytest.approx(0.6)
    assert summary["effective_cluster_count"] == 2
    assert summary["largest_cluster_fraction"] == pytest.approx(2 / 3)
    assert composition.set_index("m_element")["target_count"].to_dict() == {
        "Cr": 1,
        "Mn": 2,
    }


def test_expected_evaluability_is_post_selection_and_reports_coverage() -> None:
    history = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "target_label": [1, 0, 1, 1],
        }
    )
    scores = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "p_dft_evaluable": [0.8, 0.7, 0.25],
        }
    )

    result = expected_evaluability_at(history, scores, checkpoint=4)

    assert result["selected_target_count"] == 3
    assert result["scored_target_count"] == 2
    assert result["score_coverage"] == pytest.approx(2 / 3)
    assert result["expected_evaluable_target_count"] == pytest.approx(1.05)
    assert result["expected_evaluable_fraction_among_scored_targets"] == pytest.approx(0.525)


def test_paired_statistics_uses_exact_wilcoxon_and_fixed_bootstrap() -> None:
    differences = np.arange(1.0, 11.0)

    stats = paired_statistics(differences, bootstrap_samples=1_000, bootstrap_seed=17)

    assert stats["mean_difference"] == pytest.approx(5.5)
    assert stats["wins"] == 10
    assert stats["ties"] == 0
    assert stats["losses"] == 0
    assert stats["wilcoxon_p_two_sided_exact"] == pytest.approx(0.001953125)
    assert stats["effect_size_dz"] == pytest.approx(
        np.mean(differences) / np.std(differences, ddof=1)
    )
