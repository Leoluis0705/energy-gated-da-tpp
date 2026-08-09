from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.build_matched_seed_figures import (
    build_figure_source_data,
    make_figure,
    trajectory_bootstrap_summary,
)


def test_trajectory_bootstrap_summary_is_deterministic_and_contains_mean() -> None:
    matrix = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]])
    first = trajectory_bootstrap_summary(matrix, samples=10_000, seed=17)
    second = trajectory_bootstrap_summary(matrix, samples=10_000, seed=17)
    pd.testing.assert_frame_equal(first, second)
    assert first["mean_recovery"].tolist() == [0.0, 2.0, 4.0]
    assert np.all(first["bootstrap_ci_low"] <= first["mean_recovery"])
    assert np.all(first["bootstrap_ci_high"] >= first["mean_recovery"])
    assert first["sample_sd"].tolist() == [0.0, 1.0, 2.0]


def test_build_source_data_uses_exact_ten_seed_grid_and_zero_origin() -> None:
    rows = []
    for method, base in (("energy_gated_da_tpp", 2), ("predicted_distance_greedy", 1)):
        for seed in range(5, 15):
            for round_index in (1, 2):
                rows.append(
                    {
                        "dataset": "limo",
                        "method": method,
                        "seed": seed,
                        "round": round_index,
                        "oracle_evaluations": round_index * 16,
                        "round_target_hits": base,
                        "cumulative_target_count": round_index * base,
                    }
                )
    source = build_figure_source_data(
        pd.DataFrame(rows), dataset="limo", bootstrap_samples=1_000, bootstrap_seed=9
    )
    assert len(source) == 2 * 10 * 3
    assert set(source["seed"]) == set(range(5, 15))
    assert set(source[source["oracle_evaluations"] == 0]["cumulative_target_count"]) == {0}
    assert source["bootstrap_samples"].unique().tolist() == [1_000]
    assert np.all(source["bootstrap_ci_low"] <= source["mean_recovery"])
    assert np.all(source["bootstrap_ci_high"] >= source["mean_recovery"])


def test_formal_limo_source_has_expected_cardinality_and_checkpoint_means() -> None:
    archive = Path(__file__).resolve().parents[2]
    trajectories = pd.read_csv(archive / "results/audit/seed_variation_details.csv")
    source = build_figure_source_data(
        trajectories, dataset="limo", bootstrap_samples=1_000, bootstrap_seed=9
    )
    assert len(source) == 820
    at_80 = source[source["oracle_evaluations"] == 80].drop_duplicates(
        ["method", "oracle_evaluations"]
    )
    means = dict(zip(at_80["method"], at_80["mean_recovery"], strict=True))
    assert means == {
        "energy_gated_da_tpp": 26.0,
        "predicted_distance_greedy": 19.0,
    }


def test_title_subtitle_do_not_overlap_and_legend_stays_inside_canvas() -> None:
    archive = Path(__file__).resolve().parents[2]
    trajectories = pd.read_csv(archive / "results/audit/seed_variation_details.csv")
    source = build_figure_source_data(
        trajectories, dataset="mnoxide", bootstrap_samples=1_000, bootstrap_seed=9
    )
    fig = make_figure(source, "mnoxide")
    try:
        fig.canvas.draw()
        ax = fig.axes[0]
        renderer = fig.canvas.get_renderer()
        title_box = ax._left_title.get_window_extent(renderer)
        subtitle = next(text for text in ax.texts if text.get_text().startswith("Corrected seeds"))
        subtitle_box = subtitle.get_window_extent(renderer)
        assert not title_box.overlaps(subtitle_box)
        legend_box = ax.get_legend().get_window_extent(renderer)
        canvas_box = fig.bbox
        assert canvas_box.contains(legend_box.x0, legend_box.y0)
        assert canvas_box.contains(legend_box.x1, legend_box.y1)
    finally:
        plt.close(fig)
