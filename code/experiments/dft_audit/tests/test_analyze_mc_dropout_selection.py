import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.analyze_mc_dropout_selection import (
    analyze_mc_dropout_selection,
    render_selection_report,
)


METHODS = ("interval_hit_greedy", "energy_gated_da_tpp")


def _write_run(
    root: Path,
    *,
    k: int,
    method: str,
    sigma: list[float],
    top_b: str,
    route: str,
    autc: float,
    runtime: float,
) -> None:
    run = root / f"k_{k}" / method / "seed_0" / "attempt_1"
    run.mkdir(parents=True)
    (run / "status.json").write_text(
        json.dumps({"status": "DONE", "elapsed_seconds": runtime}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame([{"method": method, "seed": 0, "AUTC": autc}]).to_csv(
        run / "run_metrics.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "round": 1,
                "route": route,
                "direct_top_b_candidate_ids": top_b,
            }
        ]
    ).to_csv(run / "round_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "mu_eV": [-2.0, -1.9, -1.8, -1.7],
            "sigma_eV": sigma,
        }
    ).to_csv(run / f"{method}_scores_protocol_iter_1.csv", index=False)


def test_analyzer_compares_each_k_with_paired_k30_runs(tmp_path: Path) -> None:
    settings = {
        3: ([1.0, 2.0, 3.0, 4.0], "a;b", "direct", 0.70, 10.0),
        10: ([4.0, 3.0, 2.0, 1.0], "c;d", "correction", 0.75, 15.0),
        30: ([1.0, 2.0, 3.0, 4.0], "a;c", "correction", 0.80, 20.0),
    }
    for k, (sigma, top_b, gate_route, autc, runtime) in settings.items():
        for method in METHODS:
            route = gate_route if method == "energy_gated_da_tpp" else "threshold_greedy"
            _write_run(
                tmp_path,
                k=k,
                method=method,
                sigma=sigma,
                top_b=top_b,
                route=route,
                autc=autc,
                runtime=runtime,
            )

    round_detail, run_detail, summary = analyze_mc_dropout_selection(tmp_path)

    k3 = summary.loc[summary["mc_passes"] == 3].iloc[0]
    assert k3["median_uncertainty_spearman_vs_k30"] == 1.0
    assert k3["median_top_b_overlap_vs_k30"] == 0.5
    assert k3["gate_flip_rate_vs_k30"] == 1.0
    assert k3["mean_absolute_AUTC_difference_vs_k30"] == pytest.approx(0.1)
    assert k3["median_runtime_ratio_vs_k30"] == 0.5

    k30 = summary.loc[summary["mc_passes"] == 30].iloc[0]
    assert k30["median_uncertainty_spearman_vs_k30"] == 1.0
    assert k30["median_top_b_overlap_vs_k30"] == 1.0
    assert k30["gate_flip_rate_vs_k30"] == 0.0
    assert k30["mean_absolute_AUTC_difference_vs_k30"] == 0.0
    assert len(round_detail) == 6
    assert len(run_detail) == 6


def test_analyzer_rejects_incomplete_k_method_seed_grid(tmp_path: Path) -> None:
    for k in (3, 10, 30):
        _write_run(
            tmp_path,
            k=k,
            method="interval_hit_greedy",
            sigma=[1.0, 2.0, 3.0, 4.0],
            top_b="a;b",
            route="threshold_greedy",
            autc=0.8,
            runtime=10.0,
        )

    try:
        analyze_mc_dropout_selection(tmp_path)
    except ValueError as exc:
        assert "complete method-by-K-by-seed grid" in str(exc)
    else:
        raise AssertionError("incomplete development grid was accepted")


def test_render_selection_report_records_evidence_and_decision() -> None:
    summary = pd.DataFrame(
        [
            {
                "mc_passes": 3,
                "median_uncertainty_spearman_vs_k30": 0.17,
                "median_top_b_overlap_vs_k30": 0.125,
                "gate_flip_rate_vs_k30": 0.25,
                "mean_absolute_AUTC_difference_vs_k30": 0.017,
                "maximum_absolute_AUTC_difference_vs_k30": 0.036,
                "mean_runtime_seconds": 959.0,
                "median_runtime_ratio_vs_k30": 0.993,
            },
            {
                "mc_passes": 10,
                "median_uncertainty_spearman_vs_k30": 0.34,
                "median_top_b_overlap_vs_k30": 0.125,
                "gate_flip_rate_vs_k30": 0.18,
                "mean_absolute_AUTC_difference_vs_k30": 0.014,
                "maximum_absolute_AUTC_difference_vs_k30": 0.034,
                "mean_runtime_seconds": 948.0,
                "median_runtime_ratio_vs_k30": 0.981,
            },
            {
                "mc_passes": 30,
                "median_uncertainty_spearman_vs_k30": 1.0,
                "median_top_b_overlap_vs_k30": 1.0,
                "gate_flip_rate_vs_k30": 0.0,
                "mean_absolute_AUTC_difference_vs_k30": 0.0,
                "maximum_absolute_AUTC_difference_vs_k30": 0.0,
                "mean_runtime_seconds": 968.0,
                "median_runtime_ratio_vs_k30": 1.0,
            },
        ]
    )
    validation = {
        "job_count": 30,
        "status_counts": {"DONE": 30},
        "manifest_sha256": "a" * 64,
        "paired_mask_comparisons": 600,
        "issue_count": 0,
    }
    environment = {"python": "3.12", "numpy": "2.3", "scipy": "1.17"}

    report = render_selection_report(
        summary,
        validation=validation,
        environment=environment,
        selected_k=30,
        source_directory="server/results/mc_selection",
    )

    assert "Selected MC passes: `K = 30`" in report
    assert "600" in report
    assert "0.340000" in report
    assert "0.180000" in report
    assert "aaaaaaaaaaaaaaaa" in report
    assert "seeds 0-4" in report
