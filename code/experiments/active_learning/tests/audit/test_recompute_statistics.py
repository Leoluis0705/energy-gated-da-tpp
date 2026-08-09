from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.recompute_statistics import (
    build_audit_bundle,
    build_round_trajectory,
    build_statistical_report,
    compute_run_artifacts,
    compare_v33_reference,
    discover_run_directories,
    paired_statistics,
)


def test_build_round_trajectory_reconstructs_only_from_raw_history() -> None:
    history = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "iteration": [1, 1, 2, 2],
            "target_label": [1, 0, 0, 1],
        }
    )
    trajectory = build_round_trajectory(history, batch_size=2, budget=4)
    assert trajectory.to_dict("list") == {
        "round": [1, 2],
        "oracle_evaluations": [2, 4],
        "round_target_hits": [1, 1],
        "cumulative_target_count": [1, 2],
    }


def test_build_round_trajectory_rejects_noncontiguous_rounds() -> None:
    history = pd.DataFrame(
        {"id": ["a", "b"], "iteration": [1, 3], "target_label": [0, 1]}
    )
    with pytest.raises(ValueError, match="contiguous"):
        build_round_trajectory(history, batch_size=1, budget=2)


def _create_minimal_run(root: Path, dataset: str, method: str, seed: int) -> None:
    run_dir = root / "runs" / dataset / method / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({"name": dataset, "method": method, "seed": seed}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"id": [f"{dataset}-{method}-{seed}"], "iteration": [1], "target_label": [0]}
    ).to_csv(run_dir / "al_history.csv", index=False)


def test_discover_run_directories_requires_exact_seed_method_grid(tmp_path: Path) -> None:
    methods = ("energy_gated_da_tpp", "predicted_distance_greedy")
    for method in methods:
        for seed in (5, 6):
            _create_minimal_run(tmp_path, "limo", method, seed)
    records = discover_run_directories(
        roots_by_seed_range=[(tmp_path, range(5, 7))],
        datasets=("limo",),
        methods=methods,
    )
    assert [(r.dataset, r.method, r.seed) for r in records] == [
        ("limo", "energy_gated_da_tpp", 5),
        ("limo", "energy_gated_da_tpp", 6),
        ("limo", "predicted_distance_greedy", 5),
        ("limo", "predicted_distance_greedy", 6),
    ]
    (tmp_path / "runs/limo/energy_gated_da_tpp/seed_6/al_history.csv").unlink()
    with pytest.raises(FileNotFoundError):
        discover_run_directories(
            roots_by_seed_range=[(tmp_path, range(5, 7))],
            datasets=("limo",),
            methods=methods,
        )


def test_paired_statistics_uses_requested_exact_wilcoxon_parameters() -> None:
    differences = np.arange(1.0, 11.0)
    stats = paired_statistics(differences, bootstrap_samples=1_000, bootstrap_seed=7)
    assert stats["wilcoxon"] == {
        "status": "computed",
        "statistic": 0.0,
        "pvalue": 0.001953125,
        "zero_method": "wilcox",
        "correction": False,
        "alternative": "two-sided",
        "method": "exact",
    }
    assert stats["paired_sd"] == pytest.approx(np.std(differences, ddof=1))
    assert stats["effect_size_dz"] == pytest.approx(
        np.mean(differences) / np.std(differences, ddof=1)
    )


def test_paired_statistics_marks_all_zero_vector_not_applicable() -> None:
    stats = paired_statistics(np.zeros(10), bootstrap_samples=1_000, bootstrap_seed=7)
    assert stats["effect_size_dz"] == 0.0
    assert stats["effect_size_zero_variance_convention"] == "all_zero_differences_yield_dz_0"
    assert stats["wilcoxon"]["status"] == "not_applicable_all_zero"
    assert stats["wilcoxon"]["statistic"] is None
    assert stats["wilcoxon"]["pvalue"] is None


def test_compute_run_artifacts_reconstructs_metrics_and_routing(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs/limo/energy_gated_da_tpp/seed_0"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "name": "limo",
                "method": "energy_gated_da_tpp",
                "seed": 0,
                "budget": 4,
                "batch_size": 2,
                "target_count": 2,
                "checkpoint_sha256": "initial-checkpoint-hash",
                "seed_policy": "test-policy",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "iteration": [1, 1, 2, 2],
            "target_label": [1, 0, 0, 1],
        }
    ).to_csv(run_dir / "al_history.csv", index=False)
    pd.DataFrame(
        {
            "iteration": [1, 2],
            "mode": ["threshold_greedy", "diversity_aware"],
        }
    ).to_csv(run_dir / "mode_trace.csv", index=False)
    pd.DataFrame(
        {
            "round": [1, 2],
            "route": ["threshold_greedy", "diversity_aware"],
            "correction_replacement_count": [0, 1],
            "selected_unique_groups": [2, 1],
            "selected_group_repetition_rate": [0.0, 0.5],
            "correction_target_gain": [0, 1],
        }
    ).to_csv(run_dir / "round_diagnostics.csv", index=False)
    pd.DataFrame(
        {"round": [1, 2], "sha256": ["cp-1", "cp-2"], "training_seed": [200001, 200002]}
    ).to_csv(run_dir / "checkpoint_manifest.csv", index=False)
    pd.DataFrame(
        {
            "round": [1, 2],
            "sha256": ["pred-1", "pred-2"],
            "inference_seed": [100001, 100002],
        }
    ).to_csv(run_dir / "prediction_manifest.csv", index=False)

    artifacts = compute_run_artifacts(
        discover_run_directories(
            [(tmp_path, range(0, 1))],
            datasets=("limo",),
            methods=("energy_gated_da_tpp",),
        )[0]
    )

    assert artifacts.metric["AUTC"] == 0.25
    assert artifacts.metric["recovery_at_80"] == 2
    assert artifacts.metric["candidate_query_count"] == 4
    assert artifacts.metric["initial_checkpoint_sha256"] == "initial-checkpoint-hash"
    assert artifacts.metric["final_checkpoint_sha256"] == "cp-2"
    assert json.loads(artifacts.metric["first_query_by_recovery_count_json"]) == {"1": 1, "2": 4}
    assert artifacts.routing["direct_rounds"] == 1
    assert artifacts.routing["correction_rounds"] == 1
    assert artifacts.routing["effective_replacements"] == 1
    assert artifacts.routing["mean_unique_groups_per_batch"] == 1.5
    assert artifacts.routing["repetition_rate"] == 0.25
    assert artifacts.routing["repetition_denominator_selected_slots"] == 4
    assert artifacts.routing["training_seed_count"] == 2
    assert artifacts.routing["inference_seed_count"] == 2
    assert artifacts.trajectory["cumulative_target_count"].tolist() == [1, 2]


def test_formal_snapshot_contains_exact_40_run_grid() -> None:
    archive = Path(__file__).resolve().parents[2]
    evidence = archive / "baseline_snapshot/archive/experiments/reproducibility/results"
    records = discover_run_directories(
        [
            (evidence / "paired_two_dataset_confirmation_20260712", range(5, 10)),
            (evidence / "paired_two_dataset_confirmation_seeds_10_14_20260713", range(10, 15)),
        ],
        datasets=("limo", "mnoxide"),
        methods=("energy_gated_da_tpp", "predicted_distance_greedy"),
    )
    assert len(records) == 40


def test_build_audit_bundle_has_formal_cardinalities_and_raw_history_value() -> None:
    archive = Path(__file__).resolve().parents[2]
    evidence = archive / "baseline_snapshot/archive/experiments/reproducibility/results"
    records = discover_run_directories(
        [
            (evidence / "paired_two_dataset_confirmation_20260712", range(5, 10)),
            (evidence / "paired_two_dataset_confirmation_seeds_10_14_20260713", range(10, 15)),
        ],
        datasets=("limo", "mnoxide"),
        methods=("energy_gated_da_tpp", "predicted_distance_greedy"),
    )
    bundle = build_audit_bundle(records, bootstrap_samples=1_000, bootstrap_seed=11)
    assert len(bundle.per_seed_metrics) == 40
    assert len(bundle.recovery_matrix) == 40
    assert len(bundle.paired_differences) == 20
    assert len(bundle.routing_statistics) == 40
    assert len(bundle.trajectories) == 1_200
    seed5 = bundle.per_seed_metrics.query(
        "dataset == 'limo' and method == 'energy_gated_da_tpp' and seed == 5"
    ).iloc[0]
    assert seed5["AUTC"] == pytest.approx(0.7913461538461538)
    assert seed5["recovery_at_80"] == 26
    assert bundle.statistics["datasets"]["limo"]["paired"]["bootstrap_samples"] == 1_000
    assert bundle.statistics["datasets"]["mnoxide"]["paired"]["wilcoxon"]["status"] == (
        "not_applicable_all_zero"
    )
    reference = json.loads((archive / "analysis/v33_table_reference.json").read_text(encoding="utf-8"))
    comparison = compare_v33_reference(bundle, reference)
    assert len(comparison) == len(reference["entries"])
    assert set(comparison["status"]).issubset(
        {"exact", "matches_reported_rounding", "outside_reported_rounding"}
    )
    autc_row = comparison[comparison["key"] == "limo.Gate.AUTC.mean"].iloc[0]
    assert autc_row["status"] == "matches_reported_rounding"
    report = build_statistical_report(
        bundle,
        comparison,
        {
            "python": "test-python",
            "numpy": "test-numpy",
            "scipy": "test-scipy",
            "pandas": "test-pandas",
        },
    )
    assert "raw `al_history.csv`" in report
    assert "1,000" in report
    assert "bootstrap seed: `11`" in report
    assert '`zero_method="wilcox"`' in report
    assert '`method="exact"`' in report
    assert "paired effect size `dz`" in report
    assert "outside_reported_rounding" in report
