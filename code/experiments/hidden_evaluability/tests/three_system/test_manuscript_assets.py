from importlib import import_module
from pathlib import Path
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_expected_yield_table_aggregates_seeds_and_consolidates_aliases():
    """Catches exact selector aliases being reported as independent evidence."""
    module_path = ROOT / "analysis/three_system/manuscript_assets.py"
    assert module_path.is_file()
    assets = import_module("analysis.three_system.manuscript_assets")
    assert hasattr(assets, "build_expected_yield_table")
    replay = pd.DataFrame(
        [
            {
                "method": method,
                "seed": seed,
                "checkpoint": 32,
                "estimated_interval_hit_count": qualified,
                "simulated_interval_hit_count": int(qualified),
                "estimated_DFT_evaluable_count": evaluable,
                "unique_structure_clusters": clusters,
                "evidence_tier": "retrospective_ML_assisted_simulation",
            }
            for method, seed, qualified, evaluable, clusters in (
                ("group_gated_da_tpp", 15, 24.0, 28.0, 10),
                ("group_gated_da_tpp", 16, 26.0, 30.0, 12),
                ("full_gate", 15, 24.0, 28.0, 10),
                ("full_gate", 16, 26.0, 30.0, 12),
                ("dft_evaluable_greedy", 15, 20.0, 31.0, 8),
                ("dft_evaluable_greedy", 16, 22.0, 33.0, 8),
            )
        ]
    )

    table = assets.build_expected_yield_table(
        replay,
        checkpoint=32,
        equivalent_aliases={"full_gate": "group_gated_da_tpp"},
    )

    assert table["method"].tolist() == [
        "group_gated_da_tpp",
        "dft_evaluable_greedy",
    ]
    group = table.iloc[0]
    assert group["mean_expected_qualified"] == 25.0
    assert round(group["sd_expected_qualified"], 6) == 1.414214
    assert group["mean_ml_labeled_qualified"] == 25.0
    assert round(group["sd_ml_labeled_qualified"], 6) == 1.414214
    assert group["mean_expected_evaluable"] == 29.0
    assert group["mean_structure_clusters"] == 11.0
    assert group["equivalent_aliases"] == "full_gate"
    assert (
        group["evidence_tier"]
        == "retrospective_ML_assisted_simulation"
    )


def test_model_validation_table_keeps_binary_and_energy_tasks_separate():
    """Catches unlike validation metrics being merged into one score."""
    assets = import_module("analysis.three_system.manuscript_assets")
    assert hasattr(assets, "build_model_validation_table")
    binary = pd.DataFrame(
        [
            {
                "model_name": "shallow_gradient_boosting",
                "n": 20,
                "loo_roc_auc": 0.82,
                "loo_balanced_accuracy": 0.70,
                "loo_brier_score": 0.183,
                "loo_log_loss": 0.545,
            }
        ]
    )
    energy = pd.DataFrame(
        [
            {
                "model_id": "MACE-MP::composition_offset",
                "n": 10,
                "loo_mae_eV_atom": 0.0145,
                "loo_rmse_eV_atom": 0.0159,
                "prediction_interval_coverage_95": 0.70,
            }
        ]
    )

    table = assets.build_model_validation_table(binary, energy)

    assert table["task"].tolist() == [
        "DFT evaluability",
        "DFT formation-energy calibration",
    ]
    assert table.loc[0, "roc_auc"] == 0.82
    assert pd.isna(table.loc[0, "mae_eV_atom"])
    assert table.loc[1, "mae_eV_atom"] == 0.0145
    assert pd.isna(table.loc[1, "roc_auc"])
    assert table.loc[0, "evidence_tier"] == "observed_DFT_nested_LOO"
    assert table.loc[1, "evidence_tier"] == "observed_DFT_energy_LOO"


def test_actual_dft_table_uses_only_observed_selected_subset_rows():
    """Catches model-estimated yield entering the observed DFT table."""
    assets = import_module("analysis.three_system.manuscript_assets")
    assert hasattr(assets, "build_actual_dft_table")
    summary = pd.DataFrame(
        [
                {
                    "policy": "Gate",
                    "n_observed_candidates": 12,
                    "final_DFT_evaluable": 10,
                    "final_DFT_failures": 2,
                    "time_to_first_evaluable_round": 2,
                    "DFT_evaluable_round_AUC": 101.0,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
                {
                    "policy": "Greedy",
                    "n_observed_candidates": 12,
                    "final_DFT_evaluable": 10,
                    "final_DFT_failures": 2,
                    "time_to_first_evaluable_round": 1,
                "DFT_evaluable_round_AUC": 75.0,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
        ]
    )
    curve = pd.DataFrame(
        [
            {
                "policy": "Gate",
                "round": 12,
                "cumulative_DFT_evaluable": 8,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
            {
                "policy": "Greedy",
                "round": 12,
                "cumulative_DFT_evaluable": 6,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
        ]
    )

    table = assets.build_actual_dft_table(summary, curve)

    assert table.to_dict("records") == [
        {
            "policy": "Gate",
            "first_evaluable_round": 2,
            "round_12_evaluable": 8,
            "final_evaluable": 10,
            "count_round_auc": 101.0,
            "evidence_tier": "observed_DFT_retrospective_selected_subset",
        },
        {
            "policy": "Greedy",
            "first_evaluable_round": 1,
            "round_12_evaluable": 6,
            "final_evaluable": 10,
            "count_round_auc": 75.0,
            "evidence_tier": "observed_DFT_retrospective_selected_subset",
        },
    ]


def test_asset_builder_writes_separate_estimated_and_observed_outputs(tmp_path):
    """Catches manuscript assets silently mixing pseudo and observed results."""
    assets = import_module("analysis.three_system.manuscript_assets")
    assert hasattr(assets, "build_manuscript_assets")
    analysis_dir = tmp_path / "analysis"
    observed_dir = tmp_path / "observed"
    output_dir = tmp_path / "manuscript"
    analysis_dir.mkdir()
    observed_dir.mkdir()

    pd.DataFrame(
        [
            {
                "model_name": "shallow_gradient_boosting",
                "n": 20,
                "positives": 10,
                "loo_roc_auc": 0.82,
                "loo_balanced_accuracy": 0.70,
                "loo_brier_score": 0.183,
                "loo_log_loss": 0.545,
            }
        ]
    ).to_csv(analysis_dir / "dft_evaluability_model_cv.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_id": "MACE-MP::composition_offset",
                "source": "MACE-MP",
                "method": "composition_offset",
                "n": 10,
                "loo_mae_eV_atom": 0.0145,
                "loo_rmse_eV_atom": 0.0159,
                "loo_bias_eV_atom": 0.0,
                "prediction_interval_coverage_95": 0.70,
                "interval_lower": -2.3,
                "interval_upper": -1.5,
            }
        ]
    ).to_csv(analysis_dir / "formation_energy_calibration_cv.csv", index=False)
    pd.DataFrame(
        [
            {
                "model_id": "MACE-MP::composition_offset",
                "source": "MACE-MP",
                "method": "composition_offset",
                "candidate_id": "c1",
                "m_element": "Cr",
                "observed_dft_energy_eV_atom": -2.0,
                "predicted_dft_energy_eV_atom": -2.01,
                "prediction_standard_deviation_eV_atom": 0.02,
                "prediction_interval_lower_95": -2.05,
                "prediction_interval_upper_95": -1.97,
                "outer_fold": 0,
            }
        ]
    ).to_csv(
        analysis_dir / "formation_energy_calibration_oof_predictions.csv",
        index=False,
    )
    replay_rows = []
    for method in (
        "group_gated_da_tpp",
        "full_gate",
            "always_correction",
            "explore_core_set",
            "joint_qualified_greedy",
            "dft_evaluable_greedy",
    ):
        for seed, qualified, evaluable in ((15, 24.0, 28.0), (16, 26.0, 30.0)):
            for checkpoint, scale in ((16, 0.5), (32, 1.0)):
                replay_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "estimated_DFT_evaluable_count": evaluable * scale,
                            "simulated_DFT_evaluable_count": 0,
                            "estimated_interval_hit_count": qualified * scale,
                            "simulated_interval_hit_count": int(
                                round(qualified * scale)
                            ),
                        "unique_structure_clusters": 10,
                        "unique_compositions": 3,
                        "correction_rounds": 1,
                        "pseudo_oracle_sha256": f"seed-{seed}",
                        "evidence_tier": "retrospective_ML_assisted_simulation",
                    }
                )
    pd.DataFrame(replay_rows).to_csv(
        analysis_dir / "paired_baseline_replay_results.csv", index=False
    )
    selection_rows = []
    selector_candidates = {
        "group_gated_da_tpp": ("g1", "g2"),
        "full_gate": ("g1", "g2"),
        "always_correction": ("a1", "a2"),
        "explore_core_set": ("a1", "a2"),
    }
    for method, candidates in selector_candidates.items():
        for seed in (15, 16):
            for query, candidate_id in enumerate(candidates, 1):
                selection_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "query": query,
                        "candidate_id": candidate_id,
                        "proxy_model_seed": seed * 100000,
                    }
                )
    pd.DataFrame(selection_rows).to_csv(
        analysis_dir / "paired_baseline_replay_selections.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "seed": seed,
                "left_method": "group_gated_da_tpp",
                "right_method": "full_gate",
                "same_candidate_sequence": True,
                "same_proxy_model_seed_sequence": True,
                "first_difference_query": pd.NA,
            }
            for seed in (15, 16)
        ]
    ).to_csv(
        analysis_dir / "paired_selector_equivalence_audit.csv", index=False
    )
    pool_rows = [
        *[
            {"candidate_id": f"cr{i}", "m_element": "Cr"}
            for i in range(114)
        ],
        *[
            {"candidate_id": f"mn{i}", "m_element": "Mn"}
            for i in range(127)
        ],
        *[
            {"candidate_id": f"mg{i}", "m_element": "Mg"}
            for i in range(34)
        ],
    ]
    pd.DataFrame(pool_rows).to_csv(
        analysis_dir / "three_system_pool.csv", index=False
    )
    pd.DataFrame(
        [
            {"candidate_id": f"h{i}", "dft_evaluable": int(i < 10)}
            for i in range(20)
        ]
    ).to_csv(analysis_dir / "historical_dft_binary_labels.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "policy": "Gate",
                "n_observed_candidates": 12,
                "final_DFT_evaluable": 10,
                "final_DFT_failures": 2,
                "time_to_first_evaluable_round": 2,
                "DFT_evaluable_round_AUC": 101.0,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
            {
                "policy": "Greedy",
                "n_observed_candidates": 12,
                "final_DFT_evaluable": 10,
                "final_DFT_failures": 2,
                "time_to_first_evaluable_round": 1,
                "DFT_evaluable_round_AUC": 75.0,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            },
        ]
    )
    summary.to_csv(
        observed_dir / "actual_dft_gate_greedy_summary.csv", index=False
    )
    curve = pd.DataFrame(
        [
            {
                "policy": policy,
                "round": round_number,
                "cumulative_DFT_evaluable": value,
                "evidence_tier": "observed_DFT_retrospective_selected_subset",
            }
            for policy, values in (("Gate", (0, 8)), ("Greedy", (0, 6)))
            for round_number, value in zip((0, 12), values)
        ]
    )
    curve.to_csv(
        observed_dir / "actual_dft_gate_greedy_curve.csv", index=False
    )
    (output_dir / "Figures").mkdir(parents=True)
    (output_dir / "Tables/generated").mkdir(parents=True)
    (output_dir / "Figures/Figure2_v34.pdf").write_bytes(b"legacy")
    (output_dir / "Tables/generated/legacy_v34.tex").write_text(
        "legacy\n", encoding="utf-8"
    )

    assets.build_manuscript_assets(analysis_dir, observed_dir, output_dir)

    expected = pd.read_csv(output_dir / "SourceData/multifidelity_expected_yield.csv")
    observed = pd.read_csv(output_dir / "SourceData/actual_dft_ordering.csv")
    assert expected["evidence_tier"].eq(
        "retrospective_ML_assisted_simulation"
    ).all()
    assert observed["evidence_tier"].eq(
        "observed_DFT_retrospective_selected_subset"
    ).all()
    macros = (
        output_dir / "Tables/generated/multifidelity_metrics_macros.tex"
    ).read_text(encoding="utf-8")
    assert r"\newcommand{\MFPoolCount}{275}" in macros
    assert r"\newcommand{\MFCrCount}{114}" in macros
    assert r"\newcommand{\MFMnCount}{127}" in macros
    assert r"\newcommand{\MFMgCount}{34}" in macros
    assert r"\newcommand{\MFProspectiveCount}{255}" in macros
    assert r"\newcommand{\MFGroupQualifiedMean}{25.00}" in macros
    assert r"\newcommand{\MFGroupMLLabeledQualifiedMean}{25.00}" in macros
    assert r"\newcommand{\ActualSubsetCount}{12}" in macros
    assert r"\newcommand{\ActualFinalEvaluable}{10}" in macros
    assert r"\newcommand{\ActualFailureCount}{2}" in macros
    assert "explore_core_set" not in set(expected["method"])
    always = expected.loc[expected["method"] == "always_correction"].iloc[0]
    assert always["equivalent_aliases"] == "explore_core_set"
    manifest = pd.read_csv(
        output_dir / "SourceData/manuscript_asset_sha256.csv"
    )
    assert "Figures/Figure2_v34.pdf" not in set(manifest["relative_path"])
    assert "Tables/generated/legacy_v34.tex" not in set(
        manifest["relative_path"]
    )
    for relative in (
        "Figures/Figure1_multifidelity_workflow.pdf",
        "Figures/Figure2_model_validation.pdf",
        "Figures/Figure3_expected_yield.pdf",
        "Figures/Figure4_actual_dft_ordering.pdf",
        "Tables/generated/table_model_validation.tex",
        "Tables/generated/table_expected_yield.tex",
        "Tables/generated/table_actual_dft_ordering.tex",
        "SourceData/selector_equivalence_audit.csv",
        "SourceData/manuscript_asset_sha256.csv",
    ):
        path = output_dir / relative
        assert path.is_file()
        assert path.stat().st_size > 0

    first_manifest = (
        output_dir / "SourceData/manuscript_asset_sha256.csv"
    ).read_bytes()
    time.sleep(1.1)
    assets.build_manuscript_assets(analysis_dir, observed_dir, output_dir)
    second_manifest = (
        output_dir / "SourceData/manuscript_asset_sha256.csv"
    ).read_bytes()
    assert second_manifest == first_manifest
