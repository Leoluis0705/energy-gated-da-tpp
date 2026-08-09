from pathlib import Path

import pandas as pd
import pytest

from analysis.three_system import pipeline
from analysis.three_system.pipeline import (
    build_energy_calibration_table,
    largest_remainder_quotas,
    summarize_replay_results,
)
from analysis.three_system.data import (
    build_historical_binary_labels,
    build_three_system_pool,
)


ROOT = Path(__file__).resolve().parents[2]


def test_energy_calibration_table_contains_only_ten_reproducible_dft_points():
    """Catches failed or unreproducible historical attempts entering regression."""
    pool = build_three_system_pool(
        ROOT / "inputs/three_system/candidate_pool_master.csv",
        ROOT / "inputs/three_system/mlip_full_pool_results.csv",
        ROOT / "dft/audit/dft_candidate_manifest.csv",
    )
    labels = build_historical_binary_labels(
        ROOT / "dft/audit/dft_candidate_manifest.csv",
        ROOT
        / "results/post_submission_analysis/egdatpp_psfix_v1_20260719T031102Z"
        / "dft/recomputed_formation_energies.csv",
    )

    table = build_energy_calibration_table(pool, labels)

    assert len(table) == 10
    assert table["candidate_id"].is_unique
    assert set(table["m_element"]) == {"Cr", "Mn", "Mg"}
    assert {"dft_energy", "alignn", "chgnet", "mace"} <= set(table.columns)


def test_largest_remainder_quotas_match_twelve_candidate_pool_proportions():
    """Catches a prospective freeze that silently overrepresents one composition."""
    quotas = largest_remainder_quotas({"Cr": 102, "Mn": 121, "Mg": 32}, 12)

    assert quotas == {"Cr": 5, "Mn": 6, "Mg": 1}
    assert sum(quotas.values()) == 12


def test_selector_equivalence_audit_compares_sequences_within_each_seed():
    """Catches semantically identical gate labels receiving different draws."""
    assert hasattr(pipeline, "build_selector_equivalence_audit")
    selections = pd.DataFrame(
        [
            {
                "method": method,
                "seed": seed,
                "query": query,
                "candidate_id": candidate,
                "proxy_model_seed": seed * 100000,
            }
            for seed in (15, 16)
            for method in ("group_gated_da_tpp", "full_gate")
            for query, candidate in ((1, "c1"), (2, "c2"))
        ]
    )

    audit = pipeline.build_selector_equivalence_audit(selections)

    assert audit["same_candidate_sequence"].tolist() == [True, True]
    assert audit["same_proxy_model_seed_sequence"].tolist() == [True, True]
    assert audit["first_difference_query"].isna().all()


def test_selector_equivalence_validation_rejects_a_different_sequence():
    """Catches the pipeline publishing two identical selectors as different."""
    assert hasattr(pipeline, "require_selector_equivalence")
    audit = pd.DataFrame(
        [
            {
                "seed": 15,
                "same_candidate_sequence": False,
                "same_proxy_model_seed_sequence": True,
                "first_difference_query": 2,
            }
        ]
    )

    with pytest.raises(RuntimeError, match="seed 15.*query 2"):
        pipeline.require_selector_equivalence(audit)


def test_replay_summary_reports_joint_target_counts_as_first_class_endpoints():
    """Catches the joint ML-labeled endpoint being dropped during aggregation."""
    replay = pd.DataFrame(
        [
            {
                "method": "joint_qualified_greedy",
                "checkpoint": 32,
                "estimated_DFT_evaluable_count": 27.0,
                "simulated_DFT_evaluable_count": 26,
                "estimated_interval_hit_count": 23.5,
                "simulated_interval_hit_count": 22,
                "unique_structure_clusters": 8,
                "unique_compositions": 3,
            },
            {
                "method": "joint_qualified_greedy",
                "checkpoint": 32,
                "estimated_DFT_evaluable_count": 28.0,
                "simulated_DFT_evaluable_count": 27,
                "estimated_interval_hit_count": 24.5,
                "simulated_interval_hit_count": 24,
                "unique_structure_clusters": 9,
                "unique_compositions": 3,
            },
        ]
    )

    summary = summarize_replay_results(replay)
    row = summary.iloc[0]

    assert row["mean_simulated_interval_hit_count"] == pytest.approx(23.0)
    assert row["sd_simulated_interval_hit_count"] == pytest.approx(2**0.5)
    assert row["mean_estimated_interval_hit_count"] == pytest.approx(24.0)
    assert row["sd_estimated_interval_hit_count"] == pytest.approx(2**-0.5)
