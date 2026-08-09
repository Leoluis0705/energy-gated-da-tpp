from pathlib import Path

import pandas as pd

from analysis.three_system.data import (
    build_historical_binary_labels,
    build_three_system_pool,
)


ROOT = Path(__file__).resolve().parents[2]


def test_three_system_pool_filters_exact_frozen_counts_and_two_mlip_rows():
    """Catches accidental inclusion of Al/Co/Ni/Ti or loss of one MLIP result."""
    pool = build_three_system_pool(
        ROOT / "inputs/three_system/candidate_pool_master.csv",
        ROOT / "inputs/three_system/mlip_full_pool_results.csv",
        ROOT / "dft/audit/dft_candidate_manifest.csv",
    )

    assert len(pool) == 275
    assert pool["m_element"].value_counts().to_dict() == {
        "Mn": 127,
        "Cr": 114,
        "Mg": 34,
    }
    assert pool["candidate_id"].is_unique
    assert set(pool["mlip_model_count"]) == {2}
    assert int(pool["historical_dft"].sum()) == 20


def test_historical_binary_labels_enforce_reproducible_energy_requirement():
    """Catches treating an old static-only pilot as a successful DFT label."""
    labels = build_historical_binary_labels(
        ROOT / "dft/audit/dft_candidate_manifest.csv",
        ROOT
        / "results/post_submission_analysis/egdatpp_psfix_v1_20260719T031102Z"
        / "dft/recomputed_formation_energies.csv",
    )

    assert len(labels) == 20
    assert labels["candidate_id"].is_unique
    assert labels["dft_evaluable"].value_counts().to_dict() == {1: 10, 0: 10}
    pilot = labels.loc[
        labels["candidate_id"]
        == "job_029_Cr_fe_-1.337_n4_generated_crystals_cif__gen_2"
    ].iloc[0]
    assert pilot["dft_evaluable"] == 0
    assert pilot["label_reason"] == "formation_energy_not_reproducible"


def test_model_feature_columns_exclude_candidate_and_historical_policy_leakage():
    """Catches ID or historical Gate/Greedy ranks entering the model matrix."""
    pool = build_three_system_pool(
        ROOT / "inputs/three_system/candidate_pool_master.csv",
        ROOT / "inputs/three_system/mlip_full_pool_results.csv",
        ROOT / "dft/audit/dft_candidate_manifest.csv",
    )

    feature_columns = set(pool.attrs["model_feature_columns"])
    forbidden = {
        "candidate_id",
        "gate_rank",
        "greedy_rank",
        "gate_round",
        "greedy_round",
        "target_label",
    }
    assert feature_columns.isdisjoint(forbidden)
    assert {"m_element", "chgnet_final_energy_eV_atom", "mace_final_energy_eV_atom"} <= feature_columns
    assert pd.api.types.is_numeric_dtype(pool["mlip_energy_disagreement_eV_atom"])
