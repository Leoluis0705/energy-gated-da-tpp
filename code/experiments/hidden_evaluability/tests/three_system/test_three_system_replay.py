import numpy as np
import pandas as pd

from analysis.three_system.replay import (
    ALL_METHODS,
    _select_batch,
    freeze_prospective_union,
    gate_uses_correction,
    make_paired_pseudo_oracle,
    run_paired_replay,
)


def test_joint_qualified_greedy_ranks_the_joint_target_probability():
    """Catches a target-qualified Greedy baseline optimizing only one factor."""
    frame = pd.DataFrame(
        {
            "candidate_id": ["high_eval", "high_band", "joint_best"],
            "m_element": ["Cr", "Mn", "Mg"],
            "current_p_eval": [0.95, 0.55, 0.80],
            "p_interval_hit": [0.40, 0.95, 0.75],
        }
    )
    embedding = np.zeros((len(frame), 1), dtype=float)

    chosen, correction, _, _ = _select_batch(
        frame,
        embedding,
        method="joint_qualified_greedy",
        batch_size=1,
        rng=np.random.default_rng(1),
    )

    assert chosen == ["joint_best"]
    assert correction is False


def test_pseudo_oracle_is_paired_across_methods_but_changes_across_seeds():
    """Catches generating a different pseudo truth for each baseline."""
    scores = pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(20)],
            "p_dft_evaluable": np.linspace(0.1, 0.9, 20),
            "predicted_dft_energy_mean": np.linspace(-2.4, -1.4, 20),
            "predicted_dft_energy_std": np.full(20, 0.1),
        }
    )
    a = make_paired_pseudo_oracle(scores, seed=15)
    b = make_paired_pseudo_oracle(scores, seed=15)
    c = make_paired_pseudo_oracle(scores, seed=16)

    pd.testing.assert_frame_equal(a, b)
    assert not a["pseudo_dft_evaluable"].equals(c["pseudo_dft_evaluable"])


def test_gate_ablation_semantics_do_not_change_with_unrelated_diagnostic():
    """Catches Group-only reading margin or Margin-only reading concentration."""
    assert gate_uses_correction("always_correction", margin=1.0, concentration=0.0)
    assert not gate_uses_correction("group_only", margin=0.0, concentration=0.4)
    assert gate_uses_correction("group_only", margin=1.0, concentration=0.6)
    assert gate_uses_correction("margin_only", margin=0.05, concentration=0.0)
    assert gate_uses_correction("margin_only", margin=0.05, concentration=1.0)
    assert not gate_uses_correction("full_gate", margin=0.2, concentration=0.4)


def test_frozen_union_respects_cap_compositions_and_all_baselines():
    """Catches a prospective freeze that drops Mg or leaves a baseline unrepresented."""
    rows = []
    methods = sorted(ALL_METHODS)
    candidates = [
        ("cr1", "Cr", 1, 1),
        ("cr2", "Cr", 2, 2),
        ("mn1", "Mn", 3, 1),
        ("mn2", "Mn", 4, 2),
        ("mg1", "Mg", 5, 1),
        ("mg2", "Mg", 6, 2),
    ]
    for index, method in enumerate(methods):
        candidate_id, element, cluster, atoms = candidates[index % len(candidates)]
        rows.append(
            {
                "candidate_id": candidate_id,
                "m_element": element,
                "structure_matcher_cluster": cluster,
                "atom_count": atoms,
                "space_group_number": 1,
                "method": method,
                "seed": 15 + (index % 2),
                "query": 16,
            }
        )
    early = pd.DataFrame(rows)

    frozen = freeze_prospective_union(
        early,
        maximum_candidates=12,
        required_elements=("Cr", "Mn", "Mg"),
    )

    assert len(frozen) <= 12
    assert set(frozen["m_element"]) == {"Cr", "Mn", "Mg"}
    assert frozen["structure_matcher_cluster"].is_unique
    covered = set()
    for value in frozen["represented_methods"]:
        covered.update(value.split("|"))
    assert covered == set(methods)


def test_paired_replay_runs_every_method_on_the_same_seed_oracle():
    """Catches a baseline being skipped or receiving a different pseudo truth."""
    rng = np.random.default_rng(7)
    prospective = pd.DataFrame(
        {
            "candidate_id": [f"c{i:03d}" for i in range(48)],
            "m_element": np.resize(["Cr", "Mn", "Mg"], 48),
            "structure_matcher_cluster": np.arange(48) % 12,
            "atom_count": np.full(48, 7),
            "space_group_number": np.arange(48) % 5 + 1,
            "p_dft_evaluable": np.linspace(0.15, 0.85, 48),
            "p_interval_hit": np.linspace(0.8, 0.2, 48),
            "p_interval_alignn": np.linspace(0.75, 0.25, 48),
            "p_interval_mlip": np.linspace(0.7, 0.3, 48),
            "evaluability_uncertainty": np.linspace(0.05, 0.2, 48),
            "predicted_dft_energy_mean": np.linspace(-2.35, -1.45, 48),
            "predicted_dft_energy_std": np.full(48, 0.08),
            "composition_only_probability": np.resize([0.4, 0.6, 0.8], 48),
            "feature_0": rng.normal(size=48),
            "feature_1": rng.normal(size=48),
        }
    )
    history_x = pd.DataFrame(
        {
            "m_element": np.resize(["Cr", "Mn", "Mg"], 12),
            "feature_0": rng.normal(size=12),
            "feature_1": rng.normal(size=12),
        }
    )
    history_y = np.array([0, 1] * 6)

    results, selections = run_paired_replay(
        prospective,
        history_x,
        history_y,
        model_feature_columns=("m_element", "feature_0", "feature_1"),
        paired_seeds=(15,),
        methods=ALL_METHODS,
        batch_size=4,
        query_budget=8,
        checkpoints=(4, 8),
        classifier_model_name="regularized_logistic",
        bootstrap_draws=5,
    )

    assert set(results["method"]) == set(ALL_METHODS)
    assert results.groupby("method")["checkpoint"].nunique().eq(2).all()
    assert selections.groupby("method")["candidate_id"].nunique().eq(8).all()
    assert selections["pseudo_oracle_sha256"].nunique() == 1
    assert "proxy_model_seed" in selections.columns

    group_gated = selections.loc[
        selections["method"] == "group_gated_da_tpp",
        ["seed", "query", "candidate_id", "proxy_model_seed"],
    ].reset_index(drop=True)
    full_gate = selections.loc[
        selections["method"] == "full_gate",
        ["seed", "query", "candidate_id", "proxy_model_seed"],
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(group_gated, full_gate)
