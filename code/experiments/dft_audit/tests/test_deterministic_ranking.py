import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FORMAL_PROJECT = ROOT / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/project"
sys.path.append(str(FORMAL_PROJECT))

from active_learning_energy_gate_ablation import (
    deterministic_descending_indices,
    diversity_select,
)


def test_descending_score_uses_candidate_id_as_explicit_secondary_key():
    ids = ["z", "a", "m", "b"]
    scores = np.array([0.5, 0.5, 0.8, 0.5])

    indices = deterministic_descending_indices(ids, scores)

    assert [ids[index] for index in indices] == ["m", "a", "b", "z"]


def test_deterministic_order_is_independent_of_input_archive_order():
    ids_a = ["z", "a", "m"]
    ids_b = ["m", "z", "a"]
    scores = {"z": 0.5, "a": 0.5, "m": 0.8}

    ranked_a = [ids_a[index] for index in deterministic_descending_indices(ids_a, [scores[x] for x in ids_a])]
    ranked_b = [ids_b[index] for index in deterministic_descending_indices(ids_b, [scores[x] for x in ids_b])]

    assert ranked_a == ranked_b == ["m", "a", "z"]


def test_diversity_selection_uses_candidate_id_when_dynamic_scores_tie():
    ids = ["z", "a", "m"]
    p_hit = np.array([0.5, 0.5, 0.1])
    uncertainty = np.zeros(3)
    groups = ["g1", "g2", "g3"]
    similarity = np.eye(3)

    selected, *_ = diversity_select(
        ids,
        p_hit,
        uncertainty,
        groups,
        similarity,
        batch_size=2,
        prefilter_multiplier=2,
        alpha=0.0,
        beta=0.0,
        gamma=0.0,
    )

    assert [ids[index] for index in selected] == ["a", "z"]


def test_deterministic_ranking_rejects_nan_scores():
    try:
        deterministic_descending_indices(["a", "b"], [0.2, np.nan])
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("NaN score was accepted")
