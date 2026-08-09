import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FORMAL_PROJECT = ROOT / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/project"
sys.path.append(str(FORMAL_PROJECT))

from active_learning_energy_gate_ablation import choose_similarity
from analysis.audit_formal_selection import selection_score, summarize_similarity_values


def test_actual_nonnegative_l2_normalized_embeddings_bound_cosine_to_unit_interval():
    ids = ["a", "b", "zero"]
    features = {
        "a": np.array([1.0, 2.0, 0.0]),
        "b": np.array([2.0, 0.0, 1.0]),
        "zero": np.zeros(3),
    }

    normalized, mode = choose_similarity(ids, features, ["g1", "g2", "g3"])
    similarity = normalized @ normalized.T

    assert mode == "cgcnn_embedding"
    assert np.all(similarity >= 0.0)
    assert np.all(similarity <= 1.0 + 1e-12)
    assert similarity[0, 0] == pytest.approx(1.0)
    assert similarity[2, 2] == 0.0


def test_composition_fallback_is_also_nonnegative_and_l2_normalized():
    normalized, mode = choose_similarity(
        ["a", "b", "c"], {}, ["Li-O", "Li-Mn-O", "Cr-Li-O"]
    )
    similarity = normalized @ normalized.T

    assert mode == "composition"
    assert np.all(similarity >= 0.0)
    assert np.all(similarity <= 1.0 + 1e-12)


def test_hypothetical_negative_cosine_would_increase_current_equation_score():
    zero = selection_score(
        p_hit=0.4,
        uncertainty=0.5,
        max_similarity=0.0,
        group_penalty=0.0,
        alpha=0.1,
        beta=0.2,
        gamma=0.05,
    )
    negative = selection_score(
        p_hit=0.4,
        uncertainty=0.5,
        max_similarity=-0.5,
        group_penalty=0.0,
        alpha=0.1,
        beta=0.2,
        gamma=0.05,
    )

    assert negative - zero == pytest.approx(0.1)


def test_similarity_summary_counts_signs_and_quantiles():
    summary = summarize_similarity_values(np.array([-0.2, 0.0, 0.1, 0.9]))

    assert summary["n"] == 4
    assert summary["negative_count"] == 1
    assert summary["zero_count"] == 1
    assert summary["positive_count"] == 2
    assert summary["minimum"] == -0.2
    assert summary["maximum"] == 0.9
