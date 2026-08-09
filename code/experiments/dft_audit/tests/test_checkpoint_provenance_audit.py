import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch

from analysis.audit_checkpoint_provenance import (
    compare_first_round_predictions,
    exact_cif_hash_overlaps,
    extract_checkpoint_metadata,
    rank_interval_scores,
)


def test_extract_checkpoint_metadata_preserves_normalizer_and_architecture(tmp_path):
    checkpoint = {
        "epoch": 20,
        "best_mae_error": 0.123,
        "normalizer": {"mean": torch.tensor(-2.5), "std": torch.tensor(0.8)},
        "args": {
            "task": "regression",
            "data_options": [r"D:\CGCNN\mp_formation_clean"],
            "train_ratio": 0.8,
            "val_ratio": 0.2,
            "test_ratio": 0.0,
            "atom_fea_len": 64,
            "h_fea_len": 128,
            "n_conv": 3,
            "n_h": 1,
        },
        "state_dict": {"embedding.weight": torch.ones(2, 2)},
    }
    path = tmp_path / "checkpoint.pth.tar"
    torch.save(checkpoint, path)

    metadata = extract_checkpoint_metadata(path)

    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["normalizer_mean"] == pytest.approx(-2.5)
    assert metadata["normalizer_scale"] == pytest.approx(0.8)
    assert metadata["train_ratio"] == 0.8
    assert metadata["model_architecture"] == "CGCNN(64,3,128,1)"
    assert metadata["state_dict_layer_count"] == 1


def test_exact_cif_hash_overlap_is_content_based_not_filename_based(tmp_path):
    train = tmp_path / "train"
    pool = tmp_path / "pool"
    train.mkdir()
    pool.mkdir()
    (train / "mp-1.cif").write_text("same structure text", encoding="utf-8")
    (train / "mp-2.cif").write_text("training only", encoding="utf-8")
    (pool / "candidate-a.cif").write_text("same structure text", encoding="utf-8")
    (pool / "mp-2.cif").write_text("different despite filename", encoding="utf-8")

    overlaps = exact_cif_hash_overlaps(train, pool)

    assert overlaps.to_dict("records") == [
        {
            "training_candidate_id": "mp-1",
            "pool_candidate_id": "candidate-a",
            "cif_sha256": hashlib.sha256(b"same structure text").hexdigest(),
        }
    ]


def test_rank_interval_scores_uses_explicit_candidate_id_for_exact_ties():
    ids = ["candidate-z", "candidate-a", "candidate-m"]
    scores = [0.5, 0.5, 0.7]

    ranks = rank_interval_scores(ids, scores)

    assert ranks == {"candidate-m": 1, "candidate-a": 2, "candidate-z": 3}


def test_compare_first_round_predictions_is_candidate_wise_and_tolerance_aware():
    reproduced = pd.DataFrame(
        {
            "candidate_id": ["b", "a"],
            "mu_eV": [-2.0, -2.1],
            "sigma_eV": [0.4, 0.5],
            "interval_hit_score": [0.2, 0.3],
            "rank": [2, 1],
        }
    )
    archived = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "mu_eV": [-2.10000001, -2.0],
            "sigma_eV": [0.5, 0.40000001],
            "interval_hit_score": [0.30000001, 0.2],
            "rank": [1, 2],
        }
    )

    compared = compare_first_round_predictions(reproduced, archived, atol=1e-6)

    assert compared["candidate_id"].tolist() == ["a", "b"]
    assert compared["within_tolerance"].all()
    assert compared["rank_matches"].all()


def test_compare_first_round_predictions_rejects_missing_candidate():
    reproduced = pd.DataFrame(
        {"candidate_id": ["a"], "mu_eV": [0], "sigma_eV": [0], "interval_hit_score": [0], "rank": [1]}
    )
    archived = pd.DataFrame(
        {"candidate_id": ["b"], "mu_eV": [0], "sigma_eV": [0], "interval_hit_score": [0], "rank": [1]}
    )

    with pytest.raises(ValueError, match="candidate ID sets differ"):
        compare_first_round_predictions(reproduced, archived, atol=1e-6)
