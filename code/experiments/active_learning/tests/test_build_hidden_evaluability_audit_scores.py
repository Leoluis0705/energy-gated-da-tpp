import pandas as pd

from analysis.build_hidden_evaluability_audit_scores import build_hidden_scores


def test_build_hidden_scores_combines_prospective_and_selected_model_oof(tmp_path):
    candidates = tmp_path / "candidate_scores.csv"
    oof = tmp_path / "oof.csv"
    pd.DataFrame(
        [
            {
                "candidate_id": "new_a",
                "p_dft_evaluable": 0.7,
                "dft_evaluability_model": "chosen_model",
            },
            {
                "candidate_id": "new_b",
                "p_dft_evaluable": 0.2,
                "dft_evaluability_model": "chosen_model",
            },
        ]
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        [
            {"candidate_id": "historic_a", "model_name": "chosen_model", "probability": 0.6},
            {"candidate_id": "historic_b", "model_name": "chosen_model", "probability": 0.4},
            {"candidate_id": "historic_a", "model_name": "other_model", "probability": 0.9},
        ]
    ).to_csv(oof, index=False)

    combined = build_hidden_scores(candidates, oof)

    assert combined["candidate_id"].tolist() == ["historic_a", "historic_b", "new_a", "new_b"]
    assert combined.set_index("candidate_id").at["historic_a", "prediction_role"] == "selected_model_oof"
    assert combined.set_index("candidate_id").at["new_a", "prediction_role"] == "prospective_unlabeled"
    assert combined["candidate_id"].is_unique


def test_build_hidden_scores_rejects_candidate_oof_overlap(tmp_path):
    candidates = tmp_path / "candidate_scores.csv"
    oof = tmp_path / "oof.csv"
    pd.DataFrame(
        [{"candidate_id": "same", "p_dft_evaluable": 0.7, "dft_evaluability_model": "chosen"}]
    ).to_csv(candidates, index=False)
    pd.DataFrame(
        [{"candidate_id": "same", "model_name": "chosen", "probability": 0.6}]
    ).to_csv(oof, index=False)

    try:
        build_hidden_scores(candidates, oof)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("candidate/OOF overlap was accepted")
