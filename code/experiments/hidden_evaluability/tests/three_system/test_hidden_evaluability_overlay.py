import pandas as pd
import pytest

from analysis.three_system import retrospective_actual_dft as overlay


def test_hidden_overlay_counts_only_target_hits_after_selection():
    """Catches DFT evaluability being counted outside the target-hit subset."""
    assert hasattr(overlay, "build_hidden_evaluability_overlay")
    histories = pd.DataFrame(
        [
            {
                "method": "Gate",
                "seed": 5,
                "query": 1,
                "candidate_id": "target_high",
                "target_label": 1,
            },
            {
                "method": "Gate",
                "seed": 5,
                "query": 2,
                "candidate_id": "non_target",
                "target_label": 0,
            },
            {
                "method": "Gate",
                "seed": 5,
                "query": 3,
                "candidate_id": "target_low",
                "target_label": 1,
            },
        ]
    )
    hidden_scores = pd.DataFrame(
        [
            {
                "candidate_id": "target_high",
                "hidden_p_dft_evaluable": 0.8,
                "score_source": "full_model",
            },
            {
                "candidate_id": "non_target",
                "hidden_p_dft_evaluable": 0.99,
                "score_source": "full_model",
            },
            {
                "candidate_id": "target_low",
                "hidden_p_dft_evaluable": 0.4,
                "score_source": "full_model",
            },
        ]
    )

    detail, summary = overlay.build_hidden_evaluability_overlay(
        histories,
        hidden_scores,
        checkpoints=(3,),
        hard_label_threshold=0.5,
    )

    row = summary.iloc[0]
    assert row["target_hits"] == 2
    assert row["expected_DFT_evaluable_target_hits"] == pytest.approx(1.2)
    assert row["ML_labeled_DFT_evaluable_target_hits"] == 1
    assert row["mean_hidden_DFT_evaluable_probability_among_targets"] == (
        pytest.approx(0.6)
    )
    assert row["ML_labeled_DFT_evaluable_rate_among_targets"] == pytest.approx(
        0.5
    )
    assert detail["candidate_id"].tolist() == [
        "target_high",
        "non_target",
        "target_low",
    ]


def test_hidden_overlay_rejects_evaluability_columns_in_acquisition_history():
    """Catches the hidden DFT evaluator leaking into the acquisition policy."""
    assert hasattr(overlay, "build_hidden_evaluability_overlay")
    histories = pd.DataFrame(
        [
            {
                "method": "Greedy",
                "seed": 5,
                "query": 1,
                "candidate_id": "candidate",
                "target_label": 1,
                "current_p_eval": 0.9,
            }
        ]
    )
    hidden_scores = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "hidden_p_dft_evaluable": 0.9,
                "score_source": "full_model",
            }
        ]
    )

    with pytest.raises(ValueError, match="leak"):
        overlay.build_hidden_evaluability_overlay(
            histories,
            hidden_scores,
            checkpoints=(1,),
        )


def test_hidden_score_table_uses_oof_probability_for_training_candidates():
    """Catches in-sample probabilities being used for DFT-labelled candidates."""
    assert hasattr(overlay, "build_hidden_score_table")
    full_scores = pd.DataFrame(
        [
            {"candidate_id": "historical", "p_dft_evaluable": 0.99},
            {"candidate_id": "prospective", "p_dft_evaluable": 0.70},
        ]
    )
    oof_scores = pd.DataFrame(
        [
            {
                "candidate_id": "historical",
                "model_name": "selected_model",
                "probability": 0.35,
            },
            {
                "candidate_id": "historical",
                "model_name": "other_model",
                "probability": 0.80,
            },
        ]
    )

    result = overlay.build_hidden_score_table(
        full_scores,
        oof_scores,
        selected_model="selected_model",
    ).set_index("candidate_id")

    assert result.at["historical", "hidden_p_dft_evaluable"] == pytest.approx(
        0.35
    )
    assert result.at["historical", "score_source"] == "selected_model_OOF"
    assert result.at["prospective", "hidden_p_dft_evaluable"] == pytest.approx(
        0.70
    )
    assert result.at["prospective", "score_source"] == "full_model_unseen_pool"


def test_hidden_overlay_requires_scores_only_for_target_hits():
    """Non-target rows must not affect the post-selection DFT endpoint."""
    histories = pd.DataFrame(
        [
            {
                "method": "Greedy",
                "seed": 5,
                "query": 1,
                "candidate_id": "target",
                "target_label": 1,
            },
            {
                "method": "Greedy",
                "seed": 5,
                "query": 2,
                "candidate_id": "non_target_without_score",
                "target_label": 0,
            },
        ]
    )
    hidden_scores = pd.DataFrame(
        [
            {
                "candidate_id": "target",
                "hidden_p_dft_evaluable": 0.75,
                "score_source": "full_model",
            }
        ]
    )

    detail, summary = overlay.build_hidden_evaluability_overlay(
        histories,
        hidden_scores,
        checkpoints=(2,),
    )

    assert detail.loc[
        detail["candidate_id"].eq("non_target_without_score"),
        "hidden_p_dft_evaluable",
    ].isna().all()
    assert summary.iloc[0]["expected_DFT_evaluable_target_hits"] == pytest.approx(
        0.75
    )


def test_hidden_overlay_rejects_missing_score_for_target_hit():
    histories = pd.DataFrame(
        [
            {
                "method": "Greedy",
                "seed": 5,
                "query": 1,
                "candidate_id": "target_without_score",
                "target_label": 1,
            }
        ]
    )
    hidden_scores = pd.DataFrame(
        columns=[
            "candidate_id",
            "hidden_p_dft_evaluable",
            "score_source",
        ]
    )

    with pytest.raises(ValueError, match="target"):
        overlay.build_hidden_evaluability_overlay(
            histories,
            hidden_scores,
            checkpoints=(1,),
        )
