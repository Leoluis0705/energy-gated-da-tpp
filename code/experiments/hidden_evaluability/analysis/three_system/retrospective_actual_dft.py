"""Retrospective replay using only observed DFT outcomes and recorded rounds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_hidden_score_table(
    full_scores: pd.DataFrame,
    oof_scores: pd.DataFrame,
    *,
    selected_model: str,
) -> pd.DataFrame:
    """Build a post-selection evaluator, using OOF scores for training rows."""
    full_required = {"candidate_id", "p_dft_evaluable"}
    oof_required = {"candidate_id", "model_name", "probability"}
    full_missing = full_required - set(full_scores.columns)
    oof_missing = oof_required - set(oof_scores.columns)
    if full_missing:
        raise ValueError(f"full score table is missing: {sorted(full_missing)}")
    if oof_missing:
        raise ValueError(f"OOF score table is missing: {sorted(oof_missing)}")

    full = full_scores.loc[:, ["candidate_id", "p_dft_evaluable"]].copy()
    full = full.rename(
        columns={"p_dft_evaluable": "hidden_p_dft_evaluable"}
    )
    full["score_source"] = "full_model_unseen_pool"

    oof = oof_scores.loc[
        oof_scores["model_name"].astype(str).eq(str(selected_model)),
        ["candidate_id", "probability"],
    ].copy()
    if oof.empty:
        raise ValueError(f"no OOF rows found for model {selected_model}")
    oof = oof.rename(columns={"probability": "hidden_p_dft_evaluable"})
    oof["score_source"] = f"{selected_model}_OOF"

    full = full.loc[~full["candidate_id"].isin(oof["candidate_id"])]
    combined = pd.concat([full, oof], ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise ValueError("hidden score table contains duplicate candidate IDs")
    probabilities = pd.to_numeric(
        combined["hidden_p_dft_evaluable"], errors="raise"
    )
    if not probabilities.between(0.0, 1.0, inclusive="both").all():
        raise ValueError("hidden DFT-evaluability probabilities must be in [0, 1]")
    combined["hidden_p_dft_evaluable"] = probabilities
    return combined.sort_values("candidate_id", kind="mergesort").reset_index(
        drop=True
    )


def build_hidden_evaluability_overlay(
    histories: pd.DataFrame,
    hidden_scores: pd.DataFrame,
    *,
    checkpoints: tuple[int, ...],
    hard_label_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen target-search histories with a hidden DFT evaluator."""
    history_required = {
        "method",
        "seed",
        "query",
        "candidate_id",
        "target_label",
    }
    score_required = {
        "candidate_id",
        "hidden_p_dft_evaluable",
        "score_source",
    }
    missing_history = history_required - set(histories.columns)
    missing_scores = score_required - set(hidden_scores.columns)
    if missing_history:
        raise ValueError(f"history table is missing: {sorted(missing_history)}")
    if missing_scores:
        raise ValueError(
            f"hidden evaluator table is missing: {sorted(missing_scores)}"
        )
    leaked = {
        "current_p_eval",
        "p_dft_evaluable",
        "predicted_p_dft_evaluable",
        "pseudo_dft_evaluable",
        "joint_qualified_probability",
    } & set(histories.columns)
    if leaked:
        raise ValueError(
            "DFT-evaluability leakage detected in acquisition history: "
            f"{sorted(leaked)}"
        )
    if not 0.0 <= hard_label_threshold <= 1.0:
        raise ValueError("hard_label_threshold must be in [0, 1]")

    detail = histories.copy()
    detail["query"] = pd.to_numeric(detail["query"], errors="raise").astype(int)
    detail["target_label"] = pd.to_numeric(
        detail["target_label"], errors="raise"
    ).astype(int)
    if not detail["target_label"].isin([0, 1]).all():
        raise ValueError("target_label must be binary")
    detail = detail.merge(
        hidden_scores.loc[:, sorted(score_required)],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    missing = detail.loc[
        detail["target_label"].eq(1)
        & detail["hidden_p_dft_evaluable"].isna(),
        "candidate_id",
    ].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "hidden evaluator is missing target-candidate scores: "
            + ", ".join(missing.astype(str).head(10))
        )
    detail["ML_labeled_DFT_evaluable"] = (
        detail["hidden_p_dft_evaluable"].ge(hard_label_threshold).astype(int)
    )
    detail["evidence_tier"] = (
        "post_selection_hidden_ML_evaluator_not_observed_DFT"
    )
    detail = detail.sort_values(
        ["method", "seed", "query"], kind="mergesort"
    ).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for (method, seed), run in detail.groupby(
        ["method", "seed"], sort=True
    ):
        if run["query"].duplicated().any():
            raise ValueError(f"duplicate query positions for {method}, seed {seed}")
        for checkpoint in checkpoints:
            queried = run.loc[run["query"].le(int(checkpoint))]
            targets = queried.loc[queried["target_label"].eq(1)]
            rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "checkpoint": int(checkpoint),
                    "target_hits": int(len(targets)),
                    "expected_DFT_evaluable_target_hits": float(
                        targets["hidden_p_dft_evaluable"].sum()
                    ),
                    "ML_labeled_DFT_evaluable_target_hits": int(
                        targets["ML_labeled_DFT_evaluable"].sum()
                    ),
                    "mean_hidden_DFT_evaluable_probability_among_targets": (
                        float(targets["hidden_p_dft_evaluable"].mean())
                        if len(targets)
                        else np.nan
                    ),
                    "ML_labeled_DFT_evaluable_rate_among_targets": (
                        float(targets["ML_labeled_DFT_evaluable"].mean())
                        if len(targets)
                        else np.nan
                    ),
                    "evidence_tier": (
                        "post_selection_hidden_ML_evaluator_not_observed_DFT"
                    ),
                }
            )
    return detail, pd.DataFrame(rows)


def build_actual_replay(
    labels: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    interval: tuple[float, float] = (-2.3, -1.5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build Gate/Greedy curves for candidates with observed DFT and both rounds."""
    label_columns = [
        "candidate_id",
        "dft_evaluable",
        "dft_formation_energy_eV_atom",
    ]
    manifest_columns = [
        "candidate_id",
        "pilot_or_new",
        "Gate_round",
        "Greedy_round",
    ]
    candidates = labels.loc[:, label_columns].merge(
        manifest.loc[:, manifest_columns],
        on="candidate_id",
        how="inner",
        validate="one_to_one",
    )
    candidates = candidates.loc[
        candidates["pilot_or_new"].eq("new")
        & candidates["Gate_round"].notna()
        & candidates["Greedy_round"].notna()
    ].copy()
    candidates["Gate_round"] = candidates["Gate_round"].astype(int)
    candidates["Greedy_round"] = candidates["Greedy_round"].astype(int)
    candidates["dft_evaluable"] = candidates["dft_evaluable"].astype(int)
    candidates["dft_interval_qualified"] = (
        candidates["dft_evaluable"].eq(1)
        & candidates["dft_formation_energy_eV_atom"].between(
            interval[0], interval[1], inclusive="both"
        )
    ).astype(int)
    candidates["Gate_lead_rounds"] = (
        candidates["Greedy_round"] - candidates["Gate_round"]
    )
    candidates["evidence_tier"] = "observed_DFT_retrospective_selected_subset"
    candidates = candidates.sort_values("candidate_id", kind="mergesort")

    max_round = int(
        max(candidates["Gate_round"].max(), candidates["Greedy_round"].max())
    )
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for policy, round_column in (
        ("Gate", "Gate_round"),
        ("Greedy", "Greedy_round"),
    ):
        policy_curve = []
        for round_number in range(0, max_round + 1):
            observed = candidates.loc[
                candidates[round_column].le(round_number)
            ]
            policy_curve.append(
                {
                    "policy": policy,
                    "round": round_number,
                    "cumulative_DFT_attempts": len(observed),
                    "cumulative_DFT_evaluable": int(
                        observed["dft_evaluable"].sum()
                    ),
                    "cumulative_DFT_interval_qualified": int(
                        observed["dft_interval_qualified"].sum()
                    ),
                    "cumulative_DFT_failures": int(
                        observed["dft_evaluable"].eq(0).sum()
                    ),
                    "evidence_tier": (
                        "observed_DFT_retrospective_selected_subset"
                    ),
                }
            )
        curve_rows.extend(policy_curve)
        curve = pd.DataFrame(policy_curve)
        y = curve["cumulative_DFT_evaluable"].to_numpy(dtype=float)
        x = curve["round"].to_numpy(dtype=float)
        auc = float(np.trapezoid(y, x))
        final_count = int(y[-1])
        first = curve.loc[curve["cumulative_DFT_evaluable"].gt(0), "round"]
        summary_rows.append(
            {
                "policy": policy,
                "n_observed_candidates": len(candidates),
                "final_DFT_evaluable": final_count,
                "final_DFT_failures": int(
                    candidates["dft_evaluable"].eq(0).sum()
                ),
                "final_interval_qualified": int(
                    candidates["dft_interval_qualified"].sum()
                ),
                "time_to_first_evaluable_round": (
                    int(first.iloc[0]) if not first.empty else np.nan
                ),
                "DFT_evaluable_round_AUC": auc,
                "normalized_DFT_evaluable_round_AUC": (
                    auc / (max_round * final_count)
                    if max_round > 0 and final_count > 0
                    else np.nan
                ),
                "evidence_tier": (
                    "observed_DFT_retrospective_selected_subset"
                ),
            }
        )
    return (
        candidates.reset_index(drop=True),
        pd.DataFrame(curve_rows),
        pd.DataFrame(summary_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates, curve, summary = build_actual_replay(
        pd.read_csv(args.labels),
        pd.read_csv(args.manifest),
    )
    candidates.to_csv(output / "actual_dft_candidate_subset.csv", index=False)
    curve.to_csv(output / "actual_dft_gate_greedy_curve.csv", index=False)
    summary.to_csv(output / "actual_dft_gate_greedy_summary.csv", index=False)


if __name__ == "__main__":
    main()
