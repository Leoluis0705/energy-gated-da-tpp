"""Combine prospective and selected-model OOF DFT-evaluability predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_hidden_scores(candidate_path: Path, oof_path: Path) -> pd.DataFrame:
    candidates = pd.read_csv(candidate_path)
    oof = pd.read_csv(oof_path)
    models = candidates["dft_evaluability_model"].dropna().astype(str).unique().tolist()
    if len(models) != 1:
        raise ValueError("candidate scores must identify exactly one evaluability model")
    selected_model = models[0]
    prospective = candidates[["candidate_id", "p_dft_evaluable"]].copy()
    prospective["prediction_role"] = "prospective_unlabeled"
    historic = oof[oof["model_name"].astype(str).eq(selected_model)][
        ["candidate_id", "probability"]
    ].rename(columns={"probability": "p_dft_evaluable"})
    historic["prediction_role"] = "selected_model_oof"
    overlap = set(prospective["candidate_id"]).intersection(historic["candidate_id"])
    if overlap:
        raise ValueError("candidate and selected-model OOF scores overlap")
    combined = pd.concat([historic, prospective], ignore_index=True)
    if combined["candidate_id"].duplicated().any():
        raise ValueError("combined hidden-score candidate IDs must be unique")
    combined["p_dft_evaluable"] = pd.to_numeric(
        combined["p_dft_evaluable"], errors="raise"
    )
    if not combined["p_dft_evaluable"].between(0, 1, inclusive="both").all():
        raise ValueError("hidden evaluability probabilities must lie in [0, 1]")
    return combined.sort_values("candidate_id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-scores", type=Path, required=True)
    parser.add_argument("--oof-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    combined = build_hidden_scores(args.candidate_scores, args.oof_scores)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False, lineterminator="\n")
    print(f"wrote {len(combined)} hidden evaluability predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
