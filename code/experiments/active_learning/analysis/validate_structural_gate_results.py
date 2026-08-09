"""Independent validation of the structural-group feasibility experiment.

This script deliberately reconstructs checkpoints and normalized AUTC from
candidate-level acquisition histories rather than trusting the summary tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHODS = {
    "energy_gated_da_tpp",
    "gradient_norm_hybrid",
    "predicted_target_greedy",
    "structural_group_gate",
    "structural_group_gate_q95",
}
SEEDS = {111, 112, 113, 114, 115}
CHECKPOINTS = (80, 160, 240, 320)


def normalized_autc_from_history(history: pd.DataFrame, total_targets: int, budget: int) -> float:
    ordered = history.sort_values(["iteration"], kind="stable").reset_index(drop=True)
    batches = (
        ordered.groupby("iteration", sort=True)
        .agg(queries=("id", "size"), hits=("target_label", "sum"))
        .reset_index(drop=True)
    )
    batches["queries"] = batches["queries"].cumsum()
    batches["hits"] = batches["hits"].astype(int).cumsum()
    area = 0.0
    previous_queries = 0
    previous_hits = 0
    for row in batches.itertuples(index=False):
        queries = int(row.queries)
        hits = int(row.hits)
        if queries > budget:
            break
        area += (queries - previous_queries) * previous_hits
        previous_queries = queries
        previous_hits = int(hits)
    if previous_queries < budget:
        area += (budget - previous_queries) * previous_hits
    return area / float(max(1, total_targets * budget))


def sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--hidden-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    initial_rows: list[dict[str, object]] = []
    q95_violations: list[dict[str, object]] = []

    run_metric_paths = sorted(args.results.glob("*/*/seed_*/run_metrics.csv"))
    for metrics_path in run_metric_paths:
        task = metrics_path.parents[2].name
        method = metrics_path.parents[1].name
        seed = int(metrics_path.parent.name.removeprefix("seed_"))
        run_dir = metrics_path.parent
        metrics = pd.read_csv(metrics_path).iloc[0]
        history = pd.read_csv(run_dir / "al_history.csv")
        init = json.loads((run_dir / "initialization_manifest.json").read_text(encoding="utf-8"))

        total_targets = int(metrics["total_target_count"])
        budget = int(metrics["budget"])
        reconstructed_autc = normalized_autc_from_history(history, total_targets, budget)
        ordered = history.sort_values(["iteration"], kind="stable").reset_index(drop=True)
        reconstructed = {cp: int(ordered.iloc[:cp]["target_label"].astype(int).sum()) for cp in CHECKPOINTS}

        checkpoint_values: dict[str, object] = {}
        for cp in CHECKPOINTS:
            checkpoint_values[f"r{cp}_reported"] = int(metrics[f"recovery_at_{cp}"])
            checkpoint_values[f"r{cp}_reconstructed"] = reconstructed[cp]
            checkpoint_values[f"r{cp}_match"] = int(metrics[f"recovery_at_{cp}"]) == reconstructed[cp]

        run_rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "rows": len(history),
                "target_total": total_targets,
                "autc_reported": float(metrics["AUTC"]),
                "autc_reconstructed": reconstructed_autc,
                "autc_abs_error": abs(float(metrics["AUTC"]) - reconstructed_autc),
                **checkpoint_values,
                "candidate_sequence_reported": str(metrics["candidate_sequence_sha256"]),
                "candidate_sequence_line_sha256": sha256_ids(ordered["id"].astype(str).tolist()),
                "initial_candidate_order_sha256": str(init["candidate_order_sha256"]),
                "initial_ids_json": json.dumps(init["candidate_ids"], ensure_ascii=False),
                "status_done": json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "DONE",
            }
        )
        initial_rows.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "candidate_order_sha256": init["candidate_order_sha256"],
                "initial_ids_json": json.dumps(init["candidate_ids"], ensure_ascii=False),
            }
        )

        if method == "structural_group_gate_q95":
            trace_path = next(run_dir.glob("mode_trace_*.csv"))
            trace = pd.read_csv(trace_path)
            for row in trace.itertuples(index=False):
                direct = float(row.direct_batch_p_hit_sum)
                proposed = float(row.proposed_batch_p_hit_sum)
                selected = float(row.selected_batch_p_hit_sum)
                fallback = bool(int(row.quality_safeguard_fallback))
                threshold = 0.95 * direct
                threshold_ok = proposed + 1e-12 >= threshold
                selected_expected = direct if fallback else proposed
                valid = fallback == (not threshold_ok) and abs(selected - selected_expected) <= 1e-9
                if not valid:
                    q95_violations.append(
                        {
                            "task": task,
                            "seed": seed,
                            "iteration": int(row.iteration),
                            "direct": direct,
                            "proposed": proposed,
                            "selected": selected,
                            "fallback": fallback,
                        }
                    )

    runs = pd.DataFrame(run_rows)
    initials = pd.DataFrame(initial_rows)
    runs.to_csv(args.output / "independent_run_checks.csv", index=False)

    init_within_seed = (
        initials.groupby(["task", "seed"])[["candidate_order_sha256", "initial_ids_json"]]
        .nunique()
        .reset_index()
    )
    init_across_seeds = initials.groupby("task")["initial_ids_json"].nunique().to_dict()
    sequence_uniqueness = (
        runs.groupby(["task", "method"])["candidate_sequence_reported"].nunique().reset_index()
    )
    init_within_seed.to_csv(args.output / "initial_set_consistency.csv", index=False)
    sequence_uniqueness.to_csv(args.output / "sequence_independence.csv", index=False)
    pd.DataFrame(q95_violations).to_csv(args.output / "q95_violations.csv", index=False)

    oracle = pd.read_csv(args.oracle)
    task_bounds = {"mn": (-2.1, -1.9), "mg": (-2.3, -2.1)}
    composition_rows: list[dict[str, object]] = []
    for task, (low, high) in task_bounds.items():
        targets = oracle[oracle["formation_energy"].between(low, high, inclusive="both")]
        counts = targets["m_element"].value_counts()
        for element, count in counts.items():
            composition_rows.append(
                {
                    "task": task,
                    "target_low": low,
                    "target_high": high,
                    "element": element,
                    "target_count": int(count),
                    "target_share": float(count / len(targets)),
                }
            )
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(args.output / "target_composition.csv", index=False)

    hidden = pd.read_csv(args.hidden_scores)
    target_ids: set[str] = set()
    for low, high in task_bounds.values():
        target_ids.update(
            oracle.loc[oracle["formation_energy"].between(low, high, inclusive="both"), "candidate_id"].astype(str)
        )
    hidden_ids = set(hidden["candidate_id"].astype(str))

    expected_pairs = {(task, method, seed) for task in {"mn", "mg"} for method in METHODS for seed in SEEDS}
    observed_pairs = set(zip(runs["task"], runs["method"], runs["seed"], strict=False))
    checks = {
        "run_count": int(len(runs)),
        "expected_run_count": 50,
        "unique_task_method_seed": int(len(observed_pairs)),
        "missing_runs": sorted([list(value) for value in expected_pairs - observed_pairs]),
        "unexpected_runs": sorted([list(value) for value in observed_pairs - expected_pairs]),
        "all_status_done": bool(runs["status_done"].all()),
        "all_histories_320_rows": bool((runs["rows"] == 320).all()),
        "max_autc_abs_error": float(runs["autc_abs_error"].max()),
        "all_checkpoint_values_match": bool(runs[[f"r{cp}_match" for cp in CHECKPOINTS]].all().all()),
        "same_initial_set_within_task_seed": bool(
            (init_within_seed[["candidate_order_sha256", "initial_ids_json"]] == 1).all().all()
        ),
        "unique_initial_sets_per_task": {key: int(value) for key, value in init_across_seeds.items()},
        "all_method_task_sequences_differ_across_five_seeds": bool(
            (sequence_uniqueness["candidate_sequence_reported"] == 5).all()
        ),
        "q95_trace_violations": int(len(q95_violations)),
        "hidden_score_rows": int(len(hidden)),
        "hidden_score_roles": {key: int(value) for key, value in hidden["prediction_role"].value_counts().items()},
        "hidden_score_range": [float(hidden["p_dft_evaluable"].min()), float(hidden["p_dft_evaluable"].max())],
        "target_ids_missing_hidden_score": sorted(target_ids - hidden_ids),
        "target_composition": composition.to_dict(orient="records"),
    }
    (args.output / "independent_validation.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    hard_failures = [
        checks["run_count"] != 50,
        checks["missing_runs"] != [],
        not checks["all_status_done"],
        not checks["all_histories_320_rows"],
        checks["max_autc_abs_error"] > 1e-12,
        not checks["all_checkpoint_values_match"],
        not checks["same_initial_set_within_task_seed"],
        not checks["all_method_task_sequences_differ_across_five_seeds"],
        checks["q95_trace_violations"] != 0,
        checks["target_ids_missing_hidden_score"] != [],
    ]
    if any(hard_failures):
        raise SystemExit("Independent validation failed; inspect independent_validation.json")


if __name__ == "__main__":
    main()
