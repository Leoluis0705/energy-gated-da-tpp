"""Recompute fixed-horizon AUTC metrics from immutable formal histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.audit_common import (
    candidate_sequence_hash,
    left_continuous_autc,
    recovery_at,
    sha256_file,
)
from analysis.recompute_statistics import build_round_trajectory


def partial_metrics_from_history(
    history: pd.DataFrame,
    *,
    batch_size: int,
    total_targets: int,
    full_budget: int,
    horizons: Iterable[int] = (80, 160, 240, 320),
) -> dict[str, float | int]:
    if len(history) != int(full_budget):
        raise ValueError(
            f"history must contain the complete full budget ({full_budget}), found {len(history)}"
        )
    if history["id"].astype(str).duplicated().any():
        raise ValueError("history contains duplicate candidate IDs")
    requested = tuple(int(horizon) for horizon in horizons)
    if any(horizon <= 0 or horizon > int(full_budget) for horizon in requested):
        raise ValueError("each horizon must be positive and no larger than the full budget")
    trajectory = build_round_trajectory(
        history, batch_size=int(batch_size), budget=int(full_budget)
    )
    query_counts = trajectory["oracle_evaluations"].to_numpy(int)
    recoveries = trajectory["cumulative_target_count"].to_numpy(int)
    result: dict[str, float | int] = {
        "full_budget_AUTC": left_continuous_autc(
            query_counts, recoveries, int(total_targets), int(full_budget)
        ),
        "full_budget_recovery": int(recoveries[-1]),
    }
    for horizon in requested:
        result[f"AUTC_at_{horizon}"] = left_continuous_autc(
            query_counts, recoveries, int(total_targets), horizon
        )
        result[f"Recovery_at_{horizon}"] = recovery_at(
            query_counts, recoveries, horizon
        )
    return result


def recompute_formal_partial_metrics(formal_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for config_path in sorted(formal_root.rglob("run_config.json")):
        run_dir = config_path.parent
        relative = run_dir.relative_to(formal_root)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        history_path = run_dir / "al_history.csv"
        metrics_path = run_dir / "run_metrics.csv"
        if not history_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"missing history or metrics in {run_dir}")
        history = pd.read_csv(history_path)
        budget = int(config["budget"])
        available_horizons = tuple(h for h in (80, 160, 240, 320) if h <= budget)
        partial = partial_metrics_from_history(
            history,
            batch_size=int(config["batch_size"]),
            total_targets=int(config["target_count"]),
            full_budget=budget,
            horizons=available_horizons,
        )
        reported = pd.read_csv(metrics_path).iloc[0]
        reported_full = float(reported["AUTC"])
        delta = reported_full - float(partial["full_budget_AUTC"])
        if abs(delta) > 1e-12:
            raise ValueError(f"reported AUTC differs from raw-history recomputation: {run_dir}")
        row = {
            "formal_stage": relative.parts[0],
            "dataset": str(config["name"]),
            "method": str(config["method"]),
            "group_key": str(
                config.get("group_key_mode", config.get("group_key_construction", "unavailable"))
            ),
            "seed": int(config["seed"]),
            "K": int(config["mc_passes"]),
            "budget": budget,
            "batch_size": int(config["batch_size"]),
            "total_target_count": int(config["target_count"]),
            **partial,
            "reported_full_budget_AUTC": reported_full,
            "reported_minus_recomputed_full_AUTC": delta,
            "candidate_sequence_sha256": candidate_sequence_hash(
                history["id"].astype(str).tolist()
            ),
            "history_sha256": sha256_file(history_path),
            "run_config_sha256": sha256_file(config_path),
            "source_history_path": history_path.relative_to(formal_root).as_posix(),
        }
        for horizon in (80, 160, 240, 320):
            row.setdefault(f"AUTC_at_{horizon}", pd.NA)
            row.setdefault(f"Recovery_at_{horizon}", pd.NA)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["formal_stage", "dataset", "method", "group_key", "K", "seed"]
    ).reset_index(drop=True)


def reconcile_postcompute(metrics: pd.DataFrame, postcompute_path: Path) -> pd.DataFrame:
    postcompute = pd.read_csv(postcompute_path)
    keys = ["formal_stage", "dataset", "method", "group_key", "seed", "K"]
    reference = postcompute[keys + ["AUTC"]].rename(
        columns={"AUTC": "postcompute_full_budget_AUTC"}
    )
    joined = metrics.merge(reference, on=keys, how="left", validate="one_to_one")
    if joined["postcompute_full_budget_AUTC"].isna().any():
        raise ValueError("post-compute reconciliation is missing formal runs")
    joined["postcompute_minus_recomputed_full_AUTC"] = (
        joined["postcompute_full_budget_AUTC"] - joined["full_budget_AUTC"]
    )
    if joined["postcompute_minus_recomputed_full_AUTC"].abs().max() > 1e-12:
        raise ValueError("post-compute full-budget AUTC differs from raw-history recomputation")
    return joined


def _write_without_overwriting_different_content(frame: pd.DataFrame, output: Path) -> None:
    rendered = frame.to_csv(index=False, lineterminator="\n")
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite non-identical evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=root / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/results/final",
    )
    parser.add_argument(
        "--postcompute-metrics",
        type=Path,
        default=root / "results/post_submission_analysis/egdatpp_psfix_v1_20260719T031102Z/gpu/per_seed_metrics.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "results/gpu/partial_budget_metrics.csv",
    )
    args = parser.parse_args()
    metrics = recompute_formal_partial_metrics(args.formal_root)
    reconciled = reconcile_postcompute(metrics, args.postcompute_metrics)
    _write_without_overwriting_different_content(reconciled, args.output)
    print(
        json.dumps(
            {
                "run_count": len(reconciled),
                "maximum_reported_delta": float(
                    reconciled["reported_minus_recomputed_full_AUTC"].abs().max()
                ),
                "maximum_postcompute_delta": float(
                    reconciled["postcompute_minus_recomputed_full_AUTC"].abs().max()
                ),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
