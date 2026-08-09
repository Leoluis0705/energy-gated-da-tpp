#!/usr/bin/env python3
"""Aggregate only protocol-compatible corrected paired seeds 5-14."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


METHODS = ("energy_gated_da_tpp", "predicted_distance_greedy")
METHOD_LABELS = {
    "energy_gated_da_tpp": "Energy-Gated DA-TPP",
    "predicted_distance_greedy": "Predicted-Distance Greedy",
}
COLORS = {"energy_gated_da_tpp": "#176B87", "predicted_distance_greedy": "#B45F3A"}


def classify_limo(differences: np.ndarray, ci_low: float, ci_high: float) -> str:
    values = np.asarray(differences, dtype=float)
    positives = int((values > 0).sum())
    negatives = int((values < 0).sum())
    mean = float(values.mean())
    if positives >= 8 and ci_low > 0:
        return "CONSISTENT_ADVANTAGE"
    if mean > 0 and ci_low <= 0 <= ci_high:
        return "SMALL_MEAN_ADVANTAGE"
    if mean < 0 and negatives >= 8:
        return "GREEDY_ADVANTAGE"
    return "COMPARABLE_PERFORMANCE"


def classify_mnoxide(differences: np.ndarray, correction_replacements: int, margin: float = 0.01) -> str:
    values = np.asarray(differences, dtype=float)
    if correction_replacements == 0 and np.all(np.abs(values) <= margin):
        return "DIRECT_FALLBACK_CONFIRMED"
    if float(values.mean()) < -margin and int((values < -margin).sum()) >= 8:
        return "FALLBACK_WITH_DEGRADATION"
    return "FALLBACK_INCONCLUSIVE"


def trajectory_bootstrap_band(values: np.ndarray, samples: int = 20_000, seed: int = 20260713) -> dict:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("trajectory band requires a two-dimensional multi-seed matrix")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, matrix.shape[0], size=(samples, matrix.shape[0]))
    boot = matrix[indices].mean(axis=1)
    return {
        "mean": matrix.mean(axis=0),
        "std": matrix.std(axis=0, ddof=1),
        "ci_low": np.quantile(boot, 0.025, axis=0),
        "ci_high": np.quantile(boot, 0.975, axis=0),
        "n": matrix.shape[0],
    }


def bootstrap_mean_ci(values: np.ndarray, samples: int = 20_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260713)
    boot = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired_stats(rows: pd.DataFrame) -> dict:
    values = rows["paired_AUTC_difference"].to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(values)
    sd = float(values.std(ddof=1))
    effect = float(values.mean() / sd) if sd > 0 else (0.0 if np.allclose(values, 0) else math.nan)
    if wilcoxon is None or np.allclose(values, 0):
        statistic = pvalue = math.nan
    else:
        result = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "n": len(values),
        "gate_mean": float(rows["gate_AUTC"].mean()),
        "gate_sd": float(rows["gate_AUTC"].std(ddof=1)),
        "greedy_mean": float(rows["greedy_AUTC"].mean()),
        "greedy_sd": float(rows["greedy_AUTC"].std(ddof=1)),
        "paired_mean": float(values.mean()),
        "paired_median": float(np.median(values)),
        "ci_low": low,
        "ci_high": high,
        "effect_dz": effect,
        "wins": int((values > 1e-12).sum()),
        "ties": int((np.abs(values) <= 1e-12).sum()),
        "losses": int((values < -1e-12).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_pvalue": pvalue,
    }


def read_run_metrics(root: Path, seeds: range) -> pd.DataFrame:
    rows = []
    for dataset in ("limo", "mnoxide"):
        for method in METHODS:
            for seed in seeds:
                run_dir = root / "runs" / dataset / method / f"seed_{seed}"
                path = run_dir / "run_metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_csv(path)
                if len(frame) != 1:
                    raise RuntimeError(f"expected one metric row in {path}")
                row = frame.iloc[0].to_dict()
                row["source_run_dir"] = str(run_dir)
                rows.append(row)
    return pd.DataFrame(rows)


def build_paired_table(metrics: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[metrics["method"] == "energy_gated_da_tpp"]
    greedy = metrics[metrics["method"] == "predicted_distance_greedy"]
    joined = gate.merge(greedy, on=["dataset", "seed"], suffixes=("_gate", "_greedy"), validate="one_to_one")
    result = joined[["dataset", "seed"]].copy()
    result["gate_AUTC"] = joined["AUTC_gate"]
    result["greedy_AUTC"] = joined["AUTC_greedy"]
    result["paired_AUTC_difference"] = result["gate_AUTC"] - result["greedy_AUTC"]
    for checkpoint in (80, 160, 240, 320):
        result[f"gate_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_gate"]
        result[f"greedy_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_greedy"]
        result[f"paired_recovery_difference_at_{checkpoint}"] = (
            result[f"gate_recovery_at_{checkpoint}"] - result[f"greedy_recovery_at_{checkpoint}"]
        )
    result["gate_correction_rounds"] = joined["correction_rounds_gate"]
    result["gate_mean_unique_groups_per_batch"] = joined["mean_unique_groups_per_batch_gate"]
    result["greedy_mean_unique_groups_per_batch"] = joined["mean_unique_groups_per_batch_greedy"]
    result["gate_group_repetition_rate"] = joined["mean_group_repetition_rate_gate"]
    result["greedy_group_repetition_rate"] = joined["mean_group_repetition_rate_greedy"]
    result["gate_correction_replacements"] = joined["total_correction_replacements_gate"]
    result["gate_correction_target_gain"] = joined["total_correction_target_gain_gate"]
    return result.sort_values(["dataset", "seed"])


def build_effect_tables(metrics: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    limo = paired[paired["dataset"] == "limo"].copy()
    limo = limo[
        [
            "seed",
            "gate_correction_rounds",
            "gate_correction_replacements",
            "gate_correction_target_gain",
            "gate_mean_unique_groups_per_batch",
            "greedy_mean_unique_groups_per_batch",
            "gate_group_repetition_rate",
            "greedy_group_repetition_rate",
            "gate_AUTC",
            "greedy_AUTC",
            "paired_AUTC_difference",
        ]
    ]
    mn = paired[paired["dataset"] == "mnoxide"].copy()
    gate_metrics = metrics[(metrics["dataset"] == "mnoxide") & (metrics["method"] == "energy_gated_da_tpp")]
    route = gate_metrics[["seed", "direct_rounds", "correction_rounds", "direct_route_proportion"]]
    mn = mn.merge(route, on="seed", validate="one_to_one")
    mn["within_frozen_0p01_margin"] = mn["paired_AUTC_difference"].abs() <= 0.01
    return limo, mn


def load_trajectories(root_a: Path, root_b: Path) -> pd.DataFrame:
    rows = []
    for root, seeds in ((root_a, range(5, 10)), (root_b, range(10, 15))):
        for dataset in ("limo", "mnoxide"):
            for method in METHODS:
                for seed in seeds:
                    path = root / "runs" / dataset / method / f"seed_{seed}" / "summary.csv"
                    frame = pd.read_csv(path)[
                        ["oracle_evaluations", "cumulative_target_count", "round_target_hits"]
                    ].copy()
                    zero = pd.DataFrame(
                        [{"oracle_evaluations": 0, "cumulative_target_count": 0, "round_target_hits": 0}]
                    )
                    frame = pd.concat([zero, frame], ignore_index=True)
                    frame.insert(0, "seed", seed)
                    frame.insert(0, "method", method)
                    frame.insert(0, "dataset", dataset)
                    rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def build_figure_data_and_plot(trajectories: pd.DataFrame, dataset: str, output_root: Path) -> None:
    figure_dir = output_root / "figures"
    source_dir = output_root / "figure_source_data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    raw = trajectories[trajectories["dataset"] == dataset].copy()
    raw.to_csv(source_dir / f"{dataset}_corrected_recovery_per_seed.csv", index=False)
    summary_rows = []
    fig, ax = plt.subplots(figsize=(6.7, 4.25))
    for method_index, method in enumerate(METHODS):
        subset = raw[raw["method"] == method]
        pivot = subset.pivot(index="seed", columns="oracle_evaluations", values="cumulative_target_count").sort_index(axis=1)
        band = trajectory_bootstrap_band(pivot.to_numpy(dtype=float), seed=20260713 + method_index)
        queries = pivot.columns.to_numpy(dtype=int)
        for query, mean, std, low, high in zip(
            queries, band["mean"], band["std"], band["ci_low"], band["ci_high"]
        ):
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "oracle_evaluations": query,
                    "mean_recovery": mean,
                    "standard_deviation": std,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "seed_count": band["n"],
                }
            )
        ax.plot(queries, band["mean"], color=COLORS[method], linewidth=1.8, label=METHOD_LABELS[method])
        ax.fill_between(queries, band["ci_low"], band["ci_high"], color=COLORS[method], alpha=0.18, linewidth=0)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(source_dir / f"{dataset}_corrected_recovery_mean_band.csv", index=False)
    ax.set_xlabel("Oracle evaluations")
    ax.set_ylabel("Target-Window Candidates Recovered")
    ax.text(0.02, 0.96, "Li-M-O" if dataset == "limo" else "Mn-oxide", transform=ax.transAxes, va="top", fontsize=10)
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    stem = figure_dir / f"{dataset}_corrected_mean_recovery_seeds_5_14"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def stats_text(stats: dict) -> str:
    wilcoxon_text = (
        "unavailable (all differences are zero)"
        if not np.isfinite(stats["wilcoxon_pvalue"])
        else f"W={stats['wilcoxon_statistic']:.3f}, two-sided p={stats['wilcoxon_pvalue']:.6f}"
    )
    return (
        f"Gate AUTC {stats['gate_mean']:.6f} +/- {stats['gate_sd']:.6f}; "
        f"Greedy {stats['greedy_mean']:.6f} +/- {stats['greedy_sd']:.6f}; "
        f"paired mean {stats['paired_mean']:+.6f}, median {stats['paired_median']:+.6f}; "
        f"bootstrap 95% CI [{stats['ci_low']:+.6f}, {stats['ci_high']:+.6f}]; "
        f"Cohen's dz={stats['effect_dz']:.3f}; W/T/L={stats['wins']}/{stats['ties']}/{stats['losses']}; "
        f"Wilcoxon {wilcoxon_text}."
    )


def sum_runtime_hours(roots: list[Path]) -> tuple[float, float, int, int]:
    statuses = []
    wall = 0.0
    for root in roots:
        statuses.extend(json.loads(path.read_text(encoding="utf-8")) for path in root.glob("runs/*/*/seed_*/status.json"))
        completion = json.loads((root / "COMPLETION.json").read_text(encoding="utf-8"))
        wall += float(completion.get("elapsed_seconds", 0.0))
    return (
        sum(float(item.get("elapsed_seconds", 0.0)) for item in statuses) / 3600.0,
        wall / 3600.0,
        sum(item.get("status") == "DONE" for item in statuses),
        sum(item.get("status") == "FAILED" for item in statuses),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrected-root-5-9", required=True)
    parser.add_argument("--corrected-root-10-14", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root_5_9 = Path(args.corrected_root_5_9).resolve()
    root_10_14 = Path(args.corrected_root_10_14).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completion = json.loads((root_10_14 / "COMPLETION.json").read_text(encoding="utf-8"))
    if completion.get("completed") != 20 or completion.get("failed") != 0:
        raise RuntimeError("seeds 10-14 must complete 20/20 with zero failures before aggregation")

    metrics = pd.concat(
        [read_run_metrics(root_5_9, range(5, 10)), read_run_metrics(root_10_14, range(10, 15))],
        ignore_index=True,
    )
    if len(metrics) != 40 or set(metrics["seed"].astype(int)) != set(range(5, 15)):
        raise RuntimeError("corrected aggregate must contain exactly 40 run rows for seeds 5-14")
    metrics.to_csv(output_root / "CORRECTED_SEEDS_5_14_ALL_RESULTS.csv", index=False)
    paired = build_paired_table(metrics)
    paired.to_csv(output_root / "CORRECTED_PAIRED_AUTC_DIFFERENCES.csv", index=False)
    limo_effects, mn_fallback = build_effect_tables(metrics, paired)
    limo_effects.to_csv(output_root / "LIMO_CORRECTION_EFFECTS_SEEDS_5_14.csv", index=False)
    mn_fallback.to_csv(output_root / "MNOXIDE_FALLBACK_ANALYSIS_SEEDS_5_14.csv", index=False)

    trajectories = load_trajectories(root_5_9, root_10_14)
    for dataset in ("limo", "mnoxide"):
        build_figure_data_and_plot(trajectories, dataset, output_root)

    limo_rows = paired[paired["dataset"] == "limo"]
    mn_rows = paired[paired["dataset"] == "mnoxide"]
    limo_stats = paired_stats(limo_rows)
    mn_stats = paired_stats(mn_rows)
    limo_decision = classify_limo(
        limo_rows["paired_AUTC_difference"].to_numpy(), limo_stats["ci_low"], limo_stats["ci_high"]
    )
    mn_decision = classify_mnoxide(
        mn_rows["paired_AUTC_difference"].to_numpy(),
        int(mn_rows["gate_correction_replacements"].sum()),
        0.01,
    )
    runtime_hours, wall_hours, done, failed = sum_runtime_hours([root_5_9, root_10_14])
    new_runtime_hours, new_wall_hours, new_done, new_failed = sum_runtime_hours([root_10_14])

    checkpoint_text_limo = "; ".join(
        f"@{q}: {limo_rows[f'paired_recovery_difference_at_{q}'].mean():+.2f} +/- {limo_rows[f'paired_recovery_difference_at_{q}'].std(ddof=1):.2f}"
        for q in (80, 160, 240, 320)
    )
    checkpoint_text_mn = "; ".join(
        f"@{q}: {mn_rows[f'paired_recovery_difference_at_{q}'].mean():+.2f} +/- {mn_rows[f'paired_recovery_difference_at_{q}'].std(ddof=1):.2f}"
        for q in (80, 160, 240, 320)
    )
    limo_report = f"""# Corrected Seeds 5-14: Li-M-O

Only corrected, protocol-compatible seeds 5-14 are included. Legacy seeds 0-4 are excluded.

{stats_text(limo_stats)}

Mean paired recovery differences: {checkpoint_text_limo}.

Mean correction rounds: {limo_rows['gate_correction_rounds'].mean():.2f}; mean correction replacements: {limo_rows['gate_correction_replacements'].mean():.2f}; mean immediate substitution target gain: {limo_rows['gate_correction_target_gain'].mean():+.2f}.

Decision: `{limo_decision}`.
"""
    (output_root / "CORRECTED_SEEDS_5_14_LIMO_REPORT.md").write_text(limo_report, encoding="utf-8")
    mn_report = f"""# Corrected Seeds 5-14: Mn-Oxide

Only corrected, protocol-compatible seeds 5-14 are included. Legacy seeds 0-4 are excluded.

{stats_text(mn_stats)}

Mean paired recovery differences: {checkpoint_text_mn}.

Direct rounds: {int(mn_rows['gate_correction_rounds'].rsub(20).sum())}/200; correction-labeled rounds: {int(mn_rows['gate_correction_rounds'].sum())}/200; correction replacements: {int(mn_rows['gate_correction_replacements'].sum())}.

Decision: `{mn_decision}` under the frozen +/-0.01 AUTC non-degradation margin.
"""
    (output_root / "CORRECTED_SEEDS_5_14_MNOXIDE_REPORT.md").write_text(mn_report, encoding="utf-8")

    outperformed = limo_decision == "CONSISTENT_ADVANTAGE"
    comparable = mn_decision == "DIRECT_FALLBACK_CONFIRMED"
    final = f"""# Final Corrected Ten-Seed Claim Decision

## Execution

- New seeds 10-14: {new_done}/20 completed, {new_failed} failed; summed trajectory runtime {new_runtime_hours:.2f} GPU-hours; wall time {new_wall_hours:.2f} hours.
- Combined corrected seeds 5-14: {done}/40 completed, {failed} failed; summed trajectory runtime {runtime_hours:.2f} GPU-hours.
- Legacy seeds 0-4 are excluded.

## Li-M-O

{stats_text(limo_stats)}

Decision: `{limo_decision}`. Unqualified `outperformed` wording supported: **{'yes' if outperformed else 'no'}**.

## Mn-Oxide

{stats_text(mn_stats)}

Decision: `{mn_decision}`. Qualified `comparable performance` wording supported: **{'yes' if comparable else 'no'}**.
"""
    (output_root / "FINAL_CORRECTED_TEN_SEED_CLAIM_DECISION.md").write_text(final, encoding="utf-8")
    execution = {
        "new_completed": new_done,
        "new_failed": new_failed,
        "new_summed_trajectory_GPU_hours": new_runtime_hours,
        "new_wall_hours": new_wall_hours,
        "combined_corrected_completed": done,
        "combined_corrected_failed": failed,
        "combined_corrected_summed_trajectory_GPU_hours": runtime_hours,
        "limo_decision": limo_decision,
        "mnoxide_decision": mn_decision,
        "outperformed_supported": outperformed,
        "comparable_performance_supported": comparable,
    }
    (output_root / "CORRECTED_SEEDS_5_14_EXECUTION_SUMMARY.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    print(json.dumps(execution, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
