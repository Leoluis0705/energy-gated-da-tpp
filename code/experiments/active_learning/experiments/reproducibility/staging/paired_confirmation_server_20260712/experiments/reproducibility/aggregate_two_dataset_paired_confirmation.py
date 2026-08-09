#!/usr/bin/env python3
"""Aggregate legacy seeds 0-4 and corrected paired seeds 5-9 without unsafe pooling."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
except ImportError:  # Reported as unavailable rather than fabricated.
    wilcoxon = None

try:
    from .two_dataset_paired_protocol import EQUIVALENCE_MARGIN
except ImportError:
    from two_dataset_paired_protocol import EQUIVALENCE_MARGIN


LEGACY_METHODS = {
    "limo": {"energy_gated_da_tpp": "hard_gate_g050", "predicted_distance_greedy": "greedy"},
    "mnoxide": {"energy_gated_da_tpp": "hard_gate_g075", "predicted_distance_greedy": "greedy"},
}


def at_budget(history: pd.DataFrame, query_count: int) -> int:
    available = history[pd.to_numeric(history["oracle_evaluations"]) <= query_count]
    return int(available.iloc[-1]["cumulative_target_count"]) if not available.empty else 0


def load_legacy(legacy_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict] = []
    limo_effects: list[dict] = []
    mn_fallback: list[dict] = []
    for dataset, method_map in LEGACY_METHODS.items():
        for method, legacy_method in method_map.items():
            for seed in range(5):
                run_dir = legacy_root / "runs" / dataset / legacy_method / f"seed_{seed}"
                summary = pd.read_csv(run_dir / "summary.csv").iloc[0]
                history = pd.read_csv(run_dir / "al_history.csv")
                row = {
                    "protocol_cohort": "legacy_shuffle_true_seeds_0_4",
                    "protocol_compatible_for_pooled_0_9": False,
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "budget": int(summary["budget"]),
                    "batch_size": int(summary["batch_size"]),
                    "total_target_count": int(summary["total_target_count"]),
                    "recovery_at_80": at_budget(history, 80),
                    "recovery_at_160": int(summary["targets_at_160"]),
                    "recovery_at_240": at_budget(history, 240),
                    "recovery_at_320": int(summary["targets_at_320"]),
                    "final_recovery": int(summary["final_target_recovery_count"]),
                    "AUTC": float(summary["AUTC"]),
                    "direct_rounds": int(summary["direct_equivalent_rounds"]),
                    "correction_rounds": int(summary["full_correction_rounds"]),
                    "direct_route_proportion": float(summary["direct_equivalent_rounds"]) / len(history),
                    "mean_unique_groups_per_batch": float(pd.to_numeric(history["unique_groups"]).mean()),
                    "mean_group_repetition_rate": float((16 - pd.to_numeric(history["unique_groups"])).mean() / 16.0),
                    "total_correction_replacements": np.nan,
                    "total_correction_target_gain": np.nan,
                    "source_run_dir": str(run_dir),
                    "compatibility_note": "retained only; shuffle/order protocol differs from corrected seeds 5-9",
                }
                metrics.append(row)
                if dataset == "limo" and method == "energy_gated_da_tpp":
                    corrected = history[history["route_label"] == "diversity_aware"]
                    limo_effects.append(
                        {
                            "protocol_cohort": row["protocol_cohort"],
                            "seed": seed,
                            "correction_rounds": len(corrected),
                            "targets_in_corrected_batches": int(corrected["round_target_hits"].sum()),
                            "mean_unique_groups_all_batches": row["mean_unique_groups_per_batch"],
                            "mean_group_repetition_rate_all_batches": row["mean_group_repetition_rate"],
                            "correction_replacements": np.nan,
                            "correction_target_gain": np.nan,
                            "substitution_provenance": "unavailable in retained legacy runner",
                        }
                    )
                if dataset == "mnoxide" and method == "energy_gated_da_tpp":
                    mn_fallback.append(
                        {
                            "protocol_cohort": row["protocol_cohort"],
                            "seed": seed,
                            "direct_rounds": row["direct_rounds"],
                            "correction_rounds": row["correction_rounds"],
                            "direct_route_proportion": row["direct_route_proportion"],
                            "gate_AUTC": row["AUTC"],
                        }
                    )
    return pd.DataFrame(metrics), pd.DataFrame(limo_effects), pd.DataFrame(mn_fallback)


def load_corrected(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[pd.DataFrame] = []
    limo_effects: list[dict] = []
    mn_fallback: list[dict] = []
    for dataset in ("limo", "mnoxide"):
        for method in ("energy_gated_da_tpp", "predicted_distance_greedy"):
            for seed in range(5, 10):
                run_dir = output_root / "runs" / dataset / method / f"seed_{seed}"
                metric_path = run_dir / "run_metrics.csv"
                if not metric_path.exists():
                    continue
                frame = pd.read_csv(metric_path)
                frame.insert(0, "protocol_compatible_for_pooled_0_9", False)
                frame.insert(0, "protocol_cohort", "corrected_shuffle_false_seeds_5_9")
                frame["source_run_dir"] = str(run_dir)
                frame["compatibility_note"] = "paired corrected cohort; not poolable with legacy seeds 0-4"
                metrics.append(frame)
                row = frame.iloc[0]
                if dataset == "limo" and method == "energy_gated_da_tpp":
                    trajectory = pd.read_csv(run_dir / "summary.csv")
                    substitutions_path = run_dir / "correction_substitutions.csv"
                    substitutions = pd.read_csv(substitutions_path) if substitutions_path.exists() and substitutions_path.stat().st_size else pd.DataFrame()
                    corrected = trajectory[trajectory["route_choice"] == "diversity_aware"]
                    limo_effects.append(
                        {
                            "protocol_cohort": "corrected_shuffle_false_seeds_5_9",
                            "seed": seed,
                            "correction_rounds": len(corrected),
                            "targets_in_corrected_batches": int(corrected["round_target_hits"].sum()),
                            "mean_unique_groups_all_batches": float(row["mean_unique_groups_per_batch"]),
                            "mean_group_repetition_rate_all_batches": float(row["mean_group_repetition_rate"]),
                            "correction_replacements": int(row["total_correction_replacements"]),
                            "correction_target_gain": int(row["total_correction_target_gain"]),
                            "removed_target_count": int(substitutions.loc[substitutions.get("substitution_role") == "removed", "target_label"].sum()) if not substitutions.empty else 0,
                            "inserted_target_count": int(substitutions.loc[substitutions.get("substitution_role") == "inserted", "target_label"].sum()) if not substitutions.empty else 0,
                            "substitution_provenance": str(substitutions_path),
                        }
                    )
                if dataset == "mnoxide" and method == "energy_gated_da_tpp":
                    mn_fallback.append(
                        {
                            "protocol_cohort": "corrected_shuffle_false_seeds_5_9",
                            "seed": seed,
                            "direct_rounds": int(row["direct_rounds"]),
                            "correction_rounds": int(row["correction_rounds"]),
                            "direct_route_proportion": float(row["direct_route_proportion"]),
                            "gate_AUTC": float(row["AUTC"]),
                        }
                    )
    return (pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(), pd.DataFrame(limo_effects), pd.DataFrame(mn_fallback))


def paired_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[metrics["method"] == "energy_gated_da_tpp"].copy()
    greedy = metrics[metrics["method"] == "predicted_distance_greedy"].copy()
    keys = ["protocol_cohort", "dataset", "seed"]
    joined = gate.merge(greedy, on=keys, suffixes=("_gate", "_greedy"), validate="one_to_one")
    rows = joined[keys].copy()
    rows["gate_AUTC"] = joined["AUTC_gate"]
    rows["greedy_AUTC"] = joined["AUTC_greedy"]
    rows["paired_AUTC_difference_gate_minus_greedy"] = rows["gate_AUTC"] - rows["greedy_AUTC"]
    for checkpoint in (80, 160, 240, 320):
        rows[f"gate_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_gate"]
        rows[f"greedy_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_greedy"]
        rows[f"paired_recovery_difference_at_{checkpoint}"] = (
            rows[f"gate_recovery_at_{checkpoint}"] - rows[f"greedy_recovery_at_{checkpoint}"]
        )
    return rows


def bootstrap_ci(values: np.ndarray, samples: int = 20_000) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(20260712)
    means = np.mean(rng.choice(values, size=(samples, len(values)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_statistics(rows: pd.DataFrame) -> dict:
    values = rows["paired_AUTC_difference_gate_minus_greedy"].to_numpy(dtype=float)
    ci_low, ci_high = bootstrap_ci(values)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
    effect = float(np.mean(values) / std) if std and np.isfinite(std) else (0.0 if np.allclose(values, 0) else math.nan)
    wins = int((values > 1e-12).sum())
    ties = int((np.abs(values) <= 1e-12).sum())
    losses = int((values < -1e-12).sum())
    if wilcoxon is None or len(values) < 2 or np.allclose(values, 0):
        statistic, pvalue = math.nan, math.nan
    else:
        result = wilcoxon(values, alternative="two-sided", zero_method="wilcox")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "n_pairs": len(values),
        "gate_AUTC_mean": float(rows["gate_AUTC"].mean()),
        "gate_AUTC_sd": float(rows["gate_AUTC"].std(ddof=1)),
        "greedy_AUTC_mean": float(rows["greedy_AUTC"].mean()),
        "greedy_AUTC_sd": float(rows["greedy_AUTC"].std(ddof=1)),
        "paired_mean_difference": float(np.mean(values)),
        "paired_median_difference": float(np.median(values)),
        "bootstrap_95_ci_low": ci_low,
        "bootstrap_95_ci_high": ci_high,
        "paired_effect_size_dz": effect,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "wilcoxon_statistic": statistic,
        "wilcoxon_pvalue": pvalue,
    }


def format_stats(stats: dict) -> str:
    return (
        f"Gate AUTC {stats['gate_AUTC_mean']:.6f} +/- {stats['gate_AUTC_sd']:.6f}; "
        f"Greedy {stats['greedy_AUTC_mean']:.6f} +/- {stats['greedy_AUTC_sd']:.6f}; "
        f"paired mean {stats['paired_mean_difference']:+.6f}, bootstrap 95% CI "
        f"[{stats['bootstrap_95_ci_low']:+.6f}, {stats['bootstrap_95_ci_high']:+.6f}], "
        f"W/T/L={stats['wins']}/{stats['ties']}/{stats['losses']}."
    )


def write_reports(output_root: Path, all_metrics: pd.DataFrame, differences: pd.DataFrame, limo_effects: pd.DataFrame, mn_fallback: pd.DataFrame) -> None:
    reports: dict[str, str] = {}
    cohort_stats: dict[tuple[str, str], dict] = {}
    for (cohort, dataset), rows in differences.groupby(["protocol_cohort", "dataset"]):
        cohort_stats[(cohort, dataset)] = paired_statistics(rows)

    for dataset, filename, title in (
        ("limo", "PAIRED_SEEDS_0_9_LIMO_REPORT.md", "Li-M-O Paired Seeds 0-9 Inventory"),
        ("mnoxide", "PAIRED_SEEDS_0_9_MNOXIDE_REPORT.md", "Mn-Oxide Paired Seeds 0-9 Inventory"),
    ):
        lines = [f"# {title}", "", "## Protocol compatibility", "", "Seeds 0-4 and 5-9 are retained as separate cohorts and are **not pooled**. The legacy cohort used `shuffle=True` and no explicit prediction reindex; the corrected cohort uses `shuffle=False` and explicit ID reindexing. Mn-oxide also changes alpha from 0.10 to the audited 0.05.", ""]
        for cohort in ("legacy_shuffle_true_seeds_0_4", "corrected_shuffle_false_seeds_5_9"):
            stats = cohort_stats.get((cohort, dataset))
            lines.extend([f"## {cohort}", "", format_stats(stats) if stats else "Incomplete cohort; no paired statistic reported.", ""])
        if dataset == "limo":
            lines.extend(["## Correction diagnostics", "", f"Detailed correction effects: `{output_root / 'LIMO_CORRECTION_EFFECTS_SEEDS_0_9.csv'}`.", "", "The predeclared 8-of-10 decision rule cannot be applied across incompatible protocol cohorts."])
        else:
            corrected = cohort_stats.get(("corrected_shuffle_false_seeds_5_9", dataset))
            if corrected:
                direct = mn_fallback[mn_fallback["protocol_cohort"] == "corrected_shuffle_false_seeds_5_9"]
                rare = int(direct["correction_rounds"].sum()) <= max(1, int(0.05 * direct["direct_rounds"].add(direct["correction_rounds"]).sum()))
                if rare and corrected["bootstrap_95_ci_low"] >= -EQUIVALENCE_MARGIN:
                    decision = "DIRECT_FALLBACK_CONFIRMED"
                elif corrected["bootstrap_95_ci_high"] < -EQUIVALENCE_MARGIN:
                    decision = "FALLBACK_WITH_DEGRADATION"
                else:
                    decision = "FALLBACK_INCONCLUSIVE"
                lines.extend(["## Corrected-cohort fallback decision", "", f"`{decision}` using the frozen AUTC margin +/-{EQUIVALENCE_MARGIN:.2f}."])
        (output_root / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")
        reports[dataset] = filename

    corrected_limo = cohort_stats.get(("corrected_shuffle_false_seeds_5_9", "limo"))
    corrected_mn = cohort_stats.get(("corrected_shuffle_false_seeds_5_9", "mnoxide"))
    decision_lines = [
        "# Final Two-Dataset Claim Decision",
        "",
        "## Evidence boundary",
        "",
        "The requested 10-seed pooled inference is not valid because retained seeds 0-4 and corrected seeds 5-9 use different prediction-order protocols; Mn-oxide also uses a different alpha. Results are reported without averaging incompatible cohorts.",
        "",
        "## Li-M-O",
        "",
        format_stats(corrected_limo) if corrected_limo else "Corrected cohort incomplete.",
        "",
        "The manuscript may use `outperformed` only if a protocol-compatible confirmatory cohort satisfies its predeclared superiority rule. This audit does not silently treat the mixed 0-9 inventory as such a cohort.",
        "",
        "## Mn-oxide",
        "",
        format_stats(corrected_mn) if corrected_mn else "Corrected cohort incomplete.",
        "",
        f"Comparable-performance wording requires the corrected paired confidence interval to remain within the predeclared -{EQUIVALENCE_MARGIN:.2f} non-degradation boundary and correction to remain rare or absent.",
    ]
    (output_root / "FINAL_TWO_DATASET_CLAIM_DECISION.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--legacy-root", required=True)
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    legacy_root = Path(args.legacy_root).resolve()
    completion = json.loads((output_root / "COMPLETION.json").read_text(encoding="utf-8"))
    if completion.get("completed") != 20 or completion.get("failed") != 0:
        raise RuntimeError("all 20 corrected trajectories must complete before aggregation")
    legacy_metrics, legacy_limo, legacy_mn = load_legacy(legacy_root)
    corrected_metrics, corrected_limo, corrected_mn = load_corrected(output_root)
    if len(corrected_metrics) != 20:
        raise RuntimeError(f"expected 20 corrected run-metric rows, found {len(corrected_metrics)}")
    all_metrics = pd.concat([legacy_metrics, corrected_metrics], ignore_index=True)
    all_metrics.to_csv(output_root / "PAIRED_SEEDS_0_9_ALL_RESULTS.csv", index=False)
    differences = paired_rows(all_metrics)
    differences.to_csv(output_root / "PAIRED_AUTC_DIFFERENCES_BY_DATASET.csv", index=False)
    limo_effects = pd.concat([legacy_limo, corrected_limo], ignore_index=True)
    limo_effects.to_csv(output_root / "LIMO_CORRECTION_EFFECTS_SEEDS_0_9.csv", index=False)
    mn_fallback = pd.concat([legacy_mn, corrected_mn], ignore_index=True)
    greedy_lookup = differences.set_index(["protocol_cohort", "dataset", "seed"])
    mn_fallback["greedy_AUTC"] = [
        greedy_lookup.at[(row.protocol_cohort, "mnoxide", int(row.seed)), "greedy_AUTC"] for row in mn_fallback.itertuples()
    ]
    mn_fallback["paired_AUTC_difference_gate_minus_greedy"] = mn_fallback["gate_AUTC"] - mn_fallback["greedy_AUTC"]
    mn_fallback["within_predeclared_non_degradation_margin"] = mn_fallback["paired_AUTC_difference_gate_minus_greedy"] >= -EQUIVALENCE_MARGIN
    mn_fallback.to_csv(output_root / "MNOXIDE_FALLBACK_ANALYSIS_SEEDS_0_9.csv", index=False)
    write_reports(output_root, all_metrics, differences, limo_effects, mn_fallback)
    print(json.dumps({"all_results": len(all_metrics), "paired_rows": len(differences), "pooled_0_9": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
