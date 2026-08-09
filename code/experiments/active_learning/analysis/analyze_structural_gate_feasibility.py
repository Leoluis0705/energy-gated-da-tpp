"""Analyze the held-out structural-group Gate feasibility experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


TASKS = ("mn", "mg")
METHODS = (
    "predicted_target_greedy",
    "energy_gated_da_tpp",
    "structural_group_gate",
    "structural_group_gate_q95",
    "gradient_norm_hybrid",
)
REVISED_METHODS = ("structural_group_gate", "structural_group_gate_q95")
SEEDS = tuple(range(111, 116))
CHECKPOINTS = (80, 160, 240, 320)
METHOD_LABELS = {
    "predicted_target_greedy": "Greedy",
    "energy_gated_da_tpp": "Legacy Gate",
    "structural_group_gate": "Structural Gate",
    "structural_group_gate_q95": "Structural Gate + Q95",
    "gradient_norm_hybrid": "Gradient-Norm Hybrid",
}


def _method_task_summary(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    selected = frame[frame["method"].eq(method)].copy()
    expected = {(task, seed) for task in TASKS for seed in SEEDS}
    observed = set(zip(selected["task"], selected["seed"].astype(int)))
    if observed != expected:
        raise ValueError(f"{method} does not contain a complete five-seed task grid")
    return selected.groupby("task", as_index=False).agg(
        mean_autc_delta=("autc_delta", "mean"),
        mean_recovery80_delta=("recovery_at_80_delta", "mean"),
        mean_cluster80_delta=("target_cluster_at_80_delta", "mean"),
        mean_correction_gain=("cumulative_correction_target_gain", "mean"),
        worst_seed_autc_delta=("autc_delta", "min"),
    )


def classify_decision(paired_revised: pd.DataFrame) -> str:
    """Apply the frozen descriptive decision rules to revised Gate methods."""
    summaries = {
        method: _method_task_summary(paired_revised, method)
        for method in REVISED_METHODS
    }
    for summary in summaries.values():
        if (
            (summary["mean_autc_delta"] > 0).all()
            and (summary["mean_recovery80_delta"] >= 0).all()
            and (summary["mean_cluster80_delta"] >= 0).all()
            and (summary["mean_correction_gain"] >= 0).all()
        ):
            return "STRONG_GO"
    for method, summary in summaries.items():
        rows = paired_revised[paired_revised["method"].eq(method)]
        cluster_values = summary.set_index("task")["mean_cluster80_delta"]
        if (
            (summary["mean_autc_delta"] >= -0.005).all()
            and (summary["mean_recovery80_delta"] >= -1).all()
            and (cluster_values >= 0).all()
            and (cluster_values >= 1).any()
            and float(rows["autc_delta"].min()) >= -0.02
        ):
            return "CONDITIONAL_GO"
    return "STOP"


def _read_group_map(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, dtype=str)
    if frame.columns.tolist() != ["candidate_id", "group_key"]:
        raise ValueError("structural group map must have candidate_id,group_key columns")
    if len(frame) != 640 or frame["candidate_id"].duplicated().any():
        raise ValueError("structural group map must contain 640 unique candidates")
    return dict(zip(frame["candidate_id"], frame["group_key"], strict=True))


def _trace_column_sum(trace: pd.DataFrame, column: str) -> float:
    if column not in trace.columns:
        return float("nan")
    return float(pd.to_numeric(trace[column], errors="coerce").sum(min_count=1))


def _checkpoint_metrics(
    history: pd.DataFrame,
    checkpoint: int,
    group_map: dict[str, str],
    evaluability: dict[str, float],
) -> dict[str, float]:
    selected = history.iloc[: int(checkpoint)].copy()
    selected["candidate_id"] = selected["id"].astype(str)
    selected["cluster"] = selected["candidate_id"].map(group_map)
    if selected["cluster"].isna().any():
        raise ValueError("selected candidate is missing from structural group map")
    selected["target_label"] = pd.to_numeric(selected["target_label"], errors="raise").astype(int)
    targets = selected[selected["target_label"].eq(1)].copy()
    targets["p_dft_evaluable"] = targets["candidate_id"].map(evaluability)
    if targets["p_dft_evaluable"].isna().any():
        raise ValueError("recovered target is missing from hidden evaluability table")
    recovery = int(len(targets))
    return {
        f"recovery_at_{checkpoint}": recovery,
        f"recovery_rate_at_{checkpoint}": recovery,
        f"queries_per_target_at_{checkpoint}": (
            float(checkpoint / recovery) if recovery else float("inf")
        ),
        f"all_cluster_at_{checkpoint}": int(selected["cluster"].nunique()),
        f"target_cluster_at_{checkpoint}": int(targets["cluster"].nunique()),
        f"hidden_expected_evaluable_targets_at_{checkpoint}": float(
            targets["p_dft_evaluable"].sum()
        ),
        f"hidden_expected_evaluable_fraction_at_{checkpoint}": (
            float(targets["p_dft_evaluable"].mean()) if recovery else float("nan")
        ),
    }


def collect_per_seed(
    results_root: Path,
    group_map_path: Path,
    hidden_scores_path: Path,
) -> pd.DataFrame:
    group_map = _read_group_map(group_map_path)
    hidden = pd.read_csv(hidden_scores_path)
    evaluability = dict(
        zip(
            hidden["candidate_id"].astype(str),
            pd.to_numeric(hidden["p_dft_evaluable"], errors="raise"),
            strict=True,
        )
    )
    records: list[dict[str, object]] = []
    for task in TASKS:
        for method in METHODS:
            for seed in SEEDS:
                run = results_root / task / method / f"seed_{seed}"
                required = [
                    run / "run_metrics.csv",
                    run / "al_history.csv",
                    run / "round_diagnostics.csv",
                    run / "status.json",
                    run / "initialization_manifest.json",
                ]
                missing = [path for path in required if not path.is_file()]
                if missing:
                    raise FileNotFoundError("; ".join(str(path) for path in missing))
                status = json.loads((run / "status.json").read_text(encoding="utf-8"))
                if status.get("status") != "DONE":
                    raise ValueError(f"run is not DONE: {run}")
                metrics = pd.read_csv(run / "run_metrics.csv").iloc[0]
                history = pd.read_csv(run / "al_history.csv")
                if len(history) != 320:
                    raise ValueError(f"run history must contain 320 rows: {run}")
                diagnostics = pd.read_csv(run / "round_diagnostics.csv")
                trace_paths = list(run.glob("mode_trace_*.csv"))
                if len(trace_paths) != 1:
                    raise ValueError(f"run must contain one mode trace: {run}")
                trace = pd.read_csv(trace_paths[0])
                initialization = json.loads(
                    (run / "initialization_manifest.json").read_text(encoding="utf-8")
                )
                record: dict[str, object] = {
                    "task": task,
                    "method": method,
                    "seed": seed,
                    "initial_set_sha256": str(initialization["candidate_order_sha256"]),
                    "autc": float(metrics["AUTC"]),
                    "final_recovery": int(metrics["final_recovery"]),
                    "direct_rounds": int(metrics["direct_rounds"]),
                    "correction_rounds": int(metrics["correction_rounds"]),
                    "cumulative_correction_target_gain": float(
                        metrics["total_correction_target_gain"]
                    ),
                    "correction_replacements": int(metrics["total_correction_replacements"]),
                    "mean_group_repetition_rate": float(metrics["mean_group_repetition_rate"]),
                    "safeguard_fallbacks": int(
                        pd.to_numeric(
                            trace.get("quality_safeguard_fallback", pd.Series(0, index=trace.index)),
                            errors="coerce",
                        ).fillna(0).sum()
                    ),
                    "direct_p_hit_sum": float(
                        _trace_column_sum(trace, "direct_batch_p_hit_sum")
                    ),
                    "proposed_p_hit_sum": float(
                        _trace_column_sum(trace, "proposed_batch_p_hit_sum")
                    ),
                    "selected_p_hit_sum": _trace_column_sum(
                        trace, "selected_batch_p_hit_sum"
                    ),
                }
                for checkpoint in CHECKPOINTS:
                    record.update(
                        _checkpoint_metrics(history, checkpoint, group_map, evaluability)
                    )
                    total_targets = int(metrics["total_target_count"])
                    record[f"recovery_rate_at_{checkpoint}"] = (
                        record[f"recovery_at_{checkpoint}"] / total_targets
                    )
                records.append(record)
    return pd.DataFrame(records)


def build_paired(per_seed: pd.DataFrame) -> pd.DataFrame:
    greedy = per_seed[per_seed["method"].eq("predicted_target_greedy")].set_index(
        ["task", "seed"]
    )
    rows: list[dict[str, object]] = []
    for record in per_seed.itertuples(index=False):
        if record.method == "predicted_target_greedy":
            continue
        baseline = greedy.loc[(record.task, record.seed)]
        if str(record.initial_set_sha256) != str(baseline["initial_set_sha256"]):
            raise ValueError(
                f"initial-set hash mismatch for {record.task} seed {record.seed}: "
                f"{record.method} vs Greedy"
            )
        row: dict[str, object] = {
            "task": record.task,
            "method": record.method,
            "seed": int(record.seed),
            "autc_delta": float(record.autc - baseline["autc"]),
            "cumulative_correction_target_gain": float(
                record.cumulative_correction_target_gain
            ),
        }
        for checkpoint in CHECKPOINTS:
            row[f"recovery_at_{checkpoint}_delta"] = float(
                getattr(record, f"recovery_at_{checkpoint}")
                - baseline[f"recovery_at_{checkpoint}"]
            )
            row[f"recovery_rate_at_{checkpoint}_delta"] = float(
                getattr(record, f"recovery_rate_at_{checkpoint}")
                - baseline[f"recovery_rate_at_{checkpoint}"]
            )
            row[f"queries_per_target_at_{checkpoint}_delta"] = float(
                getattr(record, f"queries_per_target_at_{checkpoint}")
                - baseline[f"queries_per_target_at_{checkpoint}"]
            )
            row[f"target_cluster_at_{checkpoint}_delta"] = float(
                getattr(record, f"target_cluster_at_{checkpoint}")
                - baseline[f"target_cluster_at_{checkpoint}"]
            )
            row[f"hidden_expected_evaluable_at_{checkpoint}_delta"] = float(
                getattr(record, f"hidden_expected_evaluable_targets_at_{checkpoint}")
                - baseline[f"hidden_expected_evaluable_targets_at_{checkpoint}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_method_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (task, method), group in per_seed.groupby(["task", "method"], sort=True):
        row: dict[str, object] = {
            "task": task,
            "method": method,
            "mean_autc": float(group["autc"].mean()),
            "sd_autc": float(group["autc"].std(ddof=1)),
            "mean_final_recovery": float(group["final_recovery"].mean()),
        }
        for checkpoint in CHECKPOINTS:
            row[f"mean_recovery_at_{checkpoint}"] = float(
                group[f"recovery_at_{checkpoint}"].mean()
            )
            row[f"mean_recovery_rate_at_{checkpoint}"] = float(
                group[f"recovery_rate_at_{checkpoint}"].mean()
            )
            row[f"mean_queries_per_target_at_{checkpoint}"] = float(
                group[f"queries_per_target_at_{checkpoint}"].mean()
            )
            row[f"mean_target_cluster_at_{checkpoint}"] = float(
                group[f"target_cluster_at_{checkpoint}"].mean()
            )
            row[f"mean_hidden_evaluable_at_{checkpoint}"] = float(
                group[f"hidden_expected_evaluable_targets_at_{checkpoint}"].mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_outputs(per_seed: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    colors = {
        "predicted_target_greedy": "#222222",
        "energy_gated_da_tpp": "#a6761d",
        "structural_group_gate": "#1b9e77",
        "structural_group_gate_q95": "#1f78b4",
        "gradient_norm_hybrid": "#7570b3",
    }
    line_styles = {
        "predicted_target_greedy": "-",
        "energy_gated_da_tpp": "--",
        "structural_group_gate": "-",
        "structural_group_gate_q95": ":",
        "gradient_norm_hybrid": "-.",
    }
    task_titles = {
        "mn": "Mn-dominated proxy window\n[-2.1, -1.9] eV atom$^{-1}$",
        "mg": "Mg-anchored mixed Cr/Mg window\n[-2.3, -2.1] eV atom$^{-1}$",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for task, ax in zip(TASKS, axes, strict=True):
        subset = per_seed[per_seed["task"].eq(task)]
        for method in METHODS:
            part = subset[subset["method"].eq(method)]
            means = [part[f"recovery_at_{cp}"].mean() for cp in CHECKPOINTS]
            mins = [part[f"recovery_at_{cp}"].min() for cp in CHECKPOINTS]
            maxs = [part[f"recovery_at_{cp}"].max() for cp in CHECKPOINTS]
            ax.plot(
                CHECKPOINTS,
                means,
                marker="o",
                linestyle=line_styles[method],
                label=METHOD_LABELS[method],
                color=colors[method],
            )
            ax.fill_between(CHECKPOINTS, mins, maxs, color=colors[method], alpha=0.10)
        ax.set(xlabel="Queries", ylabel="Recovered targets", title=task_titles[task])
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Target recovery by query budget", fontsize=13)
    fig.text(0.5, 0.01, "Lines show five-seed means; shaded bands show the observed seed range.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(output_dir / "figure_recovery.pdf")
    fig.savefig(output_dir / "figure_recovery.png", dpi=300)
    plt.close(fig)

    summary = paired.groupby(["task", "method"], as_index=False)["autc_delta"].mean()
    compared_methods = [method for method in METHODS if method != "predicted_target_greedy"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(TASKS), dtype=float)
    width = 0.18
    for index, method in enumerate(compared_methods):
        values = [
            float(summary.loc[summary["task"].eq(task) & summary["method"].eq(method), "autc_delta"].iloc[0])
            for task in TASKS
        ]
        positions = x + (index - (len(compared_methods) - 1) / 2) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=colors[method],
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=3, fontsize=8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, ["Mn-dominated\nwindow", "Mg-anchored mixed\nCr/Mg window"])
    ax.set_ylabel("Mean normalized AUTC difference vs Greedy")
    ax.set_title("AUTC difference relative to Greedy")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_ylim(top=0.055)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_autc.pdf")
    fig.savefig(output_dir / "figure_autc.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for task, ax in zip(TASKS, axes, strict=True):
        subset = per_seed[per_seed["task"].eq(task)]
        for method in METHODS:
            part = subset[subset["method"].eq(method)]
            means = [part[f"target_cluster_at_{cp}"].mean() for cp in CHECKPOINTS]
            mins = [part[f"target_cluster_at_{cp}"].min() for cp in CHECKPOINTS]
            maxs = [part[f"target_cluster_at_{cp}"].max() for cp in CHECKPOINTS]
            ax.plot(
                CHECKPOINTS,
                means,
                marker="o",
                linestyle=line_styles[method],
                label=METHOD_LABELS[method],
                color=colors[method],
            )
            ax.fill_between(CHECKPOINTS, mins, maxs, color=colors[method], alpha=0.10)
        ax.set(
            xlabel="Queries",
            ylabel="Recovered target clusters",
            title=task_titles[task],
        )
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=5,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle("Structural-cluster coverage among recovered targets", fontsize=13)
    fig.text(0.5, 0.01, "Lines show five-seed means; shaded bands show the observed seed range.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    fig.savefig(output_dir / "figure_target_cluster_coverage.pdf")
    fig.savefig(output_dir / "figure_target_cluster_coverage.png", dpi=300)
    plt.close(fig)

    correction = paired.groupby(["task", "method"], as_index=False)[
        "cumulative_correction_target_gain"
    ].mean()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for index, method in enumerate(compared_methods):
        values = [
            float(
                correction.loc[
                    correction["task"].eq(task) & correction["method"].eq(method),
                    "cumulative_correction_target_gain",
                ].iloc[0]
            )
            for task in TASKS
        ]
        positions = x + (index - (len(compared_methods) - 1) / 2) * width
        bars = ax.bar(
            positions,
            values,
            width=width,
            label=METHOD_LABELS[method],
            color=colors[method],
            edgecolor="#333333",
            linewidth=0.4,
        )
        ax.bar_label(bars, labels=[f"{value:+.1f}" for value in values], padding=3, fontsize=8)
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_xticks(x, ["Mn-dominated\nwindow", "Mg-anchored mixed\nCr/Mg window"])
    ax.set_ylabel("Mean correction target gain")
    ax.set_title("Targets added or lost by correction relative to the direct batch")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.set_ylim(top=3.0)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "figure_correction_loss.pdf")
    fig.savefig(output_dir / "figure_correction_loss.png", dpi=300)
    plt.close(fig)


def _build_chinese_report(
    decision: str,
    method_summary: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> str:
    lines = [
        "# 结构分组 Gate 可行性实验报告",
        "",
        f"**冻结决策：`{decision}`**",
        "",
        "隐藏 DFT 可结算性数值仅为 post-selection 预测审计，不是真实 DFT 结果。",
        "",
        "## 方法汇总",
        "",
        method_summary.to_markdown(index=False),
        "",
        "## 相对 Greedy 的配对差值",
        "",
        paired_summary.to_markdown(index=False),
        "",
        "所有五种方法和五个 held-out seeds 均已保留；未根据结果调整区间、阈值或指标。",
    ]
    return "\n".join(lines) + "\n"


def _build_claim_boundary(decision: str) -> str:
    return (
        "# Structural Gate Claim Boundary\n\n"
        f"Decision: `{decision}`. The revision was motivated by preliminary group-representation "
        "diagnostics and frozen before held-out seeds 111–115. It is not described as preregistered "
        "or as an independent confirmation. Hidden DFT evaluability remains predicted, not observed.\n"
    )


def run_analysis(
    results_root: Path,
    group_map_path: Path,
    hidden_scores_path: Path,
    output_dir: Path,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=False)
    per_seed = collect_per_seed(results_root, group_map_path, hidden_scores_path)
    paired = build_paired(per_seed)
    method_summary = build_method_summary(per_seed)
    revised = paired[paired["method"].isin(REVISED_METHODS)].copy()
    decision = classify_decision(revised)
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False, lineterminator="\n")
    paired.to_csv(output_dir / "paired_gate_vs_greedy.csv", index=False, lineterminator="\n")
    method_summary.to_csv(output_dir / "method_summary.csv", index=False, lineterminator="\n")
    per_seed[
        [
            "task", "method", "seed", "direct_rounds", "correction_rounds",
            "correction_replacements", "cumulative_correction_target_gain",
            "mean_group_repetition_rate", "safeguard_fallbacks",
            "direct_p_hit_sum", "proposed_p_hit_sum", "selected_p_hit_sum",
        ]
    ].to_csv(output_dir / "mechanism_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "decision.json").write_text(
        json.dumps({"decision": decision, "seeds": list(SEEDS)}, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_outputs(per_seed, paired, output_dir)
    summary = paired.groupby(["task", "method"], as_index=False).agg(
        mean_autc_delta=("autc_delta", "mean"),
        mean_recovery80_delta=("recovery_at_80_delta", "mean"),
        mean_target_cluster80_delta=("target_cluster_at_80_delta", "mean"),
        mean_hidden_evaluable80_delta=("hidden_expected_evaluable_at_80_delta", "mean"),
    )
    (output_dir / "STRUCTURAL_GATE_FEASIBILITY_REPORT_ZH.md").write_text(
        _build_chinese_report(decision, method_summary, summary), encoding="utf-8"
    )
    (output_dir / "STRUCTURAL_GATE_CLAIM_BOUNDARY.md").write_text(
        _build_claim_boundary(decision), encoding="utf-8"
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--hidden-scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = run_analysis(
        args.results_root.resolve(),
        args.group_map.resolve(),
        args.hidden_scores.resolve(),
        args.output_dir.resolve(),
    )
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
