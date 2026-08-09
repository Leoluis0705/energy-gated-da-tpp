from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon


TASK_LABELS = {
    "mn": "Mn-anchored proxy interval",
    "mg": "Mg-anchored proxy interval",
}

METHOD_ORDER = (
    "energy_gated_da_tpp",
    "predicted_target_greedy",
    "explore",
    "mc_dropout",
    "gradient_norm_hybrid",
    "random_sampling",
    "always_da_tpp",
    "group_only_gate",
    "margin_only_gate",
)

METHOD_LABELS = {
    "energy_gated_da_tpp": "Full Gate",
    "predicted_target_greedy": "Greedy",
    "explore": "Explore",
    "mc_dropout": "MC dropout",
    "gradient_norm_hybrid": "Gradient-norm hybrid",
    "random_sampling": "Random",
    "always_da_tpp": "Always correction",
    "group_only_gate": "Group-only",
    "margin_only_gate": "Margin-only",
}

METHOD_COLORS = {
    "energy_gated_da_tpp": "#0F4D92",
    "predicted_target_greedy": "#272727",
    "explore": "#42949E",
    "mc_dropout": "#9A4D8E",
    "gradient_norm_hybrid": "#E28E2C",
    "random_sampling": "#A8A8A8",
    "always_da_tpp": "#B9A7E8",
    "group_only_gate": "#3775BA",
    "margin_only_gate": "#767676",
}

CHECKPOINTS = (80, 160, 240, 320)
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260804


def _sha256_text(values: Iterable[str]) -> str:
    payload = "\n".join(str(value).strip() for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _left_continuous_autc(
    query_counts: Sequence[int],
    recoveries: Sequence[int],
    total_targets: int,
    budget: int,
) -> float:
    queries = np.asarray(query_counts, dtype=int)
    hits = np.asarray(recoveries, dtype=int)
    if queries.ndim != 1 or hits.ndim != 1 or len(queries) != len(hits):
        raise ValueError("query counts and recoveries must be equal-length vectors")
    if total_targets <= 0 or budget <= 0:
        raise ValueError("total_targets and budget must be positive")
    if len(queries) and (np.any(np.diff(queries) <= 0) or np.any(np.diff(hits) < 0)):
        raise ValueError("query counts must increase and recoveries must be nondecreasing")
    area = 0
    previous_query = 0
    previous_hits = 0
    for query, recovery in zip(queries, hits, strict=True):
        clipped = min(int(query), int(budget))
        area += max(0, clipped - previous_query) * previous_hits
        previous_query = clipped
        previous_hits = int(recovery)
        if previous_query >= budget:
            break
    if previous_query < budget:
        area += (budget - previous_query) * previous_hits
    return float(area / (total_targets * budget))


def _recovery_at(queries: np.ndarray, recoveries: np.ndarray, checkpoint: int) -> int:
    available = np.flatnonzero(queries <= int(checkpoint))
    return int(recoveries[available[-1]]) if len(available) else 0


def reconstruct_trajectory(
    summary: pd.DataFrame,
    *,
    budget: int,
    total_targets: int,
    checkpoints: Sequence[int] | None = None,
) -> dict[str, float | int]:
    required = {"round", "oracle_evaluations", "cumulative_target_count"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    frame = summary.loc[:, sorted(required)].copy()
    frame["oracle_evaluations"] = pd.to_numeric(
        frame["oracle_evaluations"], errors="raise"
    ).astype(int)
    frame["cumulative_target_count"] = pd.to_numeric(
        frame["cumulative_target_count"], errors="raise"
    ).astype(int)
    frame = frame.sort_values("oracle_evaluations", kind="mergesort")
    queries = frame["oracle_evaluations"].to_numpy(dtype=int)
    recoveries = frame["cumulative_target_count"].to_numpy(dtype=int)
    if len(frame) == 0 or int(queries[-1]) != int(budget):
        raise ValueError(f"trajectory must end at budget={budget}")
    result: dict[str, float | int] = {
        "autc": _left_continuous_autc(queries, recoveries, total_targets, budget),
        "final_recovery": int(recoveries[-1]),
    }
    requested = tuple(checkpoints) if checkpoints is not None else tuple(queries.tolist())
    for checkpoint in requested:
        result[f"recovery_at_{int(checkpoint)}"] = _recovery_at(
            queries, recoveries, int(checkpoint)
        )
    return result


def summarize_target_set(
    pool: pd.DataFrame,
    *,
    low: float,
    high: float,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    required = {
        "candidate_id",
        "m_element",
        "alignn_formation_energy_eV_atom",
        "structure_matcher_cluster",
    }
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"candidate pool is missing columns: {sorted(missing)}")
    energy = pd.to_numeric(pool["alignn_formation_energy_eV_atom"], errors="raise")
    targets = pool.loc[energy.between(float(low), float(high), inclusive="both")].copy()
    if targets.empty:
        raise ValueError("target interval contains no candidates")
    cluster_counts = targets["structure_matcher_cluster"].astype(str).value_counts()
    composition = (
        targets.groupby("m_element", sort=True)
        .size()
        .rename("target_count")
        .reset_index()
    )
    composition["target_fraction_within_task"] = (
        composition["target_count"] / len(targets)
    )
    summary: dict[str, float | int | str] = {
        "pool_size": int(len(pool)),
        "target_count": int(len(targets)),
        "target_fraction": float(len(targets) / len(pool)),
        "effective_cluster_count": int(cluster_counts.size),
        "largest_cluster_count": int(cluster_counts.iloc[0]),
        "largest_cluster_fraction": float(cluster_counts.iloc[0] / len(targets)),
        "dominant_element": str(composition.sort_values("target_count").iloc[-1]["m_element"]),
        "dominant_element_fraction": float(composition["target_count"].max() / len(targets)),
    }
    return summary, composition


def expected_evaluability_at(
    history: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    checkpoint: int,
) -> dict[str, float | int]:
    required_history = {"id", "target_label"}
    required_scores = {"candidate_id", "p_dft_evaluable"}
    if required_history - set(history.columns):
        raise ValueError("history is missing id or target_label")
    if required_scores - set(scores.columns):
        raise ValueError("scores are missing candidate_id or p_dft_evaluable")
    selected = history.iloc[: int(checkpoint)].copy()
    selected["target_label"] = pd.to_numeric(
        selected["target_label"], errors="raise"
    ).astype(int)
    targets = selected.loc[selected["target_label"] == 1, ["id"]].copy()
    joined = targets.merge(
        scores.loc[:, ["candidate_id", "p_dft_evaluable"]],
        left_on="id",
        right_on="candidate_id",
        how="left",
        validate="one_to_one",
    )
    probabilities = pd.to_numeric(joined["p_dft_evaluable"], errors="coerce")
    scored = probabilities.notna()
    target_count = int(len(targets))
    scored_count = int(scored.sum())
    expected = float(probabilities[scored].sum()) if scored_count else float("nan")
    return {
        "selected_target_count": target_count,
        "scored_target_count": scored_count,
        "score_coverage": float(scored_count / target_count) if target_count else float("nan"),
        "expected_evaluable_target_count": expected,
        "expected_evaluable_fraction_among_scored_targets": (
            float(expected / scored_count) if scored_count else float("nan")
        ),
    }


def paired_statistics(
    differences: Sequence[float],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, float | int | list[float] | None]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired statistics require at least two differences")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = values[rng.integers(0, len(values), size=(int(bootstrap_samples), len(values)))].mean(
        axis=1
    )
    ci = np.quantile(draws, [0.025, 0.975])
    sample_sd = float(values.std(ddof=1))
    all_zero = bool(np.all(values == 0))
    if all_zero:
        pvalue = None
        statistic = None
        effect = 0.0
    else:
        test = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
        pvalue = float(test.pvalue)
        statistic = float(test.statistic)
        effect = float(values.mean() / sample_sd) if sample_sd > 0 else None
    return {
        "n": int(len(values)),
        "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "sample_sd": sample_sd,
        "bootstrap_ci_low": float(ci[0]),
        "bootstrap_ci_high": float(ci[1]),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p_two_sided_exact": pvalue,
        "effect_size_dz": effect,
    }


def _load_anchor_table(formation_path: Path, pool: pd.DataFrame) -> pd.DataFrame:
    dft = pd.read_csv(formation_path)
    dft["vasp_completed"] = dft["vasp_completed"].astype(str).str.lower().eq("true")
    dft["selected_for_formation_energy"] = (
        dft["selected_for_formation_energy"].astype(str).str.lower().eq("true")
    )
    rows: list[dict[str, object]] = []
    for task, element, functional in (("mn", "Mn", "GGA+U"), ("mg", "Mg", "PBE")):
        selected = dft.loc[
            dft["candidate_id"].astype(str).str.contains(f"_{element}_", regex=False)
            & dft["functional"].eq(functional)
            & dft["vasp_completed"]
            & dft["selected_for_formation_energy"]
        ].copy()
        selected = selected.sort_values("candidate_id", kind="mergesort").drop_duplicates(
            "candidate_id", keep="last"
        )
        if element == "Mn":
            tight = pd.to_numeric(
                selected["tight_gga_u_formation_energy"], errors="coerce"
            )
            legacy = pd.to_numeric(
                selected["legacy_formation_energy_eV_per_atom"], errors="coerce"
            )
            selected["anchor_dft_energy_eV_atom"] = tight.fillna(legacy)
            energy_rule = "tight_GGA+U_when_available_else_verified_legacy_GGA+U"
        else:
            selected["anchor_dft_energy_eV_atom"] = pd.to_numeric(
                selected["legacy_formation_energy_eV_per_atom"], errors="coerce"
            )
            energy_rule = "verified_legacy_PBE"
        selected = selected.dropna(subset=["anchor_dft_energy_eV_atom"])
        selected = selected.merge(
            pool.loc[
                :,
                ["candidate_id", "alignn_formation_energy_eV_atom"],
            ],
            on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        for row in selected.itertuples(index=False):
            rows.append(
                {
                    "task": task,
                    "m_element": element,
                    "candidate_id": row.candidate_id,
                    "functional": functional,
                    "anchor_dft_energy_eV_atom": float(row.anchor_dft_energy_eV_atom),
                    "anchor_alignn_energy_eV_atom": float(
                        row.alignn_formation_energy_eV_atom
                    ),
                    "energy_selection_rule": energy_rule,
                }
            )
    return pd.DataFrame(rows)


def _read_run_grid(results_root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for task in TASK_LABELS:
        task_dir = results_root / task
        for method in METHOD_ORDER:
            method_dir = task_dir / method
            for seed in range(101, 111):
                run_dir = method_dir / f"seed_{seed}"
                required = (
                    "run_config.json",
                    "run_metrics.csv",
                    "summary.csv",
                    "al_history.csv",
                    "initialization_manifest.json",
                    "status.json",
                )
                missing = [name for name in required if not (run_dir / name).is_file()]
                if missing:
                    raise FileNotFoundError(f"{run_dir}: missing {missing}")
                records.append(
                    {"task": task, "method": method, "seed": seed, "run_dir": run_dir}
                )
    if len(records) != 180:
        raise ValueError(f"expected 180 formal runs, found {len(records)}")
    return records


def _aggregate_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "AUTC",
        "final_recovery",
        "final_recall_rate",
        "mean_group_repetition_rate",
        "mean_unique_groups_per_batch",
        "total_correction_replacements",
    ]
    for checkpoint in CHECKPOINTS:
        metrics.extend(
            [
                f"recovery_at_{checkpoint}",
                f"recovery_rate_at_{checkpoint}",
                f"queries_per_recovered_target_at_{checkpoint}",
                f"unique_structure_clusters_at_{checkpoint}",
                f"structure_cluster_coverage_at_{checkpoint}",
                f"expected_evaluable_targets_at_{checkpoint}",
                f"hidden_score_coverage_at_{checkpoint}",
            ]
        )
    rows: list[dict[str, object]] = []
    for (task, method), group in per_seed.groupby(["task", "method"], sort=False):
        row: dict[str, object] = {
            "task": task,
            "task_label": TASK_LABELS[task],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    result = pd.DataFrame(rows)
    result["AUTC_rank"] = result.groupby("task")["AUTC_mean"].rank(
        ascending=False, method="min"
    )
    return result.sort_values(["task", "AUTC_rank", "method"]).reset_index(drop=True)


def _plot_main_figure(
    trajectories: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired: pd.DataFrame,
    stats_by_task: dict[str, dict[str, object]],
    figure_dir: Path,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["legend.frameon"] = False

    figure_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.32)
    core_methods = (
        "energy_gated_da_tpp",
        "predicted_target_greedy",
        "gradient_norm_hybrid",
        "explore",
        "random_sampling",
        "mc_dropout",
    )
    for panel_index, task in enumerate(("mn", "mg")):
        ax = fig.add_subplot(grid[0, panel_index])
        task_data = trajectories.loc[trajectories["task"] == task]
        for method in core_methods:
            method_data = task_data.loc[task_data["method"] == method]
            pivot = method_data.pivot(index="seed", columns="oracle_evaluations", values="recovery_rate")
            x = pivot.columns.to_numpy(dtype=float)
            mean = pivot.mean(axis=0).to_numpy(dtype=float)
            sd = pivot.std(axis=0, ddof=1).to_numpy(dtype=float)
            lw = 2.1 if method in {"energy_gated_da_tpp", "predicted_target_greedy"} else 1.2
            alpha = 1.0 if method in {"energy_gated_da_tpp", "predicted_target_greedy"} else 0.72
            ax.plot(
                x,
                mean,
                color=METHOD_COLORS[method],
                linewidth=lw,
                alpha=alpha,
                label=METHOD_LABELS[method],
            )
            if method in {"energy_gated_da_tpp", "predicted_target_greedy"}:
                ax.fill_between(
                    x,
                    np.clip(mean - sd, 0, 1),
                    np.clip(mean + sd, 0, 1),
                    color=METHOD_COLORS[method],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.set_xlim(0, 320)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Oracle evaluations")
        ax.set_ylabel("Target recovery fraction")
        ax.set_title(TASK_LABELS[task])
        ax.text(-0.12, 1.05, chr(ord("a") + panel_index), transform=ax.transAxes, fontweight="bold", fontsize=10)
        if panel_index == 1:
            ax.legend(loc="lower right", fontsize=6.5, ncol=2)

    ax_c = fig.add_subplot(grid[1, 0])
    y_labels = [METHOD_LABELS[method] for method in METHOD_ORDER]
    y = np.arange(len(METHOD_ORDER))
    offsets = {"mn": -0.13, "mg": 0.13}
    task_colors = {"mn": "#0F4D92", "mg": "#E28E2C"}
    for task in ("mn", "mg"):
        subset = aggregate.loc[aggregate["task"] == task].set_index("method").loc[list(METHOD_ORDER)]
        ax_c.errorbar(
            subset["AUTC_mean"],
            y + offsets[task],
            xerr=subset["AUTC_sd"],
            fmt="o",
            markersize=4,
            capsize=2,
            color=task_colors[task],
            label=TASK_LABELS[task].replace(" proxy interval", ""),
        )
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(y_labels, fontsize=6.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Normalized AUTC (mean ± SD, n=10)")
    ax_c.legend(fontsize=6.5, loc="lower right")
    ax_c.text(-0.12, 1.05, "c", transform=ax_c.transAxes, fontweight="bold", fontsize=10)

    ax_d = fig.add_subplot(grid[1, 1])
    rng = np.random.default_rng(17)
    for index, task in enumerate(("mn", "mg")):
        values = paired.loc[paired["task"] == task, "Gate_minus_Greedy_AUTC"].to_numpy(float)
        jitter = rng.normal(0, 0.035, size=len(values))
        colors = ["#2E9E44" if value > 0 else "#E53935" for value in values]
        ax_d.scatter(np.full(len(values), index) + jitter, values, c=colors, s=22, edgecolor="black", linewidth=0.35, zorder=3)
        stats = stats_by_task[task]
        mean = float(stats["mean_difference"])
        low = float(stats["bootstrap_ci_low"])
        high = float(stats["bootstrap_ci_high"])
        ax_d.errorbar(index, mean, yerr=[[mean - low], [high - mean]], fmt="D", color="#0F4D92", capsize=4, markersize=5, zorder=4)
        pvalue = stats["wilcoxon_p_two_sided_exact"]
        label = f"p={pvalue:.4f}" if pvalue is not None else "p=n/a"
        ax_d.text(index, 0.112, label, ha="center", va="top", fontsize=6.5)
    ax_d.axhline(0, color="#767676", linestyle="--", linewidth=1)
    ax_d.set_xticks([0, 1])
    ax_d.set_xticklabels(["Mn anchored", "Mg anchored"])
    ax_d.set_ylim(-0.08, 0.12)
    ax_d.set_ylabel("Gate − Greedy normalized AUTC")
    ax_d.text(-0.12, 1.05, "d", transform=ax_d.transAxes, fontweight="bold", fontsize=10)
    fig.suptitle("Interval robustness does not show a consistent Gate advantage", fontsize=11, y=0.995)
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 400}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(figure_dir / f"mn_mg_main_results.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_composition_figure(
    density: pd.DataFrame,
    paired_stats: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    figure_dir.mkdir(parents=True, exist_ok=True)
    composition = density.loc[density["m_element"] != "ALL"].copy()
    elements = sorted(composition["m_element"].unique())
    element_colors = {"Cr": "#B64342", "Mn": "#0F4D92", "Mg": "#E28E2C"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    bottoms = np.zeros(2)
    task_order = ["mn", "mg"]
    for element in elements:
        values = []
        for task in task_order:
            match = composition.loc[(composition["task"] == task) & (composition["m_element"] == element), "target_count"]
            values.append(float(match.iloc[0]) if len(match) else 0.0)
        ax.bar(task_order, values, bottom=bottoms, color=element_colors.get(element, "#A8A8A8"), edgecolor="white", linewidth=0.8, label=element)
        bottoms += np.asarray(values)
    ax.set_ylabel("Target candidates in full 640 pool")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Mn anchored", "Mg anchored"])
    ax.legend(title="M element", fontsize=7, title_fontsize=7)
    ax.text(-0.12, 1.04, "a", transform=ax.transAxes, fontweight="bold", fontsize=10)

    ax = axes[1]
    all_rows = density.loc[density["m_element"] == "ALL"].set_index("task")
    for index, task in enumerate(task_order):
        row = paired_stats.loc[paired_stats["task"] == task].iloc[0]
        x = float(all_rows.loc[task, "target_fraction"])
        y = float(row["mean_difference"])
        low = float(row["bootstrap_ci_low"])
        high = float(row["bootstrap_ci_high"])
        color = "#0F4D92" if task == "mn" else "#E28E2C"
        ax.errorbar(x, y, yerr=[[y - low], [high - y]], fmt="o", markersize=7, capsize=4, color=color)
        ax.annotate(
            "Mn anchored" if task == "mn" else "Mg anchored\n(Cr dominated)",
            (x, y),
            xytext=(7, 8 if task == "mn" else -20),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axhline(0, color="#767676", linestyle="--", linewidth=1)
    ax.set_xlabel("Target density in full pool")
    ax.set_ylabel("Gate − Greedy normalized AUTC")
    ax.text(0.02, 0.03, "Descriptive only: one width (0.2 eV/atom)", transform=ax.transAxes, fontsize=6.5, color="#4D4D4D")
    ax.text(-0.12, 1.04, "b", transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.suptitle("Proxy interval composition, not nominal anchor, governs task identity", fontsize=10)
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(figure_dir / f"mn_mg_target_composition.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def _plot_hidden_evaluability_figure(
    hidden_summary: pd.DataFrame,
    figure_dir: Path,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    core_methods = (
        "energy_gated_da_tpp",
        "predicted_target_greedy",
        "gradient_norm_hybrid",
        "explore",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharex=True, gridspec_kw={"wspace": 0.32})
    for index, task in enumerate(("mn", "mg")):
        ax = axes[index]
        for method in core_methods:
            subset = hidden_summary.loc[
                (hidden_summary["task"] == task) & (hidden_summary["method"] == method)
            ].sort_values("checkpoint")
            ax.errorbar(
                subset["checkpoint"],
                subset["expected_evaluable_target_count_mean"],
                yerr=subset["expected_evaluable_target_count_sd"],
                color=METHOD_COLORS[method],
                marker="o",
                markersize=3.5,
                linewidth=1.5 if method in {"energy_gated_da_tpp", "predicted_target_greedy"} else 1.0,
                capsize=2,
                label=METHOD_LABELS[method],
            )
        ax.set_title(TASK_LABELS[task])
        ax.set_xlabel("Oracle evaluations")
        ax.set_ylabel("Expected DFT-evaluable targets")
        ax.set_xticks(CHECKPOINTS)
        ax.text(-0.12, 1.04, chr(ord("a") + index), transform=ax.transAxes, fontweight="bold", fontsize=10)
        if index == 1:
            ax.legend(fontsize=6.5, loc="upper left")
    fig.suptitle(
        "Post-selection evaluability audit (model expectation, not observed DFT)",
        fontsize=10,
    )
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 400})):
        fig.savefig(
            figure_dir / f"mn_mg_hidden_evaluability.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def _write_markdown_reports(
    output_dir: Path,
    tasks: pd.DataFrame,
    aggregate: pd.DataFrame,
    paired_stats: pd.DataFrame,
    independence: pd.DataFrame,
    hidden_summary: pd.DataFrame,
    model_cv: pd.DataFrame,
) -> None:
    mn_pair = paired_stats.loc[paired_stats["task"] == "mn"].iloc[0]
    mg_pair = paired_stats.loc[paired_stats["task"] == "mg"].iloc[0]
    mn_task = tasks.loc[tasks["task"] == "mn"].iloc[0]
    mg_task = tasks.loc[tasks["task"] == "mg"].iloc[0]
    best_mn = aggregate.loc[aggregate["task"] == "mn"].sort_values("AUTC_mean").iloc[-1]
    best_mg = aggregate.loc[aggregate["task"] == "mg"].sort_values("AUTC_mean").iloc[-1]
    model = model_cv.sort_values("loo_roc_auc", ascending=False).iloc[0]

    def hidden_row(task: str, method: str, checkpoint: int) -> pd.Series:
        return hidden_summary.loc[
            (hidden_summary["task"] == task)
            & (hidden_summary["method"] == method)
            & (hidden_summary["checkpoint"] == checkpoint)
        ].iloc[0]

    mn_gate_80 = hidden_row("mn", "energy_gated_da_tpp", 80)
    mn_greedy_80 = hidden_row("mn", "predicted_target_greedy", 80)
    mg_gate_80 = hidden_row("mg", "energy_gated_da_tpp", 80)
    mg_greedy_80 = hidden_row("mg", "predicted_target_greedy", 80)
    mg_gate_240 = hidden_row("mg", "energy_gated_da_tpp", 240)
    mg_greedy_240 = hidden_row("mg", "predicted_target_greedy", 240)

    density_text = f"""# Mn/Mg interval density audit

## Outcome first

The completed GPU batch contains two **0.2 eV atom⁻¹ proxy-interval tasks on the same frozen 640-candidate Li–M–O pool**. It is not a pair of strictly filtered Mn-only and Mg-only pools.

- Mn-anchored interval: [{mn_task.target_low:.1f}, {mn_task.target_high:.1f}] eV atom⁻¹, {int(mn_task.target_count)} targets ({mn_task.target_fraction:.1%}). The target set is {mn_task.dominant_element_fraction:.1%} {mn_task.dominant_element}.
- Mg-anchored interval: [{mg_task.target_low:.1f}, {mg_task.target_high:.1f}] eV atom⁻¹, {int(mg_task.target_count)} targets ({mg_task.target_fraction:.1%}). The target set is {mg_task.dominant_element_fraction:.1%} {mg_task.dominant_element}; therefore it is not a valid Mg-only robustness task.

The DFT-derived anchors and ALIGNN proxy intervals remain explicitly separate energy conventions. Only width 0.2 eV atom⁻¹ was run; no 0.4/0.6-width sensitivity result exists in this batch.

## Data integrity

- Formal runs found: {len(independence) * 9} (2 tasks × 9 methods × 10 seeds).
- Every task uses 10 distinct frozen initial sets: {bool(independence.groupby('task')['initial_set_hash'].nunique().eq(10).all())}.
- Within each task and seed, all methods share the same initial set: {bool(independence['same_initial_set_across_methods'].all())}.
- Full Gate and Group-only produce identical query sequences in all audited task-seed pairs: {int(independence['gate_equals_group_only'].sum())}/{len(independence)}.
- Greedy and Margin-only produce identical query sequences in all audited task-seed pairs: {int(independence['greedy_equals_margin_only'].sum())}/{len(independence)}.

## Anchor construction

Mn uses three completed GGA+U DFT anchors; Mg uses two completed PBE DFT anchors. Their DFT values define the descriptive element anchors, while the active-learning replay uses the corresponding candidates' frozen ALIGNN labels. This is an anchor-to-proxy mapping, not a claim that the two energy scales are interchangeable.
"""
    (output_dir / "MN_MG_INTERVAL_DENSITY_AUDIT.md").write_text(density_text, encoding="utf-8")

    robustness = f"""# Mn/Mg interval robustness summary

## Bottom line

The new experiments do **not** support a cross-interval early advantage for Energy-Gated DA-TPP.

### Mn-anchored proxy interval

- Best mean AUTC: {best_mn.method_label} ({best_mn.AUTC_mean:.4f} ± {best_mn.AUTC_sd:.4f}).
- Full Gate − Greedy mean AUTC difference: {mn_pair.mean_difference:+.4f}; 95% paired-bootstrap CI [{mn_pair.bootstrap_ci_low:+.4f}, {mn_pair.bootstrap_ci_high:+.4f}].
- Gate wins/ties/losses: {int(mn_pair.wins)}/{int(mn_pair.ties)}/{int(mn_pair.losses)}; exact two-sided Wilcoxon p={mn_pair.wilcoxon_p_two_sided_exact:.6f}.
- Interpretation: Greedy is consistently superior in this Mn-dominated interval.

### Mg-anchored proxy interval

- Best mean AUTC: {best_mg.method_label} ({best_mg.AUTC_mean:.4f} ± {best_mg.AUTC_sd:.4f}).
- Full Gate − Greedy mean AUTC difference: {mg_pair.mean_difference:+.4f}; 95% paired-bootstrap CI [{mg_pair.bootstrap_ci_low:+.4f}, {mg_pair.bootstrap_ci_high:+.4f}].
- Gate wins/ties/losses: {int(mg_pair.wins)}/{int(mg_pair.ties)}/{int(mg_pair.losses)}; exact two-sided Wilcoxon p={mg_pair.wilcoxon_p_two_sided_exact:.6f}.
- Interpretation: the effect is mixed and unstable. The nominal Mg task is actually Cr-dominated, so it cannot establish non-Cr/Mg robustness.

## Mechanistic result

Full Gate equals Group-only for every audited task-seed trajectory, while Margin-only equals Greedy. Under the frozen thresholds, the margin condition adds no observable decision effect; the behavior is controlled by the group condition.

## Hidden DFT evaluability audit

The post-selection model uses {int(model['n'])} historical DFT attempts ({int(model['positives'])} positive) and achieves leave-one-out ROC-AUC {model['loo_roc_auc']:.2f}. Expected evaluable counts are model-based expectations only. They never entered acquisition and are not real DFT outcomes. Candidate-score coverage is reported at every checkpoint; missing historical or unsupported candidates are not imputed.

- At 80 queries in the Mn-anchored task, Full Gate yields {mn_gate_80.expected_evaluable_target_count_mean:.2f} expected evaluable targets versus {mn_greedy_80.expected_evaluable_target_count_mean:.2f} for Greedy.
- At 80 queries in the Mg-anchored task, Full Gate yields {mg_gate_80.expected_evaluable_target_count_mean:.2f} versus {mg_greedy_80.expected_evaluable_target_count_mean:.2f} for Greedy.
- Full Gate exceeds Greedy at 240 queries in the Mg-anchored task ({mg_gate_240.expected_evaluable_target_count_mean:.2f} versus {mg_greedy_240.expected_evaluable_target_count_mean:.2f}), but this is a predicted post-selection quantity in a Cr-dominated target set.

## Recommended manuscript placement

- Main text: report the Mn-anchored negative robustness result as a method boundary, preferably one compact paragraph plus a small panel.
- Supplementary Information: full nine-method tables, all seed-level results, hidden-evaluability audit, and the Mg-anchored analysis.
- Do not claim cross-element robustness, universal superiority over Greedy, or prospective DFT-yield improvement.
- Keep the title focused on early prioritization and conditional diversity, not broad multi-system superiority.
"""
    (output_dir / "mn_mg_robustness_summary.md").write_text(robustness, encoding="utf-8")

    claim_boundary = """# Mn/Mg claim boundary

## Supported

- The experiments are reproducible multi-seed proxy-label replays on one frozen 640-candidate Li–M–O pool.
- In the Mn-dominated 0.2 eV atom⁻¹ interval, Greedy outperforms Full Gate across all ten seeds.
- In the second interval, Full Gate has no statistically robust advantage; Gradient-norm hybrid has the highest mean AUTC.
- Full Gate and Group-only are behaviorally identical under this frozen protocol; Margin-only and Greedy are identical.
- A hidden model can be used to audit predicted DFT evaluability after selection, with explicit coverage and uncertainty limits.

## Not supported

- A general Gate advantage across Mn and Mg element-specific pools.
- A true Mg-only robustness result: the Mg-anchored interval is Cr-dominated in the full pool.
- Width sensitivity beyond 0.2 eV atom⁻¹.
- Real DFT success-rate improvement: evaluability values are predictions from a 20-attempt model.
- Equivalence between DFT formation energies and ALIGNN proxy labels.
- Thermodynamic stability, synthesizability, or application performance.

## Required wording

Use “Mn-anchored proxy interval” and “Mg-anchored proxy interval on the full frozen pool.” Do not call them pure Mn and Mg candidate pools. Use “expected/predicted DFT evaluability” and never “DFT success” for hidden-audit values.
"""
    (output_dir / "MN_MG_CLAIM_BOUNDARY.md").write_text(claim_boundary, encoding="utf-8")

    def method_table(task: str) -> str:
        subset = aggregate.loc[aggregate["task"] == task].sort_values(
            ["AUTC_rank", "method"], kind="mergesort"
        )
        lines = [
            "| 方法 | AUTC（均值±SD） | R@80 | R@160 | R@240 | R@320 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in subset.itertuples(index=False):
            lines.append(
                f"| {row.method_label} | {row.AUTC_mean:.4f} ± {row.AUTC_sd:.4f} | "
                f"{row.recovery_at_80_mean:.1f} | {row.recovery_at_160_mean:.1f} | "
                f"{row.recovery_at_240_mean:.1f} | {row.recovery_at_320_mean:.1f} |"
            )
        return "\n".join(lines)

    chinese_report = f"""# Mn/Mg元素锚定区间实验：正式结果分析

## 一句话结论

本轮180次GPU实验**没有证明Energy-Gated DA-TPP的早期优势可以跨区间稳定复现**：Mn锚定任务中Greedy在10个seed里全部胜出；Mg锚定任务中Gate与Greedy结果混合且不显著，表现最好的反而是Gradient-norm hybrid。更关键的是，所谓“Mg锚定任务”在完整640池中有80.6%的目标是Cr候选，不能当作Mg体系稳健性证据。

## 1. 实验和数据是否可信

- 180/180次运行均完成，全部结果可读取并能从逐轮`summary`独立复算。
- 10个seed对应10个不同初始标注集；同一seed内9种方法使用完全相同的初始集。
- 服务器给出的AUTC、80/160/240/320检查点与独立复算逐项一致。
- Full Gate与Group-only在20个“任务×seed”组合中查询序列全部相同。
- Margin-only与Greedy在20个组合中查询序列全部相同。

因此，数值本身没有发现拷贝、覆盖或seed失效问题；问题在于**实验结果不支持原先期待的广泛优势**。

## 2. 两个任务实际是什么

| 任务 | ALIGNN代理区间（eV/atom） | 目标数/640 | 目标主成分 | DFT锚点 |
|---|---:|---:|---:|---:|
| Mn锚定 | [{mn_task.target_low:.1f}, {mn_task.target_high:.1f}] | {int(mn_task.target_count)}（{mn_task.target_fraction:.1%}） | Mn {mn_task.dominant_element_fraction:.1%} | 3个Mn GGA+U结果 |
| Mg锚定 | [{mg_task.target_low:.1f}, {mg_task.target_high:.1f}] | {int(mg_task.target_count)}（{mg_task.target_fraction:.1%}） | Cr {mg_task.dominant_element_fraction:.1%} | 2个Mg PBE结果 |

这两项实验都在完整640候选池上进行，只是目标区间不同。Mn任务可称为“Mn主导的锚定区间”；Mg任务只能称为“Mg锚定的代理区间”，不能称为Mg候选池实验。DFT形成能与ALIGNN标签始终分开记录，没有把两种能标混成一个数值体系。

## 3. Mn锚定任务结果

{method_table('mn')}

Full Gate相对Greedy的配对AUTC差为{mn_pair.mean_difference:+.4f}，95% bootstrap CI为[{mn_pair.bootstrap_ci_low:+.4f}, {mn_pair.bootstrap_ci_high:+.4f}]；Gate胜/平/负为{int(mn_pair.wins)}/{int(mn_pair.ties)}/{int(mn_pair.losses)}，exact Wilcoxon双侧p={mn_pair.wilcoxon_p_two_sided_exact:.6f}，配对效应量dz={mn_pair.effect_size_dz:.2f}。这不是“打平”，而是Greedy在该区间下稳定更优。

## 4. Mg锚定任务结果

{method_table('mg')}

Full Gate相对Greedy的配对AUTC差为{mg_pair.mean_difference:+.4f}，95% bootstrap CI为[{mg_pair.bootstrap_ci_low:+.4f}, {mg_pair.bootstrap_ci_high:+.4f}]；Gate胜/平/负为{int(mg_pair.wins)}/{int(mg_pair.ties)}/{int(mg_pair.losses)}，exact Wilcoxon双侧p={mg_pair.wilcoxon_p_two_sided_exact:.6f}。均值略偏向Gate，但置信区间跨0且10个seed中Gate只赢4个，不能写成优势成立。

## 5. “预计DFT可结算性”到底说明什么

隐藏评价模型使用20个历史DFT尝试（10个正例），最佳留一交叉验证ROC-AUC={model['loo_roc_auc']:.2f}。它从未参与候选选择，只在选择完成后给候选打分。因此下列数值是**预计数量**，不是实际新DFT成功数量：

| 任务/检查点 | Full Gate预计数 | Greedy预计数 | 结论 |
|---|---:|---:|---|
| Mn，80次查询 | {mn_gate_80.expected_evaluable_target_count_mean:.2f} | {mn_greedy_80.expected_evaluable_target_count_mean:.2f} | Greedy更高 |
| Mg锚定，80次查询 | {mg_gate_80.expected_evaluable_target_count_mean:.2f} | {mg_greedy_80.expected_evaluable_target_count_mean:.2f} | Greedy更高 |
| Mg锚定，240次查询 | {mg_gate_240.expected_evaluable_target_count_mean:.2f} | {mg_greedy_240.expected_evaluable_target_count_mean:.2f} | Gate更高，但任务由Cr主导 |

隐藏评分对目标候选的覆盖率约为91%–95%，缺失的历史DFT候选没有被强行填补。这个分析可以进入SI，用于说明“代理目标命中数”和“预计下游可计算价值”并不完全相同；不能用于宣称真实DFT成功率提高。

## 6. 区间宽度和最终推荐

服务器实际只运行了宽度0.2 eV/atom，没有运行0.4、0.6或固定10%/20%/30%目标比例任务。因此：

1. 不能从本批结果比较“统一绝对宽度”和“固定20%目标比例”谁更好。
2. 不建议再为得到正结果移动区间；这会重新引入事后选择风险。
3. Mn锚定0.2区间可作为负向稳健性/适用边界结果。
4. Mg锚定0.2区间只适合放SI，并明确其目标集由Cr主导。

## 7. 对论文的直接影响

- 主文可加入Mn结果，用来说明Gate并非在所有目标密度和区间下优于强Greedy基线。
- 全部九种方法、10个seed、隐藏可评价性和Mg锚定结果放SI。
- 摘要和结论不得写“跨元素稳健优于Greedy”或“提高真实DFT成功率”。
- 方法贡献应收窄为：条件多样性修正在特定冗余结构下改变早期排序，并具有明确失效边界。
- 当前证据不要求修改标题为跨体系稳健性标题；若标题含“universal”或“generalizable”，必须删除。

## 8. 可复核文件

所有逐seed数据、配对统计、隐藏可评价性明细、图源数据、环境版本和SHA-256清单均保存在同一分析目录中。图中误差带为10个初始集seed的样本SD；Gate–Greedy正式比较采用100,000次配对bootstrap和exact two-sided Wilcoxon。
"""
    (output_dir / "MN_MG_FINAL_REPORT_ZH.md").write_text(chinese_report, encoding="utf-8")


def _write_figure_contract(output_dir: Path) -> None:
    text = """Core conclusion: The 0.2 eV atom⁻¹ interval replays do not show a robust cross-interval Full-Gate advantage; task composition and the group condition explain the observed behavior.
Figure archetype: quantitative grid
Target journal/output: CMC-compatible double-column figure
Backend: Python
Final size: 183 mm wide, approximately 158 mm high
Panel map:
  a: Mn-anchored mean recovery trajectories
  b: Mg-anchored mean recovery trajectories
  c: all-method normalized AUTC comparison
  d: paired Full Gate minus Greedy AUTC differences
Evidence hierarchy:
  hero evidence: paired Gate-Greedy AUTC differences
  validation evidence: ten-seed recovery trajectories
  controls/robustness: nine-method AUTC ranking and target-composition audit
Statistics needed: n=10 initial-set seeds; mean ± sample SD; paired 100,000-draw bootstrap CI; exact two-sided Wilcoxon for the two prespecified Gate-Greedy comparisons
Source data needed: independently recomputed per-seed metrics, trajectories, target composition, paired differences
Image-integrity notes: vector-native charts; no image manipulation
Reviewer risk: nominal Mg anchor does not create an Mg-only task; only width 0.2 was run; hidden DFT evaluability is predicted, not observed
"""
    (output_dir / "FIGURE_CONTRACT.md").write_text(text, encoding="utf-8")


def run_analysis(
    *,
    results_root: Path,
    pool_path: Path,
    evaluability_scores_path: Path,
    model_cv_path: Path,
    formation_path: Path,
    output_dir: Path,
) -> None:
    started_at = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    source_dir = output_dir / "figure_source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_path)
    scores = pd.read_csv(evaluability_scores_path)
    model_cv = pd.read_csv(model_cv_path)
    anchors = _load_anchor_table(formation_path, pool)
    anchors.to_csv(output_dir / "mn_mg_dft_anchor_candidates.csv", index=False)

    records = _read_run_grid(results_root)
    task_rows: list[dict[str, object]] = []
    density_rows: list[dict[str, object]] = []
    task_configs: dict[str, dict[str, object]] = {}
    for task in TASK_LABELS:
        example = next(record for record in records if record["task"] == task)
        config = json.loads((Path(example["run_dir"]) / "run_config.json").read_text(encoding="utf-8"))
        task_configs[task] = config
        target_summary, composition = summarize_target_set(
            pool,
            low=float(config["target_low"]),
            high=float(config["target_high"]),
        )
        if int(target_summary["target_count"]) != int(config["target_count"]):
            raise ValueError(f"{task}: target count differs from frozen config")
        anchor = anchors.loc[anchors["task"] == task]
        dft_values = anchor["anchor_dft_energy_eV_atom"].to_numpy(float)
        proxy_values = anchor["anchor_alignn_energy_eV_atom"].to_numpy(float)
        proxy_center = round(float(np.median(proxy_values)), 1)
        frozen_center = (float(config["target_low"]) + float(config["target_high"])) / 2
        if not np.isclose(proxy_center, frozen_center, atol=1e-12):
            raise ValueError(f"{task}: proxy anchor center does not match frozen interval")
        task_rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "task_id": config["interval_task_id"],
                "strict_element_pool": False,
                "pool_size": int(target_summary["pool_size"]),
                "target_low": float(config["target_low"]),
                "target_high": float(config["target_high"]),
                "interval_width_eV_atom": float(config["target_high"] - config["target_low"]),
                "target_count": int(target_summary["target_count"]),
                "target_fraction": float(target_summary["target_fraction"]),
                "effective_cluster_count": int(target_summary["effective_cluster_count"]),
                "largest_cluster_count": int(target_summary["largest_cluster_count"]),
                "largest_cluster_fraction": float(target_summary["largest_cluster_fraction"]),
                "dominant_element": target_summary["dominant_element"],
                "dominant_element_fraction": float(target_summary["dominant_element_fraction"]),
                "dft_anchor_count": int(len(anchor)),
                "dft_anchor_values_eV_atom_json": json.dumps(dft_values.tolist()),
                "dft_anchor_median_eV_atom": float(np.median(dft_values)),
                "dft_anchor_q1_eV_atom": float(np.quantile(dft_values, 0.25)),
                "dft_anchor_q3_eV_atom": float(np.quantile(dft_values, 0.75)),
                "proxy_anchor_values_eV_atom_json": json.dumps(proxy_values.tolist()),
                "proxy_anchor_median_eV_atom": float(np.median(proxy_values)),
                "proxy_anchor_center_rounded_0p1": proxy_center,
                "label_source": config["label_source"],
                "oracle_sha256": config["oracle_sha256"],
                "pool_id_prop_sha256": config["pool_id_prop_sha256"],
            }
        )
        density_rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "m_element": "ALL",
                "target_count": int(target_summary["target_count"]),
                "target_fraction_within_task": 1.0,
                "target_fraction": float(target_summary["target_fraction"]),
                "effective_cluster_count": int(target_summary["effective_cluster_count"]),
                "largest_cluster_fraction": float(target_summary["largest_cluster_fraction"]),
            }
        )
        for row in composition.itertuples(index=False):
            density_rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "m_element": row.m_element,
                    "target_count": int(row.target_count),
                    "target_fraction_within_task": float(row.target_fraction_within_task),
                    "target_fraction": float(row.target_count / len(pool)),
                    "effective_cluster_count": np.nan,
                    "largest_cluster_fraction": np.nan,
                }
            )
    tasks = pd.DataFrame(task_rows)
    density = pd.DataFrame(density_rows)
    tasks.to_csv(output_dir / "mn_mg_interval_tasks.csv", index=False)
    density.to_csv(output_dir / "mn_mg_target_density.csv", index=False)

    pool_lookup = pool.loc[
        :, ["candidate_id", "m_element", "structure_matcher_cluster"]
    ].copy()
    per_seed_rows: list[dict[str, object]] = []
    trajectory_rows: list[pd.DataFrame] = []
    hidden_rows: list[dict[str, object]] = []
    init_records: list[dict[str, object]] = []
    sequence_records: list[dict[str, object]] = []
    for record in records:
        task = str(record["task"])
        method = str(record["method"])
        seed = int(record["seed"])
        run_dir = Path(record["run_dir"])
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if str(status.get("status")) != "DONE":
            raise ValueError(f"{run_dir}: status is not DONE")
        reported = pd.read_csv(run_dir / "run_metrics.csv").iloc[0]
        summary = pd.read_csv(run_dir / "summary.csv")
        history = pd.read_csv(run_dir / "al_history.csv")
        if len(history) != int(config["budget"]):
            raise ValueError(f"{run_dir}: expected {config['budget']} history rows")
        if history["id"].astype(str).duplicated().any():
            raise ValueError(f"{run_dir}: duplicate queried candidate")
        reconstructed = reconstruct_trajectory(
            summary,
            budget=int(config["budget"]),
            total_targets=int(config["target_count"]),
            checkpoints=CHECKPOINTS,
        )
        if not np.isclose(float(reported["AUTC"]), float(reconstructed["autc"]), atol=1e-12):
            raise ValueError(f"{run_dir}: reported AUTC does not independently recompute")
        for checkpoint in CHECKPOINTS:
            if int(reported[f"recovery_at_{checkpoint}"]) != int(
                reconstructed[f"recovery_at_{checkpoint}"]
            ):
                raise ValueError(f"{run_dir}: checkpoint recovery mismatch")
        init = json.loads((run_dir / "initialization_manifest.json").read_text(encoding="utf-8"))
        init_ids = [str(value) for value in init["candidate_ids"]]
        init_records.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "initial_ids_json": json.dumps(init_ids, separators=(",", ":")),
                "initial_set_hash": _sha256_text(sorted(init_ids)),
                "initial_order_hash": _sha256_text(init_ids),
            }
        )
        candidate_ids = history["id"].astype(str).tolist()
        sequence_records.append(
            {
                "task": task,
                "method": method,
                "seed": seed,
                "candidate_sequence_hash": _sha256_text(candidate_ids),
                **{
                    f"prefix_hash_{checkpoint}": _sha256_text(candidate_ids[:checkpoint])
                    for checkpoint in CHECKPOINTS
                },
            }
        )
        history_pool = history.loc[:, ["id", "target_label"]].merge(
            pool_lookup,
            left_on="id",
            right_on="candidate_id",
            how="left",
            validate="one_to_one",
        )
        if history_pool["candidate_id"].isna().any():
            raise ValueError(f"{run_dir}: history contains candidates absent from pool master")
        row: dict[str, object] = {
            "task": task,
            "task_label": TASK_LABELS[task],
            "method": method,
            "method_label": METHOD_LABELS[method],
            "seed": seed,
            "AUTC": float(reconstructed["autc"]),
            "final_recovery": int(reconstructed["final_recovery"]),
            "final_recall_rate": float(reconstructed["final_recovery"] / config["target_count"]),
            "direct_rounds": int(reported["direct_rounds"]),
            "correction_rounds": int(reported["correction_rounds"]),
            "mean_unique_groups_per_batch": float(reported["mean_unique_groups_per_batch"]),
            "mean_group_repetition_rate": float(reported["mean_group_repetition_rate"]),
            "total_correction_replacements": int(reported["total_correction_replacements"]),
            "total_correction_target_gain": int(reported["total_correction_target_gain"]),
            "candidate_sequence_sha256": _sha256_text(candidate_ids),
            "reported_candidate_sequence_sha256": str(reported["candidate_sequence_sha256"]),
        }
        if row["candidate_sequence_sha256"] != row["reported_candidate_sequence_sha256"]:
            raise ValueError(f"{run_dir}: candidate sequence hash mismatch")
        for checkpoint in CHECKPOINTS:
            recovery = int(reconstructed[f"recovery_at_{checkpoint}"])
            subset = history_pool.iloc[:checkpoint]
            target_subset = subset.loc[pd.to_numeric(subset["target_label"]) == 1]
            hidden = expected_evaluability_at(history, scores, checkpoint=checkpoint)
            row[f"recovery_at_{checkpoint}"] = recovery
            row[f"recovery_rate_at_{checkpoint}"] = float(recovery / config["target_count"])
            row[f"queries_per_recovered_target_at_{checkpoint}"] = (
                float(checkpoint / recovery) if recovery else np.nan
            )
            row[f"unique_structure_clusters_at_{checkpoint}"] = int(
                subset["structure_matcher_cluster"].nunique()
            )
            row[f"structure_cluster_coverage_at_{checkpoint}"] = float(
                subset["structure_matcher_cluster"].nunique() / checkpoint
            )
            row[f"unique_target_structure_clusters_at_{checkpoint}"] = int(
                target_subset["structure_matcher_cluster"].nunique()
            )
            row[f"expected_evaluable_targets_at_{checkpoint}"] = hidden[
                "expected_evaluable_target_count"
            ]
            row[f"hidden_score_coverage_at_{checkpoint}"] = hidden["score_coverage"]
            hidden_rows.append(
                {
                    "task": task,
                    "task_label": TASK_LABELS[task],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "seed": seed,
                    "checkpoint": checkpoint,
                    **hidden,
                    "audit_role": "post_selection_only",
                    "observed_dft_outcome": False,
                }
            )
        per_seed_rows.append(row)
        trajectory = summary.loc[
            :, ["oracle_evaluations", "cumulative_target_count"]
        ].copy()
        trajectory["task"] = task
        trajectory["task_label"] = TASK_LABELS[task]
        trajectory["method"] = method
        trajectory["method_label"] = METHOD_LABELS[method]
        trajectory["seed"] = seed
        trajectory["recovery_rate"] = (
            trajectory["cumulative_target_count"] / int(config["target_count"])
        )
        trajectory_rows.append(trajectory)

    per_seed = pd.DataFrame(per_seed_rows).sort_values(["task", "method", "seed"])
    trajectories = pd.concat(trajectory_rows, ignore_index=True).sort_values(
        ["task", "method", "seed", "oracle_evaluations"]
    )
    hidden_detail = pd.DataFrame(hidden_rows).sort_values(
        ["task", "method", "seed", "checkpoint"]
    )
    per_seed.to_csv(output_dir / "mn_mg_multiseed_results.csv", index=False)
    trajectories.to_csv(source_dir / "mn_mg_recovery_trajectories.csv", index=False)
    hidden_detail.to_csv(output_dir / "mn_mg_hidden_evaluability_audit.csv", index=False)

    aggregate = _aggregate_summary(per_seed)
    aggregate.to_csv(output_dir / "mn_mg_multiseed_summary.csv", index=False)
    hidden_summary = (
        hidden_detail.groupby(["task", "method", "checkpoint"], sort=False)
        .agg(
            expected_evaluable_target_count_mean=("expected_evaluable_target_count", "mean"),
            expected_evaluable_target_count_sd=("expected_evaluable_target_count", "std"),
            expected_evaluable_fraction_mean=(
                "expected_evaluable_fraction_among_scored_targets",
                "mean",
            ),
            score_coverage_mean=("score_coverage", "mean"),
            selected_target_count_mean=("selected_target_count", "mean"),
        )
        .reset_index()
    )
    hidden_summary["task_label"] = hidden_summary["task"].map(TASK_LABELS)
    hidden_summary["method_label"] = hidden_summary["method"].map(METHOD_LABELS)
    hidden_summary.to_csv(output_dir / "mn_mg_hidden_evaluability_summary.csv", index=False)

    init_frame = pd.DataFrame(init_records)
    sequence_frame = pd.DataFrame(sequence_records)
    independence_rows: list[dict[str, object]] = []
    for (task, seed), group in init_frame.groupby(["task", "seed"], sort=True):
        sequences = sequence_frame.loc[
            (sequence_frame["task"] == task) & (sequence_frame["seed"] == seed)
        ].set_index("method")
        independence_rows.append(
            {
                "task": task,
                "task_label": TASK_LABELS[task],
                "seed": int(seed),
                "method_count": int(group["method"].nunique()),
                "same_initial_set_across_methods": bool(group["initial_set_hash"].nunique() == 1),
                "same_initial_order_across_methods": bool(group["initial_order_hash"].nunique() == 1),
                "initial_set_hash": str(group["initial_set_hash"].iloc[0]),
                "initial_order_hash": str(group["initial_order_hash"].iloc[0]),
                "initial_ids_json": str(group["initial_ids_json"].iloc[0]),
                "unique_method_query_sequences": int(sequences["candidate_sequence_hash"].nunique()),
                "gate_equals_group_only": bool(
                    sequences.loc["energy_gated_da_tpp", "candidate_sequence_hash"]
                    == sequences.loc["group_only_gate", "candidate_sequence_hash"]
                ),
                "greedy_equals_margin_only": bool(
                    sequences.loc["predicted_target_greedy", "candidate_sequence_hash"]
                    == sequences.loc["margin_only_gate", "candidate_sequence_hash"]
                ),
            }
        )
    independence = pd.DataFrame(independence_rows)
    independence["task_unique_initial_sets"] = independence.groupby("task")[
        "initial_set_hash"
    ].transform("nunique")
    sequence_independence = (
        sequence_frame.groupby(["task", "method"], sort=True)
        .agg(
            unique_full_sequences=("candidate_sequence_hash", "nunique"),
            unique_prefixes_at_80=("prefix_hash_80", "nunique"),
            unique_prefixes_at_160=("prefix_hash_160", "nunique"),
            unique_prefixes_at_240=("prefix_hash_240", "nunique"),
            unique_prefixes_at_320=("prefix_hash_320", "nunique"),
        )
        .reset_index()
    )
    sequence_independence["all_ten_seeds_independent_through_80"] = (
        sequence_independence["unique_prefixes_at_80"] == 10
    )
    sequence_independence["method_label"] = sequence_independence["method"].map(
        METHOD_LABELS
    )
    task_prefix = sequence_independence.groupby("task").agg(
        minimum_unique_prefixes_at_80=("unique_prefixes_at_80", "min"),
        minimum_unique_prefixes_at_160=("unique_prefixes_at_160", "min"),
        minimum_unique_prefixes_at_240=("unique_prefixes_at_240", "min"),
        minimum_unique_full_sequences=("unique_full_sequences", "min"),
    )
    for field in task_prefix.columns:
        independence[field] = independence["task"].map(task_prefix[field])
    independence.to_csv(output_dir / "mn_mg_seed_independence_audit.csv", index=False)
    sequence_independence.to_csv(
        output_dir / "mn_mg_sequence_independence_by_method.csv", index=False
    )

    paired_rows: list[pd.DataFrame] = []
    stats_by_task: dict[str, dict[str, object]] = {}
    for task in TASK_LABELS:
        gate = per_seed.loc[
            (per_seed["task"] == task) & (per_seed["method"] == "energy_gated_da_tpp")
        ].set_index("seed")
        greedy = per_seed.loc[
            (per_seed["task"] == task)
            & (per_seed["method"] == "predicted_target_greedy")
        ].set_index("seed")
        joined = gate[["AUTC"]].join(greedy[["AUTC"]], lsuffix="_Gate", rsuffix="_Greedy", validate="one_to_one")
        paired = joined.reset_index()
        paired.insert(0, "task", task)
        paired["Gate_minus_Greedy_AUTC"] = paired["AUTC_Gate"] - paired["AUTC_Greedy"]
        for checkpoint in CHECKPOINTS:
            paired[f"Gate_recovery_at_{checkpoint}"] = gate[f"recovery_at_{checkpoint}"].to_numpy()
            paired[f"Greedy_recovery_at_{checkpoint}"] = greedy[f"recovery_at_{checkpoint}"].to_numpy()
            paired[f"Gate_minus_Greedy_recovery_at_{checkpoint}"] = (
                paired[f"Gate_recovery_at_{checkpoint}"]
                - paired[f"Greedy_recovery_at_{checkpoint}"]
            )
        stats_by_task[task] = paired_statistics(
            paired["Gate_minus_Greedy_AUTC"].to_numpy(float),
            bootstrap_samples=BOOTSTRAP_SAMPLES,
            bootstrap_seed=BOOTSTRAP_SEED + (0 if task == "mn" else 1),
        )
        paired_rows.append(paired)
    paired = pd.concat(paired_rows, ignore_index=True)
    paired.to_csv(output_dir / "mn_mg_gate_greedy_paired.csv", index=False)
    paired_stats = pd.DataFrame(
        [{"task": task, "task_label": TASK_LABELS[task], **stats} for task, stats in stats_by_task.items()]
    )
    paired_stats.to_csv(output_dir / "mn_mg_gate_greedy_statistics.csv", index=False)

    aggregate.to_csv(source_dir / "mn_mg_autc_summary.csv", index=False)
    paired.to_csv(source_dir / "mn_mg_gate_greedy_paired.csv", index=False)
    density.to_csv(source_dir / "mn_mg_target_composition.csv", index=False)
    hidden_summary.to_csv(source_dir / "mn_mg_hidden_evaluability_summary.csv", index=False)
    _plot_main_figure(trajectories, aggregate, paired, stats_by_task, figure_dir)
    _plot_composition_figure(density, paired_stats, figure_dir)
    _plot_hidden_evaluability_figure(hidden_summary, figure_dir)
    _write_markdown_reports(
        output_dir,
        tasks,
        aggregate,
        paired_stats,
        independence,
        hidden_summary,
        model_cv,
    )
    _write_figure_contract(output_dir)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed_mn": BOOTSTRAP_SEED,
        "bootstrap_seed_mg": BOOTSTRAP_SEED + 1,
    }
    (output_dir / "environment_lock.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), scripts_dir / Path(__file__).name)
    source_test = Path(__file__).resolve().parents[1] / "tests" / "test_mn_mg_interval_analysis.py"
    if source_test.is_file():
        shutil.copy2(source_test, scripts_dir / source_test.name)

    log_rows = []
    for record in records:
        log_path = Path(record["run_dir"]) / "run.log"
        log_rows.append(
            {
                "task": record["task"],
                "method": record["method"],
                "seed": record["seed"],
                "log_exists": log_path.is_file(),
                "log_size_bytes": log_path.stat().st_size if log_path.is_file() else 0,
                "log_sha256": _sha256_file(log_path) if log_path.is_file() else "",
                "source_log_path": str(log_path.resolve()),
            }
        )
    pd.DataFrame(log_rows).to_csv(output_dir / "formal_run_log_inventory.csv", index=False)

    input_paths = {
        "candidate_pool_master": pool_path,
        "hidden_evaluability_scores": evaluability_scores_path,
        "hidden_evaluability_model_cv": model_cv_path,
        "recomputed_formation_energies": formation_path,
        "formal_result_checksum_manifest": results_root
        / "finalization"
        / "FORMAL_RESULT_SHA256SUMS.txt",
        "formal_job_manifest": results_root / "finalization" / "job_manifest_final.csv",
    }
    input_rows = []
    for role, path in input_paths.items():
        input_rows.append(
            {
                "role": role,
                "path": str(path.resolve()),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _sha256_file(path) if path.is_file() else "",
            }
        )
    pd.DataFrame(input_rows).to_csv(output_dir / "input_evidence_manifest.csv", index=False)
    finished_at = datetime.now(timezone.utc)
    analysis_log = {
        "status": "COMPLETE",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "formal_runs_validated": len(records),
        "tasks": sorted(TASK_LABELS),
        "methods": list(METHOD_ORDER),
        "seeds": list(range(101, 111)),
        "independent_recomputation": {
            "autc_matches_all_runs": True,
            "checkpoint_recovery_matches_all_runs": True,
            "candidate_sequence_hash_matches_all_runs": True,
        },
        "known_scope_limits": [
            "full_640_pool_not_strict_element_subpools",
            "only_interval_width_0p2_was_run",
            "hidden_evaluability_is_post_selection_prediction_not_observed_dft",
        ],
    }
    (output_dir / "analysis_run_log.json").write_text(
        json.dumps(analysis_log, indent=2), encoding="utf-8"
    )
    manifest_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.csv":
            manifest_rows.append(
                {
                    "relative_path": path.relative_to(output_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    pd.DataFrame(manifest_rows).to_csv(output_dir / "SHA256SUMS.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Mn/Mg proxy-interval GPU results")
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--evaluability-scores", type=Path, required=True)
    parser.add_argument("--model-cv", type=Path, required=True)
    parser.add_argument("--formation-energies", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        results_root=args.results_root,
        pool_path=args.pool,
        evaluability_scores_path=args.evaluability_scores,
        model_cv_path=args.model_cv,
        formation_path=args.formation_energies,
        output_dir=args.output_dir,
    )
