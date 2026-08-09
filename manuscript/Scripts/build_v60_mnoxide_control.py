from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
ARCHIVE = (
    PROJECT_ROOT
    / "gpu_mn_mg_cgcnn_20260803"
    / "paper_rebuild"
    / "baseline_snapshot"
    / "archive"
    / "experiments"
    / "reproducibility"
    / "results"
)
COHORTS = (
    ARCHIVE / "paired_two_dataset_confirmation_20260712" / "runs" / "mnoxide",
    ARCHIVE
    / "paired_two_dataset_confirmation_seeds_10_14_20260713"
    / "runs"
    / "mnoxide",
)
METHODS = {
    "energy_gated_da_tpp": "Energy-Gated DA-TPP",
    "predicted_distance_greedy": "Predicted-Target Greedy",
}
SEEDS = tuple(range(5, 15))
GROUP_MAP = (
    PROJECT_ROOT
    / "gpu_mn_mg_cgcnn_20260803"
    / "experiments"
    / "active_learning"
    / "configs"
    / "group_keys"
    / "egdatpp_psfix_v1"
    / "mnoxide_element_system_current.csv"
)

GATE_COLOR = "#9467BD"
GREEDY_COLOR = "#2CA02C"


def apply_original_figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.2,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.65,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#B0B0B0", linewidth=0.55, alpha=0.38)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3.2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_run_file(method: str, seed: int, filename: str) -> Path:
    matches = [root / method / f"seed_{seed}" / filename for root in COHORTS]
    existing = [path for path in matches if path.exists()]
    if len(existing) != 1:
        raise RuntimeError(
            f"Expected one {filename} for {method} seed {seed}; found {existing}"
        )
    return existing[0]


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[pd.DataFrame] = []
    history_rows: list[pd.DataFrame] = []
    for method in METHODS:
        for seed in SEEDS:
            metrics_path = find_run_file(method, seed, "run_metrics.csv")
            history_path = find_run_file(method, seed, "al_history.csv")

            metrics = pd.read_csv(metrics_path)
            if len(metrics) != 1:
                raise RuntimeError(f"Unexpected row count in {metrics_path}")
            metrics["source_sha256"] = sha256(metrics_path)
            metric_rows.append(metrics)

            history = pd.read_csv(history_path)
            history["seed"] = seed
            history["method"] = method
            history["query_in_round"] = history.groupby("iteration").cumcount() + 1
            history["query"] = (
                (history["iteration"].astype(int) - 1) * 16
                + history["query_in_round"].astype(int)
            )
            history["cumulative_targets"] = history["target_label"].astype(int).cumsum()
            history_rows.append(
                history[["method", "seed", "query", "id", "target_label", "cumulative_targets"]]
            )

    metrics = pd.concat(metric_rows, ignore_index=True)
    histories = pd.concat(history_rows, ignore_index=True)
    return metrics, histories


def audit_pairing(metrics: pd.DataFrame, histories: pd.DataFrame) -> pd.DataFrame:
    audit_rows = []
    for seed in SEEDS:
        gate = histories[
            (histories["seed"] == seed)
            & (histories["method"] == "energy_gated_da_tpp")
        ].sort_values("query")
        greedy = histories[
            (histories["seed"] == seed)
            & (histories["method"] == "predicted_distance_greedy")
        ].sort_values("query")
        ids_equal = gate["id"].astype(str).tolist() == greedy["id"].astype(str).tolist()
        target_paths_equal = gate["cumulative_targets"].tolist() == greedy[
            "cumulative_targets"
        ].tolist()
        batch_sets_equal = all(
            set(
                gate[((gate["query"] - 1) // 16) == batch_index]["id"].astype(str)
            )
            == set(
                greedy[((greedy["query"] - 1) // 16) == batch_index]["id"].astype(str)
            )
            for batch_index in range(20)
        )
        checkpoint_paths_equal = all(
            int(gate.loc[gate["query"] == query, "cumulative_targets"].iloc[0])
            == int(greedy.loc[greedy["query"] == query, "cumulative_targets"].iloc[0])
            for query in range(16, 321, 16)
        )

        gate_metric = metrics[
            (metrics["seed"] == seed)
            & (metrics["method"] == "energy_gated_da_tpp")
        ].iloc[0]
        greedy_metric = metrics[
            (metrics["seed"] == seed)
            & (metrics["method"] == "predicted_distance_greedy")
        ].iloc[0]
        audit_rows.append(
            {
                "seed": seed,
                "candidate_sequence_identical": ids_equal,
                "target_recovery_path_identical": target_paths_equal,
                "batch_candidate_sets_identical": batch_sets_equal,
                "batch_checkpoint_recovery_identical": checkpoint_paths_equal,
                "gate_AUTC": float(gate_metric["AUTC"]),
                "greedy_AUTC": float(greedy_metric["AUTC"]),
                "paired_AUTC_difference": float(gate_metric["AUTC"])
                - float(greedy_metric["AUTC"]),
                "gate_direct_rounds": int(gate_metric["direct_rounds"]),
                "gate_correction_rounds": int(gate_metric["correction_rounds"]),
                "effective_replacements": int(
                    gate_metric["total_correction_replacements"]
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)
    if not audit["batch_candidate_sets_identical"].all():
        raise RuntimeError("Gate and Greedy batch candidate sets are not identical")
    if not audit["batch_checkpoint_recovery_identical"].all():
        raise RuntimeError("Gate and Greedy batch-checkpoint recoveries differ")
    return audit


def batch_trajectory(histories: pd.DataFrame) -> pd.DataFrame:
    checkpoints = np.arange(16, 321, 16)
    rows: list[dict[str, float | int | str]] = []
    for method, display in METHODS.items():
        for seed in SEEDS:
            run = histories[
                (histories["method"] == method) & (histories["seed"] == seed)
            ].set_index("query")
            for query in checkpoints:
                rows.append(
                    {
                        "method": display,
                        "seed": seed,
                        "query": int(query),
                        "cumulative_targets": int(run.loc[query, "cumulative_targets"]),
                    }
                )
    return pd.DataFrame(rows)


def group_inventory() -> tuple[pd.DataFrame, pd.DataFrame]:
    group_map = pd.read_csv(GROUP_MAP, dtype=str)
    if len(group_map) != 640 or set(group_map.columns) != {"candidate_id", "group_key"}:
        raise RuntimeError("Unexpected Mn-oxide group-map schema or row count")
    sizes = group_map.groupby("group_key", as_index=False).size()
    inventory = pd.DataFrame(
        [
            {
                "candidate_count": len(group_map),
                "group_count": len(sizes),
                "singleton_group_count": int((sizes["size"] == 1).sum()),
                "singleton_candidate_fraction": float((sizes["size"] == 1).sum())
                / len(group_map),
                "maximum_group_size": int(sizes["size"].max()),
                "source_sha256": sha256(GROUP_MAP),
            }
        ]
    )
    return group_map, inventory


def write_table(metrics: pd.DataFrame, audit: pd.DataFrame) -> None:
    gate = metrics[metrics["method"] == "energy_gated_da_tpp"].copy()
    greedy = metrics[metrics["method"] == "predicted_distance_greedy"].copy()

    def mean_sd(frame: pd.DataFrame, column: str, digits: int = 3) -> str:
        values = frame[column].astype(float)
        return f"${values.mean():.{digits}f} \\pm {values.std(ddof=1):.{digits}f}$"

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Metric & Energy-Gated DA-TPP & Predicted-Target Greedy \\",
        r"\midrule",
        f"Normalized AUTC & {mean_sd(gate, 'AUTC', 6)} & {mean_sd(greedy, 'AUTC', 6)} \\\\",
        f"Targets at 80 queries & {mean_sd(gate, 'recovery_at_80', 1)} & {mean_sd(greedy, 'recovery_at_80', 1)} \\\\",
        f"Targets at 160 queries & {mean_sd(gate, 'recovery_at_160', 1)} & {mean_sd(greedy, 'recovery_at_160', 1)} \\\\",
        f"Targets at 240 queries & {mean_sd(gate, 'recovery_at_240', 1)} & {mean_sd(greedy, 'recovery_at_240', 1)} \\\\",
        f"Targets at 320 queries & {mean_sd(gate, 'recovery_at_320', 1)} & {mean_sd(greedy, 'recovery_at_320', 1)} \\\\",
        f"Direct batches & {int(audit['gate_direct_rounds'].sum())}/200 & 200/200 \\\\",
        f"Effective replacements & {int(audit['effective_replacements'].sum())} & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    (ROOT / "Tables" / "generated" / "table_v60_mnoxide_control.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    gate_by_seed = gate.set_index("seed").sort_index()
    si_lines = [
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r"Seed & AUTC (both) & $R_{80}$ & $R_{160}$ & $R_{240}$ & $R_{320}$ & Direct & Correction \\",
        r"\midrule",
    ]
    for row in audit.sort_values("seed").itertuples(index=False):
        metric = gate_by_seed.loc[row.seed]
        si_lines.append(
            f"{row.seed} & {row.gate_AUTC:.6f} & "
            f"{int(metric.recovery_at_80)} & {int(metric.recovery_at_160)} & "
            f"{int(metric.recovery_at_240)} & {int(metric.recovery_at_320)} & "
            f"{row.gate_direct_rounds} & {row.gate_correction_rounds} \\\\"
        )
    si_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (
        ROOT / "Tables" / "generated" / "table_v60_mnoxide_control_per_seed.tex"
    ).write_text("\n".join(si_lines), encoding="utf-8")


def make_figure(trajectory: pd.DataFrame, metrics: pd.DataFrame) -> None:
    apply_original_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.85), constrained_layout=True)

    styles = {
        "Energy-Gated DA-TPP": {
            "color": GATE_COLOR,
            "marker": "^",
            "linestyle": "-",
            "linewidth": 1.8,
            "zorder": 3,
        },
        "Predicted-Target Greedy": {
            "color": GREEDY_COLOR,
            "marker": "s",
            "linestyle": "--",
            "linewidth": 1.65,
            "zorder": 4,
        },
    }
    for method, style in styles.items():
        subset = trajectory[trajectory["method"] == method]
        stats = subset.groupby("query")["cumulative_targets"].agg(
            ["mean", "min", "max"]
        )
        axes[0].fill_between(
            stats.index.to_numpy(),
            stats["min"].to_numpy(),
            stats["max"].to_numpy(),
            color=style["color"],
            alpha=0.10,
            linewidth=0,
        )
        axes[0].plot(
            stats.index,
            stats["mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markevery=4,
            markersize=4.0,
            label=method,
            zorder=style["zorder"],
        )
    axes[0].set_title("a  Target recovery", loc="left", fontweight="bold")
    axes[0].set_xlabel("Queries")
    axes[0].set_ylabel("Recovered targets")
    axes[0].set_xlim(0, 320)
    axes[0].set_ylim(0, 112)
    format_axes(axes[0])
    legend = axes[0].legend(frameon=True, loc="upper left")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#B0B0B0")
    legend.get_frame().set_linewidth(0.6)
    axes[0].text(
        0.98,
        0.08,
        "mean curves overlap",
        ha="right",
        va="bottom",
        transform=axes[0].transAxes,
        color="#444444",
        fontsize=8,
    )

    gate = metrics[metrics["method"] == "energy_gated_da_tpp"].sort_values("seed")
    greedy = metrics[
        metrics["method"] == "predicted_distance_greedy"
    ].sort_values("seed")
    x = greedy["AUTC"].astype(float).to_numpy()
    y = gate["AUTC"].astype(float).to_numpy()
    lower = min(x.min(), y.min()) - 0.0004
    upper = max(x.max(), y.max()) + 0.0004
    axes[1].plot([lower, upper], [lower, upper], color="#666666", linewidth=1.0)
    axes[1].scatter(
        x,
        y,
        color=GATE_COLOR,
        marker="^",
        edgecolor="white",
        linewidth=0.5,
        s=34,
    )
    axes[1].set_title("b  Paired normalized AUTC", loc="left", fontweight="bold")
    axes[1].set_xlabel("Greedy")
    axes[1].set_ylabel("Gate")
    axes[1].set_xlim(lower, upper)
    axes[1].set_ylim(lower, upper)
    format_axes(axes[1])
    axes[1].text(
        0.04,
        0.94,
        "all seeds on identity line",
        ha="left",
        va="top",
        transform=axes[1].transAxes,
        color="#444444",
        fontsize=8,
    )

    figure_dir = ROOT / "Figures"
    for extension in ("pdf", "png", "svg", "tiff"):
        kwargs = {"dpi": 600} if extension in {"png", "tiff"} else {}
        fig.savefig(
            figure_dir / f"Figure6_v60_mnoxide_control.{extension}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(fig)


def write_manifest(outputs: list[Path]) -> None:
    rows = [
        {"file": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in outputs
    ]
    pd.DataFrame(rows).to_csv(
        ROOT / "SourceData" / "v60_mnoxide_output_sha256.csv", index=False
    )


def main() -> None:
    metrics, histories = load_metrics()
    audit = audit_pairing(metrics, histories)
    trajectory = batch_trajectory(histories)
    group_map, inventory = group_inventory()

    source_dir = ROOT / "SourceData"
    metrics_path = source_dir / "mnoxide_corrected_seed_metrics.csv"
    audit_path = source_dir / "mnoxide_seed_pairing_audit.csv"
    trajectory_path = source_dir / "mnoxide_recovery_trajectories.csv"
    group_map_path = source_dir / "mnoxide_element_system_groups.csv"
    inventory_path = source_dir / "mnoxide_group_inventory.csv"
    metrics.to_csv(metrics_path, index=False)
    audit.to_csv(audit_path, index=False)
    trajectory.to_csv(trajectory_path, index=False)
    group_map.to_csv(group_map_path, index=False)
    inventory.to_csv(inventory_path, index=False)

    write_table(metrics, audit)
    make_figure(trajectory, metrics)
    outputs = [
        metrics_path,
        audit_path,
        trajectory_path,
        group_map_path,
        inventory_path,
        ROOT / "Tables" / "generated" / "table_v60_mnoxide_control.tex",
        ROOT
        / "Tables"
        / "generated"
        / "table_v60_mnoxide_control_per_seed.tex",
        ROOT / "Figures" / "Figure6_v60_mnoxide_control.pdf",
        ROOT / "Figures" / "Figure6_v60_mnoxide_control.png",
        ROOT / "Figures" / "Figure6_v60_mnoxide_control.svg",
        ROOT / "Figures" / "Figure6_v60_mnoxide_control.tiff",
    ]
    write_manifest(outputs)


if __name__ == "__main__":
    main()
