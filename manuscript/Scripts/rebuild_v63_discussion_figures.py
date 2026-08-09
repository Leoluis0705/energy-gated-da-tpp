"""Rebuild the compact parameter and held-out figures from frozen source tables."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SourceData"
FIGURES = ROOT / "Figures"
PURPLE = "#9467BD"
GREEN = "#2CA02C"
BLUE = "#1F77B4"
ORANGE = "#FF7F0E"
RED = "#D62728"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.6,
            "ytick.labelsize": 9.6,
            "legend.fontsize": 9.4,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.55,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def format_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#B0B0B0", linewidth=0.5, alpha=0.33)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 600} if suffix == "png" else {}
        fig.savefig(FIGURES / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def figure7() -> None:
    summary = pd.read_csv(SOURCE / "v60_parameter_sensitivity_summary.csv")
    threshold = summary[summary["stage"] == "threshold"].copy()
    weights = summary[summary["stage"] == "weight"].copy()
    m_values = sorted(threshold["M0"].unique())
    g_values = sorted(threshold["G0"].unique())
    autc = threshold.pivot(index="M0", columns="G0", values="mean_AUTC_160").loc[m_values, g_values]
    rounds = threshold.pivot(index="M0", columns="G0", values="mean_correction_rounds").loc[m_values, g_values]

    purple_light = LinearSegmentedColormap.from_list("purple_light", ["#FBF9FD", "#EEE6F5", "#D4BFE5"])
    green_light = LinearSegmentedColormap.from_list("green_light", ["#FAFCFA", "#E2F2E2", "#B8DEB8"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), constrained_layout=True)
    for ax, matrix, title, number_format, cmap in (
        (axes[0], autc, r"a  Early AUTC ($q\leq160$)", ".4f", purple_light),
        (axes[1], rounds, "b  Correction-route batches", ".1f", green_light),
    ):
        x_edges = np.arange(matrix.shape[1] + 1) - 0.5
        y_edges = np.arange(matrix.shape[0] + 1) - 0.5
        mesh = ax.pcolormesh(x_edges, y_edges, matrix.values, cmap=cmap, shading="flat")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, format(matrix.iloc[row, col], number_format), ha="center", va="center", color="#111111", fontsize=9.5)
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        ax.set_xticks(range(len(g_values)), [f"{value:.2f}" for value in g_values])
        ax.set_yticks(range(len(m_values)), [f"{value:.2f}" for value in m_values])
        ax.set_xlabel(r"Group threshold $G_0$")
        ax.set_ylabel(r"Margin threshold $M_0$")
        ax.set_title(title, loc="left", fontweight="bold")
        center_row = m_values.index(1.0)
        center_col = g_values.index(0.5)
        ax.add_patch(plt.Rectangle((center_col - 0.5, center_row - 0.5), 1, 1, fill=False, edgecolor=RED, linewidth=1.5))
        ax.tick_params(direction="out", width=0.8, length=3)
        colorbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
        colorbar.solids.set_rasterized(False)

    def weight_label(row: pd.Series) -> str:
        baseline = {"alpha": 0.1, "beta": 0.2, "gamma": 0.1}
        changed = [key for key in baseline if not np.isclose(float(row[key]), baseline[key])]
        if not changed:
            return "Archived"
        key = changed[0]
        symbol = {"alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma"}[key]
        return rf"${symbol}={float(row[key]):g}$"

    weights["label"] = weights.apply(weight_label, axis=1)
    order = ["Archived", r"$\alpha=0.05$", r"$\alpha=0.2$", r"$\beta=0.1$", r"$\beta=0.4$", r"$\gamma=0.05$", r"$\gamma=0.2$"]
    weights["label"] = pd.Categorical(weights["label"], order, ordered=True)
    weights = weights.sort_values("label")
    colours = [PURPLE, BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
    markers = ["^", "o", "o", "D", "D", "s", "s"]
    for index, (_, row) in enumerate(weights.iterrows()):
        axes[2].errorbar(index, row["mean_AUTC_160"], yerr=row["sd_AUTC_160"], fmt=markers[index], capsize=2.5, color=colours[index])
    archived = float(weights.loc[weights["label"] == "Archived", "mean_AUTC_160"].iloc[0])
    axes[2].axhline(archived, color=PURPLE, linestyle="--", linewidth=1.1)
    axes[2].set_xticks(range(len(weights)), weights["label"].astype(str), rotation=42, ha="right")
    axes[2].set_ylabel("AUTC through 160 queries")
    axes[2].set_title("c  Weight sensitivity", loc="left", fontweight="bold")
    format_axes(axes[2], grid_axis="y")
    save(fig, "Figure7_v63_parameter_sensitivity_light")


def figure8() -> None:
    frame = pd.read_csv(SOURCE / "Gamma005HoldoutAnalysis" / "v60_gamma005_holdout_per_seed.csv")
    gate = frame[frame["method"] == "energy_gated_da_tpp"].set_index("seed").sort_index()
    greedy = frame[frame["method"] == "predicted_target_greedy"].set_index("seed").sort_index()
    delta = gate["AUTC_160"] - greedy["AUTC_160"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.55), constrained_layout=True)
    positive = delta >= 0
    colours = np.where(positive, PURPLE, GREEN)
    axes[0].axhline(0, color="#4A4A4A", linewidth=0.9)
    axes[0].vlines(delta.index, 0, delta.values, color=colours, linewidth=1.5, alpha=0.85)
    axes[0].scatter(delta.index[positive], delta[positive], color=PURPLE, marker="^", s=31, zorder=3)
    axes[0].scatter(delta.index[~positive], delta[~positive], color=GREEN, marker="s", s=27, zorder=3)
    axes[0].axhline(delta.mean(), color=PURPLE, linestyle="--", linewidth=1.0)
    axes[0].text(-0.12, 1.04, "a", transform=axes[0].transAxes, fontweight="bold", fontsize=10)
    axes[0].set_xlabel("Held-out seed")
    axes[0].set_ylabel(r"AUTC$_{160}$ difference (Gate $-$ Greedy)")
    axes[0].set_xticks(delta.index)
    axes[0].grid(True, axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.28)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    checkpoints = np.array([80, 160, 240, 320])
    columns = ["recovery_at_80", "recovery_at_160", "recovery_at_240", "recovery_at_320"]
    handles = []
    labels = []
    for block, label, colour, marker in ((gate, "Gate", PURPLE, "^"), (greedy, "Greedy", GREEN, "s")):
        means = np.array([block[column].mean() for column in columns])
        sds = np.array([block[column].std(ddof=1) for column in columns])
        handle = axes[1].errorbar(
            checkpoints, means, yerr=sds, color=colour, marker=marker,
            markersize=4.2, capsize=2.5, label=label,
        )
        handles.append(handle)
        labels.append(label)
    axes[1].text(-0.12, 1.04, "b", transform=axes[1].transAxes, fontweight="bold", fontsize=10)
    axes[1].set_xlabel("Query budget")
    axes[1].set_ylabel("Recovered targets")
    axes[1].set_xticks(checkpoints)
    axes[1].set_ylim(20, 82)
    axes[1].grid(True, axis="y", color="#B0B0B0", linewidth=0.45, alpha=0.28)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.76, 1.04), ncol=2, frameon=False)
    save(fig, "Figure8_v63_gamma005_holdout_difference")


def main() -> None:
    style()
    figure7()


if __name__ == "__main__":
    main()
