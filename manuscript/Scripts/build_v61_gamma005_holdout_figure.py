"""Build the manuscript figure for the frozen gamma=0.05 held-out cohort."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SourceData" / "Gamma005HoldoutAnalysis"
FIGURES = ROOT / "Figures"
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


def format_axes(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, color="#B0B0B0", linewidth=0.55, alpha=0.38)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3.2)


def main() -> None:
    frame = pd.read_csv(SOURCE / "v60_gamma005_holdout_per_seed.csv")
    expected_methods = {"energy_gated_da_tpp", "predicted_target_greedy"}
    if set(frame["method"]) != expected_methods:
        raise ValueError("unexpected method set")
    if set(frame["seed"]) != set(range(15, 25)):
        raise ValueError("expected held-out seeds 15--24")

    gate = frame[frame["method"] == "energy_gated_da_tpp"].set_index("seed").sort_index()
    greedy = frame[frame["method"] == "predicted_target_greedy"].set_index("seed").sort_index()
    if not gate.index.equals(greedy.index):
        raise ValueError("unpaired seeds")
    if gate["sequence_sha256"].nunique() != 10 or greedy["sequence_sha256"].nunique() != 10:
        raise ValueError("held-out query sequences are not independent")

    apply_original_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.0), constrained_layout=True)

    # Panel a: paired early-AUTC observations, preserving seed-level heterogeneity.
    x = np.array([0.0, 1.0])
    for seed in gate.index:
        values = [greedy.loc[seed, "AUTC_160"], gate.loc[seed, "AUTC_160"]]
        axes[0].plot(x, values, color="#8C8C8C", alpha=0.62, linewidth=0.9, zorder=1)
        axes[0].scatter(
            x[0], values[0], color=GREEDY_COLOR, marker="s", s=22, zorder=2
        )
        axes[0].scatter(
            x[1], values[1], color=GATE_COLOR, marker="^", s=25, zorder=2
        )
    means = [greedy["AUTC_160"].mean(), gate["AUTC_160"].mean()]
    axes[0].scatter(
        x[0], means[0], marker="s", s=54, facecolor="white",
        edgecolor=GREEDY_COLOR, linewidth=1.5, zorder=3
    )
    axes[0].scatter(
        x[1], means[1], marker="^", s=62, facecolor="white",
        edgecolor=GATE_COLOR, linewidth=1.5, zorder=3
    )
    axes[0].set_xticks(x, ["Greedy", r"Gate ($\gamma=0.05$)"])
    axes[0].set_ylabel("Normalized AUTC through 160 queries")
    axes[0].set_title("a  Paired held-out seeds 15--24", loc="left", fontweight="bold")
    format_axes(axes[0], grid_axis="y")
    axes[0].text(
        0.03,
        0.97,
        "Mean difference = +0.0256\nGate higher in 6/10 seeds",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.7,
    )

    # Panel b: recovery at the prespecified stopping budgets.
    checkpoints = np.array([80, 160, 240, 320])
    columns = ["recovery_at_80", "recovery_at_160", "recovery_at_240", "recovery_at_320"]
    for block, label, colour, marker in (
        (gate, "Gate", GATE_COLOR, "^"),
        (greedy, "Greedy", GREEDY_COLOR, "s"),
    ):
        means = np.array([block[column].mean() for column in columns])
        sds = np.array([block[column].std(ddof=1) for column in columns])
        axes[1].errorbar(
            checkpoints,
            means,
            yerr=sds,
            color=colour,
            marker=marker,
            linewidth=1.6,
            markersize=4.5,
            capsize=3,
            label=label,
        )
    axes[1].set_xlabel("Query budget")
    axes[1].set_ylabel("Recovered targets")
    axes[1].set_title("b  Recovery at stopping budgets", loc="left", fontweight="bold")
    axes[1].set_xticks(checkpoints)
    axes[1].set_ylim(20, 82)
    format_axes(axes[1])
    legend = axes[1].legend(frameon=True, loc="lower right")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#B0B0B0")
    legend.get_frame().set_linewidth(0.6)

    FIGURES.mkdir(parents=True, exist_ok=True)
    output = FIGURES / "Figure8_v61_gamma005_holdout"
    for suffix in ("pdf", "png", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
