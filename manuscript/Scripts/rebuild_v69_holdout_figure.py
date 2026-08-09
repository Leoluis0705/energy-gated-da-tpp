"""Rebuild the held-out comparison without a magnified inset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebuild_v63_discussion_figures import GREEN, PURPLE, save, style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "SourceData" / "Gamma005HoldoutAnalysis"


def main() -> None:
    style()
    frame = pd.read_csv(SOURCE / "v60_gamma005_holdout_per_seed.csv")
    gate = frame[frame["method"] == "energy_gated_da_tpp"].set_index("seed").sort_index()
    greedy = frame[frame["method"] == "predicted_target_greedy"].set_index("seed").sort_index()
    delta = gate["AUTC_160"] - greedy["AUTC_160"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.70), constrained_layout=True)

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
    means = {}
    sds = {}
    handles = []
    labels = []
    for block, label, colour, marker in (
        (gate, "Gate", PURPLE, "^"),
        (greedy, "Greedy", GREEN, "s"),
    ):
        means[label] = np.array([block[column].mean() for column in columns])
        sds[label] = np.array([block[column].std(ddof=1) for column in columns])
        handle = axes[1].errorbar(
            checkpoints,
            means[label],
            yerr=sds[label],
            color=colour,
            marker=marker,
            markersize=4.2,
            capsize=2.5,
            label=label,
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
    save(fig, "Figure7_v69_gamma005_holdout_clean")


if __name__ == "__main__":
    main()
