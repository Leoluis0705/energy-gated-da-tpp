"""Rebuild the held-out comparison with an honest early-budget inset."""

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

    # The late-budget separation is small, so report its magnitude without
    # truncating the main axis or implying a larger effect.
    late_delta = means["Gate"][2] - means["Greedy"][2]
    axes[1].annotate(
        rf"$\Delta={late_delta:+.1f}$",
        xy=(240, means["Gate"][2]),
        xytext=(251, 74.0),
        fontsize=7.0,
        color="#333333",
        arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.7},
    )

    # Modest inset for the prespecified early stopping budgets. It repeats the
    # same means and sample-SD bars and labels the observed mean differences.
    inset = axes[1].inset_axes([0.52, 0.08, 0.43, 0.40])
    for label, colour, marker in (("Gate", PURPLE, "^"), ("Greedy", GREEN, "s")):
        inset.errorbar(
            checkpoints[:2],
            means[label][:2],
            yerr=sds[label][:2],
            color=colour,
            marker=marker,
            markersize=3.2,
            capsize=1.8,
            linewidth=1.05,
        )
    early_deltas = means["Gate"][:2] - means["Greedy"][:2]
    inset.text(92, 34.3, rf"$\Delta={early_deltas[0]:+.1f}$", ha="center", fontsize=6.2)
    inset.text(148, 56.5, rf"$\Delta={early_deltas[1]:+.1f}$", ha="center", fontsize=6.2)
    inset.set_xlim(68, 172)
    inset.set_ylim(20, 59)
    inset.set_xticks([80, 160])
    inset.set_yticks([25, 40, 55])
    inset.tick_params(labelsize=5.8, direction="out", width=0.6, length=2)
    inset.grid(True, axis="y", color="#B0B0B0", linewidth=0.35, alpha=0.25)
    inset.set_title("Early-budget detail", loc="left", fontsize=6.6, pad=2)
    for spine in inset.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.65)

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.76, 1.04), ncol=2, frameon=False)
    save(fig, "Figure7_v68_gamma005_holdout_early_inset")


if __name__ == "__main__":
    main()
