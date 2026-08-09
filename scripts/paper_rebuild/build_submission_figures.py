#!/usr/bin/env python3
"""Build submission figures using the supervisor's original plotting style.

The plotting choices below follow:
    D:\CGCNN\cgcnn\active_learning_analysis.py

Only the data adapter, method labels, axis wording, and additional export
formats are specific to the present manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"
SOURCE_DATA = ROOT / "SourceData"

SIX_POLICY_SOURCE = SOURCE_DATA / "Figure2_six_policy_target_recovery.csv"
HIDDEN_AUDIT_SOURCE = SOURCE_DATA / "Figure3_hidden_evaluability_audit.csv"

GATE = "Energy-Gated DA-TPP"
GREEDY = "Predicted-Target Greedy"
BATCH_SIZE = 16
QUERY_BUDGET = 320
TARGET_TOTAL = 78

# Colors, markers, line styles, widths, and ordering follow the supervisor
# script. The descriptive names are those used in the manuscript.
METHODS = [
    (GATE, "Energy-Gated DA-TPP", "#9467bd", "^", "-"),
    ("Explore", "Explore", "#1f77b4", "o", "-"),
    (GREEDY, "Predicted-Target Greedy", "#2ca02c", "s", "-"),
    (
        "Modulus / Gradient-Norm Hybrid",
        "Modulus/Gradient-Norm",
        "#ff7f0e",
        "D",
        "-",
    ),
    ("MC Dropout", "MC Dropout", "#e377c2", "p", "-"),
    ("Random Sampling", "Random Sampling", "#8c564b", "x", "--"),
]


def set_supervisor_style() -> None:
    """Apply the rcParams from the original supervisor plotting script."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 10,
            "figure.titlesize": 12,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def batch_points(data: pd.DataFrame, method: str, column: str) -> pd.DataFrame:
    """Return the initial point and the recorded end of each 16-query batch."""
    subset = data.loc[data["method"].eq(method)].sort_values("query").copy()
    return subset.loc[
        subset["query"].eq(0) | subset["query"].mod(BATCH_SIZE).eq(0),
        ["query", column],
    ]


def save_all(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    target = FIGURES / stem
    fig.savefig(target.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(target.with_suffix(".pdf"), facecolor="white")
    fig.savefig(target.with_suffix(".svg"), facecolor="white")
    fig.savefig(
        target.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def validate_six_policy(data: pd.DataFrame) -> None:
    required = {"method", "query", "cumulative_targets"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing six-policy columns: {sorted(missing)}")

    expected_methods = {method for method, *_ in METHODS}
    observed_methods = set(data["method"].astype(str).unique())
    if observed_methods != expected_methods:
        raise ValueError(
            f"method mismatch: missing={sorted(expected_methods-observed_methods)}, "
            f"unexpected={sorted(observed_methods-expected_methods)}"
        )

    checkpoints = {
        GATE: [33, 50, 69, 78],
        GREEDY: [27, 47, 69, 78],
        "Explore": [29, 44, 60, 74],
        "Modulus / Gradient-Norm Hybrid": [10, 20, 39, 58],
        "MC Dropout": [8, 27, 36, 49],
        "Random Sampling": [6, 18, 26, 39],
    }
    for method, expected in checkpoints.items():
        observed = (
            data.loc[
                data["method"].eq(method)
                & data["query"].isin([80, 160, 240, 320]),
                "cumulative_targets",
            ]
            .astype(int)
            .tolist()
        )
        if observed != expected:
            raise ValueError(
                f"checkpoint mismatch for {method}: "
                f"observed={observed}, expected={expected}"
            )


def draw_six_policy() -> None:
    data = pd.read_csv(SIX_POLICY_SOURCE)
    validate_six_policy(data)

    fig, ax = plt.subplots(figsize=(12, 7))
    for method, label, color, marker, linestyle in METHODS:
        curve = batch_points(data, method, "cumulative_targets")
        ax.plot(
            curve["query"],
            curve["cumulative_targets"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=6 if marker != "x" else 4,
            linewidth=2,
            label=label,
        )

    # The supervisor script includes both a perfect-efficiency reference and
    # the total number of valid samples.
    perfect_x = np.arange(0, TARGET_TOTAL + 1)
    ax.plot(
        perfect_x,
        perfect_x,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Perfect efficiency",
    )
    ax.axhline(
        TARGET_TOTAL,
        color="#17becf",
        linestyle="--",
        linewidth=2,
        label="Total reference targets",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.set_xlabel("Label Queries Consumed")
    ax.set_ylabel("Reference Targets Identified")
    ax.set_title("Property: Formation energy  Batch_size: 16")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, QUERY_BUDGET)
    ax.set_ylim(0, TARGET_TOTAL * 1.10)
    ax.set_xticks([0, 80, 160, 240, 320])
    ax.set_yticks([0, 20, 40, 60, 80])

    fig.tight_layout()
    save_all(fig, "Figure2_six_policy_target_recovery")


def validate_hidden_audit(data: pd.DataFrame) -> pd.DataFrame:
    required = {
        "method",
        "archived_run_id",
        "query",
        "cumulative_expected_hidden_evaluable_targets",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"missing hidden-audit columns: {sorted(missing)}")

    data = data.loc[
        data["archived_run_id"].eq(1) & data["method"].isin([GATE, GREEDY])
    ].copy()
    expected = {
        GATE: [16.63, 33.87, 39.93, 46.51],
        GREEDY: [10.99, 31.60, 40.96, 46.51],
    }
    for method, expected_values in expected.items():
        observed = (
            data.loc[
                data["method"].eq(method)
                & data["query"].isin([80, 160, 240, 320]),
                "cumulative_expected_hidden_evaluable_targets",
            ]
            .astype(float)
            .tolist()
        )
        if any(
            abs(observed_value - expected_value) > 0.011
            for observed_value, expected_value in zip(observed, expected_values)
        ):
            raise ValueError(
                f"hidden-audit mismatch for {method}: "
                f"observed={observed}, expected={expected_values}"
            )
    return data


def draw_hidden_audit() -> None:
    data = validate_hidden_audit(pd.read_csv(HIDDEN_AUDIT_SOURCE))

    fig, ax = plt.subplots(figsize=(12, 7))
    styles = {
        GATE: ("Energy-Gated DA-TPP", "#9467bd", "^"),
        GREEDY: ("Predicted-Target Greedy", "#2ca02c", "s"),
    }
    for method in [GATE, GREEDY]:
        label, color, marker = styles[method]
        curve = batch_points(
            data,
            method,
            "cumulative_expected_hidden_evaluable_targets",
        )
        ax.plot(
            curve["query"],
            curve["cumulative_expected_hidden_evaluable_targets"],
            color=color,
            marker=marker,
            linestyle="-",
            markersize=6,
            linewidth=2,
            label=label,
        )

    full_score = 46.51
    ax.axhline(
        full_score,
        color="#17becf",
        linestyle="--",
        linewidth=2,
        label="Full target-set score",
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1)

    ax.set_xlabel("Label Queries Consumed")
    ax.set_ylabel("Workflow-Completion Score")
    ax.set_title("Post-selection workflow-completion score  Batch_size: 16")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, QUERY_BUDGET)
    ax.set_ylim(0, full_score * 1.10)
    ax.set_xticks([0, 80, 160, 240, 320])
    ax.set_yticks([0, 10, 20, 30, 40, 50])

    fig.tight_layout()
    save_all(fig, "Figure3_hidden_evaluability_audit")


def main() -> None:
    set_supervisor_style()
    draw_six_policy()
    draw_hidden_audit()


if __name__ == "__main__":
    main()
