from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"
SOURCE_DATA = ROOT / "SourceData"

SIX_POLICY_SOURCE = SOURCE_DATA / "Figure2_six_policy_target_recovery.csv"
PAIRED_SOURCE = SOURCE_DATA / "paired_cgcnn_histories.csv"
EVALUABILITY_SOURCE = SOURCE_DATA / "Figure1_dft_evaluability_overlay.csv"

METHOD_ORDER = [
    "Energy-Gated DA-TPP",
    "Predicted-Target Greedy",
    "Explore",
    "Modulus / Gradient-Norm Hybrid",
    "MC Dropout",
    "Random Sampling",
]

DISPLAY_NAME = {
    "Energy-Gated DA-TPP": "Energy-Gated DA-TPP",
    "Predicted-Target Greedy": "Predicted-Target Greedy",
    "Explore": "Explore",
    "Modulus / Gradient-Norm Hybrid": "Modulus/Gradient-Norm",
    "MC Dropout": "MC Dropout",
    "Random Sampling": "Random Sampling",
}

# Visual grammar follows the supplied mentor script.
METHOD_STYLE = {
    "Energy-Gated DA-TPP": dict(color="#9467BD", marker="^", linestyle="-"),
    "Predicted-Target Greedy": dict(color="#2CA02C", marker="s", linestyle="-"),
    "Explore": dict(color="#1F77B4", marker="o", linestyle="-"),
    "Modulus / Gradient-Norm Hybrid": dict(
        color="#FF7F0E", marker="D", linestyle="-"
    ),
    "MC Dropout": dict(color="#E377C2", marker="p", linestyle="-"),
    "Random Sampling": dict(color="#8C564B", marker="x", linestyle="--"),
}

TARGET_COUNT = 78
HORIZONS = (80, 160, 240, 320)
EXPECTED_AUTC = {
    80: (0.14102564, 0.12564103),
    160: (0.32051282, 0.27948718),
    240: (0.46923077, 0.45042735),
    320: (0.58269231, 0.57179487),
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.titlesize": 11.6,
            "axes.labelsize": 11.0,
            "xtick.labelsize": 10.4,
            "ytick.labelsize": 10.4,
            "legend.fontsize": 10.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "energy-gated-da-tpp-v50",
        }
    )


def format_axes(ax: mpl.axes.Axes) -> None:
    ax.grid(True, color="#B0B0B0", linewidth=0.55, alpha=0.38)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, length=3.2)


def save_all(fig: mpl.figure.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / stem
    pdf_metadata = {
        "Creator": "Energy-Gated DA-TPP v50 figure script",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata=pdf_metadata,
    )
    fig.savefig(
        path.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Creator": "Energy-Gated DA-TPP v50 figure script",
            "Date": None,
        },
    )
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Software": "Energy-Gated DA-TPP v50 figure script"},
    )
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def batch_curve(data: pd.DataFrame, value: str) -> pd.DataFrame:
    checkpoints = [0] + list(range(16, 321, 16))
    out = data.loc[data["query"].isin(checkpoints), ["query", value]].copy()
    if out["query"].tolist() != checkpoints:
        raise ValueError("trajectory does not contain every 16-query checkpoint")
    return out


def left_continuous_autc(
    queries: np.ndarray,
    recovered: np.ndarray,
    horizon: int,
    target_count: int = TARGET_COUNT,
    batch_size: int = 16,
) -> float:
    if horizon % batch_size:
        raise ValueError("horizon must be a multiple of the batch size")
    lookup = {int(q): float(y) for q, y in zip(queries, recovered, strict=True)}
    left_edges = range(0, horizon, batch_size)
    missing = [q for q in left_edges if q not in lookup]
    if missing:
        raise ValueError(f"missing AUTC checkpoints: {missing}")
    area = batch_size * sum(lookup[q] for q in left_edges)
    return area / (horizon * target_count)


def canonical_paired_curves() -> dict[str, pd.DataFrame]:
    data = pd.read_csv(PAIRED_SOURCE)
    required = {"method", "seed", "query", "target_label"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"paired source missing columns: {sorted(missing)}")

    curves: dict[str, pd.DataFrame] = {}
    for method in ("Energy-Gated DA-TPP", "Predicted-Target Greedy"):
        subset = data.loc[
            data["method"].eq(method) & data["seed"].eq(5),
            ["query", "target_label"],
        ].sort_values("query")
        if subset["query"].iloc[0] != 1 or subset["query"].iloc[-1] < 320:
            raise ValueError(f"incomplete canonical trajectory for {method}")
        subset = subset.loc[subset["query"].le(320)].copy()
        subset["cumulative_targets"] = subset["target_label"].cumsum()
        subset = pd.concat(
            [
                pd.DataFrame(
                    {"query": [0], "target_label": [0], "cumulative_targets": [0]}
                ),
                subset,
            ],
            ignore_index=True,
        )
        curves[method] = subset
    return curves


def compute_autc_table(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    gate = curves["Energy-Gated DA-TPP"]
    greedy = curves["Predicted-Target Greedy"]
    for horizon in HORIZONS:
        gate_value = left_continuous_autc(
            gate["query"].to_numpy(),
            gate["cumulative_targets"].to_numpy(),
            horizon,
        )
        greedy_value = left_continuous_autc(
            greedy["query"].to_numpy(),
            greedy["cumulative_targets"].to_numpy(),
            horizon,
        )
        expected_gate, expected_greedy = EXPECTED_AUTC[horizon]
        if not np.isclose(gate_value, expected_gate, atol=5e-8):
            raise AssertionError(
                f"Gate AUTC mismatch at {horizon}: {gate_value:.8f}"
            )
        if not np.isclose(greedy_value, expected_greedy, atol=5e-8):
            raise AssertionError(
                f"Greedy AUTC mismatch at {horizon}: {greedy_value:.8f}"
            )
        rows.append(
            {
                "horizon": horizon,
                "gate_autc": gate_value,
                "greedy_autc": greedy_value,
                "relative_gate_advantage_percent": 100
                * (gate_value / greedy_value - 1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(SOURCE_DATA / "v50_autc_checkpoints.csv", index=False, float_format="%.8f")
    return out


def draw_six_policy() -> None:
    data = pd.read_csv(SIX_POLICY_SOURCE)
    expected = set(METHOD_ORDER)
    observed = set(data["method"].unique())
    if observed != expected:
        raise ValueError(
            f"six-policy method mismatch: missing={expected-observed}, extra={observed-expected}"
        )

    fig, ax = plt.subplots(figsize=(7.18, 4.05))
    plotted: list[pd.DataFrame] = []
    for method in METHOD_ORDER:
        curve = batch_curve(
            data.loc[data["method"].eq(method)].sort_values("query"),
            "cumulative_targets",
        )
        style = METHOD_STYLE[method]
        ax.plot(
            curve["query"],
            curve["cumulative_targets"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markersize=4.2 if style["marker"] != "x" else 3.8,
            markeredgewidth=0.75,
            label=DISPLAY_NAME[method],
        )
        export = curve.copy()
        export.insert(0, "method", method)
        export.insert(2, "panel", "six_policy_archive")
        plotted.append(export)

    ax.plot(
        [0, TARGET_COUNT],
        [0, TARGET_COUNT],
        color="#D62728",
        linestyle="--",
        linewidth=1.35,
        label="Perfect efficiency",
    )
    ax.axhline(
        TARGET_COUNT,
        color="#17BECF",
        linestyle="--",
        linewidth=1.35,
        label="Target ceiling (78)",
    )
    ax.set_title("Formation-energy target recovery  |  batch size: 16")
    ax.set_xlabel("Cumulative label queries")
    ax.set_ylabel("Recovered surrogate targets")
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 84)
    ax.set_xticks([0, 80, 160, 240, 320])
    ax.set_yticks([0, 20, 40, 60, 80])
    format_axes(ax)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.988),
        ncol=2,
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#BDBDBD",
        columnspacing=0.9,
        handlelength=2.4,
        borderpad=0.55,
        labelspacing=0.38,
    )
    fig.subplots_adjust(left=0.105, right=0.988, top=0.91, bottom=0.14)
    save_all(fig, "Figure2_v50_six_policy_mentor")
    pd.concat(plotted, ignore_index=True).to_csv(
        SOURCE_DATA / "v50_recovery_figure_values.csv", index=False
    )


def draw_gate_greedy_evidence(
    curves: dict[str, pd.DataFrame], autc: pd.DataFrame
) -> None:
    evaluability = pd.read_csv(EVALUABILITY_SOURCE)
    evaluability = evaluability.loc[evaluability["archived_run_id"].eq(1)].copy()
    methods = ("Energy-Gated DA-TPP", "Predicted-Target Greedy")

    fig = plt.figure(figsize=(7.35, 3.45))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.18, 1.0], wspace=0.34)
    ax_recovery = fig.add_subplot(grid[0, 0])
    ax_eval = fig.add_subplot(grid[0, 1])
    ax_autc = fig.add_subplot(grid[0, 2])

    for method in methods:
        style = METHOD_STYLE[method]
        recovery_curve = batch_curve(curves[method], "cumulative_targets")
        ax_recovery.plot(
            recovery_curve["query"],
            recovery_curve["cumulative_targets"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.0,
            markeredgewidth=0.75,
            label=DISPLAY_NAME[method],
        )

        eval_curve = batch_curve(
            evaluability.loc[evaluability["method"].eq(method)].sort_values("query"),
            "cumulative_expected_DFT_evaluable_targets",
        )
        ax_eval.plot(
            eval_curve["query"],
            eval_curve["cumulative_expected_DFT_evaluable_targets"],
            color=style["color"],
            marker=style["marker"],
            markersize=4.0,
            markeredgewidth=0.75,
        )

    ax_recovery.axhline(
        TARGET_COUNT, color="#17BECF", linestyle="--", linewidth=1.1
    )
    ax_recovery.set_title("a  Surrogate-target recovery", loc="left", fontweight="bold")
    ax_recovery.set_xlabel("Cumulative label queries")
    ax_recovery.set_ylabel("Recovered targets")
    ax_recovery.set_xlim(0, 320)
    ax_recovery.set_ylim(0, 84)
    ax_recovery.set_xticks([0, 80, 160, 240, 320])
    ax_recovery.set_yticks([0, 20, 40, 60, 80])
    format_axes(ax_recovery)
    ax_recovery.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#BDBDBD",
        borderpad=0.45,
        handlelength=2.0,
    )
    ax_eval.set_title("b  DFT-evaluability audit", loc="left", fontweight="bold")
    ax_eval.set_xlabel("Cumulative label queries")
    ax_eval.set_ylabel("Expected evaluable targets")
    ax_eval.set_xlim(0, 320)
    ax_eval.set_ylim(0, 50)
    ax_eval.set_xticks([0, 80, 160, 240, 320])
    ax_eval.set_yticks([0, 10, 20, 30, 40, 50])
    format_axes(ax_eval)
    ax_eval.annotate(
        "16.63 vs 10.99",
        xy=(80, 16.626185),
        xytext=(100, 7.5),
        fontsize=10.2,
        arrowprops=dict(arrowstyle="-", color="#666666", linewidth=0.7),
    )

    x = np.arange(len(autc))
    width = 0.34
    gate_color = METHOD_STYLE["Energy-Gated DA-TPP"]["color"]
    greedy_color = METHOD_STYLE["Predicted-Target Greedy"]["color"]
    ax_autc.bar(
        x - width / 2,
        autc["gate_autc"],
        width,
        color=gate_color,
        edgecolor="#555555",
        linewidth=0.45,
        label="Gate",
    )
    ax_autc.bar(
        x + width / 2,
        autc["greedy_autc"],
        width,
        color=greedy_color,
        edgecolor="#555555",
        linewidth=0.45,
        label="Greedy",
    )
    for index, row in autc.iterrows():
        y = max(row["gate_autc"], row["greedy_autc"]) + 0.027
        ax_autc.text(
            index,
            y,
            f"+{row['relative_gate_advantage_percent']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#4F3B69",
        )
    ax_autc.set_title("c  AUTC by budget", loc="left", fontweight="bold")
    ax_autc.set_xlabel("Stopping budget")
    ax_autc.set_ylabel("Normalized AUTC")
    ax_autc.set_xticks(x, [str(v) for v in autc["horizon"]])
    ax_autc.set_ylim(0, 0.68)
    ax_autc.set_yticks([0, 0.2, 0.4, 0.6])
    format_axes(ax_autc)
    ax_autc.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.93,
        facecolor="white",
        edgecolor="#BDBDBD",
        borderpad=0.4,
    )

    fig.subplots_adjust(left=0.065, right=0.994, top=0.90, bottom=0.17)
    save_all(fig, "Figure3_v50_gate_greedy_evidence")


def main() -> None:
    set_style()
    curves = canonical_paired_curves()
    autc = compute_autc_table(curves)
    draw_six_policy()
    draw_gate_greedy_evidence(curves, autc)
    print(autc.to_string(index=False, float_format=lambda value: f"{value:.8f}"))


if __name__ == "__main__":
    main()
