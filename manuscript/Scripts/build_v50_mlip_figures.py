from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "Figures"
SOURCE_DATA = ROOT / "SourceData"
CALIBRATION_SOURCE = SOURCE_DATA / "energy_calibration_best_oof.csv"
RUNTIME_SOURCE = SOURCE_DATA / "mlip_full_pool_results.csv"

ELEMENT_STYLE = {
    "Cr": dict(color="#9467BD", marker="^"),
    "Mn": dict(color="#FF7F0E", marker="o"),
    "Mg": dict(color="#2A9D8F", marker="s"),
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.6,
            "axes.titlesize": 11.2,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.8,
            "ytick.labelsize": 9.8,
            "legend.fontsize": 9.6,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "energy-gated-da-tpp-v50-mlip",
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
    common = dict(bbox_inches="tight", pad_inches=0.025, facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        **common,
        metadata={
            "Creator": "Energy-Gated DA-TPP v50 MLIP figure script",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(
        path.with_suffix(".svg"),
        **common,
        metadata={
            "Creator": "Energy-Gated DA-TPP v50 MLIP figure script",
            "Date": None,
        },
    )
    fig.savefig(
        path.with_suffix(".png"),
        dpi=600,
        **common,
        metadata={"Software": "Energy-Gated DA-TPP v50 MLIP figure script"},
    )
    fig.savefig(
        path.with_suffix(".tiff"),
        dpi=600,
        **common,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def validate_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    calibration = pd.read_csv(CALIBRATION_SOURCE)
    runtime = pd.read_csv(RUNTIME_SOURCE)
    observed = calibration["observed_dft_energy_eV_atom"].to_numpy(float)
    predicted = calibration["predicted_dft_energy_eV_atom"].to_numpy(float)
    mae = np.mean(np.abs(predicted - observed))
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    covered = np.mean(
        (observed >= calibration["prediction_interval_lower_95"].to_numpy(float))
        & (observed <= calibration["prediction_interval_upper_95"].to_numpy(float))
    )
    if len(calibration) != 10:
        raise AssertionError(f"expected 10 calibration rows, found {len(calibration)}")
    if not np.isclose(mae, 0.0144870305729443, atol=1e-12):
        raise AssertionError(f"unexpected calibration MAE: {mae}")
    if not np.isclose(rmse, 0.015928147383606, atol=1e-12):
        raise AssertionError(f"unexpected calibration RMSE: {rmse}")
    if not np.isclose(covered, 0.7, atol=1e-12):
        raise AssertionError(f"unexpected nominal interval coverage: {covered}")

    summary = runtime.groupby("model_name")["wall_time_seconds"].agg(
        ["count", "median", "sum"]
    )
    for model in ("CHGNet", "MACE-MP"):
        if int(summary.loc[model, "count"]) != 640:
            raise AssertionError(f"expected 640 {model} records")
    if not np.isclose(summary.loc["CHGNet", "median"], 0.582467, atol=5e-7):
        raise AssertionError("CHGNet median changed")
    if not np.isclose(summary.loc["MACE-MP", "median"], 0.407516, atol=5e-7):
        raise AssertionError("MACE-MP median changed")
    if not np.isclose(runtime["wall_time_seconds"].sum(), 781.5453303009272):
        raise AssertionError("aggregate MLIP wall time changed")
    return calibration, runtime


def draw_calibration(calibration: pd.DataFrame) -> None:
    observed = calibration["observed_dft_energy_eV_atom"].to_numpy(float)
    predicted = calibration["predicted_dft_energy_eV_atom"].to_numpy(float)
    mae = np.mean(np.abs(predicted - observed))
    rmse = np.sqrt(np.mean((predicted - observed) ** 2))
    lo = min(observed.min(), predicted.min()) - 0.035
    hi = max(observed.max(), predicted.max()) + 0.035

    fig, ax = plt.subplots(figsize=(4.35, 3.35))
    ax.plot(
        [lo, hi],
        [lo, hi],
        color="#8A99AC",
        linestyle="--",
        linewidth=1.15,
        label="Identity",
        zorder=0,
    )
    for element in ("Cr", "Mn", "Mg"):
        part = calibration.loc[calibration["m_element"].eq(element)]
        style = ELEMENT_STYLE[element]
        lower = (
            part["predicted_dft_energy_eV_atom"]
            - part["prediction_interval_lower_95"]
        )
        upper = (
            part["prediction_interval_upper_95"]
            - part["predicted_dft_energy_eV_atom"]
        )
        ax.errorbar(
            part["observed_dft_energy_eV_atom"],
            part["predicted_dft_energy_eV_atom"],
            yerr=np.vstack([lower, upper]),
            fmt=style["marker"],
            markersize=5.0,
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            color=style["color"],
            capsize=2.0,
            elinewidth=0.9,
            label=element,
            zorder=3,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("MACE-MP leave-one-out calibration", loc="left", fontweight="bold")
    ax.set_xlabel("Observed DFT formation energy (eV atom$^{-1}$)")
    ax.set_ylabel("Calibrated MACE-MP estimate (eV atom$^{-1}$)")
    format_axes(ax)
    ax.text(
        0.035,
        0.965,
        f"n = 10\nMAE = {mae:.4f} eV atom$^{{-1}}$\nRMSE = {rmse:.4f} eV atom$^{{-1}}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.0,
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor="#BDBDBD",
            linewidth=0.7,
            alpha=0.94,
        ),
    )
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.94,
        edgecolor="#BDBDBD",
        facecolor="white",
        borderpad=0.45,
        labelspacing=0.35,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.17)
    save_all(fig, "Figure4_v50_mace_dft_calibration")


def draw_runtime(runtime: pd.DataFrame) -> None:
    order = ("CHGNet", "MACE-MP")
    values = [
        runtime.loc[runtime["model_name"].eq(model), "wall_time_seconds"].to_numpy(
            float
        )
        for model in order
    ]
    colors = ("#2A9D8F", "#FFB15A")
    fig, ax = plt.subplots(figsize=(4.55, 3.0))
    parts = ax.violinplot(
        values,
        positions=(1, 2),
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors, strict=True):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)
    ax.boxplot(
        values,
        positions=(1, 2),
        widths=0.23,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="#111111", linewidth=1.15),
        boxprops=dict(facecolor="white", edgecolor="#404040", linewidth=0.8),
        whiskerprops=dict(color="#404040", linewidth=0.8),
        capprops=dict(color="#404040", linewidth=0.8),
    )
    for x, model_values in zip((1, 2), values, strict=True):
        median = np.median(model_values)
        ax.text(
            x,
            min(4.5, np.percentile(model_values, 98) + 0.18),
            f"n = 640\nmedian {median:.2f} s",
            ha="center",
            va="bottom",
            fontsize=9.0,
        )
    ax.set_title("Recorded full-pool MLIP wall time", loc="left", fontweight="bold")
    ax.set_xticks((1, 2), order)
    ax.set_ylabel("Wall time per candidate (s)")
    ax.set_ylim(0, 5.1)
    format_axes(ax)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.15)
    save_all(fig, "FigureS_v50_mlip_runtime")


def main() -> None:
    set_style()
    calibration, runtime = validate_sources()
    draw_calibration(calibration)
    draw_runtime(runtime)
    print(
        runtime.groupby("model_name")["wall_time_seconds"]
        .agg(["count", "median", "sum"])
        .to_string()
    )


if __name__ == "__main__":
    main()
