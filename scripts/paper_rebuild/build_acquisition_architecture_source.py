#!/usr/bin/env python3
"""Build the six v34 manuscript figures from frozen post-compute evidence.

This module performs plotting and source-data packaging only.  It does not run
active-learning, GPU, or DFT workloads.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from scipy.spatial import ConvexHull


MM = 1 / 25.4
T95_N10 = 2.2621571628540993
TARGET_LOW = -2.18
TARGET_HIGH = -2.02
NEAR_TARGET_DISTANCE = 0.20

METHOD_LABELS = {
    "interval_hit_greedy": "Greedy",
    "always_da_tpp": "Always-DA-TPP",
    "margin_only_gate": "Margin-only",
    "group_only_gate": "Group-only",
    "energy_gated_da_tpp": "Full Gate",
}
METHOD_ORDER = list(METHOD_LABELS)
METHOD_STYLE = {
    "interval_hit_greedy": dict(color="#4D4D4D", marker="o", linestyle="-"),
    "always_da_tpp": dict(color="#E69F00", marker="D", linestyle="-."),
    "margin_only_gate": dict(color="#7F7F7F", marker="^", linestyle="--"),
    "group_only_gate": dict(color="#56B4E9", marker="s", linestyle="--"),
    "energy_gated_da_tpp": dict(color="#0072B2", marker="o", linestyle="-"),
}
GROUP_LABELS = {
    "element_system_current": "Element system",
    "coelement_block_multiset": "Periodic block",
    "coelement_iupac_group_set": "IUPAC group",
}


@dataclass(frozen=True)
class V34FigurePackage:
    figures: dict[str, list[Path]]
    source_data: dict[str, Path]
    qa: dict[str, dict[str, object]]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3, width=0.8)


def _panel(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.05,
        label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=9,
        va="top",
        ha="left",
    )


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def _save(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = [figure_dir / f"{stem}.pdf", figure_dir / f"{stem}.svg", figure_dir / f"{stem}.png"]
    fig.savefig(outputs[0], bbox_inches=None)
    fig.savefig(outputs[1], bbox_inches=None)
    fig.savefig(outputs[2], dpi=600, bbox_inches=None, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return outputs


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _v33_path() -> Path:
    return Path.home() / "Desktop" / "Energy_Gated_DA_TPP_CMC_organized_v33.pdf"


def _extract_v33_figure1(v33_pdf: Path) -> tuple[Image.Image, dict[str, object]]:
    document = fitz.open(v33_pdf)
    page = document[1]
    images = page.get_images(full=True)
    if not images:
        raise RuntimeError("No embedded image found on v33 page 2")
    candidates: list[tuple[int, int, bytes, str]] = []
    for image_record in images:
        extracted = document.extract_image(image_record[0])
        candidates.append(
            (
                int(extracted["width"]),
                int(extracted["height"]),
                extracted["image"],
                extracted["ext"],
            )
        )
    width, height, payload, extension = max(candidates, key=lambda item: item[0] * item[1])
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    effective_dpi = width / (180 / 25.4)
    qa = {
        "source_pdf": str(v33_pdf),
        "source_page": 2,
        "source_sha256": _sha256(v33_pdf.read_bytes()),
        "embedded_image_sha256": _sha256(payload),
        "native_width_px": width,
        "native_height_px": height,
        "effective_dpi_at_180_mm": effective_dpi,
        "native_vector": False,
        "content_layout_changed": False,
        "font_audit": "narrative labels are consistently sans-serif, but mathematical glyphs use the embedded serif math font; normalization requires the unavailable native editable source",
        "arrow_alignment_audit": "no visible misalignment at native resolution",
        "notation_audit": "p_i, M_t, G_t, M_0, G_0 and the unqueried-pool symbol R_t agree with active v34 text; cumulative recovery is denoted C_t in the text to avoid collision",
        "submission_audit": "native raster is below 600 dpi at 180 mm; PDF/SVG outputs are raster containers",
        "embedded_extension": extension,
    }
    return image, qa


def _figure1(v33_pdf: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path, dict[str, object]]:
    image, qa = _extract_v33_figure1(v33_pdf)
    native = source_dir / "Figure1_v33_embedded.png"
    native.parent.mkdir(parents=True, exist_ok=True)
    image.save(native)
    source = pd.DataFrame(
        [
            {
                "record_type": "figure_audit",
                "source_pdf": qa["source_pdf"],
                "source_page": qa["source_page"],
                "source_sha256": qa["source_sha256"],
                "embedded_image_sha256": qa["embedded_image_sha256"],
                "native_width_px": qa["native_width_px"],
                "native_height_px": qa["native_height_px"],
                "effective_dpi_at_180_mm": qa["effective_dpi_at_180_mm"],
                "native_vector": qa["native_vector"],
                "content_layout_changed": qa["content_layout_changed"],
                "font_audit": qa["font_audit"],
                "arrow_alignment_audit": qa["arrow_alignment_audit"],
                "notation_audit": qa["notation_audit"],
                "submission_audit": qa["submission_audit"],
            }
        ]
    )
    source_path = _write_csv(source, source_dir / "Figure1_source.csv")
    fig = plt.figure(figsize=(180 * MM, 120 * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation="none")
    ax.set_axis_off()
    outputs = _save(fig, figure_dir, "Figure1_v34")
    return outputs, source_path, qa


def _figure2(repo: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path]:
    source = pd.read_csv(repo / "figure_source_data" / "candidate_pool_formation_energy_distributions.csv")
    source_path = _write_csv(source, source_dir / "Figure2_source.csv")
    tasks = list(source["task"].drop_duplicates())
    fig, axes = plt.subplots(1, 2, figsize=(180 * MM, 66 * MM), constrained_layout=True)
    colors = ["#0072B2", "#009E73"]
    for index, (ax, task, color) in enumerate(zip(axes, tasks, colors)):
        rows = source.loc[source["task"] == task]
        values = rows["formation_energy_eV_per_atom"].to_numpy(float)
        edges = np.histogram_bin_edges(values, bins="fd")
        ax.hist(values, bins=edges, color=color, alpha=0.18, edgecolor=color, linewidth=0.8)
        low = float(rows["target_lower_eV_per_atom"].iloc[0])
        high = float(rows["target_upper_eV_per_atom"].iloc[0])
        ax.axvspan(low, high, color="#F0E442", alpha=0.28, linewidth=0)
        ax.axvline(low, color="#A6761D", linestyle="--", linewidth=1.0)
        ax.axvline(high, color="#A6761D", linestyle="--", linewidth=1.0)
        targets = int(rows["target_indicator"].sum())
        label = "Li–M–O generated pool" if index == 0 else "Mn-oxide control pool"
        ax.set_title(label, loc="left", fontweight="bold")
        ax.text(
            0.98,
            0.94,
            f"N = {len(rows)}\nT = {targets} ({targets / len(rows):.1%})",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.set_xlabel("Formation energy (eV atom$^{-1}$)")
        ax.set_ylabel("Candidate count")
        _clean_axis(ax)
        _panel(ax, chr(ord("a") + index))
    outputs = _save(fig, figure_dir, "Figure2_v34")
    return outputs, source_path


def _limo_data(bundle: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = pd.read_csv(bundle / "gpu" / "complete_recovery_trajectories.csv")
    trajectories = trajectories.loc[
        (trajectories["formal_stage"] == "li_m_o_ablation")
        & (trajectories["dataset"] == "limo")
        & (trajectories["K"] == 30)
        & trajectories["seed"].between(15, 24)
    ].copy()
    metrics = pd.read_csv(bundle / "gpu" / "per_seed_metrics.csv")
    metrics = metrics.loc[
        (metrics["formal_stage"] == "li_m_o_ablation")
        & (metrics["dataset"] == "limo")
        & (metrics["K"] == 30)
        & metrics["seed"].between(15, 24)
    ].copy()
    return trajectories, metrics


def _trajectory_panel(ax: plt.Axes, trajectories: pd.DataFrame, xmax: int, label: str) -> None:
    plotted = ["interval_hit_greedy", "margin_only_gate", "always_da_tpp", "energy_gated_da_tpp"]
    for method in plotted:
        rows = trajectories.loc[trajectories["method"] == method]
        style = METHOD_STYLE[method]
        if method in {"interval_hit_greedy", "always_da_tpp", "energy_gated_da_tpp"}:
            for _, seed_rows in rows.groupby("seed"):
                seed_rows = seed_rows.sort_values("oracle_evaluations")
                ax.plot(
                    seed_rows["oracle_evaluations"],
                    seed_rows["cumulative_target_count"],
                    color=style["color"],
                    alpha=0.12,
                    linewidth=0.55,
                )
        summary = rows.groupby("oracle_evaluations")["cumulative_target_count"].agg(["mean", "std", "count"]).reset_index()
        ci = T95_N10 * summary["std"].fillna(0) / np.sqrt(summary["count"])
        legend_label = METHOD_LABELS[method]
        if method == "energy_gated_da_tpp":
            legend_label = "Full Gate = Group-only"
        ax.fill_between(
            summary["oracle_evaluations"],
            summary["mean"] - ci,
            summary["mean"] + ci,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )
        ax.plot(
            summary["oracle_evaluations"],
            summary["mean"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            label=legend_label,
        )
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 81)
    ax.set_xlabel("Oracle evaluations")
    ax.set_ylabel("Targets recovered")
    if xmax > 160:
        ax.set_xticks([0, 80, 160, 240, 320, 480, 640])
    else:
        ax.set_xticks([0, 40, 80, 120, 160])
        for budget in (80, 160):
            ax.axvline(budget, color="#BDBDBD", linewidth=0.65, linestyle=":" if budget == 80 else "--", zorder=0)
    _clean_axis(ax)
    _panel(ax, label)


def _figure3(bundle: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path]:
    trajectories, metrics = _limo_data(bundle)
    trajectory_source = trajectories.copy()
    trajectory_source.insert(0, "record_type", "trajectory")
    metric_source = metrics.copy()
    metric_source.insert(0, "record_type", "per_seed_metric")
    source = pd.concat([trajectory_source, metric_source], ignore_index=True, sort=False)
    source_path = _write_csv(source, source_dir / "Figure3_source.csv")

    fig = plt.figure(figsize=(180 * MM, 112 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])
    _trajectory_panel(ax_a, trajectories, 640, "a")
    _trajectory_panel(ax_b, trajectories, 160, "b")
    handles, labels = ax_a.get_legend_handles_labels()
    ax_a.legend(handles, labels, frameon=False, loc="lower right", ncol=1)

    positions = np.arange(len(METHOD_ORDER))
    for seed in range(15, 25):
        seed_rows = metrics.loc[metrics["seed"] == seed].set_index("method")
        ys = [seed_rows.loc[method, "AUTC"] for method in METHOD_ORDER]
        ax_c.plot(positions, ys, color="#BDBDBD", linewidth=0.55, alpha=0.5, zorder=1)
    for x, method in enumerate(METHOD_ORDER):
        values = metrics.loc[metrics["method"] == method, "AUTC"].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        style = METHOD_STYLE[method]
        ax_c.scatter(
            x + jitter,
            values,
            s=18,
            facecolor="white",
            edgecolor=style["color"],
            marker=style["marker"],
            linewidth=0.8,
            zorder=3,
        )
        mean = values.mean()
        ci = T95_N10 * values.std(ddof=1) / np.sqrt(len(values))
        ax_c.errorbar(x, mean, yerr=ci, fmt="_", color=style["color"], markersize=13, capsize=3, linewidth=1.5, zorder=4)
    ax_c.set_xticks(positions, [METHOD_LABELS[method] for method in METHOD_ORDER])
    ax_c.set_ylabel("AUTC")
    ax_c.set_xlim(-0.5, len(METHOD_ORDER) - 0.5)
    _clean_axis(ax_c)
    _panel(ax_c, "c")
    outputs = _save(fig, figure_dir, "Figure3_v34")
    return outputs, source_path


def _paired_stats(bundle: Path) -> pd.DataFrame:
    document = json.loads((bundle / "gpu" / "paired_statistics.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for comparison_id, item in document["comparisons"].items():
        if not comparison_id.startswith("li_m_o_ablation:limo:"):
            continue
        method = comparison_id.split(":")[2]
        wilcoxon = item["wilcoxon"]
        rows.append(
            {
                "record_type": "paired_statistic",
                "comparison_id": comparison_id,
                "method": method,
                "paired_mean": item["paired_mean"],
                "paired_sd": item["paired_sd"],
                "bootstrap_low": item["bootstrap_ci_95_percentile"][0],
                "bootstrap_high": item["bootstrap_ci_95_percentile"][1],
                "exact_wilcoxon_p": wilcoxon["pvalue"],
                "wilcoxon_status": wilcoxon["status"],
                "dz": item["effect_size_dz"],
                "bootstrap_samples": item["bootstrap_samples"],
                "bootstrap_seed": item["bootstrap_seed"],
            }
        )
    return pd.DataFrame(rows)


def _figure4(bundle: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path]:
    _, metrics = _limo_data(bundle)
    differences = pd.read_csv(bundle / "gpu" / "paired_differences.csv")
    differences = differences.loc[
        (differences["formal_stage"] == "li_m_o_ablation")
        & differences["seed"].between(15, 24)
    ].copy()
    gate = pd.read_csv(bundle / "gpu" / "round_gate_evidence.csv")
    gate = gate.loc[
        (gate["formal_stage"] == "li_m_o_ablation")
        & gate["seed"].between(15, 24)
        & gate["method"].isin(["energy_gated_da_tpp", "margin_only_gate", "group_only_gate"])
    ].copy()
    activation = (
        gate.assign(correction=(gate["route"] == "diversity_aware").astype(float))
        .groupby(["method", "round"], as_index=False)["correction"]
        .mean()
        .rename(columns={"correction": "correction_fraction"})
    )
    stat_source = _paired_stats(bundle)
    source_frames = []
    for kind, frame in (
        ("per_seed_metric", metrics),
        ("paired_difference", differences),
        ("route_activation", activation),
    ):
        item = frame.copy()
        item.insert(0, "record_type", kind)
        source_frames.append(item)
    source_frames.append(stat_source)
    source = pd.concat(source_frames, ignore_index=True, sort=False)
    source_path = _write_csv(source, source_dir / "Figure4_source.csv")

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 122 * MM), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    positions = np.arange(len(METHOD_ORDER))
    for x, method in enumerate(METHOD_ORDER):
        values = metrics.loc[metrics["method"] == method, "AUTC"].to_numpy(float)
        jitter = np.linspace(-0.12, 0.12, len(values))
        style = METHOD_STYLE[method]
        ax_a.scatter(x + jitter, values, s=15, facecolor="white", edgecolor=style["color"], marker=style["marker"], linewidth=0.75)
        mean = values.mean()
        ci = T95_N10 * values.std(ddof=1) / np.sqrt(len(values))
        ax_a.errorbar(x, mean, yerr=ci, fmt="_", color=style["color"], markersize=12, capsize=3, linewidth=1.4)
    ax_a.set_xticks(positions, ["Greedy", "Always", "Margin", "Group", "Full"])
    ax_a.set_ylabel("AUTC")
    _clean_axis(ax_a)
    _panel(ax_a, "a")

    comparison_order = ["always_da_tpp", "margin_only_gate", "group_only_gate", "energy_gated_da_tpp"]
    y_positions = np.arange(len(comparison_order))[::-1]
    ax_b.axvline(0, color="#4D4D4D", linewidth=0.8, linestyle=":")
    for y, method in zip(y_positions, comparison_order):
        values = differences.loc[differences["method"] == method, "paired_AUTC_difference"].to_numpy(float)
        stat = stat_source.loc[stat_source["method"] == method].iloc[0]
        style = METHOD_STYLE[method]
        y_jitter = np.linspace(-0.11, 0.11, len(values))
        ax_b.scatter(values, y + y_jitter, s=13, facecolor="white", edgecolor=style["color"], marker=style["marker"], linewidth=0.7, zorder=2)
        low, high, mean = float(stat["bootstrap_low"]), float(stat["bootstrap_high"]), float(stat["paired_mean"])
        ax_b.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt=style["marker"], color=style["color"], mfc=style["color"], ms=4, capsize=3, linewidth=1.5, zorder=3)
        p_text = "not defined" if pd.isna(stat["exact_wilcoxon_p"]) else f"{float(stat['exact_wilcoxon_p']):.7g}"
        ax_b.text(
            0.044,
            y,
            f"Δ={mean:.4f}  95% CI [{low:.4f}, {high:.4f}]\np={p_text}; d$_z$={float(stat['dz']):.3f}",
            va="center",
            ha="left",
            fontsize=6.1,
        )
    ax_b.set_yticks(y_positions, [METHOD_LABELS[m] for m in comparison_order])
    ax_b.set_xlim(-0.03, 0.102)
    ax_b.set_xlabel("Paired AUTC difference vs Greedy")
    _clean_axis(ax_b)
    _panel(ax_b, "b")

    for method in METHOD_ORDER:
        rows = metrics.loc[metrics["method"] == method]
        style = METHOD_STYLE[method]
        ax_c.scatter(
            rows["correction_rounds"],
            rows["effective_replacements"],
            label=METHOD_LABELS[method],
            marker=style["marker"],
            color=style["color"],
            facecolor="white",
            linewidth=0.75,
            s=18,
            alpha=0.9,
        )
        ax_c.scatter(rows["correction_rounds"].mean(), rows["effective_replacements"].mean(), marker=style["marker"], color=style["color"], s=25)
    ax_c.set_xlabel("Correction rounds")
    ax_c.set_ylabel("Replacement count")
    ax_c.legend(frameon=False, ncol=2, loc="upper left")
    _clean_axis(ax_c)
    _panel(ax_c, "c")

    route_methods = ["energy_gated_da_tpp", "margin_only_gate", "group_only_gate"]
    matrix = np.vstack(
        [
            activation.loc[activation["method"] == method].sort_values("round")["correction_fraction"].to_numpy(float)
            for method in route_methods
        ]
    )
    cmap = LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", "#56B4E9", "#0072B2"])
    image = ax_d.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax_d.set_yticks(range(3), ["Full Gate", "Margin-only", "Group-only"])
    ax_d.set_xticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
    ax_d.set_xlabel("Acquisition round")
    ax_d.set_title("Fraction of seeds routed to correction", loc="left")
    ax_d.tick_params(direction="out", length=3, width=0.8)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax_d, fraction=0.04, pad=0.02)
    colorbar.set_ticks([0, 0.5, 1])
    colorbar.outline.set_linewidth(0.6)
    _panel(ax_d, "d")
    outputs = _save(fig, figure_dir, "Figure4_v34")
    return outputs, source_path


def _figure5(bundle: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path]:
    summary = pd.read_csv(bundle / "gpu" / "mn_group_key_sensitivity_summary.csv")
    metrics = pd.read_csv(bundle / "gpu" / "per_seed_metrics.csv")
    metrics = metrics.loc[
        (metrics["formal_stage"] == "mn_group_key")
        & (metrics["dataset"] == "mnoxide")
        & (metrics["K"] == 30)
        & metrics["seed"].between(15, 24)
    ].copy()
    greedy = metrics.loc[metrics["method"] == "interval_hit_greedy", ["seed", "AUTC"]].rename(columns={"AUTC": "Greedy_AUTC"})
    full = metrics.loc[metrics["method"] == "energy_gated_da_tpp"].merge(greedy, on="seed", how="left")
    full["paired_AUTC_difference"] = full["AUTC"] - full["Greedy_AUTC"]
    summary_source = summary.copy()
    summary_source.insert(0, "record_type", "group_summary")
    paired_source = full.copy()
    paired_source.insert(0, "record_type", "paired_difference")
    source = pd.concat([summary_source, paired_source], ignore_index=True, sort=False)
    source_path = _write_csv(source, source_dir / "Figure5_source.csv")

    fig, axes = plt.subplots(2, 2, figsize=(180 * MM, 112 * MM), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    key_order = list(GROUP_LABELS)
    ordered = summary.set_index("group_key").loc[key_order].reset_index()
    matrix_raw = np.column_stack(
        [
            ordered["group_count"].to_numpy(float),
            ordered["singleton_group_fraction"].to_numpy(float),
            ordered["formal_top_b_concentration_max"].to_numpy(float),
        ]
    )
    denominators = np.maximum(matrix_raw.max(axis=0), 1e-12)
    matrix = matrix_raw / denominators
    cmap = LinearSegmentedColormap.from_list("white_green", ["#FFFFFF", "#9ED9C5", "#009E73"])
    ax_a.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_a.set_xticks(range(3), ["Group\ncount", "Singleton\nfraction", "Max $G_t$"])
    ax_a.set_yticks(range(3), [GROUP_LABELS[key] for key in key_order])
    for row in range(3):
        annotations = [f"{matrix_raw[row, 0]:.0f}", f"{matrix_raw[row, 1]:.1%}", f"{matrix_raw[row, 2]:.3f}"]
        for col, text in enumerate(annotations):
            ax_a.text(col, row, text, ha="center", va="center", color="black", fontsize=7)
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    _panel(ax_a, "a")

    x = np.arange(3)
    ax_b.axhline(0.75, color="#D55E00", linestyle="--", linewidth=1.2, label="$M_0=0.75$")
    ax_b.plot(x, ordered["minimum_margin_score"], color="#0072B2", marker="o", linewidth=1.6, label="Minimum $M_t$")
    ax_b.set_xticks(x, [GROUP_LABELS[key] for key in key_order])
    ax_b.set_ylabel("Margin score")
    ax_b.set_ylim(0.65, max(1.22, ordered["minimum_margin_score"].max() + 0.08))
    ax_b.legend(frameon=False, loc="upper right")
    _clean_axis(ax_b)
    _panel(ax_b, "b")

    ax_c.plot(x - 0.06, ordered["correction_rounds_total"], marker="o", color="#0072B2", linestyle="none", label="Correction rounds")
    ax_c.plot(x + 0.06, ordered["effective_replacements_total"], marker="s", mfc="white", color="#E69F00", linestyle="none", label="Replacements")
    for index in range(3):
        ax_c.text(index, 0.025, "0 / 0", ha="center", va="bottom", fontsize=6.5)
    ax_c.set_xticks(x, [GROUP_LABELS[key] for key in key_order])
    ax_c.set_ylabel("Count across 10 seeds")
    ax_c.set_ylim(-0.03, 0.20)
    ax_c.set_yticks([0, 0.1, 0.2])
    ax_c.legend(frameon=False, loc="upper right")
    _clean_axis(ax_c)
    _panel(ax_c, "c")

    ax_d.axhline(0, color="#4D4D4D", linewidth=0.8, linestyle=":")
    for index, key in enumerate(key_order):
        values = full.loc[full["group_key"] == key, "paired_AUTC_difference"].to_numpy(float)
        jitter = np.linspace(-0.11, 0.11, len(values))
        ax_d.scatter(index + jitter, values, s=18, facecolor="white", edgecolor="#0072B2", linewidth=0.8)
        ax_d.text(index, 0.00018, "10 seeds at Δ=0", ha="center", va="bottom", fontsize=6.4)
    ax_d.set_xticks(x, [GROUP_LABELS[key] for key in key_order])
    ax_d.set_ylabel("Full Gate − Greedy AUTC")
    ax_d.set_ylim(-0.00045, 0.00065)
    _clean_axis(ax_d)
    _panel(ax_d, "d")
    outputs = _save(fig, figure_dir, "Figure5_v34")
    return outputs, source_path


def _candidate_label(candidate_id: str) -> str:
    match = re.search(r"job_(\d{3})_", candidate_id)
    return f"C{match.group(1)}" if match else candidate_id


def _distance_to_interval(value: float) -> float:
    if value < TARGET_LOW:
        return TARGET_LOW - value
    if value > TARGET_HIGH:
        return value - TARGET_HIGH
    return 0.0


def _plot_structure(ax: plt.Axes, cif_path: Path, candidate_label: str, formula: str) -> None:
    from pymatgen.core import Structure

    structure = Structure.from_file(cif_path)
    colors = {"Li": "#61D836", "Cr": "#2C7FB8", "Mn": "#7A5195", "O": "#D73027"}
    sizes = {"Li": 34, "Cr": 55, "Mn": 58, "O": 25}
    lattice = structure.lattice.matrix
    corners = np.array(
        [
            i * lattice[0] + j * lattice[1] + k * lattice[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    edges = []
    for i, left in enumerate(corners):
        for right in corners[i + 1 :]:
            delta = right - left
            coefficients = np.linalg.solve(lattice.T, delta)
            if np.count_nonzero(np.isclose(np.abs(coefficients), 1, atol=1e-6)) == 1 and np.count_nonzero(np.isclose(coefficients, 0, atol=1e-6)) == 2:
                edges.append((left, right))
    for left, right in edges:
        ax.plot(*zip(left, right), color="#8C8C8C", linewidth=0.45, alpha=0.65)

    plotted_neighbors: list[np.ndarray] = []
    for site in structure:
        symbol = site.specie.symbol
        if symbol not in {"Cr", "Mn"}:
            continue
        neighbors = [neighbor for neighbor in structure.get_neighbors(site, 2.35) if neighbor.specie.symbol == "O"]
        points = np.array([neighbor.coords for neighbor in neighbors])
        if len(points) >= 4:
            try:
                hull = ConvexHull(points)
                faces = [points[simplex] for simplex in hull.simplices]
                collection = Poly3DCollection(faces, alpha=0.10, facecolor=colors[symbol], edgecolor=colors[symbol], linewidth=0.35)
                ax.add_collection3d(collection)
            except Exception:
                pass
        for point in points:
            ax.plot(*zip(site.coords, point), color=colors[symbol], linewidth=0.45, alpha=0.45)
            plotted_neighbors.append(point)
    for site in structure:
        symbol = site.specie.symbol
        ax.scatter(*site.coords, s=sizes.get(symbol, 28), color=colors.get(symbol, "#777777"), edgecolor="white", linewidth=0.35, depthshade=False)
    if plotted_neighbors:
        neighbor_array = np.unique(np.round(np.vstack(plotted_neighbors), 5), axis=0)
        ax.scatter(neighbor_array[:, 0], neighbor_array[:, 1], neighbor_array[:, 2], s=sizes["O"], color=colors["O"], edgecolor="white", linewidth=0.3, depthshade=False)
    all_points = np.vstack([corners, structure.cart_coords, *([np.vstack(plotted_neighbors)] if plotted_neighbors else [])])
    mins, maxs = all_points.min(axis=0), all_points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(maxs - mins) * 0.58
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=20, azim=35)
    ax.set_axis_off()
    ax.set_title(f"{candidate_label}\n{formula}", fontsize=6.5, pad=1)


def _figure6(repo: Path, bundle: Path, figure_dir: Path, source_dir: Path) -> tuple[list[Path], Path, dict[str, object]]:
    alignn = pd.read_csv(repo / "new12_dft_final" / "NEW12_DFT_RESULTS.csv")
    alignn = alignn[["candidate_id", "formula", "alignn_formation_energy"]].drop_duplicates("candidate_id")
    energies = pd.read_csv(bundle / "dft" / "recomputed_formation_energies.csv")
    energies = energies.loc[energies["selected_for_formation_energy"] == True].copy()  # noqa: E712
    energies = energies.merge(alignn, on=["candidate_id", "formula"], how="left")
    energies["candidate_label"] = energies["candidate_id"].map(_candidate_label)
    energies["energy_standard"] = energies["functional"].replace({"GGA+U": "PBE+U"})
    energies["distance_to_target_interval_eV_per_atom"] = energies["formation_energy_eV_per_atom"].map(_distance_to_interval)
    energy_source = energies.copy()
    energy_source.insert(0, "record_type", "energy_point")

    main = pd.read_csv(bundle / "dft" / "main_text_table7_comparison.csv")
    structure_metrics = pd.read_csv(bundle / "dft" / "structure_metrics.csv")
    metric_rows: list[dict[str, object]] = []
    for _, candidate in main.iterrows():
        selected = structure_metrics.loc[
            (structure_metrics["candidate_id"] == candidate["candidate_id"])
            & (structure_metrics["functional"] == "GGA+U")
            & (structure_metrics["magnetic_initialization"] == candidate["selected_magnetic_initialization"])
        ]
        if selected.empty:
            raise RuntimeError(f"Missing selected structure metrics for {candidate['candidate_label']}")
        row = selected.iloc[0]
        label = candidate["candidate_label"]
        completed = str(candidate.get("verification_status", "")) == (
            "completed_frozen_protocol_relaxation_and_static"
        )
        relaxation_volume_change = (
            row.get("verification_relative_volume_change_percent")
            if completed
            else row["historical_selected_configuration_relative_volume_change_percent"]
        )
        relaxation_fmax = (
            row.get("verification_relaxation_Fmax_eV_A")
            if completed
            else row["historical_selected_configuration_relaxation_Fmax_eV_A"]
        )
        metric_rows.append(
            {
                "record_type": "main_candidate_metric",
                "candidate_label": label,
                "candidate_id": candidate["candidate_id"],
                "formula": row["formula"],
                "selected_magnetic_initialization": candidate["selected_magnetic_initialization"],
                "recomputed_formation_energy_eV_per_atom": candidate["recomputed_formation_energy_eV_per_atom"],
                "historical_volume_change_percent": row["historical_selected_configuration_relative_volume_change_percent"],
                "relaxation_volume_change_percent": relaxation_volume_change,
                "minimum_interatomic_distance_A": row["minimum_interatomic_distance_A"],
                "minimum_M_O_distance_A": row["minimum_M_O_distance_A"],
                "static_Fmax_eV_A": row["Fmax_eV_A_static_diagnostic"],
                "historical_relaxation_Fmax_eV_A": row["historical_selected_configuration_relaxation_Fmax_eV_A"],
                "relaxation_Fmax_eV_A": relaxation_fmax,
                "static_force_status": "frozen static diagnostic",
                "relaxation_force_status": (
                    "completed verification relaxation"
                    if completed
                    else "historical archived relaxation"
                ),
                "verification_status": (
                    "completed frozen-protocol verification"
                    if completed
                    else (
                        "archived assessment"
                        if label == "C044"
                        else "verification relaxation pending"
                    )
                ),
                "verification_cif_path": row["verification_cif_path"],
                "visualization_view_elev_deg": 20,
                "visualization_view_azim_deg": 35,
                "polyhedron_M_O_cutoff_A": 2.35,
                "visualization_software": f"pymatgen {version('pymatgen')}; matplotlib {matplotlib.__version__}",
            }
        )
    metric_source = pd.DataFrame(metric_rows)
    source = pd.concat([energy_source, metric_source], ignore_index=True, sort=False)
    source_path = _write_csv(source, source_dir / "Figure6_source.csv")

    fig = plt.figure(figsize=(180 * MM, 126 * MM), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.0, 0.92])
    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    ax_c = fig.add_subplot(grid[1, 0:3])
    structure_axes = [fig.add_subplot(grid[1, index], projection="3d") for index in (3, 4, 5)]

    identity_values = alignn["alignn_formation_energy"].dropna().to_numpy(float)
    ax_a.scatter(identity_values, identity_values, marker="o", s=18, facecolor="white", edgecolor="#4D4D4D", linewidth=0.7, label="ALIGNN reference")
    for standard, marker, color in (("PBE", "^", "#009E73"), ("PBE+U", "s", "#D55E00")):
        rows = energies.loc[energies["energy_standard"] == standard]
        ax_a.scatter(rows["alignn_formation_energy"], rows["formation_energy_eV_per_atom"], marker=marker, s=23, facecolor="white", edgecolor=color, linewidth=0.9, label=standard)
    limits = [-2.42, -1.45]
    ax_a.plot(limits, limits, color="#9E9E9E", linewidth=0.8, linestyle=":", zorder=0)
    ax_a.axvspan(TARGET_LOW, TARGET_HIGH, color="#F0E442", alpha=0.16, linewidth=0)
    ax_a.axhspan(TARGET_LOW, TARGET_HIGH, color="#F0E442", alpha=0.16, linewidth=0)
    ax_a.axhline(TARGET_LOW - NEAR_TARGET_DISTANCE, color="#A6761D", linestyle="--", linewidth=0.7)
    ax_a.axhline(TARGET_HIGH + NEAR_TARGET_DISTANCE, color="#A6761D", linestyle="--", linewidth=0.7)
    ax_a.set_xlim(limits)
    ax_a.set_ylim(limits)
    ax_a.set_xlabel("ALIGNN reference (eV atom$^{-1}$)")
    ax_a.set_ylabel("Energy under stated convention (eV atom$^{-1}$)")
    ax_a.legend(frameon=False, loc="lower right")
    _clean_axis(ax_a)
    _panel(ax_a, "a")

    selected = energies.sort_values(["candidate_id", "energy_standard"]).groupby("candidate_id", as_index=False).apply(
        lambda rows: rows.loc[rows["energy_standard"].eq("PBE+U")].iloc[0] if rows["energy_standard"].eq("PBE+U").any() else rows.iloc[0],
        include_groups=False,
    )
    selected = selected.sort_values("distance_to_target_interval_eV_per_atom")
    bx = np.arange(len(selected))
    bcolors = ["#D55E00" if standard == "PBE+U" else "#009E73" for standard in selected["energy_standard"]]
    ax_b.vlines(bx, 0, selected["distance_to_target_interval_eV_per_atom"], color=bcolors, linewidth=1.1)
    ax_b.scatter(bx, selected["distance_to_target_interval_eV_per_atom"], color=bcolors, marker="o", s=20, zorder=2)
    ax_b.axhspan(0, NEAR_TARGET_DISTANCE, color="#F0E442", alpha=0.12, linewidth=0)
    ax_b.axhline(NEAR_TARGET_DISTANCE, color="#A6761D", linestyle="--", linewidth=0.9, label="Exploratory near-target limit")
    ax_b.set_xticks(bx, selected["candidate_label"], rotation=45, ha="right")
    ax_b.set_ylabel("Distance to target interval (eV atom$^{-1}$)")
    ax_b.text(0.02, 0.05, "Exact target: distance = 0", transform=ax_b.transAxes, fontsize=6.6)
    ax_b.legend(frameon=False, loc="upper right")
    _clean_axis(ax_b)
    _panel(ax_b, "b")

    main_order = ["C044", "C120", "C214"]
    main_metrics = metric_source.set_index("candidate_label").loc[main_order]
    metric_columns = [
        "relaxation_volume_change_percent",
        "minimum_interatomic_distance_A",
        "minimum_M_O_distance_A",
        "static_Fmax_eV_A",
        "relaxation_Fmax_eV_A",
    ]
    metric_labels = [
        "ΔV\n(%)",
        "Min pair\n(Å)",
        "Min M–O\n(Å)",
        "Static\n$F_{max}$",
        "Relaxation\n$F_{max}$*",
    ]
    raw = main_metrics[metric_columns].to_numpy(float)
    spread = np.ptp(raw, axis=0)
    normalized = (raw - raw.min(axis=0)) / np.where(spread == 0, 1, spread)
    cmap = LinearSegmentedColormap.from_list("white_orange", ["#FFFFFF", "#F6C46B", "#D55E00"])
    ax_c.imshow(normalized, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax_c.set_xticks(range(len(metric_labels)), metric_labels)
    ax_c.set_yticks(range(3), main_order)
    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            suffix = "*" if row in (1, 2) and col == 4 else ""
            ax_c.text(col, row, f"{raw[row, col]:.3f}{suffix}", ha="center", va="center", fontsize=6.3)
    for spine in ax_c.spines.values():
        spine.set_visible(False)
    pending_candidates = metric_source.loc[
        metric_source["verification_status"].eq("verification relaxation pending"),
        "candidate_label",
    ].tolist()
    relaxation_note = (
        "* Historical relaxation metric; C120/C214 verification is pending."
        if pending_candidates
        else "* C044 uses the archived relaxation; C120/C214 use verification relaxations."
    )
    ax_c.text(
        0.0,
        -0.28,
        relaxation_note,
        transform=ax_c.transAxes,
        fontsize=6.2,
        ha="left",
    )
    _panel(ax_c, "c")

    for index, (ax, label) in enumerate(zip(structure_axes, main_order)):
        row = main_metrics.loc[label]
        _plot_structure(ax, Path(row["verification_cif_path"]), label, row["formula"])
        if index == 0:
            ax.text2D(-0.28, 1.05, "d", transform=ax.transAxes, fontweight="bold", fontsize=9, va="top")
    qa = {
        "visualization_software": metric_source["visualization_software"].iloc[0],
        "view_elev_deg": 20,
        "view_azim_deg": 35,
        "polyhedron_M_O_cutoff_A": 2.35,
        "candidate_cifs": {row["candidate_label"]: row["verification_cif_path"] for _, row in metric_source.iterrows()},
        "pending_candidates": pending_candidates,
    }
    outputs = _save(fig, figure_dir, "Figure6_v34")
    return outputs, source_path, qa


def build_v34_figures(
    repo_root: Path,
    output_root: Path,
    v33_pdf: Path | None = None,
    bundle_override: Path | None = None,
) -> V34FigurePackage:
    repo = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    figure_dir = output / "Figures"
    source_dir = output / "SourceData"
    bundle = (
        Path(bundle_override).resolve()
        if bundle_override is not None
        else repo
        / "results"
        / "post_submission_analysis"
        / "egdatpp_psfix_v1_20260719T031102Z"
    )
    v33 = Path(v33_pdf) if v33_pdf else _v33_path()
    _configure_style()

    figures: dict[str, list[Path]] = {}
    sources: dict[str, Path] = {}
    qa: dict[str, dict[str, object]] = {}
    figures["Figure1"], sources["Figure1"], qa["Figure1"] = _figure1(v33, figure_dir, source_dir)
    figures["Figure2"], sources["Figure2"] = _figure2(repo, figure_dir, source_dir)
    figures["Figure3"], sources["Figure3"] = _figure3(bundle, figure_dir, source_dir)
    figures["Figure4"], sources["Figure4"] = _figure4(bundle, figure_dir, source_dir)
    figures["Figure5"], sources["Figure5"] = _figure5(bundle, figure_dir, source_dir)
    figures["Figure6"], sources["Figure6"], qa["Figure6"] = _figure6(repo, bundle, figure_dir, source_dir)
    return V34FigurePackage(figures=figures, source_data=sources, qa=qa)


def build_v34_figure(
    figure_number: int,
    repo_root: Path,
    output_root: Path,
    v33_pdf: Path | None = None,
    bundle_override: Path | None = None,
) -> V34FigurePackage:
    """Build one figure and its source CSV without rebuilding the other panels."""
    repo = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    figure_dir = output / "Figures"
    source_dir = output / "SourceData"
    bundle = (
        Path(bundle_override).resolve()
        if bundle_override is not None
        else repo
        / "results"
        / "post_submission_analysis"
        / "egdatpp_psfix_v1_20260719T031102Z"
    )
    _configure_style()
    key = f"Figure{figure_number}"
    qa: dict[str, dict[str, object]] = {}
    if figure_number == 1:
        outputs, source, qa[key] = _figure1(Path(v33_pdf) if v33_pdf else _v33_path(), figure_dir, source_dir)
    elif figure_number == 2:
        outputs, source = _figure2(repo, figure_dir, source_dir)
    elif figure_number == 3:
        outputs, source = _figure3(bundle, figure_dir, source_dir)
    elif figure_number == 4:
        outputs, source = _figure4(bundle, figure_dir, source_dir)
    elif figure_number == 5:
        outputs, source = _figure5(bundle, figure_dir, source_dir)
    elif figure_number == 6:
        outputs, source, qa[key] = _figure6(repo, bundle, figure_dir, source_dir)
    else:
        raise ValueError("figure_number must be an integer from 1 through 6")
    return V34FigurePackage(figures={key: outputs}, source_data={key: source}, qa=qa)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v33-pdf", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--figure", type=int, choices=range(1, 7))
    args = parser.parse_args()
    package = (
        build_v34_figure(
            args.figure,
            args.repo_root,
            args.output,
            args.v33_pdf,
            args.bundle,
        )
        if args.figure
        else build_v34_figures(
            args.repo_root,
            args.output,
            args.v33_pdf,
            args.bundle,
        )
    )
    manifest = {
        "figures": {key: [str(path) for path in paths] for key, paths in package.figures.items()},
        "source_data": {key: str(path) for key, path in package.source_data.items()},
        "qa": package.qa,
    }
    manifest_path = Path(args.output) / "figure_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
