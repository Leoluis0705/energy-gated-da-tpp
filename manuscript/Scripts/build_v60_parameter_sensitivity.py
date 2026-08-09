"""Build the source-backed V60 gate-parameter sensitivity evidence.

The script reads completed Li--M--O development runs only.  It never edits a
run directory and it does not select new manuscript parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_SEEDS = tuple(range(5))
ARCHIVED = {"M0": 1.0, "G0": 0.5, "alpha": 0.1, "beta": 0.2, "gamma": 0.1}
PURPLE = "#9467BD"
GREEN = "#2CA02C"
BLUE = "#1F77B4"
ORANGE = "#FF7F0E"
RED = "#D62728"


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
            "legend.fontsize": 8.1,
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_stage(root: Path, stage: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for config_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if config_dir.name == "logs":
            continue
        for seed in EXPECTED_SEEDS:
            run = config_dir / f"seed_{seed}" / "attempt_1"
            status_path = run / "status.json"
            metrics_path = run / "run_metrics.csv"
            config_path = run / "run_config.json"
            summary_path = run / "summary.csv"
            missing = [path for path in (status_path, metrics_path, config_path, summary_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError("; ".join(str(path) for path in missing))
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "DONE":
                raise RuntimeError(f"run is not DONE: {run}")
            metrics = pd.read_csv(metrics_path)
            if len(metrics) != 1:
                raise ValueError(f"expected one metric row: {metrics_path}")
            metric = metrics.iloc[0]
            trajectory = pd.read_csv(summary_path)
            early = trajectory[trajectory["oracle_evaluations"] <= 160]
            if early.empty or int(early.iloc[-1]["oracle_evaluations"]) != 160:
                raise ValueError(f"missing 160-query trajectory row: {summary_path}")
            config = json.loads(config_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "stage": stage,
                    "config_id": config_dir.name,
                    "seed": seed,
                    "M0": float(config["M0"]),
                    "G0": float(config["G0"]),
                    "alpha": float(config["alpha"]),
                    "beta": float(config["beta"]),
                    "gamma": float(config["gamma"]),
                    "mc_passes": int(config["mc_passes"]),
                    "AUTC": float(metric["AUTC"]),
                    "AUTC_160": float(early.iloc[-1]["AUTC_so_far"]),
                    "recovery_at_80": int(metric["recovery_at_80"]),
                    "recovery_at_160": int(metric["recovery_at_160"]),
                    "recovery_at_240": int(metric["recovery_at_240"]),
                    "recovery_at_320": int(metric["recovery_at_320"]),
                    "correction_rounds": int(metric["correction_rounds"]),
                    "effective_replacements": int(metric["total_correction_replacements"]),
                    "sequence_sha256": str(metric["candidate_sequence_sha256"]),
                    "runtime_seconds": float(status["elapsed_seconds"]),
                    "run_config_sha256": sha256(config_path),
                    "run_metrics_sha256": sha256(metrics_path),
                    "summary_sha256": sha256(summary_path),
                    "run_path": str(run),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no runs found under {root}")
    counts = frame.groupby("config_id")["seed"].agg(lambda values: tuple(sorted(values)))
    invalid = counts[counts != EXPECTED_SEEDS]
    if not invalid.empty:
        raise ValueError(f"incomplete seed cohorts: {invalid.to_dict()}")
    return frame.sort_values(["config_id", "seed"]).reset_index(drop=True)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    fixed = ["stage", "M0", "G0", "alpha", "beta", "gamma", "mc_passes"]
    records: list[dict[str, object]] = []
    for config_id, block in per_seed.groupby("config_id", sort=True):
        if any(block[column].nunique(dropna=False) != 1 for column in fixed):
            raise ValueError(f"fixed fields vary within {config_id}")
        records.append(
            {
                "config_id": config_id,
                **{column: block.iloc[0][column] for column in fixed},
                "seed_count": len(block),
                "mean_AUTC": block["AUTC"].mean(),
                "sd_AUTC": block["AUTC"].std(ddof=1),
                "min_AUTC": block["AUTC"].min(),
                "max_AUTC": block["AUTC"].max(),
                "mean_AUTC_160": block["AUTC_160"].mean(),
                "sd_AUTC_160": block["AUTC_160"].std(ddof=1),
                "min_AUTC_160": block["AUTC_160"].min(),
                "max_AUTC_160": block["AUTC_160"].max(),
                "mean_recovery_at_80": block["recovery_at_80"].mean(),
                "sd_recovery_at_80": block["recovery_at_80"].std(ddof=1),
                "mean_correction_rounds": block["correction_rounds"].mean(),
                "sd_correction_rounds": block["correction_rounds"].std(ddof=1),
                "mean_effective_replacements": block["effective_replacements"].mean(),
                "unique_sequences": block["sequence_sha256"].nunique(),
                "mean_runtime_seconds": block["runtime_seconds"].mean(),
            }
        )
    return pd.DataFrame(records)


def label_weight(row: pd.Series) -> str:
    changed = [name for name in ("alpha", "beta", "gamma") if not np.isclose(float(row[name]), ARCHIVED[name])]
    if not changed:
        return "Archived center"
    name = changed[0]
    symbol = {"alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma"}[name]
    return rf"${symbol}={float(row[name]):g}$"


def plot(summary: pd.DataFrame, output_stem: Path) -> None:
    apply_original_figure_style()
    threshold = summary[summary["stage"] == "threshold"].copy()
    weights = summary[summary["stage"] == "weight"].copy()
    m_values = sorted(threshold["M0"].unique())
    g_values = sorted(threshold["G0"].unique())
    autc = threshold.pivot(index="M0", columns="G0", values="mean_AUTC_160").loc[m_values, g_values]
    rounds = threshold.pivot(index="M0", columns="G0", values="mean_correction_rounds").loc[m_values, g_values]

    # Export close to the final full-text width so LaTeX does not shrink labels
    # below the journal's readable 7--8 pt range.
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.05), constrained_layout=True)
    for ax, matrix, title, fmt, cmap in (
        (axes[0], autc, r"a  Early AUTC ($q\leq160$)", ".4f", "Purples"),
        (axes[1], rounds, "b  Correction-route batches", ".1f", "Greens"),
    ):
        # Use vector cells in PDF output; imshow would embed a raster heatmap.
        x_edges = np.arange(matrix.shape[1] + 1) - 0.5
        y_edges = np.arange(matrix.shape[0] + 1) - 0.5
        image = ax.pcolormesh(x_edges, y_edges, matrix.values, cmap=cmap, shading="flat")
        ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
        ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix.iloc[i, j], fmt), ha="center", va="center", fontsize=8.5)
        ax.set_xticks(range(len(g_values)), [f"{value:.2f}" for value in g_values])
        ax.set_yticks(range(len(m_values)), [f"{value:.2f}" for value in m_values])
        ax.set_xlabel(r"Group threshold $G_0$")
        ax.set_ylabel(r"Margin threshold $M_0$")
        ax.set_title(title, loc="left", fontweight="bold")
        center_i = m_values.index(ARCHIVED["M0"])
        center_j = g_values.index(ARCHIVED["G0"])
        ax.add_patch(plt.Rectangle((center_j - 0.5, center_i - 0.5), 1, 1, fill=False, edgecolor=RED, linewidth=1.8))
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#222222")
            spine.set_linewidth(0.8)
        ax.tick_params(direction="out", width=0.8, length=3.2)
        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        # Matplotlib rasterizes colorbar solids by default; keep them vector too.
        colorbar.solids.set_rasterized(False)

    weights = weights.copy()
    weights["display"] = weights.apply(label_weight, axis=1)
    order = ["Archived center", r"$\alpha=0.05$", r"$\alpha=0.2$", r"$\beta=0.1$", r"$\beta=0.4$", r"$\gamma=0.05$", r"$\gamma=0.2$"]
    weights["display"] = pd.Categorical(weights["display"], order, ordered=True)
    weights = weights.sort_values("display")
    x = np.arange(len(weights))
    colours = [PURPLE, BLUE, BLUE, ORANGE, ORANGE, GREEN, GREEN]
    markers = ["^", "o", "o", "D", "D", "s", "s"]
    for index, (_, row) in enumerate(weights.iterrows()):
        axes[2].errorbar(
            x[index],
            row["mean_AUTC_160"],
            yerr=row["sd_AUTC_160"],
            fmt=markers[index],
            capsize=3,
            color=colours[index],
        )
    axes[2].axhline(float(weights.loc[weights["display"] == "Archived center", "mean_AUTC_160"].iloc[0]), color=PURPLE, linestyle="--", linewidth=1.2)
    for separator in (0.5, 2.5, 4.5):
        axes[2].axvline(separator, color="#dddddd", linewidth=0.7)
    axes[2].set_xticks(x, weights["display"].astype(str), rotation=42, ha="right")
    axes[2].set_ylabel("AUTC through 160 queries")
    axes[2].set_title("c  Weight sensitivity", loc="left", fontweight="bold")
    format_axes(axes[2], grid_axis="y")

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png", "svg", "tiff"):
        dpi = 600 if suffix in {"png", "tiff"} else None
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_main_table(summary: pd.DataFrame, path: Path) -> None:
    threshold = summary[summary["stage"] == "threshold"]
    weights = summary[summary["stage"] == "weight"]
    center_threshold = threshold[
        np.isclose(threshold["M0"], ARCHIVED["M0"])
        & np.isclose(threshold["G0"], ARCHIVED["G0"])
    ].iloc[0]
    center_weight = weights[
        np.isclose(weights["alpha"], ARCHIVED["alpha"])
        & np.isclose(weights["beta"], ARCHIVED["beta"])
        & np.isclose(weights["gamma"], ARCHIVED["gamma"])
    ].iloc[0]
    blocks = [
        ("Archived thresholds", center_threshold),
        ("Threshold-grid minimum", threshold.loc[threshold["mean_AUTC_160"].idxmin()]),
        ("Threshold-grid maximum", threshold.loc[threshold["mean_AUTC_160"].idxmax()]),
        ("Archived weights", center_weight),
        ("Weight-sweep minimum", weights.loc[weights["mean_AUTC_160"].idxmin()]),
        ("Weight-sweep maximum", weights.loc[weights["mean_AUTC_160"].idxmax()]),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Development-cohort sensitivity of normalized AUTC and route activity.}",
        r"\label{tab:parameter-sensitivity}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Summary point & $M_0$ & $G_0$ & AUTC$_{160}$ & AUTC$_{640}$ & $R_{80}$ & Correction rounds \\",
        r"\midrule",
    ]
    for name, row in blocks:
        lines.append(
            f"{name} & {float(row['M0']):.2f} & {float(row['G0']):.2f} & "
            f"{float(row['mean_AUTC_160']):.4f} $\\pm$ {float(row['sd_AUTC_160']):.4f} & "
            f"{float(row['mean_AUTC']):.4f} $\\pm$ {float(row['sd_AUTC']):.4f} & "
            f"{float(row['mean_recovery_at_80']):.1f} $\\pm$ {float(row['sd_recovery_at_80']):.1f} & "
            f"{float(row['mean_correction_rounds']):.1f} $\\pm$ {float(row['sd_correction_rounds']):.1f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\tabnote{Values are mean $\pm$ sample SD over development seeds 0--4 with 30 MC-dropout passes. The archived result remains unchanged. The development maximum at $\gamma=0.05$ was frozen prospectively for the separate held-out comparison in Table~\ref{tab:gamma005-holdout}.}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_si_table(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Sweep & Setting & AUTC$_{160}$ & AUTC$_{640}$ & $R_{80}$ & Correction rounds & Unique paths \\",
        r"\midrule",
    ]
    for row in summary.sort_values(["stage", "config_id"]).itertuples(index=False):
        if row.stage == "threshold":
            setting = rf"$M_0={row.M0:.2f},\ G_0={row.G0:.2f}$"
        else:
            setting = rf"$\alpha={row.alpha:.2f},\ \beta={row.beta:.2f},\ \gamma={row.gamma:.2f}$"
        lines.append(
            f"{str(row.stage).capitalize()} & {setting} & "
            f"${row.mean_AUTC_160:.4f} \\pm {row.sd_AUTC_160:.4f}$ & "
            f"${row.mean_AUTC:.4f} \\pm {row.sd_AUTC:.4f}$ & "
            f"${row.mean_recovery_at_80:.1f} \\pm {row.sd_recovery_at_80:.1f}$ & "
            f"${row.mean_correction_rounds:.1f} \\pm {row.sd_correction_rounds:.1f}$ & "
            f"{int(row.unique_sequences)}/5 \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-root", type=Path, required=True)
    parser.add_argument("--weight-root", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--figure-stem", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--si-table", type=Path, required=True)
    args = parser.parse_args()

    threshold = read_stage(args.threshold_root, "threshold")
    weight = read_stage(args.weight_root, "weight")
    per_seed = pd.concat([threshold, weight], ignore_index=True)
    summary = aggregate(per_seed)
    args.source_dir.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(args.source_dir / "v60_parameter_sensitivity_per_seed.csv", index=False)
    summary.to_csv(args.source_dir / "v60_parameter_sensitivity_summary.csv", index=False)
    audit = (
        per_seed.groupby(["stage", "config_id"])
        .agg(seed_count=("seed", "size"), unique_sequences=("sequence_sha256", "nunique"))
        .reset_index()
    )
    audit["independent_sequence_check"] = np.where(audit["unique_sequences"] == audit["seed_count"], "PASS", "REVIEW")
    audit.to_csv(args.source_dir / "v60_parameter_sensitivity_seed_audit.csv", index=False)
    threshold_center = per_seed[
        (per_seed["stage"] == "threshold")
        & np.isclose(per_seed["M0"], ARCHIVED["M0"])
        & np.isclose(per_seed["G0"], ARCHIVED["G0"])
    ]
    weight_center = per_seed[
        (per_seed["stage"] == "weight")
        & np.isclose(per_seed["alpha"], ARCHIVED["alpha"])
        & np.isclose(per_seed["beta"], ARCHIVED["beta"])
        & np.isclose(per_seed["gamma"], ARCHIVED["gamma"])
    ]
    reproducibility = threshold_center.merge(
        weight_center,
        on="seed",
        suffixes=("_threshold", "_weight"),
        validate="one_to_one",
    )
    reproducibility["same_AUTC_160"] = np.isclose(
        reproducibility["AUTC_160_threshold"], reproducibility["AUTC_160_weight"]
    )
    reproducibility["same_AUTC_640"] = np.isclose(
        reproducibility["AUTC_threshold"], reproducibility["AUTC_weight"]
    )
    reproducibility["same_sequence"] = (
        reproducibility["sequence_sha256_threshold"] == reproducibility["sequence_sha256_weight"]
    )
    reproducibility.to_csv(
        args.source_dir / "v60_parameter_sensitivity_reproducibility_audit.csv",
        index=False,
    )
    plot(summary, args.figure_stem)
    write_main_table(summary, args.table)
    write_si_table(summary, args.si_table)

    outputs = [
        args.source_dir / "v60_parameter_sensitivity_per_seed.csv",
        args.source_dir / "v60_parameter_sensitivity_summary.csv",
        args.source_dir / "v60_parameter_sensitivity_seed_audit.csv",
        args.source_dir / "v60_parameter_sensitivity_reproducibility_audit.csv",
        args.figure_stem.with_suffix(".pdf"),
        args.figure_stem.with_suffix(".png"),
        args.figure_stem.with_suffix(".svg"),
        args.figure_stem.with_suffix(".tiff"),
        args.table,
        args.si_table,
    ]
    manifest = pd.DataFrame(
        [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size} for path in outputs]
    )
    manifest.to_csv(args.source_dir / "v60_parameter_sensitivity_output_sha256.csv", index=False)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
