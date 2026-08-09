from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import write_bytes_protected


METHODS = ("energy_gated_da_tpp", "predicted_distance_greedy")
METHOD_LABELS = {
    "energy_gated_da_tpp": "Full Energy-Gated DA-TPP",
    "predicted_distance_greedy": "Interval-Hit Greedy",
}
COLORS = {
    "energy_gated_da_tpp": "#0F4D92",
    "predicted_distance_greedy": "#B64342",
}
LINESTYLES = {
    "energy_gated_da_tpp": "-",
    "predicted_distance_greedy": (0, (5, 2)),
}
MARKERS = {
    "energy_gated_da_tpp": "o",
    "predicted_distance_greedy": "s",
}
DATASET_DISPLAY = {"limo": "Li-M-O", "mnoxide": "Mn-oxide"}
DATASET_BUDGET = {"limo": 640, "mnoxide": 320}
DATASET_TARGETS = {"limo": 78, "mnoxide": 111}
BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260716


def trajectory_bootstrap_summary(
    matrix: np.ndarray,
    samples: int,
    seed: int,
    chunk_size: int = 5_000,
) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("trajectory matrix must be two-dimensional with at least two seeds")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(samples, values.shape[0]))
    boot_means = np.empty((samples, values.shape[1]), dtype=float)
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        boot_means[start:stop] = values[indices[start:stop]].mean(axis=1)
    low, high = np.quantile(boot_means, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "mean_recovery": values.mean(axis=0),
            "sample_sd": values.std(axis=0, ddof=1),
            "seed_min": values.min(axis=0),
            "seed_max": values.max(axis=0),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
        }
    )


def build_figure_source_data(
    trajectories: pd.DataFrame,
    dataset: str,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    required = {
        "dataset",
        "method",
        "seed",
        "round",
        "oracle_evaluations",
        "round_target_hits",
        "cumulative_target_count",
    }
    missing = required.difference(trajectories.columns)
    if missing:
        raise ValueError(f"trajectory source is missing columns: {sorted(missing)}")
    data = trajectories[trajectories["dataset"] == dataset].copy()
    if dataset not in DATASET_BUDGET:
        raise ValueError(f"unsupported dataset: {dataset}")
    if set(data["method"].unique()) != set(METHODS):
        raise ValueError(f"{dataset}: method set does not match formal Gate/Greedy pair")
    zero_rows = []
    for method in METHODS:
        method_data = data[data["method"] == method]
        seeds = sorted(method_data["seed"].astype(int).unique().tolist())
        if seeds != list(range(5, 15)):
            raise ValueError(f"{dataset}/{method}: expected corrected seeds 5-14, found {seeds}")
        for seed in seeds:
            zero_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed": seed,
                    "round": 0,
                    "oracle_evaluations": 0,
                    "round_target_hits": 0,
                    "cumulative_target_count": 0,
                }
            )
    data = pd.concat([pd.DataFrame(zero_rows), data], ignore_index=True)
    data = data.sort_values(["method", "seed", "oracle_evaluations"]).reset_index(drop=True)
    output_frames = []
    for method in METHODS:
        method_data = data[data["method"] == method].copy()
        grids = [
            seed_frame["oracle_evaluations"].astype(int).tolist()
            for _, seed_frame in method_data.groupby("seed", sort=True)
        ]
        if any(grid != grids[0] for grid in grids[1:]):
            raise ValueError(f"{dataset}/{method}: query grids differ across seeds")
        for _, seed_frame in method_data.groupby("seed", sort=True):
            recovery = seed_frame["cumulative_target_count"].to_numpy(dtype=int)
            if np.any(np.diff(recovery) < 0):
                raise ValueError(f"{dataset}/{method}: recovery is not monotone")
        matrix = np.vstack(
            [
                seed_frame["cumulative_target_count"].to_numpy(dtype=float)
                for _, seed_frame in method_data.groupby("seed", sort=True)
            ]
        )
        summary = trajectory_bootstrap_summary(matrix, bootstrap_samples, bootstrap_seed)
        summary["oracle_evaluations"] = grids[0]
        method_data = method_data.merge(summary, on="oracle_evaluations", validate="many_to_one")
        method_data["method_label"] = METHOD_LABELS[method]
        method_data["bootstrap_samples"] = int(bootstrap_samples)
        method_data["bootstrap_seed"] = int(bootstrap_seed)
        method_data["band_definition"] = "pointwise_95_percentile_seed_bootstrap_CI_of_mean"
        method_data["analysis_set"] = "corrected_seeds_5_14"
        output_frames.append(method_data)
    columns = [
        "dataset",
        "method",
        "method_label",
        "seed",
        "round",
        "oracle_evaluations",
        "round_target_hits",
        "cumulative_target_count",
        "mean_recovery",
        "sample_sd",
        "seed_min",
        "seed_max",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_samples",
        "bootstrap_seed",
        "band_definition",
        "analysis_set",
    ]
    return pd.concat(output_frames, ignore_index=True)[columns].sort_values(
        ["method", "seed", "oracle_evaluations"]
    ).reset_index(drop=True)


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "energy-gated-da-tpp-audit-20260716",
        }
    )


def make_figure(source: pd.DataFrame, dataset: str) -> plt.Figure:
    _apply_style()
    width = 170 / 25.4
    height = 105 / 25.4
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    for method in ("predicted_distance_greedy", "energy_gated_da_tpp"):
        method_data = source[source["method"] == method]
        color = COLORS[method]
        linestyle = LINESTYLES[method]
        for _, seed_frame in method_data.groupby("seed", sort=True):
            ax.plot(
                seed_frame["oracle_evaluations"],
                seed_frame["cumulative_target_count"],
                color=color,
                linestyle=linestyle,
                linewidth=0.65,
                alpha=0.12,
                zorder=1,
            )
        aggregate = method_data.drop_duplicates("oracle_evaluations").sort_values(
            "oracle_evaluations"
        )
        x = aggregate["oracle_evaluations"].to_numpy(dtype=float)
        mean = aggregate["mean_recovery"].to_numpy(dtype=float)
        low = aggregate["bootstrap_ci_low"].to_numpy(dtype=float)
        high = aggregate["bootstrap_ci_high"].to_numpy(dtype=float)
        ax.fill_between(x, low, high, color=color, alpha=0.12, linewidth=0, zorder=2)
        marker_every = max(1, len(x) // 10)
        linewidth = 2.6 if method == "predicted_distance_greedy" else 1.7
        ax.plot(
            x,
            mean,
            label=METHOD_LABELS[method],
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=MARKERS[method],
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.8,
            markersize=3.6,
            markevery=marker_every,
            zorder=4 if method == "energy_gated_da_tpp" else 3,
        )
    budget = DATASET_BUDGET[dataset]
    target_count = DATASET_TARGETS[dataset]
    ax.axhline(
        target_count,
        color="#767676",
        linestyle=(0, (2, 2)),
        linewidth=0.9,
        zorder=0,
        label=f"Target ceiling ({target_count})",
    )
    ax.set_xlim(0, budget)
    ax.set_ylim(0, target_count * 1.055)
    ax.set_xticks(np.arange(0, budget + 1, 80))
    ax.set_xlabel("Oracle evaluations")
    ax.set_ylabel("Cumulative target recovery")
    ax.set_title(
        f"Matched-seed recovery: {DATASET_DISPLAY[dataset]}",
        loc="left",
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.0,
        1.005,
        "Corrected seeds 5-14; mean with pointwise 95% seed-bootstrap CI; n=10",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        color="#4D4D4D",
    )
    if dataset == "mnoxide":
        ax.text(
            0.03,
            0.84,
            "Paired Gate and Greedy trajectories coincide\nunder the current element-system group key",
            transform=ax.transAxes,
            fontsize=7.2,
            color="#272727",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CFCECE"},
        )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", handlelength=3.2)
    return fig


def _render_figure_bytes(fig: plt.Figure, fmt: str) -> bytes:
    buffer = io.BytesIO()
    fixed_time = datetime(2026, 7, 16, tzinfo=timezone.utc)
    if fmt == "pdf":
        metadata = {
            "Title": "Matched-seed recovery trajectories",
            "Author": "Energy-Gated DA-TPP audit",
            "Creator": "matplotlib",
            "CreationDate": fixed_time,
            "ModDate": fixed_time,
        }
    elif fmt == "svg":
        metadata = {"Title": "Matched-seed recovery trajectories", "Date": "2026-07-16"}
    else:
        metadata = {"Title": "Matched-seed recovery trajectories", "Software": "matplotlib"}
    fig.savefig(
        buffer,
        format=fmt,
        dpi=600 if fmt == "png" else None,
        bbox_inches="tight",
        facecolor="white",
        metadata=metadata,
    )
    return buffer.getvalue()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--resume-missing", action="store_true")
    args = parser.parse_args(argv)
    if args.check_existing and args.resume_missing:
        parser.error("--check-existing and --resume-missing are mutually exclusive")
    root = args.archive_root.resolve()
    trajectories = pd.read_csv(root / "results/audit/seed_variation_details.csv")
    statuses: dict[str, str] = {}
    for figure_number, dataset in ((3, "limo"), (4, "mnoxide")):
        source = build_figure_source_data(
            trajectories,
            dataset=dataset,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
        source_path = root / f"results/figure_source_data/figure{figure_number}_matched.csv"
        statuses[str(source_path.relative_to(root))] = write_bytes_protected(
            source_path,
            _csv_bytes(source),
            args.check_existing or args.resume_missing,
            create_if_missing_during_check=args.resume_missing,
        )
        fig = make_figure(source, dataset)
        try:
            for fmt in ("pdf", "png", "svg"):
                output = root / f"results/figures/figure{figure_number}_matched.{fmt}"
                statuses[str(output.relative_to(root))] = write_bytes_protected(
                    output,
                    _render_figure_bytes(fig, fmt),
                    args.check_existing or args.resume_missing,
                    create_if_missing_during_check=args.resume_missing,
                )
        finally:
            plt.close(fig)
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
