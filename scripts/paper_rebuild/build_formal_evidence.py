from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


METHODS = {
    "energy_gated_da_tpp": "Energy-Gated DA-TPP",
    "interval_hit_greedy": "Predicted-Target Greedy",
    "always_da_tpp": "Always-DA-TPP",
    "group_only_gate": "Group-only Gate",
    "margin_only_gate": "Margin-only Gate",
}
SEEDS = tuple(range(15, 25))
CHECKPOINTS = (80, 160, 240, 320)
TARGET_COUNT = 78
BATCH_SIZE = 16

COLORS = {
    "Energy-Gated DA-TPP": "#9467bd",
    "Predicted-Target Greedy": "#2ca02c",
    "Always-DA-TPP": "#1f77b4",
    "Group-only Gate": "#7b4ab0",
    "Margin-only Gate": "#1b7f2a",
}
MARKERS = {
    "Energy-Gated DA-TPP": "^",
    "Predicted-Target Greedy": "s",
    "Always-DA-TPP": "o",
    "Group-only Gate": None,
    "Margin-only Gate": None,
}
LINESTYLES = {
    "Energy-Gated DA-TPP": "-",
    "Predicted-Target Greedy": "-",
    "Always-DA-TPP": "-.",
    "Group-only Gate": "--",
    "Margin-only Gate": ":",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=Path(
            r"D:\CGCNN_Formation_Energy_AL_Archive\artifacts\gpu_server"
            r"\completed_formal_results\63729a5a4bea44b3\attempt_1\payload"
            r"\results\final\li_m_o_ablation"
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    return parser.parse_args()


def history_path(formal_root: Path, method: str, seed: int) -> Path:
    return (
        formal_root
        / method
        / f"seed_{seed}"
        / "attempt_1"
        / "al_history.csv"
    )


def autc_at_horizon(targets: np.ndarray, horizon: int) -> float:
    boundaries = np.arange(0, horizon, BATCH_SIZE)
    cumulative = np.concatenate(([0], np.cumsum(targets)))
    return float(
        BATCH_SIZE
        * cumulative[boundaries].sum()
        / (horizon * TARGET_COUNT)
    )


def full_autc(targets: np.ndarray) -> float:
    return autc_at_horizon(targets, len(targets))


def load_histories(formal_root: Path, hidden_scores: pd.DataFrame) -> pd.DataFrame:
    score_map = hidden_scores.set_index("candidate_id")[
        "hidden_p_dft_evaluable"
    ].astype(float)
    rows: list[pd.DataFrame] = []
    for method_key, method_label in METHODS.items():
        for seed in SEEDS:
            path = history_path(formal_root, method_key, seed)
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            if len(frame) != 640:
                raise ValueError(f"{path} has {len(frame)} rows, expected 640")
            target = frame["is_valid"].astype(int)
            if int(target.sum()) != TARGET_COUNT:
                raise ValueError(
                    f"{path} has {int(target.sum())} targets, expected {TARGET_COUNT}"
                )
            candidate_scores = frame["id"].map(score_map)
            missing_targets = frame.loc[
                candidate_scores.isna() & target.astype(bool), "id"
            ].tolist()
            if missing_targets:
                raise ValueError(
                    f"hidden scores missing for target candidates {missing_targets[:5]}"
                )
            candidate_scores = candidate_scores.fillna(0.0)
            out = pd.DataFrame(
                {
                    "method": method_label,
                    "method_key": method_key,
                    "seed": seed,
                    "query": np.arange(1, len(frame) + 1),
                    "candidate_id": frame["id"].astype(str),
                    "group_key": frame["group_key"].astype(str),
                    "target": target,
                    "hidden_p_dft_evaluable": candidate_scores.astype(float),
                }
            )
            out["cumulative_targets"] = out["target"].cumsum()
            out["cumulative_hidden_score"] = (
                out["target"] * out["hidden_p_dft_evaluable"]
            ).cumsum()
            out["cumulative_hidden_hard"] = (
                out["target"]
                * (out["hidden_p_dft_evaluable"] >= 0.50).astype(int)
            ).cumsum()
            rows.append(out)
    return pd.concat(rows, ignore_index=True)


def summarize(histories: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, float | int | str]] = []
    checkpoint_rows: list[dict[str, float | int | str]] = []
    for (method, seed), frame in histories.groupby(["method", "seed"], sort=False):
        frame = frame.sort_values("query")
        targets = frame["target"].to_numpy(dtype=int)
        seed_row: dict[str, float | int | str] = {
            "method": method,
            "seed": int(seed),
            "autc_320": autc_at_horizon(targets, 320),
            "autc_640": full_autc(targets),
        }
        for checkpoint in CHECKPOINTS:
            point = frame.iloc[checkpoint - 1]
            seed_row[f"targets_at_{checkpoint}"] = int(
                point["cumulative_targets"]
            )
            seed_row[f"hidden_score_at_{checkpoint}"] = float(
                point["cumulative_hidden_score"]
            )
            seed_row[f"hidden_hard_at_{checkpoint}"] = int(
                point["cumulative_hidden_hard"]
            )
        seed_rows.append(seed_row)

    per_seed = pd.DataFrame(seed_rows)
    metrics = [
        "autc_320",
        "autc_640",
        *[f"targets_at_{q}" for q in CHECKPOINTS],
        *[f"hidden_score_at_{q}" for q in CHECKPOINTS],
        *[f"hidden_hard_at_{q}" for q in CHECKPOINTS],
    ]
    for method, frame in per_seed.groupby("method", sort=False):
        for metric in metrics:
            checkpoint_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(frame[metric].mean()),
                    "sd": float(frame[metric].std(ddof=1)),
                    "minimum": float(frame[metric].min()),
                    "maximum": float(frame[metric].max()),
                    "n": int(len(frame)),
                }
            )
    return per_seed, pd.DataFrame(checkpoint_rows)


def paired_statistics(
    per_seed: pd.DataFrame, bootstrap_seed: int, resamples: int
) -> dict[str, object]:
    gate = (
        per_seed.loc[
            per_seed["method"] == "Energy-Gated DA-TPP",
            ["seed", "autc_320", "autc_640"],
        ]
        .set_index("seed")
        .sort_index()
    )
    greedy = (
        per_seed.loc[
            per_seed["method"] == "Predicted-Target Greedy",
            ["seed", "autc_320", "autc_640"],
        ]
        .set_index("seed")
        .sort_index()
    )
    if not gate.index.equals(greedy.index):
        raise ValueError("Gate and Greedy seed sets differ")

    result: dict[str, object] = {}
    rng = np.random.default_rng(bootstrap_seed)
    for metric in ("autc_320", "autc_640"):
        differences = (gate[metric] - greedy[metric]).to_numpy(dtype=float)
        indices = rng.integers(
            0, len(differences), size=(resamples, len(differences))
        )
        bootstrap_means = differences[indices].mean(axis=1)
        wilcoxon = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
        sd = float(differences.std(ddof=1))
        result[metric] = {
            "paired_differences": differences.tolist(),
            "mean_difference": float(differences.mean()),
            "sample_sd_difference": sd,
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
            "wilcoxon_statistic": float(wilcoxon.statistic),
            "wilcoxon_two_sided_exact_p": float(wilcoxon.pvalue),
            "paired_effect_size_dz": float(differences.mean() / sd),
            "wins": int((differences > 0).sum()),
            "ties": int((differences == 0).sum()),
            "losses": int((differences < 0).sum()),
        }
    result["bootstrap_seed"] = bootstrap_seed
    result["bootstrap_resamples"] = resamples
    return result


def plot_method_comparison(histories: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7))
    display_methods = [
        "Energy-Gated DA-TPP",
        "Predicted-Target Greedy",
        "Always-DA-TPP",
    ]
    for method in display_methods:
        frame = histories[histories["method"] == method]
        summary = frame.groupby("query")["cumulative_targets"].mean()
        summary = pd.concat(
            [pd.Series({0: 0.0}), summary.loc[summary.index % BATCH_SIZE == 0]]
        )
        ax.plot(
            summary.index,
            summary.values,
            label=method,
            color=COLORS[method],
            linewidth=2,
            marker=MARKERS[method],
            markersize=6,
        )
    perfect_x = np.arange(0, TARGET_COUNT + 1)
    ax.plot(
        perfect_x,
        perfect_x,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label="Perfect efficiency",
    )
    ax.axhline(
        TARGET_COUNT,
        color="#17becf",
        linestyle="--",
        linewidth=2,
        label="Target ceiling",
    )
    ax.set_title("Property: Formation energy   Batch size: 16")
    ax.set_xlabel("Cumulative label queries")
    ax.set_ylabel("Recovered surrogate targets")
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 82)
    ax.set_xticks([0, 80, 160, 240, 320])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.grid(True, alpha=0.3)
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    stem = output_dir / "Figure2_formal_target_recovery"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def plot_gate_greedy_hidden(histories: pd.DataFrame, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    methods = ["Energy-Gated DA-TPP", "Predicted-Target Greedy"]
    metrics = [
        ("cumulative_targets", "Reference targets", "Surrogate targets recovered"),
        (
            "cumulative_hidden_score",
            "Post-selection score",
            "Workflow-completion score",
        ),
    ]
    for panel, (metric, title, ylabel) in zip(axes, metrics):
        for method in methods:
            frame = histories[histories["method"] == method]
            summary = frame.groupby("query")[metric].mean()
            summary = pd.concat(
                [pd.Series({0: 0.0}), summary.loc[summary.index % BATCH_SIZE == 0]]
            )
            panel.plot(
                summary.index,
                summary.values,
                label=method,
                color=COLORS[method],
                linewidth=2,
                marker=MARKERS[method],
                markersize=6,
            )
        if metric == "cumulative_targets":
            panel.axhline(
                TARGET_COUNT,
                color="#666666",
                linestyle="--",
                linewidth=1.3,
                label="Target ceiling",
            )
            panel.set_ylim(0, 82)
            panel.set_yticks([0, 20, 40, 60, 80])
        panel.set_title(title)
        panel.set_xlabel("Cumulative label queries")
        panel.set_ylabel(ylabel)
        panel.set_xlim(0, 320)
        panel.set_xticks([0, 80, 160, 240, 320])
        panel.grid(True, alpha=0.3)
        for spine in panel.spines.values():
            spine.set_linewidth(1)
        panel.legend(loc="lower right")
    fig.suptitle("Property: Formation energy   Batch size: 16")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = output_dir / "Figure3_formal_hidden_audit"
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(
        stem.with_suffix(".tiff"),
        dpi=600,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    source_dir = project_root / "SourceData"
    figure_dir = project_root / "Figures"
    source_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    hidden_scores = pd.read_csv(
        source_dir / "hidden_dft_evaluability_scores.csv"
    )
    histories = load_histories(args.formal_root.resolve(), hidden_scores)
    per_seed, summary = summarize(histories)
    paired = paired_statistics(
        per_seed, args.bootstrap_seed, args.bootstrap_resamples
    )

    histories.to_csv(source_dir / "formal_histories.csv", index=False)
    per_seed.to_csv(source_dir / "formal_per_seed.csv", index=False)
    summary.to_csv(source_dir / "formal_summary.csv", index=False)
    (source_dir / "formal_paired_statistics.json").write_text(
        json.dumps(paired, indent=2),
        encoding="utf-8",
    )

    target_inventory = (
        histories[
            (histories["method"] == "Energy-Gated DA-TPP")
            & (histories["seed"] == SEEDS[0])
        ]
        .groupby("group_key", as_index=False)
        .agg(candidates=("candidate_id", "size"), targets=("target", "sum"))
        .sort_values(["targets", "candidates"], ascending=False)
    )
    target_inventory.to_csv(
        source_dir / "target_composition_inventory.csv", index=False
    )

    plot_method_comparison(histories, figure_dir)
    plot_gate_greedy_hidden(histories, figure_dir)


if __name__ == "__main__":
    main()
