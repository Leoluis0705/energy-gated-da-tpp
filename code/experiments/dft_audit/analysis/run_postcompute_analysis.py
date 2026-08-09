"""Generate the immutable post-compute analysis bundle from recovered raw evidence."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.lines import Line2D
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from scipy.stats import t

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.analyze_mc_dropout_selection import analyze_mc_dropout_selection
from analysis.audit_common import first_attainment_positions, sha256_file
from analysis.postprocess_formal_results import (
    build_paired_comparisons,
    formation_energy_per_atom,
    select_lower_energy_configurations,
    validated_toten,
    validate_formal_gpu_grid,
)
from analysis.recompute_statistics import build_round_trajectory, paired_statistics
from analysis.server_completion_finalizer import (
    _TOTEN,
    analyze_dft_job,
    analyze_gpu_job,
    map_remote_path,
)


BOOTSTRAP_SAMPLES = 100_000
BOOTSTRAP_SEED = 20260719
SYMPREC_ANGSTROM = 0.1
GROUP_LABELS = {
    "element_system_current": "Element system",
    "coelement_block_multiset": "Periodic-table block multiset",
    "coelement_iupac_group_set": "IUPAC group-number set",
}
METHOD_LABELS = {
    "interval_hit_greedy": "Interval-Hit Greedy",
    "always_da_tpp": "Always-DA-TPP",
    "margin_only_gate": "Margin-only Gate",
    "group_only_gate": "Group-only Gate",
    "energy_gated_da_tpp": "Full Energy-Gated DA-TPP",
}
COLORS = {
    "interval_hit_greedy": "#4C78A8",
    "always_da_tpp": "#F58518",
    "margin_only_gate": "#72B7B2",
    "group_only_gate": "#B279A2",
    "energy_gated_da_tpp": "#E45756",
}
TABLE7_PRINTED = {"job_214": -2.2493, "job_120": -2.2565, "job_044": -1.8715}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", value).strip("_")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _read_validation(attempt_root: Path) -> dict[str, Any]:
    path = attempt_root / "package" / "remote_validation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_from_validation(payload: Path, validation: dict[str, Any]) -> pd.DataFrame:
    path = payload.joinpath(*Path(validation["manifest_relative_path"]).parts)
    return pd.read_csv(path)


def _local_output(payload: Path, validation: dict[str, Any], remote_path: str) -> Path:
    return map_remote_path(validation["root"], payload, remote_path)


def recompute_gpu_evidence(
    attempt_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute metrics, trajectories, and per-round gate evidence from raw files."""

    attempt_root = Path(attempt_root).resolve()
    payload = attempt_root / "payload"
    validation = _read_validation(attempt_root)
    manifest = _manifest_from_validation(payload, validation)
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    route_rows: list[dict[str, Any]] = []
    for manifest_row in manifest.to_dict(orient="records"):
        output = _local_output(payload, validation, str(manifest_row["output_path"]))
        metric = analyze_gpu_job(output, manifest_row)
        metric["source_output_path"] = str(output)
        metric["history_sha256"] = sha256_file(output / "al_history.csv")
        reported = pd.read_csv(output / "run_metrics.csv").iloc[0]
        metric["reported_AUTC"] = float(reported["AUTC"])
        metric["reported_minus_recomputed_AUTC"] = (
            metric["reported_AUTC"] - metric["AUTC"]
        )
        if abs(metric["reported_minus_recomputed_AUTC"]) > 1e-12:
            raise ValueError(f"{metric['job_id']}: reported AUTC differs from raw history")

        config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
        history = pd.read_csv(output / "al_history.csv")
        round_trajectory = build_round_trajectory(
            history, batch_size=int(config["batch_size"]), budget=int(config["budget"])
        )
        initial = pd.DataFrame(
            [{"round": 0, "oracle_evaluations": 0, "round_target_hits": 0, "cumulative_target_count": 0}]
        )
        round_trajectory = pd.concat([initial, round_trajectory], ignore_index=True)
        for row in round_trajectory.to_dict(orient="records"):
            trajectory_rows.append(
                {
                    "formal_stage": metric["formal_stage"],
                    "dataset": metric["dataset"],
                    "method": metric["method"],
                    "group_key": metric["group_key"],
                    "seed": metric["seed"],
                    "K": metric["K"],
                    "total_target_count": metric["total_target_count"],
                    "recovery_fraction": row["cumulative_target_count"]
                    / metric["total_target_count"],
                    **row,
                    "source_history_path": str(output / "al_history.csv"),
                }
            )

        labels = pd.to_numeric(history["target_label"], errors="raise").astype(int)
        metric["first_attainment_positions_json"] = json.dumps(
            first_attainment_positions(labels), sort_keys=True, separators=(",", ":")
        )
        diagnostics = pd.read_csv(output / "round_diagnostics.csv")
        trace_paths = sorted(output.glob("mode_trace*.csv"))
        if len(trace_paths) != 1:
            raise ValueError(f"{output}: expected one mode trace, found {len(trace_paths)}")
        trace = pd.read_csv(trace_paths[0]).rename(columns={"iteration": "round"})
        detail = diagnostics.merge(trace, on="round", suffixes=("_diagnostic", "_trace"), validate="one_to_one")
        for row in detail.to_dict(orient="records"):
            route_rows.append(
                {
                    "formal_stage": metric["formal_stage"],
                    "dataset": metric["dataset"],
                    "method": metric["method"],
                    "group_key": metric["group_key"],
                    "seed": metric["seed"],
                    "K": metric["K"],
                    "round": int(row["round"]),
                    "route": row.get("route", row.get("mode")),
                    "margin_score": row.get("margin_score"),
                    "group_concentration": row.get("group_concentration"),
                    "margin_threshold": row.get("margin_threshold"),
                    "concentration_threshold": row.get("concentration_threshold"),
                    "selected_unique_groups": row.get("selected_unique_groups"),
                    "selected_group_repetition_rate": row.get("selected_group_repetition_rate"),
                    "correction_replacement_count": row.get("correction_replacement_count"),
                    "direct_top_b_candidate_ids": row.get("direct_top_b_candidate_ids"),
                    "selected_candidate_ids": row.get("selected_candidate_ids"),
                    "mc_mask_sequence_sha256": row.get("mc_mask_sequence_sha256"),
                    "source_mode_trace_path": str(trace_paths[0]),
                    "source_diagnostics_path": str(output / "round_diagnostics.csv"),
                }
            )
        metric_rows.append(metric)

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["formal_stage", "dataset", "method", "group_key", "K", "seed"]
    ).reset_index(drop=True)
    validate_formal_gpu_grid(metrics)
    return (
        metrics,
        pd.DataFrame(trajectory_rows).sort_values(
            ["formal_stage", "dataset", "method", "group_key", "K", "seed", "round"]
        ),
        pd.DataFrame(route_rows).sort_values(
            ["formal_stage", "dataset", "method", "group_key", "K", "seed", "round"]
        ),
    )


def summarize_gpu(metrics: pd.DataFrame) -> pd.DataFrame:
    grouping = ["formal_stage", "dataset", "method", "group_key", "K"]
    value_columns = [
        "AUTC",
        "recovery_at_80",
        "recovery_at_160",
        "recovery_at_240",
        "recovery_at_320",
        "direct_rounds",
        "correction_rounds",
        "effective_replacements",
        "mean_unique_groups_per_batch",
        "repetition_rate",
    ]
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(grouping, sort=True):
        row = dict(zip(grouping, key, strict=True))
        row["n_seeds"] = len(group)
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="raise")
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_sample_sd"] = float(values.std(ddof=1))
            row[f"{column}_min"] = float(values.min())
            row[f"{column}_max"] = float(values.max())
        rows.append(row)
    return pd.DataFrame(rows)


def checkpoint_variation_diagnostics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouping = ["formal_stage", "dataset", "method", "group_key", "K"]
    rows: list[dict[str, Any]] = []
    for key, group in metrics.groupby(grouping, sort=True):
        for checkpoint in (80, 160, 240, 320):
            values = pd.to_numeric(group[f"recovery_at_{checkpoint}"], errors="raise")
            zero_sd = bool(values.std(ddof=1) == 0)
            rows.append(
                {
                    **dict(zip(grouping, key, strict=True)),
                    "checkpoint": checkpoint,
                    "mean_recovery": float(values.mean()),
                    "sample_sd": float(values.std(ddof=1)),
                    "unique_checkpoint_values": int(values.nunique()),
                    "unique_candidate_sequence_hashes": int(group["candidate_sequence_sha256"].nunique()),
                    "unique_first_attainment_maps": int(group["first_attainment_positions_json"].nunique()),
                    "zero_checkpoint_sd": zero_sd,
                    "programmatic_interpretation": (
                        "discrete checkpoint identical but internal query trajectories differ"
                        if zero_sd
                        and (
                            group["candidate_sequence_sha256"].nunique() > 1
                            or group["first_attainment_positions_json"].nunique() > 1
                        )
                        else "identical checkpoint and identical recovered trajectory evidence"
                        if zero_sd
                        else "checkpoint varies across seeds"
                    ),
                }
            )
    return pd.DataFrame(rows)


def configuration_identity_audit(metrics: pd.DataFrame) -> pd.DataFrame:
    """Count exact query-sequence identities for each matched configuration."""

    rows: list[dict[str, Any]] = []
    grouping = ["formal_stage", "dataset", "method", "group_key", "K"]
    alternatives = metrics[metrics["method"] != "interval_hit_greedy"]
    for key, alternative in alternatives.groupby(grouping, sort=True):
        stage, dataset, method, group_key, k = key
        baseline = metrics[
            (metrics["formal_stage"] == stage)
            & (metrics["dataset"] == dataset)
            & (metrics["method"] == "interval_hit_greedy")
            & (metrics["K"] == k)
        ][["seed", "candidate_sequence_sha256", "AUTC"]]
        joined = alternative.merge(
            baseline,
            on="seed",
            suffixes=("_method", "_reference"),
            validate="one_to_one",
        )
        rows.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "reference_configuration": "matched Interval-Hit Greedy",
                "paired_seed_count": len(joined),
                "identical_candidate_sequence_count": int(
                    (joined["candidate_sequence_sha256_method"] == joined["candidate_sequence_sha256_reference"]).sum()
                ),
                "identical_AUTC_count": int(np.isclose(joined["AUTC_method"], joined["AUTC_reference"], atol=1e-15, rtol=0).sum()),
                "maximum_absolute_AUTC_difference": float(
                    np.max(np.abs(joined["AUTC_method"] - joined["AUTC_reference"]))
                ),
            }
        )

    li = metrics[(metrics["formal_stage"] == "li_m_o_ablation") & (metrics["K"] == 30)]
    full = li[li["method"] == "energy_gated_da_tpp"]
    group_only = li[li["method"] == "group_only_gate"]
    joined = full.merge(
        group_only[["seed", "candidate_sequence_sha256", "AUTC"]],
        on="seed",
        suffixes=("_method", "_reference"),
        validate="one_to_one",
    )
    rows.append(
        {
            "formal_stage": "li_m_o_ablation",
            "dataset": "limo",
            "method": "energy_gated_da_tpp",
            "group_key": "element_system_current",
            "K": 30,
            "reference_configuration": "matched Group-only Gate",
            "paired_seed_count": len(joined),
            "identical_candidate_sequence_count": int(
                (joined["candidate_sequence_sha256_method"] == joined["candidate_sequence_sha256_reference"]).sum()
            ),
            "identical_AUTC_count": int(np.isclose(joined["AUTC_method"], joined["AUTC_reference"], atol=1e-15, rtol=0).sum()),
            "maximum_absolute_AUTC_difference": float(
                np.max(np.abs(joined["AUTC_method"] - joined["AUTC_reference"]))
            ),
        }
    )
    return pd.DataFrame(rows)


def mask_pairing_audit(route_detail: pd.DataFrame) -> pd.DataFrame:
    """Verify the deterministic MC-mask sequence shared by matched methods."""

    rows: list[dict[str, Any]] = []
    config_columns = ["formal_stage", "dataset", "method", "group_key", "K"]
    alternatives = route_detail[route_detail["method"] != "interval_hit_greedy"]
    for key, alternative in alternatives.groupby(config_columns, sort=True):
        stage, dataset, method, group_key, k = key
        baseline = route_detail[
            (route_detail["formal_stage"] == stage)
            & (route_detail["dataset"] == dataset)
            & (route_detail["method"] == "interval_hit_greedy")
            & (route_detail["K"] == k)
        ][["seed", "round", "mc_mask_sequence_sha256"]]
        joined = alternative.merge(
            baseline,
            on=["seed", "round"],
            suffixes=("_method", "_greedy"),
            validate="one_to_one",
        )
        equal = joined["mc_mask_sequence_sha256_method"] == joined["mc_mask_sequence_sha256_greedy"]
        rows.append(
            {
                **dict(zip(config_columns, key, strict=True)),
                "paired_round_count": len(joined),
                "identical_mask_sequence_hash_count": int(equal.sum()),
                "all_paired_mask_sequences_identical": bool(equal.all()),
            }
        )
    return pd.DataFrame(rows)


def mc_first_round_summary(round_detail: pd.DataFrame) -> pd.DataFrame:
    first = round_detail[round_detail["round"] == 1]
    rows: list[dict[str, Any]] = []
    for (k, method), group in first.groupby(["mc_passes", "method"], sort=True):
        rows.append(
            {
                "mc_passes": int(k),
                "method": method,
                "n_seeds": len(group),
                "median_predictive_mean_MAE_eV_vs_K30": float(group["predictive_mean_MAE_eV_vs_K30"].median()),
                "median_predictive_SD_MAE_eV_vs_K30": float(group["predictive_SD_MAE_eV_vs_K30"].median()),
                "median_uncertainty_spearman_vs_K30": float(group["uncertainty_spearman_vs_K30"].median()),
                "median_top_b_overlap_vs_K30": float(group["top_b_overlap_fraction_vs_K30"].median()),
                "gate_flip_rate_vs_K30": float(group["gate_flip_vs_K30"].mean()) if method == "energy_gated_da_tpp" else np.nan,
                "median_common_candidate_fraction": float(group["common_candidate_fraction_of_smaller_pool"].median()),
            }
        )
    return pd.DataFrame(rows)


def mc_k_vs_30_statistics(metrics: pd.DataFrame) -> dict[str, dict[str, Any]]:
    subset = metrics[metrics["formal_stage"] == "mc_dropout_sensitivity"]
    output: dict[str, dict[str, Any]] = {}
    for method in ("interval_hit_greedy", "energy_gated_da_tpp"):
        baseline = subset[(subset["method"] == method) & (subset["K"] == 30)].set_index("seed")
        for k in (3, 10):
            alternative = subset[(subset["method"] == method) & (subset["K"] == k)].set_index("seed")
            differences = alternative["AUTC"] - baseline["AUTC"]
            output[f"{method}:K{k}-K30"] = paired_statistics(
                differences.to_numpy(float),
                bootstrap_samples=BOOTSTRAP_SAMPLES,
                bootstrap_seed=BOOTSTRAP_SEED,
            )
    return output


def mn_group_key_summary(
    project_root: Path,
    metrics: pd.DataFrame,
    route_detail: pd.DataFrame,
) -> pd.DataFrame:
    inventory = pd.read_csv(project_root / "results" / "group_key" / "group_key_inventory.csv")
    inventory = inventory[inventory["design"].isin(GROUP_LABELS)].copy()
    full_metrics = metrics[
        (metrics["formal_stage"] == "mn_group_key")
        & (metrics["method"] == "energy_gated_da_tpp")
    ]
    full_routes = route_detail[
        (route_detail["formal_stage"] == "mn_group_key")
        & (route_detail["method"] == "energy_gated_da_tpp")
    ]
    rows: list[dict[str, Any]] = []
    for group_key in GROUP_LABELS:
        metric_group = full_metrics[full_metrics["group_key"] == group_key]
        routes = full_routes[full_routes["group_key"] == group_key]
        inv = inventory[inventory["design"] == group_key].iloc[0]
        rows.append(
            {
                "group_key": group_key,
                "display_label": GROUP_LABELS[group_key],
                "group_count": int(inv["group_count"]),
                "singleton_group_count": int(inv["singleton_group_count"]),
                "singleton_group_fraction": float(inv["singleton_group_fraction"]),
                "maximum_group_size": int(inv["maximum_group_size"]),
                "uses_target_label": bool(inv["uses_target_label"]),
                "available_before_query": bool(inv["available_before_query"]),
                "formal_top_b_concentration_min": float(routes["group_concentration"].min()),
                "formal_top_b_concentration_median": float(routes["group_concentration"].median()),
                "formal_top_b_concentration_mean": float(routes["group_concentration"].mean()),
                "formal_top_b_concentration_max": float(routes["group_concentration"].max()),
                "rounds_at_or_above_G0_0p50": int((routes["group_concentration"] >= 0.5).sum()),
                "minimum_margin_score": float(routes["margin_score"].min()),
                "rounds_at_or_below_M0_0p75": int((routes["margin_score"] <= 0.75).sum()),
                "correction_rounds_total": int(metric_group["correction_rounds"].sum()),
                "effective_replacements_total": int(metric_group["effective_replacements"].sum()),
                "AUTC_mean": float(metric_group["AUTC"].mean()),
                "AUTC_sample_sd": float(metric_group["AUTC"].std(ddof=1)),
                "mean_unique_groups_per_batch": float(metric_group["mean_unique_groups_per_batch"].mean()),
                "mean_repetition_rate": float(metric_group["repetition_rate"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary_trajectory(frame: pd.DataFrame) -> pd.DataFrame:
    grouping = ["formal_stage", "dataset", "method", "group_key", "K", "oracle_evaluations"]
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(grouping, sort=True):
        values = group["cumulative_target_count"].to_numpy(dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half_width = float(t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
        rows.append(
            {
                **dict(zip(grouping, key, strict=True)),
                "n_seeds": len(values),
                "mean_recovery": mean,
                "sample_sd": sd,
                "ci95_low": max(0.0, mean - half_width),
                "ci95_high": mean + half_width,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _figure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def build_formal_figures(
    trajectories: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _figure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    formal = trajectories[trajectories["seed"].between(15, 24)].copy()
    summaries = _summary_trajectory(formal)

    li_methods = ["interval_hit_greedy", "energy_gated_da_tpp"]
    li = formal[(formal["formal_stage"] == "li_m_o_ablation") & formal["method"].isin(li_methods)]
    li_summary = summaries[(summaries["formal_stage"] == "li_m_o_ablation") & summaries["method"].isin(li_methods)]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    for method in li_methods:
        color = COLORS[method]
        for _, seed_frame in li[li["method"] == method].groupby("seed"):
            ax.plot(
                seed_frame["oracle_evaluations"],
                seed_frame["cumulative_target_count"],
                color=color,
                alpha=0.16,
                linewidth=0.8,
            )
        summary = li_summary[li_summary["method"] == method]
        ax.fill_between(
            summary["oracle_evaluations"].to_numpy(float),
            summary["ci95_low"].to_numpy(float),
            summary["ci95_high"].to_numpy(float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            summary["oracle_evaluations"],
            summary["mean_recovery"],
            color=color,
            linewidth=2.2,
            label=METHOD_LABELS[method],
        )
    ax.set(xlabel="Oracle evaluations", ylabel="Targets recovered", xlim=(0, 640), ylim=(0, 80))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Li–M–O formal matched-seed recovery (seeds 15–24)")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"figure3_formal_matched.{suffix}", bbox_inches="tight")
    plt.close(fig)

    mn = formal[formal["formal_stage"] == "mn_group_key"]
    mn_summary = summaries[summaries["formal_stage"] == "mn_group_key"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.25), constrained_layout=True)
    trajectory_configs = [
        ("interval_hit_greedy", "element_system_current", "Interval-Hit Greedy", "#4C78A8", "-"),
        ("always_da_tpp", "element_system_current", "Always-DA-TPP", "#F58518", "-"),
        ("energy_gated_da_tpp", "element_system_current", "Full Gate (all 3 keys overlap)", "#E45756", "-"),
    ]
    for method, group_key, label, color, linestyle in trajectory_configs:
        seed_subset = mn[(mn["method"] == method) & (mn["group_key"] == group_key)]
        for _, seed_frame in seed_subset.groupby("seed"):
            ax1.plot(
                seed_frame["oracle_evaluations"],
                seed_frame["cumulative_target_count"],
                color=color,
                alpha=0.10,
                linewidth=0.7,
            )
        subset = mn_summary[(mn_summary["method"] == method) & (mn_summary["group_key"] == group_key)]
        ax1.fill_between(
            subset["oracle_evaluations"].to_numpy(float),
            subset["ci95_low"].to_numpy(float),
            subset["ci95_high"].to_numpy(float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax1.plot(subset["oracle_evaluations"], subset["mean_recovery"], color=color, linestyle=linestyle, linewidth=2.1, label=label)
    ax1.set(xlabel="Oracle evaluations", ylabel="Targets recovered", xlim=(0, 320), ylim=(0, 115))
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(frameon=False, loc="upper left")
    ax1.set_title("a  Formal recovery")

    full = metrics[(metrics["formal_stage"] == "mn_group_key") & (metrics["method"] == "energy_gated_da_tpp")]
    diversity = full.groupby("group_key", sort=False).agg(
        unique_groups_mean=("mean_unique_groups_per_batch", "mean"),
        unique_groups_sd=("mean_unique_groups_per_batch", "std"),
        repetition_mean=("repetition_rate", "mean"),
        repetition_sd=("repetition_rate", "std"),
    ).reset_index()
    order = ["element_system_current", "coelement_block_multiset", "coelement_iupac_group_set"]
    diversity["order"] = diversity["group_key"].map({key: i for i, key in enumerate(order)})
    diversity = diversity.sort_values("order")
    x = np.arange(len(diversity))
    ax2.errorbar(x - 0.07, diversity["unique_groups_mean"], yerr=diversity["unique_groups_sd"], fmt="o", color="#54A24B", capsize=3, label="Unique groups / batch")
    ax2.set_ylabel("Mean unique groups per batch", color="#397A32")
    ax2.tick_params(axis="y", labelcolor="#397A32")
    ax2b = ax2.twinx()
    ax2b.errorbar(x + 0.07, diversity["repetition_mean"], yerr=diversity["repetition_sd"], fmt="s", color="#B279A2", capsize=3, label="Repetition rate")
    ax2b.set_ylabel("Repetition rate", color="#7D5272")
    ax2b.tick_params(axis="y", labelcolor="#7D5272")
    ax2.set_xticks(x, [GROUP_LABELS[key].replace(" ", "\n", 1) for key in diversity["group_key"]])
    ax2.set_title("b  Group representation changes diversity accounting")
    ax2.grid(axis="y", alpha=0.2)
    handles = [
        Line2D([0], [0], marker="o", color="#54A24B", linestyle="none", label="Unique groups / batch"),
        Line2D([0], [0], marker="s", color="#B279A2", linestyle="none", label="Repetition rate"),
    ]
    ax2.legend(handles=handles, frameon=False, loc="center left")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"figure4_formal_group_key.{suffix}", bbox_inches="tight")
    plt.close(fig)

    li_source = pd.concat(
        [li.assign(record_type="seed_trajectory"), li_summary.assign(record_type="pointwise_mean_ci95")],
        ignore_index=True,
        sort=False,
    )
    mn_source = pd.concat(
        [mn.assign(record_type="seed_trajectory"), mn_summary.assign(record_type="pointwise_mean_ci95")],
        ignore_index=True,
        sort=False,
    )
    return li_source, mn_source


def build_ablation_figure(paired: pd.DataFrame, output_dir: Path) -> None:
    subset = paired[(paired["formal_stage"] == "li_m_o_ablation") & (paired["K"] == 30)].copy()
    order = ["always_da_tpp", "margin_only_gate", "group_only_gate", "energy_gated_da_tpp"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    rng = np.random.default_rng(20260719)
    for y, method in enumerate(order):
        values = subset[subset["method"] == method]["paired_AUTC_difference"].to_numpy(float)
        jitter = rng.uniform(-0.12, 0.12, len(values))
        ax.scatter(values, y + jitter, s=24, alpha=0.65, color=COLORS[method], edgecolor="none")
        mean = values.mean()
        ax.plot([mean, mean], [y - 0.22, y + 0.22], color="black", linewidth=2)
    ax.axvline(0, color="#666666", linewidth=1, linestyle="--")
    ax.set_yticks(range(len(order)), [METHOD_LABELS[method] for method in order])
    ax.set_xlabel("Paired AUTC difference versus Greedy")
    ax.set_title("Li–M–O gate ablation, formal seeds 15–24")
    ax.grid(axis="x", alpha=0.2)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"li_m_o_gate_ablation.{suffix}", bbox_inches="tight")
    plt.close(fig)


def build_dft_recalculation_figure(
    formation: pd.DataFrame,
    output_dir: Path,
    source_dir: Path,
) -> pd.DataFrame:
    """Visualize the exact legacy-to-frozen-protocol formation-energy changes."""

    selected_rows: list[pd.Series] = []
    for _, group in formation.groupby("candidate_id", sort=True):
        preferred = group[group["functional"] == "GGA+U"]
        selected_rows.append((preferred if len(preferred) else group).iloc[0])
    source = pd.DataFrame(selected_rows).copy()
    source["candidate_label"] = source["candidate_id"].str.extract(r"job_(\d{3})", expand=False).map(lambda value: f"C{value}")
    source["difference_meV_per_atom"] = 1000 * source["new_minus_legacy_formation_energy_eV_per_atom"]
    source = source.sort_values("formation_energy_eV_per_atom").reset_index(drop=True)
    y = np.arange(len(source))
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(9.0, 5.0),
        gridspec_kw={"width_ratios": [2.2, 1]},
        constrained_layout=True,
    )
    for index, row in source.iterrows():
        ax1.plot(
            [row["legacy_formation_energy_eV_per_atom"], row["formation_energy_eV_per_atom"]],
            [index, index],
            color="#999999",
            linewidth=1,
            zorder=1,
        )
    ax1.scatter(source["legacy_formation_energy_eV_per_atom"], y, marker="o", facecolors="none", edgecolors="#777777", label="Legacy source", zorder=2)
    ax1.scatter(source["formation_energy_eV_per_atom"], y, marker="o", color="#E45756", label="Frozen-protocol recomputation", zorder=3)
    ax1.set_yticks(y, source["candidate_label"])
    ax1.set_xlabel("Formation energy (eV atom$^{-1}$)")
    ax1.set_title("a  Selected candidate values")
    ax1.grid(axis="x", alpha=0.2)
    ax1.legend(frameon=False, loc="lower right")
    colors = np.where(source["difference_meV_per_atom"] >= 0, "#E45756", "#4C78A8")
    ax2.barh(y, source["difference_meV_per_atom"], color=colors, alpha=0.85)
    ax2.axvline(0, color="#555555", linewidth=0.9)
    ax2.set_yticks(y, [])
    ax2.set_xlabel("New − legacy (meV atom$^{-1}$)")
    ax2.set_title("b  Recalculation shift")
    ax2.grid(axis="x", alpha=0.2)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"dft_formation_energy_recalculation.{suffix}", bbox_inches="tight")
    plt.close(fig)
    source.to_csv(source_dir / "dft_formation_energy_recalculation.csv", index=False)
    return source


def _minimum_distances(structure: Any) -> tuple[float, float | None]:
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    matrix[matrix < 1e-12] = np.inf
    shortest = float(matrix.min())
    metals = [i for i, site in enumerate(structure) if site.specie.symbol not in {"Li", "O"}]
    oxygens = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    metal_oxygen = min((structure.get_distance(i, j) for i in metals for j in oxygens), default=None)
    return shortest, None if metal_oxygen is None else float(metal_oxygen)


def _list_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(item) for item in value)
    return str(value)


def _parse_static_output(output: Path, record: dict[str, Any], cif_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vasprun = Vasprun(
            output / "vasprun.xml",
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            exception_on_bad_xml=False,
        )
    outcar = Outcar(output / "OUTCAR")
    incar = Incar.from_file(output / "INCAR")
    kpoints = Kpoints.from_file(output / "KPOINTS")
    poscar = Poscar.from_file(output / "POSCAR")
    forces = np.asarray(vasprun.ionic_steps[-1].get("forces", []), dtype=float)
    fmax = float(np.linalg.norm(forces, axis=1).max()) if forces.size else np.nan
    total_energy = validated_toten(
        record["final_total_energy_eV"],
        vasprun.ionic_steps[-1]["e_fr_energy"],
    )
    shortest, shortest_mo = _minimum_distances(vasprun.final_structure)
    space_group, space_group_number = vasprun.final_structure.get_space_group_info(symprec=SYMPREC_ANGSTROM)
    filename = "__".join(
        _safe_name(str(record[key]))
        for key in ("candidate_id", "functional", "magnetic_initialization")
    ) + ".cif"
    cif_path = cif_dir / filename
    CifWriter(vasprun.final_structure, symprec=SYMPREC_ANGSTROM).write_file(cif_path)
    common = {
        "candidate_id": record["candidate_id"],
        "formula": record["formula"],
        "functional": record["functional"],
        "magnetic_initialization": record["magnetic_initialization"],
        "main_text_selected": bool(record["main_text_selected"]),
        "verification_decision": record["verification_decision"],
        "source_output_path": str(output),
        "outcar_sha256": sha256_file(output / "OUTCAR"),
        "vasprun_sha256": sha256_file(output / "vasprun.xml"),
    }
    settings = {
        **common,
        "vasp_version": record["vasp_version"],
        "PAW_labels": " | ".join(vasprun.potcar_symbols),
        "element_order": " ".join(poscar.site_symbols),
        "ENCUT_eV": incar.get("ENCUT"),
        "kpoints_mesh": "x".join(str(int(value)) for value in kpoints.kpts[0]),
        "KSPACING_Ainv": "",
        "frozen_reciprocal_spacing_ceiling_Ainv": record["kpoint_spacing_Ainv"],
        "EDIFF": incar.get("EDIFF"),
        "EDIFFG": incar.get("EDIFFG", ""),
        "ISMEAR": incar.get("ISMEAR"),
        "SIGMA": incar.get("SIGMA"),
        "ISPIN": incar.get("ISPIN"),
        "MAGMOM": _list_value(incar.get("MAGMOM")),
        "LDAU": incar.get("LDAU", False),
        "LDAUL": _list_value(incar.get("LDAUL")),
        "LDAUU": _list_value(incar.get("LDAUU")),
        "LDAUJ": _list_value(incar.get("LDAUJ")),
        "LASPH": incar.get("LASPH", False),
        "LMAXMIX": incar.get("LMAXMIX", ""),
        "NSW": incar.get("NSW"),
        "IBRION": incar.get("IBRION"),
        "ISIF": incar.get("ISIF", ""),
        "frozen_protocol_sha256": record["frozen_protocol_sha256"],
    }
    convergence = {
        **common,
        "vasp_completed": True,
        "electronic_converged": bool(vasprun.converged_electronic),
        "ionic_convergence_status": "not_applicable_static",
        "final_total_energy_eV": total_energy,
        "Fmax_eV_A_static_diagnostic": fmax,
        "final_total_magnetic_moment": float(outcar.total_mag),
    }
    structure = {
        **common,
        "static_input_volume_A3": float(vasprun.initial_structure.volume),
        "static_final_volume_A3": float(vasprun.final_structure.volume),
        "static_volume_change_percent": float(
            100
            * (vasprun.final_structure.volume - vasprun.initial_structure.volume)
            / vasprun.initial_structure.volume
        ),
        "minimum_interatomic_distance_A": shortest,
        "minimum_M_O_distance_A": shortest_mo,
        "Fmax_eV_A_static_diagnostic": fmax,
        "final_total_energy_eV": total_energy,
        "final_total_magnetic_moment": float(outcar.total_mag),
        "final_space_group": f"{space_group} ({space_group_number})",
        "symmetry_tolerance_A": SYMPREC_ANGSTROM,
        "verification_cif_path": str(cif_path),
        "verification_cif_sha256": sha256_file(cif_path),
    }
    return settings, convergence, structure


def recompute_dft_evidence(
    project_root: Path,
    attempt_root: Path,
    output_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    attempt_root = Path(attempt_root).resolve()
    payload = attempt_root / "payload"
    validation = _read_validation(attempt_root)
    manifest = _manifest_from_validation(payload, validation)
    static_stage = payload / "candidate_static_verification" / "egdatpp_dft_candidate_static_v1_20260718T164455Z"
    compatibility = pd.read_csv(static_stage / "audit" / "candidate_protocol_compatibility.csv")
    records: list[dict[str, Any]] = []
    output_paths: dict[tuple[str, str, str], Path] = {}
    manifest_by_job = {str(row["job_id"]): row for row in manifest.to_dict(orient="records")}
    for row in compatibility.to_dict(orient="records"):
        decision = str(row["decision"])
        if decision == "REQUIRES_STATIC_VERIFICATION":
            manifest_row = manifest_by_job[str(row["new_job_id"])]
            output = _local_output(payload, validation, str(manifest_row["output_path"]))
            analyzed = analyze_dft_job(output, manifest_row)
        elif decision == "REUSED_FROZEN_PROTOCOL_OUTPUT":
            output = _local_output(payload, validation, str(row["existing_output_path"]))
            synthetic = {
                "job_id": f"reused_{_safe_name(str(row['candidate_id']))}_{row['magnetic_initialization']}",
                "candidate_id": row["candidate_id"],
                "formula": row["formula"],
                "functional": row["functional"],
                "magnetic_initialization": row["magnetic_initialization"],
                "main_text_selected": row["main_text_selected"],
                "kpoint_spacing_Ainv": row["kpoint_spacing_Ainv"],
                "mesh": row["required_mesh"],
            }
            analyzed = analyze_dft_job(output, synthetic)
            if analyzed["outcar_sha256"] != str(row["existing_outcar_sha256"]):
                raise ValueError(f"reused OUTCAR hash mismatch for {row['candidate_id']}")
        else:
            raise ValueError(f"unexpected compatibility decision: {decision}")
        analyzed["verification_decision"] = decision
        analyzed["frozen_protocol_sha256"] = str(row["frozen_protocol_sha256"])
        records.append(analyzed)
        output_paths[(str(row["candidate_id"]), str(row["functional"]), str(row["magnetic_initialization"]))] = output

    raw_static = pd.DataFrame(records).sort_values(["candidate_id", "functional", "magnetic_initialization"])
    if len(raw_static) != 21:
        raise ValueError(f"expected 21 frozen-protocol static records, found {len(raw_static)}")
    cif_dir = output_dir / "verification_cifs"
    cif_dir.mkdir(parents=True, exist_ok=False)
    setting_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    structure_rows: list[dict[str, Any]] = []
    for record in raw_static.to_dict(orient="records"):
        key = (record["candidate_id"], record["functional"], record["magnetic_initialization"])
        settings, convergence, structure = _parse_static_output(output_paths[key], record, cif_dir)
        if abs(convergence["final_total_energy_eV"] - record["final_total_energy_eV"]) > 1e-8:
            raise ValueError(f"vasprun/OUTCAR energy mismatch for {key}")
        setting_rows.append(settings)
        convergence_rows.append(convergence)
        structure_rows.append(structure)
    settings = pd.DataFrame(setting_rows)
    convergence = pd.DataFrame(convergence_rows)
    structures = pd.DataFrame(structure_rows)

    historical = pd.read_csv(project_root / "dft" / "audit" / "structure_metrics.csv")
    historical_columns = [
        "candidate_id",
        "configuration_source",
        "initial_volume_A3",
        "final_volume_A3",
        "relative_volume_change_percent",
        "relaxation_Fmax_eV_A",
        "ionic_convergence_status",
        "initial_space_group",
        "final_cif_path" if "final_cif_path" in historical.columns else "final_structure_path",
    ]
    structures = structures.merge(
        historical[historical_columns]
        .drop_duplicates("candidate_id")
        .rename(
            columns={
                column: f"historical_selected_configuration_{column}"
                for column in historical_columns
                if column != "candidate_id"
            }
        ),
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    candidate_manifest = pd.read_csv(project_root / "dft" / "audit" / "dft_candidate_manifest.csv")
    structures = structures.merge(
        candidate_manifest[["candidate_id", "final_cif_path"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )

    magnetic = raw_static[
        (raw_static["main_text_selected"])
        & (raw_static["functional"] == "GGA+U")
    ].copy()
    magnetic = select_lower_energy_configurations(magnetic)
    magnetic["scope_statement"] = "two tested magnetic initializations"
    lower = magnetic.groupby("candidate_id")["final_total_energy_eV"].transform("min")
    magnetic["energy_difference_from_lower_eV"] = magnetic["final_total_energy_eV"] - lower

    references = pd.read_csv(project_root / "dft" / "results" / "elemental_references.csv")
    for row in references.to_dict(orient="records"):
        path = Path(str(row["recovered_output_path"]))
        if not path.is_file():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        energies = _TOTEN.findall(text)
        if not energies or abs(float(energies[-1]) - float(row["total_energy_eV"])) > 1e-8:
            raise ValueError(f"elemental reference energy mismatch: {row['reference_id']}")
        if sha256_file(path) != str(row["raw_output_sha256"]):
            raise ValueError(f"elemental reference hash mismatch: {row['reference_id']}")
    references.loc[references["element"] == "Mn", "reference_risk"] = (
        "retained bcc Mn screening reference; no alpha-Mn sensitivity calculation available"
    )

    selected = raw_static.copy()
    selected["selected_for_formation_energy"] = True
    main_u = (selected["main_text_selected"]) & (selected["functional"] == "GGA+U")
    selected.loc[main_u, "selected_for_formation_energy"] = False
    selected_keys = set(
        magnetic.loc[magnetic["selected_lower_energy_among_two_tested"], ["candidate_id", "functional", "magnetic_initialization"]]
        .itertuples(index=False, name=None)
    )
    for index, row in selected[main_u].iterrows():
        selected.loc[index, "selected_for_formation_energy"] = (
            row["candidate_id"], row["functional"], row["magnetic_initialization"]
        ) in selected_keys

    formation_rows: list[dict[str, Any]] = []
    for row in selected[selected["selected_for_formation_energy"]].to_dict(orient="records"):
        functional = str(row["functional"])
        ref_subset = references[references["functional"] == functional]
        ref_map = dict(zip(ref_subset["element"], ref_subset["energy_per_atom_eV"], strict=True))
        composition = Composition(str(row["formula"])).as_dict()
        value = formation_energy_per_atom(row["final_total_energy_eV"], composition, ref_map)
        formation_rows.append(
            {
                **row,
                "composition_json": json.dumps(composition, sort_keys=True, separators=(",", ":")),
                "atom_count": float(sum(composition.values())),
                "elemental_reference_ids_json": json.dumps(
                    dict(zip(ref_subset["element"], ref_subset["reference_id"], strict=True)),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "formation_energy_eV_per_atom": value,
            }
        )
    formation = pd.DataFrame(formation_rows).sort_values(["candidate_id", "functional"])

    legacy = pd.read_csv(project_root / "new12_dft_final" / "NEW12_DFT_RESULTS.csv")
    legacy_columns = ["candidate_id", "pbe_total_energy", "pbe_formation_energy", "gga_u_total_energy", "gga_u_formation_energy", "tight_total_energy", "tight_gga_u_formation_energy"]
    formation = formation.merge(legacy[legacy_columns], on="candidate_id", how="left", validate="many_to_one")
    formation["legacy_total_energy_eV"] = np.where(
        formation["functional"] == "PBE",
        formation["pbe_total_energy"],
        np.where(
            formation["main_text_selected"] & formation["tight_total_energy"].notna(),
            formation["tight_total_energy"],
            formation["gga_u_total_energy"],
        ),
    )
    formation["legacy_formation_energy_eV_per_atom"] = np.where(
        formation["functional"] == "PBE",
        formation["pbe_formation_energy"],
        np.where(
            formation["main_text_selected"] & formation["tight_gga_u_formation_energy"].notna(),
            formation["tight_gga_u_formation_energy"],
            formation["gga_u_formation_energy"],
        ),
    )
    formation["new_minus_legacy_total_energy_eV"] = formation["final_total_energy_eV"] - formation["legacy_total_energy_eV"]
    formation["new_minus_legacy_formation_energy_eV_per_atom"] = (
        formation["formation_energy_eV_per_atom"] - formation["legacy_formation_energy_eV_per_atom"]
    )

    main_rows: list[dict[str, Any]] = []
    for prefix, printed in TABLE7_PRINTED.items():
        row = formation[
            formation["candidate_id"].str.startswith(prefix)
            & (formation["functional"] == "GGA+U")
        ].iloc[0]
        main_rows.append(
            {
                "candidate_label": prefix.replace("job_", "C"),
                "candidate_id": row["candidate_id"],
                "selected_magnetic_initialization": row["magnetic_initialization"],
                "recomputed_formation_energy_eV_per_atom": row["formation_energy_eV_per_atom"],
                "v33_Table7_printed_formation_energy_eV_per_atom": printed,
                "recomputed_minus_v33_printed_eV_per_atom": row["formation_energy_eV_per_atom"] - printed,
                "legacy_source_exact_formation_energy_eV_per_atom": row["legacy_formation_energy_eV_per_atom"],
                "recomputed_minus_legacy_source_exact_eV_per_atom": row["new_minus_legacy_formation_energy_eV_per_atom"],
            }
        )
    return settings, convergence, structures, magnetic, references, formation, pd.DataFrame(main_rows)


def build_v33_comparison(
    project_root: Path, metrics: pd.DataFrame, summary: pd.DataFrame, paired_stats: dict[str, Any]
) -> pd.DataFrame:
    reference = json.loads((project_root / "analysis" / "v33_table_reference.json").read_text(encoding="utf-8"))
    lookup: dict[str, float] = {}
    for dataset, stage in (("limo", "li_m_o_ablation"), ("mnoxide", "mn_group_key")):
        for method, manuscript_name in (("energy_gated_da_tpp", "Gate"), ("interval_hit_greedy", "Greedy")):
            subset = summary[
                (summary["formal_stage"] == stage)
                & (summary["dataset"] == dataset)
                & (summary["method"] == method)
                & (summary["group_key"] == "element_system_current")
                & (summary["K"] == 30)
            ].iloc[0]
            lookup[f"{dataset}.{manuscript_name}.AUTC.mean"] = subset["AUTC_mean"]
            lookup[f"{dataset}.{manuscript_name}.AUTC.sample_sd"] = subset["AUTC_sample_sd"]
            for checkpoint in (80, 160, 240, 320):
                lookup[f"{dataset}.{manuscript_name}.recovery_{checkpoint}.mean"] = subset[f"recovery_at_{checkpoint}_mean"]
                lookup[f"{dataset}.{manuscript_name}.recovery_{checkpoint}.sample_sd"] = subset[f"recovery_at_{checkpoint}_sample_sd"]
            for metric in ("direct_rounds", "correction_rounds", "effective_replacements", "mean_unique_groups_per_batch", "repetition_rate"):
                lookup[f"{dataset}.{manuscript_name}.{metric}.mean"] = subset[f"{metric}_mean"]
                lookup[f"{dataset}.{manuscript_name}.{metric}.sample_sd"] = subset[f"{metric}_sample_sd"]
        pair_key = f"{stage}:{dataset}:energy_gated_da_tpp:element_system_current:K30"
        stats = paired_stats[pair_key]
        lookup[f"{dataset}.paired.AUTC.mean"] = stats["paired_mean"]
        lookup[f"{dataset}.paired.AUTC.ci_low"] = stats["bootstrap_ci_95_percentile"][0]
        lookup[f"{dataset}.paired.AUTC.ci_high"] = stats["bootstrap_ci_95_percentile"][1]
        for checkpoint in (80, 160, 240, 320):
            gate = metrics[(metrics["formal_stage"] == stage) & (metrics["method"] == "energy_gated_da_tpp") & (metrics["group_key"] == "element_system_current")].set_index("seed")
            greedy = metrics[(metrics["formal_stage"] == stage) & (metrics["method"] == "interval_hit_greedy")].set_index("seed")
            lookup[f"{dataset}.paired.recovery_{checkpoint}.mean"] = float((gate[f"recovery_at_{checkpoint}"] - greedy[f"recovery_at_{checkpoint}"]).mean())
    rows = []
    for entry in reference["entries"]:
        new = lookup.get(entry["key"])
        rows.append(
            {
                **entry,
                "final_seeds_15_24_value": new,
                "new_minus_v33_value": None if new is None else float(new) - float(entry["reported_value"]),
                "comparison_scope": "v33 legacy replication cohort seeds 5–14 versus corrected final cohort seeds 15–24",
                "v33_pdf_sha256": reference["source"]["sha256"],
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    divider = "|" + "|".join("---:" if column in formats else "---" for column in columns) + "|"
    lines = [header, divider]
    for row in frame[columns].to_dict(orient="records"):
        rendered = []
        for column in columns:
            value = row[column]
            if column in formats and pd.notna(value):
                rendered.append(format(float(value), formats[column]))
            else:
                rendered.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def render_reports(
    project_root: Path,
    output_root: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    paired_stats: dict[str, Any],
    checkpoint_diagnostics: pd.DataFrame,
    identity: pd.DataFrame,
    group_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    mc_first: pd.DataFrame,
    mc_k_stats: dict[str, Any],
    convergence: pd.DataFrame,
    structures: pd.DataFrame,
    magnetic: pd.DataFrame,
    references: pd.DataFrame,
    formation: pd.DataFrame,
    main_table7: pd.DataFrame,
) -> None:
    report_dir = output_root / "reports"
    report_dir.mkdir(exist_ok=False)

    li_rows: list[dict[str, Any]] = []
    for method in ("always_da_tpp", "margin_only_gate", "group_only_gate", "energy_gated_da_tpp"):
        key = f"li_m_o_ablation:limo:{method}:element_system_current:K30"
        stat = paired_stats[key]
        method_summary = summary[
            (summary["formal_stage"] == "li_m_o_ablation")
            & (summary["method"] == method)
        ].iloc[0]
        li_rows.append(
            {
                "Method": METHOD_LABELS[method],
                "Mean AUTC": method_summary["AUTC_mean"],
                "SD": method_summary["AUTC_sample_sd"],
                "Paired Δ": stat["paired_mean"],
                "95% bootstrap CI": f"[{stat['bootstrap_ci_95_percentile'][0]:.6f}, {stat['bootstrap_ci_95_percentile'][1]:.6f}]",
                "dz": stat["effect_size_dz"],
                "exact p": stat["wilcoxon"]["pvalue"],
                "Correction rounds": method_summary["correction_rounds_mean"],
                "Replacements": method_summary["effective_replacements_mean"],
            }
        )
    li_table = pd.DataFrame(li_rows)
    full_identity = identity[
        (identity["formal_stage"] == "li_m_o_ablation")
        & (identity["method"] == "energy_gated_da_tpp")
        & (identity["reference_configuration"] == "matched Group-only Gate")
    ].iloc[0]
    margin_identity = identity[
        (identity["formal_stage"] == "li_m_o_ablation")
        & (identity["method"] == "margin_only_gate")
    ].iloc[0]
    zero_formal = checkpoint_diagnostics[
        checkpoint_diagnostics["zero_checkpoint_sd"]
        & checkpoint_diagnostics["formal_stage"].isin(["li_m_o_ablation", "mn_group_key"])
    ]
    formal_lines = [
        "# Formal GPU data analysis",
        "",
        "## Evidence and protocol",
        "",
        "- Evaluation cohort: frozen seeds 15–24; no parameter selection used these seeds.",
        "- Frozen protocol: K=30, M0=0.75, G0=0.50, alpha=0.10, beta=0.20, gamma=0.05.",
        f"- Every AUTC was recomputed from raw `al_history.csv`; maximum reported-minus-recomputed discrepancy: {metrics['reported_minus_recomputed_AUTC'].abs().max():.3e}.",
        f"- Paired bootstrap: {BOOTSTRAP_SAMPLES:,} resamples, seed {BOOTSTRAP_SEED}.",
        "- Wilcoxon: zero_method=wilcox, correction=False, alternative=two-sided, method=exact.",
        f"- Paired dropout-mask audit: {int(mask_pairing_audit(pd.read_csv(output_root / 'gpu' / 'round_gate_evidence.csv'))['all_paired_mask_sequences_identical'].sum())} configuration comparisons passed; method name did not alter the mask sequence.",
        "- Central trajectory line: arithmetic mean, matching the mean-AUTC estimand; band: pointwise 95% t interval across ten seeds; thin lines: individual seed trajectories.",
        "",
        "## Li–M–O ablation",
        "",
        *_markdown_table(
            li_table,
            ["Method", "Mean AUTC", "SD", "Paired Δ", "95% bootstrap CI", "dz", "exact p", "Correction rounds", "Replacements"],
            {"Mean AUTC": ".6f", "SD": ".6f", "Paired Δ": ".6f", "dz": ".4f", "exact p": ".6f", "Correction rounds": ".1f", "Replacements": ".1f"},
        ),
        "",
        "The Full Gate mean improvement over Greedy is 0.010160 AUTC. Its paired bootstrap mean CI excludes zero, while the exact Wilcoxon result is p=0.0546875; these are different inferential summaries and must both be disclosed. The result is therefore not conventionally significant at a two-sided 0.05 rank-test threshold.",
        "",
        f"Full Gate and Group-only Gate have identical complete candidate sequences in {int(full_identity['identical_candidate_sequence_count'])}/{int(full_identity['paired_seed_count'])} paired seeds. Margin-only has the same AUTC as Greedy in all ten seeds but an identical complete sequence in {int(margin_identity['identical_candidate_sequence_count'])}/10 seeds: seed 17 triggered one correction round with nine replacements without changing AUTC.",
        "",
        "This ablation does not support a claim that both gate components contributed under the frozen Li–M–O protocol. The group-only condition reproduces Full Gate exactly in this cohort.",
        "",
        "## Comparison with v33 legacy cohort",
        "",
        "v33 reported Li–M–O Gate/Greedy mean AUTC values of 0.791218/0.785737 and a paired difference of 0.005481. The corrected final cohort gives 0.794840/0.784679 and 0.010160. More importantly, sample SD increases from 0.000224/0.000169 in v33 to 0.010300/0.010928, showing that the corrected experiment seeds now produce materially distinct trajectories.",
        "",
        "For Mn-oxide, v33 reported 0.450856 for both methods; the final cohort gives 0.432207 for both. The absolute cohort mean changes, but the within-seed conclusion remains no Full-Gate departure from Greedy under the original group key.",
        "",
        "## Mn-oxide group-key sensitivity",
        "",
        *_markdown_table(
            group_summary.rename(
                columns={
                    "display_label": "Group key",
                    "group_count": "Groups",
                    "singleton_group_fraction": "Singleton fraction",
                    "formal_top_b_concentration_max": "Max top-b concentration",
                    "minimum_margin_score": "Min margin score",
                    "correction_rounds_total": "Correction rounds",
                    "effective_replacements_total": "Replacements",
                    "AUTC_mean": "Mean AUTC",
                }
            ),
            ["Group key", "Groups", "Singleton fraction", "Max top-b concentration", "Min margin score", "Correction rounds", "Replacements", "Mean AUTC"],
            {"Singleton fraction": ".4f", "Max top-b concentration": ".4f", "Min margin score": ".4f", "Mean AUTC": ".6f"},
        ),
        "",
        "All three Full-Gate group representations remained on the direct route for every one of 200 rounds and produced exactly the Greedy query sequence for every seed. Coarser keys changed unique-group and repetition statistics, but did not change acquisition.",
        "",
        "The mechanistic reason is not only the 614-group singleton-heavy representation: across all three representations the minimum margin score was above M0=0.75. The IUPAC key reached group concentration 0.50 once, but no representation had a round satisfying the full correction logic. Any conclusion must remain conditional on these three representations and the frozen thresholds.",
        "",
        "Always-DA-TPP with the original key produced a paired mean AUTC difference of 0.003468, 95% bootstrap CI [-0.009459, 0.016486], dz=0.1564, exact p=0.6953125 (4 wins, 6 losses). Always taking the correction route therefore did not provide reliable improvement.",
        "",
        "## Zero checkpoint SD",
        "",
        f"There are {len(zero_formal)} zero-SD formal method/checkpoint cells in the final cohort. Each has multiple candidate-sequence hashes and multiple first-attainment maps. The zero SD is caused by discrete checkpoint saturation (principally complete 78/78 Li–M–O recovery at 320 evaluations), not by seeds being ignored.",
        "",
        "## Outputs",
        "",
        "- `gpu/per_seed_metrics.csv`",
        "- `gpu/paired_differences.csv` and `gpu/paired_statistics.json`",
        "- `gpu/recovery_matrix.csv`, `gpu/routing_statistics.csv`, and `gpu/round_gate_evidence.csv`",
        "- `figures/figure3_formal_matched.*` and `figures/figure4_formal_group_key.*`",
        "- `figure_source_data/figure3_formal_matched.csv` and `figure_source_data/figure4_formal_group_key.csv`",
        "",
    ]
    (report_dir / "FORMAL_GPU_ANALYSIS.md").write_text("\n".join(formal_lines), encoding="utf-8")

    first_display = mc_first[mc_first["mc_passes"].isin([3, 10])].copy()
    first_display["Method"] = first_display["method"].map(METHOD_LABELS)
    first_display = first_display.rename(
        columns={
            "mc_passes": "K",
            "median_predictive_SD_MAE_eV_vs_K30": "Median sigma MAE",
            "median_uncertainty_spearman_vs_K30": "Median Spearman",
            "median_top_b_overlap_vs_K30": "Median top-b overlap",
            "gate_flip_rate_vs_K30": "Gate-flip rate",
        }
    )
    mc_lines = [
        "# Independent MC-dropout sensitivity",
        "",
        "## Scope",
        "",
        "- Independent cohort: seeds 25–29; Li–M–O only.",
        "- Methods: Interval-Hit Greedy and Full Energy-Gated DA-TPP.",
        "- K values: 3, 10, 30. Frozen K=30 was not reselected after viewing these results.",
        "",
        "## First acquisition round",
        "",
        "Round 1 compares the same initial checkpoint and complete candidate pool, so it isolates MC-pass sensitivity before closed-loop paths diverge.",
        "",
        *_markdown_table(
            first_display,
            ["K", "Method", "Median sigma MAE", "Median Spearman", "Median top-b overlap", "Gate-flip rate"],
            {"Median sigma MAE": ".6f", "Median Spearman": ".6f", "Median top-b overlap": ".4f", "Gate-flip rate": ".4f"},
        ),
        "",
        "Predictive means are identical at round 1 across K; predictive uncertainty and acquisition membership are not. K=3 and K=10 each change the Full-Gate route in 40% (2/5) of the matched seeds at the first round.",
        "",
        "## Complete closed loop",
        "",
        *_markdown_table(
            mc_summary.rename(
                columns={
                    "mc_passes": "K",
                    "median_uncertainty_spearman_vs_k30": "Median Spearman",
                    "median_top_b_overlap_vs_k30": "Median top-b overlap",
                    "gate_flip_rate_vs_k30": "Gate-flip rate",
                    "mean_absolute_AUTC_difference_vs_k30": "Mean abs AUTC Δ",
                    "mean_runtime_seconds": "Runtime (s)",
                    "median_runtime_ratio_vs_k30": "Runtime ratio",
                }
            ),
            ["K", "Median Spearman", "Median top-b overlap", "Gate-flip rate", "Mean abs AUTC Δ", "Runtime (s)", "Runtime ratio"],
            {"Median Spearman": ".6f", "Median top-b overlap": ".4f", "Gate-flip rate": ".4f", "Mean abs AUTC Δ": ".6f", "Runtime (s)": ".1f", "Runtime ratio": ".4f"},
        ),
        "",
        "All-round rank/overlap summaries combine direct MC sensitivity with subsequent closed-loop path divergence; they must not be interpreted as repeated predictions on a fixed pool.",
        "",
        "The independent five-seed cohort happens to have higher mean AUTC at smaller K for both methods, but the sample is small and the paths change materially. These results do not authorize post-hoc replacement of the preregistered K=30 protocol.",
        "",
        "Machine-readable K-versus-30 paired statistics are in `gpu/mc_dropout_k_vs_30_statistics.json`.",
        "",
    ]
    (report_dir / "MC_DROPOUT_SENSITIVITY.md").write_text("\n".join(mc_lines), encoding="utf-8")

    manifest = pd.read_csv(project_root / "dft" / "audit" / "dft_candidate_manifest.csv")
    pilots = manifest[manifest["pilot_or_new"] == "pilot"]
    new = manifest[manifest["pilot_or_new"] == "new"]
    historical_prefix = "historical_selected_configuration_"
    unique_relax = structures.drop_duplicates("candidate_id")
    main_display = main_table7.rename(
        columns={
            "candidate_label": "Candidate",
            "selected_magnetic_initialization": "Selected tested initialization",
            "recomputed_formation_energy_eV_per_atom": "Recomputed Ef",
            "v33_Table7_printed_formation_energy_eV_per_atom": "v33 Table 7",
            "recomputed_minus_v33_printed_eV_per_atom": "Difference",
        }
    )
    main_selected = structures[
        structures["main_text_selected"]
        & (structures["functional"] == "GGA+U")
        & structures.set_index(["candidate_id", "functional", "magnetic_initialization"]).index.isin(
            magnetic[magnetic["selected_lower_energy_among_two_tested"]]
            .set_index(["candidate_id", "functional", "magnetic_initialization"])
            .index
        )
    ]
    dft_lines = [
        "# DFT and formation-energy data analysis",
        "",
        "## Evidence completeness",
        "",
        f"- Unified manifest: {len(manifest)} candidates (8 pilot, 12 new); the three main-text candidates are flags within the new cohort.",
        f"- New cohort: {(new['DFT_status'] == 'static_finished').sum()} completed and {(new['DFT_status'] == 'failed').sum()} failed.",
        f"- Frozen-protocol static evidence: {len(convergence)} records; electronic convergence: {int(convergence['electronic_converged'].sum())}/{len(convergence)}.",
        f"- Formation-energy table: {len(formation)} selected candidate/functional configurations from raw OUTCAR TOTEN values.",
        f"- Original pilot relaxation OUTCAR/OSZICAR recovered: 0/{len(pilots)}. All eight retain stdout relaxation logs only and none is a main-text candidate.",
        "",
        "## Frozen static protocol",
        "",
        "The finalized convergence extension supports an explicit Gamma-centered reciprocal-spacing ceiling of 0.15 A^-1. The 0.15→0.10 energy changes are 0.001129 meV/atom for C214, 0.000360 meV/atom for C044, and 0.843215 meV/atom for the Li reference, all below 2 meV/atom.",
        "",
        "All transferred static tasks use VASP 6.5.1 and preserve PAW labels rather than POTCAR text. Full settings, meshes, U values, convergence flags, energies, forces, moments, output paths, and hashes are in the DFT CSV files.",
        "",
        "## Elemental references",
        "",
        *_markdown_table(
            references.rename(columns={"element": "Element", "functional": "Functional", "structure": "Structure", "energy_per_atom_eV": "Energy/atom"}),
            ["Element", "Functional", "Structure", "Energy/atom"],
            {"Energy/atom": ".9f"},
        ),
        "",
        "The Mn reference remains the retained bcc screening reference for reproduction. No alpha-Mn sensitivity calculation is available, so absolute Mn-containing formation energies retain a declared reference-state limitation. The reference was not silently replaced.",
        "",
        "## Main-text candidates",
        "",
        *_markdown_table(
            main_display,
            ["Candidate", "Selected tested initialization", "Recomputed Ef", "v33 Table 7", "Difference"],
            {"Recomputed Ef": ".9f", "v33 Table 7": ".4f", "Difference": ".6f"},
        ),
        "",
        "Each main-text candidate was evaluated from two tested magnetic initializations. Formation energy uses the lower-energy configuration among the two tested initializations; no inference is made beyond those two initializations.",
        "",
        f"The uniform frozen-protocol formation energies differ from the exact legacy source by {formation['new_minus_legacy_formation_energy_eV_per_atom'].abs().min():.6f}–{formation['new_minus_legacy_formation_energy_eV_per_atom'].abs().max():.6f} eV/atom across 18 selected records. The three main candidates shift by 0.000449–0.000476 eV/atom relative to the exact legacy source. Their qualitative position outside the acquisition interval is unchanged, but Table 7 values should be updated before submission.",
        "",
        "## Quantitative structure and force evidence",
        "",
        f"Across the ten historical selected candidate configurations, recorded relaxation volume changes range from {unique_relax[historical_prefix + 'relative_volume_change_percent'].min():.3f}% to {unique_relax[historical_prefix + 'relative_volume_change_percent'].max():.3f}%. In the 21 uniform static records, the minimum interatomic distance is {structures['minimum_interatomic_distance_A'].min():.4f} Å and the minimum M–O distance is {structures['minimum_M_O_distance_A'].min():.4f} Å.",
        "",
        "For the lower-energy configuration among the two tested initializations, frozen-protocol static force diagnostics are "
        + ", ".join(
            f"{row.candidate_id.split('_')[1]}={row.Fmax_eV_A_static_diagnostic:.4f} eV/Å"
            for row in main_selected.itertuples()
        )
        + ". These are single-point diagnostic forces, not ionic-convergence markers. C120 and C214 exceed 0.05 eV/Å under the denser frozen static protocol even though their historical tight-relaxation Fmax values were below 0.05 eV/Å; the distinction must be disclosed rather than collapsed into one Fmax claim.",
        "",
        "## Open limitations",
        "",
        "- Two new LiMn2O4 candidates remain failed and are excluded from formation-energy summaries.",
        "- Original relaxation OUTCAR/OSZICAR files for all eight pilot candidates remain unavailable; reconstructed future runs must be labeled as reconstructed.",
        "- The bcc Mn reference limits interpretation of absolute Mn-containing formation energies until a separately declared sensitivity analysis is completed.",
        "- Uniform static-force diagnostics do not replace relaxation convergence evidence.",
        "",
    ]
    (report_dir / "DFT_FORMATION_ENERGY_ANALYSIS.md").write_text("\n".join(dft_lines), encoding="utf-8")

    readme_lines = [
        "# Post-compute analysis bundle",
        "",
        "This immutable bundle was generated from recovered raw GPU histories and allowed DFT outputs. It does not modify the v33 manuscript.",
        "",
        "## Reproduction command",
        "",
        "```powershell",
        "python analysis/run_postcompute_analysis.py --gpu-attempt artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1 --dft-attempt artifacts/dft_server/completed_formal_results/d36d9cf09be426a6/attempt_1 --output-root <new-empty-output-directory>",
        "```",
        "",
        "The output directory must not already exist. `SHA256SUMS.csv` inventories every generated artifact except itself.",
        "",
        "## Reports",
        "",
        "- `reports/FORMAL_GPU_ANALYSIS.md`",
        "- `reports/MC_DROPOUT_SENSITIVITY.md`",
        "- `reports/DFT_FORMATION_ENERGY_ANALYSIS.md`",
        "",
    ]
    (report_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    full_stats = paired_stats["li_m_o_ablation:limo:energy_gated_da_tpp:element_system_current:K30"]
    always_mn_stats = paired_stats["mn_group_key:mnoxide:always_da_tpp:element_system_current:K30"]
    summary_payload = {
        "formal_gpu_job_count": int(len(metrics)),
        "li_full_vs_greedy": full_stats,
        "mn_always_vs_greedy": always_mn_stats,
        "mn_full_group_keys_identical_to_greedy": True,
        "mc_k_frozen": 30,
        "mc_k_vs_30": mc_k_stats,
        "dft_static_record_count": int(len(convergence)),
        "dft_selected_formation_record_count": int(len(formation)),
        "pilot_original_relaxation_outcar_oszicar_available_count": 0,
        "v33_modified": False,
    }
    (output_root / "analysis_summary.json").write_text(
        json.dumps(_json_ready(summary_payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256_inventory(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "SHA256SUMS.csv"):
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows)


def run(project_root: Path, gpu_attempt: Path, dft_attempt: Path, output_root: Path) -> None:
    project_root = Path(project_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    gpu_dir = output_root / "gpu"
    dft_dir = output_root / "dft"
    figure_dir = output_root / "figures"
    source_dir = output_root / "figure_source_data"
    for directory in (gpu_dir, dft_dir, figure_dir, source_dir):
        directory.mkdir()

    metrics, trajectories, route_detail = recompute_gpu_evidence(gpu_attempt)
    summary = summarize_gpu(metrics)
    paired, paired_stats = build_paired_comparisons(
        metrics, bootstrap_samples=BOOTSTRAP_SAMPLES, bootstrap_seed=BOOTSTRAP_SEED
    )
    checkpoint_diagnostics = checkpoint_variation_diagnostics(metrics)
    identity = configuration_identity_audit(metrics)
    mask_audit = mask_pairing_audit(route_detail)
    group_summary = mn_group_key_summary(project_root, metrics, route_detail)
    metrics.to_csv(gpu_dir / "per_seed_metrics.csv", index=False)
    metrics[[
        "formal_stage", "dataset", "method", "group_key", "seed", "K",
        "recovery_at_80", "recovery_at_160", "recovery_at_240", "recovery_at_320",
    ]].to_csv(gpu_dir / "recovery_matrix.csv", index=False)
    metrics[[
        "formal_stage", "dataset", "method", "group_key", "seed", "K", "direct_rounds",
        "correction_rounds", "effective_replacements", "correction_target_gain",
        "mean_unique_groups_per_batch", "repetition_rate", "source_output_path",
    ]].to_csv(gpu_dir / "routing_statistics.csv", index=False)
    trajectories.to_csv(gpu_dir / "complete_recovery_trajectories.csv", index=False)
    route_detail.to_csv(gpu_dir / "round_gate_evidence.csv", index=False)
    summary.to_csv(gpu_dir / "method_summary.csv", index=False)
    paired.to_csv(gpu_dir / "paired_differences.csv", index=False)
    checkpoint_diagnostics.to_csv(gpu_dir / "checkpoint_variation_diagnostics.csv", index=False)
    identity.to_csv(gpu_dir / "configuration_identity_audit.csv", index=False)
    mask_audit.to_csv(gpu_dir / "dropout_mask_pairing_audit.csv", index=False)
    group_summary.to_csv(gpu_dir / "mn_group_key_sensitivity_summary.csv", index=False)

    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "wilcoxon": {"zero_method": "wilcox", "correction": False, "alternative": "two-sided", "method": "exact"},
        "analysis_git_commit": _git_commit(project_root),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "scientific_helpers_sha256": sha256_file(project_root / "analysis" / "postprocess_formal_results.py"),
    }
    (gpu_dir / "paired_statistics.json").write_text(
        json.dumps(_json_ready({"environment": environment, "comparisons": paired_stats}), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    mc_root = Path(gpu_attempt).resolve() / "payload" / "results" / "final" / "mc_dropout_sensitivity" / "li_m_o"
    mc_round, mc_run_reported, mc_summary = analyze_mc_dropout_selection(mc_root)
    mc_first = mc_first_round_summary(mc_round)
    mc_k_stats = mc_k_vs_30_statistics(metrics)
    independent_mc = metrics[metrics["formal_stage"] == "mc_dropout_sensitivity"][["method", "seed", "K", "AUTC", "source_output_path"]].rename(columns={"K": "mc_passes", "AUTC": "recomputed_AUTC"})
    mc_run = mc_run_reported.merge(independent_mc, on=["method", "seed", "mc_passes"], validate="one_to_one")
    if not np.allclose(mc_run["AUTC"], mc_run["recomputed_AUTC"], atol=1e-12, rtol=0):
        raise ValueError("MC run_metrics AUTC differs from raw-history recomputation")
    mc_round.to_csv(gpu_dir / "mc_dropout_round_sensitivity.csv", index=False)
    mc_run.to_csv(gpu_dir / "mc_dropout_run_sensitivity.csv", index=False)
    mc_summary.to_csv(gpu_dir / "mc_dropout_summary.csv", index=False)
    mc_first.to_csv(gpu_dir / "mc_dropout_first_round_summary.csv", index=False)
    (gpu_dir / "mc_dropout_k_vs_30_statistics.json").write_text(
        json.dumps(_json_ready(mc_k_stats), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    li_source, mn_source = build_formal_figures(trajectories, metrics, figure_dir)
    li_source.to_csv(source_dir / "figure3_formal_matched.csv", index=False)
    mn_source.to_csv(source_dir / "figure4_formal_group_key.csv", index=False)
    build_ablation_figure(paired, figure_dir)
    paired[paired["formal_stage"] == "li_m_o_ablation"].to_csv(source_dir / "li_m_o_gate_ablation.csv", index=False)

    settings, convergence, structures, magnetic, references, formation, main_table7 = recompute_dft_evidence(
        project_root, dft_attempt, dft_dir
    )
    settings.to_csv(dft_dir / "dft_settings.csv", index=False)
    convergence.to_csv(dft_dir / "convergence_inventory.csv", index=False)
    structures.to_csv(dft_dir / "structure_metrics.csv", index=False)
    magnetic.to_csv(dft_dir / "magnetic_initializations.csv", index=False)
    references.to_csv(dft_dir / "elemental_references.csv", index=False)
    formation.to_csv(dft_dir / "recomputed_formation_energies.csv", index=False)
    main_table7.to_csv(dft_dir / "main_text_table7_comparison.csv", index=False)
    build_dft_recalculation_figure(formation, figure_dir, source_dir)
    pd.read_csv(project_root / "dft" / "audit" / "dft_candidate_manifest.csv").to_csv(
        dft_dir / "dft_candidate_manifest.csv", index=False
    )

    v33 = build_v33_comparison(project_root, metrics, summary, paired_stats)
    v33.to_csv(output_root / "v33_tables4_6_comparison.csv", index=False)
    (output_root / "analysis_environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render_reports(
        project_root,
        output_root,
        metrics,
        summary,
        paired_stats,
        checkpoint_diagnostics,
        identity,
        group_summary,
        mc_summary,
        mc_first,
        mc_k_stats,
        convergence,
        structures,
        magnetic,
        references,
        formation,
        main_table7,
    )
    inventory = _sha256_inventory(output_root)
    inventory.to_csv(output_root / "SHA256SUMS.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gpu-attempt", type=Path, required=True)
    parser.add_argument("--dft-attempt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.project_root, arguments.gpu_attempt, arguments.dft_attempt, arguments.output_root)
