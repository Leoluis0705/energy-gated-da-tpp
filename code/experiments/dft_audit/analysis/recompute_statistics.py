from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import (
    bootstrap_mean_ci,
    candidate_sequence_hash,
    first_attainment_positions,
    left_continuous_autc,
    recovery_at,
    sha256_file,
    write_bytes_protected,
)


@dataclass(frozen=True)
class RunDirectory:
    dataset: str
    method: str
    seed: int
    path: Path


@dataclass(frozen=True)
class RunArtifacts:
    metric: dict
    routing: dict
    trajectory: pd.DataFrame


@dataclass(frozen=True)
class AuditBundle:
    per_seed_metrics: pd.DataFrame
    recovery_matrix: pd.DataFrame
    paired_differences: pd.DataFrame
    routing_statistics: pd.DataFrame
    trajectories: pd.DataFrame
    statistics: dict


def build_round_trajectory(history: pd.DataFrame, batch_size: int, budget: int) -> pd.DataFrame:
    required = {"id", "iteration", "target_label"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"raw history is missing columns: {sorted(missing)}")
    frame = history.iloc[:budget].copy()
    iterations = pd.to_numeric(frame["iteration"], errors="raise").astype(int)
    expected = list(range(1, int(iterations.max()) + 1)) if len(iterations) else []
    actual = sorted(iterations.unique().tolist())
    if actual != expected:
        raise ValueError("history iterations must be contiguous from 1")
    grouped = frame.assign(iteration=iterations).groupby("iteration", sort=True)
    rows: list[dict[str, int]] = []
    cumulative = 0
    for round_index, batch in grouped:
        expected_size = min(batch_size, budget - (int(round_index) - 1) * batch_size)
        if len(batch) != expected_size:
            raise ValueError(f"round {round_index} has {len(batch)} rows; expected {expected_size}")
        labels = pd.to_numeric(batch["target_label"], errors="raise").astype(int)
        if not labels.isin([0, 1]).all():
            raise ValueError("target_label must be binary")
        round_hits = int(labels.sum())
        cumulative += round_hits
        rows.append(
            {
                "round": int(round_index),
                "oracle_evaluations": min(int(round_index) * batch_size, budget),
                "round_target_hits": round_hits,
                "cumulative_target_count": cumulative,
            }
        )
    return pd.DataFrame(rows)


def discover_run_directories(
    roots_by_seed_range: Sequence[tuple[Path, Iterable[int]]],
    datasets: Sequence[str],
    methods: Sequence[str],
) -> list[RunDirectory]:
    records: list[RunDirectory] = []
    for dataset in datasets:
        for method in methods:
            for root, seeds in roots_by_seed_range:
                for seed in seeds:
                    path = Path(root) / "runs" / dataset / method / f"seed_{seed}"
                    for filename in ("run_config.json", "al_history.csv"):
                        required = path / filename
                        if not required.is_file():
                            raise FileNotFoundError(required)
                    records.append(RunDirectory(dataset, method, int(seed), path))
    return records


def _read_required_csv(run_dir: Path, filename: str) -> pd.DataFrame:
    path = run_dir / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def compute_run_artifacts(record: RunDirectory) -> RunArtifacts:
    config_path = record.path / "run_config.json"
    history_path = record.path / "al_history.csv"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    for key, expected in (
        ("name", record.dataset),
        ("method", record.method),
        ("seed", record.seed),
    ):
        if config.get(key) != expected:
            raise ValueError(f"{config_path}: {key}={config.get(key)!r}, expected {expected!r}")
    budget = int(config["budget"])
    batch_size = int(config["batch_size"])
    total_targets = int(config["target_count"])
    history = pd.read_csv(history_path)
    if len(history) != budget:
        raise ValueError(f"{history_path}: expected {budget} rows, found {len(history)}")
    if history["id"].astype(str).duplicated().any():
        raise ValueError(f"{history_path}: duplicate candidate IDs")

    labels = pd.to_numeric(history["target_label"], errors="raise").astype(int)
    trajectory = build_round_trajectory(history, batch_size=batch_size, budget=budget)
    query_counts = trajectory["oracle_evaluations"].to_numpy(dtype=int)
    recoveries = trajectory["cumulative_target_count"].to_numpy(dtype=int)

    mode_trace = _read_required_csv(record.path, "mode_trace.csv")
    diagnostics = _read_required_csv(record.path, "round_diagnostics.csv")
    checkpoints = _read_required_csv(record.path, "checkpoint_manifest.csv")
    predictions = _read_required_csv(record.path, "prediction_manifest.csv")
    rounds = len(trajectory)
    for filename, frame in (
        ("mode_trace.csv", mode_trace),
        ("round_diagnostics.csv", diagnostics),
        ("checkpoint_manifest.csv", checkpoints),
        ("prediction_manifest.csv", predictions),
    ):
        if len(frame) != rounds:
            raise ValueError(f"{record.path / filename}: expected {rounds} rows, found {len(frame)}")

    routes = diagnostics["route"].astype(str)
    trace_routes = mode_trace["mode"].astype(str)
    if routes.tolist() != trace_routes.tolist():
        raise ValueError(f"{record.path}: route mismatch between diagnostics and mode trace")
    allowed_routes = {"threshold_greedy", "diversity_aware"}
    if not set(routes).issubset(allowed_routes):
        raise ValueError(f"{record.path}: unexpected route values {sorted(set(routes) - allowed_routes)}")

    unique_groups = pd.to_numeric(diagnostics["selected_unique_groups"], errors="raise").astype(int)
    replacements = pd.to_numeric(
        diagnostics["correction_replacement_count"], errors="raise"
    ).astype(int)
    correction_gain = pd.to_numeric(diagnostics["correction_target_gain"], errors="raise").astype(int)
    repetition_numerator = int((batch_size - unique_groups).sum())
    repetition_denominator = int(rounds * batch_size)

    candidate_ids = history["id"].astype(str).tolist()
    attainment = first_attainment_positions(labels.tolist())
    trajectory_payload = "".join(
        f"{int(query)}:{int(recovery)}\n"
        for query, recovery in zip(query_counts, recoveries, strict=True)
    )
    final_checkpoint = checkpoints.sort_values("round").iloc[-1]
    checkpoint_training_seeds = pd.to_numeric(checkpoints["training_seed"], errors="raise").astype(int)
    prediction_inference_seeds = pd.to_numeric(predictions["inference_seed"], errors="raise").astype(int)

    metric = {
        "dataset": record.dataset,
        "method": record.method,
        "seed": record.seed,
        "budget": budget,
        "batch_size": batch_size,
        "total_target_count": total_targets,
        "candidate_query_count": len(history),
        "unique_candidate_query_count": int(history["id"].astype(str).nunique()),
        "final_recovery": int(recoveries[-1]),
        "AUTC": left_continuous_autc(query_counts, recoveries, total_targets, budget),
        "candidate_sequence_sha256": candidate_sequence_hash(candidate_ids),
        "round_recovery_trajectory_sha256": candidate_sequence_hash([trajectory_payload]),
        "first_query_by_recovery_count_json": json.dumps(attainment, sort_keys=True, separators=(",", ":")),
        "initial_checkpoint_sha256": str(config.get("checkpoint_sha256", "")),
        "final_checkpoint_sha256": str(final_checkpoint["sha256"]),
        "history_sha256": sha256_file(history_path),
        "run_config_sha256": sha256_file(config_path),
        "checkpoint_manifest_sha256": sha256_file(record.path / "checkpoint_manifest.csv"),
        "prediction_manifest_sha256": sha256_file(record.path / "prediction_manifest.csv"),
        "source_run_dir": str(record.path.resolve()),
    }
    for checkpoint in (80, 160, 240, 320):
        metric[f"recovery_at_{checkpoint}"] = recovery_at(query_counts, recoveries, checkpoint)

    routing = {
        "dataset": record.dataset,
        "method": record.method,
        "seed": record.seed,
        "rounds": rounds,
        "batch_size": batch_size,
        "direct_rounds": int((routes == "threshold_greedy").sum()),
        "correction_rounds": int((routes == "diversity_aware").sum()),
        "effective_replacements": int(replacements.sum()),
        "correction_target_gain": int(correction_gain.sum()),
        "mean_unique_groups_per_batch": float(unique_groups.mean()),
        "repetition_rate": float(repetition_numerator / repetition_denominator),
        "repetition_numerator_repeated_slots": repetition_numerator,
        "repetition_denominator_selected_slots": repetition_denominator,
        "unique_group_aggregation": "arithmetic_mean_of_per_round_unique_group_counts",
        "repetition_aggregation": "sum(batch_size-unique_groups)/sum(batch_size)",
        "training_seed_count": int(len(checkpoint_training_seeds)),
        "training_seed_unique_count": int(checkpoint_training_seeds.nunique()),
        "training_seeds_json": json.dumps(checkpoint_training_seeds.tolist(), separators=(",", ":")),
        "inference_seed_count": int(len(prediction_inference_seeds)),
        "inference_seed_unique_count": int(prediction_inference_seeds.nunique()),
        "inference_seeds_json": json.dumps(prediction_inference_seeds.tolist(), separators=(",", ":")),
        "seed_policy": str(config.get("seed_policy", "")),
        "round_diagnostics_sha256": sha256_file(record.path / "round_diagnostics.csv"),
        "mode_trace_sha256": sha256_file(record.path / "mode_trace.csv"),
        "source_run_dir": str(record.path.resolve()),
    }
    trajectory.insert(0, "seed", record.seed)
    trajectory.insert(0, "method", record.method)
    trajectory.insert(0, "dataset", record.dataset)
    return RunArtifacts(metric=metric, routing=routing, trajectory=trajectory)


def paired_statistics(
    differences: Sequence[float] | np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("paired statistics require at least two one-dimensional differences")
    low, high = bootstrap_mean_ci(values, bootstrap_samples, bootstrap_seed)
    paired_sd = float(values.std(ddof=1))
    all_zero = bool(np.all(values == 0))
    if paired_sd > 0:
        effect = float(values.mean() / paired_sd)
        convention = "standard_mean_over_sample_sd"
    elif all_zero:
        effect = 0.0
        convention = "all_zero_differences_yield_dz_0"
    else:
        effect = None
        convention = "undefined_nonzero_mean_with_zero_sample_sd"

    if all_zero:
        wilcoxon_result = {
            "status": "not_applicable_all_zero",
            "statistic": None,
            "pvalue": None,
            "zero_method": "wilcox",
            "correction": False,
            "alternative": "two-sided",
            "method": "exact",
        }
    else:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )
        wilcoxon_result = {
            "status": "computed",
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "zero_method": "wilcox",
            "correction": False,
            "alternative": "two-sided",
            "method": "exact",
        }
    return {
        "n": int(len(values)),
        "paired_mean": float(values.mean()),
        "paired_median": float(np.median(values)),
        "paired_sd": paired_sd,
        "bootstrap_ci_95_percentile": [low, high],
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(bootstrap_seed),
        "effect_size_dz": effect,
        "effect_size_zero_variance_convention": convention,
        "wins": int((values > 0).sum()),
        "ties": int((values == 0).sum()),
        "losses": int((values < 0).sum()),
        "wilcoxon": wilcoxon_result,
    }


def _sample_summary(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _build_paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    gate = metrics[metrics["method"] == "energy_gated_da_tpp"].copy()
    greedy = metrics[metrics["method"] == "predicted_distance_greedy"].copy()
    joined = gate.merge(
        greedy,
        on=["dataset", "seed"],
        suffixes=("_Gate", "_Greedy"),
        validate="one_to_one",
    )
    paired = joined[["dataset", "seed"]].copy()
    paired["Gate_AUTC"] = joined["AUTC_Gate"]
    paired["Greedy_AUTC"] = joined["AUTC_Greedy"]
    paired["paired_AUTC_difference"] = paired["Gate_AUTC"] - paired["Greedy_AUTC"]
    for checkpoint in (80, 160, 240, 320):
        paired[f"Gate_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_Gate"]
        paired[f"Greedy_recovery_at_{checkpoint}"] = joined[f"recovery_at_{checkpoint}_Greedy"]
        paired[f"paired_recovery_difference_at_{checkpoint}"] = (
            paired[f"Gate_recovery_at_{checkpoint}"] - paired[f"Greedy_recovery_at_{checkpoint}"]
        )
    paired["Gate_candidate_sequence_sha256"] = joined["candidate_sequence_sha256_Gate"]
    paired["Greedy_candidate_sequence_sha256"] = joined["candidate_sequence_sha256_Greedy"]
    return paired.sort_values(["dataset", "seed"]).reset_index(drop=True)


def build_audit_bundle(
    records: Sequence[RunDirectory],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> AuditBundle:
    artifacts = [compute_run_artifacts(record) for record in records]
    metrics = pd.DataFrame([item.metric for item in artifacts]).sort_values(
        ["dataset", "method", "seed"]
    ).reset_index(drop=True)
    expected_keys = {(dataset, method, seed) for dataset in ("limo", "mnoxide") for method in (
        "energy_gated_da_tpp",
        "predicted_distance_greedy",
    ) for seed in range(5, 15)}
    actual_keys = set(metrics[["dataset", "method", "seed"]].itertuples(index=False, name=None))
    if len(records) == 40 and actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"formal run grid mismatch; missing={missing}, extra={extra}")

    routes = pd.DataFrame([item.routing for item in artifacts]).sort_values(
        ["dataset", "method", "seed"]
    ).reset_index(drop=True)
    trajectories = pd.concat([item.trajectory for item in artifacts], ignore_index=True).sort_values(
        ["dataset", "method", "seed", "round"]
    ).reset_index(drop=True)
    recovery_columns = [
        "dataset",
        "method",
        "seed",
        "recovery_at_80",
        "recovery_at_160",
        "recovery_at_240",
        "recovery_at_320",
        "final_recovery",
    ]
    recovery = metrics[recovery_columns].copy()
    paired = _build_paired_differences(metrics)

    dataset_statistics: dict[str, dict] = {}
    for dataset in sorted(metrics["dataset"].unique()):
        dataset_metrics = metrics[metrics["dataset"] == dataset]
        dataset_paired = paired[paired["dataset"] == dataset]
        method_summaries = {}
        for method in ("energy_gated_da_tpp", "predicted_distance_greedy"):
            values = dataset_metrics[dataset_metrics["method"] == method]
            method_summaries[method] = {"AUTC": _sample_summary(values["AUTC"])}
            for checkpoint in (80, 160, 240, 320):
                method_summaries[method][f"recovery_at_{checkpoint}"] = _sample_summary(
                    values[f"recovery_at_{checkpoint}"]
                )
        paired_result = paired_statistics(
            dataset_paired["paired_AUTC_difference"].to_numpy(dtype=float),
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        paired_result["recovery_differences"] = {
            str(checkpoint): _sample_summary(
                dataset_paired[f"paired_recovery_difference_at_{checkpoint}"]
            )
            for checkpoint in (80, 160, 240, 320)
        }
        dataset_statistics[dataset] = {
            "methods": method_summaries,
            "paired": paired_result,
        }

    statistics = {
        "analysis_set": {
            "datasets": ["limo", "mnoxide"],
            "methods": ["energy_gated_da_tpp", "predicted_distance_greedy"],
            "seeds": list(range(5, 15)),
            "run_count": int(len(metrics)),
            "paired_seed_count_per_dataset": 10,
        },
        "AUTC_definition": "left_continuous_round_checkpoint_area/(budget*total_targets)",
        "standard_deviation": "sample_sd_ddof_1",
        "bootstrap": {
            "resamples": int(bootstrap_samples),
            "seed": int(bootstrap_seed),
            "unit": "matched_seed_pair",
            "interval": "percentile_2.5_97.5",
        },
        "datasets": dataset_statistics,
    }
    return AuditBundle(metrics, recovery, paired, routes, trajectories, statistics)


def _computed_reference_values(bundle: AuditBundle) -> dict[str, float]:
    values: dict[str, float] = {}
    method_names = {
        "Gate": "energy_gated_da_tpp",
        "Greedy": "predicted_distance_greedy",
    }
    for dataset in ("limo", "mnoxide"):
        dataset_stats = bundle.statistics["datasets"][dataset]
        for label, method in method_names.items():
            method_stats = dataset_stats["methods"][method]
            values[f"{dataset}.{label}.AUTC.mean"] = method_stats["AUTC"]["mean"]
            values[f"{dataset}.{label}.AUTC.sample_sd"] = method_stats["AUTC"]["sample_sd"]
            for checkpoint in (80, 160, 240, 320):
                recovery_stats = method_stats[f"recovery_at_{checkpoint}"]
                values[f"{dataset}.{label}.recovery_{checkpoint}.mean"] = recovery_stats["mean"]
                values[f"{dataset}.{label}.recovery_{checkpoint}.sample_sd"] = recovery_stats[
                    "sample_sd"
                ]
        paired = dataset_stats["paired"]
        values[f"{dataset}.paired.AUTC.mean"] = paired["paired_mean"]
        values[f"{dataset}.paired.AUTC.ci_low"] = paired["bootstrap_ci_95_percentile"][0]
        values[f"{dataset}.paired.AUTC.ci_high"] = paired["bootstrap_ci_95_percentile"][1]
        for checkpoint in (80, 160, 240, 320):
            values[f"{dataset}.paired.recovery_{checkpoint}.mean"] = paired[
                "recovery_differences"
            ][str(checkpoint)]["mean"]

        dataset_routes = bundle.routing_statistics[bundle.routing_statistics["dataset"] == dataset]
        for label, method in method_names.items():
            method_routes = dataset_routes[dataset_routes["method"] == method]
            for field in (
                "direct_rounds",
                "correction_rounds",
                "effective_replacements",
                "mean_unique_groups_per_batch",
                "repetition_rate",
            ):
                summary = _sample_summary(method_routes[field])
                values[f"{dataset}.{label}.{field}.mean"] = summary["mean"]
                values[f"{dataset}.{label}.{field}.sample_sd"] = summary["sample_sd"]
    return values


def compare_v33_reference(bundle: AuditBundle, reference: dict) -> pd.DataFrame:
    computed = _computed_reference_values(bundle)
    rows: list[dict] = []
    for entry in reference["entries"]:
        key = str(entry["key"])
        if key not in computed:
            raise KeyError(f"no computed value for v33 reference key {key}")
        reported_value = float(entry["reported_value"])
        computed_value = float(computed[key])
        decimals = int(entry["printed_decimals"])
        difference = computed_value - reported_value
        tolerance = 0.5 * (10.0 ** (-decimals))
        if difference == 0:
            status = "exact"
        elif abs(difference) <= tolerance + np.finfo(float).eps:
            status = "matches_reported_rounding"
        else:
            status = "outside_reported_rounding"
        rows.append(
            {
                "table": int(entry["table"]),
                "key": key,
                "label": str(entry["label"]),
                "reported_value": reported_value,
                "computed_value": computed_value,
                "raw_difference_computed_minus_reported": difference,
                "printed_decimals": decimals,
                "rounding_tolerance": tolerance,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "not applicable"
    return f"{float(value):.12g}"


def build_statistical_report(
    bundle: AuditBundle,
    comparison: pd.DataFrame,
    software: dict[str, str],
) -> str:
    bootstrap = bundle.statistics["bootstrap"]
    lines = [
        "# Statistical recalculation report",
        "",
        "## Scope and computation source",
        "",
        "All formal metrics in this report were independently reconstructed from the 40 raw `al_history.csv` files and their corresponding run configuration, routing, checkpoint-manifest, and prediction-manifest records. Existing manuscript result JSON and `run_metrics.csv` files were not used as numerical inputs. They remain comparison evidence only.",
        "",
        f"- analysis set: corrected seeds 5-14; 2 datasets x 2 methods x 10 seeds = {len(bundle.per_seed_metrics)} runs",
        "- AUTC: left-continuous area at completed round checkpoints, normalized by `budget * total_targets`",
        "- standard deviation: sample SD (`ddof=1`)",
        f"- paired bootstrap: {bootstrap['resamples']:,} percentile resamples; bootstrap seed: `{bootstrap['seed']}`; matched seed pair is the resampling unit",
        "- Wilcoxon call: `zero_method=\"wilcox\"`, `correction=False`, `alternative=\"two-sided\"`, `method=\"exact\"`",
        "- paired effect size `dz`: mean paired AUTC difference divided by its sample SD",
        "",
        "## Software used for this recalculation",
        "",
    ]
    for key in ("python", "numpy", "scipy", "pandas"):
        lines.append(f"- {key}: `{software[key]}`")

    lines.extend(
        [
            "",
            "## Independent AUTC results",
            "",
            "| Dataset | Gate mean +/- sample SD | Greedy mean +/- sample SD | Paired mean | 95% percentile CI | dz | Wilcoxon W | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("limo", "mnoxide"):
        stats = bundle.statistics["datasets"][dataset]
        gate = stats["methods"]["energy_gated_da_tpp"]["AUTC"]
        greedy = stats["methods"]["predicted_distance_greedy"]["AUTC"]
        paired = stats["paired"]
        wilcoxon_data = paired["wilcoxon"]
        ci = paired["bootstrap_ci_95_percentile"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    f"{gate['mean']:.12f} +/- {gate['sample_sd']:.12f}",
                    f"{greedy['mean']:.12f} +/- {greedy['sample_sd']:.12f}",
                    f"{paired['paired_mean']:.12f}",
                    f"[{ci[0]:.12f}, {ci[1]:.12f}]",
                    _format_number(paired["effect_size_dz"]),
                    _format_number(wilcoxon_data["statistic"]),
                    _format_number(wilcoxon_data["pvalue"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "For the all-zero Mn-oxide paired vector, `dz=0` is an explicit reporting convention and the signed-rank test is `not_applicable_all_zero`; SciPy is not called on the undefined all-zero vector.",
            "",
            "## Recovery checkpoints",
            "",
            "| Dataset | Checkpoint | Gate mean +/- sample SD | Greedy mean +/- sample SD | Paired mean difference |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("limo", "mnoxide"):
        stats = bundle.statistics["datasets"][dataset]
        for checkpoint in (80, 160, 240, 320):
            gate = stats["methods"]["energy_gated_da_tpp"][f"recovery_at_{checkpoint}"]
            greedy = stats["methods"]["predicted_distance_greedy"][f"recovery_at_{checkpoint}"]
            paired = stats["paired"]["recovery_differences"][str(checkpoint)]
            lines.append(
                f"| {dataset} | {checkpoint} | {gate['mean']:.6f} +/- {gate['sample_sd']:.6f} | "
                f"{greedy['mean']:.6f} +/- {greedy['sample_sd']:.6f} | {paired['mean']:.6f} |"
            )

    status_counts = comparison["status"].value_counts().to_dict()
    lines.extend(
        [
            "",
            "## Reconciliation with v33 Tables 4-6",
            "",
            "The PDF values below are comparison-only. Every transcribed value is listed, including zero raw differences. Status definitions are `exact`, `matches_reported_rounding`, and `outside_reported_rounding`.",
            "",
            f"Status counts: exact={status_counts.get('exact', 0)}, matches_reported_rounding={status_counts.get('matches_reported_rounding', 0)}, outside_reported_rounding={status_counts.get('outside_reported_rounding', 0)}.",
            "",
            "| Table | Metric | Reported | Recomputed | Raw difference | Status |",
            "|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.table} | {row.label} | {row.reported_value:.12g} | {row.computed_value:.12g} | "
            f"{row.raw_difference_computed_minus_reported:.12g} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Output map",
            "",
            "- `results/audit/per_seed_metrics.csv`: independently reconstructed per-run metrics and evidence hashes",
            "- `results/audit/recovery_matrix.csv`: Recovery@80/160/240/320 by dataset, method, and seed",
            "- `results/audit/paired_differences.csv`: paired AUTC and recovery differences",
            "- `results/audit/paired_statistics.json`: full-precision statistics, method parameters, versions, and provenance",
            "- `results/audit/routing_statistics.csv`: route, replacement, group, repetition, and seed-manifest summaries",
            "- `results/audit/seed_variation_details.csv`: round-level recovery trajectories used by the seed audit and figures",
            "- `results/audit/v33_table_comparison.csv`: complete reconciliation table shown above",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _formal_records(archive_root: Path) -> list[RunDirectory]:
    evidence = archive_root / "baseline_snapshot/archive/experiments/reproducibility/results"
    return discover_run_directories(
        [
            (evidence / "paired_two_dataset_confirmation_20260712", range(5, 10)),
            (evidence / "paired_two_dataset_confirmation_seeds_10_14_20260713", range(10, 15)),
        ],
        datasets=("limo", "mnoxide"),
        methods=("energy_gated_da_tpp", "predicted_distance_greedy"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    archive_root = args.archive_root.resolve()
    output_dir = archive_root / "results/audit"
    report_path = archive_root / "docs/STATISTICAL_RECALCULATION_REPORT.md"

    records = _formal_records(archive_root)
    bundle = build_audit_bundle(records, args.bootstrap_samples, args.bootstrap_seed)
    reference_path = archive_root / "analysis/v33_table_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    source_pdf = Path(reference["source"]["path"])
    if not source_pdf.is_file():
        raise FileNotFoundError(f"v33 comparison PDF is missing: {source_pdf}")
    actual_pdf_hash = sha256_file(source_pdf)
    if actual_pdf_hash != reference["source"]["sha256"]:
        raise RuntimeError("v33 comparison PDF hash differs from the frozen reference")
    comparison = compare_v33_reference(bundle, reference)
    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
    statistics = json.loads(json.dumps(bundle.statistics))
    statistics["software"] = software
    statistics["input_provenance"] = {
        "calculation_inputs": [
            "al_history.csv",
            "run_config.json",
            "mode_trace.csv",
            "round_diagnostics.csv",
            "checkpoint_manifest.csv",
            "prediction_manifest.csv",
        ],
        "excluded_as_calculation_inputs": [
            "run_metrics.csv",
            "summary.csv",
            "FINAL_STATISTICAL_QA.json",
            "manuscript tables",
        ],
        "run_count": len(records),
        "source_run_directories": [str(record.path.resolve()) for record in records],
        "evidence_manifest_sha256": sha256_file(archive_root / "docs/evidence_sha256.csv"),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
    }
    statistics["v33_reference"] = {
        **reference["source"],
        "reference_json_sha256": sha256_file(reference_path),
        "comparison_status_counts": {
            str(key): int(value) for key, value in comparison["status"].value_counts().items()
        },
    }
    report = build_statistical_report(bundle, comparison, software)

    outputs = {
        output_dir / "per_seed_metrics.csv": _csv_bytes(bundle.per_seed_metrics),
        output_dir / "recovery_matrix.csv": _csv_bytes(bundle.recovery_matrix),
        output_dir / "paired_differences.csv": _csv_bytes(bundle.paired_differences),
        output_dir / "paired_statistics.json": _json_bytes(statistics),
        output_dir / "routing_statistics.csv": _csv_bytes(bundle.routing_statistics),
        output_dir / "seed_variation_details.csv": _csv_bytes(bundle.trajectories),
        output_dir / "v33_table_comparison.csv": _csv_bytes(comparison),
        report_path: report.encode("utf-8"),
    }
    statuses = {
        str(path.relative_to(archive_root)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
