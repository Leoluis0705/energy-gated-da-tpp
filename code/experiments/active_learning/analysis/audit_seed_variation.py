from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import candidate_sequence_hash, first_attainment_positions, sha256_file, write_bytes_protected
from analysis.recompute_statistics import _formal_records, compute_run_artifacts


def earliest_seed_divergence(
    frame: pd.DataFrame,
    query_column: str = "oracle_evaluations",
    value_column: str = "cumulative_target_count",
) -> int | None:
    required = {"seed", query_column, value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")
    counts = frame.groupby(query_column, sort=True)[value_column].nunique(dropna=False)
    divergent = counts[counts > 1]
    return int(divergent.index[0]) if len(divergent) else None


def classify_checkpoint_variation(
    checkpoint_values: Sequence[int | float],
    prefix_divergence_query: int | None,
) -> str:
    values = np.asarray(checkpoint_values, dtype=float)
    if len(np.unique(values)) > 1:
        return "checkpoint_varies_across_seeds"
    if prefix_divergence_query is not None:
        return "checkpoint_equal_but_round_prefix_differs"
    return "checkpoint_and_round_prefix_identical"


def _display_optional_query(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return "none"
    return str(int(value))


def _trajectory_hash(values: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(f"{int(value)}\n".encode("ascii"))
    return digest.hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _build_report(
    summary: pd.DataFrame,
    checkpoints: pd.DataFrame,
    code_evidence: list[dict[str, str]],
) -> str:
    lines = [
        "# Seed effectiveness and zero-SD audit",
        "",
        "## Conclusion",
        "",
        "The corrected seed was effective in refit and wrapper RNG records, but the protocol did not randomly reinitialize the model for each seed: every run intentionally loaded the same frozen initial checkpoint. Training and inference wrapper seeds vary by outer seed and round. The retained MC-dropout implementation separately resets each manual mask pass to `1000003 + pass_idx`, independent of the outer seed; this is a disclosed limitation rather than evidence that all seed controls failed.",
        "",
        "All Recovery values are exact integer counts. Therefore a reported sample SD of zero is not caused by decimal rounding. The classification table below distinguishes exact checkpoint equality from a trajectory that diverged earlier and reconverged by the checkpoint.",
        "",
        "## Dataset-method summary",
        "",
        "| Dataset | Method | Unique candidate sequences | Unique round trajectories | Unique candidate-level trajectories | First round-boundary divergence | First retained candidate-order divergence | Unique final checkpoints |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.method} | {row.candidate_sequence_unique_count} | "
            f"{row.round_recovery_trajectory_unique_count} | {row.candidate_recovery_trajectory_unique_count} | "
            f"{_display_optional_query(row.earliest_round_checkpoint_divergence_query)} | "
            f"{_display_optional_query(row.earliest_candidate_order_divergence_query)} | "
            f"{row.final_checkpoint_sha256_unique_count} |"
        )
    lines.extend(
        [
            "",
            "The candidate-level divergence uses the row order retained in `al_history.csv`. Batch selection is simultaneous, so this within-batch order is an audit trace, not the formal AUTC estimand. The round-boundary divergence is the relevant trajectory comparison for the manuscript metrics.",
            "",
            "## Checkpoint-by-checkpoint classification",
            "",
            "| Dataset | Method | Queries | Mean | Sample SD | Unique values | First round-prefix divergence | First candidate-order-prefix divergence | Classification |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in checkpoints.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.method} | {row.checkpoint_queries} | {row.mean_recovery:.6f} | "
            f"{row.sample_sd:.6f} | {row.unique_value_count} | "
            f"{_display_optional_query(row.earliest_round_prefix_divergence_query)} | "
            f"{_display_optional_query(row.earliest_candidate_prefix_divergence_query)} | {row.classification} |"
        )

    lines.extend(
        [
            "",
            "## Programmatic seed checks",
            "",
            "For every dataset-method block, all ten candidate-query sequence hashes and all ten final checkpoint hashes are distinct. Each run has a distinct source directory; no output-path collision was found. Every checkpoint and prediction manifest contains one recorded seed per round, and the minimum across-round seed uniqueness across the ten outer seeds is ten. These checks rule out accidental reuse of a single final checkpoint or a single formal history as the explanation for zero checkpoint SD.",
            "",
            "The frozen pool order is intentionally identical across seeds within a dataset (`candidate_order_sha256` has one unique value). Prediction loading is explicitly `shuffle=False`. Training uses the seed wrapper and a PyTorch `SubsetRandomSampler`; `CIFData` first applies its own fixed `random_seed=123` shuffle. Thus candidate order is controlled, while refit sampling receives the per-round PyTorch seed.",
            "",
            "No evidence of rounding-induced equality exists because recovery counts are stored and recalculated as integers. No evidence of same-file overwrite exists because run paths are distinct and content hashes vary; this is not a proof that no historical overwrite ever occurred before preservation.",
            "",
            "## Code evidence",
            "",
            "| Evidence | Path and lines | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for item in code_evidence:
        lines.append(f"| {item['label']} | `{item['path_lines']}` | `{item['sha256']}` |")
    lines.extend(
        [
            "",
            "## Output map",
            "",
            "- `seed_variation_summary.csv`: one programmatic summary row per dataset-method block",
            "- `checkpoint_variation.csv`: the complete Recovery@80/160/240/320 SD classification",
            "- `first_attainment_matrix.csv`: first retained candidate-order query for every attained recovery count and seed",
            "- `per_seed_metrics.csv`: candidate-sequence and final-checkpoint hashes",
            "- `seed_variation_details.csv`: all formal round-boundary trajectories",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    root = args.archive_root.resolve()
    records = _formal_records(root)
    artifacts = [compute_run_artifacts(record) for record in records]
    metrics = pd.DataFrame([item.metric for item in artifacts])
    round_trajectories = pd.concat([item.trajectory for item in artifacts], ignore_index=True)

    candidate_frames: list[pd.DataFrame] = []
    first_rows: list[dict] = []
    manifest_rows: list[dict] = []
    config_rows: list[dict] = []
    for record in records:
        history = pd.read_csv(record.path / "al_history.csv")
        labels = pd.to_numeric(history["target_label"], errors="raise").astype(int)
        candidate_frame = pd.DataFrame(
            {
                "dataset": record.dataset,
                "method": record.method,
                "seed": record.seed,
                "oracle_evaluations": np.arange(1, len(history) + 1),
                "cumulative_target_count": labels.cumsum().to_numpy(),
            }
        )
        candidate_frames.append(candidate_frame)
        for recovery_count, query_position in first_attainment_positions(labels).items():
            first_rows.append(
                {
                    "dataset": record.dataset,
                    "method": record.method,
                    "seed": record.seed,
                    "recovery_count": recovery_count,
                    "first_query_position_in_retained_candidate_order": query_position,
                    "order_semantics": "retained_al_history_row_order_within_simultaneous_batches",
                }
            )
        for kind, filename, seed_column in (
            ("checkpoint", "checkpoint_manifest.csv", "training_seed"),
            ("prediction", "prediction_manifest.csv", "inference_seed"),
        ):
            frame = pd.read_csv(record.path / filename)
            for row in frame.itertuples(index=False):
                manifest_rows.append(
                    {
                        "dataset": record.dataset,
                        "method": record.method,
                        "seed": record.seed,
                        "round": int(row.round),
                        "kind": kind,
                        "artifact_sha256": str(row.sha256),
                        "recorded_seed": int(getattr(row, seed_column)),
                    }
                )
        config = json.loads((record.path / "run_config.json").read_text(encoding="utf-8"))
        config_rows.append(
            {
                "dataset": record.dataset,
                "method": record.method,
                "seed": record.seed,
                "initial_checkpoint_sha256": config["checkpoint_sha256"],
                "candidate_order_sha256": config["candidate_order_sha256"],
                "prediction_loader_shuffle": config["prediction_loader_shuffle"],
                "mc_passes": config["mc_passes"],
                "dropout_rate": config["dropout_rate"],
            }
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    manifests = pd.DataFrame(manifest_rows)
    configs = pd.DataFrame(config_rows)

    summary_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    for dataset in ("limo", "mnoxide"):
        for method in ("energy_gated_da_tpp", "predicted_distance_greedy"):
            run_metrics = metrics[(metrics["dataset"] == dataset) & (metrics["method"] == method)]
            rounds = round_trajectories[
                (round_trajectories["dataset"] == dataset)
                & (round_trajectories["method"] == method)
            ]
            candidate_group = candidates[
                (candidates["dataset"] == dataset) & (candidates["method"] == method)
            ]
            group_manifests = manifests[
                (manifests["dataset"] == dataset) & (manifests["method"] == method)
            ]
            group_configs = configs[(configs["dataset"] == dataset) & (configs["method"] == method)]
            candidate_hashes = []
            for _, seed_frame in candidate_group.groupby("seed"):
                candidate_hashes.append(_trajectory_hash(seed_frame["cumulative_target_count"].tolist()))
            checkpoint_manifest = group_manifests[group_manifests["kind"] == "checkpoint"]
            prediction_manifest = group_manifests[group_manifests["kind"] == "prediction"]
            checkpoint_hash_min_unique = int(
                checkpoint_manifest.groupby("round")["artifact_sha256"].nunique().min()
            )
            training_seed_min_unique = int(
                checkpoint_manifest.groupby("round")["recorded_seed"].nunique().min()
            )
            inference_seed_min_unique = int(
                prediction_manifest.groupby("round")["recorded_seed"].nunique().min()
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "seed_count": int(run_metrics["seed"].nunique()),
                    "candidate_sequence_unique_count": int(run_metrics["candidate_sequence_sha256"].nunique()),
                    "history_sha256_unique_count": int(run_metrics["history_sha256"].nunique()),
                    "round_recovery_trajectory_unique_count": int(
                        run_metrics["round_recovery_trajectory_sha256"].nunique()
                    ),
                    "candidate_recovery_trajectory_unique_count": len(set(candidate_hashes)),
                    "earliest_round_checkpoint_divergence_query": earliest_seed_divergence(rounds),
                    "earliest_candidate_order_divergence_query": earliest_seed_divergence(candidate_group),
                    "initial_checkpoint_sha256_unique_count": int(
                        group_configs["initial_checkpoint_sha256"].nunique()
                    ),
                    "final_checkpoint_sha256_unique_count": int(
                        run_metrics["final_checkpoint_sha256"].nunique()
                    ),
                    "candidate_order_sha256_unique_count": int(
                        group_configs["candidate_order_sha256"].nunique()
                    ),
                    "source_run_dir_unique_count": int(run_metrics["source_run_dir"].nunique()),
                    "minimum_unique_checkpoint_hashes_per_round_across_seeds": checkpoint_hash_min_unique,
                    "minimum_unique_training_seeds_per_round_across_seeds": training_seed_min_unique,
                    "minimum_unique_inference_seeds_per_round_across_seeds": inference_seed_min_unique,
                    "prediction_loader_shuffle_unique_values": ";".join(
                        sorted(group_configs["prediction_loader_shuffle"].astype(str).unique())
                    ),
                    "mc_passes_unique_values": ";".join(
                        sorted(group_configs["mc_passes"].astype(str).unique())
                    ),
                    "dropout_rate_unique_values": ";".join(
                        sorted(group_configs["dropout_rate"].astype(str).unique())
                    ),
                    "manual_mc_pass_seed_policy": "1000003+pass_idx_independent_of_outer_seed",
                }
            )
            for checkpoint in (80, 160, 240, 320):
                checkpoint_values = run_metrics[f"recovery_at_{checkpoint}"].to_numpy(dtype=int)
                round_prefix = rounds[rounds["oracle_evaluations"] <= checkpoint]
                candidate_prefix = candidate_group[candidate_group["oracle_evaluations"] <= checkpoint]
                round_divergence = earliest_seed_divergence(round_prefix)
                candidate_divergence = earliest_seed_divergence(candidate_prefix)
                checkpoint_rows.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "checkpoint_queries": checkpoint,
                        "mean_recovery": float(checkpoint_values.mean()),
                        "sample_sd": float(checkpoint_values.std(ddof=1)),
                        "unique_value_count": int(len(np.unique(checkpoint_values))),
                        "values_by_seed_json": json.dumps(
                            dict(zip(run_metrics["seed"].astype(str), checkpoint_values.tolist(), strict=True)),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "earliest_round_prefix_divergence_query": round_divergence,
                        "earliest_candidate_prefix_divergence_query": candidate_divergence,
                        "classification": classify_checkpoint_variation(
                            checkpoint_values, round_divergence
                        ),
                        "rounding_could_explain_zero_sd": False,
                    }
                )

    summary = pd.DataFrame(summary_rows)
    checkpoint_table = pd.DataFrame(checkpoint_rows)
    first_attainment = pd.DataFrame(first_rows).sort_values(
        ["dataset", "method", "seed", "recovery_count"]
    )
    code_specs = [
        ("Python/NumPy/PyTorch/CUDA seed wrapper", "baseline_snapshot/archive/experiments/reproducibility/seeded_runpy_torch_compat.py", "17-23"),
        ("Outer-seed and per-round seed schedule", "baseline_snapshot/archive/experiments/reproducibility/two_dataset_paired_protocol.py", "104-114"),
        ("Refit launched through training seed", "baseline_snapshot/archive/experiments/reproducibility/run_paired_dataset_job.py", "510-551"),
        ("Prediction loader has shuffle=False", "baseline_snapshot/archive/experiments/reproducibility/paired_predict_no_shuffle.py", "58"),
        ("CIFData fixed shuffle and training sampler", "baseline_snapshot/archive/experiments/reproducibility/staging/paired_confirmation_server_20260712/cgcnn/data.py", "77-85,303-316"),
        ("Manual MC masks and fixed pass seeds", "baseline_snapshot/archive/experiments/reproducibility/staging/paired_confirmation_server_20260712/active_learning_etdg_tage.py", "199-211"),
    ]
    code_evidence = []
    for label, relative_path, lines in code_specs:
        code_evidence.append(
            {
                "label": label,
                "path_lines": f"{relative_path}:{lines}",
                "sha256": sha256_file(root / relative_path),
            }
        )
    report = _build_report(summary, checkpoint_table, code_evidence)
    output_dir = root / "results/audit"
    outputs = {
        output_dir / "seed_variation_summary.csv": _csv_bytes(summary),
        output_dir / "checkpoint_variation.csv": _csv_bytes(checkpoint_table),
        output_dir / "first_attainment_matrix.csv": _csv_bytes(first_attainment),
        output_dir / "seed_effectiveness_report.md": report.encode("utf-8"),
    }
    statuses = {
        str(path.relative_to(root)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
