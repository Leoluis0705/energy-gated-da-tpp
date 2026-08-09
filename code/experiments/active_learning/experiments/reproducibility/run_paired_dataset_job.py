#!/usr/bin/env python3
"""Run one frozen, ID-aligned paired-confirmation trajectory."""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .two_dataset_paired_protocol import (
        candidate_order_digest,
        clean_id,
        dataset_configs,
        paired_round_seeds,
        reindex_predictions,
        repetition_rate,
        sha256_file,
    )
    from .protocol_artifacts import (
        PROTOCOL_VERSION,
        score_artifact_path,
        trace_artifact_path,
        validate_protocol_version,
    )
    from .formal_protocol import FormalProtocolError, load_formal_protocol
    from .interval_task_protocol import load_initial_set_ids, load_interval_task
except ImportError:  # Direct script execution on the experiment server.
    from two_dataset_paired_protocol import (
        candidate_order_digest,
        clean_id,
        dataset_configs,
        paired_round_seeds,
        reindex_predictions,
        repetition_rate,
        sha256_file,
    )
    from protocol_artifacts import (
        PROTOCOL_VERSION,
        score_artifact_path,
        trace_artifact_path,
        validate_protocol_version,
    )
    from formal_protocol import FormalProtocolError, load_formal_protocol
    from interval_task_protocol import load_initial_set_ids, load_interval_task


HERE = Path(__file__).resolve().parent
SEED_WRAPPER = HERE / "seeded_runpy_torch_compat.py"
PREDICT_SCRIPT = HERE / "paired_predict_no_shuffle.py"
METHOD_SPECS = {
    "energy_gated_da_tpp": {
        "display_name": "Energy-Gated DA-TPP",
        "selection_method_name": "energy_gated_da_tpp",
        "ablation_mode": "full",
    },
    "predicted_distance_greedy": {
        "display_name": "Predicted-Distance Greedy",
        "selection_method_name": "predicted_distance_greedy",
        "ablation_mode": "p_hit_greedy",
    },
    "predicted_target_greedy": {
        "display_name": "Predicted-Target Greedy",
        "selection_method_name": "predicted_target_greedy",
        "ablation_mode": "p_hit_greedy",
    },
    "interval_hit_greedy": {
        "display_name": "Interval-Hit Greedy",
        "selection_method_name": "interval_hit_greedy",
        "ablation_mode": "p_hit_greedy",
    },
    "always_da_tpp": {
        "display_name": "Always-DA-TPP",
        "selection_method_name": "always_da_tpp",
        "ablation_mode": "always_diversity",
    },
    "margin_only_gate": {
        "display_name": "Margin-only Gate",
        "selection_method_name": "margin_only_gate",
        "ablation_mode": "gate_no_concentration",
    },
    "group_only_gate": {
        "display_name": "Group-only Gate",
        "selection_method_name": "group_only_gate",
        "ablation_mode": "gate_no_margin",
    },
    "explore": {
        "display_name": "Explore (KMeans++ + Greedy)",
        "selection_method_name": "explore",
        "ablation_mode": "explore",
    },
    "mc_dropout": {
        "display_name": "MC Dropout",
        "selection_method_name": "mc_dropout",
        "ablation_mode": "mc_dropout",
    },
    "gradient_norm_hybrid": {
        "display_name": "Gradient-Norm Hybrid (Modulus)",
        "selection_method_name": "gradient_norm_hybrid",
        "ablation_mode": "gradient_norm_hybrid",
    },
    "random_sampling": {
        "display_name": "Random Sampling",
        "selection_method_name": "random_sampling",
        "ablation_mode": "random_sampling",
    },
    "structural_group_gate": {
        "display_name": "Structural-Group Gate",
        "selection_method_name": "structural_group_gate",
        "ablation_mode": "full",
        "quality_safeguard_fraction": None,
    },
    "structural_group_gate_q95": {
        "display_name": "Structural-Group Gate + Q95",
        "selection_method_name": "structural_group_gate_q95",
        "ablation_mode": "full",
        "quality_safeguard_fraction": 0.95,
    },
}

SELECTION_HISTORY_COLUMNS = [
    "id",
    "iteration",
    "target_feature",
    "pseudo_feature",
    "actual_feature",
    "distance",
    "selection_method",
    "P_hit",
    "U_i",
    "group_key",
    "mode",
    "ablation_mode",
    "protocol_version",
    "experiment_seed",
    "model_refit_index",
    "mc_passes",
    "mc_mask_seeds_json",
    "mc_mask_sequence_sha256",
    "mc_seed_policy_version",
    "normalizer_location",
    "normalizer_scale",
]


def selector_protocol_arguments(
    *,
    experiment_seed: int,
    acquisition_round: int,
    model_refit_index: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> list[str]:
    if int(experiment_seed) < 0 or int(acquisition_round) < 1 or int(model_refit_index) < 0:
        raise ValueError("selector seed state must use non-negative seed/refit and positive round")
    version = validate_protocol_version(protocol_version)
    return [
        "--experiment-seed",
        str(int(experiment_seed)),
        "--model-refit-index",
        str(int(model_refit_index)),
        "--protocol-version",
        version,
    ]


def run_cmd(command: list[str], cwd: Path, log, env: dict[str, str] | None = None) -> None:
    log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {subprocess.list2cmdline(command)}\n")
    log.flush()
    subprocess.run(command, cwd=cwd, env=env, check=True, stdout=log, stderr=log)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_status(path: Path, **values: object) -> None:
    write_json(path, values)


def load_oracle(path: Path, low: float, high: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    id_col = "candidate_id" if "candidate_id" in frame.columns else ("id" if "id" in frame.columns else frame.columns[0])
    value_col = next(
        (name for name in ("formation_energy", "formation_energy_per_atom", "oracle_value", "target") if name in frame.columns),
        frame.columns[1],
    )
    frame = frame.copy()
    frame["id"] = frame[id_col].map(clean_id)
    frame["oracle_value"] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["oracle_value"]).copy()
    frame["target_label"] = frame["oracle_value"].between(low, high, inclusive="both").astype(int)
    if frame["id"].duplicated().any():
        raise ValueError("duplicate candidate IDs in reference-label table")
    return frame


def prepare_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    torch.save(
        {
            "epoch": 0,
            "state_dict": checkpoint["state_dict"],
            "best_mae_error": float("inf"),
            "normalizer": checkpoint.get("normalizer"),
            "args": checkpoint.get("args", {}),
        },
        destination,
    )


def reset_checkpoint_for_training(path: Path) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["epoch"] = 0
    checkpoint["best_mae_error"] = float("inf")
    torch.save(checkpoint, path)


def copy_pool(source: Path, destination: Path, atom_init: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(source, destination)
    if not (destination / "atom_init.json").exists():
        shutil.copy2(atom_init, destination / "atom_init.json")


def compute_normalized_autc(trajectory: pd.DataFrame, total_targets: int, budget: int) -> float:
    area = 0.0
    previous_queries = 0
    previous_hits = 0
    for row in trajectory.sort_values("oracle_evaluations").itertuples(index=False):
        queries = min(int(row.oracle_evaluations), int(budget))
        area += max(0, queries - previous_queries) * previous_hits
        previous_queries = queries
        previous_hits = int(row.cumulative_target_count)
        if previous_queries >= budget:
            break
    if previous_queries < budget:
        area += (budget - previous_queries) * previous_hits
    return area / float(max(1, total_targets * budget))


def derive_round_audit(
    scores: pd.DataFrame,
    oracle: pd.DataFrame,
    round_index: int,
    route: str,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    frame = scores.copy()
    frame["id"] = frame["id"].map(clean_id)
    p_hit = pd.to_numeric(frame["P_hit"], errors="raise").to_numpy(dtype=float)
    direct_order = np.argsort(-p_hit)[: min(batch_size, len(frame))]
    direct_ids = frame.iloc[direct_order]["id"].tolist()
    selected_ids = frame.loc[pd.to_numeric(frame["selected"], errors="raise") == 1, "id"].tolist()
    direct_set = set(direct_ids)
    selected_set = set(selected_ids)
    removed_ids = [value for value in direct_ids if value not in selected_set]
    inserted_ids = [value for value in selected_ids if value not in direct_set]
    oracle_map = oracle.set_index("id")["target_label"].astype(int).to_dict()
    score_map = frame.set_index("id")
    selected_groups = [str(score_map.at[value, "group_key"]) for value in selected_ids]
    direct_hits = sum(int(oracle_map[value]) for value in direct_ids)
    selected_hits = sum(int(oracle_map[value]) for value in selected_ids)
    diagnostic = {
        "round": int(round_index),
        "route": str(route),
        "direct_top_b_candidate_ids": ";".join(direct_ids),
        "selected_candidate_ids": ";".join(selected_ids),
        "removed_by_correction_ids": ";".join(removed_ids),
        "inserted_by_correction_ids": ";".join(inserted_ids),
        "correction_replacement_count": len(removed_ids),
        "direct_top_b_target_hits": direct_hits,
        "selected_target_hits": selected_hits,
        "correction_target_gain": selected_hits - direct_hits,
        "selected_group_keys": ";".join(selected_groups),
        "selected_unique_groups": len(set(selected_groups)),
        "selected_group_repetition_rate": repetition_rate(selected_groups),
    }
    substitution_rows: list[dict] = []
    for role, values in (("removed", removed_ids), ("inserted", inserted_ids)):
        for candidate_id in values:
            substitution_rows.append(
                {
                    "round": int(round_index),
                    "route": str(route),
                    "substitution_role": role,
                    "candidate_id": candidate_id,
                    "group_key": str(score_map.at[candidate_id, "group_key"]),
                    "target_label": int(oracle_map[candidate_id]),
                    "P_hit": float(score_map.at[candidate_id, "P_hit"]),
                }
            )
    return diagnostic, pd.DataFrame(substitution_rows)


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def update_selected_history(path: Path, oracle: pd.DataFrame, method: str) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    history = pd.read_csv(path)
    if history.empty:
        return
    history["id"] = history["id"].map(clean_id)
    mapping = oracle.set_index("id")
    history["actual_feature"] = history["id"].map(mapping["oracle_value"])
    history["target_label"] = history["id"].map(mapping["target_label"]).astype(int)
    history["is_valid"] = history["target_label"]
    history["selection_method"] = method
    history.to_csv(path, index=False)


def materialize_initial_set(
    *,
    initial_ids: list[str],
    oracle: pd.DataFrame,
    cif_dir: Path,
    train_dir: Path,
    history_path: Path,
    method: str,
    seed: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> None:
    """Move one frozen seed-specific initial set into the training history."""
    pool_path = cif_dir / "id_prop.csv"
    pool = pd.read_csv(pool_path, header=None, names=["id", "prop"])
    pool["id"] = pool["id"].map(clean_id)
    wanted = [clean_id(value) for value in initial_ids]
    if len(wanted) != len(set(wanted)) or not set(wanted).issubset(set(pool["id"])):
        raise RuntimeError("frozen initial set is not a unique subset of the eligible pool")
    oracle_by_id = oracle.set_index("id")
    selected = pool.set_index("id").loc[wanted].reset_index()
    selected["prop"] = selected["id"].map(oracle_by_id["oracle_value"])
    selected[["id", "prop"]].to_csv(train_dir / "id_prop.csv", header=False, index=False)
    for candidate_id in wanted:
        source = cif_dir / f"{candidate_id}.cif"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, train_dir / source.name)
    remaining = pool[~pool["id"].isin(wanted)]
    remaining[["id", "prop"]].to_csv(pool_path, header=False, index=False)
    history = pd.DataFrame(
        {
            "id": wanted,
            "iteration": 0,
            "target_feature": float("nan"),
            "pseudo_feature": float("nan"),
            "actual_feature": [float(oracle_by_id.at[value, "oracle_value"]) for value in wanted],
            "distance": float("nan"),
            "selection_method": "frozen_initial_set",
            "P_hit": float("nan"),
            "U_i": float("nan"),
            "group_key": "INITIAL_UNASSIGNED",
            "mode": "initialization",
            "ablation_mode": "initialization",
            "protocol_version": validate_protocol_version(protocol_version),
            "experiment_seed": int(seed),
            "model_refit_index": -1,
            "mc_passes": 0,
            "mc_mask_seeds_json": "[]",
            "mc_mask_sequence_sha256": "",
            "mc_seed_policy_version": "not_applicable_initial_set",
            "normalizer_location": float("nan"),
            "normalizer_scale": float("nan"),
        },
        columns=SELECTION_HISTORY_COLUMNS,
    )
    history.to_csv(history_path, index=False)


def build_summary(
    run_dir: Path,
    dataset: str,
    method: str,
    seed: int,
    oracle: pd.DataFrame,
    budget: int,
    batch_size: int,
    initial_set_size: int = 0,
    checkpoints: tuple[int, ...] = (80, 160, 240, 320),
    protocol_version: str = PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(run_dir / "al_history.csv")
    history["id"] = history["id"].map(clean_id)
    trace = pd.read_csv(trace_artifact_path(run_dir, protocol_version=protocol_version))
    audit = pd.read_csv(run_dir / "round_diagnostics.csv")
    rows: list[dict] = []
    cumulative = 0
    for round_index in sorted(history["iteration"].astype(int).unique()):
        selected = history[history["iteration"].astype(int) == round_index].copy()
        hits = int(selected["target_label"].sum())
        cumulative += hits
        is_initial = round_index == 0 and int(initial_set_size) > 0
        if is_initial:
            margin = float("nan")
            concentration = float("nan")
            route = "initialization"
            unique_groups = int(selected["group_key"].nunique()) if "group_key" in selected else 0
            repetition = 0.0
            replacements = 0
            target_gain = 0
            oracle_evaluations = int(initial_set_size)
        else:
            trace_row = trace[trace["iteration"].astype(int) == round_index].iloc[-1]
            audit_row = audit[audit["round"].astype(int) == round_index].iloc[-1]
            margin = float(trace_row["margin_score"])
            concentration = float(trace_row["group_concentration"])
            route = str(trace_row["mode"])
            unique_groups = int(audit_row["selected_unique_groups"])
            repetition = float(audit_row["selected_group_repetition_rate"])
            replacements = int(audit_row["correction_replacement_count"])
            target_gain = int(audit_row["correction_target_gain"])
            oracle_evaluations = int(initial_set_size) + round_index * batch_size
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "round": round_index,
                "oracle_evaluations": min(oracle_evaluations, budget),
                "selected_candidate_ids": ";".join(selected["id"].tolist()),
                "selected_target_indicators": ";".join(selected["target_label"].astype(str).tolist()),
                "round_target_hits": hits,
                "cumulative_target_count": cumulative,
                "M_t": margin,
                "G_t": concentration,
                "route_choice": route,
                "unique_groups_per_batch": unique_groups,
                "group_repetition_rate": repetition,
                "correction_replacement_count": replacements,
                "correction_target_gain": target_gain,
                "checkpoint_path": str(run_dir / "checkpoints" / f"checkpoint_after_round_{round_index:03d}.pth.tar"),
                "prediction_path": str(run_dir / f"test_results_iter_{round_index}.csv"),
            }
        )
    trajectory = pd.DataFrame(rows)
    trajectory["AUTC_so_far"] = [
        compute_normalized_autc(trajectory.iloc[: index + 1], int(oracle["target_label"].sum()), int(row.oracle_evaluations))
        for index, row in enumerate(trajectory.itertuples(index=False))
    ]
    trajectory.to_csv(run_dir / "summary.csv", index=False)

    def at_budget(query_count: int) -> int:
        available = trajectory[trajectory["oracle_evaluations"] <= query_count]
        return int(available.iloc[-1]["cumulative_target_count"]) if not available.empty else 0

    modes = trajectory["route_choice"].astype(str)
    metric_row = {
                "dataset": dataset,
                "method": method,
                "seed": seed,
                "budget": budget,
                "batch_size": batch_size,
                "total_target_count": int(oracle["target_label"].sum()),
                "final_recovery": int(trajectory.iloc[-1]["cumulative_target_count"]),
                "AUTC": compute_normalized_autc(trajectory, int(oracle["target_label"].sum()), budget),
                "direct_rounds": int((modes == "threshold_greedy").sum()),
                "correction_rounds": int((modes == "diversity_aware").sum()),
                "direct_route_proportion": float((modes == "threshold_greedy").mean()),
                "mean_unique_groups_per_batch": float(trajectory["unique_groups_per_batch"].mean()),
                "mean_group_repetition_rate": float(trajectory["group_repetition_rate"].mean()),
                "total_correction_replacements": int(trajectory["correction_replacement_count"].sum()),
                "total_correction_target_gain": int(trajectory["correction_target_gain"].sum()),
    }
    for query_count in checkpoints:
        metric_row[f"recovery_at_{int(query_count)}"] = at_budget(int(query_count))
    metrics = pd.DataFrame([metric_row])
    metrics.to_csv(run_dir / "run_metrics.csv", index=False)
    return trajectory, metrics


def environment_info() -> dict:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }


def run_job(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    run_dir = Path(args.run_dir).resolve()
    config = dataset_configs()[args.dataset]
    formal_protocol = load_formal_protocol(args.protocol_config) if args.protocol_config else None
    protocol_version = validate_protocol_version(
        formal_protocol.protocol_version if formal_protocol is not None else PROTOCOL_VERSION
    )
    if formal_protocol is not None:
        config = formal_protocol.resolve_dataset_config(config, method=args.method, seed=args.seed)
    interval_task = load_interval_task(args.task_config) if args.task_config else None
    if interval_task is not None:
        config = interval_task.apply(config)
    method_spec = METHOD_SPECS[args.method]
    status_path = run_dir / "status.json"
    if status_path.exists() and (run_dir / "run_metrics.csv").exists():
        old = json.loads(status_path.read_text(encoding="utf-8"))
        if old.get("status") == "DONE":
            print(f"SKIP_DONE {run_dir}")
            return
    run_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = project_root / config.pool_relative_path
    oracle_path = project_root / config.oracle_relative_path
    checkpoint_source = project_root / "checkpoint_formation_clean.pth.tar"
    atom_init = project_root / "atom_init.json"
    selector_script = project_root / "active_learning_energy_gate_ablation.py"
    group_key_map = (
        project_root / formal_protocol.group_key_map_relative_path
        if formal_protocol is not None and formal_protocol.group_key_map_relative_path
        else None
    )
    initial_sets_path = (
        project_root / interval_task.initial_sets_relative_path
        if interval_task is not None
        else None
    )
    required = [pool_dir / "id_prop.csv", oracle_path, checkpoint_source, atom_init, selector_script, project_root / "main.py", PREDICT_SCRIPT]
    if group_key_map is not None:
        required.append(group_key_map)
    if initial_sets_path is not None:
        required.append(initial_sets_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing inputs: " + "; ".join(missing))
    oracle = load_oracle(oracle_path, config.target_low, config.target_high)
    expected_pool_size = interval_task.pool_size if interval_task is not None else 640
    if len(oracle) != expected_pool_size or int(oracle["target_label"].sum()) != config.target_count:
        raise RuntimeError("reference-label table does not match frozen candidate/target counts")
    source_ids = pd.read_csv(pool_dir / "id_prop.csv", header=None, usecols=[0]).iloc[:, 0].map(clean_id).tolist()
    if len(source_ids) != expected_pool_size or len(set(source_ids)) != expected_pool_size or set(source_ids) != set(oracle["id"]):
        raise RuntimeError("candidate pool and reference-label IDs do not form the frozen ID mapping")

    run_config = {
        **config.to_dict(),
        "method": args.method,
        "method_display_name": method_spec["display_name"],
        "seed": args.seed,
        "project_root": str(project_root),
        "pool_path": str(pool_dir),
        "pool_id_prop_sha256": sha256_file(pool_dir / "id_prop.csv"),
        "candidate_order_sha256": candidate_order_digest(source_ids),
        "oracle_path": str(oracle_path),
        "oracle_sha256": sha256_file(oracle_path),
        "checkpoint_path": str(checkpoint_source),
        "checkpoint_sha256": sha256_file(checkpoint_source),
        "predict_script": str(PREDICT_SCRIPT),
        "predict_script_sha256": sha256_file(PREDICT_SCRIPT),
        "selector_script": str(selector_script),
        "selector_script_sha256": sha256_file(selector_script),
        "protocol_version": protocol_version,
        "formal_protocol_path": str(formal_protocol.source_path) if formal_protocol is not None else None,
        "formal_protocol_sha256": formal_protocol.sha256 if formal_protocol is not None else None,
        "formal_protocol_phase": formal_protocol.phase if formal_protocol is not None else "legacy_embedded",
        "formal_protocol_frozen": formal_protocol.frozen if formal_protocol is not None else None,
        "interval_task_path": str(interval_task.source_path) if interval_task is not None else None,
        "interval_task_sha256": interval_task.sha256 if interval_task is not None else None,
        "interval_task_id": interval_task.task_id if interval_task is not None else None,
        "initial_sets_path": str(initial_sets_path) if initial_sets_path is not None else None,
        "initial_sets_sha256": sha256_file(initial_sets_path) if initial_sets_path is not None else None,
        "group_key_mode": formal_protocol.group_key_mode if formal_protocol is not None else config.group_key_construction,
        "group_key_map_path": str(group_key_map) if group_key_map is not None else None,
        "group_key_map_sha256": sha256_file(group_key_map) if group_key_map is not None else None,
        "seed_policy": "base=seed*1000000; inference=base+100000+round; training=base+200000+round",
        "command_line": subprocess.list2cmdline(sys.argv),
    }
    write_json(run_dir / "run_config.json", run_config)
    write_json(run_dir / "environment.json", environment_info())
    (run_dir / "command.txt").write_text(subprocess.list2cmdline(sys.argv) + "\n", encoding="utf-8")
    started = time.time()
    write_status(status_path, status="RUNNING", started_at=time.strftime("%Y-%m-%d %H:%M:%S"), dataset=args.dataset, method=args.method, seed=args.seed)
    try:
        cif_dir = run_dir / "cifs"
        train_dir = run_dir / "train_dir"
        work_dir = run_dir / "work"
        checkpoint_dir = run_dir / "checkpoints"
        copy_pool(pool_dir, cif_dir, atom_init)
        train_dir.mkdir(exist_ok=True)
        work_dir.mkdir(exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        train_csv = train_dir / "id_prop.csv"
        if not train_csv.exists():
            pd.DataFrame(columns=["id", "prop"]).to_csv(train_csv, header=False, index=False)
            shutil.copy2(atom_init, train_dir / "atom_init.json")
        current_checkpoint = run_dir / "current.pth.tar"
        if not current_checkpoint.exists():
            prepare_checkpoint(checkpoint_source, current_checkpoint)
        history_path = run_dir / "al_history.csv"
        base_env = os.environ.copy()
        base_env["ACTIVE_HISTORY"] = str(history_path)
        base_env["CURRENT_STRATEGY"] = method_spec["selection_method_name"]
        base_env["TOTAL_POOL_SIZE"] = "640"
        base_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        base_env["PYTHONPATH"] = os.pathsep.join([str(project_root), str(HERE.parents[1]), base_env.get("PYTHONPATH", "")])
        with (run_dir / "run.log").open("a", encoding="utf-8") as log:
            initial_set_size = interval_task.initial_set_size if interval_task is not None else 0
            initialization_marker = run_dir / "initialization_manifest.json"
            if interval_task is not None and not initialization_marker.exists():
                if history_path.exists() and history_path.stat().st_size:
                    raise RuntimeError("partial interval initialization found without a manifest; use a fresh run directory")
                initial_ids = load_initial_set_ids(
                    initial_sets_path,
                    seed=args.seed,
                    pool_ids=source_ids,
                    expected_size=initial_set_size,
                )
                materialize_initial_set(
                    initial_ids=initial_ids,
                    oracle=oracle,
                    cif_dir=cif_dir,
                    train_dir=train_dir,
                    history_path=history_path,
                    method=args.method,
                    seed=args.seed,
                    protocol_version=protocol_version,
                )
                reset_checkpoint_for_training(current_checkpoint)
                initial_seeds = paired_round_seeds(args.seed, 0)
                initial_env = base_env.copy()
                initial_env["PYTHONHASHSEED"] = str(args.seed)
                run_cmd(
                    [
                        sys.executable,
                        str(SEED_WRAPPER),
                        "--seed",
                        str(initial_seeds["training_seed"]),
                        str(project_root / "main.py"),
                        "--epochs",
                        str(config.refit_epochs),
                        "--lr",
                        str(config.learning_rate),
                        "--batch-size",
                        str(config.cgcnn_batch_size),
                        "--train-ratio",
                        "1",
                        "--val-ratio",
                        "0",
                        "--test-ratio",
                        "0",
                        "--task",
                        "regression",
                        "--resume",
                        str(current_checkpoint),
                        str(train_dir),
                    ],
                    cwd=work_dir,
                    log=log,
                    env=initial_env,
                )
                shutil.move(str(work_dir / "checkpoint.pth.tar"), current_checkpoint)
                initial_checkpoint = checkpoint_dir / "checkpoint_after_initial_set.pth.tar"
                shutil.copy2(current_checkpoint, initial_checkpoint)
                write_json(
                    initialization_marker,
                    {
                        "seed": args.seed,
                        "candidate_ids": initial_ids,
                        "candidate_order_sha256": candidate_order_digest(initial_ids),
                        "training_seed": initial_seeds["training_seed"],
                        "checkpoint_path": str(initial_checkpoint),
                        "checkpoint_sha256": sha256_file(initial_checkpoint),
                    },
                )
            completed_rows = len(pd.read_csv(history_path)) if history_path.exists() and history_path.stat().st_size else 0
            acquisition_rows = max(0, completed_rows - initial_set_size)
            if acquisition_rows % config.batch_size:
                raise RuntimeError("history row count is not aligned to the frozen batch size")
            start_round = acquisition_rows // config.batch_size + 1
            for round_index in range(start_round, config.rounds + 1):
                current_id_path = cif_dir / "id_prop.csv"
                if not current_id_path.exists() or current_id_path.stat().st_size == 0:
                    break
                current_ids = pd.read_csv(current_id_path, header=None, usecols=[0]).iloc[:, 0].map(clean_id).tolist()
                seeds = paired_round_seeds(args.seed, round_index)
                round_env = base_env.copy()
                round_env["PYTHONHASHSEED"] = str(args.seed)
                run_cmd(
                    [
                        sys.executable,
                        str(SEED_WRAPPER),
                        "--seed",
                        str(seeds["inference_seed"]),
                        str(PREDICT_SCRIPT),
                        str(current_checkpoint),
                        str(cif_dir),
                        "--batch-size",
                        str(config.cgcnn_batch_size),
                    ],
                    cwd=work_dir,
                    log=log,
                    env=round_env,
                )
                prediction_path = run_dir / f"test_results_iter_{round_index}.csv"
                shutil.move(str(work_dir / "test_results.csv"), prediction_path)
                prediction = pd.read_csv(prediction_path, header=None, names=["id", "actual", "prediction"])
                prediction = reindex_predictions(prediction, current_ids)
                prediction.to_csv(prediction_path, header=False, index=False)
                append_frame(
                    run_dir / "prediction_manifest.csv",
                    pd.DataFrame(
                        [
                            {
                                "round": round_index,
                                "path": str(prediction_path),
                                "sha256": sha256_file(prediction_path),
                                "candidate_count": len(prediction),
                                "candidate_order_sha256": candidate_order_digest(prediction["id"].tolist()),
                                "inference_seed": seeds["inference_seed"],
                            }
                        ]
                    ),
                )
                pseudo_dir = run_dir / "pseudo_dir"
                if pseudo_dir.exists():
                    shutil.rmtree(pseudo_dir)
                pseudo_dir.mkdir()
                for cif in cif_dir.glob("*.cif"):
                    shutil.copy2(cif, pseudo_dir / cif.name)
                prediction[["id", "prediction"]].to_csv(pseudo_dir / "id_prop.csv", header=False, index=False)
                shutil.copy2(atom_init, pseudo_dir / "atom_init.json")
                run_cmd(
                    [
                        sys.executable,
                        str(SEED_WRAPPER),
                        "--seed",
                        str(seeds["inference_seed"]),
                        str(selector_script),
                        "--model",
                        str(current_checkpoint),
                        "--original-dir",
                        str(cif_dir),
                        "--pseudo-dir",
                        str(pseudo_dir),
                        "--test-results",
                        str(prediction_path),
                        "--output-dir",
                        str(train_dir),
                        "--target-feature",
                        str((config.target_low + config.target_high) / 2.0),
                        "--target-low",
                        str(config.target_low),
                        "--target-high",
                        str(config.target_high),
                        "--selection-size",
                        str(config.batch_size),
                        "--iteration",
                        str(round_index),
                        *selector_protocol_arguments(
                            experiment_seed=args.seed,
                            acquisition_round=round_index,
                            model_refit_index=round_index - 1,
                            protocol_version=protocol_version,
                        ),
                        "--mc-passes",
                        str(config.mc_passes),
                        "--dropout-rate",
                        str(config.dropout_rate),
                        "--score-log-dir",
                        str(run_dir),
                        "--selection-method-name",
                        method_spec["selection_method_name"],
                        "--ablation-mode",
                        method_spec["ablation_mode"],
                        "--margin-threshold",
                        str(config.M0),
                        "--concentration-threshold",
                        str(config.G0),
                        "--alpha",
                        str(config.alpha),
                        "--beta",
                        str(config.beta),
                        "--gamma",
                        str(config.gamma),
                        *(
                            [
                                "--quality-safeguard-fraction",
                                str(method_spec["quality_safeguard_fraction"]),
                            ]
                            if method_spec.get("quality_safeguard_fraction") is not None
                            else []
                        ),
                        *(["--group-key-map", str(group_key_map)] if group_key_map is not None else []),
                    ],
                    cwd=project_root,
                    log=log,
                    env=round_env,
                )
                score_path = score_artifact_path(
                    run_dir,
                    method_name=method_spec["selection_method_name"],
                    iteration=round_index,
                    protocol_version=protocol_version,
                )
                scores = pd.read_csv(score_path)
                trace = pd.read_csv(
                    trace_artifact_path(run_dir, protocol_version=protocol_version)
                )
                route = str(trace[trace["iteration"].astype(int) == round_index].iloc[-1]["mode"])
                diagnostic, substitutions = derive_round_audit(scores, oracle, round_index, route, config.batch_size)
                diagnostic.update({"dataset": args.dataset, "method": args.method, "seed": args.seed})
                append_frame(run_dir / "round_diagnostics.csv", pd.DataFrame([diagnostic]))
                if not substitutions.empty:
                    substitutions.insert(0, "seed", args.seed)
                    substitutions.insert(0, "method", args.method)
                    substitutions.insert(0, "dataset", args.dataset)
                    append_frame(run_dir / "correction_substitutions.csv", substitutions)

                train_frame = pd.read_csv(train_csv, header=None, names=["id", "prop"])
                train_frame["id"] = train_frame["id"].map(clean_id)
                oracle_values = oracle.set_index("id")["oracle_value"].to_dict()
                train_frame["prop"] = train_frame["id"].map(oracle_values)
                if train_frame["prop"].isna().any():
                    raise RuntimeError("selected candidate missing from reference-label table")
                train_frame.to_csv(train_csv, header=False, index=False)
                update_selected_history(history_path, oracle, args.method)
                reset_checkpoint_for_training(current_checkpoint)
                run_cmd(
                    [
                        sys.executable,
                        str(SEED_WRAPPER),
                        "--seed",
                        str(seeds["training_seed"]),
                        str(project_root / "main.py"),
                        "--epochs",
                        str(config.refit_epochs),
                        "--lr",
                        str(config.learning_rate),
                        "--batch-size",
                        str(config.cgcnn_batch_size),
                        "--train-ratio",
                        "1",
                        "--val-ratio",
                        "0",
                        "--test-ratio",
                        "0",
                        "--task",
                        "regression",
                        "--resume",
                        str(current_checkpoint),
                        str(train_dir),
                    ],
                    cwd=work_dir,
                    log=log,
                    env=round_env,
                )
                shutil.move(str(work_dir / "checkpoint.pth.tar"), current_checkpoint)
                round_checkpoint = checkpoint_dir / f"checkpoint_after_round_{round_index:03d}.pth.tar"
                shutil.copy2(current_checkpoint, round_checkpoint)
                append_frame(
                    run_dir / "checkpoint_manifest.csv",
                    pd.DataFrame(
                        [
                            {
                                "round": round_index,
                                "path": str(round_checkpoint),
                                "sha256": sha256_file(round_checkpoint),
                                "training_seed": seeds["training_seed"],
                            }
                        ]
                    ),
                )
        update_selected_history(history_path, oracle, args.method)
        _, metrics = build_summary(
            run_dir,
            args.dataset,
            args.method,
            args.seed,
            oracle,
            config.budget,
            config.batch_size,
            initial_set_size=interval_task.initial_set_size if interval_task is not None else 0,
            checkpoints=interval_task.checkpoints if interval_task is not None else (80, 160, 240, 320),
            protocol_version=protocol_version,
        )
        completed_history = pd.read_csv(history_path)
        metrics["candidate_sequence_sha256"] = candidate_order_digest(completed_history["id"].tolist())
        metrics["formal_protocol_sha256"] = formal_protocol.sha256 if formal_protocol is not None else None
        metrics.to_csv(run_dir / "run_metrics.csv", index=False)
        write_status(
            status_path,
            status="DONE",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            ended_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=time.time() - started,
            dataset=args.dataset,
            method=args.method,
            seed=args.seed,
            final_recovery=int(metrics.iloc[0]["final_recovery"]),
            AUTC=float(metrics.iloc[0]["AUTC"]),
        )
    except Exception as error:
        write_status(
            status_path,
            status="FAILED",
            started_at=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
            ended_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_seconds=time.time() - started,
            dataset=args.dataset,
            method=args.method,
            seed=args.seed,
            error=str(error),
        )
        import traceback

        (run_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def parse_experiment_seed(value: str) -> int:
    seed = int(value)
    if seed not in range(30) and seed not in range(101, 116):
        raise argparse.ArgumentTypeError(
            "experiment seed must be in a declared cohort (0..29 or 101..115)"
        )
    return seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=sorted(dataset_configs()), required=True)
    parser.add_argument("--method", choices=sorted(METHOD_SPECS), required=True)
    parser.add_argument("--seed", type=parse_experiment_seed, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--protocol-config")
    parser.add_argument("--task-config")
    args = parser.parse_args()
    run_job(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
