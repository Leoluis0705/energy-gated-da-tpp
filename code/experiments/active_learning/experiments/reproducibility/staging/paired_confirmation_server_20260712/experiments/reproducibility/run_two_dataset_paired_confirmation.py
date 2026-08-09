#!/usr/bin/env python3
"""Launch exactly 20 frozen paired-confirmation trajectories (seeds 5-9)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .two_dataset_paired_protocol import (
        EQUIVALENCE_MARGIN,
        build_planned_runs,
        candidate_order_digest,
        clean_id,
        dataset_configs,
        sha256_file,
        sha256_tree,
    )
except ImportError:
    from two_dataset_paired_protocol import (
        EQUIVALENCE_MARGIN,
        build_planned_runs,
        candidate_order_digest,
        clean_id,
        dataset_configs,
        sha256_file,
        sha256_tree,
    )


HERE = Path(__file__).resolve().parent
JOB_RUNNER = HERE / "run_paired_dataset_job.py"
PREDICT_SCRIPT = HERE / "paired_predict_no_shuffle.py"
DEFAULT_OUTPUT_NAME = "paired_two_dataset_confirmation_20260712"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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


def run_command(project_root: Path, output_root: Path, row: dict) -> list[str]:
    return [
        sys.executable,
        str(JOB_RUNNER),
        "--project-root",
        str(project_root),
        "--dataset",
        row["dataset"],
        "--method",
        row["method"],
        "--seed",
        str(row["seed"]),
        "--run-dir",
        str(output_root / "runs" / row["dataset"] / row["method"] / f"seed_{row['seed']}"),
    ]


def frozen_input_record(project_root: Path, dataset: str) -> dict:
    config = dataset_configs()[dataset]
    pool = project_root / config.pool_relative_path
    oracle = project_root / config.oracle_relative_path
    checkpoint = project_root / "checkpoint_formation_clean.pth.tar"
    ids = pd.read_csv(pool / "id_prop.csv", header=None, usecols=[0]).iloc[:, 0].map(clean_id).tolist()
    if len(ids) != 640 or len(set(ids)) != 640:
        raise RuntimeError(f"{dataset}: frozen pool must contain 640 unique candidate IDs")
    reference = pd.read_csv(oracle)
    id_col = "candidate_id" if "candidate_id" in reference.columns else ("id" if "id" in reference.columns else reference.columns[0])
    reference_ids = reference[id_col].map(clean_id).tolist()
    if set(ids) != set(reference_ids):
        raise RuntimeError(f"{dataset}: candidate pool and reference-label ID sets differ")
    return {
        **config.to_dict(),
        "pool_path": str(pool),
        "pool_tree_sha256": sha256_tree(pool),
        "pool_id_prop_sha256": sha256_file(pool / "id_prop.csv"),
        "candidate_count": len(ids),
        "candidate_order_sha256": candidate_order_digest(ids),
        "candidate_order_source": "id_prop.csv row order",
        "oracle_path": str(oracle),
        "oracle_sha256": sha256_file(oracle),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
    }


def prepare_manifest(project_root: Path, output_root: Path, max_workers: int) -> dict:
    required = [
        project_root / "main.py",
        project_root / "active_learning_energy_gate_ablation.py",
        project_root / "active_learning_etdg_tage.py",
        project_root / "checkpoint_formation_clean.pth.tar",
        project_root / "atom_init.json",
        JOB_RUNNER,
        PREDICT_SCRIPT,
        HERE / "seeded_runpy_torch_compat.py",
        HERE / "two_dataset_paired_protocol.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + "; ".join(missing))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = build_planned_runs(range(5, 10))
    if len(rows) != 20:
        raise RuntimeError("planned matrix must contain exactly 20 trajectories")
    for row in rows:
        row["run_dir"] = str(output_root / "runs" / row["dataset"] / row["method"] / f"seed_{row['seed']}")
        row["command"] = subprocess.list2cmdline(run_command(project_root, output_root, row))
    planned_path = output_root / "PLANNED_RUNS.csv"
    pd.DataFrame(rows).to_csv(planned_path, index=False)

    inputs = {dataset: frozen_input_record(project_root, dataset) for dataset in ("limo", "mnoxide")}
    manifest = {
        "scope": "exactly_20_required_trajectories",
        "created_before_seed_5_9_results": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seeds": [5, 6, 7, 8, 9],
        "datasets": ["limo", "mnoxide"],
        "methods": ["energy_gated_da_tpp", "predicted_distance_greedy"],
        "optional_always_correction_launched": False,
        "trajectory_count": 20,
        "max_workers": max_workers,
        "mnoxide_AUTC_non_degradation_margin": EQUIVALENCE_MARGIN,
        "margin_justification": (
            "0.01 is one percentage point on normalized AUTC and is smaller than the retained "
            "Mn-oxide between-seed SD (approximately 0.027-0.033)."
        ),
        "mnoxide_fallback_decision_rule": {
            "DIRECT_FALLBACK_CONFIRMED": (
                "correction rounds are at most 5% of corrected-cohort Mn rounds and the paired-bootstrap "
                "95% CI lower bound for Gate minus Greedy AUTC is at least -0.01"
            ),
            "FALLBACK_WITH_DEGRADATION": "paired-bootstrap 95% CI upper bound is below -0.01",
            "otherwise": "FALLBACK_INCONCLUSIVE",
        },
        "prediction_loader_shuffle": False,
        "explicit_candidate_id_reindex": True,
        "seed_policy": "base=seed*1000000; inference=base+100000+round; training=base+200000+round",
        "pair_controls": [
            "same frozen pool and candidate ordering",
            "same reference-label mapping",
            "same initial checkpoint and empty initial labeled set",
            "same Python/NumPy/PyTorch/CUDA/DataLoader seed schedule",
            "same training and evaluation implementation within each pair",
        ],
        "group_key_construction": "sorted element-system key parsed from each candidate CIF",
        "expected_runtime": {
            "historical_mn_minutes_per_trajectory": "16-19",
            "estimated_limo_minutes_per_trajectory": "32-40",
            "estimated_total_trajectory_minutes": "480-590",
            "estimated_wall_clock_with_4_workers": "2.5-3.5 hours, subject to server I/O and GPU contention",
        },
        "legacy_seeds_0_4_compatibility": {
            "pooling_allowed": False,
            "reason": (
                "Retained seeds 0-4 used the project-root prediction path with shuffle=True and no explicit "
                "candidate-order reindex. Mn-oxide also used alpha=0.10 rather than the audited alpha=0.05. "
                "They remain an archived cohort and will not be pooled silently with corrected seeds 5-9."
            ),
            "limo_differences": ["prediction_loader_shuffle", "explicit_candidate_id_reindex"],
            "mnoxide_differences": ["prediction_loader_shuffle", "explicit_candidate_id_reindex", "alpha"],
        },
        "inputs": inputs,
        "source_hashes": {
            "job_runner": sha256_file(JOB_RUNNER),
            "predict_no_shuffle": sha256_file(PREDICT_SCRIPT),
            "seed_wrapper": sha256_file(HERE / "seeded_runpy_torch_compat.py"),
            "protocol": sha256_file(HERE / "two_dataset_paired_protocol.py"),
            "selector": sha256_file(project_root / "active_learning_energy_gate_ablation.py"),
            "feature_uncertainty": sha256_file(project_root / "active_learning_etdg_tage.py"),
            "cgcnn_main": sha256_file(project_root / "main.py"),
        },
        "planned_runs_csv": str(planned_path),
        "planned_runs_sha256": sha256_file(planned_path),
        "environment_at_freeze": environment_info(),
    }
    manifest_path = output_root / "FROZEN_LAUNCH_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = ["seeds", "datasets", "methods", "trajectory_count", "mnoxide_AUTC_non_degradation_margin", "inputs", "source_hashes"]
        changed = [key for key in immutable if existing.get(key) != manifest.get(key)]
        if changed:
            raise RuntimeError(f"existing frozen manifest differs in immutable fields: {changed}")
        return existing
    write_json(manifest_path, manifest)
    markdown = f"""# Frozen Two-Dataset Paired Launch Manifest

- Required trajectories: **20** (two datasets x two methods x seeds 5-9)
- Optional Always Correction: **not launched**
- Prediction DataLoader: `shuffle=False`; explicit one-to-one candidate-ID reindexing enabled
- Mn-oxide AUTC non-degradation margin: **+/-{EQUIVALENCE_MARGIN:.2f}**, frozen before new results
- Li-M-O fixed selector: `M0=1.0, G0=0.50, alpha=0.10, beta=0.20, gamma=0.10`
- Mn-oxide fixed selector: `M0=1.0, G0=0.75, alpha=0.05, beta=0.20, gamma=0.10`
- Legacy seeds 0-4 pooling: **not allowed** because the prediction-order protocol differs; Mn alpha also differs
- Expected wall time with {max_workers} workers: approximately 2.5-3.5 hours

Exact inputs, SHA-256 values, commands and output paths are in `FROZEN_LAUNCH_MANIFEST.json` and `PLANNED_RUNS.csv`.
"""
    (output_root / "FROZEN_LAUNCH_MANIFEST.md").write_text(markdown, encoding="utf-8")
    return manifest


def execute_pair(project_root: Path, output_root: Path, dataset: str, seed: int) -> list[dict]:
    results: list[dict] = []
    for method in ("energy_gated_da_tpp", "predicted_distance_greedy"):
        row = {"dataset": dataset, "method": method, "seed": seed}
        command = run_command(project_root, output_root, row)
        started = time.time()
        try:
            process = subprocess.run(command, cwd=project_root, text=True, capture_output=True)
            results.append(
                {
                    **row,
                    "return_code": process.returncode,
                    "elapsed_seconds": time.time() - started,
                    "stdout": process.stdout[-4000:],
                    "stderr": process.stderr[-4000:],
                    "command": subprocess.list2cmdline(command),
                }
            )
        except Exception as error:
            results.append(
                {
                    **row,
                    "return_code": -1,
                    "elapsed_seconds": time.time() - started,
                    "stdout": "",
                    "stderr": str(error),
                    "command": subprocess.list2cmdline(command),
                }
            )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-workers", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "experiments" / "reproducibility" / "results" / DEFAULT_OUTPUT_NAME
    manifest = prepare_manifest(project_root, output_root, args.max_workers)
    print(pd.read_csv(output_root / "PLANNED_RUNS.csv").to_string(index=False))
    if args.prepare_only:
        print(f"PREPARED_ONLY {output_root}")
        return 0

    started = time.time()
    write_json(
        output_root / "LAUNCHER_STATUS.json",
        {"status": "RUNNING", "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "planned": 20, "completed": 0, "failed": 0},
    )
    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(execute_pair, project_root, output_root, dataset, seed): (dataset, seed)
            for dataset in ("limo", "mnoxide")
            for seed in range(5, 10)
        }
        for future in as_completed(futures):
            all_results.extend(future.result())
            completed = sum(int(row["return_code"] == 0) for row in all_results)
            failed = sum(int(row["return_code"] != 0) for row in all_results)
            write_json(
                output_root / "LAUNCHER_STATUS.json",
                {
                    "status": "RUNNING",
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                    "last_update_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "planned": 20,
                    "attempted": len(all_results),
                    "completed": completed,
                    "failed": failed,
                },
            )
    result_frame = pd.DataFrame(all_results).sort_values(["dataset", "seed", "method"])
    result_frame.to_csv(output_root / "LAUNCH_RESULTS.csv", index=False)
    failures = result_frame[result_frame["return_code"] != 0].copy()
    failures.to_csv(output_root / "failed_runs.csv", index=False)
    completed = int((result_frame["return_code"] == 0).sum())
    failed = int((result_frame["return_code"] != 0).sum())
    final_status = {
        "status": "DONE" if completed == 20 and failed == 0 else "COMPLETED_WITH_FAILURES",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "planned": int(manifest["trajectory_count"]),
        "completed": completed,
        "failed": failed,
    }
    write_json(output_root / "LAUNCHER_STATUS.json", final_status)
    write_json(output_root / "COMPLETION.json", final_status)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
