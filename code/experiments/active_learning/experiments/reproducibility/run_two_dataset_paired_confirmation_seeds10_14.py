#!/usr/bin/env python3
"""Launch exactly 20 corrected paired trajectories for seeds 10-14."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

try:
    from .run_two_dataset_paired_confirmation import environment_info, frozen_input_record
    from .two_dataset_paired_protocol import EQUIVALENCE_MARGIN, METHODS, dataset_configs, sha256_file
except ImportError:
    from run_two_dataset_paired_confirmation import environment_info, frozen_input_record
    from two_dataset_paired_protocol import EQUIVALENCE_MARGIN, METHODS, dataset_configs, sha256_file


HERE = Path(__file__).resolve().parent
CORE_RUNNER = HERE / "run_paired_dataset_job.py"
JOB_WRAPPER = HERE / "run_paired_dataset_job_seed10_14.py"
PREDICT_SCRIPT = HERE / "paired_predict_no_shuffle.py"
SEED_WRAPPER = HERE / "seeded_runpy_torch_compat.py"
PROTOCOL_SCRIPT = HERE / "two_dataset_paired_protocol.py"
DEFAULT_OUTPUT_NAME = "paired_two_dataset_confirmation_seeds_10_14_20260713"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_new_run_matrix() -> list[dict]:
    return [
        {"dataset": dataset, "method": method, "seed": seed}
        for dataset in ("limo", "mnoxide")
        for seed in range(10, 15)
        for method in METHODS
    ]


def validate_protocol_match(previous: dict, current: dict) -> None:
    def normalized_inputs(value: dict) -> dict:
        return {
            dataset: {
                key: item
                for key, item in record.items()
                if not key.endswith("_path") and key not in {"pool_path", "oracle_path", "checkpoint_path"}
            }
            for dataset, record in value.items()
        }

    changed = []
    for field in ["prediction_loader_shuffle", "explicit_candidate_id_reindex", "source_hashes"]:
        if previous.get(field) != current.get(field):
            changed.append(field)
    if normalized_inputs(previous.get("inputs", {})) != normalized_inputs(current.get("inputs", {})):
        changed.append("inputs")
    if changed:
        raise RuntimeError(f"corrected seeds 10-14 protocol differs from seeds 5-9 in: {changed}")


def core_source_hashes(project_root: Path) -> dict:
    return {
        "job_runner": sha256_file(CORE_RUNNER),
        "predict_no_shuffle": sha256_file(PREDICT_SCRIPT),
        "seed_wrapper": sha256_file(SEED_WRAPPER),
        "protocol": sha256_file(PROTOCOL_SCRIPT),
        "selector": sha256_file(project_root / "active_learning_energy_gate_ablation.py"),
        "feature_uncertainty": sha256_file(project_root / "active_learning_etdg_tage.py"),
        "cgcnn_main": sha256_file(project_root / "main.py"),
    }


def job_command(project_root: Path, output_root: Path, row: dict) -> list[str]:
    return [
        sys.executable,
        str(JOB_WRAPPER),
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


def prepare_manifest(
    project_root: Path,
    output_root: Path,
    previous_manifest_path: Path,
    max_workers: int,
) -> dict:
    if not previous_manifest_path.exists():
        raise FileNotFoundError(previous_manifest_path)
    previous = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
    inputs = {dataset: frozen_input_record(project_root, dataset) for dataset in ("limo", "mnoxide")}
    comparison = {
        "prediction_loader_shuffle": False,
        "explicit_candidate_id_reindex": True,
        "inputs": inputs,
        "source_hashes": core_source_hashes(project_root),
    }
    validate_protocol_match(previous, comparison)

    rows = build_new_run_matrix()
    if len(rows) != 20 or len({(r["dataset"], r["method"], r["seed"]) for r in rows}) != 20:
        raise RuntimeError("new corrected matrix must contain exactly 20 unique trajectories")
    output_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        row["run_dir"] = str(output_root / "runs" / row["dataset"] / row["method"] / f"seed_{row['seed']}")
        row["command"] = subprocess.list2cmdline(job_command(project_root, output_root, row))
    planned_path = output_root / "PLANNED_RUNS.csv"
    pd.DataFrame(rows).to_csv(planned_path, index=False)

    manifest = {
        "scope": "exactly_20_corrected_trajectories_seeds_10_14",
        "created_before_seed_10_14_results": True,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seeds": [10, 11, 12, 13, 14],
        "datasets": ["limo", "mnoxide"],
        "methods": list(METHODS),
        "trajectory_count": 20,
        "max_workers": max_workers,
        "optional_or_extra_methods_launched": False,
        "legacy_seeds_0_4_excluded": True,
        "combine_after_completion": "corrected seeds 5-14 only",
        "prediction_loader_shuffle": False,
        "explicit_candidate_id_reindex": True,
        "mc_dropout_id_mapping": "variance dictionary keyed by candidate ID; prediction table explicitly reindexed",
        "seed_policy": "base=seed*1000000; inference=base+100000+round; training=base+200000+round",
        "mnoxide_AUTC_non_degradation_margin": EQUIVALENCE_MARGIN,
        "decision_rules": {
            "Li-M-O CONSISTENT_ADVANTAGE": "at least 8/10 positive paired AUTC differences and bootstrap 95% CI above zero",
            "Li-M-O SMALL_MEAN_ADVANTAGE": "positive mean paired difference but CI includes zero",
            "Li-M-O COMPARABLE_PERFORMANCE": "difference close to zero with mixed outcomes",
            "Li-M-O GREEDY_ADVANTAGE": "negative paired mean with consistent direction",
            "Mn DIRECT_FALLBACK_CONFIRMED": "correction absent or operationally inactive and paired AUTC within +/-0.01",
            "Mn FALLBACK_WITH_DEGRADATION": "consistent loss outside -0.01 margin",
        },
        "expected_runtime": {
            "based_on_corrected_seeds_5_9_summed_trajectory_hours": 7.37,
            "expected_summed_GPU_hours": "7.0-8.0",
            "expected_wall_hours_with_4_workers": "2.3-2.8",
        },
        "inputs": inputs,
        "source_hashes": comparison["source_hashes"],
        "new_wrapper_sha256": sha256_file(JOB_WRAPPER),
        "launcher_sha256": sha256_file(Path(__file__).resolve()),
        "previous_corrected_manifest_path": str(previous_manifest_path),
        "previous_corrected_manifest_sha256": sha256_file(previous_manifest_path),
        "protocol_match_to_corrected_seeds_5_9": True,
        "planned_runs_csv": str(planned_path),
        "planned_runs_sha256": sha256_file(planned_path),
        "environment_at_freeze": environment_info(),
        "gpu_telemetry": "5-second device-level samples per run; concurrent jobs share the device",
    }
    manifest_path = output_root / "FROZEN_CORRECTED_SEEDS_10_14_MANIFEST.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = [
            "seeds",
            "datasets",
            "methods",
            "trajectory_count",
            "prediction_loader_shuffle",
            "explicit_candidate_id_reindex",
            "inputs",
            "source_hashes",
            "decision_rules",
        ]
        changed = [key for key in immutable if existing.get(key) != manifest.get(key)]
        if changed:
            raise RuntimeError(f"existing frozen manifest differs in immutable fields: {changed}")
        return existing
    write_json(manifest_path, manifest)
    (output_root / "FROZEN_CORRECTED_SEEDS_10_14_MANIFEST.md").write_text(
        "# Frozen Corrected Seeds 10-14 Manifest\n\n"
        "- Exactly 20 trajectories: two datasets x two methods x seeds 10-14.\n"
        "- Core runner, candidate pools, labels, checkpoint, selector, seed policy and evaluation match corrected seeds 5-9.\n"
        "- `shuffle=False` and explicit candidate-ID reindexing remain active.\n"
        "- Mn-oxide non-degradation margin remains frozen at +/-0.01 AUTC.\n"
        "- Legacy seeds 0-4 and all extra methods are excluded.\n"
        "- Expected wall time with four workers: 2.3-2.8 hours.\n\n"
        "Exact hashes, commands and output paths are recorded in the JSON manifest and `PLANNED_RUNS.csv`.\n",
        encoding="utf-8",
    )
    return manifest


def execute_pair(project_root: Path, output_root: Path, dataset: str, seed: int) -> list[dict]:
    results: list[dict] = []
    for method in METHODS:
        row = {"dataset": dataset, "method": method, "seed": seed}
        command = job_command(project_root, output_root, row)
        started = time.time()
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
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--previous-corrected-manifest", required=True)
    parser.add_argument("--max-workers", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else project_root / "experiments" / "reproducibility" / "results" / DEFAULT_OUTPUT_NAME
    manifest = prepare_manifest(
        project_root,
        output_root,
        Path(args.previous_corrected_manifest).resolve(),
        args.max_workers,
    )
    planned = pd.read_csv(output_root / "PLANNED_RUNS.csv")
    print(planned.to_string(index=False))
    if args.prepare_only:
        return 0

    started = time.time()
    write_json(output_root / "LAUNCHER_STATUS.json", {"status": "RUNNING", "planned": 20, "completed": 0, "failed": 0})
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(execute_pair, project_root, output_root, dataset, seed): (dataset, seed)
            for dataset in ("limo", "mnoxide")
            for seed in range(10, 15)
        }
        for future in as_completed(futures):
            try:
                results.extend(future.result())
            except Exception as error:
                dataset, seed = futures[future]
                results.append(
                    {
                        "dataset": dataset,
                        "method": "pair_launcher_exception",
                        "seed": seed,
                        "return_code": -1,
                        "elapsed_seconds": 0.0,
                        "stdout": "",
                        "stderr": str(error),
                        "command": "",
                    }
                )
            write_json(
                output_root / "LAUNCHER_STATUS.json",
                {
                    "status": "RUNNING",
                    "planned": 20,
                    "attempted": len(results),
                    "completed": sum(row["return_code"] == 0 for row in results),
                    "failed": sum(row["return_code"] != 0 for row in results),
                    "last_update_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
    result_frame = pd.DataFrame(results).sort_values(["dataset", "seed", "method"])
    result_frame.to_csv(output_root / "LAUNCH_RESULTS.csv", index=False)
    failures = result_frame[result_frame["return_code"] != 0]
    failures.to_csv(output_root / "failed_runs.csv", index=False)
    completion = {
        "status": "DONE" if len(result_frame) == 20 and failures.empty else "COMPLETED_WITH_FAILURES",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "ended_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "planned": int(manifest["trajectory_count"]),
        "completed": int((result_frame["return_code"] == 0).sum()),
        "failed": int(len(failures)),
    }
    write_json(output_root / "LAUNCHER_STATUS.json", completion)
    write_json(output_root / "COMPLETION.json", completion)
    return 0 if completion["failed"] == 0 and completion["completed"] == 20 else 1


if __name__ == "__main__":
    raise SystemExit(main())
