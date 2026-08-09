"""Build collision-safe Mn/Mg CGCNN job manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METHODS = (
    "energy_gated_da_tpp",
    "predicted_target_greedy",
    "always_da_tpp",
    "group_only_gate",
    "margin_only_gate",
    "explore",
    "mc_dropout",
    "gradient_norm_hybrid",
    "random_sampling",
)
TASKS = (
    ("mn", "mn_interval_w0p2_gpu.json", "mn_interval_w0p2_gpu_smoke.json"),
    ("mg", "mg_interval_w0p2_gpu.json", "mg_interval_w0p2_gpu_smoke.json"),
)


def build_manifest_rows(
    *,
    project_root: str,
    execution_root: str,
    python: str,
    smoke: bool,
    protocol_sha256: str,
) -> list[dict[str, str]]:
    project = Path(project_root)
    execution = Path(execution_root)
    protocol = project / "configs/mn_mg_interval_gpu_protocol.json"
    tasks = TASKS[:1] if smoke else TASKS
    seeds = (101,) if smoke else tuple(range(101, 111))
    rows: list[dict[str, str]] = []
    for task_token, full_config, smoke_config in tasks:
        task_config = project / "configs" / (smoke_config if smoke else full_config)
        for method in METHODS:
            for seed in seeds:
                job_id = f"{task_token}__{method}__seed_{seed}"
                cohort = "smoke_grid" if smoke else "formal_w0p2"
                output_path = execution / "results" / cohort / task_token / method / f"seed_{seed}"
                log_path = execution / "logs" / cohort / f"{job_id}.controller.log"
                command = [
                    str(python),
                    str(project / "experiments/reproducibility/run_paired_dataset_job.py"),
                    "--project-root",
                    str(project),
                    "--dataset",
                    "limo",
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--run-dir",
                    str(output_path),
                    "--protocol-config",
                    str(protocol),
                    "--task-config",
                    str(task_config),
                ]
                rows.append(
                    {
                        "job_id": job_id,
                        "task": task_token,
                        "dataset": "limo",
                        "method": method,
                        "seed": str(seed),
                        "gpu_id": str(len(rows) % 2),
                        "config_hash": str(protocol_sha256),
                        "status": "PENDING",
                        "start_time": "",
                        "end_time": "",
                        "exit_code": "",
                        "log_path": str(log_path),
                        "output_path": str(output_path),
                        "sha256": "",
                        "command_json": json.dumps(command, separators=(",", ":")),
                        "cwd": str(project),
                        "attempt": "0",
                        "pid": "",
                        "failure_reason": "",
                        "env_json": "{}",
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--execution-root", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    rows = build_manifest_rows(
        project_root=args.project_root,
        execution_root=args.execution_root,
        python=args.python,
        smoke=args.smoke,
        protocol_sha256=args.protocol_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, lineterminator="\n")
    print(f"WROTE {len(rows)} jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

