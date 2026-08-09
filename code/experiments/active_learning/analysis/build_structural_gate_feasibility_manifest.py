"""Build the frozen 50-job structural-group feasibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Sequence

import pandas as pd


METHOD_PROTOCOLS = {
    "predicted_target_greedy": "legacy_protocol.json",
    "energy_gated_da_tpp": "legacy_protocol.json",
    "structural_group_gate": "structural_protocol.json",
    "structural_group_gate_q95": "structural_q95_protocol.json",
    "gradient_norm_hybrid": "legacy_protocol.json",
}
TASK_CONFIGS = {"mn": "mn_task.json", "mg": "mg_task.json"}
SEEDS = tuple(range(111, 116))


def _config_hash(protocol: Path | PurePosixPath, task: Path | PurePosixPath) -> str:
    digest = hashlib.sha256()
    for path in (protocol, task):
        local = Path(path)
        digest.update(local.read_bytes() if local.is_file() else str(path).encode("utf-8"))
    return digest.hexdigest()


def build_manifest_rows(
    *,
    project_root: str,
    execution_root: str,
    python: str,
    gpu_ids: Sequence[int],
    config_source_root: Path | None = None,
) -> list[dict[str, str]]:
    gpu_tokens = tuple(str(int(value)) for value in gpu_ids)
    if not gpu_tokens or len(gpu_tokens) != len(set(gpu_tokens)):
        raise ValueError("GPU IDs must be non-empty and unique")
    project = PurePosixPath(str(project_root).replace("\\", "/"))
    execution = PurePosixPath(str(execution_root).replace("\\", "/"))
    config_dir = project / "configs" / "structural_group_feasibility"
    rows: list[dict[str, str]] = []
    for task, task_name in TASK_CONFIGS.items():
        task_config = config_dir / task_name
        for method, protocol_name in METHOD_PROTOCOLS.items():
            protocol = config_dir / protocol_name
            hash_protocol = (
                Path(config_source_root) / protocol_name
                if config_source_root is not None
                else protocol
            )
            hash_task = (
                Path(config_source_root) / task_name
                if config_source_root is not None
                else task_config
            )
            for seed in SEEDS:
                job_id = f"{task}__{method}__seed_{seed}"
                output = (
                    execution
                    / "results"
                    / "structural_group_feasibility_v1"
                    / task
                    / method
                    / f"seed_{seed}"
                )
                log = (
                    execution
                    / "logs"
                    / "structural_group_feasibility_v1"
                    / f"{job_id}.controller.log"
                )
                command = [
                    str(python),
                    (project / "experiments/reproducibility/run_paired_dataset_job.py").as_posix(),
                    "--project-root",
                    project.as_posix(),
                    "--dataset",
                    "limo",
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--run-dir",
                    output.as_posix(),
                    "--protocol-config",
                    protocol.as_posix(),
                    "--task-config",
                    task_config.as_posix(),
                ]
                rows.append(
                    {
                        "job_id": job_id,
                        "task": task,
                        "dataset": "limo",
                        "method": method,
                        "seed": str(seed),
                        "gpu_id": gpu_tokens[len(rows) % len(gpu_tokens)],
                        "config_hash": _config_hash(hash_protocol, hash_task),
                        "status": "PENDING",
                        "start_time": "",
                        "end_time": "",
                        "exit_code": "",
                        "log_path": log.as_posix(),
                        "output_path": output.as_posix(),
                        "sha256": "",
                        "command_json": json.dumps(command, separators=(",", ":")),
                        "cwd": project.as_posix(),
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
    parser.add_argument("--gpu-ids", nargs="+", type=int, required=True)
    parser.add_argument("--config-source-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build_manifest_rows(
        project_root=args.project_root,
        execution_root=args.execution_root,
        python=args.python,
        gpu_ids=args.gpu_ids,
        config_source_root=args.config_source_root,
    )
    frame = pd.DataFrame(rows)
    if len(frame) != 50 or frame["job_id"].duplicated().any() or frame["output_path"].duplicated().any():
        raise RuntimeError("manifest grid is incomplete or contains collisions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite manifest: {args.output}")
    frame.to_csv(args.output, index=False, lineterminator="\n")
    print(f"WROTE {len(frame)} jobs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
