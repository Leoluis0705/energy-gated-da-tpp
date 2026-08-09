"""Generate the preregistered 30-job GPU MC-dropout development bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd


METHODS = ("interval_hit_greedy", "energy_gated_da_tpp")
K_VALUES = (3, 10, 30)
SEEDS = tuple(range(5))


def protocol_payloads() -> list[dict[str, object]]:
    common: dict[str, object] = {
        "protocol_version": "egdatpp_psfix_v1",
        "phase": "mc_dropout_development",
        "dataset": "limo",
        "allowed_seeds": list(SEEDS),
        "allowed_methods": list(METHODS),
        "M0": 1.0,
        "G0": 0.50,
        "alpha": 0.10,
        "beta": 0.20,
        "gamma": 0.10,
        "group_key_mode": "element_system_current",
        "group_key_map_relative_path": None,
        "frozen": False,
    }
    return [{**common, "mc_passes": value} for value in K_VALUES]


def _write_json_yaml_exclusive(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def build_development_bundle(
    *,
    project_root: str,
    output_root: str,
    local_configs_root: Path,
    remote_configs_root: str,
    manifest_path: Path,
    git_commit: str,
) -> pd.DataFrame:
    local_configs_root = Path(local_configs_root)
    manifest_path = Path(manifest_path)
    remote_project = PurePosixPath(project_root)
    remote_output = PurePosixPath(output_root)
    remote_configs = PurePosixPath(remote_configs_root)
    config_records: dict[int, tuple[PurePosixPath, str]] = {}
    for payload in protocol_payloads():
        k = int(payload["mc_passes"])
        name = f"mc_k{k}.yaml"
        digest = _write_json_yaml_exclusive(local_configs_root / name, payload)
        config_records[k] = (remote_configs / name, digest)

    rows: list[dict[str, object]] = []
    runner = remote_project / "experiments/reproducibility/run_paired_dataset_job.py"
    python = PurePosixPath("/root/miniconda3/bin/python")
    for k in K_VALUES:
        config_path, config_hash = config_records[k]
        for method in METHODS:
            for seed in SEEDS:
                job_id = f"gpu_dev_k{k}_{method}_seed{seed:02d}"
                attempt_root = remote_output / f"k_{k}" / method / f"seed_{seed}" / "attempt_1"
                log_path = remote_output / "logs" / f"{job_id}.log"
                command = [
                    str(python),
                    str(runner),
                    "--project-root",
                    str(remote_project),
                    "--dataset",
                    "limo",
                    "--method",
                    method,
                    "--seed",
                    str(seed),
                    "--run-dir",
                    str(attempt_root),
                    "--protocol-config",
                    str(config_path),
                ]
                rows.append(
                    {
                        "job_id": job_id,
                        "dataset": "limo",
                        "method": method,
                        "group_key": "element_system_current",
                        "seed": seed,
                        "K": k,
                        "config_hash": config_hash,
                        "git_commit": git_commit,
                        "gpu_id": 0,
                        "status": "PENDING",
                        "start_time": "",
                        "end_time": "",
                        "exit_code": "",
                        "log_path": str(log_path),
                        "output_path": str(attempt_root),
                        "sha256": "",
                        "command_json": json.dumps(command, separators=(",", ":")),
                        "cwd": str(remote_project),
                        "attempt": 1,
                        "pid": "",
                        "failure_reason": "",
                    }
                )
    frame = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--local-configs-root", type=Path, required=True)
    parser.add_argument("--remote-configs-root", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    args = parser.parse_args()
    frame = build_development_bundle(
        project_root=args.project_root,
        output_root=args.output_root,
        local_configs_root=args.local_configs_root,
        remote_configs_root=args.remote_configs_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
    )
    print(json.dumps({"jobs": len(frame), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

