"""Generate the preregistered nine-job threshold-calibration screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd


M0_VALUES = (0.75, 1.00, 1.25)
G0_VALUES = (0.40, 0.50, 0.60)
METHOD = "energy_gated_da_tpp"
DEVELOPMENT_SEEDS = tuple(range(5))


def _token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def protocol_payloads(*, mc_passes: int) -> list[dict[str, object]]:
    if mc_passes not in {3, 10, 30}:
        raise ValueError("mc_passes must be one of 3, 10, 30")
    common: dict[str, object] = {
        "protocol_version": "egdatpp_psfix_v1",
        "phase": "threshold_calibration",
        "dataset": "limo",
        "allowed_seeds": list(DEVELOPMENT_SEEDS),
        "allowed_methods": [METHOD],
        "mc_passes": mc_passes,
        "alpha": 0.10,
        "beta": 0.20,
        "gamma": 0.10,
        "group_key_mode": "element_system_current",
        "group_key_map_relative_path": None,
        "frozen": False,
    }
    return [{**common, "M0": m0, "G0": g0} for m0 in M0_VALUES for g0 in G0_VALUES]


def _write_config(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def build_threshold_screen_bundle(
    *,
    project_root: str,
    output_root: str,
    local_configs_root: Path,
    remote_configs_root: str,
    manifest_path: Path,
    git_commit: str,
    mc_passes: int,
) -> pd.DataFrame:
    remote_project = PurePosixPath(project_root)
    remote_output = PurePosixPath(output_root)
    remote_configs = PurePosixPath(remote_configs_root)
    runner = remote_project / "experiments/reproducibility/run_paired_dataset_job.py"
    rows: list[dict[str, object]] = []
    for payload in protocol_payloads(mc_passes=mc_passes):
        m0 = float(payload["M0"])
        g0 = float(payload["G0"])
        config_id = f"m0_{_token(m0)}_g0_{_token(g0)}"
        config_name = f"threshold_{config_id}.yaml"
        config_hash = _write_config(Path(local_configs_root) / config_name, payload)
        remote_config = remote_configs / config_name
        job_id = f"gpu_cal_threshold_{config_id}_seed00"
        run_dir = remote_output / config_id / "seed_0" / "attempt_1"
        command = [
            "/root/miniconda3/bin/python",
            str(runner),
            "--project-root",
            str(remote_project),
            "--dataset",
            "limo",
            "--method",
            METHOD,
            "--seed",
            "0",
            "--run-dir",
            str(run_dir),
            "--protocol-config",
            str(remote_config),
        ]
        rows.append(
            {
                "job_id": job_id,
                "dataset": "limo",
                "method": METHOD,
                "group_key": "element_system_current",
                "seed": 0,
                "K": mc_passes,
                "config_hash": config_hash,
                "git_commit": git_commit,
                "gpu_id": 0,
                "status": "PENDING",
                "start_time": "",
                "end_time": "",
                "exit_code": "",
                "log_path": str(remote_output / "logs" / f"{job_id}.log"),
                "output_path": str(run_dir),
                "sha256": "",
                "command_json": json.dumps(command, separators=(",", ":")),
                "cwd": str(remote_project),
                "attempt": 1,
                "pid": "",
                "failure_reason": "",
                "M0": m0,
                "G0": g0,
                "alpha": 0.10,
                "beta": 0.20,
                "gamma": 0.10,
                "calibration_stage": "threshold_seed0_screen",
            }
        )
    frame = pd.DataFrame(rows)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="") as handle:
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
    parser.add_argument("--mc-passes", type=int, required=True)
    args = parser.parse_args()
    frame = build_threshold_screen_bundle(
        project_root=args.project_root,
        output_root=args.output_root,
        local_configs_root=args.local_configs_root,
        remote_configs_root=args.remote_configs_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
        mc_passes=args.mc_passes,
    )
    print(json.dumps({"jobs": len(frame), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
