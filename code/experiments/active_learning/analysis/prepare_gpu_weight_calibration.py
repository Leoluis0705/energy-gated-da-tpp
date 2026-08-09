"""Generate the preregistered seven-job local weight-calibration screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

import pandas as pd


METHOD = "energy_gated_da_tpp"
DEVELOPMENT_SEEDS = tuple(range(5))
WEIGHT_VARIANTS: dict[str, tuple[float, float, float]] = {
    "baseline": (0.10, 0.20, 0.10),
    "alpha_0p05": (0.05, 0.20, 0.10),
    "alpha_0p20": (0.20, 0.20, 0.10),
    "beta_0p10": (0.10, 0.10, 0.10),
    "beta_0p40": (0.10, 0.40, 0.10),
    "gamma_0p05": (0.10, 0.20, 0.05),
    "gamma_0p20": (0.10, 0.20, 0.20),
}


def weight_protocol_payloads(
    *, mc_passes: int, m0: float, g0: float
) -> list[dict[str, object]]:
    if mc_passes not in {3, 10, 30}:
        raise ValueError("mc_passes must be one of 3, 10, 30")
    common: dict[str, object] = {
        "protocol_version": "egdatpp_psfix_v1",
        "phase": "weight_calibration",
        "dataset": "limo",
        "allowed_seeds": list(DEVELOPMENT_SEEDS),
        "allowed_methods": [METHOD],
        "mc_passes": mc_passes,
        "M0": float(m0),
        "G0": float(g0),
        "group_key_mode": "element_system_current",
        "group_key_map_relative_path": None,
        "frozen": False,
    }
    return [
        {
            **common,
            "variant_id": variant_id,
            "alpha": weights[0],
            "beta": weights[1],
            "gamma": weights[2],
        }
        for variant_id, weights in WEIGHT_VARIANTS.items()
    ]


def _write_config(path: Path, payload: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def build_weight_screen_bundle(
    *,
    project_root: str,
    output_root: str,
    local_configs_root: Path,
    remote_configs_root: str,
    manifest_path: Path,
    git_commit: str,
    mc_passes: int,
    m0: float,
    g0: float,
) -> pd.DataFrame:
    remote_project = PurePosixPath(project_root)
    remote_output = PurePosixPath(output_root)
    remote_configs = PurePosixPath(remote_configs_root)
    runner = remote_project / "experiments/reproducibility/run_paired_dataset_job.py"
    rows: list[dict[str, object]] = []
    for payload in weight_protocol_payloads(mc_passes=mc_passes, m0=m0, g0=g0):
        variant_id = str(payload["variant_id"])
        config_name = f"weight_{variant_id}.yaml"
        protocol_payload = {key: value for key, value in payload.items() if key != "variant_id"}
        config_hash = _write_config(Path(local_configs_root) / config_name, protocol_payload)
        remote_config = remote_configs / config_name
        job_id = f"gpu_cal_weight_{variant_id}_seed00"
        run_dir = remote_output / variant_id / "seed_0" / "attempt_1"
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
                "M0": float(m0),
                "G0": float(g0),
                "alpha": float(payload["alpha"]),
                "beta": float(payload["beta"]),
                "gamma": float(payload["gamma"]),
                "variant_id": variant_id,
                "calibration_stage": "weight_seed0_screen",
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
    parser.add_argument("--m0", type=float, required=True)
    parser.add_argument("--g0", type=float, required=True)
    args = parser.parse_args()
    frame = build_weight_screen_bundle(
        project_root=args.project_root,
        output_root=args.output_root,
        local_configs_root=args.local_configs_root,
        remote_configs_root=args.remote_configs_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
        mc_passes=args.mc_passes,
        m0=args.m0,
        g0=args.g0,
    )
    print(json.dumps({"jobs": len(frame), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
