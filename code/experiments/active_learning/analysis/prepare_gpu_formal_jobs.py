"""Generate the three approved formal GPU job manifests from frozen protocols."""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from experiments.reproducibility.formal_protocol import FormalProtocol, load_formal_protocol


def _validate_protocol(
    protocol: FormalProtocol, *, dataset: str, phase: str, methods: set[str]
) -> None:
    if protocol.dataset != dataset or protocol.phase != phase or not protocol.frozen:
        raise ValueError(f"unexpected frozen protocol scope: {protocol.source_path}")
    if not methods.issubset(protocol.allowed_methods):
        raise ValueError(f"protocol does not allow required methods: {protocol.source_path}")


def _job_row(
    *,
    project_root: PurePosixPath,
    output_path: PurePosixPath,
    log_path: PurePosixPath,
    job_id: str,
    dataset: str,
    method: str,
    group_key: str,
    seed: int,
    protocol: FormalProtocol,
    git_commit: str,
    formal_stage: str,
) -> dict[str, object]:
    runner = project_root / "experiments/reproducibility/run_paired_dataset_job.py"
    command = [
        "/root/miniconda3/bin/python",
        str(runner),
        "--project-root",
        str(project_root),
        "--dataset",
        dataset,
        "--method",
        method,
        "--seed",
        str(seed),
        "--run-dir",
        str(output_path),
        "--protocol-config",
        protocol.source_path.as_posix(),
    ]
    return {
        "job_id": job_id,
        "dataset": dataset,
        "method": method,
        "group_key": group_key,
        "seed": seed,
        "K": protocol.mc_passes,
        "config_hash": protocol.sha256,
        "git_commit": git_commit,
        "gpu_id": 0,
        "status": "PENDING",
        "start_time": "",
        "end_time": "",
        "exit_code": "",
        "log_path": str(log_path),
        "output_path": str(output_path),
        "sha256": "",
        "command_json": json.dumps(command, separators=(",", ":")),
        "cwd": str(project_root),
        "attempt": 1,
        "pid": "",
        "failure_reason": "",
        "formal_stage": formal_stage,
        "protocol_path": protocol.source_path.as_posix(),
    }


def _write_manifest(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def build_formal_job_manifests(
    *,
    project_root: str,
    output_root: str,
    manifest_directory: Path,
    git_commit: str,
    limo_protocol: Path,
    mn_original_protocol: Path,
    mn_block_protocol: Path,
    mn_iupac_protocol: Path,
    mc_protocols: dict[int, Path],
) -> dict[str, pd.DataFrame]:
    project = PurePosixPath(project_root)
    output = PurePosixPath(output_root)
    limo = load_formal_protocol(limo_protocol)
    mn_original = load_formal_protocol(mn_original_protocol)
    mn_block = load_formal_protocol(mn_block_protocol)
    mn_iupac = load_formal_protocol(mn_iupac_protocol)
    mc = {int(k): load_formal_protocol(path) for k, path in mc_protocols.items()}

    limo_methods = {
        "interval_hit_greedy",
        "always_da_tpp",
        "margin_only_gate",
        "group_only_gate",
        "energy_gated_da_tpp",
    }
    _validate_protocol(limo, dataset="limo", phase="formal_evaluation", methods=limo_methods)
    _validate_protocol(
        mn_original,
        dataset="mnoxide",
        phase="formal_evaluation",
        methods={"interval_hit_greedy", "always_da_tpp", "energy_gated_da_tpp"},
    )
    _validate_protocol(
        mn_block,
        dataset="mnoxide",
        phase="formal_evaluation",
        methods={"energy_gated_da_tpp"},
    )
    _validate_protocol(
        mn_iupac,
        dataset="mnoxide",
        phase="formal_evaluation",
        methods={"energy_gated_da_tpp"},
    )
    if set(mc) != {3, 10, 30}:
        raise ValueError("MC sensitivity protocols must contain K=3, 10, and 30")
    for k, protocol in mc.items():
        _validate_protocol(
            protocol,
            dataset="limo",
            phase="mc_dropout_sensitivity",
            methods={"interval_hit_greedy", "energy_gated_da_tpp"},
        )
        if protocol.mc_passes != k:
            raise ValueError(f"MC protocol K mismatch: {protocol.source_path}")

    limo_rows: list[dict[str, object]] = []
    limo_root = output / "results/final/li_m_o_ablation"
    for method in sorted(limo_methods):
        for seed in range(15, 25):
            job_id = f"gpu_final_limo_{method}_seed{seed:02d}"
            run = limo_root / method / f"seed_{seed}" / "attempt_1"
            limo_rows.append(
                _job_row(
                    project_root=project,
                    output_path=run,
                    log_path=limo_root / "logs" / f"{job_id}.log",
                    job_id=job_id,
                    dataset="limo",
                    method=method,
                    group_key=limo.group_key_mode,
                    seed=seed,
                    protocol=limo,
                    git_commit=git_commit,
                    formal_stage="li_m_o_ablation",
                )
            )

    mn_specs = [
        ("original", "interval_hit_greedy", mn_original),
        ("original", "always_da_tpp", mn_original),
        ("original", "energy_gated_da_tpp", mn_original),
        ("block", "energy_gated_da_tpp", mn_block),
        ("iupac", "energy_gated_da_tpp", mn_iupac),
    ]
    mn_rows: list[dict[str, object]] = []
    mn_root = output / "results/final/mn_group_key"
    for group_token, method, protocol in mn_specs:
        for seed in range(15, 25):
            job_id = f"gpu_final_mn_{group_token}_{method}_seed{seed:02d}"
            run = mn_root / group_token / method / f"seed_{seed}" / "attempt_1"
            mn_rows.append(
                _job_row(
                    project_root=project,
                    output_path=run,
                    log_path=mn_root / "logs" / f"{job_id}.log",
                    job_id=job_id,
                    dataset="mnoxide",
                    method=method,
                    group_key=protocol.group_key_mode,
                    seed=seed,
                    protocol=protocol,
                    git_commit=git_commit,
                    formal_stage="mn_group_key",
                )
            )

    mc_rows: list[dict[str, object]] = []
    mc_root = output / "results/final/mc_dropout_sensitivity/li_m_o"
    for k in (3, 10, 30):
        protocol = mc[k]
        for method in ("interval_hit_greedy", "energy_gated_da_tpp"):
            for seed in range(25, 30):
                job_id = f"gpu_final_mc_limo_k{k}_{method}_seed{seed:02d}"
                run = mc_root / f"k_{k}" / method / f"seed_{seed}" / "attempt_1"
                mc_rows.append(
                    _job_row(
                        project_root=project,
                        output_path=run,
                        log_path=mc_root / "logs" / f"{job_id}.log",
                        job_id=job_id,
                        dataset="limo",
                        method=method,
                        group_key=protocol.group_key_mode,
                        seed=seed,
                        protocol=protocol,
                        git_commit=git_commit,
                        formal_stage="mc_dropout_sensitivity",
                    )
                )

    frames = {
        "li_m_o_ablation": pd.DataFrame(limo_rows),
        "mn_group_key": pd.DataFrame(mn_rows),
        "mc_dropout_sensitivity": pd.DataFrame(mc_rows),
    }
    combined = pd.concat(frames.values(), ignore_index=True)
    if len(combined) != 130:
        raise ValueError(f"approved formal grid must contain 130 jobs, found {len(combined)}")
    if not combined["job_id"].is_unique or not combined["output_path"].is_unique:
        raise ValueError("formal job IDs and output paths must be unique")
    frames["combined"] = combined

    names = {
        "li_m_o_ablation": "gpu_final_li_m_o_ablation.csv",
        "mn_group_key": "gpu_final_mn_group_key.csv",
        "mc_dropout_sensitivity": "gpu_final_mc_dropout_sensitivity.csv",
        "combined": "gpu_final_jobs_manifest.csv",
    }
    for name, frame in frames.items():
        _write_manifest(Path(manifest_directory) / names[name], frame)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--manifest-directory", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--limo-protocol", type=Path, required=True)
    parser.add_argument("--mn-original-protocol", type=Path, required=True)
    parser.add_argument("--mn-block-protocol", type=Path, required=True)
    parser.add_argument("--mn-iupac-protocol", type=Path, required=True)
    parser.add_argument("--mc-k3-protocol", type=Path, required=True)
    parser.add_argument("--mc-k10-protocol", type=Path, required=True)
    parser.add_argument("--mc-k30-protocol", type=Path, required=True)
    args = parser.parse_args()
    frames = build_formal_job_manifests(
        project_root=args.project_root,
        output_root=args.output_root,
        manifest_directory=args.manifest_directory,
        git_commit=args.git_commit,
        limo_protocol=args.limo_protocol,
        mn_original_protocol=args.mn_original_protocol,
        mn_block_protocol=args.mn_block_protocol,
        mn_iupac_protocol=args.mn_iupac_protocol,
        mc_protocols={
            3: args.mc_k3_protocol,
            10: args.mc_k10_protocol,
            30: args.mc_k30_protocol,
        },
    )
    print(json.dumps({name: len(frame) for name, frame in frames.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
