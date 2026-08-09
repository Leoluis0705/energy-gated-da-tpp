#!/usr/bin/env python3
"""Supervise C120/C214 relaxations and launch only dependency-gated statics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from analysis.dft_finalization_runner import build_queue_manifest
from analysis.postprocess_dft_verification_relaxations import analyze_verification_relaxations
from analysis.prepare_dft_finalization_inputs import build_candidate_static_inputs


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def require_relaxations_done(frame: pd.DataFrame) -> None:
    if len(frame) != 4 or frame["job_id"].astype(str).duplicated().any():
        raise ValueError("relaxation queue must contain exactly four unique jobs")
    if not frame["status"].astype(str).eq("DONE").all() or not pd.to_numeric(
        frame["exit_code"], errors="raise"
    ).eq(0).all():
        raise ValueError("relaxation jobs are not all DONE with exit code 0")


def normalize_relaxation_input_paths(
    frame: pd.DataFrame, manifest_path: Path
) -> pd.DataFrame:
    """Resolve copied server inputs by job ID instead of stale client absolute paths."""

    normalized = frame.copy(deep=True)
    inputs = Path(manifest_path).resolve().parent / "inputs"
    resolved: list[str] = []
    for job_id in normalized["job_id"].astype(str):
        directory = (inputs / job_id).resolve()
        missing = [
            str(directory / name)
            for name in ("initial.POSCAR", "initial.cif")
            if not (directory / name).is_file()
        ]
        if missing:
            raise FileNotFoundError("missing copied relaxation input evidence: " + "; ".join(missing))
        resolved.append(str(directory))
    normalized["input_dir"] = resolved
    return normalized


def stage_directory(root: Path, name: str, prefix: str) -> Path:
    candidate = Path(name)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != name
        or not name.startswith(prefix)
    ):
        raise ValueError(f"unsafe stage directory: {name}")
    return Path(root) / name


def assemble_server_potcar(
    components: Sequence[tuple[str, Path, str]],
    output: Path,
) -> dict[str, Any]:
    """Concatenate licensed server-side PAW files and return payload-free provenance."""

    if [element for element, _, _ in components] != ["Li", "Cr", "O"]:
        raise ValueError("candidate POTCAR element order must be Li, Cr, O")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    total = 0
    with output.open("xb") as target:
        for element, source, expected_label in components:
            path = Path(source)
            data = path.read_bytes()
            text = data[:8192].decode("latin-1", errors="replace")
            title = next(
                (
                    line.split("=", 1)[1].strip()
                    for line in text.splitlines()
                    if "TITEL" in line and "=" in line
                ),
                None,
            )
            if title is None or expected_label not in title:
                raise ValueError(f"unexpected PAW label for {element}: {title!r}")
            target.write(data)
            digest.update(data)
            total += len(data)
            rows.append(
                {
                    "element": element,
                    "source_path": str(path),
                    "TITEL": title,
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return {
        "created_at_utc": _utc_now(),
        "element_order": ["Li", "Cr", "O"],
        "components": rows,
        "combined_server_path": str(output),
        "combined_size_bytes": total,
        "combined_sha256": digest.hexdigest(),
        "payload_transfer": "not_transferred",
        "retention_policy": "deleted when frozen-static controller exits",
    }


def _pid_alive(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (ValueError, OSError):
        return False
    return True


def _write_heartbeat(directory: Path, frame: pd.DataFrame, *, phase: str) -> None:
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    _write_json_exclusive(
        directory / f"heartbeat_{token}.json",
        {
            "timestamp": _utc_now(),
            "phase": phase,
            "status_counts": frame["status"].astype(str).value_counts().sort_index().to_dict(),
            "free_disk_bytes": shutil.disk_usage(directory).free,
        },
    )


def _write_analysis_outputs(
    input_manifest: pd.DataFrame,
    queue_manifest: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    output.mkdir(parents=True, exist_ok=False)
    metrics, convergence, review, dependencies = analyze_verification_relaxations(
        input_manifest,
        queue_manifest,
        cif_root=output / "final_cifs",
    )
    metrics.to_csv(output / "structure_metrics.csv", index=False, lineterminator="\n")
    convergence.to_csv(output / "convergence_inventory.csv", index=False, lineterminator="\n")
    _write_json_exclusive(output / "structural_review.json", review)
    _write_json_exclusive(output / "dependency_results.json", dependencies)
    return metrics, review, dependencies


def run_supervisor(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    supervisor = stage_directory(
        run_root, args.supervisor_directory_name, "candidate_pipeline_supervisor"
    )
    supervisor.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        supervisor / "supervisor_config.json",
        {
            "started_at_utc": _utc_now(),
            "git_commit": args.git_commit,
            "relaxation_runner_git_commit": args.relaxation_runner_git_commit,
            "concurrency": args.concurrency,
            "openblas_threads": args.openblas_threads,
            "poll_seconds": args.poll_seconds,
            "heartbeat_seconds": args.heartbeat_seconds,
            "gpu_rerun": False,
            "alpha_formal_tasks": "not_authorized",
        },
    )
    queue_path = Path(args.relaxation_queue_manifest).resolve()
    last_heartbeat = 0.0
    while True:
        frame = pd.read_csv(queue_path, dtype=str, keep_default_na=False)
        unfinished = frame["status"].astype(str).isin(["PENDING", "RUNNING"]).any()
        if not unfinished:
            break
        if frame["status"].astype(str).eq("RUNNING").any() and not _pid_alive(
            Path(args.relaxation_controller_pid_file)
        ):
            raise RuntimeError("relaxation controller exited while jobs remained RUNNING")
        now = time.monotonic()
        if now - last_heartbeat >= float(args.heartbeat_seconds):
            _write_heartbeat(supervisor, frame, phase="waiting_for_relaxations")
            last_heartbeat = now
        time.sleep(float(args.poll_seconds))

    require_relaxations_done(frame)
    inputs = normalize_relaxation_input_paths(
        pd.read_csv(args.relaxation_input_manifest, dtype=str, keep_default_na=False),
        Path(args.relaxation_input_manifest),
    )
    analysis_root = stage_directory(
        run_root / "candidate_relaxation",
        args.relaxation_postprocess_directory_name,
        "postprocess_attempt_",
    )
    metrics, review, dependencies = _write_analysis_outputs(inputs, frame, analysis_root)
    if not review["static_launch_authorized"]:
        _write_json_exclusive(
            supervisor / "final_status.json",
            {
                "status": "PAUSED_BY_STRUCTURAL_REVIEW",
                "ended_at_utc": _utc_now(),
                "pause_reasons": review["pause_reasons"],
                "static_jobs_launched": 0,
            },
        )
        return 2

    full_manifest = pd.read_csv(args.full_manifest, dtype=str, keep_default_na=False)
    static_root = run_root / "candidate_static"
    static_inputs = build_candidate_static_inputs(
        full_manifest,
        dependency_results=dependencies,
        structural_review=analysis_root / "structural_review.json",
        relaxation_metrics=metrics,
        output_root=static_root,
        git_commit=args.git_commit,
    )
    potcar = Path(args.temporary_potcar).resolve()
    components = [
        ("Li", Path(args.li_potcar), "PAW_PBE Li_sv"),
        ("Cr", Path(args.cr_potcar), "PAW_PBE Cr_pv"),
        ("O", Path(args.o_potcar), "PAW_PBE O"),
    ]
    provenance = assemble_server_potcar(components, potcar)
    _write_json_exclusive(static_root / "POTCAR_provenance.json", provenance)
    static_queue = build_queue_manifest(
        static_inputs,
        output_root=static_root / "outputs",
        log_root=static_root / "logs",
        runner_path=Path(args.vasp_task_runner),
        python_executable=args.python_executable,
        vasp_executable=args.vasp_executable,
        potcar_source=potcar,
        openblas_threads=args.openblas_threads,
    )
    static_queue_path = static_root / "jobs_manifest.csv"
    static_queue.to_csv(static_queue_path, index=False, lineterminator="\n")
    queue_command = [
        args.python_executable,
        args.queue_runner,
        "--manifest",
        str(static_queue_path),
        "--audit-dir",
        str(static_root / "audit"),
        "--output-root",
        str(static_root / "outputs"),
        "--concurrency",
        str(args.concurrency),
        "--resource-kind",
        "cpu",
        "--sample-interval-seconds",
        "5",
        "--progress-interval-seconds",
        str(args.heartbeat_seconds),
        "--minimum-free-bytes",
        str(10 * 1024**3),
        "--maximum-root-bytes",
        str(10 * 1024**3),
        "--baseline-throughput-per-hour",
        "0.1",
        "--throughput-floor-fraction",
        "0.70",
    ]
    _write_json_exclusive(
        static_root / "controller_command.json",
        {
            "argv": queue_command,
            "git_commit": args.git_commit,
            "relaxation_runner_git_commit": args.relaxation_runner_git_commit,
        },
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(args.relaxation_code_root).resolve())
    try:
        with (static_root / "controller.log").open("xb") as log:
            completed = subprocess.run(
                queue_command,
                cwd=Path(args.relaxation_code_root),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
    finally:
        potcar.unlink(missing_ok=True)
    final_static = pd.read_csv(static_queue_path, dtype=str, keep_default_na=False)
    if completed.returncode != 0:
        raise RuntimeError(f"frozen-static controller exited {completed.returncode}")
    require_relaxations_done(final_static)
    _write_json_exclusive(
        supervisor / "final_status.json",
        {
            "status": "STATIC_TASKS_DONE_PENDING_POSTPROCESSING",
            "ended_at_utc": _utc_now(),
            "relaxation_jobs_done": 4,
            "static_jobs_done": 4,
            "temporary_potcar_retained": potcar.exists(),
        },
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--full-manifest", required=True)
    parser.add_argument("--relaxation-input-manifest", required=True)
    parser.add_argument("--relaxation-queue-manifest", required=True)
    parser.add_argument("--relaxation-controller-pid-file", required=True)
    parser.add_argument("--relaxation-code-root", required=True)
    parser.add_argument("--queue-runner", required=True)
    parser.add_argument("--vasp-task-runner", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--vasp-executable", required=True)
    parser.add_argument("--li-potcar", required=True)
    parser.add_argument("--cr-potcar", required=True)
    parser.add_argument("--o-potcar", required=True)
    parser.add_argument("--temporary-potcar", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--relaxation-runner-git-commit", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--openblas-threads", type=int, default=8)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--supervisor-directory-name", default="candidate_pipeline_supervisor"
    )
    parser.add_argument(
        "--relaxation-postprocess-directory-name", default="postprocess_attempt_1"
    )
    args = parser.parse_args()
    if args.concurrency != 4 or args.openblas_threads != 8:
        raise ValueError("approved DFT concurrency is four tasks with eight OpenBLAS threads each")
    try:
        return run_supervisor(args)
    except Exception as error:
        directory = stage_directory(
            Path(args.run_root).resolve(),
            args.supervisor_directory_name,
            "candidate_pipeline_supervisor",
        )
        if directory.is_dir() and not (directory / "failure.json").exists():
            _write_json_exclusive(
                directory / "failure.json",
                {
                    "status": "FAILED",
                    "ended_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
