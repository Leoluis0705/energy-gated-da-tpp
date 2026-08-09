"""Run a collision-safe CSV job queue with resource and stop-gate auditing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.run_concurrency_benchmark import (
    _cpu_percent,
    _query_gpu,
    _read_cpu_times,
    _read_disk_bytes,
    _read_memory,
    _read_process_tree,
    sha256_tree,
)


ALLOWED_STATUSES = {"PENDING", "RUNNING", "DONE", "FAILED", "CANCELLED"}
REQUIRED_COLUMNS = {
    "job_id",
    "config_hash",
    "status",
    "start_time",
    "end_time",
    "exit_code",
    "log_path",
    "output_path",
    "sha256",
    "command_json",
    "cwd",
    "attempt",
    "pid",
    "failure_reason",
}


class QueueConfigurationError(ValueError):
    """Raised when a queue manifest cannot be executed safely."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_jobs(frame: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise QueueConfigurationError(f"manifest is missing columns: {sorted(missing)}")
    if frame.empty:
        raise QueueConfigurationError("manifest has no jobs")
    for column in ("job_id", "log_path", "output_path"):
        if not frame[column].astype(str).is_unique:
            raise QueueConfigurationError(f"manifest requires unique {column}")
    invalid = set(frame["status"].astype(str)).difference(ALLOWED_STATUSES)
    if invalid:
        raise QueueConfigurationError(f"invalid statuses: {sorted(invalid)}")
    for record in frame.itertuples(index=False):
        try:
            command = json.loads(str(record.command_json))
        except json.JSONDecodeError as error:
            raise QueueConfigurationError(f"invalid command_json for {record.job_id}") from error
        if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
            raise QueueConfigurationError(f"command_json for {record.job_id} must be a non-empty string list")
        if not str(record.config_hash):
            raise QueueConfigurationError(f"config_hash is empty for {record.job_id}")
        if "env_json" in frame.columns and str(record.env_json):
            try:
                environment = json.loads(str(record.env_json))
            except json.JSONDecodeError as error:
                raise QueueConfigurationError(f"invalid env_json for {record.job_id}") from error
            if not isinstance(environment, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in environment.items()
            ):
                raise QueueConfigurationError(f"env_json for {record.job_id} must be a string mapping")


def _pid_is_alive(value: object) -> bool:
    try:
        pid = int(str(value))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _run_config_matches_manifest(run_config: dict[str, Any], expected_hash: str) -> bool:
    if str(run_config.get("formal_protocol_sha256", "")) == expected_hash:
        return True
    protocol_path = Path(str(run_config.get("formal_protocol_path", "")))
    task_path = Path(str(run_config.get("interval_task_path", "")))
    if not protocol_path.is_file() or not task_path.is_file():
        return False
    combined_hash = hashlib.sha256(
        protocol_path.read_bytes() + task_path.read_bytes()
    ).hexdigest()
    return combined_hash == expected_hash


def reconcile_resume_state(frame: pd.DataFrame, *, resource_kind: str) -> pd.DataFrame:
    result = frame.copy()
    if resource_kind == "gpu":
        stale_failures = result["status"].eq("FAILED") & result["failure_reason"].eq(
            "stale_running_process_missing"
        )
        for index in result.index[stale_failures]:
            output = Path(str(result.at[index, "output_path"]))
            status_path = output / "status.json"
            config_path = output / "run_config.json"
            if not status_path.is_file() or not config_path.is_file():
                continue
            status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
            if status != "DONE" or not _run_config_matches_manifest(
                run_config, str(result.at[index, "config_hash"])
            ):
                continue
            result.at[index, "status"] = "DONE"
            result.at[index, "sha256"] = sha256_tree(output) or ""
            result.at[index, "failure_reason"] = (
                "reconciled_verified_done_output_after_controller_crash;"
                "process_exit_code_unavailable"
            )
    for index in result.index[result["status"].eq("DONE")]:
        if resource_kind != "gpu":
            continue
        output = Path(str(result.at[index, "output_path"]))
        status_path = output / "status.json"
        config_path = output / "run_config.json"
        if not status_path.is_file() or not config_path.is_file():
            raise QueueConfigurationError(f"DONE evidence is incomplete for {result.at[index, 'job_id']}")
        status = json.loads(status_path.read_text(encoding="utf-8")).get("status")
        run_config = json.loads(config_path.read_text(encoding="utf-8"))
        if status != "DONE" or not _run_config_matches_manifest(
            run_config, str(result.at[index, "config_hash"])
        ):
            raise QueueConfigurationError(f"DONE config hash mismatch for {result.at[index, 'job_id']}")
    for index in result.index[result["status"].eq("RUNNING")]:
        if _pid_is_alive(result.at[index, "pid"]):
            raise QueueConfigurationError(
                f"active RUNNING job requires its original controller: {result.at[index, 'job_id']}"
            )
        result.at[index, "status"] = "FAILED"
        result.at[index, "end_time"] = utc_now()
        result.at[index, "failure_reason"] = "stale_running_process_missing"
    return result


def evaluate_safety(
    *,
    free_bytes: int,
    minimum_free_bytes: int,
    root_bytes: int,
    maximum_root_bytes: int,
    completed_jobs: int,
    throughput_per_hour: float | None,
    baseline_throughput_per_hour: float,
    throughput_floor_fraction: float,
    error_flags: set[str],
) -> str | None:
    if error_flags:
        return "numerical_or_accelerator_error:" + ",".join(sorted(error_flags))
    if int(free_bytes) < int(minimum_free_bytes):
        return "free_disk_below_limit"
    if int(root_bytes) > int(maximum_root_bytes):
        return "output_root_exceeds_budget"
    if (
        int(completed_jobs) >= 3
        and throughput_per_hour is not None
        and float(throughput_per_hour) < float(baseline_throughput_per_hour) * float(throughput_floor_fraction)
    ):
        return "throughput_below_floor"
    return None


def cohort_throughput(
    *,
    now: float,
    window_started: float,
    completed_jobs: int,
    window_completed_jobs: int,
    cohort_size: int,
) -> float | None:
    newly_completed = int(completed_jobs) - int(window_completed_jobs)
    if newly_completed < int(cohort_size):
        return None
    return newly_completed * 3600.0 / max(float(now) - float(window_started), 1e-9)


def _atomic_write_manifest(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def _append_csv(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False, lineterminator="\n")


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except FileNotFoundError:
            continue
    return total


def _error_flags(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    text = path.read_text(encoding="utf-8", errors="ignore").lower()
    flags = set()
    if "out of memory" in text or "cuda oom" in text:
        flags.add("oom")
    if "cuda error" in text or "cublas_status" in text:
        flags.add("cuda")
    if re.search(r"\bnan\b", text):
        flags.add("nan")
    return flags


def _job_error_flags(log_path: Path, output_path: Path) -> set[str]:
    flags = _error_flags(log_path)
    for name in ("run.log", "traceback.txt", "error.log"):
        flags.update(_error_flags(output_path / name))
    return flags


def _write_progress(
    *,
    audit_dir: Path,
    frame: pd.DataFrame,
    started: float,
    initial_completed_jobs: int,
    peak_running_jobs: int,
    root_bytes: int,
    free_bytes: int,
    pause_reason: str | None,
    resource_snapshot: dict[str, Any],
) -> dict[str, Any]:
    now = time.monotonic()
    counts = Counter(frame["status"].astype(str))
    completed = int(counts.get("DONE", 0))
    session_completed = max(0, completed - int(initial_completed_jobs))
    elapsed = max(now - started, 1e-9)
    throughput = session_completed * 3600.0 / elapsed if session_completed else None
    remaining = int(counts.get("PENDING", 0) + counts.get("RUNNING", 0))
    eta = remaining / throughput if throughput and throughput > 0 else None
    summary = {
        "timestamp": utc_now(),
        "status_counts": {status: int(counts.get(status, 0)) for status in sorted(ALLOWED_STATUSES)},
        "throughput_jobs_per_hour": throughput,
        "remaining_wall_hours": eta,
        "output_root_bytes": int(root_bytes),
        "free_disk_bytes": int(free_bytes),
        "peak_running_jobs": int(peak_running_jobs),
        "pause_reason": pause_reason,
        "resource_snapshot": resource_snapshot,
        "failed_jobs": frame.loc[frame["status"].eq("FAILED"), ["job_id", "failure_reason"]].to_dict("records"),
    }
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    with (audit_dir / f"progress_{token}.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary


def run_queue(
    *,
    manifest_path: Path,
    audit_dir: Path,
    output_root: Path,
    concurrency: int,
    resource_kind: str,
    sample_interval_seconds: float,
    progress_interval_seconds: float,
    minimum_free_bytes: int,
    maximum_root_bytes: int,
    baseline_throughput_per_hour: float,
    throughput_floor_fraction: float,
) -> dict[str, Any]:
    if int(concurrency) < 1:
        raise QueueConfigurationError("concurrency must be positive")
    manifest_path = Path(manifest_path).resolve()
    audit_dir = Path(audit_dir).resolve()
    output_root = Path(output_root).resolve()
    frame = pd.read_csv(manifest_path, keep_default_na=False)
    validate_jobs(frame)
    frame = reconcile_resume_state(frame, resource_kind=resource_kind)
    _atomic_write_manifest(frame, manifest_path)
    audit_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_completed = int(frame["status"].eq("DONE").sum())
    throughput_window_started = started
    throughput_window_completed = initial_completed
    previous_cpu = _read_cpu_times()
    previous_disk = _read_disk_bytes()
    next_progress = started
    next_disk_check = started
    root_bytes = _tree_size(output_root)
    free_bytes = shutil.disk_usage(output_root).free
    running: dict[int, tuple[subprocess.Popen[Any], Any]] = {}
    peak_running = 0
    error_flags: set[str] = set()
    pause_reason: str | None = None
    failure_counts: Counter[str] = Counter(
        frame.loc[frame["status"].eq("FAILED"), "config_hash"].astype(str)
    )
    last_summary: dict[str, Any] | None = None
    latest_resource: dict[str, Any] = {}

    while True:
        for index, (process, log_handle) in list(running.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            log_handle.close()
            output = Path(str(frame.at[index, "output_path"]))
            status_file = output / "status.json"
            declared_status = None
            if status_file.is_file():
                try:
                    declared_status = json.loads(status_file.read_text(encoding="utf-8")).get("status")
                except (json.JSONDecodeError, OSError):
                    declared_status = None
            success = int(exit_code) == 0 and declared_status in (None, "DONE")
            frame.at[index, "status"] = "DONE" if success else "FAILED"
            frame.at[index, "end_time"] = utc_now()
            frame.at[index, "exit_code"] = str(int(exit_code))
            frame.at[index, "sha256"] = sha256_tree(output) or ""
            if not success:
                reason = f"exit_code={exit_code}; declared_status={declared_status}"
                frame.at[index, "failure_reason"] = reason
                failure_counts[str(frame.at[index, "config_hash"])] += 1
            error_flags.update(
                _job_error_flags(
                    Path(str(frame.at[index, "log_path"])),
                    output,
                )
            )
            del running[index]
            _atomic_write_manifest(frame, manifest_path)

        if any(count >= 3 for count in failure_counts.values()):
            pause_reason = "three_same_config_failures"

        now = time.monotonic()
        for index in running:
            error_flags.update(
                _job_error_flags(
                    Path(str(frame.at[index, "log_path"])),
                    Path(str(frame.at[index, "output_path"])),
                )
            )
        completed = int(frame["status"].eq("DONE").sum())
        completed_in_throughput_window = completed - throughput_window_completed
        throughput = cohort_throughput(
            now=now,
            window_started=throughput_window_started,
            completed_jobs=completed,
            window_completed_jobs=throughput_window_completed,
            cohort_size=concurrency,
        )
        if now >= next_disk_check:
            root_bytes = _tree_size(output_root)
            free_bytes = shutil.disk_usage(output_root).free
            pause_reason = pause_reason or evaluate_safety(
                free_bytes=free_bytes,
                minimum_free_bytes=minimum_free_bytes,
                root_bytes=root_bytes,
                maximum_root_bytes=maximum_root_bytes,
                completed_jobs=completed_in_throughput_window,
                throughput_per_hour=throughput,
                baseline_throughput_per_hour=baseline_throughput_per_hour,
                throughput_floor_fraction=throughput_floor_fraction,
                error_flags=error_flags,
            )
            if throughput is not None:
                throughput_window_started = now
                throughput_window_completed = completed
            next_disk_check = now + min(300.0, max(sample_interval_seconds, 0.01) * 10.0)

        if pause_reason is None:
            available = int(concurrency) - len(running)
            pending = list(frame.index[frame["status"].eq("PENDING")])[: max(0, available)]
            for index in pending:
                command = json.loads(str(frame.at[index, "command_json"]))
                cwd = Path(str(frame.at[index, "cwd"]))
                log_path = Path(str(frame.at[index, "log_path"]))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("x", encoding="utf-8")
                env = os.environ.copy()
                if "env_json" in frame.columns and str(frame.at[index, "env_json"]):
                    env.update(json.loads(str(frame.at[index, "env_json"])))
                if resource_kind == "gpu":
                    env.update(
                        {
                            "CUDA_VISIBLE_DEVICES": str(frame.at[index, "gpu_id"]),
                            "OMP_NUM_THREADS": "8",
                            "MKL_NUM_THREADS": "8",
                            "OPENBLAS_NUM_THREADS": "8",
                            "NUMEXPR_NUM_THREADS": "8",
                            "PYTHONUNBUFFERED": "1",
                        }
                    )
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name == "posix",
                )
                running[index] = (process, log_handle)
                frame.at[index, "status"] = "RUNNING"
                frame.at[index, "start_time"] = utc_now()
                frame.at[index, "pid"] = str(int(process.pid))
                _atomic_write_manifest(frame, manifest_path)
            peak_running = max(peak_running, len(running))

        current_cpu = _read_cpu_times()
        current_disk = _read_disk_bytes()
        process_stats = _read_process_tree(process.pid for process, _ in running.values())
        memory_total, memory_available = _read_memory()
        sample: dict[str, Any] = {
            "timestamp": utc_now(),
            "running_jobs": len(running),
            "cpu_percent": _cpu_percent(previous_cpu, current_cpu),
            "memory_used_bytes": memory_total - memory_available,
            "memory_available_bytes": memory_available,
            "process_rss_bytes": process_stats["rss_bytes"],
            "process_threads": process_stats["threads"],
            "disk_read_delta_bytes": max(0, current_disk[0] - previous_disk[0]),
            "disk_write_delta_bytes": max(0, current_disk[1] - previous_disk[1]),
            "output_root_bytes": root_bytes,
            "free_disk_bytes": free_bytes,
        }
        if resource_kind == "gpu":
            sample.update(_query_gpu())
        _append_csv(audit_dir / "resource_samples.csv", sample)
        latest_resource = sample
        previous_cpu = current_cpu
        previous_disk = current_disk

        if now >= next_progress or pause_reason is not None:
            last_summary = _write_progress(
                audit_dir=audit_dir,
                frame=frame,
                started=started,
                initial_completed_jobs=initial_completed,
                peak_running_jobs=peak_running,
                root_bytes=root_bytes,
                free_bytes=free_bytes,
                pause_reason=pause_reason,
                resource_snapshot=latest_resource,
            )
            next_progress = now + float(progress_interval_seconds)

        unfinished = frame["status"].isin(["PENDING", "RUNNING"]).any()
        if not unfinished or (pause_reason is not None and not running):
            break
        time.sleep(float(sample_interval_seconds))

    last_summary = _write_progress(
        audit_dir=audit_dir,
        frame=frame,
        started=started,
        initial_completed_jobs=initial_completed,
        peak_running_jobs=peak_running,
        root_bytes=_tree_size(output_root),
        free_bytes=shutil.disk_usage(output_root).free,
        pause_reason=pause_reason,
        resource_snapshot=latest_resource,
    )
    last_summary["controller_elapsed_seconds"] = time.monotonic() - started
    last_summary["resource_kind"] = resource_kind
    with (audit_dir / "controller_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(last_summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return last_summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--resource-kind", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=10.0)
    parser.add_argument("--progress-interval-seconds", type=float, default=7200.0)
    parser.add_argument("--minimum-free-bytes", type=int, default=10 * 1024**3)
    parser.add_argument("--maximum-root-bytes", type=int, required=True)
    parser.add_argument("--baseline-throughput-per-hour", type=float, required=True)
    parser.add_argument("--throughput-floor-fraction", type=float, default=0.70)
    args = parser.parse_args()
    summary = run_queue(
        manifest_path=args.manifest,
        audit_dir=args.audit_dir,
        output_root=args.output_root,
        concurrency=args.concurrency,
        resource_kind=args.resource_kind,
        sample_interval_seconds=args.sample_interval_seconds,
        progress_interval_seconds=args.progress_interval_seconds,
        minimum_free_bytes=args.minimum_free_bytes,
        maximum_root_bytes=args.maximum_root_bytes,
        baseline_throughput_per_hour=args.baseline_throughput_per_hour,
        throughput_floor_fraction=args.throughput_floor_fraction,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if summary.get("pause_reason") else 0


if __name__ == "__main__":
    raise SystemExit(main())
