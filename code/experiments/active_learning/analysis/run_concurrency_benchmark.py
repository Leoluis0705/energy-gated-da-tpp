#!/usr/bin/env python3
"""Launch isolated commands and sample host/GPU resources without touching numerics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CGROUP_ROOT = Path("/sys/fs/cgroup")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str | None:
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted((item for item in root.rglob("*") if item.is_file())):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest must contain at least one task")
    interval = float(manifest.get("sample_interval_seconds", 0.0))
    if interval <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    required = {"task_id", "command", "cwd", "env", "log_path", "output_dir"}
    for task in tasks:
        missing = required - set(task)
        if missing:
            raise ValueError(f"task is missing fields: {sorted(missing)}")
        if not isinstance(task["command"], list) or not task["command"]:
            raise ValueError("task command must be a non-empty list")
    for field in ("task_id", "log_path", "output_dir"):
        values = [str(task[field]) for task in tasks]
        if len(values) != len(set(values)):
            raise ValueError(f"tasks require unique {field} values")
    if str(manifest.get("resource_kind", "cpu")) == "gpu":
        seeds = [task.get("experiment_seed") for task in tasks]
        if None in seeds or len(seeds) != len(set(seeds)):
            raise ValueError("concurrent GPU tasks require distinct experiment_seed values")


def summarize_mode(task_elapsed_seconds: Iterable[float], mode_elapsed_seconds: float) -> dict[str, float | int]:
    values = [float(value) for value in task_elapsed_seconds]
    elapsed = float(mode_elapsed_seconds)
    return {
        "completed_tasks": len(values),
        "mode_elapsed_seconds": elapsed,
        "mean_task_elapsed_seconds": sum(values) / len(values) if values else math.nan,
        "min_task_elapsed_seconds": min(values) if values else math.nan,
        "max_task_elapsed_seconds": max(values) if values else math.nan,
        "trajectories_per_hour": len(values) * 3600.0 / elapsed if elapsed > 0 else math.nan,
    }


def _read_cpu_times() -> tuple[int, int]:
    cpu_max = CGROUP_ROOT / "cpu.max"
    cpu_stat = CGROUP_ROOT / "cpu.stat"
    if cpu_max.is_file() and cpu_stat.is_file():
        quota_raw, period_raw = cpu_max.read_text(encoding="utf-8").split()[:2]
        if quota_raw != "max":
            quota_cores = int(quota_raw) / int(period_raw)
            usage = next(
                int(line.split()[1])
                for line in cpu_stat.read_text(encoding="utf-8").splitlines()
                if line.startswith("usage_usec ")
            )
            capacity = int(time.monotonic() * 1_000_000 * quota_cores)
            return capacity, capacity - usage
    proc_stat = Path("/proc/stat")
    if proc_stat.is_file():
        fields = proc_stat.read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle
    import psutil

    values = psutil.cpu_times()
    total = int(sum(values) * 1000)
    idle = int((values.idle + getattr(values, "iowait", 0.0)) * 1000)
    return total, idle


def _cpu_percent(previous: tuple[int, int], current: tuple[int, int]) -> float:
    total = current[0] - previous[0]
    idle = current[1] - previous[1]
    return 0.0 if total <= 0 else 100.0 * (total - idle) / total


def _read_memory() -> tuple[int, int]:
    memory_max = CGROUP_ROOT / "memory.max"
    memory_current = CGROUP_ROOT / "memory.current"
    if memory_max.is_file() and memory_current.is_file():
        maximum = memory_max.read_text(encoding="utf-8").strip()
        if maximum != "max":
            total = int(maximum)
            current = int(memory_current.read_text(encoding="utf-8").strip())
            return total, max(0, total - current)
    if not Path("/proc/meminfo").is_file():
        import psutil

        memory = psutil.virtual_memory()
        return int(memory.total), int(memory.available)
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    return values["MemTotal"], values["MemAvailable"]


def _read_disk_bytes() -> tuple[int, int]:
    read_sectors = 0
    write_sectors = 0
    sys_block = Path("/sys/block")
    if not sys_block.is_dir():
        import psutil

        counters = psutil.disk_io_counters()
        return (int(counters.read_bytes), int(counters.write_bytes)) if counters else (0, 0)
    for device in sys_block.iterdir():
        if device.name.startswith(("loop", "ram", "fd")):
            continue
        try:
            fields = (device / "stat").read_text(encoding="utf-8").split()
            read_sectors += int(fields[2])
            write_sectors += int(fields[6])
        except (FileNotFoundError, IndexError, ValueError):
            continue
    return read_sectors * 512, write_sectors * 512


def _read_process_tree(root_pids: Iterable[int]) -> dict[str, int]:
    process_data: dict[int, dict[str, int]] = {}
    proc = Path("/proc")
    if not proc.is_dir():
        import psutil

        selected = []
        for pid in root_pids:
            try:
                process = psutil.Process(int(pid))
                selected.extend([process, *process.children(recursive=True)])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        totals = {"rss_bytes": 0, "read_bytes": 0, "write_bytes": 0, "threads": 0}
        for process in {item.pid: item for item in selected}.values():
            try:
                totals["rss_bytes"] += int(process.memory_info().rss)
                totals["threads"] += int(process.num_threads())
                io = process.io_counters()
                totals["read_bytes"] += int(io.read_bytes)
                totals["write_bytes"] += int(io.write_bytes)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return totals
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status: dict[str, str] = {}
            for line in (entry / "status").read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    status[key] = value.strip()
            io_values: dict[str, int] = {}
            for line in (entry / "io").read_text(encoding="utf-8", errors="ignore").splitlines():
                key, value = line.split(":", 1)
                io_values[key] = int(value.strip())
            process_data[int(entry.name)] = {
                "ppid": int(status.get("PPid", "0")),
                "rss_bytes": int(status.get("VmRSS", "0 kB").split()[0]) * 1024,
                "threads": int(status.get("Threads", "0")),
                "read_bytes": io_values.get("read_bytes", 0),
                "write_bytes": io_values.get("write_bytes", 0),
            }
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    selected = {int(pid) for pid in root_pids}
    changed = True
    while changed:
        changed = False
        for pid, data in process_data.items():
            if pid not in selected and data["ppid"] in selected:
                selected.add(pid)
                changed = True
    totals = {"rss_bytes": 0, "read_bytes": 0, "write_bytes": 0, "threads": 0}
    for pid in selected:
        for key in totals:
            totals[key] += process_data.get(pid, {}).get(key, 0)
    return totals


def _query_gpu() -> dict[str, float | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        row = subprocess.run(command, check=True, capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
        fields = [value.strip() for value in row.split(",")]
        parsed: list[float | None] = []
        for value in fields:
            try:
                parsed.append(float(value))
            except ValueError:
                parsed.append(None)
        return {
            "gpu_utilization_percent": parsed[0],
            "gpu_memory_used_mib": parsed[1],
            "gpu_memory_total_mib": parsed[2],
            "gpu_power_w": parsed[3],
        }
    except (FileNotFoundError, IndexError, subprocess.SubprocessError):
        return {
            "gpu_utilization_percent": None,
            "gpu_memory_used_mib": None,
            "gpu_memory_total_mib": None,
            "gpu_power_w": None,
        }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _scan_error_flags(paths: Iterable[Path]) -> dict[str, bool]:
    text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in paths if path.is_file()).lower()
    return {
        "oom_detected": "out of memory" in text or "cuda oom" in text,
        "cuda_error_detected": "cuda error" in text or "cublas_status" in text,
    }


def _terminate_processes(processes: dict[str, subprocess.Popen[Any]]) -> None:
    for process in processes.values():
        if process.poll() is not None:
            continue
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(process.poll() is None for process in processes.values()):
        time.sleep(0.1)
    for process in processes.values():
        if process.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass


def run_benchmark(manifest: dict[str, Any], output_root: str | Path) -> dict[str, Any]:
    validate_manifest(manifest)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_json(root / "command_manifest.snapshot.json", manifest)
    resource_kind = str(manifest.get("resource_kind", "cpu"))
    interval = float(manifest["sample_interval_seconds"])
    processes: dict[str, subprocess.Popen[Any]] = {}
    log_handles: dict[str, Any] = {}
    task_state: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    mode_started_wall = time.time()
    mode_started = time.monotonic()
    previous_cpu = _read_cpu_times()
    previous_disk = _read_disk_bytes()
    try:
        for task in manifest["tasks"]:
            task_id = str(task["task_id"])
            output_dir = Path(task["output_dir"]).resolve()
            if output_dir.exists():
                raise FileExistsError(f"output_dir already exists: {output_dir}")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            log_path = Path(task["log_path"]).resolve()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("xb")
            environment = os.environ.copy()
            environment.update({str(key): str(value) for key, value in task.get("env", {}).items()})
            started = time.monotonic()
            process = subprocess.Popen(
                [str(value) for value in task["command"]],
                cwd=str(Path(task["cwd"]).resolve()),
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name == "posix"),
            )
            processes[task_id] = process
            log_handles[task_id] = log_handle
            task_state[task_id] = {
                "task_id": task_id,
                "experiment_seed": task.get("experiment_seed"),
                "pid": process.pid,
                "started_at": utc_now(),
                "started_monotonic": started,
                "ended_at": None,
                "ended_monotonic": None,
                "log_path": str(log_path),
                "output_dir": str(output_dir),
                "command": json.dumps(task["command"], ensure_ascii=False),
            }
        next_sample = time.monotonic()
        while any(process.poll() is None for process in processes.values()):
            now = time.monotonic()
            for task_id, process in processes.items():
                if process.poll() is not None and task_state[task_id]["ended_monotonic"] is None:
                    task_state[task_id]["ended_monotonic"] = now
                    task_state[task_id]["ended_at"] = utc_now()
            if now >= next_sample:
                current_cpu = _read_cpu_times()
                current_disk = _read_disk_bytes()
                memory_total, memory_available = _read_memory()
                process_totals = _read_process_tree(process.pid for process in processes.values())
                load1, load5, load15 = os.getloadavg() if hasattr(os, "getloadavg") else (math.nan,) * 3
                row: dict[str, Any] = {
                    "timestamp_utc": utc_now(),
                    "elapsed_seconds": now - mode_started,
                    "running_tasks": sum(process.poll() is None for process in processes.values()),
                    "cpu_percent": _cpu_percent(previous_cpu, current_cpu),
                    "load1": load1,
                    "load5": load5,
                    "load15": load15,
                    "memory_used_bytes": memory_total - memory_available,
                    "memory_available_bytes": memory_available,
                    "memory_percent": 100.0 * (memory_total - memory_available) / memory_total,
                    "disk_read_bytes_total": current_disk[0],
                    "disk_write_bytes_total": current_disk[1],
                    "disk_read_bytes_delta": current_disk[0] - previous_disk[0],
                    "disk_write_bytes_delta": current_disk[1] - previous_disk[1],
                    "process_rss_bytes": process_totals["rss_bytes"],
                    "process_read_bytes": process_totals["read_bytes"],
                    "process_write_bytes": process_totals["write_bytes"],
                    "process_threads": process_totals["threads"],
                }
                row.update(_query_gpu() if resource_kind == "gpu" else {
                    "gpu_utilization_percent": None,
                    "gpu_memory_used_mib": None,
                    "gpu_memory_total_mib": None,
                    "gpu_power_w": None,
                })
                samples.append(row)
                previous_cpu = current_cpu
                previous_disk = current_disk
                next_sample = now + interval
            time.sleep(min(0.1, interval / 2.0))
        finished = time.monotonic()
        for task_id, process in processes.items():
            if task_state[task_id]["ended_monotonic"] is None:
                task_state[task_id]["ended_monotonic"] = finished
                task_state[task_id]["ended_at"] = utc_now()
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for handle in log_handles.values():
            handle.close()

    task_rows: list[dict[str, Any]] = []
    for task_id, state in task_state.items():
        process = processes[task_id]
        log_path = Path(state["log_path"])
        output_dir = Path(state["output_dir"])
        task_rows.append({
            "task_id": task_id,
            "experiment_seed": state["experiment_seed"],
            "pid": state["pid"],
            "status": "DONE" if process.returncode == 0 else "FAILED",
            "exit_code": process.returncode,
            "started_at": state["started_at"],
            "ended_at": state["ended_at"],
            "elapsed_seconds": state["ended_monotonic"] - state["started_monotonic"],
            "log_path": str(log_path),
            "log_sha256": sha256_file(log_path),
            "output_dir": str(output_dir),
            "output_tree_sha256": sha256_tree(output_dir),
            "command": state["command"],
        })
    _write_csv(root / "resource_samples.csv", samples)
    _write_csv(root / "task_results.csv", task_rows)
    completed = [row for row in task_rows if row["status"] == "DONE"]
    summary: dict[str, Any] = {
        "benchmark_id": manifest["benchmark_id"],
        "resource_kind": resource_kind,
        "started_at": datetime.fromtimestamp(mode_started_wall, timezone.utc).isoformat(),
        "ended_at": utc_now(),
        **summarize_mode([row["elapsed_seconds"] for row in completed], time.monotonic() - mode_started),
        "requested_tasks": len(task_rows),
        "failed_tasks": len(task_rows) - len(completed),
        "average_cpu_percent": sum(float(row["cpu_percent"]) for row in samples) / len(samples),
        "peak_cpu_percent": max(float(row["cpu_percent"]) for row in samples),
        "peak_system_memory_used_bytes": max(int(row["memory_used_bytes"]) for row in samples),
        "minimum_system_memory_available_bytes": min(int(row["memory_available_bytes"]) for row in samples),
        "peak_process_rss_bytes": max(int(row["process_rss_bytes"]) for row in samples),
        "peak_process_threads": max(int(row["process_threads"]) for row in samples),
        "process_read_bytes_during_mode": max(int(row["process_read_bytes"]) for row in samples),
        "process_write_bytes_during_mode": max(int(row["process_write_bytes"]) for row in samples),
        "disk_read_bytes_during_mode": sum(max(0, int(row["disk_read_bytes_delta"])) for row in samples),
        "disk_write_bytes_during_mode": sum(max(0, int(row["disk_write_bytes_delta"])) for row in samples),
    }
    gpu_util = [float(row["gpu_utilization_percent"]) for row in samples if row["gpu_utilization_percent"] is not None]
    gpu_memory = [float(row["gpu_memory_used_mib"]) for row in samples if row["gpu_memory_used_mib"] is not None]
    summary.update({
        "average_gpu_utilization_percent": sum(gpu_util) / len(gpu_util) if gpu_util else None,
        "peak_gpu_utilization_percent": max(gpu_util) if gpu_util else None,
        "peak_gpu_memory_used_mib": max(gpu_memory) if gpu_memory else None,
        **_scan_error_flags(Path(row["log_path"]) for row in task_rows),
    })
    _write_json(root / "benchmark_summary.json", summary)
    inventory_rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "sha256sums.csv")
    ]
    _write_csv(root / "sha256sums.csv", inventory_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary = run_benchmark(manifest, args.output_root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
