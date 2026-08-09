#!/usr/bin/env python3
"""Thin seeds 10-14 wrapper around the frozen corrected paired runner."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd

try:
    from . import run_paired_dataset_job as frozen_runner
except ImportError:
    import run_paired_dataset_job as frozen_runner


class DeviceGpuSampler:
    """Record device-level GPU telemetry without touching experiment RNG state."""

    def __init__(self, output_path: Path, interval_seconds: float = 5.0) -> None:
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.rows: list[dict] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-telemetry", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(10.0, self.interval_seconds * 2))
        frame = pd.DataFrame(self.rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=["sample_time", "gpu_utilization_percent", "memory_used_mib", "power_draw_w"]
            )
        frame.to_csv(self.output_path, index=False)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                ).strip().splitlines()[0]
                utilization, memory_used, power_draw = [float(value.strip()) for value in text.split(",")]
                self.rows.append(
                    {
                        "sample_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "gpu_utilization_percent": utilization,
                        "memory_used_mib": memory_used,
                        "power_draw_w": power_draw,
                    }
                )
            except Exception:
                self.rows.append(
                    {
                        "sample_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "gpu_utilization_percent": float("nan"),
                        "memory_used_mib": float("nan"),
                        "power_draw_w": float("nan"),
                    }
                )
            self.stop_event.wait(self.interval_seconds)


def add_runtime_gpu_summary(run_dir: Path) -> None:
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    telemetry = pd.read_csv(run_dir / "gpu_usage.csv")
    numeric = telemetry[["gpu_utilization_percent", "memory_used_mib", "power_draw_w"]].apply(
        pd.to_numeric, errors="coerce"
    )
    summary = {
        "telemetry_scope": "device-level shared GPU; concurrent paired jobs may contribute",
        "runtime_seconds": float(status.get("elapsed_seconds", 0.0)),
        "sample_count": int(len(telemetry)),
        "mean_gpu_utilization_percent": float(numeric["gpu_utilization_percent"].mean()),
        "max_gpu_utilization_percent": float(numeric["gpu_utilization_percent"].max()),
        "mean_memory_used_mib": float(numeric["memory_used_mib"].mean()),
        "max_memory_used_mib": float(numeric["memory_used_mib"].max()),
        "mean_power_draw_w": float(numeric["power_draw_w"].mean()),
        "max_power_draw_w": float(numeric["power_draw_w"].max()),
    }
    (run_dir / "gpu_usage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metrics_path = run_dir / "run_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    for key, value in summary.items():
        metrics[key] = value
    metrics.to_csv(metrics_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=sorted(frozen_runner.dataset_configs()), required=True)
    parser.add_argument("--method", choices=sorted(frozen_runner.METHOD_SPECS), required=True)
    parser.add_argument("--seed", type=int, choices=range(10, 15), required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    sampler = DeviceGpuSampler(run_dir / "gpu_usage.csv")
    sampler.start()
    try:
        frozen_runner.run_job(args)
    finally:
        sampler.stop()
    if (run_dir / "run_metrics.csv").exists() and (run_dir / "status.json").exists():
        add_runtime_gpu_summary(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
