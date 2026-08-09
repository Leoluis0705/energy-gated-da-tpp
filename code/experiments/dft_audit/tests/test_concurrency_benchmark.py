import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from analysis import run_concurrency_benchmark as benchmark


def _task(tmp_path: Path, task_id: str, output_name: str, exit_code: int = 0) -> dict:
    output_dir = tmp_path / output_name
    code = (
        "from pathlib import Path; "
        f"p=Path({str(output_dir)!r}); p.mkdir(parents=True); "
        "(p/'value.txt').write_text('deterministic\\n'); "
        f"raise SystemExit({exit_code})"
    )
    return {
        "task_id": task_id,
        "experiment_seed": int(task_id.rsplit("_", 1)[-1]),
        "command": [sys.executable, "-c", code],
        "cwd": str(tmp_path),
        "env": {},
        "log_path": str(tmp_path / f"{task_id}.log"),
        "output_dir": str(output_dir),
    }


def _manifest(tmp_path: Path, tasks: list[dict]) -> dict:
    return {
        "benchmark_id": "synthetic_c2",
        "resource_kind": "cpu",
        "sample_interval_seconds": 0.05,
        "tasks": tasks,
    }


def test_summarize_mode_uses_completion_window_for_throughput():
    summary = benchmark.summarize_mode([10.0, 12.0], mode_elapsed_seconds=12.0)
    assert summary["completed_tasks"] == 2
    assert summary["trajectories_per_hour"] == pytest.approx(600.0)


def test_validate_manifest_rejects_duplicate_output_directories(tmp_path):
    first = _task(tmp_path, "task_0", "same")
    second = _task(tmp_path, "task_1", "other")
    second["output_dir"] = first["output_dir"]
    with pytest.raises(ValueError, match="unique output_dir"):
        benchmark.validate_manifest(_manifest(tmp_path, [first, second]))


def test_validate_manifest_rejects_duplicate_gpu_experiment_seeds(tmp_path):
    first = _task(tmp_path, "task_0", "one")
    second = _task(tmp_path, "task_1", "two")
    second["experiment_seed"] = first["experiment_seed"]
    manifest = _manifest(tmp_path, [first, second])
    manifest["resource_kind"] = "gpu"
    with pytest.raises(ValueError, match="distinct experiment_seed"):
        benchmark.validate_manifest(manifest)


def test_run_benchmark_writes_samples_results_summary_and_hashes(tmp_path):
    root = tmp_path / "benchmark"
    manifest = _manifest(tmp_path, [_task(tmp_path, "task_0", "run0")])
    result = benchmark.run_benchmark(manifest, root)

    assert result["failed_tasks"] == 0
    assert result["completed_tasks"] == 1
    assert "process_read_bytes_during_mode" in result
    assert "process_write_bytes_during_mode" in result
    assert (root / "resource_samples.csv").is_file()
    assert (root / "task_results.csv").is_file()
    assert (root / "benchmark_summary.json").is_file()
    assert (root / "command_manifest.snapshot.json").is_file()
    assert (root / "sha256sums.csv").is_file()
    rows = pd.read_csv(root / "task_results.csv")
    assert rows.loc[0, "exit_code"] == 0
    assert rows.loc[0, "status"] == "DONE"
    assert json.loads((root / "benchmark_summary.json").read_text())["completed_tasks"] == 1


def test_run_benchmark_retains_failed_task_log_and_exit_code(tmp_path):
    root = tmp_path / "benchmark"
    manifest = _manifest(tmp_path, [_task(tmp_path, "task_0", "run0", exit_code=7)])
    result = benchmark.run_benchmark(manifest, root)

    assert result["failed_tasks"] == 1
    rows = pd.read_csv(root / "task_results.csv")
    assert rows.loc[0, "exit_code"] == 7
    assert rows.loc[0, "status"] == "FAILED"
    assert Path(rows.loc[0, "log_path"]).is_file()


def test_read_memory_prefers_cgroup_limit_and_current_usage(tmp_path, monkeypatch):
    (tmp_path / "memory.max").write_text(str(8 * 1024**3), encoding="utf-8")
    (tmp_path / "memory.current").write_text(str(2 * 1024**3), encoding="utf-8")
    monkeypatch.setattr(benchmark, "CGROUP_ROOT", tmp_path)

    total, available = benchmark._read_memory()

    assert total == 8 * 1024**3
    assert available == 6 * 1024**3


def test_read_cpu_times_normalizes_cgroup_usage_to_cpu_quota(tmp_path, monkeypatch):
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (tmp_path / "cpu.stat").write_text("usage_usec 100000\n", encoding="utf-8")
    monkeypatch.setattr(benchmark, "CGROUP_ROOT", tmp_path)
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: 10.0)
    first = benchmark._read_cpu_times()
    (tmp_path / "cpu.stat").write_text("usage_usec 200000\n", encoding="utf-8")
    monkeypatch.setattr(benchmark.time, "monotonic", lambda: 11.0)
    second = benchmark._read_cpu_times()

    assert benchmark._cpu_percent(first, second) == pytest.approx(5.0)
