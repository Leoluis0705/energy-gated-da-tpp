from pathlib import Path
import json

import pandas as pd
import pytest

from analysis import analyze_concurrency_benchmarks as analysis


def _valid_modes(throughput: dict[int, float]) -> list[dict]:
    return [
        {
            "concurrency": concurrency,
            "trajectories_per_hour": value,
            "failed_tasks": 0,
            "oom_detected": False,
            "cuda_error_detected": False,
            "reproducible": True,
            "memory_headroom_fraction": 0.5,
            "oversubscribed": False,
        }
        for concurrency, value in throughput.items()
    ]


def test_choose_concurrency_requires_five_percent_gain_and_valid_outputs():
    rows = _valid_modes({1: 4.0, 2: 7.8, 3: 8.0})
    assert analysis.choose_concurrency(rows) == 2


def test_choose_concurrency_rejects_mode_with_scientific_mismatch():
    rows = _valid_modes({1: 4.0, 2: 7.5})
    rows[1]["reproducible"] = False
    assert analysis.choose_concurrency(rows) == 1


def test_compare_dft_energies_rejects_difference_above_tolerance():
    with pytest.raises(ValueError, match="energy mismatch"):
        analysis.compare_dft_energies([-3.0, -3.0 + 2e-8], tolerance_ev=1e-8)


def _write_gpu_run(root: Path, prediction_offset: float = 0.0) -> None:
    root.mkdir()
    pd.DataFrame([{"seed": 0, "AUTC": 0.5, "final_recovery": 20}]).to_csv(root / "run_metrics.csv", index=False)
    pd.DataFrame(
        [{"round": 1, "selected_candidate_ids": "a;b", "checkpoint_path": str(root / "checkpoint"), "prediction_path": str(root / "prediction")}]
    ).to_csv(root / "summary.csv", index=False)
    pd.DataFrame([{"id": "a", "iteration": 1, "target_label": 1}]).to_csv(root / "al_history.csv", index=False)
    pd.DataFrame([{"round": 1, "route": "diversity_aware"}]).to_csv(root / "round_diagnostics.csv", index=False)
    pd.DataFrame([{"iteration": 1, "mode": "diversity_aware"}]).to_csv(root / "mode_trace_egdatpp_psfix_v1.csv", index=False)
    pd.DataFrame([["a", 0.0, 1.0 + prediction_offset]]).to_csv(root / "test_results_iter_1.csv", index=False, header=False)


def test_compare_gpu_runs_ignores_paths_but_detects_prediction_change(tmp_path):
    reference = tmp_path / "reference"
    same = tmp_path / "same"
    changed = tmp_path / "changed"
    _write_gpu_run(reference)
    _write_gpu_run(same)
    _write_gpu_run(changed, prediction_offset=1e-5)

    assert analysis.compare_gpu_runs(reference, same)["reproducible"] is True
    mismatch = analysis.compare_gpu_runs(reference, changed, numeric_tolerance=1e-10)
    assert mismatch["reproducible"] is False
    assert "test_results_iter_1.csv" in mismatch["mismatched_files"]


def test_estimate_wall_hours_uses_measured_throughput_and_scaling():
    result = analysis.estimate_wall_hours(30, measured_trajectories_per_hour=6.0, scaling_factor=1.2)
    assert result == pytest.approx(6.0)


def _write_mode(root: Path, name: str, concurrency: int, throughput: float, elapsed: float, gpu: bool) -> None:
    monitor = root / name / "monitor"
    monitor.mkdir(parents=True)
    summary = {
        "requested_tasks": concurrency,
        "completed_tasks": concurrency,
        "failed_tasks": 0,
        "mode_elapsed_seconds": elapsed,
        "trajectories_per_hour": throughput,
        "average_cpu_percent": 10.0 * concurrency,
        "peak_cpu_percent": 20.0 * concurrency,
        "peak_system_memory_used_bytes": 1024**3 * concurrency,
        "minimum_system_memory_available_bytes": 8 * 1024**3,
        "peak_process_rss_bytes": 512 * 1024**2 * concurrency,
        "peak_process_threads": 9 * concurrency,
        "process_read_bytes_during_mode": 1024,
        "process_write_bytes_during_mode": 2048,
        "disk_read_bytes_during_mode": 4096,
        "disk_write_bytes_during_mode": 8192,
        "average_gpu_utilization_percent": 2.0 * concurrency if gpu else None,
        "peak_gpu_utilization_percent": 20.0 * concurrency if gpu else None,
        "peak_gpu_memory_used_mib": 700.0 * concurrency if gpu else None,
        "oom_detected": False,
        "cuda_error_detected": False,
    }
    (monitor / "benchmark_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    pd.DataFrame(
        [{"task_id": f"task_{i}", "status": "DONE", "exit_code": 0, "elapsed_seconds": elapsed - i * 0.1} for i in range(concurrency)]
    ).to_csv(monitor / "task_results.csv", index=False)


def test_generate_reports_contains_recommendations_estimates_and_four_day_answer(tmp_path):
    gpu = tmp_path / "gpu"
    dft = tmp_path / "dft"
    for concurrency, throughput in ((1, 4.0), (2, 7.8), (3, 11.0)):
        _write_mode(gpu, f"gpu_c{concurrency}", concurrency, throughput, 900.0, gpu=True)
    (gpu / "gpu_reproducibility_checks.json").write_text(
        json.dumps({"check": {"reproducible": True, "compared_files": 45, "missing_files": [], "mismatched_files": [], "numeric_tolerance": 1e-10}}),
        encoding="utf-8",
    )
    (gpu / "gpu_c3_manifest.json").write_text(json.dumps({"git_commit": "gpu123"}), encoding="utf-8")
    for concurrency, throughput in ((1, 340.0), (2, 690.0), (4, 1380.0)):
        _write_mode(dft, f"dft_c{concurrency}", concurrency, throughput, 10.5, gpu=False)
    (dft / "energy_consistency.json").write_text(
        json.dumps({"count": 7, "energy_ev": -3.8, "spread_ev": 0.0, "tolerance_ev": 1e-8, "consistent": True}),
        encoding="utf-8",
    )
    (dft / "dft_c4_manifest.json").write_text(json.dumps({"git_commit": "dft123"}), encoding="utf-8")
    output = tmp_path / "docs"

    paths = analysis.generate_reports(gpu, dft, output, git_commit="abc123")

    assert {path.name for path in paths} == {
        "GPU_CONCURRENCY_BENCHMARK.md",
        "DFT_CONCURRENCY_BENCHMARK.md",
        "UPDATED_WALLTIME_ESTIMATE.md",
    }
    gpu_text = (output / "GPU_CONCURRENCY_BENCHMARK.md").read_text(encoding="utf-8")
    dft_text = (output / "DFT_CONCURRENCY_BENCHMARK.md").read_text(encoding="utf-8")
    estimate = (output / "UPDATED_WALLTIME_ESTIMATE.md").read_text(encoding="utf-8")
    assert "推荐 GPU 并发数：**3**" in gpu_text
    assert "gpu123" in gpu_text and "abc123" in gpu_text
    assert "推荐 DFT 并发数：**4**" in dft_text
    assert "dft123" in dft_text and "abc123" in dft_text
    assert "完整 mandatory GPU" in estimate
    assert "连续开启服务器 4 天：**否**" in estimate
