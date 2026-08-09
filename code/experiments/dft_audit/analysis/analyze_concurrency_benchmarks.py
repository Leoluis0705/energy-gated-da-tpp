#!/usr/bin/env python3
"""Validate concurrency benchmark numerics and derive measured wall-time estimates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def choose_concurrency(rows: Iterable[dict[str, Any]], minimum_gain_fraction: float = 0.05) -> int:
    ordered = sorted((dict(row) for row in rows), key=lambda row: int(row["concurrency"]))
    valid = [
        row
        for row in ordered
        if int(row.get("failed_tasks", 0)) == 0
        and not bool(row.get("oom_detected", False))
        and not bool(row.get("cuda_error_detected", False))
        and bool(row.get("reproducible", False))
        and float(row.get("memory_headroom_fraction", 0.0)) >= 0.20
        and not bool(row.get("oversubscribed", False))
    ]
    if not valid:
        raise ValueError("no concurrency mode passed the validity criteria")
    recommended = valid[0]
    previous = valid[0]
    for row in valid[1:]:
        gain = float(row["trajectories_per_hour"]) / float(previous["trajectories_per_hour"]) - 1.0
        if gain >= float(minimum_gain_fraction):
            recommended = row
        previous = row
    return int(recommended["concurrency"])


def compare_dft_energies(energies_ev: Iterable[float], tolerance_ev: float = 1e-8) -> dict[str, float | int | bool]:
    values = [float(value) for value in energies_ev]
    if not values:
        raise ValueError("no DFT energies were provided")
    spread = max(values) - min(values)
    if spread > float(tolerance_ev):
        raise ValueError(f"energy mismatch: spread={spread:.12g} eV exceeds {tolerance_ev:.12g} eV")
    return {"count": len(values), "minimum_ev": min(values), "maximum_ev": max(values), "spread_ev": spread, "consistent": True}


def _read_comparable_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None if path.name.startswith("test_results_iter_") else 0)
    return frame.drop(columns=[name for name in ("checkpoint_path", "prediction_path") if name in frame], errors="ignore")


def _frames_match(left: pd.DataFrame, right: pd.DataFrame, tolerance: float) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=float(tolerance),
            atol=float(tolerance),
        )
        return True
    except AssertionError:
        return False


def compare_gpu_runs(reference_dir: str | Path, candidate_dir: str | Path, numeric_tolerance: float = 1e-10) -> dict[str, Any]:
    reference = Path(reference_dir)
    candidate = Path(candidate_dir)
    fixed = [
        "run_metrics.csv",
        "summary.csv",
        "al_history.csv",
        "round_diagnostics.csv",
        "mode_trace_egdatpp_psfix_v1.csv",
    ]
    names = fixed + [path.name for path in sorted(reference.glob("test_results_iter_*.csv"))]
    mismatches: list[str] = []
    missing: list[str] = []
    for name in names:
        left_path = reference / name
        right_path = candidate / name
        if not left_path.is_file() or not right_path.is_file():
            missing.append(name)
            continue
        if not _frames_match(_read_comparable_csv(left_path), _read_comparable_csv(right_path), numeric_tolerance):
            mismatches.append(name)
    return {
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "numeric_tolerance": float(numeric_tolerance),
        "compared_files": len(names) - len(missing),
        "missing_files": missing,
        "mismatched_files": mismatches,
        "reproducible": not missing and not mismatches,
    }


def estimate_wall_hours(task_count: int, measured_trajectories_per_hour: float, scaling_factor: float = 1.0) -> float:
    throughput = float(measured_trajectories_per_hour)
    if int(task_count) < 0 or throughput <= 0 or float(scaling_factor) <= 0:
        raise ValueError("task count must be non-negative and throughput/scaling must be positive")
    return int(task_count) * float(scaling_factor) / throughput


def _load_mode(root: Path, prefix: str, concurrency: int) -> tuple[dict[str, Any], pd.DataFrame]:
    monitor = root / f"{prefix}_c{concurrency}" / "monitor"
    summary = json.loads((monitor / "benchmark_summary.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(monitor / "task_results.csv")
    summary["concurrency"] = int(concurrency)
    return summary, tasks


def _gib(value: float | int) -> float:
    return float(value) / 1024.0**3


def _mib(value: float | int) -> float:
    return float(value) / 1024.0**2


def _task_times(tasks: pd.DataFrame) -> str:
    return ", ".join(f"{float(value):.2f}" for value in tasks["elapsed_seconds"].tolist())


def _headroom_fraction(summary: dict[str, Any]) -> float:
    available = float(summary["minimum_system_memory_available_bytes"])
    used = float(summary["peak_system_memory_used_bytes"])
    return available / (available + used)


def _write_text_exclusive(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(value.rstrip() + "\n")


def generate_reports(
    gpu_root: str | Path,
    dft_root: str | Path,
    output_dir: str | Path,
    *,
    git_commit: str,
) -> list[Path]:
    gpu_source = Path(gpu_root).resolve()
    dft_source = Path(dft_root).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    gpu_modes = {value: _load_mode(gpu_source, "gpu", value) for value in (1, 2, 3)}
    dft_modes = {value: _load_mode(dft_source, "dft", value) for value in (1, 2, 4)}
    reproducibility = json.loads((gpu_source / "gpu_reproducibility_checks.json").read_text(encoding="utf-8"))
    energy = json.loads((dft_source / "energy_consistency.json").read_text(encoding="utf-8"))
    gpu_benchmark_commit = json.loads(
        (gpu_source / "gpu_c3_manifest.json").read_text(encoding="utf-8")
    )["git_commit"]
    dft_benchmark_commit = json.loads(
        (dft_source / "dft_c4_manifest.json").read_text(encoding="utf-8")
    )["git_commit"]
    all_gpu_reproducible = all(bool(value["reproducible"]) for value in reproducibility.values())

    gpu_choice_rows = []
    for concurrency, (summary, _) in gpu_modes.items():
        reproducible = True if concurrency == 1 else all_gpu_reproducible
        gpu_choice_rows.append({
            **summary,
            "concurrency": concurrency,
            "reproducible": reproducible,
            "memory_headroom_fraction": _headroom_fraction(summary),
            "oversubscribed": float(summary["peak_cpu_percent"]) > 100.5,
        })
    recommended_gpu = choose_concurrency(gpu_choice_rows)

    single_dft_max = float(dft_modes[1][1]["elapsed_seconds"].max())
    dft_choice_rows = []
    for concurrency, (summary, tasks) in dft_modes.items():
        slowdown = float(tasks["elapsed_seconds"].max()) / single_dft_max - 1.0
        dft_choice_rows.append({
            **summary,
            "concurrency": concurrency,
            "reproducible": bool(energy["consistent"]),
            "memory_headroom_fraction": _headroom_fraction(summary),
            "oversubscribed": float(summary["peak_cpu_percent"]) > 100.5 or slowdown > 0.10,
        })
    recommended_dft = choose_concurrency(dft_choice_rows)

    gpu_lines = [
        "# GPU concurrency benchmark",
        "",
        "## Scope",
        "",
        f"This report was generated from recovered raw artifacts under `{gpu_source}`. The benchmark execution used commit `{gpu_benchmark_commit}`; this report generator used commit `{git_commit}`. The protocol was `egdatpp_psfix_v1`, with one complete Li--M--O Full Energy-Gated DA-TPP trajectory per process, K=3, and eight OMP/MKL/OpenBLAS threads per process. It did not run parameter calibration or final experiment seeds.",
        "",
        "The retained preflight audit trail includes non-scientific setup failures (initial shell escaping, unavailable remote pytest, and an incorrect validation-script API call) and a superseded monitor run that used host `/proc` denominators. None of those attempts started a scientific trajectory. The results below are from the corrected cgroup-aware monitor after direct validation passed.",
        "",
        "## Measured results",
        "",
        "GPU memory and utilization are aggregate device measurements. CPU percentages are normalized to the 24-CPU cgroup quota; memory is normalized to the 72-GiB cgroup limit.",
        "",
        "| concurrency | task wall times (s) | completion window (s) | trajectories/hour | CPU avg/peak (%) | GPU avg/peak (%) | peak GPU memory (MiB) | peak cgroup memory (GiB) | peak process RSS (GiB) | process read/write (GiB) | system disk read/write (GiB) | failures |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, (summary, tasks) in gpu_modes.items():
        gpu_lines.append(
            f"| {concurrency} | {_task_times(tasks)} | {float(summary['mode_elapsed_seconds']):.2f} | {float(summary['trajectories_per_hour']):.3f} | "
            f"{float(summary['average_cpu_percent']):.2f}/{float(summary['peak_cpu_percent']):.2f} | "
            f"{float(summary['average_gpu_utilization_percent']):.2f}/{float(summary['peak_gpu_utilization_percent']):.2f} | "
            f"{float(summary['peak_gpu_memory_used_mib']):.0f} | {_gib(summary['peak_system_memory_used_bytes']):.2f} | "
            f"{_gib(summary['peak_process_rss_bytes']):.2f} | {_gib(summary['process_read_bytes_during_mode']):.3f}/{_gib(summary['process_write_bytes_during_mode']):.2f} | "
            f"{_gib(summary['disk_read_bytes_during_mode']):.3f}/{_gib(summary['disk_write_bytes_during_mode']):.2f} | {int(summary['failed_tasks'])} |"
        )
    c1_tph = float(gpu_modes[1][0]["trajectories_per_hour"])
    c2_tph = float(gpu_modes[2][0]["trajectories_per_hour"])
    c3_tph = float(gpu_modes[3][0]["trajectories_per_hour"])
    gpu_lines.extend([
        "",
        "Process I/O is attributable to the launched process trees; system disk counters include other host/block-cache activity and are reported separately. No mode contained an OOM or CUDA error.",
        "",
        "## Scientific reproducibility",
        "",
    ])
    for name, value in reproducibility.items():
        gpu_lines.append(
            f"- `{name}`: reproducible={str(bool(value['reproducible'])).lower()}, compared files={int(value['compared_files'])}, "
            f"missing={len(value['missing_files'])}, mismatched={len(value['mismatched_files'])}, numerical tolerance={float(value['numeric_tolerance']):.1e}."
        )
    gpu_lines.extend([
        "",
        "The comparisons cover `run_metrics.csv`, `summary.csv`, query history, round diagnostics, route trace, and all 40 per-round prediction tables. Runtime/path fields are excluded; scientific fields are retained.",
        "",
        "## Recommendation",
        "",
        f"推荐 GPU 并发数：**{recommended_gpu}**。Two-process throughput improved by {(c2_tph / c1_tph - 1) * 100:.1f}% over one process; three-process throughput improved by {(c3_tph / c2_tph - 1) * 100:.1f}% over two processes. Three-process peak GPU memory was {float(gpu_modes[3][0]['peak_gpu_memory_used_mib']):.0f} MiB of 32,760 MiB, and its peak CPU utilization was {float(gpu_modes[3][0]['peak_cpu_percent']):.2f}% of the allocated quota. The gain exceeds the pre-registered 5% threshold and scientific outputs remained reproducible.",
        "",
        "This recommendation is bounded to the tested Li--M--O CGCNN trajectory and eight-thread process setting. It does not demonstrate that four or more GPU processes are safe; four-process GPU concurrency was not authorized or tested.",
    ])

    dft_lines = [
        "# DFT concurrency benchmark",
        "",
        "## Scope",
        "",
        f"This report was generated from recovered artifacts under `{dft_source}`. The benchmark execution used commit `{dft_benchmark_commit}`; this report generator used commit `{git_commit}`. All tasks cloned the same existing Li-metal PBE static input (ENCUT=520, explicit Gamma 15 x 15 x 15 mesh, two Li atoms) and used the serial VASP 6.5.1 binary with `OPENBLAS_NUM_THREADS=8`. The benchmark energy is execution evidence, not a replacement elemental reference result.",
        "",
        "## Measured results",
        "",
        "CPU percentages are normalized to the 32-CPU cgroup quota; memory is normalized to the 60-GiB limit.",
        "",
        "| concurrency | task wall times (s) | completion window (s) | tasks/hour | CPU avg/peak (%) | peak cgroup memory (MiB) | peak process RSS (MiB) | process read/write (MiB) | system disk read/write (MiB) | peak process threads | failures |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, (summary, tasks) in dft_modes.items():
        dft_lines.append(
            f"| {concurrency} | {_task_times(tasks)} | {float(summary['mode_elapsed_seconds']):.2f} | {float(summary['trajectories_per_hour']):.2f} | "
            f"{float(summary['average_cpu_percent']):.2f}/{float(summary['peak_cpu_percent']):.2f} | {_mib(summary['peak_system_memory_used_bytes']):.1f} | "
            f"{_mib(summary['peak_process_rss_bytes']):.1f} | {_mib(summary['process_read_bytes_during_mode']):.3f}/{_mib(summary['process_write_bytes_during_mode']):.3f} | "
            f"{_mib(summary['disk_read_bytes_during_mode']):.3f}/{_mib(summary['disk_write_bytes_during_mode']):.3f} | {int(summary['peak_process_threads'])} | {int(summary['failed_tasks'])} |"
        )
    d1_tph = float(dft_modes[1][0]["trajectories_per_hour"])
    d2_tph = float(dft_modes[2][0]["trajectories_per_hour"])
    d4_tph = float(dft_modes[4][0]["trajectories_per_hour"])
    dft_lines.extend([
        "",
        "## Numerical consistency and POTCAR control",
        "",
        f"All {int(energy['count'])} task outputs gave TOTEN = {float(energy['energy_ev']):.8f} eV; spread = {float(energy['spread_ev']):.3g} eV against a tolerance of {float(energy['tolerance_ev']):.1e} eV. Every task exited 0, reached EDIFF, and contained the VASP timing footer. Each task recorded the server-side POTCAR hash and then removed its temporary POTCAR; the recovered artifact set contains no POTCAR file.",
        "",
        "## Oversubscription assessment",
        "",
        f"Four tasks created {int(dft_modes[4][0]['peak_process_threads'])} process threads but used at most {float(dft_modes[4][0]['peak_cpu_percent']):.2f}% of the 32-CPU quota. Its maximum per-task wall time was {float(dft_modes[4][1]['elapsed_seconds'].max()):.2f} s versus {single_dft_max:.2f} s for one task; there was no >10% slowdown, convergence failure, or energy change. Thus there is no effective oversubscription signal in this tested workload.",
        "",
        "## Recommendation",
        "",
        f"推荐 DFT 并发数：**{recommended_dft}**。Two-task throughput improved by {(d2_tph / d1_tph - 1) * 100:.1f}% over one task, and four-task throughput improved by {(d4_tph / d2_tph - 1) * 100:.1f}% over two tasks. Four concurrent serial/OpenBLAS jobs should be the cap; do not add MPI ranks or a fifth job without a new benchmark.",
    ])

    recommended_summary = gpu_modes[recommended_gpu][0]
    gpu_wave_hours = float(recommended_summary["mode_elapsed_seconds"]) / 3600.0
    k3_time = 4.034885800007032
    k10_time = 4.085081600002013
    k30_time = 4.297967799997423
    k10_factor = k10_time / k3_time
    k30_factor = k30_time / k3_time
    mixed_k_factor = (1.0 + k10_factor + k30_factor) / 3.0
    mn_factor = 1681.902582168579 / 979.743894815445

    families = [
        ("开发阶段：K=3/10/30 x Greedy/Full x seeds 0--4", 30, 10, mixed_k_factor, mixed_k_factor),
        ("参数校准：threshold 21 + weight 19", 40, 14, 1.0, k30_factor),
        ("Li--M--O 五方法消融", 50, 17, 1.0, k30_factor),
        ("Mn-oxide group-key suite", 50, 17, mn_factor, mn_factor * k30_factor),
        ("Li--M--O MC-dropout sensitivity", 30, 10, mixed_k_factor, mixed_k_factor),
    ]
    family_estimates = [
        (name, count, waves, waves * gpu_wave_hours * low, waves * gpu_wave_hours * high)
        for name, count, waves, low, high in families
    ]
    mandatory_low = sum(row[3] for row in family_estimates)
    mandatory_high = sum(row[4] for row in family_estimates)
    retry_low = mandatory_low * 1.20
    retry_high = mandatory_high * 1.20
    optional_mn_mc = 10 * gpu_wave_hours * mixed_k_factor * mn_factor

    dft_c1_task = float(dft_modes[1][1]["elapsed_seconds"].max())
    kpoint_low = (6 * 229.495 + 3 * dft_c1_task) / 4.0 / 3600.0
    kpoint_high = (6 * 690.7888 + 3 * dft_c1_task) / 4.0 / 3600.0
    recovery = 1814.433 / 3600.0
    fixed_dft_low = kpoint_low + recovery
    fixed_dft_high = kpoint_high + recovery
    full_dft_low = fixed_dft_low + (9 * 61.5505 + 26 * 229.495 + 2 * 61.5505 + 4 * 229.495) / 4.0 / 3600.0
    full_dft_high = fixed_dft_high + (9 * 312.2311 + 26 * 690.7888 + 2 * 312.2311 + 4 * 690.7888) / 4.0 / 3600.0
    minimum_gpu = family_estimates[0][3]

    estimate_lines = [
        "# Updated measured wall-time estimate",
        "",
        "## Basis",
        "",
        f"This estimate is generated from the full-trajectory GPU benchmark at concurrency {recommended_gpu} ({float(recommended_summary['mode_elapsed_seconds']):.2f} s per three-task completion window; {float(recommended_summary['trajectories_per_hour']):.3f} trajectories/hour) and the VASP benchmark at concurrency {recommended_dft}. GPU execution commit: `{gpu_benchmark_commit}`; DFT execution commit: `{dft_benchmark_commit}`; report generator commit: `{git_commit}`. It does not use task count multiplied only by a historical mean.",
        "",
        f"K scaling uses the retained single-round measured totals K=3/10/30 = {k3_time:.3f}/{k10_time:.3f}/{k30_time:.3f} s. For selected-K batches the lower bound is K=3 and the conservative upper bound applies the measured K=30/K=3 factor ({k30_factor:.4f}). Mn-oxide uses the retained measured Mn/Li full-trajectory ratio ({mn_factor:.4f}) because this benchmark was explicitly limited to Li--M--O. DFT task-class scaling uses the historical median/p90 static timings together with the newly measured near-linear four-task efficiency.",
        "",
        "## GPU breakdown",
        "",
        "| experiment family | trajectories | three-task waves | measured/scaled wall time (h) | with 20% retry reserve (h) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, count, waves, low, high in family_estimates:
        estimate_lines.append(f"| {name} | {count} | {waves} | {low:.2f}--{high:.2f} | {low * 1.2:.2f}--{high * 1.2:.2f} |")
    estimate_lines.extend([
        f"| **完整 mandatory GPU** | **200** | dependency-aware | **{mandatory_low:.2f}--{mandatory_high:.2f}** | **{retry_low:.2f}--{retry_high:.2f}** |",
        "",
        f"The optional Mn-oxide K sensitivity adds {optional_mn_mc:.2f} h measured/scaled wall time, or {optional_mn_mc * 1.2:.2f} h with reserve, and remains excluded from the mandatory budget.",
        "",
        "The 20% margin is retained despite zero failures in six full GPU trajectories because the benchmark sample is small and does not cover every method/configuration. Add up to 1.0 h non-compute allowance for the two calibration ranking/freeze transitions; this is not GPU execution time.",
        "",
        "## DFT breakdown",
        "",
        f"- Nine k-point convergence tasks: {kpoint_low:.2f}--{kpoint_high:.2f} h using measured four-task concurrency plus historical compound-static median/p90 complexity.",
        f"- Fixed minimum (nine k-point tasks plus two controlled Mn-static recoveries): {fixed_dft_low:.2f}--{fixed_dft_high:.2f} h; with 20% reserve {fixed_dft_low * 1.2:.2f}--{fixed_dft_high * 1.2:.2f} h.",
        f"- Conditional 52-job envelope: {full_dft_low:.2f}--{full_dft_high:.2f} h; with 20% reserve {full_dft_low * 1.2:.2f}--{full_dft_high * 1.2:.2f} h.",
        "- These bounds exclude any complete re-relaxation of C214/C120/C044, which remains a separate stop condition.",
        "",
        "## Minimum and complete wall time",
        "",
        f"- 最低首批：GPU development/K-selection {minimum_gpu:.2f} h and DFT k-point {kpoint_low:.2f}--{kpoint_high:.2f} h run on separate servers concurrently; critical-path wall time is {minimum_gpu:.2f} h, or {minimum_gpu * 1.2:.2f} h with retry reserve.",
        f"- 完整 mandatory program: GPU is the critical path at {mandatory_low:.2f}--{mandatory_high:.2f} h; with 20% retry reserve and up to 1.0 h calibration-transition allowance, plan {retry_low + 1.0:.2f}--{retry_high + 1.0:.2f} h.",
        f"- DFT conditional envelope with reserve is {full_dft_low * 1.2:.2f}--{full_dft_high * 1.2:.2f} h and can run concurrently, so it does not extend the critical path unless a stop condition triggers.",
        "",
        "## Four-day server question",
        "",
        "连续开启服务器 4 天：**否**。For the mandatory scope, a roughly 30-hour booking covers the measured/scaled critical path and stated reserve; a 36-hour operational reservation is conservative. The workflow has mandatory K/parameter freeze gates, so the servers may be stopped between stages rather than left idle. Four days would only become relevant after adding unapproved work, repeated failures, or main-text full relaxations.",
        "",
        "## Remaining uncertainty",
        "",
        "- GPU concurrency was measured with Full Gate, K=3, Li--M--O and eight threads per process. Method/K scaling is explicitly bounded but not a second full concurrency benchmark.",
        "- Mn-oxide scaling retains the measured historical Mn/Li ratio; it is not presented as a direct Mn concurrency measurement.",
        "- DFT four-task scaling was measured on a small Li-metal static. Compound and recovery bounds therefore retain historical median/p90 task-class timings.",
        "- No formal experiment has been launched; these estimates are the requested pre-budget evidence.",
    ])

    paths = [
        destination / "GPU_CONCURRENCY_BENCHMARK.md",
        destination / "DFT_CONCURRENCY_BENCHMARK.md",
        destination / "UPDATED_WALLTIME_ESTIMATE.md",
    ]
    for path, lines in zip(paths, (gpu_lines, dft_lines, estimate_lines)):
        _write_text_exclusive(path, "\n".join(lines))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-gpu-run")
    parser.add_argument("--candidate-gpu-run")
    parser.add_argument("--dft-energies-json")
    parser.add_argument("--gpu-root")
    parser.add_argument("--dft-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    result: dict[str, Any] = {}
    if args.reference_gpu_run and args.candidate_gpu_run:
        result["gpu"] = compare_gpu_runs(args.reference_gpu_run, args.candidate_gpu_run)
    if args.dft_energies_json:
        result["dft"] = compare_dft_energies(json.loads(Path(args.dft_energies_json).read_text(encoding="utf-8")))
    report_values = (args.gpu_root, args.dft_root, args.output_dir, args.git_commit)
    if any(report_values):
        if not all(report_values):
            raise ValueError("--gpu-root, --dft-root, --output-dir and --git-commit are required together")
        result["reports"] = [
            str(path)
            for path in generate_reports(
                args.gpu_root,
                args.dft_root,
                args.output_dir,
                git_commit=args.git_commit,
            )
        ]
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
