#!/usr/bin/env python3
"""Generate the alpha-Mn NELM=1 cost/memory probe report from raw evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


RESTRICTED = {"POTCAR", "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1", "AECCAR2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _one(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"expected one {label}, found {len(paths)}")
    return paths[0]


def _cgroup_value(payload: dict[str, Any], path: str) -> str | None:
    value = payload.get("files", {}).get(path)
    return None if value is None else str(value)


def summarize_alpha_probe(probe_root: str | Path) -> dict[str, Any]:
    root = Path(probe_root).resolve()
    attempt = root / "attempt_2"
    metadata = json.loads((attempt / "attempt_metadata.json").read_text(encoding="utf-8"))
    result_path = _one(list((attempt / "outputs").glob("*/attempt_1/task_result.json")), "attempt-2 task result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    samples = pd.read_csv(attempt / "audit" / "resource_samples.csv")
    cgroup_path = attempt / "audit" / "cgroup_memory_after_probe.json"
    cgroup = json.loads(cgroup_path.read_text(encoding="utf-8"))
    events = _cgroup_value(cgroup, "/sys/fs/cgroup/memory.events") or ""
    memory_max_text = _cgroup_value(cgroup, "/sys/fs/cgroup/memory.max")
    memory_max = int(memory_max_text) if memory_max_text and memory_max_text.isdigit() else None
    peak_rss = int(pd.to_numeric(samples["process_rss_bytes"], errors="raise").max())
    peak_used = int(pd.to_numeric(samples["memory_used_bytes"], errors="raise").max())
    elapsed = float(result["elapsed_seconds"])
    completed_step = result.get("final_toten_ev") is not None and result.get("timing_footer_present") is True
    cgroup_oom = "oom_kill 0" not in events and "oom_kill" in events
    if int(result.get("exit_code", 0)) == -9 and not cgroup_oom:
        classification = "external_SIGKILL_during_high_memory_growth_cgroup_oom_not_proven"
    elif int(result.get("exit_code", 0)) == -9:
        classification = "cgroup_OOM_SIGKILL"
    else:
        classification = "non_SIGKILL_failure"
    return {
        "probe_scope": "alpha-Mn MAGNDATA 1.85 noncollinear NELM=1 cost-memory probe",
        "scientific_result": False,
        "NELM": int(metadata["NELM"]),
        "config_hash": metadata["attempt_2_config_hash"],
        "vasp_ncl_sha256": metadata["vasp_ncl_sha256"],
        "status": result["status"],
        "exit_code": int(result["exit_code"]),
        "elapsed_seconds_to_termination": elapsed,
        "completed_scf_step": completed_step,
        "final_toten_ev": result.get("final_toten_ev"),
        "electronic_converged": bool(result.get("electronic_converged")),
        "timing_footer_present": bool(result.get("timing_footer_present")),
        "peak_process_rss_bytes": peak_rss,
        "peak_process_rss_GiB": peak_rss / 1024**3,
        "peak_sampled_container_memory_used_bytes": peak_used,
        "peak_sampled_container_memory_used_GiB": peak_used / 1024**3,
        "peak_process_threads": int(pd.to_numeric(samples["process_threads"], errors="raise").max()),
        "cgroup_memory_max_bytes": memory_max,
        "cgroup_memory_max_GiB": None if memory_max is None else memory_max / 1024**3,
        "cgroup_memory_events_after_probe": events,
        "failure_classification": classification,
        "allocation_core_hours_to_termination": elapsed * 8.0 / 3600.0,
        "actual_cpu_hours_to_termination": None,
        "actual_cpu_hours_note": "resource samples contain host-wide CPU utilization, not per-process CPU time",
        "formal_reference_cpu_hours_estimate": None,
        "formal_reference_estimate_note": "not estimable because no electronic step or energy completed",
        "formal_alpha_reference_tasks": "not approved and not launched",
        "potcar_retained": bool(result.get("potcar_retained")),
        "task_result_path": str(result_path),
        "resource_samples_path": str(attempt / "audit" / "resource_samples.csv"),
        "cgroup_snapshot_path": str(cgroup_path),
    }


def build_report(summary: dict[str, Any]) -> str:
    completed_text = (
        f"{summary['elapsed_seconds_to_termination']:.3f} s per completed NELM=1 step"
        if summary["completed_scf_step"]
        else "not measurable: no NELM=1 electronic step completed"
    )
    return f"""# Alpha-Mn NELM=1 cost and memory probe

## Result

- Scope: `{summary['probe_scope']}`.
- This probe is **not a scientific reference-energy result**.
- Exit: `{summary['status']}` with code `{summary['exit_code']}` after
  `{summary['elapsed_seconds_to_termination']:.6f} s`.
- Single-step wall time: **{completed_text}**.
- Final energy, electronic convergence, and VASP timing footer: unavailable / false.
- Failure classification: `{summary['failure_classification']}`.

The correct VASP 6.5.1 non-collinear executable started the 58-atom, 5x5x5
calculation, but the process was terminated by `SIGKILL` while sampled process-tree
RSS was rising. Peak sampled process-tree RSS was
`{summary['peak_process_rss_bytes']}` bytes (`{summary['peak_process_rss_GiB']:.3f} GiB`);
peak sampled container memory use was `{summary['peak_sampled_container_memory_used_GiB']:.3f} GiB`.
The post-probe cgroup counters recorded no cgroup OOM kill, and kernel logs were not
accessible. Therefore the evidence supports an external `SIGKILL` under high memory
pressure, but does **not** prove that the cgroup OOM killer was the source.

## Cost interpretation

- Eight-thread allocation-equivalent consumption before termination:
  `{summary['allocation_core_hours_to_termination']:.6f}` core-hours.
- Actual CPU-hours: unavailable because the sampler recorded host-wide CPU use,
  not process CPU time.
- Total CPU-hours for a converged alpha-Mn reference static: **not estimable** from
  this probe because no electronic step completed.
- Safe memory requirement: not bounded; the task had already reached
  `{summary['peak_process_rss_GiB']:.3f} GiB` RSS before termination.
- Convergence risk under the unchanged frozen protocol and current 60-GiB cgroup:
  high; feasibility was not demonstrated.

## Authorization consequence

The four formal alpha-Mn reference static tasks are **not approved and not launched**.
No retry with changed `LREAL`, k-point mesh, PAW data, ENCUT, magnetic initialization,
or other scientific setting was attempted. A higher-memory server or an explicitly
approved protocol change is required before a new formal alpha-Mn request can be
costed scientifically.

## Provenance

- Probe config SHA-256: `{summary['config_hash']}`
- `vasp_ncl` SHA-256: `{summary['vasp_ncl_sha256']}`
- Task result: `{summary['task_result_path']}`
- Resource samples: `{summary['resource_samples_path']}`
- Post-probe cgroup snapshot: `{summary['cgroup_snapshot_path']}`
- POTCAR retained in recovered output: `{str(summary['potcar_retained']).lower()}`
"""


def write_hash_manifest(root: Path, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for item in sorted(Path(root).rglob("*")):
        if not item.is_file() or item.resolve() == Path(path).resolve():
            continue
        if item.name.upper() in RESTRICTED or item.name.upper().startswith("POTCAR."):
            raise ValueError(f"restricted VASP payload found in recovered evidence: {item}")
        rows.append(
            {
                "relative_path": item.relative_to(root).as_posix(),
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--hashes", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_alpha_probe(args.probe_root)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(summary), encoding="utf-8", newline="\n")
    write_hash_manifest(args.probe_root, args.hashes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
