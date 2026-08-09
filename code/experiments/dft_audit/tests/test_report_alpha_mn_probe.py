from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.report_alpha_mn_probe import build_report, summarize_alpha_probe


def test_alpha_probe_report_does_not_extrapolate_an_incomplete_scf_step(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt_2"
    output = attempt / "outputs" / "probe" / "attempt_1"
    audit = attempt / "audit"
    output.mkdir(parents=True)
    audit.mkdir(parents=True)
    (attempt / "attempt_metadata.json").write_text(
        json.dumps({"NELM": 1, "attempt_2_config_hash": "config", "vasp_ncl_sha256": "vasp"}),
        encoding="utf-8",
    )
    (output / "task_result.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "exit_code": -9,
                "elapsed_seconds": 22.5,
                "final_toten_ev": None,
                "electronic_converged": False,
                "timing_footer_present": False,
                "potcar_retained": False,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"process_rss_bytes": 1024, "memory_used_bytes": 2048, "process_threads": 9},
            {"process_rss_bytes": 51_000_000_000, "memory_used_bytes": 52_000_000_000, "process_threads": 9},
        ]
    ).to_csv(audit / "resource_samples.csv", index=False)
    (audit / "cgroup_memory_after_probe.json").write_text(
        json.dumps(
            {
                "files": {
                    "/sys/fs/cgroup/memory.max": "64424509440",
                    "/sys/fs/cgroup/memory.events": "oom 0\noom_kill 0",
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_alpha_probe(tmp_path)
    report = build_report(summary)

    assert summary["completed_scf_step"] is False
    assert summary["formal_reference_cpu_hours_estimate"] is None
    assert summary["failure_classification"] == "external_SIGKILL_during_high_memory_growth_cgroup_oom_not_proven"
    assert "not estimable" in report
    assert "not approved and not launched" in report
