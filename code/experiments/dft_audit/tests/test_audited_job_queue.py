import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

import analysis.run_audited_job_queue as queue_module
from analysis.run_audited_job_queue import (
    QueueConfigurationError,
    evaluate_safety,
    reconcile_resume_state,
    run_queue,
    validate_jobs,
)


def task_row(tmp_path, index):
    output = tmp_path / f"out_{index}"
    log = tmp_path / f"log_{index}.txt"
    code = (
        "import json,pathlib,time;"
        f"p=pathlib.Path(r'{output}');p.mkdir(parents=True);"
        "time.sleep(0.05);"
        "(p/'status.json').write_text(json.dumps({'status':'DONE'}));"
        "(p/'run_config.json').write_text(json.dumps({'formal_protocol_sha256':'cfg'}))"
    )
    return {
        "job_id": f"job_{index}",
        "dataset": "limo",
        "method": "energy_gated_da_tpp",
        "group_key": "element_system_current",
        "seed": index,
        "K": 3,
        "config_hash": "cfg",
        "git_commit": "abc",
        "gpu_id": 0,
        "status": "PENDING",
        "start_time": "",
        "end_time": "",
        "exit_code": "",
        "log_path": str(log),
        "output_path": str(output),
        "sha256": "",
        "command_json": json.dumps([sys.executable, "-c", code]),
        "cwd": str(tmp_path),
        "attempt": 1,
        "pid": "",
        "failure_reason": "",
    }


def test_validate_jobs_rejects_output_collisions(tmp_path):
    frame = pd.DataFrame([task_row(tmp_path, 0), task_row(tmp_path, 1)])
    frame.loc[1, "output_path"] = frame.loc[0, "output_path"]
    with pytest.raises(QueueConfigurationError, match="unique output_path"):
        validate_jobs(frame)


def test_validate_jobs_rejects_non_mapping_environment(tmp_path):
    frame = pd.DataFrame([task_row(tmp_path, 0)])
    frame["env_json"] = [json.dumps(["OPENBLAS_NUM_THREADS=8"])]
    with pytest.raises(QueueConfigurationError, match="env_json"):
        validate_jobs(frame)


def test_safety_gate_covers_disk_throughput_and_numerical_errors():
    assert evaluate_safety(
        free_bytes=9,
        minimum_free_bytes=10,
        root_bytes=1,
        maximum_root_bytes=100,
        completed_jobs=0,
        throughput_per_hour=None,
        baseline_throughput_per_hour=11.2,
        throughput_floor_fraction=0.70,
        error_flags=set(),
    ) == "free_disk_below_limit"
    assert evaluate_safety(
        free_bytes=100,
        minimum_free_bytes=10,
        root_bytes=101,
        maximum_root_bytes=100,
        completed_jobs=0,
        throughput_per_hour=None,
        baseline_throughput_per_hour=11.2,
        throughput_floor_fraction=0.70,
        error_flags=set(),
    ) == "output_root_exceeds_budget"
    assert evaluate_safety(
        free_bytes=100,
        minimum_free_bytes=10,
        root_bytes=1,
        maximum_root_bytes=100,
        completed_jobs=3,
        throughput_per_hour=7.0,
        baseline_throughput_per_hour=11.2,
        throughput_floor_fraction=0.70,
        error_flags=set(),
    ) == "throughput_below_floor"
    assert evaluate_safety(
        free_bytes=100,
        minimum_free_bytes=10,
        root_bytes=1,
        maximum_root_bytes=100,
        completed_jobs=0,
        throughput_per_hour=None,
        baseline_throughput_per_hour=11.2,
        throughput_floor_fraction=0.70,
        error_flags={"nan"},
    ) == "numerical_or_accelerator_error:nan"


def test_throughput_gate_waits_for_a_complete_new_concurrency_cohort_on_resume():
    window_started = 100.0
    historical_done = 6

    assert queue_module.cohort_throughput(
        now=1000.0,
        window_started=window_started,
        completed_jobs=historical_done,
        window_completed_jobs=historical_done,
        cohort_size=3,
    ) is None
    assert queue_module.cohort_throughput(
        now=1050.0,
        window_started=window_started,
        completed_jobs=historical_done + 2,
        window_completed_jobs=historical_done,
        cohort_size=3,
    ) is None
    assert queue_module.cohort_throughput(
        now=1100.0,
        window_started=window_started,
        completed_jobs=historical_done + 3,
        window_completed_jobs=historical_done,
        cohort_size=3,
    ) == pytest.approx(10.8)


def test_progress_throughput_excludes_jobs_done_before_resumed_controller(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    frame = pd.DataFrame([task_row(tmp_path, index) for index in range(9)])
    frame.loc[:5, "status"] = "DONE"

    summary = queue_module._write_progress(
        audit_dir=audit,
        frame=frame,
        started=time.monotonic() - 3600.0,
        initial_completed_jobs=6,
        peak_running_jobs=0,
        root_bytes=1,
        free_bytes=100,
        pause_reason=None,
        resource_snapshot={},
    )

    assert summary["throughput_jobs_per_hour"] is None
    assert summary["remaining_wall_hours"] is None


def test_resume_rejects_done_config_mismatch_and_marks_stale_running(tmp_path):
    done = task_row(tmp_path, 0)
    done["status"] = "DONE"
    output = Path(done["output_path"])
    output.mkdir()
    (output / "status.json").write_text(json.dumps({"status": "DONE"}), encoding="utf-8")
    (output / "run_config.json").write_text(
        json.dumps({"formal_protocol_sha256": "different"}), encoding="utf-8"
    )
    with pytest.raises(QueueConfigurationError, match="DONE config hash mismatch"):
        reconcile_resume_state(pd.DataFrame([done]), resource_kind="gpu")

    stale = task_row(tmp_path, 1)
    stale["status"] = "RUNNING"
    stale["pid"] = "99999999"
    reconciled = reconcile_resume_state(pd.DataFrame([stale]), resource_kind="cpu")
    assert reconciled.loc[0, "status"] == "FAILED"
    assert reconciled.loc[0, "failure_reason"] == "stale_running_process_missing"


def test_resume_recovers_verified_gpu_completion_after_controller_crash(tmp_path):
    stale = task_row(tmp_path, 0)
    stale["status"] = "FAILED"
    stale["failure_reason"] = "stale_running_process_missing"
    stale["exit_code"] = ""
    output = Path(stale["output_path"])
    output.mkdir()
    (output / "status.json").write_text(json.dumps({"status": "DONE"}), encoding="utf-8")
    (output / "run_config.json").write_text(
        json.dumps({"formal_protocol_sha256": "cfg"}), encoding="utf-8"
    )

    reconciled = reconcile_resume_state(pd.DataFrame([stale]), resource_kind="gpu")

    assert reconciled.loc[0, "status"] == "DONE"
    assert reconciled.loc[0, "exit_code"] == ""
    assert reconciled.loc[0, "failure_reason"] == (
        "reconciled_verified_done_output_after_controller_crash;"
        "process_exit_code_unavailable"
    )
    assert reconciled.loc[0, "sha256"]


def test_tree_size_ignores_file_removed_between_discovery_and_stat(tmp_path, monkeypatch):
    stable = tmp_path / "stable.bin"
    volatile = tmp_path / "volatile.bin"
    stable.write_bytes(b"stable")
    volatile.write_bytes(b"volatile")
    original_stat = Path.stat
    volatile_stat_calls = 0

    def racing_stat(path, *args, **kwargs):
        nonlocal volatile_stat_calls
        if path == volatile:
            volatile_stat_calls += 1
            if volatile_stat_calls >= 2:
                raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)

    assert queue_module._tree_size(tmp_path) == stable.stat().st_size


def test_queue_runs_at_configured_concurrency_and_records_progress(tmp_path):
    manifest = tmp_path / "jobs.csv"
    pd.DataFrame([task_row(tmp_path, index) for index in range(4)]).to_csv(manifest, index=False)
    audit = tmp_path / "audit"

    summary = run_queue(
        manifest_path=manifest,
        audit_dir=audit,
        output_root=tmp_path,
        concurrency=2,
        resource_kind="cpu",
        sample_interval_seconds=0.02,
        progress_interval_seconds=0.05,
        minimum_free_bytes=0,
        maximum_root_bytes=10**9,
        baseline_throughput_per_hour=1.0,
        throughput_floor_fraction=0.0,
    )

    finished = pd.read_csv(manifest)
    assert finished["status"].eq("DONE").all()
    assert summary["peak_running_jobs"] == 2
    assert (audit / "resource_samples.csv").is_file()
    progress = sorted(audit.glob("progress_*.json"))
    assert progress
    snapshot = json.loads(progress[-1].read_text(encoding="utf-8"))["resource_snapshot"]
    assert "cpu_percent" in snapshot
    assert "memory_used_bytes" in snapshot
