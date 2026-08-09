from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.server_experiment_watchdog import (
    _remote_resume_script,
    _remote_snapshot_script,
    build_ssh_command,
    decide_action,
    run_watchdog,
)


GIB = 1024**3


def gpu_policy() -> dict[str, object]:
    return {
        "role": "gpu",
        "minimum_free_bytes": 10 * GIB,
        "maximum_root_bytes": 12 * GIB,
    }


def dft_policy() -> dict[str, object]:
    return {
        "role": "dft",
        "minimum_free_bytes": 10 * GIB,
        "maximum_root_bytes": 10 * GIB,
    }


def safe_snapshot(
    *,
    controller_alive: bool,
    pending: int = 0,
    running: int = 0,
    done: int = 0,
) -> dict[str, object]:
    return {
        "controller_alive": controller_alive,
        "status_counts": {
            "PENDING": pending,
            "RUNNING": running,
            "DONE": done,
            "FAILED": 0,
            "CANCELLED": 0,
        },
        "live_running_pids": [],
        "error_flags": [],
        "pause_reason": None,
        "free_bytes": 50 * GIB,
        "root_bytes": 2 * GIB,
        "required_protocol_present": True,
    }


def test_gpu_dead_controller_with_safe_pending_work_is_resumable():
    decision = decide_action(
        safe_snapshot(controller_alive=False, pending=4),
        gpu_policy(),
    )

    assert decision.action == "RESUME_GPU_CONTROLLER"
    assert decision.reason == "safe_controller_absence"
    assert decision.repair_allowed is True


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"error_flags": ["oom"]}, "numerical_or_accelerator_error"),
        ({"live_running_pids": [123]}, "orphan_worker"),
        ({"pause_reason": "throughput_below_floor"}, "controller_safety_pause"),
        ({"free_bytes": 9 * GIB}, "free_disk_below_limit"),
        ({"root_bytes": 13 * GIB}, "output_root_exceeds_budget"),
    ],
)
def test_safety_flags_and_live_orphans_are_never_auto_repaired(change, reason):
    snapshot = safe_snapshot(controller_alive=False, pending=4)
    snapshot.update(change)

    decision = decide_action(snapshot, gpu_policy())

    assert decision.action == "ALERT"
    assert decision.reason == reason
    assert decision.repair_allowed is False


def test_three_same_config_failures_are_never_auto_repaired():
    snapshot = safe_snapshot(controller_alive=False, pending=4)
    snapshot["same_config_failure_max"] = 3

    decision = decide_action(snapshot, gpu_policy())

    assert decision.action == "ALERT"
    assert decision.reason == "three_same_config_failures"
    assert decision.repair_allowed is False


def test_live_gpu_controller_is_healthy_without_starting_another_controller():
    decision = decide_action(
        safe_snapshot(controller_alive=True, pending=4, running=3, done=9),
        gpu_policy(),
    )

    assert decision.action == "HEALTHY"
    assert decision.repair_allowed is False


def test_dft_completed_kpoint_batch_is_scientific_hold_until_protocol_is_frozen():
    snapshot = safe_snapshot(controller_alive=False, done=9)
    snapshot["required_protocol_present"] = False

    decision = decide_action(snapshot, dft_policy())

    assert decision.action == "SCIENTIFIC_HOLD"
    assert decision.reason == "dft_protocol_not_frozen"
    assert decision.repair_allowed is False


def test_dft_pending_work_is_monitor_only():
    decision = decide_action(
        safe_snapshot(controller_alive=False, pending=2, done=9),
        dft_policy(),
    )

    assert decision.action == "ALERT"
    assert decision.reason == "dft_monitor_only"
    assert decision.repair_allowed is False


def test_gpu_manifest_without_pending_work_stops_at_stage_boundary():
    decision = decide_action(
        safe_snapshot(controller_alive=False, done=30),
        gpu_policy(),
    )

    assert decision.action == "STAGE_COMPLETE"
    assert decision.reason == "no_pending_gpu_work"
    assert decision.repair_allowed is False


def server_config(tmp_path: Path, *, role: str = "gpu") -> dict[str, object]:
    root = f"/remote/{role}"
    return {
        "role": role,
        "host": f"{role}.example.test",
        "port": 2200 if role == "gpu" else 2201,
        "user": "root",
        "remote_python": f"/opt/{role}/python",
        "ssh_executable": "ssh.exe",
        "ssh_key_path": str(tmp_path / "id_ed25519"),
        "known_hosts_path": str(tmp_path / "known_hosts"),
        "root": root,
        "manifest": f"{root}/jobs/jobs.csv",
        "minimum_free_bytes": 10 * GIB,
        "maximum_root_bytes": (12 if role == "gpu" else 10) * GIB,
        "required_protocol_path": f"{root}/configs/frozen.yaml",
        "auto_resume": role == "gpu",
        "controller": {
            "python": "/opt/python",
            "script": f"{root}/project/analysis/run_audited_job_queue.py",
            "cwd": f"{root}/project",
            "arguments": [
                "--concurrency",
                "3",
                "--resource-kind",
                "gpu",
            ],
        },
    }


def write_config(tmp_path: Path, servers: dict[str, dict[str, object]]) -> Path:
    path = tmp_path / "watchdog.json"
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


class RecordingRunner:
    def __init__(self, snapshots: dict[str, dict[str, object]]):
        self.snapshots = snapshots
        self.resume_calls = 0
        self.commands: list[list[str]] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        script = kwargs["input"]
        host = next(key for key in self.snapshots if key in command)
        if "WATCHDOG_REMOTE_SNAPSHOT" in script:
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(self.snapshots[host]),
                stderr="",
            )
        if "WATCHDOG_REMOTE_RESUME" in script:
            self.resume_calls += 1
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(
                    {
                        "started": True,
                        "pid": 456,
                        "audit_dir": "/remote/audit/resume",
                    }
                ),
                stderr="",
            )
        raise AssertionError("unexpected remote script")


def test_ssh_command_uses_key_strict_hosts_and_no_password(tmp_path):
    command = build_ssh_command(server_config(tmp_path))
    joined = " ".join(command)

    assert "BatchMode=yes" in joined
    assert "StrictHostKeyChecking=yes" in joined
    assert "password" not in joined.lower()
    assert command[-1] == "root@gpu.example.test"


def test_only_resumable_gpu_decision_calls_resume_and_writes_audit(tmp_path):
    gpu = server_config(tmp_path)
    snapshot = safe_snapshot(controller_alive=False, pending=2)
    snapshot["manifest_sha256"] = "abc123"
    runner = RecordingRunner({"root@gpu.example.test": snapshot})
    output = tmp_path / "audit"

    report = run_watchdog(
        write_config(tmp_path, {"gpu": gpu}),
        output,
        ssh_runner=runner,
    )

    assert report["servers"]["gpu"]["action"] == "RESUME_GPU_CONTROLLER"
    assert report["servers"]["gpu"]["repair"]["started"] is True
    assert runner.resume_calls == 1
    assert (output / "last_check.json").is_file()
    assert (output / "history.jsonl").is_file()
    assert len(list((output / "checks").glob("*.json"))) == 1


def test_remote_python_is_selected_per_server(tmp_path):
    gpu = server_config(tmp_path)
    snapshot = safe_snapshot(controller_alive=True, pending=2)
    snapshot["manifest_sha256"] = "abc123"
    runner = RecordingRunner({"root@gpu.example.test": snapshot})

    run_watchdog(
        write_config(tmp_path, {"gpu": gpu}),
        tmp_path / "audit",
        ssh_runner=runner,
    )

    assert runner.commands[0][-2:] == ["/opt/gpu/python", "-"]


def test_dft_and_done_jobs_never_trigger_resume(tmp_path):
    dft = server_config(tmp_path, role="dft")
    snapshot = safe_snapshot(controller_alive=False, done=9)
    snapshot["required_protocol_present"] = False
    snapshot["manifest_sha256"] = "def456"
    runner = RecordingRunner({"root@dft.example.test": snapshot})

    report = run_watchdog(
        write_config(tmp_path, {"dft": dft}),
        tmp_path / "audit",
        ssh_runner=runner,
    )

    assert report["servers"]["dft"]["action"] == "SCIENTIFIC_HOLD"
    assert runner.resume_calls == 0


def test_repository_config_has_two_servers_and_no_credentials():
    path = Path(__file__).resolve().parents[1] / "configs" / "server_watchdog.json"
    raw = path.read_text(encoding="utf-8")
    config = json.loads(raw)

    assert set(config["servers"]) == {"gpu", "dft"}
    assert config["servers"]["gpu"]["auto_resume"] is True
    assert config["servers"]["dft"]["auto_resume"] is False
    assert "password" not in raw.lower()


def test_generated_remote_python_scripts_compile(tmp_path):
    gpu = server_config(tmp_path)

    compile(_remote_snapshot_script(gpu), "<watchdog-snapshot>", "exec")
    compile(
        _remote_resume_script(gpu, expected_manifest_sha256="0" * 64),
        "<watchdog-resume>",
        "exec",
    )


def test_generated_scripts_resolve_a_dynamic_manifest_pointer(tmp_path):
    gpu = server_config(tmp_path)
    gpu["manifest_pointer"] = "/remote/gpu/jobs/ACTIVE_GPU_MANIFEST.txt"

    snapshot = _remote_snapshot_script(gpu)
    resume = _remote_resume_script(gpu, expected_manifest_sha256="0" * 64)

    for script in (snapshot, resume):
        assert "ACTIVE_GPU_MANIFEST.txt" in script
        assert "configured_manifest" in script
        assert "manifest_pointer" in script
        compile(script, "<dynamic-manifest-watchdog>", "exec")


def test_wrapper_and_installer_define_non_overlapping_thirty_minute_task():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts" / "run_server_watchdog.ps1").read_text(encoding="utf-8")
    installer = (root / "scripts" / "install_server_watchdog_task.ps1").read_text(
        encoding="utf-8"
    )

    assert "server_experiment_watchdog.py" in wrapper
    assert "server_watchdog.json" in wrapper
    assert "System.Threading.Mutex" in wrapper
    assert "UTF8Encoding" in wrapper
    assert "AppendAllText" in wrapper
    assert "Tee-Object -FilePath" not in wrapper
    assert "RepetitionInterval (New-TimeSpan -Minutes 30)" in installer
    assert "MultipleInstances IgnoreNew" in installer
    assert "ExecutionTimeLimit (New-TimeSpan -Minutes 10)" in installer
