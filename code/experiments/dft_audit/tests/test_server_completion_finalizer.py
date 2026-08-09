from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from analysis.server_completion_finalizer import (
    analyze_dft_job,
    analyze_gpu_job,
    analyze_recovered_role,
    assess_completion,
    assess_role_completion,
    authorize_shutdown,
    build_scp_command,
    build_remote_stream_command,
    build_shutdown_script,
    completion_evidence_kind,
    finalize_ready_role,
    build_remote_package_script,
    is_restricted_artifact,
    map_remote_path,
    resume_authorized_shutdown,
    run_finalization_once,
    safe_extract_archive,
    sha256_tree,
    scientific_holds_from_gpu_analysis,
    verify_allowed_inventory,
)


def complete_snapshot(*, done: int = 2) -> dict[str, object]:
    return {
        "manifest_rows": done,
        "status_counts": {
            "PENDING": 0,
            "RUNNING": 0,
            "DONE": done,
            "FAILED": 0,
            "CANCELLED": 0,
        },
        "controller_alive": False,
        "live_running_pids": [],
        "active_vasp_pids": [],
        "stale_running_job_ids": [],
        "error_flags": [],
        "pause_reason": None,
        "same_config_failure_max": 0,
        "required_protocol_present": True,
    }


def test_completion_gate_requires_both_manifests_to_be_fully_done():
    gpu = complete_snapshot(done=130)
    dft = complete_snapshot(done=19)

    gate = assess_completion({"gpu": gpu, "dft": dft})

    assert gate.ready is True
    assert gate.state == "READY"
    assert gate.reasons == ()


@pytest.mark.parametrize(
    ("change", "expected_reason"),
    [
        (
            {
                "status_counts": {
                    "PENDING": 1,
                    "RUNNING": 0,
                    "DONE": 1,
                    "FAILED": 0,
                    "CANCELLED": 0,
                }
            },
            "pending_or_running",
        ),
        (
            {
                "status_counts": {
                    "PENDING": 0,
                    "RUNNING": 0,
                    "DONE": 1,
                    "FAILED": 1,
                    "CANCELLED": 0,
                }
            },
            "failed_or_cancelled",
        ),
        ({"controller_alive": True}, "controller_or_worker_still_alive"),
        ({"live_running_pids": [123]}, "controller_or_worker_still_alive"),
        ({"active_vasp_pids": [456]}, "controller_or_worker_still_alive"),
        ({"error_flags": ["nan"]}, "remote_error_or_pause"),
        ({"pause_reason": "throughput_below_floor"}, "remote_error_or_pause"),
        ({"required_protocol_present": False}, "protocol_missing"),
        ({"ssh_error": "offline"}, "ssh_error"),
    ],
)
def test_completion_gate_blocks_shutdown_on_incomplete_or_unsafe_state(
    change, expected_reason
):
    gpu = complete_snapshot()
    gpu.update(change)

    gate = assess_completion({"gpu": gpu, "dft": complete_snapshot()})

    assert gate.ready is False
    assert gate.state in {"WAIT", "BLOCKED"}
    assert any(expected_reason in reason for reason in gate.reasons)


@pytest.mark.parametrize(
    "name",
    [
        "POTCAR",
        "POTCAR.gz",
        "WAVECAR",
        "CHGCAR",
        "CHG",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
    ],
)
def test_restricted_vasp_artifacts_are_never_transferable(name):
    assert is_restricted_artifact(Path("job") / name) is True


def test_allowed_inventory_detects_missing_extra_and_hash_mismatches(tmp_path):
    root = tmp_path / "payload"
    root.mkdir()
    good = root / "results.csv"
    good.write_text("a,b\n1,2\n", encoding="utf-8")
    row = {
        "relative_path": "results.csv",
        "size_bytes": str(good.stat().st_size),
        "sha256": hashlib.sha256(good.read_bytes()).hexdigest(),
    }

    report = verify_allowed_inventory(root, [row])
    assert report["verified_files"] == 1

    good.write_text("z,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_allowed_inventory(root, [row])

    good.write_text("a,b\n1,2\n", encoding="utf-8")
    (root / "unexpected.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected local files"):
        verify_allowed_inventory(root, [row])


def test_remote_path_mapping_rejects_paths_outside_the_frozen_root(tmp_path):
    local_root = tmp_path / "payload"
    mapped = map_remote_path("/remote/formal", local_root, "/remote/formal/results/job")
    assert mapped == local_root / "results" / "job"

    with pytest.raises(ValueError, match="outside remote root"):
        map_remote_path("/remote/formal", local_root, "/remote/other/job")


def _write_gpu_job(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "run_config.json").write_text(
        json.dumps(
            {
                "name": "limo",
                "method": "energy_gated_da_tpp",
                "seed": 15,
                "budget": 4,
                "batch_size": 2,
                "target_count": 3,
                "formal_protocol_sha256": "protocol-hash",
            }
        ),
        encoding="utf-8",
    )
    with (path / "al_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "iteration", "target_label"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "a", "iteration": 1, "target_label": 1},
                {"id": "b", "iteration": 1, "target_label": 0},
                {"id": "c", "iteration": 2, "target_label": 1},
                {"id": "d", "iteration": 2, "target_label": 1},
            ]
        )
    with (path / "round_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "route",
                "selected_unique_groups",
                "correction_replacement_count",
                "correction_target_gain",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "route": "threshold_greedy",
                    "selected_unique_groups": 2,
                    "correction_replacement_count": 0,
                    "correction_target_gain": 0,
                },
                {
                    "route": "diversity_aware",
                    "selected_unique_groups": 1,
                    "correction_replacement_count": 1,
                    "correction_target_gain": 1,
                },
            ]
        )
    for name in ("prediction_manifest.csv", "checkpoint_manifest.csv"):
        (path / name).write_text("round,sha256\n1,a\n2,b\n", encoding="utf-8")


def test_gpu_local_analysis_recomputes_metrics_from_raw_history(tmp_path):
    output = tmp_path / "gpu"
    _write_gpu_job(output)
    row = {
        "job_id": "gpu-final-1",
        "dataset": "limo",
        "method": "energy_gated_da_tpp",
        "group_key": "element_system_current",
        "seed": "15",
        "K": "30",
        "formal_stage": "li_m_o_ablation",
        "config_hash": "config-hash",
    }

    result = analyze_gpu_job(output, row)

    assert result["AUTC"] == pytest.approx(1.0 / 6.0)
    assert result["recovery_at_4"] == 3
    assert result["direct_rounds"] == 1
    assert result["correction_rounds"] == 1
    assert result["effective_replacements"] == 1
    assert result["candidate_sequence_sha256"]


def test_dft_local_analysis_requires_completed_outcar_and_extracts_energy(tmp_path):
    output = tmp_path / "dft"
    output.mkdir()
    (output / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
    (output / "KPOINTS").write_text("mesh\n0\nGamma\n4 4 4\n0 0 0\n", encoding="utf-8")
    (output / "POSCAR").write_text("test\n", encoding="utf-8")
    (output / "OUTCAR").write_text(
        "vasp.6.5.1\n free  energy   TOTEN  =      -12.345678 eV\n"
        "General timing and accounting informations for this job:\n",
        encoding="utf-8",
    )
    row = {
        "job_id": "dft-1",
        "candidate_id": "C044",
        "formula": "LiMn2O4",
        "functional": "GGA+U",
        "magnetic_initialization": "state_afm",
        "main_text_selected": "True",
        "kpoint_spacing_Ainv": "0.15",
        "mesh": "4x4x4",
    }

    result = analyze_dft_job(output, row)

    assert result["vasp_completed"] is True
    assert result["final_total_energy_eV"] == pytest.approx(-12.345678)
    assert result["vasp_version"] == "6.5.1"

    (output / "OUTCAR").write_text("free  energy TOTEN = -1 eV\n", encoding="utf-8")
    with pytest.raises(ValueError, match="timing footer"):
        analyze_dft_job(output, row)


def test_remote_packager_rechecks_jobs_and_excludes_licensed_and_large_vasp_files():
    server = {
        "role": "dft",
        "root": "/remote/formal",
        "manifest": "/remote/formal/jobs/jobs.csv",
        "manifest_pointer": "/remote/formal/jobs/ACTIVE_DFT_MANIFEST.txt",
    }

    script = build_remote_package_script(server, "/remote-packages/dft")

    compile(script, "<remote-finalizer>", "exec")
    assert "REMOTE_COMPLETION_PACKAGE_V1" in script
    assert "sha256_tree" in script
    assert "status_counts" in script
    assert "stream_script_path" in script
    assert "completed_evidence.tar" not in script
    assert "marker_path.open('x'" in script
    for name in ("POTCAR", "WAVECAR", "CHGCAR", "AECCAR0"):
        assert name in script


def test_shutdown_authorization_requires_verified_copy_analysis_and_no_scientific_hold():
    ready = FinalizationGateForTest(ready=True)
    summary = {
        "remote_validation_passed": True,
        "archive_sha256_verified": True,
        "allowed_inventory_verified": True,
        "local_analysis_passed": True,
        "scientific_holds": [],
    }
    assert authorize_shutdown(ready, summary) is True

    for field in (
        "remote_validation_passed",
        "archive_sha256_verified",
        "allowed_inventory_verified",
        "local_analysis_passed",
    ):
        changed = dict(summary)
        changed[field] = False
        assert authorize_shutdown(ready, changed) is False
    changed = dict(summary)
    changed["scientific_holds"] = ["Li full gate reverses the legacy direction"]
    assert authorize_shutdown(ready, changed) is False
    assert authorize_shutdown(FinalizationGateForTest(ready=False), summary) is False


class FinalizationGateForTest:
    def __init__(self, *, ready: bool):
        self.ready = ready


def test_finalizer_wrapper_and_installer_define_non_overlapping_thirty_minute_task():
    root = Path(__file__).resolve().parents[1]
    wrapper = (root / "scripts" / "run_server_completion_finalizer.ps1").read_text(
        encoding="utf-8"
    )
    installer = (
        root / "scripts" / "install_server_completion_finalizer_task.ps1"
    ).read_text(encoding="utf-8")

    assert "server_completion_finalizer.py" in wrapper
    assert "--execute" in wrapper
    assert "--shutdown-after-success" in wrapper
    assert "System.Threading.Mutex" in wrapper
    assert "RepetitionInterval (New-TimeSpan -Minutes 30)" in installer
    assert "MultipleInstances IgnoreNew" in installer
    assert "ExecutionTimeLimit (New-TimeSpan -Hours 8)" in installer


def test_safe_archive_extraction_rejects_path_traversal(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("evidence", encoding="utf-8")
    safe_tar = tmp_path / "safe.tar"
    with tarfile.open(safe_tar, "w") as archive:
        archive.add(source, arcname="results/source.txt")
    destination = tmp_path / "safe-output"

    safe_extract_archive(safe_tar, destination)
    assert (destination / "results" / "source.txt").read_text(
        encoding="utf-8"
    ) == "evidence"

    unsafe_tar = tmp_path / "unsafe.tar"
    with tarfile.open(unsafe_tar, "w") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = 0
        archive.addfile(info)
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract_archive(unsafe_tar, tmp_path / "unsafe-output")


def test_recovered_gpu_role_analysis_validates_tree_hash_and_writes_raw_recalculation(
    tmp_path,
):
    payload = tmp_path / "payload"
    output = payload / "results" / "job" / "attempt_1"
    _write_gpu_job(output)
    manifest = payload / "jobs" / "gpu.csv"
    manifest.parent.mkdir(parents=True)
    row = {
        "job_id": "gpu-final-1",
        "dataset": "limo",
        "method": "energy_gated_da_tpp",
        "group_key": "element_system_current",
        "seed": "15",
        "K": "30",
        "formal_stage": "li_m_o_ablation",
        "config_hash": "config-hash",
        "status": "DONE",
        "exit_code": "0",
        "failure_reason": "",
        "output_path": "/remote/formal/results/job/attempt_1",
        "sha256": sha256_tree(output),
    }
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    validation = {
        "root": "/remote/formal",
        "manifest_relative_path": "jobs/gpu.csv",
        "manifest_rows": 1,
        "job_tree_hashes_verified": 1,
    }
    analysis_dir = tmp_path / "analysis"

    report = analyze_recovered_role("gpu", payload, validation, analysis_dir)

    assert report["status"] == "PASS"
    assert report["analyzed_jobs"] == 1
    recalculated = pd_read_csv(analysis_dir / "gpu_per_trajectory_analysis.csv")
    assert recalculated[0]["AUTC"] == pytest.approx(1.0 / 6.0)


def pd_read_csv(path: Path) -> list[dict[str, object]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def test_shutdown_script_flushes_disks_before_poweroff():
    script = build_shutdown_script()

    compile(script, "<remote-shutdown>", "exec")
    assert "os.sync()" in script
    assert "/bin/bash" in script
    assert "/usr/bin/shutdown" in script
    assert "/sbin/shutdown" not in script
    assert "FINALIZATION_SHUTDOWN_AUTHORIZED" in script


def test_incomplete_finalizer_run_writes_wait_state_without_packaging(tmp_path):
    config = tmp_path / "watchdog.json"
    config.write_text(
        json.dumps(
            {
                "servers": {
                    "gpu": {"role": "gpu"},
                    "dft": {"role": "dft"},
                }
            }
        ),
        encoding="utf-8",
    )
    snapshots = {
        "gpu": complete_snapshot(done=130),
        "dft": complete_snapshot(done=19),
    }
    snapshots["gpu"]["status_counts"] = {
        "PENDING": 1,
        "RUNNING": 0,
        "DONE": 129,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    snapshots["dft"]["status_counts"] = {
        "PENDING": 15,
        "RUNNING": 4,
        "DONE": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    output = tmp_path / "finalizer"

    report = run_finalization_once(
        config,
        output,
        execute=True,
        shutdown_after_success=True,
        snapshot_collector=lambda server: snapshots[str(server["role"])],
        role_finalizer=lambda **_: pytest.fail("finalization must not start"),
    )

    assert report["state"] == "WAIT"
    assert (output / "last_finalizer_check.json").is_file()
    assert not (output / "FINALIZATION_COMPLETE.json").exists()


def test_completed_gpu_can_finalize_while_dft_continues(tmp_path):
    config = tmp_path / "watchdog.json"
    config.write_text(
        json.dumps(
            {
                "servers": {
                    "gpu": {"role": "gpu"},
                    "dft": {"role": "dft"},
                }
            }
        ),
        encoding="utf-8",
    )
    snapshots = {
        "gpu": complete_snapshot(done=130),
        "dft": complete_snapshot(done=19),
    }
    snapshots["dft"]["status_counts"] = {
        "PENDING": 15,
        "RUNNING": 4,
        "DONE": 0,
        "FAILED": 0,
        "CANCELLED": 0,
    }
    calls: list[str] = []

    report = run_finalization_once(
        config,
        tmp_path / "finalizer",
        execute=True,
        shutdown_after_success=True,
        snapshot_collector=lambda server: snapshots[str(server["role"])],
        role_finalizer=lambda **kwargs: (
            calls.append(kwargs["role"])
            or {"state": "ROLE_COMPLETE", "role": kwargs["role"]}
        ),
    )

    assert calls == ["gpu"]
    assert report["state"] == "PARTIAL"
    assert report["roles"]["gpu"]["state"] == "ROLE_COMPLETE"
    assert report["roles"]["dft"]["state"] == "WAIT"


def test_single_role_gate_accepts_a_quiescent_complete_gpu_manifest():
    gate = assess_role_completion("gpu", complete_snapshot(done=130))

    assert gate.ready is True
    assert gate.state == "READY"


def test_ready_role_shuts_down_only_after_recovery_and_analysis_gates(tmp_path):
    events: list[str] = []
    config = {"servers": {"gpu": {"role": "gpu"}}}

    result = finalize_ready_role(
        config=config,
        output_dir=tmp_path,
        role="gpu",
        snapshot=complete_snapshot(done=130),
        shutdown_after_success=True,
        recovery_runner=lambda **_: events.append("recover") or {"status": "PASS"},
        summary_runner=lambda *_: (
            events.append("analyze")
            or {
                "remote_validation_passed": True,
                "archive_sha256_verified": True,
                "allowed_inventory_verified": True,
                "local_analysis_passed": True,
                "scientific_holds": [],
            }
        ),
        shutdown_runner=lambda _: (
            events.append("shutdown") or {"shutdown_started": True}
        ),
    )

    assert events == ["recover", "analyze", "shutdown"]
    assert result["state"] == "ROLE_COMPLETE"
    assert (tmp_path / "SHUTDOWN_AUTHORIZED_gpu.json").is_file()
    assert (tmp_path / "ROLE_FINALIZATION_COMPLETE_gpu.json").is_file()


def test_only_the_documented_controller_crash_reconciliation_may_lack_exit_code():
    standard = {"status": "DONE", "exit_code": "0", "failure_reason": ""}
    reconciled = {
        "status": "DONE",
        "exit_code": "",
        "failure_reason": (
            "reconciled_verified_done_output_after_controller_crash;"
            "process_exit_code_unavailable"
        ),
    }

    assert completion_evidence_kind(standard) == "standard_exit_zero"
    assert completion_evidence_kind(reconciled) == "verified_reconciled_output"
    with pytest.raises(ValueError, match="unacceptable completion evidence"):
        completion_evidence_kind(
            {"status": "DONE", "exit_code": "", "failure_reason": "unknown"}
        )


def test_scp_command_uses_strict_key_authentication_and_contains_no_password(tmp_path):
    server = {
        "host": "gpu.example.test",
        "port": 2200,
        "user": "root",
        "ssh_executable": "C:/Windows/System32/OpenSSH/ssh.exe",
        "ssh_key_path": str(tmp_path / "id_ed25519"),
        "known_hosts_path": str(tmp_path / "known_hosts"),
    }

    command = build_scp_command(server, "/remote/result.tar", tmp_path / "result.tar")
    joined = " ".join(command)

    assert command[0].endswith("scp.exe")
    assert "BatchMode=yes" in joined
    assert "StrictHostKeyChecking=yes" in joined
    assert "password" not in joined.lower()
    assert "root@gpu.example.test:/remote/result.tar" in command


def test_remote_stream_command_uses_server_python_and_binary_stdout(tmp_path):
    server = {
        "host": "gpu.example.test",
        "port": 2200,
        "user": "root",
        "remote_python": "/opt/conda/bin/python",
        "ssh_executable": "C:/Windows/System32/OpenSSH/ssh.exe",
        "ssh_key_path": str(tmp_path / "id_ed25519"),
        "known_hosts_path": str(tmp_path / "known_hosts"),
    }

    command = build_remote_stream_command(server, "/remote/stream_allowed_files.py")

    assert command[-2:] == ["/opt/conda/bin/python", "/remote/stream_allowed_files.py"]
    assert "root@gpu.example.test" in command


def test_scientific_hold_detects_reversed_li_full_gate_direction():
    rows = []
    for seed in range(15, 25):
        rows.append(
            {
                "formal_stage": "li_m_o_ablation",
                "dataset": "limo",
                "method": "interval_hit_greedy",
                "group_key": "element_system_current",
                "seed": seed,
                "K": 30,
                "AUTC": 0.80,
            }
        )
        rows.append(
            {
                "formal_stage": "li_m_o_ablation",
                "dataset": "limo",
                "method": "energy_gated_da_tpp",
                "group_key": "element_system_current",
                "seed": seed,
                "K": 30,
                "AUTC": 0.79,
            }
        )

    holds = scientific_holds_from_gpu_analysis(rows)

    assert len(holds) == 1
    assert "reverses" in holds[0]


def test_authorized_shutdown_resume_skips_server_with_recorded_acceptance(tmp_path):
    output = tmp_path / "finalizer"
    output.mkdir()
    (output / "SHUTDOWN_AUTHORIZED.json").write_text(
        json.dumps({"authorized_at": "2026-07-19T00:00:00Z"}), encoding="utf-8"
    )
    (output / "shutdown_dft.json").write_text(
        json.dumps({"shutdown_started": True}), encoding="utf-8"
    )
    calls: list[str] = []
    config = {
        "servers": {
            "gpu": {"role": "gpu"},
            "dft": {"role": "dft"},
        }
    }

    result = resume_authorized_shutdown(
        config,
        output,
        shutdown_runner=lambda server: (
            calls.append(server["role"]) or {"shutdown_started": True}
        ),
    )

    assert calls == ["gpu"]
    assert result["state"] == "COMPLETE_SHUTDOWN_REQUESTED"
    assert (output / "shutdown_gpu.json").is_file()
    assert (output / "FINALIZATION_COMPLETE.json").is_file()
