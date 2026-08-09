"""Monitor approved remote experiment queues and apply bounded repairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class WatchdogDecision:
    """A deterministic action derived from one remote server snapshot."""

    action: str
    reason: str
    repair_allowed: bool


def decide_action(
    snapshot: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> WatchdogDecision:
    """Return the only action permitted by the current evidence and policy."""

    if snapshot.get("ssh_error"):
        return WatchdogDecision("ALERT", "ssh_error", False)
    if snapshot.get("error_flags"):
        return WatchdogDecision("ALERT", "numerical_or_accelerator_error", False)
    if snapshot.get("pause_reason"):
        return WatchdogDecision("ALERT", "controller_safety_pause", False)
    if int(snapshot.get("same_config_failure_max", 0)) >= 3:
        return WatchdogDecision("ALERT", "three_same_config_failures", False)
    if snapshot.get("live_running_pids") and not snapshot.get("controller_alive"):
        return WatchdogDecision("ALERT", "orphan_worker", False)
    if int(snapshot.get("free_bytes", 0)) < int(policy["minimum_free_bytes"]):
        return WatchdogDecision("ALERT", "free_disk_below_limit", False)
    if int(snapshot.get("root_bytes", 0)) > int(policy["maximum_root_bytes"]):
        return WatchdogDecision("ALERT", "output_root_exceeds_budget", False)

    counts = snapshot["status_counts"]
    if policy["role"] == "dft":
        if int(counts.get("PENDING", 0)) or int(counts.get("RUNNING", 0)):
            return WatchdogDecision("ALERT", "dft_monitor_only", False)
        if not snapshot.get("required_protocol_present", False):
            return WatchdogDecision("SCIENTIFIC_HOLD", "dft_protocol_not_frozen", False)
        return WatchdogDecision("COMPLETE", "manifest_complete", False)

    if snapshot.get("controller_alive"):
        return WatchdogDecision("HEALTHY", "controller_alive", False)
    if int(counts.get("PENDING", 0)):
        return WatchdogDecision("RESUME_GPU_CONTROLLER", "safe_controller_absence", True)
    return WatchdogDecision("STAGE_COMPLETE", "no_pending_gpu_work", False)


def build_ssh_command(server: Mapping[str, Any]) -> list[str]:
    """Build a non-interactive, strict-host-key SSH command without secrets."""

    key_path = str(Path(str(server["ssh_key_path"])).expanduser())
    known_hosts = str(Path(str(server["known_hosts_path"])).expanduser())
    return [
        str(server.get("ssh_executable", "ssh.exe")),
        "-F",
        "NUL",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-i",
        key_path,
        "-p",
        str(server["port"]),
        f"{server['user']}@{server['host']}",
    ]


def _remote_snapshot_script(server: Mapping[str, Any]) -> str:
    manifest = str(server["manifest"])
    manifest_pointer = str(server.get("manifest_pointer", ""))
    root = str(server["root"])
    required_protocol = str(server.get("required_protocol_path", ""))
    return textwrap.dedent(
        fr"""
        # WATCHDOG_REMOTE_SNAPSHOT
        import csv
        import hashlib
        import json
        import os
        import re
        import shlex
        import shutil
        import subprocess
        from collections import Counter
        from datetime import datetime, timezone
        from pathlib import Path

        configured_manifest = Path({manifest!r})
        manifest_pointer = Path({manifest_pointer!r}) if {manifest_pointer!r} else None
        if manifest_pointer is not None:
            active_manifest = manifest_pointer.read_text(encoding='utf-8').strip()
            if not active_manifest:
                raise ValueError('active manifest pointer is empty')
            manifest = Path(active_manifest)
            if not manifest.is_absolute():
                raise ValueError('active manifest pointer must contain an absolute path')
        else:
            manifest = configured_manifest
        root = Path({root!r})
        required_protocol = Path({required_protocol!r}) if {required_protocol!r} else None

        def pid_alive(value):
            try:
                pid = int(str(value))
                if pid <= 0:
                    return False
                os.kill(pid, 0)
                return True
            except (TypeError, ValueError, OSError, ProcessLookupError):
                return False

        def tail_text(path, limit=262144):
            try:
                with Path(path).open('rb') as handle:
                    handle.seek(0, 2)
                    size = handle.tell()
                    handle.seek(max(0, size - limit))
                    return handle.read().decode('utf-8', errors='ignore').lower()
            except OSError:
                return ''

        with manifest.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter((row.get('status') or '').strip() for row in rows)

        process_result = subprocess.run(
            ['ps', '-eo', 'pid=,args='],
            text=True,
            capture_output=True,
            check=False,
        )
        controller_pids = []
        controller_audit_dirs = []
        active_vasp_pids = []
        for line in process_result.stdout.splitlines():
            fields = line.strip().split(None, 1)
            if len(fields) != 2 or not fields[0].isdigit():
                continue
            pid, command_text = int(fields[0]), fields[1]
            if 'run_audited_job_queue.py' in command_text and str(manifest) in command_text:
                controller_pids.append(pid)
                try:
                    arguments = shlex.split(command_text)
                    if '--audit-dir' in arguments:
                        controller_audit_dirs.append(arguments[arguments.index('--audit-dir') + 1])
                except (ValueError, IndexError):
                    pass
            executable = Path(command_text.split(None, 1)[0]).name
            if executable in {{'vasp_std', 'vasp_gam', 'vasp_ncl'}}:
                active_vasp_pids.append(pid)

        live_running_pids = []
        stale_running_job_ids = []
        for row in rows:
            if row.get('status') != 'RUNNING':
                continue
            if pid_alive(row.get('pid')):
                live_running_pids.append(int(row['pid']))
            else:
                stale_running_job_ids.append(row.get('job_id', ''))

        candidate_status_files = []
        if controller_audit_dirs:
            for audit_dir in controller_audit_dirs:
                directory = Path(audit_dir)
                candidate_status_files.extend(directory.glob('progress_*.json'))
                candidate_status_files.extend(directory.glob('controller_summary.json'))
        else:
            audit_root = root / 'audit'
            if audit_root.is_dir():
                candidate_status_files.extend(audit_root.rglob('progress_*.json'))
                candidate_status_files.extend(audit_root.rglob('controller_summary.json'))
        latest_status = {{}}
        latest_status_path = ''
        if candidate_status_files:
            latest = max(candidate_status_files, key=lambda path: path.stat().st_mtime_ns)
            latest_status_path = str(latest)
            try:
                latest_status = json.loads(latest.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                latest_status = {{}}

        error_flags = set()
        for row in rows:
            if row.get('status') not in {{'RUNNING', 'FAILED'}}:
                continue
            paths = [Path(row.get('log_path') or '')]
            output = Path(row.get('output_path') or '')
            paths.extend(output / name for name in ('run.log', 'traceback.txt', 'error.log'))
            text = '\n'.join(tail_text(path) for path in paths if str(path))
            if 'out of memory' in text or 'cuda oom' in text:
                error_flags.add('oom')
            if 'cuda error' in text or 'cublas_status' in text:
                error_flags.add('cuda')
            if re.search(r'\bnan\b', text):
                error_flags.add('nan')

        failure_counts = Counter(
            row.get('config_hash', '') for row in rows if row.get('status') == 'FAILED'
        )
        du = subprocess.run(
            ['du', '-sb', str(root)],
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            root_bytes = int(du.stdout.split()[0])
        except (ValueError, IndexError):
            root_bytes = 0

        snapshot = {{
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'manifest_path': str(manifest),
            'configured_manifest_path': str(configured_manifest),
            'manifest_pointer_path': str(manifest_pointer) if manifest_pointer else None,
            'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'manifest_rows': len(rows),
            'status_counts': {{
                status: int(counts.get(status, 0))
                for status in ('PENDING', 'RUNNING', 'DONE', 'FAILED', 'CANCELLED')
            }},
            'controller_alive': bool(controller_pids),
            'controller_pids': controller_pids,
            'controller_audit_dirs': controller_audit_dirs,
            'live_running_pids': live_running_pids,
            'stale_running_job_ids': stale_running_job_ids,
            'active_vasp_pids': active_vasp_pids,
            'error_flags': sorted(error_flags),
            'same_config_failure_max': max(failure_counts.values(), default=0),
            'pause_reason': latest_status.get('pause_reason'),
            'throughput_per_hour': latest_status.get('throughput_jobs_per_hour'),
            'latest_status_path': latest_status_path,
            'root_bytes': root_bytes,
            'free_bytes': int(shutil.disk_usage(root).free),
            'required_protocol_present': bool(required_protocol and required_protocol.is_file()),
        }}
        print(json.dumps(snapshot, sort_keys=True))
        """
    ).strip() + "\n"


def _remote_resume_script(
    server: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
) -> str:
    controller = server["controller"]
    return textwrap.dedent(
        fr"""
        # WATCHDOG_REMOTE_RESUME
        import csv
        import fcntl
        import hashlib
        import json
        import os
        import shutil
        import subprocess
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        configured_manifest = Path({str(server['manifest'])!r})
        manifest_pointer = Path({str(server.get('manifest_pointer', ''))!r}) if {str(server.get('manifest_pointer', ''))!r} else None
        if manifest_pointer is not None:
            active_manifest = manifest_pointer.read_text(encoding='utf-8').strip()
            if not active_manifest:
                raise ValueError('active manifest pointer is empty')
            manifest = Path(active_manifest)
            if not manifest.is_absolute():
                raise ValueError('active manifest pointer must contain an absolute path')
        else:
            manifest = configured_manifest
        root = Path({str(server['root'])!r})
        expected_sha256 = {expected_manifest_sha256!r}
        minimum_free_bytes = {int(server['minimum_free_bytes'])}
        maximum_root_bytes = {int(server['maximum_root_bytes'])}
        audit_root = root / 'audit'
        audit_root.mkdir(parents=True, exist_ok=True)

        def pid_alive(value):
            try:
                pid = int(str(value))
                if pid <= 0:
                    return False
                os.kill(pid, 0)
                return True
            except (TypeError, ValueError, OSError, ProcessLookupError):
                return False

        lock_handle = (audit_root / 'watchdog_resume.lock').open('a+', encoding='utf-8')
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({{'started': False, 'reason': 'resume_lock_busy'}}))
            raise SystemExit(0)

        current_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if current_sha256 != expected_sha256:
            print(json.dumps({{'started': False, 'reason': 'manifest_changed'}}))
            raise SystemExit(0)

        with manifest.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if not any(row.get('status') == 'PENDING' for row in rows):
            print(json.dumps({{'started': False, 'reason': 'no_pending_work'}}))
            raise SystemExit(0)
        if any(row.get('status') == 'RUNNING' and pid_alive(row.get('pid')) for row in rows):
            print(json.dumps({{'started': False, 'reason': 'live_running_worker'}}))
            raise SystemExit(0)

        processes = subprocess.run(
            ['ps', '-eo', 'args='], text=True, capture_output=True, check=False
        ).stdout
        if any(
            'run_audited_job_queue.py' in line and str(manifest) in line
            for line in processes.splitlines()
        ):
            print(json.dumps({{'started': False, 'reason': 'controller_already_alive'}}))
            raise SystemExit(0)

        free_bytes = int(shutil.disk_usage(root).free)
        du = subprocess.run(['du', '-sb', str(root)], text=True, capture_output=True, check=False)
        try:
            root_bytes = int(du.stdout.split()[0])
        except (ValueError, IndexError):
            root_bytes = maximum_root_bytes + 1
        if free_bytes < minimum_free_bytes or root_bytes > maximum_root_bytes:
            print(json.dumps({{'started': False, 'reason': 'disk_safety_gate'}}))
            raise SystemExit(0)

        token = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        audit_dir = audit_root / f'gpu_development_watchdog_{{token}}'
        audit_dir.mkdir(exist_ok=False)
        log_path = audit_dir / 'controller.log'
        pid_path = audit_dir / f'controller_{{token}}.pid'
        command = [
            {str(controller['python'])!r},
            {str(controller['script'])!r},
            '--manifest',
            str(manifest),
            '--audit-dir',
            str(audit_dir),
            '--output-root',
            str(root),
            *{list(controller['arguments'])!r},
        ]
        with log_path.open('x', encoding='utf-8') as log_handle:
            process = subprocess.Popen(
                command,
                cwd={str(controller['cwd'])!r},
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(str(process.pid) + '\n', encoding='utf-8')
        time.sleep(2.0)
        alive = pid_alive(process.pid)
        print(json.dumps({{
            'started': alive,
            'reason': 'started' if alive else 'controller_exited_during_start',
            'pid': process.pid,
            'audit_dir': str(audit_dir),
            'log_path': str(log_path),
            'pid_path': str(pid_path),
            'manifest_sha256': current_sha256,
        }}, sort_keys=True))
        """
    ).strip() + "\n"


def _run_remote_python(
    server: Mapping[str, Any],
    script: str,
    *,
    ssh_runner: Any,
) -> dict[str, Any]:
    command = [*build_ssh_command(server), str(server.get("remote_python", "python3")), "-"]
    try:
        completed = ssh_runner(
            command,
            input=script,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"ssh_error": f"{type(error).__name__}: {error}"}
    if int(completed.returncode) != 0:
        error_text = str(completed.stderr).strip()[-2000:]
        return {"ssh_error": f"exit_code={completed.returncode}: {error_text}"}
    lines = [line for line in str(completed.stdout).splitlines() if line.strip()]
    if not lines:
        return {"ssh_error": "remote command returned no JSON"}
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as error:
        return {"ssh_error": f"invalid remote JSON: {error}"}


def collect_snapshot(
    server: Mapping[str, Any],
    *,
    ssh_runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Collect a read-only snapshot from one configured server."""

    return _run_remote_python(
        server,
        _remote_snapshot_script(server),
        ssh_runner=ssh_runner,
    )


def resume_gpu_controller(
    server: Mapping[str, Any],
    *,
    expected_manifest_sha256: str,
    ssh_runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Start only the unchanged approved GPU controller after remote rechecks."""

    return _run_remote_python(
        server,
        _remote_resume_script(
            server,
            expected_manifest_sha256=expected_manifest_sha256,
        ),
        ssh_runner=ssh_runner,
    )


def _git_commit() -> str | None:
    project_root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = output_dir / "checks"
    checks.mkdir(exist_ok=True)
    token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    (checks / f"check_{token}.json").write_text(text, encoding="utf-8")
    temporary = output_dir / "last_check.json.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_dir / "last_check.json")
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True) + "\n")


def run_watchdog(
    config_path: Path,
    output_dir: Path,
    *,
    ssh_runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Run one complete inspection and any bounded, evidence-backed repair."""

    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "git_commit": _git_commit(),
        "servers": {},
    }
    for name, server in config["servers"].items():
        snapshot = collect_snapshot(server, ssh_runner=ssh_runner)
        decision = decide_action(snapshot, server)
        entry: dict[str, Any] = {
            "action": decision.action,
            "reason": decision.reason,
            "repair_allowed": decision.repair_allowed,
            "snapshot": snapshot,
        }
        if decision.repair_allowed and bool(server.get("auto_resume", False)):
            entry["repair"] = resume_gpu_controller(
                server,
                expected_manifest_sha256=str(snapshot["manifest_sha256"]),
                ssh_runner=ssh_runner,
            )
        report["servers"][name] = entry
    _write_report(output_dir, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_watchdog(args.config, args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
