"""Safely gate, validate, recover, and analyze completed server experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import textwrap
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import (
    candidate_sequence_hash,
    left_continuous_autc,
    recovery_at,
)
from analysis.recompute_statistics import build_round_trajectory, paired_statistics
from analysis.server_experiment_watchdog import (
    build_ssh_command,
    collect_snapshot,
)


RESTRICTED_VASP_BASENAMES = frozenset(
    {
        "POTCAR",
        "WAVECAR",
        "CHGCAR",
        "CHG",
        "AECCAR0",
        "AECCAR1",
        "AECCAR2",
    }
)
RECONCILED_CONTROLLER_CRASH_REASON = (
    "reconciled_verified_done_output_after_controller_crash;"
    "process_exit_code_unavailable"
)


@dataclass(frozen=True)
class FinalizationGate:
    ready: bool
    state: str
    reasons: tuple[str, ...]


def completion_evidence_kind(row: Mapping[str, Any]) -> str:
    """Classify the only two accepted terminal evidence patterns."""

    status = str(row.get("status") or "").strip()
    exit_code = str(row.get("exit_code") or "").strip()
    reason = str(row.get("failure_reason") or "").strip()
    if status == "DONE" and exit_code == "0" and not reason:
        return "standard_exit_zero"
    if (
        status == "DONE"
        and not exit_code
        and reason == RECONCILED_CONTROLLER_CRASH_REASON
    ):
        return "verified_reconciled_output"
    raise ValueError(
        "unacceptable completion evidence: "
        f"status={status!r}, exit_code={exit_code!r}, failure_reason={reason!r}"
    )


def authorize_shutdown(gate: Any, summary: Mapping[str, Any]) -> bool:
    """Return true only after every non-destructive finalization gate passed."""

    required = (
        "remote_validation_passed",
        "archive_sha256_verified",
        "allowed_inventory_verified",
        "local_analysis_passed",
    )
    return bool(
        getattr(gate, "ready", False)
        and all(bool(summary.get(field, False)) for field in required)
        and not summary.get("scientific_holds")
    )


def build_remote_package_script(
    server: Mapping[str, Any],
    package_base: str,
) -> str:
    """Generate the remote, evidence-preserving package operation."""

    root = str(server["root"])
    manifest = str(server["manifest"])
    pointer = str(server.get("manifest_pointer", ""))
    role = str(server["role"])
    restricted = sorted(RESTRICTED_VASP_BASENAMES)
    reconciled_reason = RECONCILED_CONTROLLER_CRASH_REASON
    stream_helper = (
        textwrap.dedent(
            rf"""
        import csv
        import hashlib
        import json
        import sys
        import tarfile
        from datetime import datetime, timezone
        from pathlib import Path, PurePosixPath

        root = Path({root!r}).resolve()
        package_dir = Path(__file__).resolve().parent
        inventory_path = package_dir / 'allowed_files_sha256.csv'

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        with inventory_path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        with tarfile.open(fileobj=sys.stdout.buffer, mode='w|', format=tarfile.PAX_FORMAT) as archive:
            for row in rows:
                relative = PurePosixPath(row['relative_path'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError(f'unsafe inventory path: {{relative}}')
                path = root.joinpath(*relative.parts).resolve()
                path.relative_to(root)
                if not path.is_file() or path.is_symlink():
                    raise RuntimeError(f'allowed source is no longer a regular file: {{path}}')
                if path.stat().st_size != int(row['size_bytes']):
                    raise RuntimeError(f'allowed source size changed: {{path}}')
                if sha256_file(path) != row['sha256']:
                    raise RuntimeError(f'allowed source hash changed: {{path}}')
                archive.add(path, arcname=relative.as_posix(), recursive=False)
        sys.stdout.buffer.flush()
        marker = {{
            'schema': 'REMOTE_COMPLETION_STREAM_V1',
            'completed_at': datetime.now(timezone.utc).isoformat(),
            'inventory_sha256': sha256_file(inventory_path),
            'streamed_file_count': len(rows),
            'streamed_bytes': sum(int(row['size_bytes']) for row in rows),
        }}
        marker_path = package_dir / 'stream_completed.json'
        if marker_path.is_file():
            existing = json.loads(marker_path.read_text(encoding='utf-8'))
            if existing.get('inventory_sha256') != marker['inventory_sha256']:
                raise RuntimeError('existing stream marker uses a different inventory')
        else:
            with marker_path.open('x', encoding='utf-8') as handle:
                handle.write(json.dumps(marker, indent=2, sort_keys=True) + '\n')
        """
        ).strip()
        + "\n"
    )
    return (
        textwrap.dedent(
            rf"""
        # REMOTE_COMPLETION_PACKAGE_V1
        import csv
        import hashlib
        import json
        import os
        import subprocess
        from collections import Counter
        from datetime import datetime, timezone
        from pathlib import Path

        role = {role!r}
        root = Path({root!r}).resolve()
        configured_manifest = Path({manifest!r})
        manifest_pointer = Path({pointer!r}) if {pointer!r} else None
        package_base = Path({str(package_base)!r})
        restricted_names = set({restricted!r})
        reconciled_reason = {reconciled_reason!r}

        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open('rb') as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()

        def sha256_tree(directory):
            directory = Path(directory)
            digest = hashlib.sha256()
            for path in sorted(item for item in directory.rglob('*') if item.is_file()):
                relative = path.relative_to(directory).as_posix().encode('utf-8')
                digest.update(len(relative).to_bytes(8, 'big'))
                digest.update(relative)
                digest.update(bytes.fromhex(sha256_file(path)))
            return digest.hexdigest()

        def pid_alive(value):
            try:
                pid = int(str(value))
                if pid <= 0:
                    return False
                os.kill(pid, 0)
                return True
            except (TypeError, ValueError, OSError, ProcessLookupError):
                return False

        if manifest_pointer is not None:
            active = manifest_pointer.read_text(encoding='utf-8').strip()
            if not active:
                raise RuntimeError('active manifest pointer is empty')
            manifest_path = Path(active)
        else:
            manifest_path = configured_manifest
        manifest_path = manifest_path.resolve()
        manifest_path.relative_to(root)
        with manifest_path.open(newline='', encoding='utf-8-sig') as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise RuntimeError('active manifest is empty')
        status_counts = Counter((row.get('status') or '').strip() for row in rows)
        if status_counts != Counter({{'DONE': len(rows)}}):
            raise RuntimeError(f'manifest is not fully DONE: {{dict(status_counts)}}')
        completion_evidence = {{}}
        for row in rows:
            exit_code = (row.get('exit_code') or '').strip()
            reason = (row.get('failure_reason') or '').strip()
            if exit_code == '0' and not reason:
                kind = 'standard_exit_zero'
            elif not exit_code and reason == reconciled_reason:
                kind = 'verified_reconciled_output'
            else:
                raise RuntimeError(
                    f"unacceptable completion evidence for {{row.get('job_id')}}: "
                    f"exit_code={{exit_code!r}}, failure_reason={{reason!r}}"
                )
            completion_evidence[row.get('job_id')] = kind
        if any(pid_alive(row.get('pid')) for row in rows):
            raise RuntimeError('a manifest worker PID is still alive')

        processes = subprocess.run(
            ['ps', '-eo', 'args='], text=True, capture_output=True, check=False
        ).stdout.splitlines()
        if any('run_audited_job_queue.py' in line and str(manifest_path) in line for line in processes):
            raise RuntimeError('queue controller is still alive')
        if role == 'dft' and any(
            Path(line.strip().split(None, 1)[0]).name in {{'vasp_std', 'vasp_gam', 'vasp_ncl'}}
            for line in processes if line.strip()
        ):
            raise RuntimeError('a VASP process is still alive')

        manifest_sha256 = sha256_file(manifest_path)
        package_dir = package_base / manifest_sha256
        ready_path = package_dir / 'package_ready.json'
        if ready_path.is_file():
            ready = json.loads(ready_path.read_text(encoding='utf-8'))
            for path_key, hash_key in (
                ('inventory_path', 'inventory_sha256'),
                ('excluded_inventory_path', 'excluded_inventory_sha256'),
                ('stream_script_path', 'stream_script_sha256'),
            ):
                evidence_path = Path(ready[path_key])
                if not evidence_path.is_file() or sha256_file(evidence_path) != ready[hash_key]:
                    raise RuntimeError(f'existing ready package failed verification: {{path_key}}')
            print(json.dumps(ready, sort_keys=True))
            raise SystemExit(0)
        if package_dir.exists():
            raise RuntimeError(f'incomplete package directory is preserved for audit: {{package_dir}}')
        package_dir.mkdir(parents=True, exist_ok=False)

        job_validation = []
        for row in rows:
            output = Path(row['output_path']).resolve()
            output.relative_to(root)
            if not output.is_dir():
                raise FileNotFoundError(output)
            kind = completion_evidence[row.get('job_id')]
            if kind == 'verified_reconciled_output':
                status_path = output / 'status.json'
                status_payload = json.loads(status_path.read_text(encoding='utf-8'))
                if status_payload.get('status') != 'DONE':
                    raise RuntimeError(
                        f"reconciled output status is not DONE for {{row.get('job_id')}}"
                    )
            expected = (row.get('sha256') or '').strip()
            if len(expected) != 64:
                raise RuntimeError(f"{{row.get('job_id')}} has no output tree SHA-256")
            actual = sha256_tree(output)
            if actual != expected:
                raise RuntimeError(
                    f"output tree hash mismatch for {{row.get('job_id')}}: {{expected}} != {{actual}}"
                )
            job_validation.append({{
                'job_id': row.get('job_id'),
                'output_path': str(output),
                'sha256': actual,
                'completion_evidence_kind': kind,
            }})

        allowed = []
        excluded = []
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                excluded.append((path.relative_to(root).as_posix(), 0, '', 'symlink'))
                continue
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            digest = sha256_file(path)
            name = path.name.upper()
            if name in restricted_names or name.startswith('POTCAR.'):
                excluded.append((relative, stat.st_size, digest, 'restricted_vasp_artifact'))
            else:
                allowed.append((path, relative, stat.st_size, digest))

        inventory_path = package_dir / 'allowed_files_sha256.csv'
        with inventory_path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['relative_path', 'size_bytes', 'sha256'])
            writer.writerows((relative, size, digest) for _, relative, size, digest in allowed)
        excluded_path = package_dir / 'excluded_artifacts_sha256.csv'
        with excluded_path.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(['relative_path', 'size_bytes', 'sha256', 'reason'])
            writer.writerows(excluded)

        stream_script_path = package_dir / 'stream_allowed_files.py'
        stream_script_path.write_text({stream_helper!r}, encoding='utf-8')
        stream_completion_path = package_dir / 'stream_completed.json'
        if sha256_file(manifest_path) != manifest_sha256:
            raise RuntimeError('manifest changed during packaging')

        ready = {{
            'schema': 'REMOTE_COMPLETION_PACKAGE_V1',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'role': role,
            'root': str(root),
            'manifest_path': str(manifest_path),
            'manifest_relative_path': manifest_path.relative_to(root).as_posix(),
            'manifest_sha256': manifest_sha256,
            'manifest_rows': len(rows),
            'status_counts': dict(status_counts),
            'job_tree_hashes_verified': len(job_validation),
            'inventory_path': str(inventory_path),
            'inventory_sha256': sha256_file(inventory_path),
            'excluded_inventory_path': str(excluded_path),
            'excluded_inventory_sha256': sha256_file(excluded_path),
            'stream_script_path': str(stream_script_path),
            'stream_script_sha256': sha256_file(stream_script_path),
            'stream_completion_path': str(stream_completion_path),
            'allowed_file_count': len(allowed),
            'allowed_bytes': sum(item[2] for item in allowed),
            'excluded_file_count': len(excluded),
            'job_validation': job_validation,
        }}
        temporary = package_dir / 'package_ready.json.tmp'
        temporary.write_text(
            json.dumps(ready, indent=2, sort_keys=True) + '\n', encoding='utf-8'
        )
        temporary.replace(ready_path)
        print(json.dumps(ready, sort_keys=True))
        """
        ).strip()
        + "\n"
    )


def assess_role_completion(role: str, snapshot: Mapping[str, Any]) -> FinalizationGate:
    """Assess one server independently so completed resources need not idle."""

    reasons: list[str] = []
    if snapshot.get("ssh_error"):
        return FinalizationGate(False, "BLOCKED", (f"{role}:ssh_error",))
    if not snapshot.get("required_protocol_present", False):
        reasons.append(f"{role}:protocol_missing")
    if snapshot.get("error_flags") or snapshot.get("pause_reason"):
        reasons.append(f"{role}:remote_error_or_pause")
    if int(snapshot.get("same_config_failure_max", 0)):
        reasons.append(f"{role}:recorded_failed_configuration")
    if snapshot.get("stale_running_job_ids"):
        reasons.append(f"{role}:stale_running_job")

    counts = snapshot.get("status_counts") or {}
    failed = int(counts.get("FAILED", 0))
    cancelled = int(counts.get("CANCELLED", 0))
    pending = int(counts.get("PENDING", 0))
    running = int(counts.get("RUNNING", 0))
    done = int(counts.get("DONE", 0))
    rows = int(snapshot.get("manifest_rows", 0))
    if failed or cancelled:
        reasons.append(f"{role}:failed_or_cancelled")
    if pending or running:
        reasons.append(f"{role}:pending_or_running")
    if rows <= 0 or done != rows:
        reasons.append(f"{role}:done_count_does_not_match_manifest")
    if (
        snapshot.get("controller_alive")
        or snapshot.get("live_running_pids")
        or snapshot.get("active_vasp_pids")
    ):
        reasons.append(f"{role}:controller_or_worker_still_alive")
    if not reasons:
        return FinalizationGate(True, "READY", ())
    blocking_markers = (
        "failed",
        "cancelled",
        "error",
        "pause",
        "protocol_missing",
        "ssh_error",
        "stale",
        "missing_server_snapshot",
    )
    blocked = any(
        any(marker in reason for marker in blocking_markers) for reason in reasons
    )
    return FinalizationGate(False, "BLOCKED" if blocked else "WAIT", tuple(reasons))


def assess_completion(snapshots: Mapping[str, Mapping[str, Any]]) -> FinalizationGate:
    """Require both approved manifests to be fully successful and quiescent."""

    reasons: list[str] = []
    states: list[str] = []
    for role in ("gpu", "dft"):
        snapshot = snapshots.get(role)
        if snapshot is None:
            reasons.append(f"{role}:missing_server_snapshot")
            states.append("BLOCKED")
            continue
        gate = assess_role_completion(role, snapshot)
        reasons.extend(gate.reasons)
        states.append(gate.state)
    if not reasons:
        return FinalizationGate(True, "READY", ())
    return FinalizationGate(
        False,
        "BLOCKED" if "BLOCKED" in states else "WAIT",
        tuple(reasons),
    )


def is_restricted_artifact(path: Path | PurePosixPath) -> bool:
    """Return whether an artifact must stay off Git and the local archive."""

    name = path.name.upper()
    return name in RESTRICTED_VASP_BASENAMES or name.startswith("POTCAR.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory with the queue controller's path-aware algorithm."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def safe_extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract only regular files into a new directory without path traversal."""

    archive_path = Path(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()
    try:
        with tarfile.open(archive_path, mode="r") as archive:
            for member in archive.getmembers():
                relative = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise ValueError(f"unsafe archive member: {member.name}")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise ValueError(f"archive would overwrite a local file: {target}")
                resolved_parent = target.parent.resolve()
                try:
                    resolved_parent.relative_to(destination_root)
                except ValueError as error:
                    raise ValueError(f"unsafe archive member: {member.name}") from error
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"archive member has no file body: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
    except Exception:
        # Preserve a failed extraction for audit; the caller will not reuse it.
        raise


def verify_allowed_inventory(
    root: Path,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, int]:
    """Verify an extracted payload exactly against its remote file inventory."""

    root = Path(root).resolve()
    expected: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        relative = PurePosixPath(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe inventory path: {relative}")
        if is_restricted_artifact(relative):
            raise ValueError(f"restricted artifact listed for transfer: {relative}")
        key = relative.as_posix()
        if key in expected:
            raise ValueError(f"duplicate inventory path: {key}")
        expected[key] = row

    actual = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
    }
    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    if extra:
        raise ValueError(f"unexpected local files: {extra[:10]}")
    if missing:
        raise ValueError(f"missing local files: {missing[:10]}")

    total_bytes = 0
    for relative, row in expected.items():
        path = actual[relative]
        size = path.stat().st_size
        expected_size = int(row["size_bytes"])
        if size != expected_size:
            raise ValueError(
                f"size mismatch for {relative}: expected {expected_size}, found {size}"
            )
        digest = _sha256_file(path)
        expected_digest = str(row["sha256"])
        if digest != expected_digest:
            raise ValueError(
                f"hash mismatch for {relative}: expected {expected_digest}, found {digest}"
            )
        total_bytes += size
    return {"verified_files": len(expected), "verified_bytes": total_bytes}


def map_remote_path(remote_root: str, local_root: Path, remote_path: str) -> Path:
    """Map an absolute remote path into its verified local payload root."""

    root = PurePosixPath(remote_root)
    path = PurePosixPath(remote_path)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"remote path is outside remote root: {remote_path}"
        ) from error
    return Path(local_root).joinpath(*relative.parts)


def build_scp_command(
    server: Mapping[str, Any],
    remote_path: str,
    local_path: Path,
) -> list[str]:
    """Build a non-interactive SCP command with the watchdog's SSH policy."""

    ssh_executable = Path(str(server.get("ssh_executable", "ssh.exe")))
    scp_executable = str(ssh_executable.with_name("scp.exe"))
    key_path = str(Path(str(server["ssh_key_path"])).expanduser())
    known_hosts = str(Path(str(server["known_hosts_path"])).expanduser())
    return [
        scp_executable,
        "-F",
        "NUL",
        "-i",
        key_path,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-P",
        str(server["port"]),
        f"{server['user']}@{server['host']}:{remote_path}",
        str(Path(local_path)),
    ]


def build_remote_stream_command(
    server: Mapping[str, Any],
    remote_script_path: str,
) -> list[str]:
    """Build the SSH command whose stdout is the evidence tar stream."""

    return [
        *build_ssh_command(server),
        str(server.get("remote_python", "python3")),
        str(remote_script_path),
    ]


def _required_csv(path: Path, name: str) -> pd.DataFrame:
    file_path = path / name
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return pd.read_csv(file_path)


def analyze_gpu_job(output: Path, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute one trajectory summary from raw history and route records."""

    output = Path(output)
    config_path = output / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_values = {
        "name": str(manifest_row["dataset"]),
        "method": str(manifest_row["method"]),
        "seed": int(manifest_row["seed"]),
    }
    for key, expected in expected_values.items():
        if config.get(key) != expected:
            raise ValueError(
                f"{config_path}: {key}={config.get(key)!r}, expected {expected!r}"
            )

    budget = int(config["budget"])
    batch_size = int(config["batch_size"])
    total_targets = int(config["target_count"])
    history = _required_csv(output, "al_history.csv")
    if len(history) != budget:
        raise ValueError(
            f"{output}: history has {len(history)} rows, expected {budget}"
        )
    if history["id"].astype(str).duplicated().any():
        raise ValueError(f"{output}: duplicate candidate IDs")
    labels = pd.to_numeric(history["target_label"], errors="raise").astype(int)
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{output}: target_label must be binary")

    trajectory = build_round_trajectory(history, batch_size=batch_size, budget=budget)
    rounds = len(trajectory)
    diagnostics = _required_csv(output, "round_diagnostics.csv")
    predictions = _required_csv(output, "prediction_manifest.csv")
    checkpoints = _required_csv(output, "checkpoint_manifest.csv")
    for name, frame in (
        ("round_diagnostics.csv", diagnostics),
        ("prediction_manifest.csv", predictions),
        ("checkpoint_manifest.csv", checkpoints),
    ):
        if len(frame) != rounds:
            raise ValueError(
                f"{output / name}: expected {rounds} rows, found {len(frame)}"
            )

    routes = diagnostics["route"].astype(str)
    allowed_routes = {"threshold_greedy", "diversity_aware"}
    if not set(routes).issubset(allowed_routes):
        raise ValueError(f"{output}: unexpected route values")
    unique_groups = pd.to_numeric(
        diagnostics["selected_unique_groups"], errors="raise"
    ).astype(int)
    replacements = pd.to_numeric(
        diagnostics["correction_replacement_count"], errors="raise"
    ).astype(int)
    correction_gain = pd.to_numeric(
        diagnostics["correction_target_gain"], errors="raise"
    ).astype(int)

    queries = trajectory["oracle_evaluations"].to_numpy(dtype=int)
    recoveries = trajectory["cumulative_target_count"].to_numpy(dtype=int)
    result: dict[str, Any] = {
        "job_id": str(manifest_row["job_id"]),
        "formal_stage": str(manifest_row.get("formal_stage", "")),
        "dataset": str(manifest_row["dataset"]),
        "method": str(manifest_row["method"]),
        "group_key": str(manifest_row.get("group_key", "")),
        "seed": int(manifest_row["seed"]),
        "K": int(manifest_row["K"]),
        "config_hash": str(manifest_row.get("config_hash", "")),
        "formal_protocol_sha256": str(config.get("formal_protocol_sha256", "")),
        "budget": budget,
        "batch_size": batch_size,
        "total_target_count": total_targets,
        "final_recovery": int(recoveries[-1]),
        "AUTC": left_continuous_autc(queries, recoveries, total_targets, budget),
        "candidate_sequence_sha256": candidate_sequence_hash(
            history["id"].astype(str).tolist()
        ),
        "direct_rounds": int((routes == "threshold_greedy").sum()),
        "correction_rounds": int((routes == "diversity_aware").sum()),
        "effective_replacements": int(replacements.sum()),
        "correction_target_gain": int(correction_gain.sum()),
        "mean_unique_groups_per_batch": float(unique_groups.mean()),
        "repetition_rate": float(
            (batch_size - unique_groups).sum() / (rounds * batch_size)
        ),
        "source_output_path": str(output.resolve()),
    }
    for checkpoint in sorted({80, 160, 240, 320, budget}):
        result[f"recovery_at_{checkpoint}"] = recovery_at(
            queries, recoveries, checkpoint
        )
    return result


_VASP_VERSION = re.compile(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", re.IGNORECASE)
_TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)\s+eV", re.IGNORECASE)


def analyze_dft_job(output: Path, manifest_row: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one transferred VASP static job and extract its final energy."""

    output = Path(output)
    for name in ("INCAR", "KPOINTS", "POSCAR", "OUTCAR"):
        path = output / name
        if not path.is_file():
            raise FileNotFoundError(path)
    outcar = output / "OUTCAR"
    text = outcar.read_text(encoding="utf-8", errors="ignore")
    footer = "General timing and accounting informations for this job"
    if footer not in text:
        raise ValueError(f"{outcar}: VASP timing footer is missing")
    energies = _TOTEN.findall(text)
    if not energies:
        raise ValueError(f"{outcar}: no TOTEN value found")
    version = _VASP_VERSION.search(text)
    if version is None:
        raise ValueError(f"{outcar}: VASP version is missing")
    return {
        "job_id": str(manifest_row["job_id"]),
        "candidate_id": str(manifest_row["candidate_id"]),
        "formula": str(manifest_row["formula"]),
        "functional": str(manifest_row["functional"]),
        "magnetic_initialization": str(manifest_row["magnetic_initialization"]),
        "main_text_selected": str(manifest_row.get("main_text_selected", "")).lower()
        == "true",
        "kpoint_spacing_Ainv": float(manifest_row["kpoint_spacing_Ainv"]),
        "mesh": str(manifest_row["mesh"]),
        "vasp_version": version.group(1),
        "vasp_completed": True,
        "final_total_energy_eV": float(energies[-1]),
        "outcar_sha256": _sha256_file(outcar),
        "source_output_path": str(output.resolve()),
    }


def analyze_recovered_role(
    role: str,
    payload_root: Path,
    validation: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate one recovered manifest and write a raw-evidence analysis table."""

    if role not in {"gpu", "dft"}:
        raise ValueError(f"unsupported role: {role}")
    payload_root = Path(payload_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_relative = PurePosixPath(str(validation["manifest_relative_path"]))
    if manifest_relative.is_absolute() or ".." in manifest_relative.parts:
        raise ValueError("unsafe manifest_relative_path")
    manifest_path = payload_root.joinpath(*manifest_relative.parts)
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected_rows = int(validation["manifest_rows"])
    if len(rows) != expected_rows:
        raise ValueError(
            f"recovered manifest has {len(rows)} rows, expected {expected_rows}"
        )
    if int(validation.get("job_tree_hashes_verified", 0)) != expected_rows:
        raise ValueError("remote output tree validation count is incomplete")

    results: list[dict[str, Any]] = []
    for row in rows:
        evidence_kind = completion_evidence_kind(row)
        output = map_remote_path(
            str(validation["root"]), payload_root, str(row["output_path"])
        )
        if evidence_kind == "verified_reconciled_output":
            status_path = output / "status.json"
            status_payload = json.loads(status_path.read_text(encoding="utf-8"))
            if status_payload.get("status") != "DONE":
                raise ValueError(
                    f"{row.get('job_id')}: reconciled output status is not DONE"
                )
        if role == "gpu":
            expected_hash = str(row.get("sha256") or "")
            actual_hash = sha256_tree(output)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"{row.get('job_id')}: local output tree hash mismatch"
                )
            result = analyze_gpu_job(output, row)
            rounds = int(result["budget"]) // int(result["batch_size"])
            if result["method"] == "interval_hit_greedy" and (
                int(result["direct_rounds"]) != rounds
                or int(result["correction_rounds"]) != 0
            ):
                raise ValueError("Greedy did not bypass every correction route")
            if result["method"] == "always_da_tpp" and (
                int(result["correction_rounds"]) != rounds
                or int(result["direct_rounds"]) != 0
            ):
                raise ValueError("Always-DA-TPP did not use every correction route")
            reported_path = output / "run_metrics.csv"
            if reported_path.is_file():
                reported = pd.read_csv(reported_path)
                if len(reported) != 1:
                    raise ValueError(f"{reported_path}: expected one metrics row")
                difference = float(reported.iloc[0]["AUTC"]) - float(result["AUTC"])
                result["reported_AUTC_difference"] = difference
                if abs(difference) > 1e-12:
                    raise ValueError(f"{row.get('job_id')}: reported AUTC mismatch")
        else:
            result = analyze_dft_job(output, row)
        result["completion_evidence_kind"] = evidence_kind
        results.append(result)

    frame = pd.DataFrame(results)
    filename = (
        "gpu_per_trajectory_analysis.csv"
        if role == "gpu"
        else "dft_static_job_analysis.csv"
    )
    frame.to_csv(output_dir / filename, index=False, lineterminator="\n")
    report = {
        "role": role,
        "status": "PASS",
        "analyzed_jobs": len(results),
        "manifest_path": str(manifest_path),
        "analysis_csv": str((output_dir / filename).resolve()),
    }
    (output_dir / f"{role}_analysis_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def build_shutdown_script() -> str:
    """Return the remote shutdown command used only after local authorization."""

    return (
        textwrap.dedent(
            """
        # FINALIZATION_SHUTDOWN_AUTHORIZED
        import hashlib
        import json
        import os
        import subprocess
        import sys

        os.sync()
        shutdown_script = '/usr/bin/shutdown'
        if not os.path.isfile(shutdown_script):
            raise FileNotFoundError(shutdown_script)
        with open(shutdown_script, 'rb') as handle:
            shutdown_sha256 = hashlib.sha256(handle.read()).hexdigest()
        helper = (
            "import subprocess,time; time.sleep(3); "
            "subprocess.run(['/bin/bash','/usr/bin/shutdown'], check=False)"
        )
        command = [sys.executable, '-c', helper]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(json.dumps({
            'shutdown_started': True,
            'pid': process.pid,
            'method': 'platform_container_shutdown_wrapper',
            'shutdown_script': shutdown_script,
            'shutdown_script_sha256': shutdown_sha256,
        }), flush=True)
        """
        ).strip()
        + "\n"
    )


def scientific_holds_from_gpu_analysis(
    rows: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Detect the pre-registered main-direction reversal before shutdown."""

    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return ["GPU analysis is empty"]
    subset = frame[
        (frame["formal_stage"] == "li_m_o_ablation")
        & (frame["dataset"] == "limo")
        & frame["method"].isin(["energy_gated_da_tpp", "interval_hit_greedy"])
    ]
    pivot = subset.pivot(index="seed", columns="method", values="AUTC")
    required = {"energy_gated_da_tpp", "interval_hit_greedy"}
    if len(pivot) != 10 or not required.issubset(pivot.columns):
        return ["Li-M-O Full Gate versus Greedy paired cohort is incomplete"]
    paired_mean = float(
        (pivot["energy_gated_da_tpp"] - pivot["interval_hit_greedy"]).mean()
    )
    if paired_mean < -1e-12:
        return [
            "Li-M-O Full Gate minus Greedy mean AUTC reverses the legacy conclusion direction "
            f"({paired_mean:.12g})"
        ]
    return []


def _validate_formal_gpu_grid(frame: pd.DataFrame) -> None:
    if len(frame) != 130 or frame["job_id"].duplicated().any():
        raise ValueError("formal GPU analysis must contain 130 unique jobs")
    expected: set[tuple[str, str, str, str, int, int]] = set()
    for seed in range(15, 25):
        for method in (
            "interval_hit_greedy",
            "always_da_tpp",
            "margin_only_gate",
            "group_only_gate",
            "energy_gated_da_tpp",
        ):
            expected.add(
                (
                    "li_m_o_ablation",
                    "limo",
                    method,
                    "element_system_current",
                    seed,
                    30,
                )
            )
        for method, group_key in (
            ("interval_hit_greedy", "element_system_current"),
            ("always_da_tpp", "element_system_current"),
            ("energy_gated_da_tpp", "element_system_current"),
            ("energy_gated_da_tpp", "coelement_block_multiset"),
            ("energy_gated_da_tpp", "coelement_iupac_group_set"),
        ):
            expected.add(("mn_group_key", "mnoxide", method, group_key, seed, 30))
    for seed in range(25, 30):
        for method in ("interval_hit_greedy", "energy_gated_da_tpp"):
            for passes in (3, 10, 30):
                expected.add(
                    (
                        "mc_dropout_sensitivity",
                        "limo",
                        method,
                        "element_system_current",
                        seed,
                        passes,
                    )
                )
    actual = set(
        frame[
            ["formal_stage", "dataset", "method", "group_key", "seed", "K"]
        ].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError(
            f"formal GPU grid mismatch: missing={sorted(expected - actual)[:5]}, "
            f"extra={sorted(actual - expected)[:5]}"
        )


def _write_gpu_statistical_summary(frame: pd.DataFrame, output_dir: Path) -> Path:
    _validate_formal_gpu_grid(frame)
    grouping = ["formal_stage", "dataset", "method", "group_key", "K"]
    summary = (
        frame.groupby(grouping, as_index=False)["AUTC"]
        .agg(n="count", mean_AUTC="mean", sample_sd_AUTC="std")
        .sort_values(grouping)
    )
    summary.to_csv(
        output_dir / "gpu_method_summary.csv", index=False, lineterminator="\n"
    )

    comparisons: list[dict[str, Any]] = []
    for stage, stage_frame in frame.groupby("formal_stage"):
        for (method, group_key, passes), method_frame in stage_frame.groupby(
            ["method", "group_key", "K"]
        ):
            if method == "interval_hit_greedy":
                continue
            greedy = stage_frame[
                (stage_frame["method"] == "interval_hit_greedy")
                & (stage_frame["K"] == passes)
            ][["seed", "AUTC"]].rename(columns={"AUTC": "greedy_AUTC"})
            joined = method_frame[["seed", "AUTC"]].merge(
                greedy, on="seed", validate="one_to_one"
            )
            differences = joined["AUTC"].to_numpy(dtype=float) - joined[
                "greedy_AUTC"
            ].to_numpy(dtype=float)
            statistics = paired_statistics(
                differences,
                bootstrap_samples=100_000,
                bootstrap_seed=20260719,
            )
            comparisons.append(
                {
                    "formal_stage": stage,
                    "dataset": str(method_frame.iloc[0]["dataset"]),
                    "method": method,
                    "group_key": group_key,
                    "K": int(passes),
                    "n_pairs": len(joined),
                    "statistics": statistics,
                }
            )
    path = output_dir / "gpu_paired_statistics.json"
    path.write_text(
        json.dumps(
            {
                "bootstrap": {"resamples": 100_000, "seed": 20260719},
                "comparisons": comparisons,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_finalizer_report(output_dir: Path, report: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = output_dir / "last_finalizer_check.json.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(output_dir / "last_finalizer_check.json")
    with (output_dir / "finalizer_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(report, sort_keys=True, ensure_ascii=False) + "\n")


def _run_remote_python_long(
    server: Mapping[str, Any],
    script: str,
    *,
    timeout_seconds: int = 8 * 60 * 60,
    accept_json_on_disconnect: bool = False,
) -> dict[str, Any]:
    command = [
        *build_ssh_command(server),
        str(server.get("remote_python", "python3")),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=script,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    payload: dict[str, Any] | None = None
    if lines:
        try:
            payload = json.loads(lines[-1])
        except json.JSONDecodeError:
            payload = None
    if completed.returncode != 0 and not (accept_json_on_disconnect and payload):
        stderr = completed.stderr.strip()[-4000:]
        raise RuntimeError(
            f"remote command failed for {server['role']}: "
            f"exit_code={completed.returncode}: {stderr}"
        )
    if payload is None:
        raise RuntimeError(f"remote command for {server['role']} returned no JSON")
    return payload


def _copy_remote_file(
    server: Mapping[str, Any],
    remote_path: str,
    local_path: Path,
) -> None:
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        raise FileExistsError(local_path)
    completed = subprocess.run(
        build_scp_command(server, remote_path, local_path),
        text=True,
        capture_output=True,
        timeout=8 * 60 * 60,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SCP failed for {server['role']}:{remote_path}: "
            f"{completed.stderr.strip()[-4000:]}"
        )


def _stream_remote_archive(
    server: Mapping[str, Any],
    remote_script_path: str,
    local_archive_path: Path,
) -> None:
    local_archive_path = Path(local_archive_path)
    local_archive_path.parent.mkdir(parents=True, exist_ok=True)
    if local_archive_path.exists():
        raise FileExistsError(local_archive_path)
    with local_archive_path.open("xb") as archive_handle:
        completed = subprocess.run(
            build_remote_stream_command(server, remote_script_path),
            stdout=archive_handle,
            stderr=subprocess.PIPE,
            timeout=8 * 60 * 60,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()[-4000:]
        raise RuntimeError(
            f"remote evidence stream failed for {server['role']}: "
            f"exit_code={completed.returncode}: {stderr}"
        )


def _next_recovery_attempt(role_root: Path) -> Path:
    role_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(path for path in role_root.glob("attempt_*") if path.is_dir())
    if len(attempts) >= 3:
        raise RuntimeError(
            f"three preserved local recovery attempts already exist under {role_root}"
        )
    attempt = role_root / f"attempt_{len(attempts) + 1}"
    attempt.mkdir(exist_ok=False)
    return attempt


def _recover_completed_role(
    *,
    server: Mapping[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    role = str(server["role"])
    remote_package_base = f"/root/autodl-tmp/egdatpp_completion_packages_v1/{role}"
    validation = _run_remote_python_long(
        server,
        build_remote_package_script(server, remote_package_base),
    )
    if validation.get("schema") != "REMOTE_COMPLETION_PACKAGE_V1":
        raise ValueError(f"{role}: unexpected remote package schema")
    if int(validation.get("job_tree_hashes_verified", 0)) != int(
        validation.get("manifest_rows", -1)
    ):
        raise ValueError(f"{role}: remote job tree validation is incomplete")
    manifest_sha = str(validation["manifest_sha256"])
    role_root = (
        project_root
        / "artifacts"
        / f"{role}_server"
        / "completed_formal_results"
        / manifest_sha[:16]
    )
    complete_path = role_root / "LOCAL_RECOVERY_COMPLETE.json"
    if complete_path.is_file():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    attempt = _next_recovery_attempt(role_root)
    package_dir = attempt / "package"
    package_dir.mkdir()
    validation_path = package_dir / "remote_validation.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    remote_files = {
        "inventory": str(validation["inventory_path"]),
        "excluded_inventory": str(validation["excluded_inventory_path"]),
        "stream_script": str(validation["stream_script_path"]),
    }
    local_files = {
        key: package_dir / Path(remote).name for key, remote in remote_files.items()
    }
    command_record = {
        key: build_scp_command(server, remote, local_files[key])
        for key, remote in remote_files.items()
    }
    archive_path = package_dir / f"{role}_completed_evidence.tar"
    command_record["archive_stream"] = build_remote_stream_command(
        server, str(validation["stream_script_path"])
    )
    (attempt / "transfer_commands.json").write_text(
        json.dumps(command_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for key, remote in remote_files.items():
        _copy_remote_file(server, remote, local_files[key])
    inventory_path = local_files["inventory"]
    if _sha256_file(inventory_path) != str(validation["inventory_sha256"]):
        raise ValueError(f"{role}: inventory SHA-256 differs from the server")
    excluded_path = local_files["excluded_inventory"]
    if _sha256_file(excluded_path) != str(validation["excluded_inventory_sha256"]):
        raise ValueError(f"{role}: excluded inventory SHA-256 differs from the server")

    _stream_remote_archive(server, str(validation["stream_script_path"]), archive_path)
    stream_marker_path = package_dir / "stream_completed.json"
    _copy_remote_file(
        server, str(validation["stream_completion_path"]), stream_marker_path
    )
    stream_marker = json.loads(stream_marker_path.read_text(encoding="utf-8"))
    if stream_marker.get("schema") != "REMOTE_COMPLETION_STREAM_V1":
        raise ValueError(f"{role}: unexpected stream completion schema")
    if stream_marker.get("inventory_sha256") != str(validation["inventory_sha256"]):
        raise ValueError(f"{role}: stream used a different inventory")
    if int(stream_marker.get("streamed_file_count", -1)) != int(
        validation["allowed_file_count"]
    ):
        raise ValueError(f"{role}: streamed file count mismatch")
    if int(stream_marker.get("streamed_bytes", -1)) != int(validation["allowed_bytes"]):
        raise ValueError(f"{role}: streamed byte count mismatch")
    archive_digest = _sha256_file(archive_path)
    (package_dir / "local_archive.sha256").write_text(
        f"{archive_digest}  {archive_path.name}\n", encoding="utf-8"
    )

    payload_root = attempt / "payload"
    safe_extract_archive(archive_path, payload_root)
    with inventory_path.open(newline="", encoding="utf-8-sig") as handle:
        inventory_rows = list(csv.DictReader(handle))
    inventory_report = verify_allowed_inventory(payload_root, inventory_rows)
    restricted_local = [
        path.relative_to(payload_root).as_posix()
        for path in payload_root.rglob("*")
        if path.is_file() and is_restricted_artifact(path)
    ]
    if restricted_local:
        raise ValueError(
            f"restricted VASP files reached local storage: {restricted_local[:5]}"
        )

    analysis_dir = attempt / "analysis"
    analysis_report = analyze_recovered_role(
        role, payload_root, validation, analysis_dir
    )
    result = {
        "role": role,
        "status": "PASS",
        "completed_at": _utc_now(),
        "remote_validation_passed": True,
        "archive_sha256_verified": True,
        "allowed_inventory_verified": True,
        "local_analysis_passed": analysis_report.get("status") == "PASS",
        "manifest_sha256": manifest_sha,
        "archive_sha256": archive_digest,
        "inventory": inventory_report,
        "attempt_dir": str(attempt.resolve()),
        "payload_root": str(payload_root.resolve()),
        "analysis": analysis_report,
    }
    temporary = role_root / "LOCAL_RECOVERY_COMPLETE.json.tmp"
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(complete_path)
    return result


def _shutdown_server(server: Mapping[str, Any]) -> dict[str, Any]:
    result = _run_remote_python_long(
        server,
        build_shutdown_script(),
        timeout_seconds=60,
        accept_json_on_disconnect=True,
    )
    if not result.get("shutdown_started"):
        raise RuntimeError(f"{server['role']}: shutdown command was not accepted")
    return result


def resume_authorized_shutdown(
    config: Mapping[str, Any],
    output_dir: Path,
    *,
    shutdown_runner: Any = _shutdown_server,
) -> dict[str, Any]:
    """Finish an already authorized two-server shutdown without repeating success."""

    output_dir = Path(output_dir)
    authorization_path = output_dir / "SHUTDOWN_AUTHORIZED.json"
    if not authorization_path.is_file():
        raise FileNotFoundError(authorization_path)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    shutdown_results: dict[str, Any] = {}
    for role in ("dft", "gpu"):
        result_path = output_dir / f"shutdown_{role}.json"
        if result_path.is_file():
            shutdown_results[role] = json.loads(result_path.read_text(encoding="utf-8"))
            continue
        result = shutdown_runner(config["servers"][role])
        if not result.get("shutdown_started"):
            raise RuntimeError(f"{role}: shutdown command was not accepted")
        temporary = output_dir / f"shutdown_{role}.json.tmp"
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(result_path)
        shutdown_results[role] = result
    complete = {
        "state": "COMPLETE_SHUTDOWN_REQUESTED",
        "completed_at": _utc_now(),
        "shutdown_authorization": authorization,
        "shutdown_results": shutdown_results,
    }
    temporary = output_dir / "FINALIZATION_COMPLETE.json.tmp"
    temporary.write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_dir / "FINALIZATION_COMPLETE.json")
    return complete


def _execute_completed_finalization(
    *,
    config: Mapping[str, Any],
    gate: FinalizationGate,
    output_dir: Path,
    shutdown_after_success: bool,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    recoveries = {
        role: _recover_completed_role(server=server, project_root=project_root)
        for role, server in config["servers"].items()
    }
    gpu_csv = Path(recoveries["gpu"]["analysis"]["analysis_csv"])
    gpu_frame = pd.read_csv(gpu_csv)
    statistics_path = _write_gpu_statistical_summary(gpu_frame, output_dir)
    scientific_holds = scientific_holds_from_gpu_analysis(
        gpu_frame.to_dict(orient="records")
    )
    summary = {
        "remote_validation_passed": all(
            bool(item["remote_validation_passed"]) for item in recoveries.values()
        ),
        "archive_sha256_verified": all(
            bool(item["archive_sha256_verified"]) for item in recoveries.values()
        ),
        "allowed_inventory_verified": all(
            bool(item["allowed_inventory_verified"]) for item in recoveries.values()
        ),
        "local_analysis_passed": all(
            bool(item["local_analysis_passed"]) for item in recoveries.values()
        ),
        "scientific_holds": scientific_holds,
        "recoveries": recoveries,
        "gpu_paired_statistics": str(statistics_path.resolve()),
    }
    (output_dir / "local_validation_and_analysis.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not authorize_shutdown(gate, summary):
        return {
            "state": "SCIENTIFIC_HOLD" if scientific_holds else "VALIDATION_BLOCKED",
            "finalization_summary": summary,
        }
    if not shutdown_after_success:
        return {"state": "RECOVERED_AND_ANALYZED", "finalization_summary": summary}

    authorization = {
        "authorized_at": _utc_now(),
        "reason": "both manifests complete; remote trees, transferred archives, inventories, and local analyses passed",
        "summary_sha256": _sha256_file(
            output_dir / "local_validation_and_analysis.json"
        ),
    }
    (output_dir / "SHUTDOWN_AUTHORIZED.json").write_text(
        json.dumps(authorization, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete = resume_authorized_shutdown(config, output_dir)
    complete["finalization_summary"] = summary
    return complete


def _build_role_finalization_summary(
    role: str,
    recovery: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    scientific_holds: list[str] = []
    statistics_path: str | None = None
    if role == "gpu":
        gpu_csv = Path(str(recovery["analysis"]["analysis_csv"]))
        gpu_frame = pd.read_csv(gpu_csv)
        statistics_path = str(
            _write_gpu_statistical_summary(gpu_frame, output_dir).resolve()
        )
        scientific_holds = scientific_holds_from_gpu_analysis(
            gpu_frame.to_dict(orient="records")
        )
    summary = {
        "role": role,
        "remote_validation_passed": bool(recovery["remote_validation_passed"]),
        "archive_sha256_verified": bool(recovery["archive_sha256_verified"]),
        "allowed_inventory_verified": bool(recovery["allowed_inventory_verified"]),
        "local_analysis_passed": bool(recovery["local_analysis_passed"]),
        "scientific_holds": scientific_holds,
        "recovery": recovery,
        "gpu_paired_statistics": statistics_path,
    }
    path = output_dir / f"local_validation_and_analysis_{role}.json"
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def finalize_ready_role(
    *,
    config: Mapping[str, Any],
    output_dir: Path,
    role: str,
    snapshot: Mapping[str, Any],
    shutdown_after_success: bool,
    recovery_runner: Any = _recover_completed_role,
    summary_runner: Any = _build_role_finalization_summary,
    shutdown_runner: Any = _shutdown_server,
) -> dict[str, Any]:
    """Recover, analyze, and stop one independently completed server."""

    output_dir = Path(output_dir)
    role_complete_path = output_dir / f"ROLE_FINALIZATION_COMPLETE_{role}.json"
    if role_complete_path.is_file():
        return json.loads(role_complete_path.read_text(encoding="utf-8"))
    gate = assess_role_completion(role, snapshot)
    if not gate.ready:
        return {"state": gate.state, "role": role, "reasons": list(gate.reasons)}

    server = config["servers"][role]
    project_root = Path(__file__).resolve().parents[1]
    recovery = recovery_runner(server=server, project_root=project_root)
    summary = summary_runner(role, recovery, output_dir)
    if not authorize_shutdown(gate, summary):
        return {
            "state": "SCIENTIFIC_HOLD"
            if summary.get("scientific_holds")
            else "VALIDATION_BLOCKED",
            "role": role,
            "finalization_summary": summary,
        }
    if not shutdown_after_success:
        return {
            "state": "RECOVERED_AND_ANALYZED",
            "role": role,
            "finalization_summary": summary,
        }

    summary_path = output_dir / f"local_validation_and_analysis_{role}.json"
    if summary_path.is_file():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing_summary != summary:
            raise RuntimeError(f"{role}: persisted analysis summary changed")
    else:
        with summary_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    authorization = {
        "role": role,
        "authorized_at": _utc_now(),
        "reason": "role manifest complete; remote trees, transferred inventory, and local analysis passed",
        "summary_sha256": _sha256_file(summary_path),
    }
    authorization_path = output_dir / f"SHUTDOWN_AUTHORIZED_{role}.json"
    if not authorization_path.is_file():
        with authorization_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    shutdown_path = output_dir / f"shutdown_{role}.json"
    if shutdown_path.is_file():
        shutdown_result = json.loads(shutdown_path.read_text(encoding="utf-8"))
    else:
        shutdown_result = shutdown_runner(server)
        if not shutdown_result.get("shutdown_started"):
            raise RuntimeError(f"{role}: shutdown command was not accepted")
        temporary = output_dir / f"shutdown_{role}.json.tmp"
        temporary.write_text(
            json.dumps(shutdown_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(shutdown_path)
    completed = {
        "state": "ROLE_COMPLETE",
        "role": role,
        "completed_at": _utc_now(),
        "finalization_summary": summary,
        "shutdown_result": shutdown_result,
    }
    temporary = output_dir / f"ROLE_FINALIZATION_COMPLETE_{role}.json.tmp"
    temporary.write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(role_complete_path)
    return completed


def run_finalization_once(
    config_path: Path,
    output_dir: Path,
    *,
    execute: bool,
    shutdown_after_success: bool,
    snapshot_collector: Any = collect_snapshot,
    role_finalizer: Any = finalize_ready_role,
) -> dict[str, Any]:
    """Finalize each ready server independently and preserve unfinished work."""

    config_path = Path(config_path).resolve()
    output_dir = Path(output_dir).resolve()
    complete_path = output_dir / "FINALIZATION_COMPLETE.json"
    if complete_path.is_file():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (output_dir / "SHUTDOWN_AUTHORIZED.json").is_file():
        report = resume_authorized_shutdown(config, output_dir)
        _write_finalizer_report(output_dir, report)
        return report
    snapshots: dict[str, Any] = {}
    roles: dict[str, Any] = {}
    reasons: list[str] = []
    for role in ("gpu", "dft"):
        role_complete_path = output_dir / f"ROLE_FINALIZATION_COMPLETE_{role}.json"
        if role_complete_path.is_file():
            roles[role] = json.loads(role_complete_path.read_text(encoding="utf-8"))
            continue
        server = config["servers"][role]
        snapshot = snapshot_collector(server)
        snapshots[role] = snapshot
        gate = assess_role_completion(role, snapshot)
        reasons.extend(gate.reasons)
        if not gate.ready:
            roles[role] = {
                "state": gate.state,
                "role": role,
                "reasons": list(gate.reasons),
            }
            continue
        if not execute:
            roles[role] = {"state": "READY_DRY_RUN", "role": role, "reasons": []}
            continue
        roles[role] = role_finalizer(
            config=config,
            output_dir=output_dir,
            role=role,
            snapshot=snapshot,
            shutdown_after_success=shutdown_after_success,
        )

    complete_roles = [
        role for role, entry in roles.items() if entry.get("state") == "ROLE_COMPLETE"
    ]
    if len(complete_roles) == 2:
        overall_state = "COMPLETE_SHUTDOWN_REQUESTED"
    elif complete_roles:
        overall_state = "PARTIAL"
    elif any(
        entry.get("state") in {"BLOCKED", "SCIENTIFIC_HOLD", "VALIDATION_BLOCKED"}
        for entry in roles.values()
    ):
        overall_state = "BLOCKED"
    else:
        overall_state = "WAIT"
    report: dict[str, Any] = {
        "checked_at": _utc_now(),
        "execute": bool(execute),
        "shutdown_after_success": bool(shutdown_after_success),
        "state": overall_state,
        "reasons": reasons,
        "snapshots": snapshots,
        "roles": roles,
    }
    if overall_state == "COMPLETE_SHUTDOWN_REQUESTED":
        temporary = output_dir / "FINALIZATION_COMPLETE.json.tmp"
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(complete_path)
    _write_finalizer_report(output_dir, report)
    return report


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--shutdown-after-success", action="store_true")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args(argv)
    if args.version:
        print("server-completion-finalizer-v1")
        return 0
    if args.config is None or args.output_dir is None:
        parser.error("--config and --output-dir are required")
    report = run_finalization_once(
        args.config,
        args.output_dir,
        execute=args.execute,
        shutdown_after_success=args.shutdown_after_success,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
