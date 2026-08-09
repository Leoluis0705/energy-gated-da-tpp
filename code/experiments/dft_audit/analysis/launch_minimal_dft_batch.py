#!/usr/bin/env python3
"""Upload, launch, monitor, and retrieve the frozen minimal DFT batch."""

from __future__ import annotations

import argparse
import base64
import csv
import getpass
import hashlib
import json
import os
import shlex
import socket
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import paramiko

from analysis.audit_minimal_dft_remote import EXPECTED_HOST_KEY, host_key_sha256


RESTRICTED_REMOTE_NAMES = {
    "POTCAR",
    "WAVECAR",
    "CHGCAR",
    "CHG",
    "AECCAR0",
    "AECCAR1",
    "AECCAR2",
}
TERMINAL_STATES = {"COMPLETE", "STOPPED_BY_PROBE_GATE"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connect(
    host: str, port: int, user: str, password: str, expected_host_key: str
) -> paramiko.Transport:
    sock = socket.create_connection((host, port), timeout=20)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=20)
    observed = host_key_sha256(transport.get_remote_server_key())
    if observed != expected_host_key:
        transport.close()
        sock.close()
        raise RuntimeError(
            f"SSH host-key mismatch: observed {observed}; expected {expected_host_key}"
        )
    transport.auth_password(user, password)
    if not transport.is_authenticated():
        transport.close()
        sock.close()
        raise RuntimeError("SSH password authentication failed")
    transport.set_keepalive(30)
    return transport


def remote_exec(
    transport: paramiko.Transport, command: str, timeout: int = 300
) -> tuple[int, str, str]:
    channel = transport.open_session(timeout=20)
    try:
        channel.settimeout(timeout)
        channel.exec_command(command)
        stdout = channel.makefile("rb").read().decode("utf-8", "replace")
        stderr = channel.makefile_stderr("rb").read().decode("utf-8", "replace")
        status = int(channel.recv_exit_status())
    finally:
        channel.close()
    return status, stdout, stderr


def remote_mkdirs(sftp: paramiko.SFTPClient, path: str) -> None:
    current = PurePosixPath("/")
    for part in PurePosixPath(path).parts[1:]:
        current /= part
        try:
            sftp.stat(current.as_posix())
        except FileNotFoundError:
            sftp.mkdir(current.as_posix())


def upload_tree(
    sftp: paramiko.SFTPClient, local_root: Path, remote_root: str
) -> list[dict[str, Any]]:
    inventory = []
    remote_mkdirs(sftp, remote_root)
    for path in sorted(local_root.rglob("*")):
        relative = path.relative_to(local_root).as_posix()
        target = (PurePosixPath(remote_root) / relative).as_posix()
        if path.is_dir():
            remote_mkdirs(sftp, target)
            continue
        if path.name.upper() in RESTRICTED_REMOTE_NAMES:
            raise RuntimeError(f"restricted file found in local upload tree: {path}")
        remote_mkdirs(sftp, PurePosixPath(target).parent.as_posix())
        sftp.put(str(path), target)
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return inventory


def upload_file(
    sftp: paramiko.SFTPClient, local: Path, remote: str
) -> dict[str, Any]:
    remote_mkdirs(sftp, PurePosixPath(remote).parent.as_posix())
    sftp.put(str(local), remote)
    return {
        "relative_path": PurePosixPath(remote).name,
        "size_bytes": local.stat().st_size,
        "sha256": sha256(local),
    }


def read_remote_json(sftp: paramiko.SFTPClient, path: str) -> dict[str, Any]:
    with sftp.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _remote_file_sha256(
    transport: paramiko.Transport, path: str
) -> tuple[int, str]:
    status, stdout, _ = remote_exec(
        transport, f"sha256sum -- {shlex.quote(path)}", timeout=60
    )
    return status, stdout.split()[0] if stdout.strip() else ""


def verify_upload(
    transport: paramiko.Transport,
    package_root: str,
    inventory: list[dict[str, Any]],
) -> None:
    for row in inventory:
        path = (
            PurePosixPath(package_root) / str(row["relative_path"])
        ).as_posix()
        status, observed = _remote_file_sha256(transport, path)
        if status or observed.lower() != str(row["sha256"]).lower():
            raise RuntimeError(f"remote upload hash mismatch: {row['relative_path']}")


def _remote_walk(
    sftp: paramiko.SFTPClient, remote_root: str
) -> list[tuple[str, paramiko.SFTPAttributes]]:
    output = []
    pending = [PurePosixPath(remote_root)]
    while pending:
        root = pending.pop()
        for item in sftp.listdir_attr(root.as_posix()):
            path = root / item.filename
            if stat.S_ISLNK(item.st_mode):
                continue
            if stat.S_ISDIR(item.st_mode):
                pending.append(path)
            elif stat.S_ISREG(item.st_mode):
                output.append((path.as_posix(), item))
    return sorted(output)


def download_results(
    sftp: paramiko.SFTPClient, remote_root: str, local_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = []
    excluded = []
    local_root.mkdir(parents=True, exist_ok=False)
    for remote_path, attrs in _remote_walk(sftp, remote_root):
        relative = PurePosixPath(remote_path).relative_to(PurePosixPath(remote_root))
        name = relative.name.upper()
        row = {
            "relative_path": relative.as_posix(),
            "size_bytes": int(attrs.st_size),
        }
        if name in RESTRICTED_REMOTE_NAMES or name.startswith("POTCAR."):
            row["reason"] = "restricted_or_large_vasp_artifact"
            excluded.append(row)
            continue
        target = local_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(target))
        row["sha256"] = sha256(target)
        allowed.append(row)
    return allowed, excluded


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, encoding="utf-8"
    ).strip()


def launch(args: argparse.Namespace) -> int:
    repo = Path(args.repo_root).resolve()
    protocol_path = repo / "manifests" / "dft_protocol_frozen.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("overall_status") != "PASS":
        raise RuntimeError("local frozen DFT protocol is not PASS")
    if protocol.get("same_scale_status") != "UNRESOLVED":
        raise RuntimeError("unexpected SAME_SCALE_STATUS; re-audit before launch")
    tag_commit = _git_output(repo, "rev-list", "-n", "1", "minimal-dft-5-frozen")
    if not tag_commit:
        raise RuntimeError("minimal-dft-5-frozen tag is missing")
    protocol_hash = sha256(protocol_path)
    candidate_manifest = repo / "manifests" / "minimal_dft_5_candidates.csv"
    candidates = list(csv_rows(candidate_manifest))
    by_order = {int(row["frozen_order"]): row["candidate_id"] for row in candidates}
    probes = [by_order[2], by_order[4]]
    remaining = [by_order[1], by_order[3], by_order[5]]

    remote_audit = json.loads(
        (repo / "reports" / "dft_remote_audit.json").read_text(encoding="utf-8")
    )
    if not remote_audit.get("pass"):
        raise RuntimeError("remote audit is not PASS")
    label_to_element = {
        "Li_sv": "Li",
        "Cr_pv": "Cr",
        "Mn_pv": "Mn",
        "Mg_pv": "Mg",
        "O": "O",
    }
    potcar_paths = {}
    potcar_hashes = {}
    for label, element in label_to_element.items():
        matches = remote_audit["paw_status"][label]["matches"]
        if len(matches) != 1:
            raise RuntimeError(f"expected one audited server POTCAR for {label}")
        potcar_paths[element] = matches[0]["path"]
        potcar_hashes[element] = matches[0]["sha256"]

    password = getpass.getpass("DFT server password (not stored): ")
    transport = connect(
        args.host, args.port, args.user, password, args.expected_host_key
    )
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"minimal_dft_5_{protocol_hash[:12]}_{timestamp}"
        remote_root = (
            PurePosixPath(args.remote_base) / run_id
        ).as_posix()
        package_root = (PurePosixPath(remote_root) / "package").as_posix()
        execution_root = (PurePosixPath(remote_root) / "execution").as_posix()
        try:
            sftp.stat(remote_root)
        except FileNotFoundError:
            remote_mkdirs(sftp, remote_root)
        else:
            raise FileExistsError(f"remote run root already exists: {remote_root}")

        package_inventory = upload_tree(
            sftp, repo / "dft" / "minimal_dft_5_frozen" / "inputs",
            (PurePosixPath(package_root) / "inputs").as_posix(),
        )
        for row in package_inventory:
            row["relative_path"] = (
                PurePosixPath("inputs") / str(row["relative_path"])
            ).as_posix()
        fixed_files = {
            "dft_protocol_frozen.json": protocol_path,
            "minimal_dft_5_candidates.csv": candidate_manifest,
            "minimal_dft_5_candidates.json": (
                repo / "manifests" / "minimal_dft_5_candidates.json"
            ),
            "elemental_references.csv": repo / "dft" / "results" / "elemental_references.csv",
            "run_minimal_dft_remote.py": repo / "analysis" / "run_minimal_dft_remote.py",
        }
        for name, local in fixed_files.items():
            row = upload_file(
                sftp, local, (PurePosixPath(package_root) / name).as_posix()
            )
            row["relative_path"] = name
            package_inventory.append(row)
        verify_upload(transport, package_root, package_inventory)
        dry_command = (
            "test -x /root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std"
            " && python3 -m py_compile "
            + shlex.quote(
                (PurePosixPath(package_root) / "run_minimal_dft_remote.py").as_posix()
            )
        )
        status, _, error = remote_exec(transport, dry_command, timeout=120)
        if status:
            raise RuntimeError(f"remote dry-run failed: {error.strip()}")

        command = [
            "python3",
            (PurePosixPath(package_root) / "run_minimal_dft_remote.py").as_posix(),
            "--run-root",
            execution_root,
            "--package-root",
            package_root,
            "--protocol-sha256",
            protocol_hash,
            "--vasp",
            "/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std",
            "--potcar-paths-json",
            json.dumps(potcar_paths, separators=(",", ":")),
            "--potcar-hashes-json",
            json.dumps(potcar_hashes, separators=(",", ":")),
            "--probe-candidates",
            ",".join(probes),
            "--remaining-candidates",
            ",".join(remaining),
            "--stage-timeout-seconds",
            str(args.stage_timeout_seconds),
            "--memory-limit-kb",
            str(args.memory_limit_kb),
        ]
        launch_record = {
            "run_id": run_id,
            "launched_at_utc": utc_now(),
            "host": args.host,
            "port": args.port,
            "remote_root": remote_root,
            "package_root": package_root,
            "execution_root": execution_root,
            "protocol_sha256": protocol_hash,
            "candidate_manifest_sha256": sha256(candidate_manifest),
            "frozen_tag_commit": tag_commit,
            "probe_candidates": probes,
            "remaining_candidates": remaining,
            "command": command,
            "potcar_policy": "server-side only; bodies never uploaded or downloaded",
        }
        local_launch_path = repo / "reports" / "minimal_dft_5_launch.json"
        local_launch_path.write_text(
            json.dumps(launch_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        remote_launch = (PurePosixPath(remote_root) / "launch_record.json").as_posix()
        with sftp.open(remote_launch, "wb") as handle:
            handle.write(
                (json.dumps(launch_record, indent=2, sort_keys=True) + "\n").encode(
                    "utf-8"
                )
            )
        quoted = " ".join(shlex.quote(value) for value in command)
        supervisor_log = (PurePosixPath(remote_root) / "supervisor.log").as_posix()
        pid_path = (PurePosixPath(remote_root) / "supervisor.pid").as_posix()
        launch_shell = (
            f"nohup {quoted} > {shlex.quote(supervisor_log)} 2>&1 "
            f"< /dev/null & echo $! > {shlex.quote(pid_path)}"
        )
        status, _, error = remote_exec(transport, launch_shell, timeout=60)
        if status:
            raise RuntimeError(f"remote launch failed: {error.strip()}")
        print(
            json.dumps(
                {
                    "event": "launched",
                    "run_id": run_id,
                    "remote_root": remote_root,
                    "probes": probes,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        last_snapshot = None
        status_path = (PurePosixPath(execution_root) / "status.json").as_posix()
        while True:
            time.sleep(args.poll_seconds)
            try:
                snapshot = read_remote_json(sftp, status_path)
            except (FileNotFoundError, OSError, EOFError):
                snapshot = None
            if snapshot != last_snapshot and snapshot is not None:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "status": snapshot.get("status"),
                            "phase": snapshot.get("current_phase"),
                            "candidate": snapshot.get("current_candidate"),
                            "completed": snapshot.get("completed_candidates"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                last_snapshot = snapshot
            if snapshot and snapshot.get("status") in TERMINAL_STATES:
                break
            alive_command = (
                f"test -s {shlex.quote(pid_path)} && "
                f"kill -0 $(cat {shlex.quote(pid_path)}) 2>/dev/null"
            )
            alive_status, _, _ = remote_exec(transport, alive_command, timeout=30)
            if alive_status and snapshot is None:
                raise RuntimeError("remote supervisor exited before creating status.json")
            if alive_status and snapshot and snapshot.get("status") not in TERMINAL_STATES:
                raise RuntimeError(
                    f"remote supervisor exited unexpectedly in state {snapshot.get('status')}"
                )

        local_artifact = (
            repo / "artifacts" / "minimal_dft_5" / run_id
        )
        allowed, excluded = download_results(
            sftp, remote_root, local_artifact
        )
        (local_artifact / "downloaded_inventory.json").write_text(
            json.dumps(
                {"allowed": allowed, "excluded": excluded},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        local_pointer = {
            **launch_record,
            "terminal_status": snapshot,
            "local_artifact_root": str(local_artifact),
            "retrieved_at_utc": utc_now(),
            "downloaded_files": len(allowed),
            "excluded_files": len(excluded),
        }
        (repo / "reports" / "minimal_dft_5_retrieval.json").write_text(
            json.dumps(local_pointer, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "retrieved",
                    "terminal_status": snapshot.get("status"),
                    "local_artifact_root": str(local_artifact),
                    "downloaded_files": len(allowed),
                    "excluded_files": len(excluded),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if snapshot.get("status") == "COMPLETE" else 2
    finally:
        password = ""
        sftp.close()
        transport.close()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--expected-host-key", default=EXPECTED_HOST_KEY)
    parser.add_argument(
        "--remote-base",
        default="/root/autodl-tmp/Energy_Gated_DA_TPP_DFT_Audit",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stage-timeout-seconds", type=int, default=6 * 3600)
    parser.add_argument("--memory-limit-kb", type=int, default=50 * 1024 * 1024)
    return launch(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
