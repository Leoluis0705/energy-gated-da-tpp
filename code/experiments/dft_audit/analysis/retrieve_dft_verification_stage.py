#!/usr/bin/env python3
"""Recover a terminal DFT verification stage with remote/local SHA-256 checks."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import paramiko


RESTRICTED_BASENAMES = frozenset(
    {"POTCAR", "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1", "AECCAR2"}
)
TERMINAL_WATCHER_STATES = {
    "POSTPROCESS_DONE": True,
    "POSTPROCESS_DONE_PAPER_UPDATE_PAUSED": False,
    "BLOCKED_BY_UPSTREAM_GATE": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_transfer_allowed(path: PurePosixPath) -> bool:
    """Keep licensed PAW payloads and large restart/density files server-side."""

    name = path.name.upper()
    return name not in RESTRICTED_BASENAMES and not name.startswith("POTCAR.")


def safe_local_target(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe remote relative path: {relative}")
    target = Path(root).joinpath(*path.parts)
    target.resolve().relative_to(Path(root).resolve())
    return target


def terminal_watcher_state(payload: dict[str, Any]) -> tuple[str, bool]:
    status = str(payload.get("status", ""))
    if status not in TERMINAL_WATCHER_STATES:
        raise ValueError(f"verification watcher is not terminal: {status!r}")
    return status, TERMINAL_WATCHER_STATES[status]


def _safe_stage_name(name: str, prefix: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or len(path.parts) != 1 or not name.startswith(prefix):
        raise ValueError(f"unsafe stage directory: {name}")
    return name


def _remote_inventory_source(remote_root: str) -> str:
    return f'''import base64, hashlib, json, os, zlib
from pathlib import Path
root = Path({remote_root!r}).resolve()
restricted = {sorted(RESTRICTED_BASENAMES)!r}
allowed = []
excluded = []
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
for path in sorted(root.rglob("*")):
    if path.is_symlink():
        excluded.append({{"relative_path": path.relative_to(root).as_posix(), "reason": "symlink"}})
        continue
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    stat = path.stat()
    row = {{"relative_path": relative, "size_bytes": stat.st_size,
           "mtime_epoch": stat.st_mtime, "sha256": sha256(path)}}
    name = path.name.upper()
    if name in restricted or name.startswith("POTCAR."):
        row["reason"] = "restricted_vasp_artifact"
        excluded.append(row)
    else:
        allowed.append(row)
payload = {{"remote_root": str(root), "allowed": allowed, "excluded": excluded}}
raw = json.dumps(payload, sort_keys=True).encode("utf-8")
print(base64.b64encode(zlib.compress(raw, 9)).decode("ascii"))
'''


def _exec(client: paramiko.SSHClient, command: str, timeout: int = 300) -> str:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"remote command exited {status}: {error.strip()}")
    return output


def _read_remote_json(sftp: paramiko.SFTPClient, path: str) -> dict[str, Any]:
    with sftp.open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def retrieve(args: argparse.Namespace) -> dict[str, Any]:
    password = os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(f"password environment variable is not set: {args.password_env}")
    local_root = Path(args.local_root).resolve()
    local_root.mkdir(parents=True, exist_ok=False)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            args.host,
            port=args.port,
            username=args.user,
            password=password,
            look_for_keys=False,
            allow_agent=False,
            timeout=20,
            auth_timeout=20,
            banner_timeout=20,
        )
        server_key = client.get_transport().get_remote_server_key()
        host_key_sha256 = "SHA256:" + base64.b64encode(
            hashlib.sha256(server_key.asbytes()).digest()
        ).decode("ascii").rstrip("=")
        sftp = client.open_sftp()
        watcher_name = _safe_stage_name(
            args.watcher_directory_name, "candidate_postprocess_watcher"
        )
        watcher_path = (
            PurePosixPath(args.remote_root)
            / watcher_name
            / "final_status.json"
        ).as_posix()
        watcher = _read_remote_json(sftp, watcher_path)
        watcher_status, paper_update_authorized = terminal_watcher_state(watcher)
        process_check = _exec(
            client,
            "ps -eo args= | grep -E '[v]asp_(std|gam|ncl)|"
            "[r]un_dft_verification_supervisor|[w]atch_dft_verification_postprocess' || true",
            timeout=60,
        ).strip()
        if process_check:
            raise RuntimeError("verification VASP/supervisor processes are still alive")

        source = _remote_inventory_source(args.remote_root)
        encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
        command = (
            f"{shlex.quote(args.remote_python)} -c "
            + shlex.quote(f"import base64;exec(base64.b64decode({encoded!r}))")
        )
        compressed = _exec(client, command, timeout=args.inventory_timeout).strip()
        import zlib

        inventory = json.loads(zlib.decompress(base64.b64decode(compressed)).decode("utf-8"))
        allowed = list(inventory["allowed"])
        excluded = list(inventory["excluded"])
        for row in allowed:
            relative = str(row["relative_path"])
            if not is_transfer_allowed(PurePosixPath(relative)):
                raise RuntimeError(f"remote inventory misclassified restricted file: {relative}")
            target = safe_local_target(local_root / "payload", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            remote_path = (PurePosixPath(args.remote_root) / relative).as_posix()
            sftp.get(remote_path, str(target))
            if target.stat().st_size != int(row["size_bytes"]):
                raise RuntimeError(f"downloaded size mismatch: {relative}")
            if _sha256(target) != str(row["sha256"]):
                raise RuntimeError(f"downloaded SHA-256 mismatch: {relative}")
        sftp.close()
    finally:
        client.close()

    allowed_fields = ["relative_path", "size_bytes", "mtime_epoch", "sha256"]
    excluded_fields = allowed_fields + ["reason"]
    _write_csv(local_root / "remote_allowed_sha256.csv", allowed, allowed_fields)
    _write_csv(local_root / "remote_excluded_sha256.csv", excluded, excluded_fields)
    local_rows = []
    for path in sorted((local_root / "payload").rglob("*")):
        if path.is_file():
            relative = path.relative_to(local_root / "payload").as_posix()
            local_rows.append(
                {"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            )
    _write_csv(
        local_root / "local_sha256.csv",
        local_rows,
        ["relative_path", "size_bytes", "sha256"],
    )
    metadata = {
        "schema": "DFT_VERIFICATION_STAGE_RECOVERY_V1",
        "recovered_at_utc": _utc_now(),
        "host": args.host,
        "port": args.port,
        "user": args.user,
        "host_key_sha256": host_key_sha256,
        "remote_root": args.remote_root,
        "local_root": str(local_root),
        "watcher_status": watcher_status,
        "paper_conclusion_update_authorized": paper_update_authorized,
        "allowed_file_count": len(allowed),
        "allowed_bytes": sum(int(row["size_bytes"]) for row in allowed),
        "excluded_file_count": len(excluded),
        "verified_file_count": len(local_rows),
        "hash_mismatches": 0,
        "password_recorded": False,
        "potcar_payload_transferred": False,
    }
    with (local_root / "recovery_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--user", default="root")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--remote-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--password-env", default="DFT_SSH_PASSWORD")
    parser.add_argument("--inventory-timeout", type=int, default=1800)
    parser.add_argument(
        "--watcher-directory-name", default="candidate_postprocess_watcher"
    )
    args = parser.parse_args()
    metadata = retrieve(args)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
