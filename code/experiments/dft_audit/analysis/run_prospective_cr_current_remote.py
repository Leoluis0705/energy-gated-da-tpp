"""Remote execution helpers for the prospective Cr self-consistent batch."""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
from pathlib import Path
from typing import Iterable

import paramiko
from pymatgen.core import Structure

from analysis.audit_current_potcars_remote import EXPECTED_HOST_KEY, host_key_sha256


def choose_restart_geometry(attempt_dirs: Iterable[Path]) -> Path | None:
    """Choose the newest parseable, non-empty CONTCAR from ordered attempts."""

    for attempt in reversed([Path(value) for value in attempt_dirs]):
        contcar = attempt / "CONTCAR"
        if not contcar.is_file() or contcar.stat().st_size == 0:
            continue
        try:
            Structure.from_file(contcar)
        except Exception:
            continue
        return contcar
    return None


def _connect(
    host: str, port: int, user: str, key_path: Path
) -> tuple[paramiko.Transport, socket.socket]:
    key = paramiko.Ed25519Key.from_private_key_file(str(key_path.expanduser()))
    sock = socket.create_connection((host, port), timeout=20)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=20)
    observed = host_key_sha256(transport.get_remote_server_key())
    if observed != EXPECTED_HOST_KEY:
        transport.close()
        sock.close()
        raise RuntimeError(f"host-key mismatch: {observed}")
    transport.auth_publickey(user, key)
    if not transport.is_authenticated():
        transport.close()
        sock.close()
        raise RuntimeError("SSH authentication failed")
    return transport, sock


def _mkdirs(sftp: paramiko.SFTPClient, remote_path: str) -> None:
    parts = Path(remote_path).as_posix().strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def upload_package(
    sftp: paramiko.SFTPClient, local_root: Path, remote_root: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(local_root.rglob("*")):
        relative = path.relative_to(local_root).as_posix()
        destination = remote_root.rstrip("/") + "/" + relative
        if path.is_dir():
            _mkdirs(sftp, destination)
            continue
        if path.name == "POTCAR":
            raise RuntimeError("local input package unexpectedly contains POTCAR")
        _mkdirs(sftp, destination.rsplit("/", 1)[0])
        sftp.put(str(path), destination)
        records.append(
            {
                "local": str(path),
                "remote": destination,
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def exec_remote(
    transport: paramiko.Transport, command: str, timeout: int = 60
) -> tuple[int, str, str]:
    channel = transport.open_session(timeout=20)
    try:
        channel.settimeout(timeout)
        channel.exec_command(command)
        stdout = channel.makefile("rb").read().decode("utf-8", "replace")
        stderr = channel.makefile_stderr("rb").read().decode(
            "utf-8", "replace"
        )
        status = int(channel.recv_exit_status())
        return status, stdout, stderr
    finally:
        channel.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", required=True)
    parser.add_argument("--local-package", required=True)
    parser.add_argument(
        "--remote-root",
        default="/root/autodl-tmp/prospective_cr_discovery_v2",
    )
    parser.add_argument("--segment-seconds", type=int, default=21000)
    parser.add_argument("--phase2-workers", type=int, default=2)
    parser.add_argument("--deploy-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--repair-controller-json-stress", action="store_true")
    args = parser.parse_args()

    transport, sock = _connect(
        args.host, args.port, args.user, Path(args.key)
    )
    try:
        if args.repair_controller_json_stress:
            stop_command = (
                f"root={args.remote_root}; "
                "pids=$(pgrep -f '^python3 remote_current_dft_controller.py "
                f"--root {args.remote_root} ' || true); "
                "for p in $pids; do "
                "for c in $(pgrep -P $p || true); do kill -TERM $c || true; "
                "done; done; "
                "sleep 5; "
                "for p in $pids; do kill -TERM $p || true; done; "
                "sleep 3; "
                "for p in $pids; do kill -KILL $p 2>/dev/null || true; done; "
                "find \"$root/runs\" -name CONTCAR -type f -size +0c "
                "-printf '%p ' -exec sha256sum {} \\; "
                "> \"$root/repair_001_preserved_contcars.txt\""
            )
            status, stdout, stderr = exec_remote(
                transport, stop_command, timeout=30
            )
            if status:
                raise RuntimeError(
                    f"targeted controller stop failed: {stdout} {stderr}"
                )
            sftp = paramiko.SFTPClient.from_transport(transport)
            try:
                source = (
                    Path(__file__).resolve().parent
                    / "remote_current_dft_controller.py"
                )
                sftp.put(
                    str(source),
                    args.remote_root + "/remote_current_dft_controller.py",
                )
            finally:
                sftp.close()
            start_command = (
                f"cd {args.remote_root} && "
                "setsid -f python3 remote_current_dft_controller.py "
                f"--root {args.remote_root} "
                f"--segment-seconds {args.segment_seconds} "
                f"--phase2-workers {args.phase2_workers} "
                "> controller_repair_001.log 2>&1 < /dev/null; "
                "echo REPAIR_001_RESTARTED"
            )
            status, stdout, stderr = exec_remote(
                transport, start_command, timeout=30
            )
            if status:
                raise RuntimeError(
                    f"controller repair restart failed: {stdout} {stderr}"
                )
            print(stdout, end="")
            return 0
        if args.status_only:
            status, stdout, stderr = exec_remote(
                transport,
                f"root={args.remote_root}; "
                "echo CONTROLLER_STATE; "
                "python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); "
                "print(json.dumps({\"status\":s.get(\"status\"),"
                "\"phase\":s.get(\"phase\"),"
                "\"current_entity\":s.get(\"current_entity\"),"
                "\"job_092_gate\":s.get(\"job_092_gate\"),"
                "\"failure_reason\":s.get(\"failure_reason\"),"
                "\"entities\":{k:v.get(\"status\") for k,v in "
                "s.get(\"entities\",{}).items()}},sort_keys=True))' "
                "\"$root/controller_state.json\" 2>/dev/null; "
                "echo RECENT_OSZICAR; "
                "find \"$root/runs\" -name OSZICAR -type f -size +0c "
                "-printf '%T@ %p\\n' "
                "2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | "
                "xargs -r tail -n 4; "
                "echo RECENT_OUTCAR; "
                "find \"$root/runs\" -name OUTCAR -type f -size +0c "
                "-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | "
                "cut -d' ' -f2- | xargs -r tail -n 8; "
                "echo CURRENT_IBZKPT; "
                "find \"$root/runs\" -name IBZKPT -type f "
                "-printf '%T@ %p\\n' 2>/dev/null | sort -n | tail -1 | "
                "cut -d' ' -f2- | xargs -r head -n 4; "
                "echo PROCESSES; "
                "ps -eo pid=,etime=,pcpu=,rss=,args= | "
                "grep -E '[r]emote_current_dft_controller|[v]asp_std'",
            )
            print(stdout, end="")
            if stderr:
                print(stderr, file=os.sys.stderr, end="")
            return status
        if args.inspect_only:
            status, stdout, stderr = exec_remote(
                transport,
                "if [ -e /root/autodl-tmp/prospective_cr_discovery_v2 ]; "
                "then echo EXISTS; "
                "ls -la /root/autodl-tmp/prospective_cr_discovery_v2; "
                "find /root/autodl-tmp/prospective_cr_discovery_v2 "
                "-maxdepth 3 -type f | head -40; "
                "echo CONTROLLER_STATE; "
                "cat /root/autodl-tmp/prospective_cr_discovery_v2/"
                "controller_state.json 2>/dev/null; "
                "echo CONTROLLER_LOGS; "
                "tail -n 30 /root/autodl-tmp/prospective_cr_discovery_v2/"
                "controller.log 2>/dev/null; "
                "tail -n 30 /root/autodl-tmp/prospective_cr_discovery_v2/"
                "controller_repair_001.log 2>/dev/null; "
                "echo RECENT_OSZICAR; "
                "find /root/autodl-tmp/prospective_cr_discovery_v2/runs "
                "-name OSZICAR -type f -print -exec tail -n 3 {} \\; "
                "2>/dev/null | tail -30; "
                "echo RECENT_VASP_STDIO; "
                "find /root/autodl-tmp/prospective_cr_discovery_v2/runs "
                "\\( -name vasp.stdout -o -name vasp.stderr \\) "
                "-type f -print -exec tail -n 8 {} \\; "
                "2>/dev/null | tail -40; "
                "ps -eo pid=,etime=,pcpu=,rss=,args= | "
                "grep -E '[r]emote_current_dft_controller|[v]asp_std'; "
                "else echo ABSENT; fi; "
                "file /root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std; "
                "ldd /root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std "
                "| head -15; "
                "echo ALL_VASP_STD_BINARIES; "
                "find /root/software -type f -name vasp_std -perm -u+x "
                "-print 2>/dev/null; "
                "echo MPI_LAUNCHERS; "
                "command -v mpirun || true; command -v mpiexec || true",
            )
            print(stdout, end="")
            if stderr:
                print(stderr, file=os.sys.stderr, end="")
            return status
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            _mkdirs(sftp, args.remote_root)
            package_records = upload_package(
                sftp,
                Path(args.local_package),
                args.remote_root + "/input_package",
            )
            repo_root = Path(__file__).resolve().parents[1]
            script_sources = [
                repo_root / "analysis/remote_current_dft_controller.py",
                repo_root / "analysis/compute_self_consistent_fe.py",
                repo_root / "analysis/recompute_self_consistent_fe_decimal.py",
            ]
            for source in script_sources:
                sftp.put(
                    str(source),
                    args.remote_root + "/" + source.name,
                )
            deployment = {
                "remote_root": args.remote_root,
                "uploaded_file_count": len(package_records),
                "contains_potcar": False,
            }
            encoded = base64.b64encode(
                (json.dumps(deployment, indent=2) + "\n").encode("utf-8")
            ).decode("ascii")
            command = (
                f"printf %s {encoded!r} | base64 -d > "
                f"{args.remote_root}/deployment.json"
            )
            status, _, stderr = exec_remote(transport, command)
            if status:
                raise RuntimeError(stderr)
        finally:
            sftp.close()

        if args.deploy_only:
            print(json.dumps(deployment, sort_keys=True))
            return 0
        command = (
            f"cd {args.remote_root} && "
            "nohup python3 remote_current_dft_controller.py "
            f"--root {args.remote_root} "
            f"--segment-seconds {args.segment_seconds} "
            f"--phase2-workers {args.phase2_workers} "
            "> controller.log 2>&1 < /dev/null & echo $!"
        )
        status, stdout, stderr = exec_remote(transport, command)
        if status:
            raise RuntimeError(stderr)
        print(
            json.dumps(
                {
                    **deployment,
                    "controller_pid": int(stdout.strip().splitlines()[-1]),
                    "started": True,
                },
                sort_keys=True,
            )
        )
        return 0
    finally:
        transport.close()
        sock.close()


if __name__ == "__main__":
    raise SystemExit(main())
