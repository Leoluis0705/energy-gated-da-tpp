#!/usr/bin/env python3
"""Read-only audit of the authorized DFT server.

The password is requested with a no-echo prompt and is never written to disk.
POTCAR files are inspected only on the remote host; only paths, TITEL labels,
sizes, and SHA-256 digests are returned.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path

import paramiko


EXPECTED_HOST_KEY = "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
EXPECTED_VASP_SHA256 = (
    "2abdcfedd1c3e7962a56404bd14cc340dcb170867720921fcea8ec7058ef3d94"
)
EXPECTED_POTCAR_SHA256 = {
    "Li_sv": "201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b",
    "Cr_pv": "836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b",
    "Mn_pv": "5da6f925d338804d4244b231b43927f3fccb021320fc01db0f865b4eda5a756d",
    "Mg_pv": "9c1ab7b832a2e0c6a472613699d255f5c01190d69448d45cb2d6283bcfff58ac",
    "O": "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64",
}
EXPECTED_TITEL_SUBSTRINGS = {
    "Li_sv": "PAW_PBE Li_sv",
    "Cr_pv": "PAW_PBE Cr_pv",
    "Mn_pv": "PAW_PBE Mn_pv",
    "Mg_pv": "PAW_PBE Mg_pv",
    "O": "PAW_PBE O ",
}


def host_key_sha256(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def _exec(transport: paramiko.Transport, command: str, timeout: int = 300) -> str:
    channel = transport.open_session(timeout=20)
    try:
        channel.settimeout(timeout)
        channel.exec_command(command)
        stdout = channel.makefile("rb").read().decode("utf-8", "replace")
        stderr = channel.makefile_stderr("rb").read().decode("utf-8", "replace")
        status = channel.recv_exit_status()
    finally:
        channel.close()
    if status:
        raise RuntimeError(
            f"remote read-only audit exited {status}: {stderr.strip() or stdout.strip()}"
        )
    return stdout


def _remote_source() -> str:
    return r'''
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

def run(argv):
    proc = subprocess.run(argv, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    return {"returncode": proc.returncode, "output": proc.stdout.strip()}

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def first_titel(path):
    with path.open("r", encoding="latin-1", errors="replace") as handle:
        for _ in range(20):
            line = handle.readline()
            if not line:
                break
            if "TITEL" in line:
                return line.strip()
    return ""

def cgroup_value(path):
    p = Path(path)
    return p.read_text().strip() if p.exists() else None

vasp = Path("/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std")
potcars = []
for root in [Path("/root/software"), Path("/root/vasp"), Path("/opt/vasp")]:
    if not root.exists():
        continue
    for path in root.rglob("POTCAR"):
        if not path.is_file():
            continue
        titel = first_titel(path)
        if any(label in titel for label in
               ["PAW_PBE Li_sv", "PAW_PBE Cr_pv", "PAW_PBE Mn_pv",
                "PAW_PBE Mg_pv", "PAW_PBE O "]):
            potcars.append({
                "path": str(path),
                "titel": titel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

existing_outcar = None
for root in [Path("/root/dft_limo_new12_20260712"), Path("/root/dft_limo")]:
    if root.exists():
        existing_outcar = next(root.rglob("OUTCAR"), None)
        if existing_outcar:
            break
vasp_version = ""
if existing_outcar:
    with existing_outcar.open("r", encoding="latin-1", errors="replace") as handle:
        for _ in range(80):
            line = handle.readline()
            if "vasp." in line.lower():
                vasp_version = line.strip()
                break

memory_max = cgroup_value("/sys/fs/cgroup/memory.max")
cpu_max = cgroup_value("/sys/fs/cgroup/cpu.max")
disk = shutil.disk_usage("/root/autodl-tmp")
processes = run(["bash", "-lc",
    "ps -eo args= | grep -E '[v]asp_(std|gam|ncl)|[r]un_minimal_dft' || true"])
python_probe = run(["python3", "-c",
    "import json,sys; import pymatgen; "
    "print(json.dumps({'python':sys.version.split()[0],"
    "'pymatgen':getattr(pymatgen,'__version__','import-ok')}))"])
scheduler = {
    name: shutil.which(name)
    for name in ["sbatch", "squeue", "qsub", "qstat"]
}

payload = {
    "hostname": platform.node(),
    "kernel": platform.release(),
    "nproc": os.cpu_count(),
    "cpu_max": cpu_max,
    "memory_max": memory_max,
    "autodl_tmp": {
        "total_bytes": disk.total,
        "free_bytes": disk.free,
    },
    "vasp": {
        "path": str(vasp),
        "exists": vasp.is_file(),
        "executable": os.access(vasp, os.X_OK),
        "sha256": sha256(vasp) if vasp.is_file() else None,
        "version_evidence": vasp_version,
    },
    "potcars": sorted(potcars, key=lambda row: row["path"]),
    "python_probe": python_probe,
    "scheduler": scheduler,
    "running_processes": processes["output"],
    "remote_roots": {
        path: Path(path).exists()
        for path in ["/root/dft_limo_new12_20260712", "/root/dft_limo",
                     "/root/autodl-tmp"]
    },
}
print(json.dumps(payload, sort_keys=True))
'''


def evaluate(payload: dict, fingerprint: str) -> dict:
    observed: dict[str, list[dict]] = {label: [] for label in EXPECTED_POTCAR_SHA256}
    for row in payload.get("potcars", []):
        for label, title in EXPECTED_TITEL_SUBSTRINGS.items():
            if title in str(row.get("titel", "")):
                observed[label].append(row)

    paw_status = {}
    for label, expected_hash in EXPECTED_POTCAR_SHA256.items():
        matching = [
            row
            for row in observed[label]
            if str(row.get("sha256", "")).lower() == expected_hash.lower()
        ]
        paw_status[label] = {
            "pass": bool(matching),
            "expected_sha256": expected_hash,
            "matches": matching,
        }

    checks = {
        "host_key": fingerprint == EXPECTED_HOST_KEY,
        "vasp_exists": bool(payload.get("vasp", {}).get("exists")),
        "vasp_executable": bool(payload.get("vasp", {}).get("executable")),
        "vasp_sha256": (
            str(payload.get("vasp", {}).get("sha256", "")).lower()
            == EXPECTED_VASP_SHA256
        ),
        "vasp_version_6_5_1": "6.5.1" in str(
            payload.get("vasp", {}).get("version_evidence", "")
        ),
        "paw_labels_and_hashes": all(row["pass"] for row in paw_status.values()),
        "python_probe": payload.get("python_probe", {}).get("returncode") == 0,
        "work_root_exists": bool(
            payload.get("remote_roots", {}).get("/root/autodl-tmp")
        ),
        "no_running_minimal_batch": not str(
            payload.get("running_processes", "")
        ).strip(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "pass": all(checks.values()),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_key_sha256": fingerprint,
        "checks": checks,
        "paw_status": paw_status,
        "inventory": payload,
        "execution_path": {
            "vasp": "/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std",
            "mode": "serial OpenBLAS",
            "openblas_threads": 8,
            "scheduler": "none",
        },
        "potcar_policy": (
            "server-side assembly only; only labels, paths, sizes, and hashes audited"
        ),
    }


def audit(args: argparse.Namespace) -> int:
    password = getpass.getpass("DFT server password (not stored): ")
    sock = socket.create_connection((args.host, args.port), timeout=20)
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=20)
        fingerprint = host_key_sha256(transport.get_remote_server_key())
        if fingerprint != args.expected_host_key:
            raise RuntimeError(
                f"SSH host-key mismatch: observed {fingerprint}; expected "
                f"{args.expected_host_key}"
            )
        transport.auth_password(args.user, password)
        if not transport.is_authenticated():
            raise RuntimeError("SSH password authentication failed")
        encoded = base64.b64encode(_remote_source().encode("utf-8")).decode("ascii")
        command = (
            "python3 -c "
            + repr(f"import base64;exec(base64.b64decode({encoded!r}))")
        )
        raw = _exec(transport, command, timeout=args.timeout)
    finally:
        password = ""
        transport.close()
        sock.close()

    payload = json.loads(raw)
    result = evaluate(payload, fingerprint)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "hostname": payload.get("hostname"),
                "vasp_sha256_pass": result["checks"]["vasp_sha256"],
                "paw_labels_and_hashes_pass": result["checks"][
                    "paw_labels_and_hashes"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--expected-host-key", default=EXPECTED_HOST_KEY)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
