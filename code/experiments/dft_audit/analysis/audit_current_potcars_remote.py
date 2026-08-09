"""Audit modern PAW-PBE POTCAR metadata on the licensed VASP server.

Only compact metadata leave the server. POTCAR bodies are never returned,
downloaded, or written locally.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import paramiko


EXPECTED_HOST_KEY = "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc"
EXPECTED_VASP_SHA256 = (
    "2abdcfedd1c3e7962a56404bd14cc340dcb170867720921fcea8ec7058ef3d94"
)
EXPECTED_POTCARS = {
    "Li": {
        "titel": "PAW_PBE Li_sv 10Sep2004",
        "zval": 3.0,
        "sha256": "201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b",
        "path": "/root/software/potpaw_PBE/Li_sv/POTCAR",
    },
    "Cr": {
        "titel": "PAW_PBE Cr_pv 02Aug2007",
        "zval": 12.0,
        "sha256": "836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b",
        "path": "/root/software/potpaw_PBE/Cr_pv/POTCAR",
    },
    "O": {
        "titel": "PAW_PBE O 08Apr2002",
        "zval": 6.0,
        "sha256": "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64",
        "path": "/root/software/potpaw_PBE/O/POTCAR",
    },
}


def host_key_sha256(key: paramiko.PKey) -> str:
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def evaluate_potcar_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the exact modern PAW-PBE metadata and derive the cutoff."""

    failures: list[str] = []
    if payload.get("host_key_sha256") != EXPECTED_HOST_KEY:
        failures.append("host_key_mismatch")
    if payload.get("license_scope") != "USER_AUTHORIZED_LICENSED_VASP_SERVER":
        failures.append("license_scope_unconfirmed")
    library = payload.get("library", {})
    if not library.get("exists"):
        failures.append("potcar_library_missing")
    if library.get("source_class") != "server-installed VASP PAW-PBE library":
        failures.append("potcar_library_source_unconfirmed")

    observed = payload.get("potcars", {})
    enmax_values: list[float] = []
    for element, expected in EXPECTED_POTCARS.items():
        row = observed.get(element)
        if not isinstance(row, dict):
            failures.append(f"{element}:missing")
            continue
        for field in ("titel", "sha256", "path"):
            if str(row.get(field)) != str(expected[field]):
                failures.append(f"{element}:{field}_mismatch")
        if not math.isclose(
            float(row.get("zval", float("nan"))),
            float(expected["zval"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            failures.append(f"{element}:zval_mismatch")
        if row.get("lexch") != "PE":
            failures.append(f"{element}:lexch_not_PE")
        if row.get("functional") != "PBE":
            failures.append(f"{element}:functional_not_PBE")
        if row.get("potential_type") != "PAW":
            failures.append(f"{element}:potential_type_not_PAW")
        if int(row.get("size_bytes", 0)) <= 0:
            failures.append(f"{element}:empty_file")
        try:
            enmax = float(row["enmax_eV"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{element}:invalid_enmax")
        else:
            if enmax <= 0:
                failures.append(f"{element}:invalid_enmax")
            enmax_values.append(enmax)

    vasp = payload.get("vasp", {})
    if not vasp.get("exists") or not vasp.get("executable"):
        failures.append("vasp_unavailable")
    if str(vasp.get("version")) != "6.5.1":
        failures.append("vasp_version_mismatch")
    if str(vasp.get("sha256")) != EXPECTED_VASP_SHA256:
        failures.append("vasp_sha256_mismatch")

    maximum_enmax = max(enmax_values, default=float("nan"))
    formula_cutoff = max(520.0, 1.3 * maximum_enmax)
    frozen_cutoff = math.ceil(formula_cutoff) if math.isfinite(formula_cutoff) else None
    return {
        **payload,
        "schema": "CURRENT_POTCAR_HASH_MANIFEST_V1",
        "protocol_name": "CURRENT_SELF_CONSISTENT_PAW_PBE_U",
        "status": "PASS" if not failures else "FAIL",
        "failure_reasons": failures,
        "maximum_ENMAX_eV": maximum_enmax,
        "encut_formula": "max(520 eV, 1.3 * maximum POTCAR ENMAX)",
        "encut_formula_eV": formula_cutoff,
        "encut_eV": frozen_cutoff,
        "potcar_body_transferred": False,
        "exact_mp_compatibility_claimed": False,
    }


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
from pymatgen.io.vasp.inputs import PotcarSingle

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

paths = {
    "Li": Path("/root/software/potpaw_PBE/Li_sv/POTCAR"),
    "Cr": Path("/root/software/potpaw_PBE/Cr_pv/POTCAR"),
    "O": Path("/root/software/potpaw_PBE/O/POTCAR"),
}
potcars = {}
for element, path in paths.items():
    parsed = PotcarSingle.from_file(path)
    potcars[element] = {
        "path": str(path.resolve()),
        "titel": parsed.TITEL,
        "lexch": parsed.LEXCH,
        "zval": float(parsed.ZVAL),
        "enmax_eV": float(parsed.ENMAX),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "functional": parsed.functional,
        "potential_type": parsed.potential_type,
        "symbol": parsed.symbol,
        "owner_uid": path.stat().st_uid,
        "mode_octal": oct(path.stat().st_mode & 0o777),
    }

version = ""
version_evidence = ""
for root in [Path("/root/autodl-tmp"), Path("/root/dft_limo")]:
    if not root.exists():
        continue
    for outcar in root.rglob("OUTCAR"):
        try:
            with outcar.open("r", encoding="latin-1", errors="replace") as handle:
                head = "".join(handle.readlines()[:100])
        except OSError:
            continue
        match = re.search(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", head, re.I)
        if match:
            version = match.group(1)
            version_evidence = str(outcar)
            break
    if version:
        break

vasp = Path("/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std")
library = Path("/root/software/potpaw_PBE")
running = subprocess.run(
    ["bash", "-lc",
     "ps -eo pid=,args= | grep -E '[v]asp_(std|gam|ncl)|[r]un_prospective' || true"],
    text=True, capture_output=True, check=False,
).stdout.strip()
payload = {
    "hostname": platform.node(),
    "license_scope": "USER_AUTHORIZED_LICENSED_VASP_SERVER",
    "library": {
        "path": str(library.resolve()),
        "exists": library.is_dir(),
        "source_class": "server-installed VASP PAW-PBE library",
        "owner_uid": library.stat().st_uid if library.exists() else None,
        "mode_octal": oct(library.stat().st_mode & 0o777) if library.exists() else None,
    },
    "potcars": potcars,
    "vasp": {
        "path": str(vasp),
        "exists": vasp.is_file(),
        "executable": os.access(vasp, os.X_OK),
        "sha256": sha256(vasp) if vasp.is_file() else None,
        "version": version,
        "version_evidence_path": version_evidence,
    },
    "resources": {
        "nproc_visible": os.cpu_count(),
        "cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip(),
        "memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "autodl_tmp_free_bytes": shutil.disk_usage("/root/autodl-tmp").free,
    },
    "scheduler": {
        name: shutil.which(name)
        for name in ["sbatch", "squeue", "qsub", "qstat"]
    },
    "running_vasp_or_batch_processes": running,
    "remote_python": subprocess.run(
        ["python3", "-c",
         "import sys,importlib.metadata as m; "
         "print(sys.version.split()[0]+'|'+m.version('pymatgen'))"],
        text=True, capture_output=True, check=False,
    ).stdout.strip(),
}
print(json.dumps(payload, sort_keys=True))
'''


def collect_remote_inventory(
    *,
    host: str,
    port: int,
    user: str,
    key_path: Path,
    expected_host_key: str = EXPECTED_HOST_KEY,
) -> dict[str, Any]:
    key = paramiko.Ed25519Key.from_private_key_file(str(key_path.expanduser()))
    sock = socket.create_connection((host, port), timeout=20)
    transport = paramiko.Transport(sock)
    try:
        transport.start_client(timeout=20)
        fingerprint = host_key_sha256(transport.get_remote_server_key())
        if fingerprint != expected_host_key:
            raise RuntimeError(
                f"SSH host-key mismatch: {fingerprint} != {expected_host_key}"
            )
        transport.auth_publickey(user, key)
        if not transport.is_authenticated():
            raise RuntimeError("SSH public-key authentication failed")
        encoded = base64.b64encode(_remote_source().encode("utf-8")).decode("ascii")
        channel = transport.open_session(timeout=20)
        try:
            channel.settimeout(600)
            channel.exec_command(
                "python3 -c "
                + repr(f"import base64;exec(base64.b64decode({encoded!r}))")
            )
            stdout = channel.makefile("rb").read().decode("utf-8", "replace")
            stderr = (
                channel.makefile_stderr("rb").read().decode("utf-8", "replace")
            )
            status = int(channel.recv_exit_status())
        finally:
            channel.close()
        if status:
            raise RuntimeError(
                f"remote POTCAR audit exited {status}: {stderr.strip()}"
            )
    finally:
        transport.close()
        sock.close()

    payload = json.loads(stdout)
    payload.update(
        {
            "host": host,
            "port": port,
            "user": user,
            "host_key_sha256": fingerprint,
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    return evaluate_potcar_inventory(payload)


def write_audit_report(audit: dict[str, Any], path: Path) -> None:
    rows = []
    for element in ("Li", "Cr", "O"):
        row = audit["potcars"][element]
        rows.append(
            f"| {element} | `{row['titel']}` | `{row['lexch']}` | "
            f"{row['zval']:.3f} | {row['enmax_eV']:.3f} | "
            f"`{row['sha256']}` | `{row['path']}` |"
        )
    text = f"""# Current POTCAR Audit

- Status: **{audit["status"]}**
- Protocol: `CURRENT_SELF_CONSISTENT_PAW_PBE_U`
- Server: `{audit["user"]}@{audit["host"]}:{audit["port"]}`
- Host key: `{audit["host_key_sha256"]}`
- VASP: `{audit["vasp"]["version"]}` (`{audit["vasp"]["sha256"]}`)
- Library source: `{audit["library"]["source_class"]}` at `{audit["library"]["path"]}`
- License scope: `{audit["license_scope"]}` (user-authorized licensed server)
- POTCAR bodies transferred: `no`
- Exact MP compatibility claimed: `no`

| Element | TITEL | LEXCH | ZVAL | ENMAX (eV) | SHA-256 | Server path |
|---|---|---:|---:|---:|---|---|
{chr(10).join(rows)}

The largest ENMAX is {audit["maximum_ENMAX_eV"]:.3f} eV.
`max(520, 1.3 × ENMAX_max) = {audit["encut_formula_eV"]:.4f} eV`;
the frozen operational ENCUT is the upward-rounded `{audit["encut_eV"]} eV`.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="reports/CURRENT_POTCAR_AUDIT.md")
    args = parser.parse_args()

    audit = collect_remote_inventory(
        host=args.host,
        port=args.port,
        user=args.user,
        key_path=Path(args.key),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_audit_report(audit, Path(args.report))
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(output),
                "report": args.report,
                "encut_eV": audit["encut_eV"],
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
