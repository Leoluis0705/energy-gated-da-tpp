#!/usr/bin/env python3
"""Run one isolated serial VASP task without retaining POTCAR in its output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


STRUCTURE_INPUTS = ("INCAR", "KPOINTS", "POSCAR")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _parse_outcar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"final_toten_ev": None, "electronic_converged": False, "timing_footer_present": False}
    text = path.read_text(encoding="utf-8", errors="ignore")
    energies = re.findall(r"free\s+energy\s+TOTEN\s*=\s*([+\-0-9.Ee]+)\s+eV", text)
    return {
        "final_toten_ev": float(energies[-1]) if energies else None,
        "electronic_converged": "aborting loop because EDIFF is reached" in text,
        "timing_footer_present": "General timing and accounting" in text,
    }


def run_vasp_task(
    input_dir: str | Path,
    output_dir: str | Path,
    command: Sequence[str],
    environment: dict[str, str] | None = None,
    potcar_source: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(input_dir).resolve()
    output = Path(output_dir).resolve()
    potcar = Path(potcar_source).resolve() if potcar_source is not None else source / "POTCAR"
    missing = [name for name in STRUCTURE_INPUTS if not (source / name).is_file()]
    if not potcar.is_file():
        missing.append("POTCAR")
    if missing:
        raise FileNotFoundError(f"missing VASP benchmark inputs: {missing}")
    if not command:
        raise ValueError("VASP command must not be empty")
    output.mkdir(parents=True, exist_ok=False)
    input_sources = {name: source / name for name in STRUCTURE_INPUTS}
    input_sources["POTCAR"] = potcar
    input_manifest = {
        name: {"source_path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in input_sources.items()
    }
    _write_json_exclusive(output / "input_manifest.json", input_manifest)
    (output / "command.txt").write_text(json.dumps(list(command), ensure_ascii=False) + "\n", encoding="utf-8")
    for name in STRUCTURE_INPUTS:
        shutil.copy2(source / name, output / name)
    shutil.copy2(potcar, output / "POTCAR")
    started_at = _utc_now()
    started = time.monotonic()
    exit_code: int | None = None
    error: str | None = None
    log_path = output / "vasp.stdout_stderr.log"
    try:
        process_environment = None
        if environment is not None:
            import os

            process_environment = os.environ.copy()
            process_environment.update({str(key): str(value) for key, value in environment.items()})
        with log_path.open("xb") as log:
            completed = subprocess.run(
                [str(value) for value in command],
                cwd=output,
                env=process_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        exit_code = int(completed.returncode)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        (output / "POTCAR").unlink(missing_ok=True)
    parsed = _parse_outcar(output / "OUTCAR")
    result = {
        "status": "DONE" if exit_code == 0 else "FAILED",
        "exit_code": exit_code,
        "error": error,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "command": list(command),
        "output_dir": str(output),
        "potcar_retained": (output / "POTCAR").exists(),
        **parsed,
    }
    _write_json_exclusive(output / "task_result.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--command-json", required=True)
    parser.add_argument("--potcar-source")
    args = parser.parse_args()
    command = json.loads(args.command_json)
    if not isinstance(command, list):
        raise ValueError("command JSON must be a list")
    result = run_vasp_task(
        args.input_dir,
        args.output_dir,
        command,
        potcar_source=args.potcar_source,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "DONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
