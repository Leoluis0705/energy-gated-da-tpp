"""Generate a common-spacing explicit-mesh VASP k-point convergence batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from pymatgen.core import Structure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def explicit_mesh(reciprocal_lengths: Sequence[float], spacing_Ainv: float) -> tuple[int, int, int]:
    values = tuple(float(value) for value in reciprocal_lengths)
    spacing = float(spacing_Ainv)
    if len(values) != 3 or spacing <= 0 or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("reciprocal lengths must contain three positive finite values and spacing must be positive")
    return tuple(max(1, int(math.ceil(value / spacing))) for value in values)  # type: ignore[return-value]


def _write_kpoints(path: Path, mesh: tuple[int, int, int], spacing: float) -> None:
    text = (
        f"Explicit Gamma mesh from reciprocal spacing <= {spacing:.2f} A^-1\n"
        "0\n"
        "Gamma\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _safe_system_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"unsafe system name: {value!r}")
    return value


def build_kpoint_bundle(
    *,
    system_sources: Mapping[str, Path],
    spacings: Sequence[float],
    work_root: Path,
    manifest_path: Path,
    git_commit: str,
    python_executable: str,
    runner_path: Path,
    vasp_command: Sequence[str],
) -> pd.DataFrame:
    spacing_values = tuple(float(value) for value in spacings)
    if len(system_sources) != 3 or not spacing_values:
        raise ValueError("k-point batch must contain exactly three systems and at least one spacing")
    if len(set(spacing_values)) != len(spacing_values):
        raise ValueError("k-point spacings must be unique")
    if not vasp_command:
        raise ValueError("VASP command must not be empty")
    root = Path(work_root).resolve()
    rows: list[dict[str, object]] = []
    for system, raw_source in system_sources.items():
        name = _safe_system_name(str(system))
        source = Path(raw_source).resolve()
        required = [source / "INCAR", source / "POSCAR", source / "POTCAR"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing server-side VASP source inputs: " + "; ".join(missing))
        structure = Structure.from_file(source / "POSCAR")
        reciprocal_lengths = structure.lattice.reciprocal_lattice.abc
        for spacing in spacing_values:
            spacing_value = float(spacing)
            spacing_token = f"{spacing_value:.2f}".replace(".", "p")
            mesh = explicit_mesh(reciprocal_lengths, spacing_value)
            mesh_text = "x".join(str(value) for value in mesh)
            input_dir = root / "inputs" / name / f"spacing_{spacing_token}"
            input_dir.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source / "INCAR", input_dir / "INCAR")
            shutil.copy2(source / "POSCAR", input_dir / "POSCAR")
            _write_kpoints(input_dir / "KPOINTS", mesh, spacing_value)
            provenance = {
                "system": name,
                "source_dir": str(source),
                "spacing_Ainv": spacing_value,
                "mesh": list(mesh),
                "reciprocal_lengths_Ainv": list(reciprocal_lengths),
                "atom_count": len(structure),
                "INCAR_sha256": _sha256(source / "INCAR"),
                "POSCAR_sha256": _sha256(source / "POSCAR"),
                "KPOINTS_sha256": _sha256(input_dir / "KPOINTS"),
                "POTCAR_source_path": str(source / "POTCAR"),
                "POTCAR_sha256": _sha256(source / "POTCAR"),
                "potcar_content_retained_in_generated_inputs": False,
            }
            provenance_path = input_dir / "input_provenance.json"
            with provenance_path.open("x", encoding="utf-8") as handle:
                json.dump(provenance, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
            config_hash = hashlib.sha256(
                json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            job_id = f"dft_kpoint_{name}_spacing_{spacing_token}"
            output_dir = root / "results" / name / f"spacing_{spacing_token}" / "attempt_1"
            log_path = root / "logs" / f"{job_id}.log"
            command = [
                str(python_executable),
                str(runner_path),
                "--input-dir",
                str(input_dir),
                "--potcar-source",
                str(source / "POTCAR"),
                "--output-dir",
                str(output_dir),
                "--command-json",
                json.dumps(list(vasp_command), separators=(",", ":")),
            ]
            rows.append(
                {
                    "job_id": job_id,
                    "dataset": "dft_kpoint_convergence",
                    "method": "serial_vasp_static",
                    "group_key": "not_applicable",
                    "seed": "",
                    "K": mesh_text,
                    "config_hash": config_hash,
                    "git_commit": git_commit,
                    "gpu_id": "",
                    "status": "PENDING",
                    "start_time": "",
                    "end_time": "",
                    "exit_code": "",
                    "log_path": str(log_path),
                    "output_path": str(output_dir),
                    "sha256": "",
                    "command_json": json.dumps(command, separators=(",", ":")),
                    "cwd": str(root),
                    "attempt": 1,
                    "pid": "",
                    "failure_reason": "",
                    "env_json": json.dumps(
                        {
                            "OPENBLAS_NUM_THREADS": "8",
                            "OMP_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "8",
                            "GOTO_NUM_THREADS": "8",
                        },
                        separators=(",", ":"),
                    ),
                    "system": name,
                    "kpoint_spacing_Ainv": spacing_value,
                    "mesh": mesh_text,
                    "atom_count": len(structure),
                    "source_dir": str(source),
                    "potcar_sha256": provenance["POTCAR_sha256"],
                    "input_provenance_path": str(provenance_path),
                }
            )
    frame = pd.DataFrame(rows)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", action="append", required=True, help="NAME=/server/source/directory")
    parser.add_argument("--spacing", action="append", type=float, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--vasp-command-json", required=True)
    args = parser.parse_args()
    systems: dict[str, Path] = {}
    for value in args.system:
        if "=" not in value:
            raise ValueError("--system must use NAME=PATH")
        name, path = value.split("=", 1)
        systems[name] = Path(path)
    command = json.loads(args.vasp_command_json)
    if not isinstance(command, list):
        raise ValueError("VASP command JSON must be a list")
    frame = build_kpoint_bundle(
        system_sources=systems,
        spacings=args.spacing,
        work_root=args.work_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
        python_executable=args.python_executable,
        runner_path=args.runner_path,
        vasp_command=command,
    )
    print(json.dumps({"jobs": len(frame), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
