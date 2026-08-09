"""Build immutable candidate static-verification jobs under a frozen DFT protocol."""

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

from analysis.prepare_dft_kpoint_jobs import explicit_mesh


KPOINT_RULE = "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing"
MANIFEST_COLUMNS = [
    "job_id", "dataset", "method", "group_key", "seed", "K", "config_hash",
    "git_commit", "gpu_id", "status", "start_time", "end_time", "exit_code",
    "log_path", "output_path", "sha256", "command_json", "cwd", "attempt",
    "pid", "failure_reason", "env_json", "candidate_id", "formula", "functional",
    "magnetic_initialization", "main_text_selected", "kpoint_spacing_Ainv", "mesh",
    "source_dir", "source_incar_sha256", "source_structure_sha256", "potcar_sha256",
    "frozen_protocol_sha256", "input_provenance_path",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _safe_component(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_+-]+", "_", value).strip("_")
    return token[:48] or "candidate"


def _candidate_index(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for directory in sorted(root.glob("candidate_*")):
        metadata = directory / "candidate_metadata.json"
        if not metadata.is_file():
            continue
        candidate_id = str(json.loads(metadata.read_text(encoding="utf-8"))["candidate_id"])
        if candidate_id in result:
            raise ValueError(f"duplicate candidate source for {candidate_id}")
        result[candidate_id] = directory.resolve()
    return result


def build_required_static_records(
    candidate_manifest: pd.DataFrame,
    magnetic_initializations: pd.DataFrame,
    source_root: Path,
) -> list[dict[str, object]]:
    """Select static-only verification inputs without inferring new magnetic states."""

    root = Path(source_root).resolve()
    regular = _candidate_index(root / "candidate_inputs")
    tight = _candidate_index(root / "shortlist_tight" / "shortlist_inputs")
    selected_states = {
        str(row["candidate_id"]): str(row["initialization_label_as_recorded"])
        for _, row in magnetic_initializations.iterrows()
        if _truthy(row.get("selected_lower_energy_among_two"))
    }
    records: list[dict[str, object]] = []
    eligible = candidate_manifest[
        (candidate_manifest["pilot_or_new"].astype(str) == "new")
        & (candidate_manifest["DFT_status"].astype(str) == "static_finished")
    ]
    for _, row in eligible.sort_values("candidate_id").iterrows():
        candidate_id = str(row["candidate_id"])
        if candidate_id not in regular:
            raise FileNotFoundError(f"missing retained new-candidate source for {candidate_id}")
        formula = str(row["formula"])
        main_text = _truthy(row.get("main_text_selected"))
        base = regular[candidate_id]
        records.append(
            {
                "candidate_id": candidate_id,
                "formula": formula,
                "functional": "PBE",
                "magnetic_initialization": "default",
                "source_dir": str(base / "stages" / "02_pbe_static"),
                "main_text_selected": main_text,
            }
        )
        # Mg has no Hubbard correction in the retained protocol. Its PBE result is
        # the U=0 result and is not duplicated as a second VASP calculation.
        if "Mg" in formula:
            continue
        if main_text:
            if candidate_id not in tight:
                raise FileNotFoundError(f"missing two tested magnetic initializations for {candidate_id}")
            for state in ("state_fm", "state_afm"):
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "formula": formula,
                        "functional": "GGA+U",
                        "magnetic_initialization": state,
                        "source_dir": str(tight[candidate_id] / "magnetic_states" / state / "02_static"),
                        "main_text_selected": True,
                    }
                )
        elif candidate_id in selected_states:
            state = selected_states[candidate_id]
            if candidate_id not in tight:
                raise FileNotFoundError(f"missing retained selected magnetic state for {candidate_id}")
            records.append(
                {
                    "candidate_id": candidate_id,
                    "formula": formula,
                    "functional": "GGA+U",
                    "magnetic_initialization": state,
                    "source_dir": str(tight[candidate_id] / "magnetic_states" / state / "02_static"),
                    "main_text_selected": False,
                }
            )
        else:
            records.append(
                {
                    "candidate_id": candidate_id,
                    "formula": formula,
                    "functional": "GGA+U",
                    "magnetic_initialization": "default",
                    "source_dir": str(base / "stages" / "04_gga_u_static"),
                    "main_text_selected": False,
                }
            )
    keys = [f"{row['candidate_id']}|{row['functional']}|{row['magnetic_initialization']}" for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("candidate static plan contains duplicate scientific tasks")
    return records


def _load_protocol(path: Path) -> tuple[dict[str, object], float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("frozen") is not True or payload.get("kpoint_rule") != KPOINT_RULE:
        raise ValueError("a frozen DFT protocol with the approved explicit-mesh rule is required")
    spacing = float(payload["kpoint_spacing_Ainv"])
    if spacing <= 0 or not math.isfinite(spacing):
        raise ValueError("invalid frozen k-point spacing")
    return payload, spacing


def load_vasp_command(*, command_json: str | None, command_file: Path | None) -> list[str]:
    if (command_json is None) == (command_file is None):
        raise ValueError("provide exactly one VASP command source")
    raw = command_json if command_json is not None else Path(command_file).read_text(encoding="utf-8")
    command = json.loads(raw)
    if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
        raise ValueError("VASP command must be a non-empty JSON string list")
    return command


def _write_kpoints(path: Path, mesh: tuple[int, int, int], spacing: float) -> None:
    path.write_text(
        f"Explicit Gamma mesh from frozen reciprocal spacing <= {spacing:.2f} A^-1\n"
        f"0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n",
        encoding="utf-8",
        newline="\n",
    )


def _mesh_from_kpoints(path: Path) -> tuple[int, int, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
    for line in reversed(lines):
        parts = line.split()
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)  # type: ignore[return-value]
    raise ValueError(f"cannot parse explicit mesh from {path}")


def _validate_reuse(output: Path, source: Path, mesh: tuple[int, int, int]) -> str:
    result_path = output / "task_result.json"
    manifest_path = output / "input_manifest.json"
    if not result_path.is_file() or not manifest_path.is_file() or not (output / "OUTCAR").is_file():
        raise ValueError(f"incomplete reuse output: {output}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not (
        result.get("status") == "DONE"
        and int(result.get("exit_code", 1)) == 0
        and result.get("electronic_converged") is True
        and result.get("timing_footer_present") is True
        and result.get("potcar_retained") is False
        and not (output / "POTCAR").exists()
    ):
        raise ValueError(f"reuse output did not pass completion checks: {output}")
    inputs = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("INCAR", "POSCAR", "POTCAR"):
        if inputs.get(name, {}).get("sha256") != _sha256(source / name):
            raise ValueError(f"reuse {name} hash mismatch for {output}")
    if _mesh_from_kpoints(output / "KPOINTS") != mesh:
        raise ValueError(f"reuse k-point mesh mismatch for {output}")
    return _sha256(output / "OUTCAR")


def build_candidate_static_bundle(
    *,
    records: Sequence[Mapping[str, object]],
    reuse_outputs: Mapping[str, str],
    frozen_protocol_path: Path,
    work_root: Path,
    manifest_path: Path,
    compatibility_path: Path,
    git_commit: str,
    python_executable: str,
    runner_path: Path,
    vasp_command: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate only missing static jobs and prove any reused result is identical."""

    protocol, spacing = _load_protocol(Path(frozen_protocol_path))
    if not vasp_command:
        raise ValueError("VASP command must not be empty")
    root = Path(work_root).resolve()
    protocol_path = Path(frozen_protocol_path).resolve()
    protocol_sha = _sha256(protocol_path)
    manifest_rows: list[dict[str, object]] = []
    compatibility_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in records:
        row = dict(raw)
        key = f"{row['candidate_id']}|{row['functional']}|{row['magnetic_initialization']}"
        if key in seen:
            raise ValueError(f"duplicate static task: {key}")
        seen.add(key)
        source = Path(str(row["source_dir"])).resolve()
        missing = [str(source / name) for name in ("INCAR", "POSCAR", "POTCAR") if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError("missing retained candidate input: " + "; ".join(missing))
        structure = Structure.from_file(source / "POSCAR")
        mesh = explicit_mesh(structure.lattice.reciprocal_lattice.abc, spacing)
        mesh_text = "x".join(str(value) for value in mesh)
        source_hashes = {name: _sha256(source / name) for name in ("INCAR", "POSCAR", "POTCAR")}
        common = {
            "candidate_id": str(row["candidate_id"]),
            "formula": str(row["formula"]),
            "functional": str(row["functional"]),
            "magnetic_initialization": str(row["magnetic_initialization"]),
            "main_text_selected": bool(row["main_text_selected"]),
            "source_dir": str(source),
            "required_mesh": mesh_text,
            "kpoint_spacing_Ainv": spacing,
            "source_incar_sha256": source_hashes["INCAR"],
            "source_structure_sha256": source_hashes["POSCAR"],
            "potcar_sha256": source_hashes["POTCAR"],
            "frozen_protocol_sha256": protocol_sha,
        }
        if key in reuse_outputs:
            output = Path(str(reuse_outputs[key])).resolve()
            outcar_sha = _validate_reuse(output, source, mesh)
            compatibility_rows.append(
                {
                    **common,
                    "decision": "REUSED_FROZEN_PROTOCOL_OUTPUT",
                    "existing_output_path": str(output),
                    "existing_outcar_sha256": outcar_sha,
                    "new_job_id": "",
                }
            )
            continue

        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
        job_id = f"dft_candidate_{_safe_component(str(row['candidate_id']))[:24]}_{digest}"
        input_dir = root / "inputs" / job_id
        input_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source / "INCAR", input_dir / "INCAR")
        shutil.copy2(source / "POSCAR", input_dir / "POSCAR")
        _write_kpoints(input_dir / "KPOINTS", mesh, spacing)
        provenance = {
            **common,
            "scientific_task_key": key,
            "kpoint_rule": str(protocol["kpoint_rule"]),
            "mesh": list(mesh),
            "generated_kpoints_sha256": _sha256(input_dir / "KPOINTS"),
            "potcar_content_retained_in_generated_inputs": False,
        }
        provenance_path = input_dir / "input_provenance.json"
        provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        config_hash = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        output_dir = root / "results" / job_id / "attempt_1"
        command = [
            str(python_executable), str(runner_path), "--input-dir", str(input_dir),
            "--potcar-source", str(source / "POTCAR"), "--output-dir", str(output_dir),
            "--command-json", json.dumps(list(vasp_command), separators=(",", ":")),
        ]
        manifest_rows.append(
            {
                "job_id": job_id,
                "dataset": "dft_candidate_static_verification",
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
                "log_path": str(root / "logs" / f"{job_id}.log"),
                "output_path": str(output_dir),
                "sha256": "",
                "command_json": json.dumps(command, separators=(",", ":")),
                "cwd": str(root),
                "attempt": 1,
                "pid": "",
                "failure_reason": "",
                "env_json": json.dumps(
                    {"OPENBLAS_NUM_THREADS": "8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "8", "GOTO_NUM_THREADS": "8"},
                    separators=(",", ":"),
                ),
                "candidate_id": str(row["candidate_id"]),
                "formula": str(row["formula"]),
                "functional": str(row["functional"]),
                "magnetic_initialization": str(row["magnetic_initialization"]),
                "main_text_selected": bool(row["main_text_selected"]),
                "kpoint_spacing_Ainv": spacing,
                "mesh": mesh_text,
                "source_dir": str(source),
                "source_incar_sha256": source_hashes["INCAR"],
                "source_structure_sha256": source_hashes["POSCAR"],
                "potcar_sha256": source_hashes["POTCAR"],
                "frozen_protocol_sha256": protocol_sha,
                "input_provenance_path": str(provenance_path),
            }
        )
        compatibility_rows.append(
            {
                **common,
                "decision": "REQUIRES_STATIC_VERIFICATION",
                "existing_output_path": "",
                "existing_outcar_sha256": "",
                "new_job_id": job_id,
            }
        )

    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    compatibility = pd.DataFrame(compatibility_rows)
    manifest_file = Path(manifest_path)
    compatibility_file = Path(compatibility_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    compatibility_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("x", encoding="utf-8", newline="") as handle:
        manifest.to_csv(handle, index=False, lineterminator="\n")
    with compatibility_file.open("x", encoding="utf-8", newline="") as handle:
        compatibility.to_csv(handle, index=False, lineterminator="\n")
    return manifest, compatibility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--magnetic-initializations", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reuse-map", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    command_group = parser.add_mutually_exclusive_group(required=True)
    command_group.add_argument("--vasp-command-json")
    command_group.add_argument("--vasp-command-file", type=Path)
    args = parser.parse_args()
    candidate_manifest = pd.read_csv(args.candidate_manifest, dtype=str, keep_default_na=False)
    magnetic = pd.read_csv(args.magnetic_initializations, dtype=str, keep_default_na=False)
    reuse = json.loads(args.reuse_map.read_text(encoding="utf-8"))
    if not isinstance(reuse, dict):
        raise ValueError("reuse map must be a JSON object")
    command = load_vasp_command(command_json=args.vasp_command_json, command_file=args.vasp_command_file)
    records = build_required_static_records(candidate_manifest, magnetic, args.source_root)
    manifest, compatibility = build_candidate_static_bundle(
        records=records,
        reuse_outputs={str(key): str(value) for key, value in reuse.items()},
        frozen_protocol_path=args.frozen_protocol,
        work_root=args.work_root,
        manifest_path=args.manifest,
        compatibility_path=args.compatibility,
        git_commit=args.git_commit,
        python_executable=args.python_executable,
        runner_path=args.runner_path,
        vasp_command=command,
    )
    print(json.dumps({"required_static_records": len(compatibility), "new_jobs": len(manifest), "reused": len(compatibility) - len(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
