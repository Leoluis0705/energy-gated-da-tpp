"""Prepare audited elemental-reference VASP jobs from a frozen DFT protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from pymatgen.core import Structure

from analysis.prepare_dft_kpoint_jobs import explicit_mesh


EXPECTED_REFERENCES = {
    "PBE_Li_metal": ("Li_metal", "Li", "PBE", None, False),
    "PBE_Cr_metal": ("Cr_metal", "Cr", "PBE", None, False),
    "PBE_Mn_metal": ("Mn_metal", "Mn", "PBE", None, False),
    "PBE_Mg_metal": ("Mg_metal", "Mg", "PBE", None, False),
    "PBE_O2_molecule": ("O2_molecule", "O", "PBE", None, True),
    "GGA_U_Li_metal": ("Li_metal", "Li", "GGA+U", 0.0, False),
    "GGA_U_Cr_metal": ("Cr_metal", "Cr", "GGA+U", 3.7, False),
    "GGA_U_Mn_metal": ("Mn_metal", "Mn", "GGA+U", 3.9, False),
    "GGA_U_O2_molecule": ("O2_molecule", "O", "GGA+U", 0.0, True),
}
KPOINT_RULE = "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_reference_id(value: object) -> str:
    reference_id = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", reference_id):
        raise ValueError(f"unsafe reference_id: {reference_id!r}")
    return reference_id


def _normalise_ueff(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _validate_plan(reference_plan: Sequence[Mapping[str, object]]) -> None:
    if len(reference_plan) != len(EXPECTED_REFERENCES):
        raise ValueError("reference plan must contain exactly the nine retained reference protocols")
    records = {str(record.get("reference_id")): record for record in reference_plan}
    if len(records) != len(reference_plan) or set(records) != set(EXPECTED_REFERENCES):
        raise ValueError("reference plan must contain exactly the nine retained reference protocols")
    for reference_id, expected in EXPECTED_REFERENCES.items():
        record = records[reference_id]
        observed = (
            str(record.get("reference_name")),
            str(record.get("element")),
            str(record.get("functional")),
            _normalise_ueff(record.get("Ueff_eV")),
            bool(record.get("molecular_gamma_only")),
        )
        if observed != expected:
            raise ValueError(f"reference plan metadata mismatch for {reference_id}: {observed!r}")


def _load_protocol(path: Path) -> tuple[dict[str, object], float]:
    protocol_path = Path(path)
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    if payload.get("frozen") is not True or payload.get("kpoint_rule") != KPOINT_RULE:
        raise ValueError("a frozen DFT protocol with the approved explicit-mesh rule is required")
    spacing = float(payload["kpoint_spacing_Ainv"])
    if spacing <= 0:
        raise ValueError("frozen DFT protocol contains an invalid k-point spacing")
    return payload, spacing


def _write_kpoints(
    path: Path,
    *,
    mesh: tuple[int, int, int],
    spacing: float,
    molecular_gamma_only: bool,
) -> None:
    if molecular_gamma_only:
        title = "Gamma-only isolated molecular reference"
    else:
        title = f"Explicit Gamma mesh from frozen reciprocal spacing <= {spacing:.2f} A^-1"
    path.write_text(
        f"{title}\n0\nGamma\n{mesh[0]} {mesh[1]} {mesh[2]}\n",
        encoding="utf-8",
        newline="\n",
    )


def build_elemental_reference_bundle(
    *,
    reference_plan: Sequence[Mapping[str, object]],
    frozen_protocol_path: Path,
    work_root: Path,
    manifest_path: Path,
    git_commit: str,
    python_executable: str,
    runner_path: Path,
    vasp_command: Sequence[str],
    selected_reference_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Create pending static jobs while retaining only POTCAR paths and hashes."""

    _validate_plan(reference_plan)
    protocol, spacing = _load_protocol(Path(frozen_protocol_path))
    if not vasp_command:
        raise ValueError("VASP command must not be empty")
    root = Path(work_root).resolve()
    protocol_path = Path(frozen_protocol_path).resolve()
    protocol_sha256 = _sha256(protocol_path)
    selected = None
    if selected_reference_ids is not None:
        selected_list = [str(value) for value in selected_reference_ids]
        if len(selected_list) != len(set(selected_list)):
            raise ValueError("selected reference IDs must be unique")
        unknown = set(selected_list).difference(EXPECTED_REFERENCES)
        if unknown:
            raise ValueError(f"unknown selected reference IDs: {sorted(unknown)}")
        selected = set(selected_list)
    rows: list[dict[str, object]] = []

    for raw_record in reference_plan:
        record = dict(raw_record)
        reference_id = _safe_reference_id(record["reference_id"])
        if selected is not None and reference_id not in selected:
            continue
        source = Path(str(record["source_dir"])).resolve()
        incar_filename = str(record.get("incar_filename") or "INCAR")
        if Path(incar_filename).name != incar_filename:
            raise ValueError(f"unsafe INCAR filename for {reference_id}")
        structure_filename = str(record.get("structure_filename") or "POSCAR")
        if Path(structure_filename).name != structure_filename:
            raise ValueError(f"unsafe structure filename for {reference_id}")
        source_incar = source / incar_filename
        source_structure = source / structure_filename
        source_potcar = source / "POTCAR"
        missing = [
            str(path)
            for path in (source_incar, source_structure, source_potcar)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError("missing server-side reference input: " + "; ".join(missing))

        structure = Structure.from_file(source_structure)
        molecular_gamma_only = bool(record["molecular_gamma_only"])
        mesh = (
            (1, 1, 1)
            if molecular_gamma_only
            else explicit_mesh(structure.lattice.reciprocal_lattice.abc, spacing)
        )
        mesh_text = "x".join(str(value) for value in mesh)
        kpoint_basis = (
            "isolated_molecule_gamma_only_retained_protocol"
            if molecular_gamma_only
            else "frozen_reciprocal_spacing_rule"
        )
        input_dir = root / "inputs" / reference_id
        input_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_incar, input_dir / "INCAR")
        shutil.copy2(source_structure, input_dir / "POSCAR")
        _write_kpoints(
            input_dir / "KPOINTS",
            mesh=mesh,
            spacing=spacing,
            molecular_gamma_only=molecular_gamma_only,
        )

        provenance = {
            "reference_id": reference_id,
            "reference_name": str(record["reference_name"]),
            "element": str(record["element"]),
            "functional": str(record["functional"]),
            "Ueff_eV": _normalise_ueff(record.get("Ueff_eV")),
            "structure": str(record["structure"]),
            "magnetic_setup": str(record["magnetic_setup"]),
            "paw_label": str(record["paw_label"]),
            "source_dir": str(source),
            "source_incar_filename": incar_filename,
            "source_structure_filename": structure_filename,
            "source_structure_path": str(source_structure),
            "atom_count": len(structure),
            "frozen_protocol_path": str(protocol_path),
            "frozen_protocol_sha256": protocol_sha256,
            "kpoint_rule": str(protocol["kpoint_rule"]),
            "kpoint_spacing_Ainv": spacing,
            "kpoint_basis": kpoint_basis,
            "mesh": list(mesh),
            "INCAR_sha256": _sha256(source_incar),
            "POSCAR_sha256": _sha256(source_structure),
            "source_structure_sha256": _sha256(source_structure),
            "KPOINTS_sha256": _sha256(input_dir / "KPOINTS"),
            "POTCAR_source_path": str(source_potcar),
            "POTCAR_sha256": _sha256(source_potcar),
            "potcar_content_retained_in_generated_inputs": False,
        }
        provenance_path = input_dir / "input_provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        config_hash = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        job_id = f"dft_reference_{reference_id}"
        output_dir = root / "results" / reference_id / "attempt_1"
        command = [
            str(python_executable),
            str(runner_path),
            "--input-dir",
            str(input_dir),
            "--potcar-source",
            str(source_potcar),
            "--output-dir",
            str(output_dir),
            "--command-json",
            json.dumps(list(vasp_command), separators=(",", ":")),
        ]
        rows.append(
            {
                "job_id": job_id,
                "dataset": "dft_elemental_reference_verification",
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
                    {
                        "OPENBLAS_NUM_THREADS": "8",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "8",
                        "GOTO_NUM_THREADS": "8",
                    },
                    separators=(",", ":"),
                ),
                "reference_id": reference_id,
                "reference_name": str(record["reference_name"]),
                "element": str(record["element"]),
                "functional": str(record["functional"]),
                "Ueff_eV": _normalise_ueff(record.get("Ueff_eV")),
                "structure": str(record["structure"]),
                "magnetic_setup": str(record["magnetic_setup"]),
                "paw_label": str(record["paw_label"]),
                "kpoint_spacing_Ainv": spacing,
                "kpoint_basis": kpoint_basis,
                "mesh": mesh_text,
                "atom_count": len(structure),
                "source_dir": str(source),
                "structure_source_filename": structure_filename,
                "structure_source_path": str(source_structure),
                "structure_source_sha256": provenance["source_structure_sha256"],
                "potcar_sha256": provenance["POTCAR_sha256"],
                "frozen_protocol_sha256": protocol_sha256,
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
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--vasp-command-json", required=True)
    parser.add_argument("--reference-id", action="append", dest="reference_ids")
    args = parser.parse_args()
    reference_plan = json.loads(args.reference_plan.read_text(encoding="utf-8"))
    if not isinstance(reference_plan, list):
        raise ValueError("reference plan must be a JSON list")
    vasp_command = json.loads(args.vasp_command_json)
    if not isinstance(vasp_command, list):
        raise ValueError("VASP command JSON must be a list")
    frame = build_elemental_reference_bundle(
        reference_plan=reference_plan,
        frozen_protocol_path=args.frozen_protocol,
        work_root=args.work_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
        python_executable=args.python_executable,
        runner_path=args.runner_path,
        vasp_command=vasp_command,
        selected_reference_ids=args.reference_ids,
    )
    print(json.dumps({"jobs": len(frame), "manifest": str(args.manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
