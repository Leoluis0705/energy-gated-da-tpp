#!/usr/bin/env python3
"""Quantify and gate the approved C120/C214 verification relaxations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher


FORCE_THRESHOLD_EV_A = 0.05
SYMMETRY_TOLERANCE_A = 0.1
MATCHER_SETTINGS = {
    "ltol": 0.2,
    "stol": 0.3,
    "angle_tol": 5.0,
    "primitive_cell": False,
    "scale": True,
    "attempt_supercell": False,
    "allow_subset": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return float(values[-1]) if values else None


def _last_fmax(text: str) -> float | None:
    blocks = re.findall(
        r"TOTAL-FORCE\s*\(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-{5,})",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    norms: list[float] = []
    for line in blocks[-1].splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            force = np.asarray([float(value) for value in fields[-3:]], dtype=float)
        except ValueError:
            continue
        norms.append(float(np.linalg.norm(force)))
    return max(norms) if norms else None


def _minimum_distances(structure: Structure) -> tuple[float, float | None]:
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    matrix[np.eye(len(structure), dtype=bool)] = np.inf
    shortest = float(matrix.min())
    metals = [i for i, site in enumerate(structure) if site.specie.symbol in {"Cr", "Mn", "Mg"}]
    oxygens = [i for i, site in enumerate(structure) if site.specie.symbol == "O"]
    metal_oxygen = min((matrix[i, j] for i in metals for j in oxygens), default=None)
    return shortest, None if metal_oxygen is None else float(metal_oxygen)


def _space_group(structure: Structure) -> str:
    analyzer = SpacegroupAnalyzer(structure, symprec=SYMMETRY_TOLERANCE_A)
    return f"{analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})"


def _maximum_internal_displacement(initial: Structure, final: Structure) -> float:
    if len(initial) != len(final):
        raise ValueError("initial and final structures have different atom counts")
    if [site.specie.symbol for site in initial] != [site.specie.symbol for site in final]:
        raise ValueError("VASP site order/species changed unexpectedly")
    delta = np.asarray(final.frac_coords - initial.frac_coords, dtype=float)
    delta -= np.rint(delta)
    cartesian = delta @ np.asarray(final.lattice.matrix, dtype=float)
    return float(np.linalg.norm(cartesian, axis=1).max())


def _lattice_fields(prefix: str, structure: Structure) -> dict[str, float]:
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    return {
        f"{prefix}_a_A": float(a),
        f"{prefix}_b_A": float(b),
        f"{prefix}_c_A": float(c),
        f"{prefix}_alpha_deg": float(alpha),
        f"{prefix}_beta_deg": float(beta),
        f"{prefix}_gamma_deg": float(gamma),
    }


def _analyze_one(input_row: pd.Series, queue_row: pd.Series, cif_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    job_id = str(input_row["job_id"])
    source = Path(str(input_row["input_dir"])).resolve()
    output = Path(str(queue_row["output_path"])).resolve()
    required = [
        source / "initial.POSCAR",
        source / "initial.cif",
        output / "INCAR",
        output / "KPOINTS",
        output / "POSCAR",
        output / "CONTCAR",
        output / "OUTCAR",
        output / "OSZICAR",
        output / "task_result.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing verification evidence: " + "; ".join(missing))

    result = json.loads((output / "task_result.json").read_text(encoding="utf-8"))
    text = (output / "OUTCAR").read_text(encoding="utf-8", errors="ignore")
    oszicar = (output / "OSZICAR").read_text(encoding="utf-8", errors="ignore")
    initial = Structure.from_file(source / "initial.POSCAR")
    final = Structure.from_file(output / "CONTCAR")
    matcher = StructureMatcher(**MATCHER_SETTINGS)
    fmax = _last_fmax(text)
    completed = (
        str(queue_row["status"]) == "DONE"
        and int(queue_row["exit_code"]) == 0
        and result.get("status") == "DONE"
        and int(result.get("exit_code", 1)) == 0
        and result.get("timing_footer_present") is True
        and result.get("potcar_retained") is False
        and not (output / "POTCAR").exists()
    )
    electronic = completed and result.get("electronic_converged") is True
    ionic_marker = "reached required accuracy - stopping structural energy minimisation" in text
    ionic = bool(electronic and ionic_marker and fmax is not None and fmax <= FORCE_THRESHOLD_EV_A + 1e-9)
    shortest, shortest_mo = _minimum_distances(final)
    total_energy = _last_float(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)", text)
    magnetic_moment = _last_float(r"\bmag=\s*([-+0-9.Ee]+)", oszicar)
    if magnetic_moment is None:
        magnetic_moment = _last_float(
            r"number of electron\s+[-+0-9.Ee]+\s+magnetization\s+([-+0-9.Ee]+)", text
        )
    version = re.search(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", text[:20_000], flags=re.IGNORECASE)
    final_cif = cif_root / f"{job_id}.cif"
    CifWriter(final, symprec=SYMMETRY_TOLERANCE_A).write_file(final_cif)
    structural_match = bool(matcher.fit(initial, final))
    metric = {
        "job_id": job_id,
        "candidate_id": str(input_row["candidate_id"]),
        "magnetic_initialization": str(input_row["magnetic_initialization"]),
        **_lattice_fields("initial", initial),
        **_lattice_fields("final", final),
        "initial_volume_A3": float(initial.volume),
        "final_volume_A3": float(final.volume),
        "relative_volume_change_percent": float(100.0 * (final.volume - initial.volume) / initial.volume),
        "maximum_internal_displacement_A": _maximum_internal_displacement(initial, final),
        "minimum_interatomic_distance_A": shortest,
        "minimum_M_O_distance_A": shortest_mo,
        "Fmax_eV_A": fmax,
        "initial_space_group": _space_group(initial),
        "final_space_group": _space_group(final),
        "space_group_changed": _space_group(initial) != _space_group(final),
        "structural_match_initial_final": structural_match,
        "final_total_energy_eV_relaxation": total_energy,
        "final_total_magnetic_moment": magnetic_moment,
        "symmetry_tolerance_A": SYMMETRY_TOLERANCE_A,
        "source_input_path": str(source),
        "source_output_path": str(output),
        "initial_poscar_sha256": _sha256(source / "initial.POSCAR"),
        "contcar_sha256": _sha256(output / "CONTCAR"),
        "outcar_sha256": _sha256(output / "OUTCAR"),
        "oszicar_sha256": _sha256(output / "OSZICAR"),
        "vasprun_sha256": _sha256(output / "vasprun.xml") if (output / "vasprun.xml").is_file() else "",
        "initial_cif_path": str(source / "initial.cif"),
        "final_cif_path": str(final_cif),
        "final_cif_sha256": _sha256(final_cif),
        "displacement_definition": "minimum-image fractional displacement mapped through final lattice; homogeneous cell strain removed",
    }
    convergence = {
        "job_id": job_id,
        "candidate_id": str(input_row["candidate_id"]),
        "magnetic_initialization": str(input_row["magnetic_initialization"]),
        "queue_status": str(queue_row["status"]),
        "queue_exit_code": int(queue_row["exit_code"]),
        "vasp_version": version.group(1) if version else "",
        "vasp_completed": completed,
        "electronic_converged": bool(electronic),
        "ionic_marker_present": ionic_marker,
        "ionic_converged": ionic,
        "Fmax_eV_A": fmax,
        "force_threshold_eV_A": FORCE_THRESHOLD_EV_A,
        "timing_footer_present": result.get("timing_footer_present") is True,
        "potcar_retained": (output / "POTCAR").exists(),
        "failure_reason": "" if completed and electronic and ionic else "completion_or_convergence_gate_failed",
        "source_output_path": str(output),
    }
    return metric, convergence


def analyze_verification_relaxations(
    input_manifest: pd.DataFrame,
    queue_manifest: pd.DataFrame,
    *,
    cif_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, dict[str, Any]]]:
    """Return metrics, convergence, an automatic static gate, and dependency records."""

    expected = {
        f"dft_{candidate.lower()}_{state}_verification_relax"
        for candidate in ("C120", "C214")
        for state in ("state_fm", "state_afm")
    }
    if set(input_manifest["job_id"].astype(str)) != expected:
        raise ValueError("input manifest is not the four approved verification relaxations")
    if set(queue_manifest["job_id"].astype(str)) != expected:
        raise ValueError("queue manifest does not match approved verification relaxations")
    if input_manifest["job_id"].duplicated().any() or queue_manifest["job_id"].duplicated().any():
        raise ValueError("verification manifests contain duplicate jobs")
    cif_root = Path(cif_root)
    cif_root.mkdir(parents=True, exist_ok=False)
    queue = queue_manifest.set_index("job_id", drop=False)
    metrics: list[dict[str, Any]] = []
    convergence: list[dict[str, Any]] = []
    pause_reasons: list[str] = []
    structures: dict[str, Structure] = {}
    dependencies: dict[str, dict[str, Any]] = {}
    for _, input_row in input_manifest.sort_values("job_id").iterrows():
        job_id = str(input_row["job_id"])
        metric, status = _analyze_one(input_row, queue.loc[job_id], cif_root)
        metrics.append(metric)
        convergence.append(status)
        structures[job_id] = Structure.from_file(Path(metric["source_output_path"]) / "CONTCAR")
        dependencies[job_id] = {
            "status": "DONE" if status["vasp_completed"] else "FAILED",
            "exit_code": status["queue_exit_code"],
            "electronic_converged": status["electronic_converged"],
            "ionic_converged": status["ionic_converged"],
            "output_path": metric["source_output_path"],
            "contcar_sha256": metric["contcar_sha256"],
        }
        if not status["vasp_completed"]:
            pause_reasons.append(f"{job_id}: VASP completion gate failed")
        if not status["electronic_converged"]:
            pause_reasons.append(f"{job_id}: electronic convergence gate failed")
        if not status["ionic_converged"]:
            pause_reasons.append(f"{job_id}: ionic/Fmax gate failed")
        if not metric["structural_match_initial_final"]:
            pause_reasons.append(f"{job_id}: initial/final structures do not match")

    matcher = StructureMatcher(**MATCHER_SETTINGS)
    branch_comparisons: list[dict[str, Any]] = []
    for candidate in ("C120", "C214"):
        fm = f"dft_{candidate.lower()}_state_fm_verification_relax"
        afm = f"dft_{candidate.lower()}_state_afm_verification_relax"
        match = bool(matcher.fit(structures[fm], structures[afm]))
        branch_comparisons.append(
            {
                "candidate_id": candidate,
                "initialization_scope": "two tested magnetic initializations",
                "final_structures_match": match,
            }
        )
        if not match:
            pause_reasons.append(f"{candidate}: distinct final structural branches between two tested magnetic initializations")

    metrics_frame = pd.DataFrame(metrics).sort_values(["candidate_id", "magnetic_initialization"]).reset_index(drop=True)
    convergence_frame = pd.DataFrame(convergence).sort_values(["candidate_id", "magnetic_initialization"]).reset_index(drop=True)
    unique_reasons = list(dict.fromkeys(pause_reasons))
    review = {
        "review_generated_from_outputs": True,
        "static_launch_authorized": not unique_reasons,
        "pause_reasons": unique_reasons,
        "force_threshold_eV_A": FORCE_THRESHOLD_EV_A,
        "symmetry_tolerance_A": SYMMETRY_TOLERANCE_A,
        "structure_matcher": MATCHER_SETTINGS,
        "branch_comparisons": branch_comparisons,
        "formation_energy_shift_gate_eV_per_atom": 0.02,
        "formation_energy_gate_status": "pending_frozen_static_outputs",
    }
    return metrics_frame, convergence_frame, review, dependencies


def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json_exclusive(payload: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, required=True)
    parser.add_argument("--cif-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--convergence", type=Path, required=True)
    parser.add_argument("--structural-review", type=Path, required=True)
    parser.add_argument("--dependency-results", type=Path, required=True)
    args = parser.parse_args()
    inputs = pd.read_csv(args.input_manifest, dtype=str, keep_default_na=False)
    queue = pd.read_csv(args.queue_manifest, dtype=str, keep_default_na=False)
    metrics, convergence, review, dependencies = analyze_verification_relaxations(
        inputs,
        queue,
        cif_root=args.cif_root,
    )
    _write_csv_exclusive(metrics, args.metrics)
    _write_csv_exclusive(convergence, args.convergence)
    _write_json_exclusive(review, args.structural_review)
    _write_json_exclusive(dependencies, args.dependency_results)
    print(json.dumps({"jobs": len(metrics), "static_launch_authorized": review["static_launch_authorized"], "pause_reasons": review["pause_reasons"]}, indent=2))
    return 0 if review["static_launch_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
