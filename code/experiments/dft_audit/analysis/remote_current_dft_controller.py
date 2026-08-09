"""Server-side controller for CURRENT_SELF_CONSISTENT_PAW_PBE_U.

This file is uploaded to the licensed VASP server. POTCAR files are assembled
and consumed there and are never included in downloaded result bundles.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from compute_self_consistent_fe import compute_formation_energy
from recompute_self_consistent_fe_decimal import (
    recompute_formation_energy_decimal,
)


PROTOCOL = "CURRENT_SELF_CONSISTENT_PAW_PBE_U"
FIRST_CANDIDATE = (
    "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_3"
)
REMAINING = [
    "job_196_Cr_fe_-0.819_n4_generated_crystals_cif__gen_1",
    "job_234_Cr_fe_-1.123_n4_generated_crystals_cif__gen_3",
    "job_079_Cr_fe_-0.854_n4_generated_crystals_cif__gen_1",
    "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0",
]
POTCARS = {
    "Li": {
        "path": Path("/root/software/potpaw_PBE/Li_sv/POTCAR"),
        "sha256": "201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b",
    },
    "Cr": {
        "path": Path("/root/software/potpaw_PBE/Cr_pv/POTCAR"),
        "sha256": "836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b",
    },
    "O": {
        "path": Path("/root/software/potpaw_PBE/O/POTCAR"),
        "sha256": "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64",
    },
}
VASP = Path("/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std")
VASP_SHA256 = (
    "2abdcfedd1c3e7962a56404bd14cc340dcb170867720921fcea8ec7058ef3d94"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_server_assets() -> dict[str, Any]:
    failures: list[str] = []
    observed: dict[str, Any] = {}
    for element, expected in POTCARS.items():
        path = expected["path"]
        digest = sha256_file(path) if path.is_file() else None
        observed[element] = {"path": str(path), "sha256": digest}
        if digest != expected["sha256"]:
            failures.append(f"{element}:POTCAR_SHA256_DRIFT")
    binary_hash = sha256_file(VASP) if VASP.is_file() else None
    if binary_hash != VASP_SHA256:
        failures.append("VASP_BINARY_SHA256_DRIFT")
    return {
        "checked_at_utc": utc_now(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "potcars": observed,
        "vasp": {"path": str(VASP), "sha256": binary_hash},
    }


def assemble_potcar(poscar_path: Path, destination: Path) -> dict[str, Any]:
    symbols = Poscar.from_file(poscar_path).site_symbols
    records = []
    with destination.open("wb") as output:
        for symbol in symbols:
            source = POTCARS[symbol]["path"]
            observed = sha256_file(source)
            if observed != POTCARS[symbol]["sha256"]:
                raise RuntimeError(f"{symbol} POTCAR hash drift")
            with source.open("rb") as handle:
                shutil.copyfileobj(handle, output)
            records.append(
                {
                    "element": symbol,
                    "source_path": str(source),
                    "source_sha256": observed,
                }
            )
    return {
        "assembled_sha256": sha256_file(destination),
        "components": records,
    }


def parse_stage(attempt: Path, initial_structure: Structure) -> dict[str, Any]:
    vasprun = Vasprun(
        attempt / "vasprun.xml",
        parse_dos=False,
        parse_eigen=True,
        parse_projected_eigen=False,
        exception_on_bad_xml=True,
    )
    outcar = Outcar(attempt / "OUTCAR")
    final = vasprun.final_structure
    forces = vasprun.ionic_steps[-1].get("forces", [])
    fmax = max(
        (
            math.sqrt(sum(float(component) ** 2 for component in vector))
            for vector in forces
        ),
        default=float("nan"),
    )
    distance_matrix = final.distance_matrix
    dmin = min(
        float(distance_matrix[i, j])
        for i in range(len(final))
        for j in range(i + 1, len(final))
    )
    gap, cbm, vbm, direct = vasprun.eigenvalue_band_properties
    magnetization = outcar.magnetization or []
    local_moments = [
        float(row.get("tot", float("nan"))) for row in magnetization
    ]
    total_moment = (
        float(sum(value for value in local_moments if math.isfinite(value)))
        if local_moments
        else float("nan")
    )
    stress = vasprun.ionic_steps[-1].get("stress")
    if hasattr(stress, "tolist"):
        stress = stress.tolist()
    entropy_term = None
    if vasprun.ionic_steps:
        electronic = vasprun.ionic_steps[-1].get("electronic_steps", [])
        if electronic:
            last = electronic[-1]
            energy_without_entropy = last.get("e_wo_entrp")
            free_energy = last.get("e_fr_energy")
            if energy_without_entropy is not None and free_energy is not None:
                entropy_term = float(energy_without_entropy - free_energy)
    return {
        "parsed_at_utc": utc_now(),
        "energy_eV": float(vasprun.final_energy),
        "energy_per_atom_eV": float(vasprun.final_energy / len(final)),
        "atom_count": len(final),
        "electronic_converged": bool(vasprun.converged_electronic),
        "ionic_converged": bool(vasprun.converged_ionic),
        "fmax_eV_A": fmax,
        "stress_kbar": stress,
        "total_magnetic_moment_muB": total_moment,
        "local_magnetic_moments_muB": local_moments,
        "band_gap_eV": float(gap),
        "metallic": bool(float(gap) <= 1e-6),
        "direct_gap": bool(direct),
        "cbm_eV": float(cbm),
        "vbm_eV": float(vbm),
        "entropy_term_eV_cell": entropy_term,
        "minimum_interatomic_distance_A": dmin,
        "initial_volume_A3": float(initial_structure.volume),
        "final_volume_A3": float(final.volume),
        "volume_change_percent": float(
            100.0 * (final.volume / initial_structure.volume - 1.0)
        ),
        "structure_collapse": bool(
            dmin <= 1.2
            or final.volume / initial_structure.volume <= 0.5
            or final.volume / initial_structure.volume >= 2.0
        ),
        "final_formula": final.composition.formula,
    }


def run_vasp(
    attempt: Path,
    *,
    segment_seconds: int,
) -> dict[str, Any]:
    started = time.time()
    stdout_path = attempt / "vasp.stdout"
    stderr_path = attempt / "vasp.stderr"
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            [str(VASP)],
            cwd=attempt,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "16",
                "OPENBLAS_NUM_THREADS": "16",
                "MKL_NUM_THREADS": "16",
            },
        )
        try:
            return_code = process.wait(timeout=segment_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                return_code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                return_code = process.wait()
    return {
        "started_at_epoch": started,
        "ended_at_epoch": time.time(),
        "elapsed_seconds": time.time() - started,
        "return_code": return_code,
        "walltime_segment_expired": timed_out,
    }


def latest_valid_contcar(attempts: list[Path]) -> Path | None:
    for attempt in reversed(attempts):
        candidate = attempt / "CONTCAR"
        if not candidate.is_file() or candidate.stat().st_size == 0:
            continue
        try:
            Structure.from_file(candidate)
        except Exception:
            continue
        return candidate
    return None


def run_stage(
    root: Path,
    job_id: str,
    *,
    initial_poscar: Path,
    segment_seconds: int,
    max_attempts: int,
) -> dict[str, Any]:
    input_dir = root / "input_package" / "inputs" / job_id
    stage_root = root / "runs" / job_id
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_name = job_id.rsplit("__", 1)[-1]
    is_relax = stage_name == "relax" or job_id.endswith("_relax")
    original = Structure.from_file(initial_poscar)
    attempts: list[Path] = sorted(stage_root.glob("attempt_*"))
    summary_path = stage_root / "stage_summary.json"
    if summary_path.is_file():
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        if current.get("status") == "PASS":
            return current

    for attempt_number in range(len(attempts) + 1, max_attempts + 1):
        attempt = stage_root / f"attempt_{attempt_number:04d}"
        attempt.mkdir(parents=True, exist_ok=False)
        source_poscar = initial_poscar
        if is_relax and attempts:
            continuation = latest_valid_contcar(attempts)
            if continuation is None:
                result = {
                    "job_id": job_id,
                    "status": "FAIL",
                    "failure_reason": "NO_VALID_CONTCAR_FOR_CONTINUATION",
                    "attempt_count": len(attempts),
                }
                atomic_json(summary_path, result)
                return result
            source_poscar = continuation
        shutil.copy2(source_poscar, attempt / "POSCAR")
        shutil.copy2(input_dir / "INCAR", attempt / "INCAR")
        shutil.copy2(input_dir / "KPOINTS", attempt / "KPOINTS")
        potcar_record = assemble_potcar(attempt / "POSCAR", attempt / "POTCAR")
        input_record = {
            "protocol": PROTOCOL,
            "job_id": job_id,
            "attempt": attempt_number,
            "created_at_utc": utc_now(),
            "poscar_source": str(source_poscar),
            "continuation": source_poscar != initial_poscar,
            "sha256": {
                filename: sha256_file(attempt / filename)
                for filename in ("INCAR", "KPOINTS", "POSCAR", "POTCAR")
            },
            "potcar": potcar_record,
        }
        atomic_json(attempt / "input_hashes.json", input_record)
        runtime = run_vasp(attempt, segment_seconds=segment_seconds)
        atomic_json(attempt / "runtime.json", runtime)
        parsed: dict[str, Any] | None = None
        parse_error = ""
        try:
            parsed = parse_stage(attempt, original)
            atomic_json(attempt / "parsed.json", parsed)
            Structure.from_file(attempt / "CONTCAR").to(
                filename=attempt / "final.cif", fmt="cif"
            )
        except Exception:
            parse_error = traceback.format_exc()
            (attempt / "parse_error.txt").write_text(
                parse_error, encoding="utf-8"
            )
        attempts.append(attempt)

        stage_pass = bool(
            parsed
            and runtime["return_code"] == 0
            and parsed["electronic_converged"]
            and not parsed["structure_collapse"]
            and (
                not is_relax
                or (
                    parsed["ionic_converged"]
                    and parsed["fmax_eV_A"] <= 0.05
                )
            )
        )
        if stage_pass:
            result = {
                "job_id": job_id,
                "status": "PASS",
                "attempt_count": len(attempts),
                "final_attempt": str(attempt),
                "final_contcar": str(attempt / "CONTCAR"),
                "parsed": parsed,
                "runtime": runtime,
            }
            atomic_json(summary_path, result)
            return result

        if not runtime["walltime_segment_expired"]:
            reason = (
                "VASP_NONZERO_RETURN"
                if runtime["return_code"] != 0
                else "PERSISTENT_NUMERICAL_OR_CONVERGENCE_FAILURE"
            )
            result = {
                "job_id": job_id,
                "status": "FAIL",
                "failure_reason": reason,
                "attempt_count": len(attempts),
                "last_attempt": str(attempt),
                "last_parsed": parsed,
                "parse_error": parse_error,
            }
            atomic_json(summary_path, result)
            return result
        if not is_relax:
            # Static calculations have no newer geometry, but a wall-time stop
            # is not a scientific failure. Retry the same fixed structure.
            pass

    result = {
        "job_id": job_id,
        "status": "FAIL",
        "failure_reason": "MAX_WALLTIME_SEGMENTS_EXHAUSTED",
        "attempt_count": len(attempts),
    }
    atomic_json(summary_path, result)
    return result


def run_entity(
    root: Path,
    entity: str,
    *,
    segment_seconds: int,
    include_dense: bool,
) -> dict[str, Any]:
    if entity.startswith("reference_"):
        element = entity.split("_")[1]
        relax_id = f"reference_{element}_relax"
        static_id = f"reference_{element}_static_0p15"
    else:
        relax_id = f"{entity}__relax"
        static_id = f"{entity}__static_0p15"
    relax_input = root / "input_package" / "inputs" / relax_id / "POSCAR"
    relax = run_stage(
        root,
        relax_id,
        initial_poscar=relax_input,
        segment_seconds=segment_seconds,
        max_attempts=20,
    )
    if relax["status"] != "PASS":
        return {"entity": entity, "status": "FAIL", "relax": relax}
    relaxed_geometry = Path(relax["final_contcar"])
    static = run_stage(
        root,
        static_id,
        initial_poscar=relaxed_geometry,
        segment_seconds=segment_seconds,
        max_attempts=5,
    )
    result: dict[str, Any] = {
        "entity": entity,
        "status": static["status"],
        "relax": relax,
        "static_0p15": static,
    }
    if static["status"] != "PASS":
        return result
    if include_dense:
        dense_id = f"{entity}__static_0p10"
        dense = run_stage(
            root,
            dense_id,
            initial_poscar=relaxed_geometry,
            segment_seconds=segment_seconds,
            max_attempts=5,
        )
        result["static_0p10"] = dense
        if dense["status"] != "PASS":
            result["status"] = "FAIL"
        else:
            difference = abs(
                static["parsed"]["energy_per_atom_eV"]
                - dense["parsed"]["energy_per_atom_eV"]
            )
            result["kpoint_energy_difference_eV_atom"] = difference
            result["kpoint_converged_2meV_atom"] = difference <= 0.002
            if difference > 0.002:
                result["status"] = "FAIL"
                result["failure_reason"] = "KPOINT_CONVERGENCE_GT_2MEV_ATOM"
    return result


def formation_rows(
    entities: dict[str, dict[str, Any]],
    candidate_ids: list[str],
) -> list[dict[str, Any]]:
    li = entities["reference_Li"]["static_0p15"]["parsed"]
    cr = entities["reference_Cr"]["static_0p15"]["parsed"]
    oxygen = entities["reference_O"]["static_0p15"]["parsed"]
    li_atoms = li["atom_count"]
    cr_atoms = cr["atom_count"]
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        if entities.get(candidate_id, {}).get("status") != "PASS":
            continue
        candidate = entities[candidate_id]["static_0p15"]["parsed"]
        row = {
            "candidate_id": candidate_id,
            "candidate_total_energy_eV": candidate["energy_eV"],
            "Li_energy_per_atom_eV": li["energy_eV"] / li_atoms,
            "Cr_energy_per_atom_eV": cr["energy_eV"] / cr_atoms,
            "O2_energy_per_molecule_eV": oxygen["energy_eV"],
        }
        primary = compute_formation_energy(row)
        independent = recompute_formation_energy_decimal(row)
        row.update(
            {
                "formation_energy_primary_eV_atom": primary,
                "formation_energy_decimal_eV_atom": independent,
                "absolute_difference_eV_atom": abs(primary - independent),
                "roundtrip_pass": abs(primary - independent) <= 1e-6,
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--segment-seconds", type=int, default=21000)
    parser.add_argument("--phase2-workers", type=int, default=2)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    state_path = root / "controller_state.json"
    state: dict[str, Any] = {
        "protocol": PROTOCOL,
        "started_at_utc": utc_now(),
        "status": "STARTING",
        "phase": "asset_audit",
        "entities": {},
    }
    atomic_json(state_path, state)

    asset_audit = validate_server_assets()
    state["asset_audit"] = asset_audit
    if asset_audit["status"] != "PASS":
        state.update({"status": "STOPPED", "failure_reason": "ASSET_DRIFT"})
        atomic_json(state_path, state)
        return 2

    state.update({"status": "RUNNING", "phase": "reference_prerequisites"})
    atomic_json(state_path, state)
    for reference in ("reference_Li", "reference_Cr", "reference_O"):
        state["current_entity"] = reference
        atomic_json(state_path, state)
        result = run_entity(
            root,
            reference,
            segment_seconds=args.segment_seconds,
            include_dense=False,
        )
        state["entities"][reference] = result
        atomic_json(state_path, state)
        if result["status"] != "PASS":
            state.update(
                {
                    "status": "STOPPED",
                    "phase": "reference_prerequisites",
                    "failure_reason": f"{reference}_FAILED",
                }
            )
            atomic_json(state_path, state)
            return 3

    state.update({"phase": "job_092_gate", "current_entity": FIRST_CANDIDATE})
    atomic_json(state_path, state)
    first = run_entity(
        root,
        FIRST_CANDIDATE,
        segment_seconds=args.segment_seconds,
        include_dense=True,
    )
    state["entities"][FIRST_CANDIDATE] = first
    atomic_json(state_path, state)
    if first["status"] != "PASS":
        state.update(
            {
                "status": "STOPPED",
                "failure_reason": "JOB_092_GATE_FAILED",
            }
        )
        atomic_json(state_path, state)
        return 4

    first_fe = formation_rows(state["entities"], [FIRST_CANDIDATE])
    write_csv(root / "results" / "formation_energy_checks.csv", first_fe)
    if not first_fe or not first_fe[0]["roundtrip_pass"]:
        state.update(
            {
                "status": "STOPPED",
                "failure_reason": "JOB_092_FORMATION_ROUNDTRIP_FAILED",
            }
        )
        atomic_json(state_path, state)
        return 5

    state.update(
        {
            "phase": "remaining_four",
            "job_092_gate": "PASS",
            "current_entity": None,
        }
    )
    atomic_json(state_path, state)
    with ThreadPoolExecutor(max_workers=args.phase2_workers) as executor:
        futures = {
            executor.submit(
                run_entity,
                root,
                candidate,
                segment_seconds=args.segment_seconds,
                include_dense=False,
            ): candidate
            for candidate in REMAINING
        }
        for future in as_completed(futures):
            candidate = futures[future]
            try:
                result = future.result()
            except Exception:
                result = {
                    "entity": candidate,
                    "status": "FAIL",
                    "failure_reason": "CONTROLLER_EXCEPTION",
                    "traceback": traceback.format_exc(),
                }
            state["entities"][candidate] = result
            atomic_json(state_path, state)

    all_candidates = [FIRST_CANDIDATE, *REMAINING]
    checks = formation_rows(state["entities"], all_candidates)
    write_csv(root / "results" / "formation_energy_checks.csv", checks)
    state.update(
        {
            "status": (
                "COMPLETE"
                if all(
                    state["entities"].get(candidate, {}).get("status") == "PASS"
                    for candidate in all_candidates
                )
                else "COMPLETE_WITH_FAILURES"
            ),
            "phase": "complete",
            "completed_at_utc": utc_now(),
        }
    )
    atomic_json(state_path, state)
    return 0 if state["status"] == "COMPLETE" else 6


if __name__ == "__main__":
    raise SystemExit(main())
