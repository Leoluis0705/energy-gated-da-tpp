#!/usr/bin/env python3
"""Recompute C120/C214 verification energies after gated frozen statics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Incar, Kpoints, Poscar

from analysis.postprocess_dft_verification_relaxations import (
    FORCE_THRESHOLD_EV_A,
    SYMMETRY_TOLERANCE_A,
    _last_float,
    _last_fmax,
    _minimum_distances,
    _sha256,
    _space_group,
)


FORMATION_SHIFT_THRESHOLD_EV_ATOM = 0.02


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _candidate_alias(value: str) -> str:
    match = re.search(r"(?:^|_)job_(120|214)(?:_|$)", str(value))
    if match:
        return f"C{match.group(1)}"
    if str(value) in {"C120", "C214"}:
        return str(value)
    raise ValueError(f"cannot map candidate identifier: {value}")


def _candidate_alias_or_none(value: str) -> str | None:
    try:
        return _candidate_alias(value)
    except ValueError:
        return None


def _reference_energies(references: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    expected = {"Li": "GGA_U_Li_metal", "Cr": "GGA_U_Cr_metal", "O": "GGA_U_O2_molecule"}
    energies: dict[str, float] = {}
    hashes: dict[str, str] = {}
    indexed = references.set_index("reference_id", drop=False)
    for element, reference_id in expected.items():
        if reference_id not in indexed.index:
            raise ValueError(f"missing frozen elemental reference: {reference_id}")
        row = indexed.loc[reference_id]
        if not _truthy(row["electronic_converged"]) or not _truthy(row["timing_footer_present"]):
            raise ValueError(f"elemental reference did not converge: {reference_id}")
        if str(row["element"]) != element:
            raise ValueError(f"element mismatch in reference {reference_id}")
        energies[element] = float(row["energy_per_atom_eV"])
        hashes[element] = str(row.get("raw_output_sha256", ""))
    return energies, hashes


def _mesh(path: Path) -> str:
    points = Kpoints.from_file(path)
    if not points.kpts:
        raise ValueError(f"missing explicit mesh: {path}")
    return "x".join(str(int(round(float(value)))) for value in points.kpts[0])


def _parse_static(input_row: pd.Series, queue_row: pd.Series, cif_root: Path) -> dict[str, Any]:
    job_id = str(input_row["job_id"])
    output = Path(str(queue_row["output_path"])).resolve()
    required = [output / name for name in ("INCAR", "KPOINTS", "POSCAR", "OUTCAR", "OSZICAR", "task_result.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen-static evidence: " + "; ".join(missing))
    result = json.loads((output / "task_result.json").read_text(encoding="utf-8"))
    outcar_text = (output / "OUTCAR").read_text(encoding="utf-8", errors="ignore")
    oszicar_text = (output / "OSZICAR").read_text(encoding="utf-8", errors="ignore")
    incar = Incar.from_file(output / "INCAR")
    poscar = Poscar.from_file(output / "POSCAR")
    final_path = output / "CONTCAR" if (output / "CONTCAR").is_file() and (output / "CONTCAR").stat().st_size else output / "POSCAR"
    final = Structure.from_file(final_path)
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
    energy = _last_float(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)", outcar_text)
    if energy is None:
        raise ValueError(f"missing final TOTEN: {job_id}")
    fmax = _last_fmax(outcar_text)
    magnetic = _last_float(r"\bmag=\s*([-+0-9.Ee]+)", oszicar_text)
    if magnetic is None:
        magnetic = _last_float(
            r"number of electron\s+[-+0-9.Ee]+\s+magnetization\s+([-+0-9.Ee]+)", outcar_text
        )
    shortest, shortest_mo = _minimum_distances(final)
    version = re.search(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", outcar_text[:20_000], flags=re.IGNORECASE)
    paw_labels = list(dict.fromkeys(match.strip() for match in re.findall(r"TITEL\s*=\s*(.+)", outcar_text)))
    cif = cif_root / f"{job_id}.cif"
    CifWriter(final, symprec=SYMMETRY_TOLERANCE_A).write_file(cif)
    return {
        "job_id": job_id,
        "candidate_id": str(input_row["candidate_id"]),
        "magnetic_initialization": str(input_row["magnetic_initialization"]),
        "initialization_scope": "two tested magnetic initializations",
        "dependency_job_id": str(input_row["dependency_job_id"]),
        "vasp_version": version.group(1) if version else "",
        "vasp_completed": bool(completed),
        "electronic_converged": bool(electronic),
        "final_total_energy_eV": float(energy),
        "Fmax_eV_A_static_diagnostic": fmax,
        "final_total_magnetic_moment": magnetic,
        "static_input_volume_A3": float(poscar.structure.volume),
        "static_final_volume_A3": float(final.volume),
        "static_volume_change_percent": float(100 * (final.volume - poscar.structure.volume) / poscar.structure.volume),
        "minimum_interatomic_distance_A": shortest,
        "minimum_M_O_distance_A": shortest_mo,
        "final_space_group": _space_group(final),
        "symmetry_tolerance_A": SYMMETRY_TOLERANCE_A,
        "ENCUT_eV": float(incar["ENCUT"]),
        "EDIFF": float(incar["EDIFF"]),
        "NSW": int(incar.get("NSW", 0)),
        "IBRION": int(incar.get("IBRION", -1)),
        "ISMEAR": int(incar["ISMEAR"]),
        "SIGMA": float(incar["SIGMA"]),
        "LDAUU": " ".join(str(value) for value in incar.get("LDAUU", [])),
        "kpoints_mesh": _mesh(output / "KPOINTS"),
        "PAW_labels": " | ".join(paw_labels),
        "element_order": " ".join(poscar.site_symbols),
        "source_output_path": str(output),
        "outcar_sha256": _sha256(output / "OUTCAR"),
        "oszicar_sha256": _sha256(output / "OSZICAR"),
        "vasprun_sha256": _sha256(output / "vasprun.xml") if (output / "vasprun.xml").is_file() else "",
        "final_cif_path": str(cif),
        "final_cif_sha256": _sha256(cif),
        "potcar_retained": (output / "POTCAR").exists(),
    }


def analyze_verification_statics(
    static_input_manifest: pd.DataFrame,
    static_queue_manifest: pd.DataFrame,
    relaxation_metrics: pd.DataFrame,
    historical_magnetic: pd.DataFrame,
    elemental_references: pd.DataFrame,
    *,
    cif_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    expected = {
        f"dft_{candidate.lower()}_{state}_verification_static"
        for candidate in ("C120", "C214")
        for state in ("state_fm", "state_afm")
    }
    if set(static_input_manifest["job_id"].astype(str)) != expected:
        raise ValueError("static input manifest is not the four approved jobs")
    if set(static_queue_manifest["job_id"].astype(str)) != expected:
        raise ValueError("static queue manifest is not the four approved jobs")
    cif_root = Path(cif_root)
    cif_root.mkdir(parents=True, exist_ok=False)
    queue = static_queue_manifest.set_index("job_id", drop=False)
    records = [
        _parse_static(row, queue.loc[str(row["job_id"])], cif_root)
        for _, row in static_input_manifest.sort_values("job_id").iterrows()
    ]
    static = pd.DataFrame(records).sort_values(["candidate_id", "magnetic_initialization"]).reset_index(drop=True)
    relax = relaxation_metrics.copy()
    relax["candidate_id"] = relax["candidate_id"].map(_candidate_alias)
    relax = relax.set_index(["candidate_id", "magnetic_initialization"])
    for index, row in static.iterrows():
        key = (row["candidate_id"], row["magnetic_initialization"])
        if key not in relax.index:
            raise ValueError(f"missing matching relaxation metric: {key}")
        for column in (
            "final_total_energy_eV_relaxation",
            "final_volume_A3",
            "relative_volume_change_percent",
            "maximum_internal_displacement_A",
            "Fmax_eV_A",
            "final_space_group",
        ):
            static.loc[index, f"verification_relaxation_{column}"] = relax.loc[key, column]

    historical = historical_magnetic.copy()
    historical = historical[historical["functional"].astype(str).eq("GGA+U")]
    historical["candidate_alias"] = historical["candidate_id"].map(
        _candidate_alias_or_none
    )
    historical = historical[historical["candidate_alias"].notna()].copy()
    if historical.duplicated(["candidate_alias", "magnetic_initialization"]).any():
        raise ValueError("historical magnetic evidence contains duplicate states")
    historical = historical.set_index(["candidate_alias", "magnetic_initialization"])
    references, reference_hashes = _reference_energies(elemental_references)
    reference_sum = references["Li"] + 2 * references["Cr"] + 4 * references["O"]
    formation_rows: list[dict[str, Any]] = []
    for row in static.itertuples(index=False):
        key = (row.candidate_id, row.magnetic_initialization)
        if key not in historical.index:
            raise ValueError(f"missing historical frozen-static state: {key}")
        old = historical.loc[key]
        old_energy = float(old["final_total_energy_eV"])
        new_energy = float(row.final_total_energy_eV)
        old_formation = (old_energy - reference_sum) / 7.0
        new_formation = (new_energy - reference_sum) / 7.0
        formation_rows.append(
            {
                "candidate_id": row.candidate_id,
                "formula": "LiCr2O4",
                "functional": "GGA+U",
                "magnetic_initialization": row.magnetic_initialization,
                "initialization_scope": "two tested magnetic initializations",
                "historical_total_energy_eV": old_energy,
                "verification_total_energy_eV": new_energy,
                "new_minus_historical_total_energy_eV": new_energy - old_energy,
                "historical_formation_energy_eV_per_atom": old_formation,
                "new_formation_energy_eV_per_atom": new_formation,
                "new_minus_historical_formation_energy_eV_per_atom": new_formation - old_formation,
                "reference_sum_eV_per_formula_unit": reference_sum,
                "elemental_reference_ids_json": json.dumps(
                    {"Li": "GGA_U_Li_metal", "Cr": "GGA_U_Cr_metal", "O": "GGA_U_O2_molecule"},
                    sort_keys=True,
                ),
                "elemental_reference_hashes_json": json.dumps(reference_hashes, sort_keys=True),
                "historical_source_output_path": str(old["source_output_path"]),
                "historical_outcar_sha256": str(old["outcar_sha256"]),
                "verification_source_output_path": row.source_output_path,
                "verification_outcar_sha256": row.outcar_sha256,
                "historical_selected_lower_energy_among_two": _truthy(old["selected_lower_energy_among_two_tested"]),
            }
        )
    formation = pd.DataFrame(formation_rows).sort_values(["candidate_id", "magnetic_initialization"]).reset_index(drop=True)

    magnetic_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    pause_reasons: list[str] = []
    for row in static.itertuples(index=False):
        if not row.vasp_completed or not row.electronic_converged:
            pause_reasons.append(f"{row.job_id}: frozen static did not complete or converge")
        if row.Fmax_eV_A_static_diagnostic is None or float(row.Fmax_eV_A_static_diagnostic) > FORCE_THRESHOLD_EV_A + 1e-9:
            pause_reasons.append(f"{row.job_id}: frozen-static diagnostic Fmax exceeds 0.05 eV/A")
    for candidate in ("C120", "C214"):
        candidate_static = static[static["candidate_id"].eq(candidate)].copy()
        lower_index = candidate_static["final_total_energy_eV"].astype(float).idxmin()
        lower_state = str(static.loc[lower_index, "magnetic_initialization"])
        lower_energy = float(static.loc[lower_index, "final_total_energy_eV"])
        for row in candidate_static.itertuples(index=False):
            magnetic_rows.append(
                {
                    "candidate_id": candidate,
                    "formula": "LiCr2O4",
                    "magnetic_initialization": row.magnetic_initialization,
                    "scope_statement": "two tested magnetic initializations",
                    "final_total_energy_eV": row.final_total_energy_eV,
                    "final_total_magnetic_moment": row.final_total_magnetic_moment,
                    "selected_lower_energy_among_two_tested": row.magnetic_initialization == lower_state,
                    "energy_difference_from_lower_eV": float(row.final_total_energy_eV) - lower_energy,
                    "source_output_path": row.source_output_path,
                    "outcar_sha256": row.outcar_sha256,
                }
            )
        old_selected = formation[
            formation["candidate_id"].eq(candidate)
            & formation["historical_selected_lower_energy_among_two"].astype(bool)
        ]
        if len(old_selected) != 1:
            raise ValueError(f"historical selected state is not unique for {candidate}")
        new_selected = formation[
            formation["candidate_id"].eq(candidate)
            & formation["magnetic_initialization"].eq(lower_state)
        ].iloc[0]
        old_row = old_selected.iloc[0]
        shift = float(new_selected["new_formation_energy_eV_per_atom"]) - float(
            old_row["historical_formation_energy_eV_per_atom"]
        )
        selected_rows.append(
            {
                "candidate_id": candidate,
                "selection_scope": "lower-energy configuration among the two tested initializations",
                "historical_selected_initialization": old_row["magnetic_initialization"],
                "new_selected_initialization": lower_state,
                "selected_initialization_changed": old_row["magnetic_initialization"] != lower_state,
                "historical_selected_total_energy_eV": old_row["historical_total_energy_eV"],
                "new_selected_total_energy_eV": new_selected["verification_total_energy_eV"],
                "historical_selected_formation_energy_eV_per_atom": old_row["historical_formation_energy_eV_per_atom"],
                "new_selected_formation_energy_eV_per_atom": new_selected["new_formation_energy_eV_per_atom"],
                "selected_formation_energy_shift_eV_per_atom": shift,
                "absolute_shift_threshold_eV_per_atom": FORMATION_SHIFT_THRESHOLD_EV_ATOM,
            }
        )
        if abs(shift) > FORMATION_SHIFT_THRESHOLD_EV_ATOM:
            pause_reasons.append(
                f"{candidate}: selected formation-energy shift {shift:.6f} eV/atom exceeds 0.02 eV/atom"
            )
    magnetic = pd.DataFrame(magnetic_rows).sort_values(["candidate_id", "magnetic_initialization"]).reset_index(drop=True)
    selected = pd.DataFrame(selected_rows).sort_values("candidate_id").reset_index(drop=True)
    unique_reasons = list(dict.fromkeys(pause_reasons))
    review = {
        "review_generated_from_raw_static_outputs": True,
        "paper_conclusion_update_authorized": not unique_reasons,
        "pause_reasons": unique_reasons,
        "formation_energy_shift_threshold_eV_per_atom": FORMATION_SHIFT_THRESHOLD_EV_ATOM,
        "static_force_diagnostic_threshold_eV_A": FORCE_THRESHOLD_EV_A,
        "reference_convention": "retained frozen GGA+U Li, Cr, and O elemental references",
        "magnetic_scope": "two tested magnetic initializations",
    }
    return static, magnetic, formation, selected, review


def build_report(selected: pd.DataFrame, review: dict[str, Any]) -> str:
    rows = []
    for row in selected.itertuples(index=False):
        rows.append(
            f"| {row.candidate_id} | {row.historical_selected_initialization} | {row.new_selected_initialization} | "
            f"{row.historical_selected_formation_energy_eV_per_atom:.9f} | "
            f"{row.new_selected_formation_energy_eV_per_atom:.9f} | "
            f"{row.selected_formation_energy_shift_eV_per_atom:+.9f} |"
        )
    pauses = "\n".join(f"- {reason}" for reason in review["pause_reasons"]) or "- None."
    return """# Main-candidate verification relaxation and frozen-static report

## Scope

C120 and C214 were each evaluated with two tested magnetic initializations. The
reported selected result is the lower-energy configuration among the two tested
initializations; this is not an exhaustive magnetic search or a ground-state claim.

## Selected frozen-static comparison

| candidate | historical initialization | new initialization | historical Ef (eV/atom) | new Ef (eV/atom) | shift (eV/atom) |
|---|---|---|---:|---:|---:|
""" + "\n".join(rows) + f"""

The 0.02 eV/atom stop rule is evaluated from raw VASP total energies using the
retained frozen GGA+U Li, Cr, and O elemental references. Static forces are reported
as diagnostic forces, not ionic-convergence markers.

## Conclusion-update gate

- Authorized: `{str(review['paper_conclusion_update_authorized']).lower()}`
- Pause reasons:
{pauses}
"""


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _write_json(payload: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-input-manifest", type=Path, required=True)
    parser.add_argument("--static-queue-manifest", type=Path, required=True)
    parser.add_argument("--relaxation-metrics", type=Path, required=True)
    parser.add_argument("--historical-magnetic", type=Path, required=True)
    parser.add_argument("--elemental-references", type=Path, required=True)
    parser.add_argument("--cif-root", type=Path, required=True)
    parser.add_argument("--static-metrics", type=Path, required=True)
    parser.add_argument("--magnetic", type=Path, required=True)
    parser.add_argument("--formation", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    outputs = analyze_verification_statics(
        pd.read_csv(args.static_input_manifest, dtype=str, keep_default_na=False),
        pd.read_csv(args.static_queue_manifest, dtype=str, keep_default_na=False),
        pd.read_csv(args.relaxation_metrics),
        pd.read_csv(args.historical_magnetic),
        pd.read_csv(args.elemental_references),
        cif_root=args.cif_root,
    )
    static, magnetic, formation, selected, review = outputs
    _write_csv(static, args.static_metrics)
    _write_csv(magnetic, args.magnetic)
    _write_csv(formation, args.formation)
    _write_csv(selected, args.selected)
    _write_json(review, args.review)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(selected, review), encoding="utf-8", newline="\n")
    print(json.dumps(review, indent=2, ensure_ascii=False))
    return 0 if review["paper_conclusion_update_authorized"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
