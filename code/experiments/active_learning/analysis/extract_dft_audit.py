from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Incar, Kpoints, Poscar
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file, write_bytes_protected
from analysis.build_dft_manifest import build_dft_manifest


@dataclass(frozen=True)
class StageRecord:
    candidate_id: str
    cohort: str
    calculation_id: str
    stage_role: str
    incar: Path
    kpoints: Path
    poscar: Path
    contcar: Path | None
    outcar: Path | None
    oszicar: Path | None
    vasprun: Path | None
    potcar_spec: Path | None
    output_provenance: str


@dataclass(frozen=True)
class DftAuditTables:
    settings: pd.DataFrame
    convergence: pd.DataFrame
    structures: pd.DataFrame
    magnetic: pd.DataFrame
    references: pd.DataFrame


def _rel(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _existing(path: Path) -> Path | None:
    return path if path.is_file() else None


def _serialise(value: Any) -> Any:
    if value is None:
        return pd.NA
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _incar_values(path: Path) -> dict[str, Any]:
    incar = Incar.from_file(path)
    values = {key.upper(): value for key, value in incar.items()}
    return values


def _kpoints_values(path: Path, incar: dict[str, Any]) -> tuple[str, str, Any]:
    kspacing = _serialise(incar.get("KSPACING"))
    if not path.is_file():
        return "", "", kspacing
    points = Kpoints.from_file(path)
    mesh = ""
    if points.kpts:
        first = points.kpts[0]
        if len(first) == 3:
            mesh = "x".join(str(int(round(float(value)))) for value in first)
    return mesh, str(points.style), kspacing


def _element_order(path: Path) -> str:
    return " ".join(Poscar.from_file(path).site_symbols)


def _paw_labels(path: Path | None, outcar_text: str = "") -> str:
    labels: list[str] = []
    if path is not None and path.is_file():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "PAW_" in line:
                label = line[line.index("PAW_") :].strip()
                if label not in labels:
                    labels.append(label)
    if not labels and outcar_text:
        for match in re.findall(r"TITEL\s*=\s*(.+)", outcar_text):
            label = match.strip()
            if label not in labels:
                labels.append(label)
    return " | ".join(labels)


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return float(matches[-1]) if matches else None


def _last_fmax(text: str) -> float | None:
    blocks = re.findall(
        r"TOTAL-FORCE\s*\(eV/Angst\)\s*\n\s*-+\s*\n(.*?)(?=\n\s*-{5,})",
        text,
        flags=re.DOTALL,
    )
    if not blocks:
        return None
    forces: list[float] = []
    for line in blocks[-1].splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        try:
            xyz = [float(value) for value in fields[-3:]]
        except ValueError:
            continue
        forces.append(float(np.linalg.norm(xyz)))
    return max(forces) if forces else None


def _outcar_values(outcar: Path | None, oszicar: Path | None) -> dict[str, Any]:
    if outcar is None or not outcar.is_file():
        return {
            "text": "",
            "vasp_version": pd.NA,
            "electronic_converged": pd.NA,
            "ionic_marker": pd.NA,
            "vasp_completed": pd.NA,
            "total_energy_eV": pd.NA,
            "Fmax_eV_A": pd.NA,
            "total_magnetic_moment": pd.NA,
        }
    text = outcar.read_text(encoding="utf-8", errors="ignore")
    version_match = re.search(r"\bvasp\.([0-9]+(?:\.[0-9]+)+)", text[:20_000], re.IGNORECASE)
    magnetic_moment: float | None = None
    if oszicar is not None and oszicar.is_file():
        oszicar_text = oszicar.read_text(encoding="utf-8", errors="ignore")
        magnetic_moment = _last_float(r"\bmag=\s*([-+0-9.Ee]+)", oszicar_text)
    if magnetic_moment is None:
        magnetic_moment = _last_float(
            r"number of electron\s+[-+0-9.Ee]+\s+magnetization\s+([-+0-9.Ee]+)", text
        )
    return {
        "text": text,
        "vasp_version": version_match.group(1) if version_match else pd.NA,
        "electronic_converged": "aborting loop because EDIFF is reached" in text,
        "ionic_marker": "reached required accuracy - stopping structural energy minimisation" in text,
        "vasp_completed": "General timing and accounting informations for this job" in text,
        "total_energy_eV": _last_float(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)", text),
        "Fmax_eV_A": _last_fmax(text),
        "total_magnetic_moment": magnetic_moment,
    }


def _discover_pilot_stages(root: Path, manifest: pd.DataFrame) -> list[StageRecord]:
    stages: list[StageRecord] = []
    for row in manifest.query("pilot_or_new == 'pilot'").itertuples(index=False):
        directory = root / row.raw_job_path
        stages.extend(
            [
                StageRecord(
                    row.candidate_id,
                    "pilot",
                    "pilot_pbe_relax",
                    "relax",
                    directory / "INCAR_relax",
                    directory / "KPOINTS",
                    directory / "POSCAR",
                    directory / "CONTCAR",
                    None,
                    None,
                    None,
                    _existing(directory / "POTCAR.spec"),
                    "original_relaxation_artifact_unavailable; relax.log retained",
                ),
                StageRecord(
                    row.candidate_id,
                    "pilot",
                    "pilot_pbe_static",
                    "static",
                    directory / "INCAR_static",
                    directory / "KPOINTS",
                    directory / "CONTCAR",
                    directory / "CONTCAR",
                    directory / "OUTCAR",
                    directory / "OSZICAR",
                    directory / "vasprun.xml",
                    _existing(directory / "POTCAR.spec"),
                    "retained original static output; prior relaxation output was overwritten",
                ),
            ]
        )
        gga = directory / "gga_u"
        if gga.is_dir():
            stages.extend(
                [
                    StageRecord(
                        row.candidate_id,
                        "pilot",
                        "pilot_gga_u_relax",
                        "relax",
                        gga / "INCAR_relax_gga_u",
                        gga / "KPOINTS",
                        directory / "CONTCAR",
                        gga / "CONTCAR",
                        None,
                        None,
                        None,
                        _existing(gga / "POTCAR.spec"),
                        "original_relaxation_artifact_unavailable; relax.log retained",
                    ),
                    StageRecord(
                        row.candidate_id,
                        "pilot",
                        "pilot_gga_u_static",
                        "static",
                        gga / "INCAR_static_gga_u",
                        gga / "KPOINTS",
                        gga / "CONTCAR",
                        gga / "CONTCAR",
                        gga / "OUTCAR",
                        gga / "OSZICAR",
                        gga / "vasprun.xml",
                        _existing(gga / "POTCAR.spec"),
                        "retained original static output; prior relaxation output was overwritten",
                    ),
                ]
            )
        repair = directory / "static_repair_attempt"
        if repair.is_dir():
            stages.append(
                StageRecord(
                    row.candidate_id,
                    "pilot",
                    "pilot_static_repair",
                    "static",
                    repair / "INCAR",
                    repair / "KPOINTS",
                    repair / "POSCAR",
                    _existing(repair / "CONTCAR"),
                    _existing(repair / "OUTCAR"),
                    _existing(repair / "OSZICAR"),
                    _existing(repair / "vasprun.xml"),
                    _existing(repair / "POTCAR.spec"),
                    "retained failed repair attempt",
                )
            )
    return stages


def _new_calculation_id(relative_directory: Path) -> str:
    parts = relative_directory.parts
    if parts[0] == "stages":
        return parts[-1]
    if parts[0] == "tight_magnetic_states":
        return f"tight_{parts[1]}_{parts[2]}"
    return "_".join(parts)


def _discover_new_stages(root: Path, manifest: pd.DataFrame) -> list[StageRecord]:
    stages: list[StageRecord] = []
    for row in manifest.query("pilot_or_new == 'new'").itertuples(index=False):
        directory = root / row.raw_job_path
        for incar in sorted(directory.rglob("INCAR")):
            calculation_dir = incar.parent
            relative = calculation_dir.relative_to(directory)
            calculation_id = _new_calculation_id(relative)
            stage_role = "relax" if "relax" in calculation_id else "static"
            outcar = _existing(calculation_dir / "OUTCAR")
            output_provenance = (
                "stage-specific original output"
                if outcar is not None
                else "stage inputs retained; output unavailable in archived job bundle"
            )
            stages.append(
                StageRecord(
                    row.candidate_id,
                    "new",
                    calculation_id,
                    stage_role,
                    incar,
                    calculation_dir / "KPOINTS",
                    calculation_dir / "POSCAR",
                    _existing(calculation_dir / "CONTCAR"),
                    outcar,
                    _existing(calculation_dir / "OSZICAR"),
                    _existing(calculation_dir / "vasprun.xml"),
                    _existing(calculation_dir / "POTCAR.spec"),
                    output_provenance,
                )
            )
    return stages


def discover_candidate_stages(root: Path, manifest: pd.DataFrame) -> list[StageRecord]:
    records = _discover_pilot_stages(root, manifest) + _discover_new_stages(root, manifest)
    for record in records:
        for path in (record.incar, record.kpoints, record.poscar):
            if not path.is_file():
                raise FileNotFoundError(path)
    return sorted(records, key=lambda item: (item.cohort, item.candidate_id, item.calculation_id))


def _settings_and_convergence(root: Path, stages: list[StageRecord]) -> tuple[pd.DataFrame, pd.DataFrame]:
    setting_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    for stage in stages:
        incar = _incar_values(stage.incar)
        out = _outcar_values(stage.outcar, stage.oszicar)
        kpoints_mesh, kpoints_style, kspacing = _kpoints_values(stage.kpoints, incar)
        nsw = int(incar.get("NSW", 0))
        if stage.outcar is None:
            ionic_status: Any = pd.NA
            failure_reason = (
                "original_relaxation_artifact_unavailable"
                if stage.cohort == "pilot"
                else "stage_output_unavailable_in_archived_job_bundle"
            )
        elif stage.stage_role == "static" or nsw == 0:
            ionic_status = "not_applicable_static"
            failure_reason = "" if out["electronic_converged"] else "electronic convergence marker absent"
        else:
            ionic_status = "converged" if out["ionic_marker"] else "convergence marker absent"
            failures = []
            if not out["electronic_converged"]:
                failures.append("electronic convergence marker absent")
            if not out["ionic_marker"]:
                failures.append("ionic convergence marker absent")
            failure_reason = "; ".join(failures)

        functional = "GGA+U" if bool(incar.get("LDAU", False)) else "PBE"
        common = {
            "candidate_id": stage.candidate_id,
            "cohort": stage.cohort,
            "calculation_id": stage.calculation_id,
            "stage_role": stage.stage_role,
        }
        setting_rows.append(
            {
                **common,
                "functional": functional,
                "vasp_version": out["vasp_version"],
                "paw_labels": _paw_labels(stage.potcar_spec, out["text"]),
                "element_order": _element_order(stage.poscar),
                "ENCUT": _serialise(incar.get("ENCUT")),
                "kpoints_mesh": kpoints_mesh,
                "kpoints_style": kpoints_style,
                "KSPACING": kspacing,
                "EDIFF": _serialise(incar.get("EDIFF")),
                "EDIFFG": _serialise(incar.get("EDIFFG")),
                "ISMEAR": _serialise(incar.get("ISMEAR")),
                "SIGMA": _serialise(incar.get("SIGMA")),
                "ISPIN": _serialise(incar.get("ISPIN")),
                "MAGMOM": _serialise(incar.get("MAGMOM")),
                "MAGMOM_explicit": "MAGMOM" in incar,
                "LDAU": bool(incar.get("LDAU", False)),
                "LDAUL": _serialise(incar.get("LDAUL")),
                "LDAUU": _serialise(incar.get("LDAUU")),
                "LDAUJ": _serialise(incar.get("LDAUJ")),
                "LASPH": _serialise(incar.get("LASPH")),
                "LMAXMIX": _serialise(incar.get("LMAXMIX")),
                "NSW": nsw,
                "IBRION": _serialise(incar.get("IBRION")),
                "ISIF": _serialise(incar.get("ISIF")),
                "incar_path": _rel(stage.incar, root),
                "kpoints_path": _rel(stage.kpoints, root),
                "potcar_spec_path": _rel(stage.potcar_spec, root),
                "poscar_path": _rel(stage.poscar, root),
                "contcar_path": _rel(stage.contcar, root),
                "outcar_path": _rel(stage.outcar, root),
                "oszicar_path": _rel(stage.oszicar, root),
                "vasprun_path": _rel(stage.vasprun, root),
                "outcar_available": stage.outcar is not None,
                "vasprun_available": stage.vasprun is not None,
                "output_provenance": stage.output_provenance,
            }
        )
        convergence_rows.append(
            {
                **common,
                "functional": functional,
                "outcar_available": stage.outcar is not None,
                "vasp_completed": out["vasp_completed"],
                "electronic_converged": out["electronic_converged"],
                "ionic_convergence_status": ionic_status,
                "final_total_energy_eV": out["total_energy_eV"],
                "Fmax_eV_A": out["Fmax_eV_A"],
                "final_total_magnetic_moment": out["total_magnetic_moment"],
                "failure_reason": failure_reason,
                "outcar_path": _rel(stage.outcar, root),
                "outcar_sha256": sha256_file(stage.outcar) if stage.outcar is not None else "",
                "vasprun_path": _rel(stage.vasprun, root),
                "output_provenance": stage.output_provenance,
            }
        )
    return pd.DataFrame(setting_rows), pd.DataFrame(convergence_rows)


def _minimum_distances(structure: Structure) -> tuple[float, float | None]:
    matrix = np.asarray(structure.distance_matrix, dtype=float)
    mask = ~np.eye(len(structure), dtype=bool)
    minimum = float(matrix[mask].min())
    metal_indices = [
        index for index, site in enumerate(structure) if site.specie.symbol in {"Cr", "Mn", "Mg"}
    ]
    oxygen_indices = [index for index, site in enumerate(structure) if site.specie.symbol == "O"]
    metal_oxygen = [matrix[i, j] for i in metal_indices for j in oxygen_indices]
    return minimum, float(min(metal_oxygen)) if metal_oxygen else None


def _space_group(structure: Structure) -> str:
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    return f"{analyzer.get_space_group_symbol()} ({analyzer.get_space_group_number()})"


def _stage_map(stages: list[StageRecord]) -> dict[tuple[str, str], StageRecord]:
    return {(stage.candidate_id, stage.calculation_id): stage for stage in stages}


def _lower_tight_state(candidate_id: str, stages: dict[tuple[str, str], StageRecord]) -> str:
    energies: dict[str, float] = {}
    for state in ("state_afm", "state_fm"):
        stage = stages[(candidate_id, f"tight_{state}_02_static")]
        value = _outcar_values(stage.outcar, stage.oszicar)["total_energy_eV"]
        if value is None or pd.isna(value):
            raise ValueError(f"missing tight-state energy for {candidate_id} {state}")
        energies[state] = float(value)
    return min(energies, key=energies.get)


def _selected_configuration(
    row: Any, stages: dict[tuple[str, str], StageRecord]
) -> tuple[StageRecord, StageRecord, str]:
    candidate_id = row.candidate_id
    if row.pilot_or_new == "pilot":
        relax = stages[(candidate_id, "pilot_pbe_relax")]
        static = stages[(candidate_id, "pilot_pbe_static")]
        return relax, static, "pilot primary PBE configuration"
    tight_key = (candidate_id, "tight_state_fm_02_static")
    if tight_key in stages:
        state = _lower_tight_state(candidate_id, stages)
        return (
            stages[(candidate_id, f"tight_{state}_01_tight_relax")],
            stages[(candidate_id, f"tight_{state}_02_static")],
            "lower-energy configuration among the two tested initializations",
        )
    if row.DFT_status == "failed":
        return (
            stages[(candidate_id, "01_pbe_relax")],
            stages[(candidate_id, "02_pbe_static")],
            "retained failed-candidate PBE configuration",
        )
    if row.formula == "LiMg2O4":
        return (
            stages[(candidate_id, "01_pbe_relax")],
            stages[(candidate_id, "02_pbe_static")],
            "PBE configuration",
        )
    return (
        stages[(candidate_id, "03_gga_u_relax")],
        stages[(candidate_id, "04_gga_u_static")],
        "GGA+U configuration",
    )


def _build_structure_metrics(
    root: Path, manifest: pd.DataFrame, stage_records: list[StageRecord]
) -> pd.DataFrame:
    stages = _stage_map(stage_records)
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest.itertuples(index=False):
        relax, static, selection = _selected_configuration(manifest_row, stages)
        if relax.contcar is None:
            raise FileNotFoundError(f"selected final structure missing for {manifest_row.candidate_id}")
        initial = Structure.from_file(relax.poscar)
        final = Structure.from_file(relax.contcar)
        minimum, metal_oxygen = _minimum_distances(final)
        relax_out = _outcar_values(relax.outcar, relax.oszicar)
        out = _outcar_values(static.outcar, static.oszicar)
        if relax.outcar is not None and not pd.isna(relax_out["Fmax_eV_A"]):
            reported_fmax = relax_out["Fmax_eV_A"]
            fmax_source = "final relaxation OUTCAR"
        else:
            reported_fmax = out["Fmax_eV_A"]
            fmax_source = "final static OUTCAR; original relaxation OUTCAR unavailable"
        rows.append(
            {
                "candidate_id": manifest_row.candidate_id,
                "formula": manifest_row.formula,
                "cohort": manifest_row.pilot_or_new,
                "configuration_source": selection,
                "relaxation_calculation_id": relax.calculation_id,
                "static_calculation_id": static.calculation_id,
                "initial_volume_A3": float(initial.volume),
                "final_volume_A3": float(final.volume),
                "relative_volume_change_percent": float(
                    100.0 * (final.volume - initial.volume) / initial.volume
                ),
                "minimum_interatomic_distance_A": minimum,
                "minimum_M_O_distance_A": metal_oxygen,
                "Fmax_eV_A": reported_fmax,
                "Fmax_source": fmax_source,
                "relaxation_Fmax_eV_A": relax_out["Fmax_eV_A"],
                "final_static_Fmax_eV_A": out["Fmax_eV_A"],
                "electronic_converged": out["electronic_converged"],
                "ionic_convergence_status": (
                    "unverifiable_original_artifact_missing"
                    if relax.outcar is None
                    else (
                        "converged"
                        if relax_out["ionic_marker"]
                        else "convergence marker absent"
                    )
                ),
                "final_total_energy_eV": out["total_energy_eV"],
                "final_total_magnetic_moment": out["total_magnetic_moment"],
                "initial_space_group": _space_group(initial),
                "final_space_group": _space_group(final),
                "initial_structure_path": _rel(relax.poscar, root),
                "final_structure_path": _rel(relax.contcar, root),
                "final_static_outcar_path": _rel(static.outcar, root),
                "metric_definition": (
                    "periodic minimum-image distances; Fmax from relaxation output when available, "
                    "otherwise final static output; energy from final static output"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["cohort", "candidate_id"]).reset_index(drop=True)


def _build_magnetic_initializations(
    root: Path, stages: list[StageRecord]
) -> pd.DataFrame:
    mapping = _stage_map(stages)
    candidate_ids = sorted(
        {
            candidate_id
            for candidate_id, calculation_id in mapping
            if calculation_id.startswith("tight_state_")
        }
    )
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        state_rows: list[dict[str, Any]] = []
        for state in ("state_afm", "state_fm"):
            relax = mapping[(candidate_id, f"tight_{state}_01_tight_relax")]
            static = mapping[(candidate_id, f"tight_{state}_02_static")]
            relax_incar = _incar_values(relax.incar)
            relax_out = _outcar_values(relax.outcar, relax.oszicar)
            static_out = _outcar_values(static.outcar, static.oszicar)
            final_structure = Structure.from_file(relax.contcar) if relax.contcar else None
            state_rows.append(
                {
                    "candidate_id": candidate_id,
                    "initialization_label_as_recorded": state,
                    "scope_statement": "two tested magnetic initializations",
                    "initial_MAGMOM": _serialise(relax_incar.get("MAGMOM")),
                    "electronic_converged": static_out["electronic_converged"],
                    "ionic_converged": bool(relax_out["ionic_marker"]),
                    "total_energy_eV": static_out["total_energy_eV"],
                    "Fmax_eV_A": relax_out["Fmax_eV_A"],
                    "final_static_Fmax_eV_A": static_out["Fmax_eV_A"],
                    "final_total_magnetic_moment": static_out["total_magnetic_moment"],
                    "final_space_group": _space_group(final_structure) if final_structure else pd.NA,
                    "relax_outcar_path": _rel(relax.outcar, root),
                    "static_outcar_path": _rel(static.outcar, root),
                    "static_outcar_sha256": sha256_file(static.outcar) if static.outcar else "",
                }
            )
        lower = min(float(item["total_energy_eV"]) for item in state_rows)
        for item in state_rows:
            selected = math.isclose(float(item["total_energy_eV"]), lower, abs_tol=1e-10)
            item["selected_lower_energy_among_two"] = selected
            item["energy_difference_from_lower_eV"] = float(item["total_energy_eV"]) - lower
            item["selection_statement"] = (
                "lower-energy configuration among the two tested initializations" if selected else ""
            )
            rows.append(item)
    return pd.DataFrame(rows).sort_values(
        ["candidate_id", "initialization_label_as_recorded"]
    ).reset_index(drop=True)


def _reference_directories(root: Path) -> list[tuple[str, str, str, Path]]:
    records: list[tuple[str, str, str, Path]] = []
    names = {"Li": "Li_metal", "Cr": "Cr_metal", "Mn": "Mn_metal", "O": "O2_molecule"}
    for functional, folder in (("PBE", "reference_calculations_pbe"), ("GGA+U", "reference_calculations_gga_u")):
        for element, name in names.items():
            records.append(
                (
                    functional,
                    element,
                    name,
                    root / "tmp/dft_candidate_inventory/remote" / folder / name,
                )
            )
    records.append(
        (
            "PBE",
            "Mg",
            "Mg_metal",
            root
            / "new12_dft_screening_snapshot/extracted/reference_calculations_pbe/Mg_metal/02_pbe_static",
        )
    )
    return records


def _build_elemental_references(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for functional, element, name, directory in _reference_directories(root):
        outcar = directory / "OUTCAR"
        oszicar = _existing(directory / "OSZICAR")
        incar_path = directory / "INCAR_static"
        if not incar_path.is_file():
            incar_path = directory / "INCAR"
        poscar = directory / "POSCAR"
        contcar = directory / "CONTCAR"
        kpoints = directory / "KPOINTS"
        potcar_spec = _existing(directory / "POTCAR.spec")
        if not all(path.is_file() for path in (outcar, incar_path, poscar, contcar, kpoints)):
            raise FileNotFoundError(f"elemental reference evidence incomplete: {directory}")
        incar = _incar_values(incar_path)
        out = _outcar_values(outcar, oszicar)
        structure = Structure.from_file(contcar)
        total_energy = float(out["total_energy_eV"])
        atoms = len(structure)
        mesh, style, kspacing = _kpoints_values(kpoints, incar)
        structure_label = (
            "O2 molecule in periodic cell" if element == "O" else _space_group(structure)
        )
        rows.append(
            {
                "reference_name": name,
                "element": element,
                "structure": structure_label,
                "magnetic_setup": (
                    f"ISPIN={_serialise(incar.get('ISPIN'))}; MAGMOM={_serialise(incar.get('MAGMOM'))}"
                ),
                "paw_label": _paw_labels(potcar_spec, out["text"]),
                "functional": functional,
                "Ueff_eV": _serialise(incar.get("LDAUU")),
                "total_energy_eV": total_energy,
                "atoms_per_cell": atoms,
                "energy_per_atom_eV": total_energy / atoms,
                "ENCUT": _serialise(incar.get("ENCUT")),
                "kpoints_mesh": mesh,
                "kpoints_style": style,
                "KSPACING": kspacing,
                "electronic_converged": out["electronic_converged"],
                "raw_output_path": _rel(outcar, root),
                "raw_output_sha256": sha256_file(outcar),
                "incar_path": _rel(incar_path, root),
                "poscar_path": _rel(poscar, root),
                "contcar_path": _rel(contcar, root),
                "notes": (
                    "internal bcc Mn screening reference; not alpha-Mn/Materials Project compatible"
                    if element == "Mn"
                    else "internal same-protocol elemental reference"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["functional", "element"]).reset_index(drop=True)


def build_dft_audit_tables(root: Path) -> DftAuditTables:
    archive_root = Path(root).resolve()
    manifest = build_dft_manifest(archive_root)
    stages = discover_candidate_stages(archive_root, manifest)
    settings, convergence = _settings_and_convergence(archive_root, stages)
    structures = _build_structure_metrics(archive_root, manifest, stages)
    magnetic = _build_magnetic_initializations(archive_root, stages)
    references = _build_elemental_references(archive_root)
    return DftAuditTables(settings, convergence, structures, magnetic, references)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    root = args.archive_root.resolve()
    tables = build_dft_audit_tables(root)
    outputs = {
        root / "dft/audit/dft_settings.csv": _csv_bytes(tables.settings),
        root / "dft/audit/convergence_inventory.csv": _csv_bytes(tables.convergence),
        root / "dft/audit/structure_metrics.csv": _csv_bytes(tables.structures),
        root / "dft/audit/magnetic_initializations.csv": _csv_bytes(tables.magnetic),
        root / "dft/audit/elemental_references.csv": _csv_bytes(tables.references),
    }
    statuses = {
        str(path.relative_to(root)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
