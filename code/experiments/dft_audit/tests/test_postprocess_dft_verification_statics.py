from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from analysis.postprocess_dft_verification_statics import analyze_verification_statics


def _structure() -> Structure:
    return Structure(
        Lattice.cubic(8.0),
        ["Li", "Cr", "Cr", "O", "O", "O", "O"],
        [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75], [0.1, 0.1, 0.1],
         [0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]],
    )


def _write_static(root: Path, job_id: str, energy: float) -> tuple[dict[str, object], dict[str, object]]:
    input_dir = root / "inputs" / job_id
    output = root / "outputs" / job_id / "attempt_1"
    input_dir.mkdir(parents=True)
    output.mkdir(parents=True)
    structure = _structure()
    Poscar(structure).write_file(input_dir / "POSCAR")
    (input_dir / "INCAR").write_text(
        "ENCUT=520\nEDIFF=1E-6\nIBRION=-1\nNSW=0\nISMEAR=0\nSIGMA=0.05\n"
        "LDAU=.TRUE.\nLDAUL=-1 2 -1\nLDAUU=0.0 3.7 0.0\nLDAUJ=0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    mesh = "15 9 9" if "c120" in job_id else "15 10 9"
    (input_dir / "KPOINTS").write_text(f"mesh\n0\nGamma\n{mesh}\n", encoding="utf-8")
    for name in ("POSCAR", "INCAR", "KPOINTS"):
        (output / name).write_bytes((input_dir / name).read_bytes())
    Poscar(structure).write_file(output / "CONTCAR")
    force_rows = "\n".join(
        f" {site.x:10.5f} {site.y:10.5f} {site.z:10.5f} 0.01000 0.00000 0.00000"
        for site in structure
    )
    (output / "OUTCAR").write_text(
        "vasp.6.5.1\n"
        "aborting loop because EDIFF is reached\n"
        f"free  energy   TOTEN  = {energy:15.8f} eV\n"
        "TOTAL-FORCE (eV/Angst)\n ---------------------------------------------\n"
        + force_rows
        + "\n ---------------------------------------------\n"
        "General timing and accounting informations for this job\n",
        encoding="utf-8",
    )
    (output / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 mag= 4.2\n", encoding="utf-8")
    (output / "vasprun.xml").write_text("<modeling/>\n", encoding="utf-8")
    (output / "task_result.json").write_text(
        json.dumps(
            {
                "status": "DONE",
                "exit_code": 0,
                "electronic_converged": True,
                "timing_footer_present": True,
                "potcar_retained": False,
            }
        ),
        encoding="utf-8",
    )
    dependency = job_id.replace("_static", "_relax")
    candidate = "C120" if "c120" in job_id else "C214"
    state = "state_afm" if "state_afm" in job_id else "state_fm"
    return (
        {
            "job_id": job_id,
            "candidate_id": candidate,
            "magnetic_initialization": state,
            "dependency_job_id": dependency,
            "input_dir": str(input_dir),
        },
        {"job_id": job_id, "status": "DONE", "exit_code": "0", "output_path": str(output)},
    )


def _inputs(root: Path, *, shifted_energy: bool = False):
    energies = {
        ("C120", "state_fm"): -49.1090 - (0.21 if shifted_energy else 0.0),
        ("C120", "state_afm"): -49.0530,
        ("C214", "state_fm"): -49.0590,
        ("C214", "state_afm"): -49.0000,
    }
    inputs, queue = [], []
    relax_rows = []
    for (candidate, state), energy in energies.items():
        job = f"dft_{candidate.lower()}_{state}_verification_static"
        rows = _write_static(root, job, energy)
        inputs.append(rows[0])
        queue.append(rows[1])
        relax_rows.append(
            {
                "job_id": job.replace("_static", "_relax"),
                "candidate_id": candidate,
                "magnetic_initialization": state,
                "final_total_energy_eV_relaxation": energy + 0.001,
                "final_volume_A3": 512.0,
                "relative_volume_change_percent": 0.0,
                "maximum_internal_displacement_A": 0.0,
                "Fmax_eV_A": 0.04,
                "final_space_group": "Pm-3m (221)",
            }
        )
    historical = pd.DataFrame(
        [
            {
                "candidate_id": f"job_{candidate[1:]}_Cr_fixture",
                "functional": "GGA+U",
                "magnetic_initialization": state,
                "final_total_energy_eV": (
                    -49.1092 if candidate == "C120" and state == "state_fm" else
                    -49.0532 if candidate == "C120" else
                    -49.0590 if state == "state_fm" else -49.0002
                ),
                "selected_lower_energy_among_two_tested": state == "state_fm",
                "outcar_sha256": f"historical-{candidate}-{state}",
                "source_output_path": f"/historical/{candidate}/{state}",
            }
            for candidate in ("C120", "C214")
            for state in ("state_fm", "state_afm")
        ]
    )
    historical = pd.concat(
        [
            historical,
            pd.DataFrame(
                [
                    {
                        "candidate_id": "job_044_Mn_fixture",
                        "functional": "GGA+U",
                        "magnetic_initialization": "state_fm",
                        "final_total_energy_eV": -47.85,
                        "selected_lower_energy_among_two_tested": True,
                        "outcar_sha256": "historical-C044-state_fm",
                        "source_output_path": "/historical/C044/state_fm",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    references = pd.DataFrame(
        [
            {"reference_id": "GGA_U_Li_metal", "element": "Li", "energy_per_atom_eV": -1.9,
             "electronic_converged": True, "timing_footer_present": True, "raw_output_sha256": "li"},
            {"reference_id": "GGA_U_Cr_metal", "element": "Cr", "energy_per_atom_eV": -5.8,
             "electronic_converged": True, "timing_footer_present": True, "raw_output_sha256": "cr"},
            {"reference_id": "GGA_U_O2_molecule", "element": "O", "energy_per_atom_eV": -4.9,
             "electronic_converged": True, "timing_footer_present": True, "raw_output_sha256": "o"},
        ]
    )
    return pd.DataFrame(inputs), pd.DataFrame(queue), pd.DataFrame(relax_rows), historical, references


def test_static_postprocessor_recomputes_statewise_and_selected_formation_energies(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    static, magnetic, formation, selected, review = analyze_verification_statics(
        *inputs,
        cif_root=tmp_path / "cifs",
    )

    assert len(static) == len(magnetic) == len(formation) == 4
    assert len(selected) == 2
    assert selected["new_selected_initialization"].tolist() == ["state_fm", "state_fm"]
    assert review["paper_conclusion_update_authorized"] is True
    assert review["pause_reasons"] == []
    c120 = formation.query("candidate_id == 'C120' and magnetic_initialization == 'state_fm'").iloc[0]
    assert c120["new_formation_energy_eV_per_atom"] == pytest.approx((-49.1090 + 1.9 + 2 * 5.8 + 4 * 4.9) / 7)
    assert c120["new_minus_historical_formation_energy_eV_per_atom"] == pytest.approx(0.0002 / 7)


def test_selected_formation_energy_shift_over_threshold_pauses_conclusion_update(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, shifted_energy=True)
    _, _, _, selected, review = analyze_verification_statics(*inputs, cif_root=tmp_path / "cifs")

    assert abs(selected.loc[selected["candidate_id"] == "C120", "selected_formation_energy_shift_eV_per_atom"].iloc[0]) > 0.02
    assert review["paper_conclusion_update_authorized"] is False
    assert any("0.02" in reason for reason in review["pause_reasons"])
