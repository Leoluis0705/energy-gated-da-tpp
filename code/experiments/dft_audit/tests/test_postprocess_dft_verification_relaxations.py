from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from analysis.postprocess_dft_verification_relaxations import (
    analyze_verification_relaxations,
)
from analysis.prepare_dft_finalization_inputs import build_candidate_static_inputs


def _structure(*, shifted: bool = False) -> Structure:
    coordinates = [
        [0.00, 0.00, 0.00],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [0.10, 0.10, 0.10],
        [0.40, 0.40, 0.40],
        [0.60, 0.60, 0.60],
        [0.90, 0.90, 0.90],
    ]
    if shifted:
        coordinates[3] = [0.47, 0.10, 0.10]
    return Structure(
        Lattice.from_parameters(8.0, 8.2, 8.4, 90, 91, 92),
        ["Li", "Cr", "Cr", "O", "O", "O", "O"],
        coordinates,
    )


def _outcar(structure: Structure, *, fmax: float, converged: bool = True) -> str:
    force_rows = []
    for index, site in enumerate(structure):
        force = fmax if index == 0 else 0.0
        force_rows.append(
            f" {site.x:12.6f} {site.y:12.6f} {site.z:12.6f}"
            f" {force:12.6f} {0.0:12.6f} {0.0:12.6f}"
        )
    convergence = (
        "aborting loop because EDIFF is reached\n"
        "reached required accuracy - stopping structural energy minimisation\n"
        if converged
        else ""
    )
    return (
        "vasp.6.5.1 10Mar25\n"
        + convergence
        + "free  energy   TOTEN  =      -100.123456 eV\n"
        + "TOTAL-FORCE (eV/Angst)\n"
        + " -------------------------------------------------------------------\n"
        + "\n".join(force_rows)
        + "\n -------------------------------------------------------------------\n"
        + "number of electron       50.000 magnetization       4.250\n"
        + "General timing and accounting informations for this job\n"
    )


def _write_job(
    root: Path,
    job_id: str,
    candidate: str,
    state: str,
    *,
    fmax: float = 0.04,
    shifted: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    input_dir = root / "inputs" / job_id
    output_dir = root / "outputs" / job_id / "attempt_1"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    initial = _structure()
    final = _structure(shifted=shifted)
    Poscar(initial).write_file(input_dir / "POSCAR")
    Poscar(initial).write_file(input_dir / "initial.POSCAR")
    initial.to(filename=input_dir / "initial.cif", fmt="cif", symprec=None)
    (input_dir / "INCAR").write_text(
        "ENCUT = 520\nEDIFF = 1E-6\nEDIFFG = -0.05\nNSW = 160\n",
        encoding="utf-8",
    )
    mesh = "15 9 9" if candidate == "C120" else "15 10 9"
    (input_dir / "KPOINTS").write_text(f"mesh\n0\nGamma\n{mesh}\n", encoding="utf-8")
    for name in ("INCAR", "KPOINTS", "POSCAR"):
        (output_dir / name).write_bytes((input_dir / name).read_bytes())
    Poscar(final).write_file(output_dir / "CONTCAR")
    (output_dir / "OUTCAR").write_text(_outcar(final, fmax=fmax), encoding="utf-8")
    (output_dir / "OSZICAR").write_text(" 1 F= -.100E+03 E0= -.100E+03  mag= 4.250\n", encoding="utf-8")
    (output_dir / "vasprun.xml").write_text("<modeling/>\n", encoding="utf-8")
    (output_dir / "task_result.json").write_text(
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
    input_row = {
        "job_id": job_id,
        "candidate_id": candidate,
        "magnetic_initialization": state,
        "input_dir": str(input_dir),
        "config_hash": f"hash-{job_id}",
        "git_commit": "abc123",
    }
    queue_row = {
        "job_id": job_id,
        "status": "DONE",
        "exit_code": "0",
        "output_path": str(output_dir),
    }
    return input_row, queue_row


def _four_jobs(root: Path, *, bad_force: bool = False, shifted_branch: bool = False):
    inputs = []
    queue = []
    for candidate in ("C120", "C214"):
        for state in ("state_fm", "state_afm"):
            job_id = f"dft_{candidate.lower()}_{state}_verification_relax"
            rows = _write_job(
                root,
                job_id,
                candidate,
                state,
                fmax=0.06 if bad_force and candidate == "C120" and state == "state_fm" else 0.04,
                shifted=shifted_branch and candidate == "C214" and state == "state_afm",
            )
            inputs.append(rows[0])
            queue.append(rows[1])
    return pd.DataFrame(inputs), pd.DataFrame(queue)


def _static_manifest() -> pd.DataFrame:
    rows = []
    for candidate in ("C120", "C214"):
        for state in ("state_fm", "state_afm"):
            dependency = f"dft_{candidate.lower()}_{state}_verification_relax"
            rows.append(
                {
                    "job_id": dependency.replace("_relax", "_static"),
                    "task_group": "main_candidate_verification_static",
                    "candidate_id": candidate,
                    "candidate_long_id": candidate,
                    "calculation_type": "verification_static",
                    "NELM": 160,
                    "scientific_result": True,
                    "dependency_job_ids": dependency,
                    "status": "PENDING",
                    "magnetic_initialization": state,
                    "expected_mesh": "15x9x9" if candidate == "C120" else "15x10x9",
                    "ENCUT_eV": 520,
                    "EDIFF": 1e-6,
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "ALGO": "Normal",
                    "MAGMOM_definition": (
                        "1*0.6 2*5.0 4*0.6"
                        if state == "state_fm"
                        else "1*0.6 1*5.0 1*-5.0 4*0.6"
                    ),
                    "LDAUL": "-1 2 -1",
                    "LDAUU": "0.0 3.7 0.0",
                    "LDAUJ": "0.0 0.0 0.0",
                    "base_protocol_sha256": "frozen-sha",
                }
            )
    return pd.DataFrame(rows)


def test_safe_relaxations_generate_quantitative_review_and_static_inputs(tmp_path: Path) -> None:
    inputs, queue = _four_jobs(tmp_path)
    metrics, convergence, review, dependencies = analyze_verification_relaxations(
        inputs,
        queue,
        cif_root=tmp_path / "cifs",
    )

    assert len(metrics) == 4
    assert convergence["ionic_converged"].all()
    assert metrics["maximum_internal_displacement_A"].max() == pytest.approx(0.0)
    assert review["review_generated_from_outputs"] is True
    assert review["static_launch_authorized"] is True
    assert review["pause_reasons"] == []
    assert all(value["ionic_converged"] is True for value in dependencies.values())

    review_path = tmp_path / "structural_review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    built = build_candidate_static_inputs(
        _static_manifest(),
        dependency_results=dependencies,
        structural_review=review_path,
        relaxation_metrics=metrics,
        output_root=tmp_path / "static",
        git_commit="abc123",
    )

    assert len(built) == 4
    for row in built.itertuples(index=False):
        directory = Path(row.input_dir)
        assert (directory / "POSCAR").read_bytes() == (
            Path(metrics.set_index("job_id").loc[row.dependency_job_id, "source_output_path"])
            / "CONTCAR"
        ).read_bytes()
        incar = (directory / "INCAR").read_text(encoding="utf-8")
        assert "IBRION = -1" in incar
        assert "NSW = 0" in incar
        assert "EDIFFG" not in incar
        assert not (directory / "POTCAR").exists()


@pytest.mark.parametrize("bad_force,shifted_branch", [(True, False), (False, True)])
def test_force_or_distinct_magnetic_structure_branch_blocks_static_launch(
    tmp_path: Path,
    bad_force: bool,
    shifted_branch: bool,
) -> None:
    inputs, queue = _four_jobs(tmp_path, bad_force=bad_force, shifted_branch=shifted_branch)
    metrics, _, review, dependencies = analyze_verification_relaxations(
        inputs,
        queue,
        cif_root=tmp_path / "cifs",
    )
    assert review["static_launch_authorized"] is False
    assert review["pause_reasons"]
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    with pytest.raises(ValueError, match="dependency|structural review"):
        build_candidate_static_inputs(
            _static_manifest(),
            dependency_results=dependencies,
            structural_review=review_path,
            relaxation_metrics=metrics,
            output_root=tmp_path / "static",
            git_commit="abc123",
        )
