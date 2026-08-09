from __future__ import annotations

import json
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from analysis.prepare_dft_elemental_reference_jobs import (
    build_elemental_reference_bundle,
)


REFERENCE_ROWS = (
    ("PBE_Li_metal", "Li_metal", "Li", "PBE", None, False),
    ("PBE_Cr_metal", "Cr_metal", "Cr", "PBE", None, False),
    ("PBE_Mn_metal", "Mn_metal", "Mn", "PBE", None, False),
    ("PBE_Mg_metal", "Mg_metal", "Mg", "PBE", None, False),
    ("PBE_O2_molecule", "O2_molecule", "O", "PBE", None, True),
    ("GGA_U_Li_metal", "Li_metal", "Li", "GGA+U", 0.0, False),
    ("GGA_U_Cr_metal", "Cr_metal", "Cr", "GGA+U", 3.7, False),
    ("GGA_U_Mn_metal", "Mn_metal", "Mn", "GGA+U", 3.9, False),
    ("GGA_U_O2_molecule", "O2_molecule", "O", "GGA+U", 0.0, True),
)


def _write_source(path: Path, element: str, *, molecule: bool) -> None:
    path.mkdir(parents=True)
    if molecule:
        structure = Structure(
            Lattice.cubic(15.0),
            ["O", "O"],
            [[0.5, 0.5, 0.46], [0.5, 0.5, 0.54]],
        )
    else:
        structure = Structure(
            Lattice.cubic(3.0),
            [element, element],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )
    structure.to(filename=path / "POSCAR", fmt="poscar")
    (path / "INCAR").write_text("ENCUT = 520\nNSW = 0\n", encoding="utf-8")
    (path / "POTCAR").write_bytes(f"test PAW data for {element}\n".encode())


def _plan(tmp_path: Path) -> list[dict[str, object]]:
    plan = []
    for reference_id, name, element, functional, ueff, molecule in REFERENCE_ROWS:
        source = tmp_path / "sources" / reference_id
        _write_source(source, element, molecule=molecule)
        plan.append(
            {
                "reference_id": reference_id,
                "reference_name": name,
                "element": element,
                "functional": functional,
                "Ueff_eV": ueff,
                "structure": "test structure",
                "magnetic_setup": "test magnetic setup",
                "paw_label": f"PAW_PBE {element}",
                "source_dir": str(source),
                "incar_filename": "INCAR",
                "molecular_gamma_only": molecule,
            }
        )
    return plan


def _protocol(path: Path, *, frozen: bool = True) -> Path:
    path.write_text(
        json.dumps(
            {
                "frozen": frozen,
                "kpoint_rule": "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing",
                "kpoint_spacing_Ainv": 0.15,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_nine_reference_jobs_without_copying_potcar(tmp_path: Path) -> None:
    work = tmp_path / "work"
    manifest = work / "jobs" / "dft_elemental_reference_jobs.csv"

    frame = build_elemental_reference_bundle(
        reference_plan=_plan(tmp_path),
        frozen_protocol_path=_protocol(tmp_path / "protocol.json"),
        work_root=work,
        manifest_path=manifest,
        git_commit="abc123",
        python_executable="/opt/python",
        runner_path=Path("/audit/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/vasp_std"],
    )

    assert len(frame) == 9
    assert set(frame["status"]) == {"PENDING"}
    assert set(frame["functional"]) == {"PBE", "GGA+U"}
    assert not list(work.rglob("POTCAR"))
    assert manifest.is_file()
    assert frame["config_hash"].nunique() == 9

    molecules = frame[frame["reference_name"] == "O2_molecule"]
    metals = frame[frame["reference_name"] != "O2_molecule"]
    assert set(molecules["mesh"]) == {"1x1x1"}
    assert set(molecules["kpoint_basis"]) == {"isolated_molecule_gamma_only_retained_protocol"}
    assert all(int(mesh.split("x")[0]) > 1 for mesh in metals["mesh"])
    assert set(metals["kpoint_basis"]) == {"frozen_reciprocal_spacing_rule"}

    for row in frame.to_dict(orient="records"):
        command = json.loads(row["command_json"])
        assert "--potcar-source" in command
        assert command[command.index("--potcar-source") + 1].endswith("POTCAR")
        provenance = json.loads(Path(row["input_provenance_path"]).read_text())
        assert provenance["potcar_content_retained_in_generated_inputs"] is False
        assert len(provenance["POTCAR_sha256"]) == 64


def test_explicit_structure_filename_is_copied_and_recorded(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    record = next(row for row in plan if row["reference_id"] == "GGA_U_Cr_metal")
    source = Path(str(record["source_dir"]))
    historical = Structure(
        Lattice.cubic(4.0),
        ["Cr", "Cr"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    historical.to(filename=source / "CONTCAR", fmt="poscar")
    record["structure_filename"] = "CONTCAR"
    work = tmp_path / "work"

    frame = build_elemental_reference_bundle(
        reference_plan=plan,
        frozen_protocol_path=_protocol(tmp_path / "protocol.json"),
        work_root=work,
        manifest_path=work / "jobs.csv",
        git_commit="abc123",
        python_executable="/opt/python",
        runner_path=Path("/audit/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/vasp_std"],
    )

    row = frame.set_index("reference_id").loc["GGA_U_Cr_metal"]
    copied = Structure.from_file(work / "inputs" / "GGA_U_Cr_metal" / "POSCAR")
    provenance = json.loads(Path(row["input_provenance_path"]).read_text())
    assert copied.volume == pytest.approx(historical.volume)
    assert row["structure_source_filename"] == "CONTCAR"
    assert provenance["source_structure_filename"] == "CONTCAR"
    assert provenance["source_structure_path"] == str((source / "CONTCAR").resolve())


def test_selected_reference_ids_build_only_requested_jobs(tmp_path: Path) -> None:
    work = tmp_path / "work"

    frame = build_elemental_reference_bundle(
        reference_plan=_plan(tmp_path),
        frozen_protocol_path=_protocol(tmp_path / "protocol.json"),
        work_root=work,
        manifest_path=work / "jobs.csv",
        git_commit="abc123",
        python_executable="/opt/python",
        runner_path=Path("/audit/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/vasp_std"],
        selected_reference_ids=["GGA_U_Cr_metal", "GGA_U_Mn_metal"],
    )

    assert frame["reference_id"].tolist() == ["GGA_U_Cr_metal", "GGA_U_Mn_metal"]
    assert len(list((work / "inputs").iterdir())) == 2


def test_refuses_unfrozen_protocol(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="frozen DFT protocol"):
        build_elemental_reference_bundle(
            reference_plan=_plan(tmp_path),
            frozen_protocol_path=_protocol(tmp_path / "protocol.json", frozen=False),
            work_root=tmp_path / "work",
            manifest_path=tmp_path / "work" / "jobs.csv",
            git_commit="abc123",
            python_executable="/opt/python",
            runner_path=Path("/audit/run_vasp_benchmark_task.py"),
            vasp_command=["/licensed/vasp_std"],
        )


def test_refuses_incomplete_reference_plan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly the nine retained reference protocols"):
        build_elemental_reference_bundle(
            reference_plan=_plan(tmp_path)[:-1],
            frozen_protocol_path=_protocol(tmp_path / "protocol.json"),
            work_root=tmp_path / "work",
            manifest_path=tmp_path / "work" / "jobs.csv",
            git_commit="abc123",
            python_executable="/opt/python",
            runner_path=Path("/audit/run_vasp_benchmark_task.py"),
            vasp_command=["/licensed/vasp_std"],
        )
