from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis.prepare_dft_candidate_static_jobs import (
    build_candidate_static_bundle,
    build_required_static_records,
    load_vasp_command,
)


def _write_static_source(path: Path, *, magmom: str = "1*0.6 2*5.0 4*0.6") -> None:
    path.mkdir(parents=True)
    (path / "INCAR").write_text(
        "ENCUT = 520\nEDIFF = 1E-6\nNSW = 0\nIBRION = -1\n"
        f"ISPIN = 2\nMAGMOM = {magmom}\n",
        encoding="utf-8",
    )
    (path / "POSCAR").write_text(
        "fixture\n1\n4 0 0\n0 5 0\n0 0 6\nLi Cr O\n1 2 4\nDirect\n"
        "0 0 0\n0.2 0.2 0.2\n0.4 0.4 0.4\n0.1 0.1 0.1\n"
        "0.3 0.3 0.3\n0.5 0.5 0.5\n0.7 0.7 0.7\n",
        encoding="utf-8",
    )
    (path / "POTCAR").write_bytes(b"licensed-fixture")


def _candidate(root: Path, folder: str, candidate_id: str) -> Path:
    path = root / "candidate_inputs" / folder
    path.mkdir(parents=True)
    (path / "candidate_metadata.json").write_text(
        json.dumps({"candidate_id": candidate_id}) + "\n", encoding="utf-8"
    )
    return path


def _tight_candidate(root: Path, folder: str, candidate_id: str) -> Path:
    path = root / "shortlist_tight" / "shortlist_inputs" / folder
    path.mkdir(parents=True)
    (path / "candidate_metadata.json").write_text(
        json.dumps({"candidate_id": candidate_id}) + "\n", encoding="utf-8"
    )
    return path


def test_required_records_use_two_tested_states_only_for_main_candidates(tmp_path: Path) -> None:
    source = tmp_path / "new12"
    main = _candidate(source, "candidate_001_main", "main")
    mg = _candidate(source, "candidate_002_mg", "mg")
    selected = _candidate(source, "candidate_003_selected", "selected")
    for candidate in (main, mg, selected):
        _write_static_source(candidate / "stages" / "02_pbe_static")
    _write_static_source(selected / "stages" / "04_gga_u_static")

    main_tight = _tight_candidate(source, "candidate_001_main", "main")
    selected_tight = _tight_candidate(source, "candidate_003_selected", "selected")
    for state in ("state_fm", "state_afm"):
        _write_static_source(main_tight / "magnetic_states" / state / "02_static")
    _write_static_source(selected_tight / "magnetic_states" / "state_fm" / "02_static")

    manifest = pd.DataFrame(
        [
            {"candidate_id": "main", "pilot_or_new": "new", "formula": "LiCr2O4", "DFT_status": "static_finished", "main_text_selected": "True"},
            {"candidate_id": "mg", "pilot_or_new": "new", "formula": "LiMg2O4", "DFT_status": "static_finished", "main_text_selected": "False"},
            {"candidate_id": "selected", "pilot_or_new": "new", "formula": "LiMn2O4", "DFT_status": "static_finished", "main_text_selected": "False"},
            {"candidate_id": "failed", "pilot_or_new": "new", "formula": "LiMn2O4", "DFT_status": "failed", "main_text_selected": "False"},
        ]
    )
    magnetic = pd.DataFrame(
        [
            {"candidate_id": "main", "initialization_label_as_recorded": "state_fm", "selected_lower_energy_among_two": "True"},
            {"candidate_id": "main", "initialization_label_as_recorded": "state_afm", "selected_lower_energy_among_two": "False"},
            {"candidate_id": "selected", "initialization_label_as_recorded": "state_fm", "selected_lower_energy_among_two": "True"},
        ]
    )

    records = build_required_static_records(manifest, magnetic, source)
    keys = {(row["candidate_id"], row["functional"], row["magnetic_initialization"]) for row in records}

    assert keys == {
        ("main", "PBE", "default"),
        ("main", "GGA+U", "state_fm"),
        ("main", "GGA+U", "state_afm"),
        ("mg", "PBE", "default"),
        ("selected", "PBE", "default"),
        ("selected", "GGA+U", "state_fm"),
    }


def test_bundle_reuses_only_exact_frozen_protocol_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_static_source(source)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "frozen": True,
                "kpoint_rule": "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing",
                "kpoint_spacing_Ainv": 0.15,
            }
        ),
        encoding="utf-8",
    )
    existing = tmp_path / "existing"
    existing.mkdir()
    input_manifest = {
        name: {
            "sha256": hashlib.sha256((source / name).read_bytes()).hexdigest()
        }
        for name in ("INCAR", "POSCAR", "POTCAR")
    }
    # 2*pi/4, 2*pi/5, 2*pi/6 at spacing 0.15 -> 11 x 9 x 7.
    kpoints = existing / "KPOINTS"
    kpoints.write_text("frozen\n0\nGamma\n11 9 7\n", encoding="utf-8")
    input_manifest["KPOINTS"] = {"sha256": hashlib.sha256(kpoints.read_bytes()).hexdigest()}
    (existing / "input_manifest.json").write_text(json.dumps(input_manifest), encoding="utf-8")
    (existing / "task_result.json").write_text(
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
    (existing / "OUTCAR").write_text("General timing and accounting\n", encoding="utf-8")

    records = [
        {
            "candidate_id": "candidate_a",
            "formula": "LiCr2O4",
            "functional": "GGA+U",
            "magnetic_initialization": "state_fm",
            "source_dir": str(source),
            "main_text_selected": True,
        }
    ]
    reuse = {"candidate_a|GGA+U|state_fm": str(existing)}
    manifest, compatibility = build_candidate_static_bundle(
        records=records,
        reuse_outputs=reuse,
        frozen_protocol_path=protocol,
        work_root=tmp_path / "work",
        manifest_path=tmp_path / "jobs.csv",
        compatibility_path=tmp_path / "compatibility.csv",
        git_commit="abc123",
        python_executable="python3",
        runner_path=Path("/runner.py"),
        vasp_command=["vasp_std"],
    )

    assert manifest.empty
    assert compatibility.loc[0, "decision"] == "REUSED_FROZEN_PROTOCOL_OUTPUT"
    assert compatibility.loc[0, "required_mesh"] == "11x9x7"
    assert not list((tmp_path / "work").rglob("POTCAR"))


def test_bundle_generates_pending_job_without_copying_potcar(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_static_source(source)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "frozen": True,
                "kpoint_rule": "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing",
                "kpoint_spacing_Ainv": 0.15,
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "candidate_id": "candidate_a",
            "formula": "LiCr2O4",
            "functional": "PBE",
            "magnetic_initialization": "default",
            "source_dir": str(source),
            "main_text_selected": False,
        }
    ]

    manifest, compatibility = build_candidate_static_bundle(
        records=records,
        reuse_outputs={},
        frozen_protocol_path=protocol,
        work_root=tmp_path / "work",
        manifest_path=tmp_path / "jobs.csv",
        compatibility_path=tmp_path / "compatibility.csv",
        git_commit="abc123",
        python_executable="python3",
        runner_path=Path("/runner.py"),
        vasp_command=["vasp_std"],
    )

    assert len(manifest) == 1
    assert manifest.loc[0, "status"] == "PENDING"
    assert manifest.loc[0, "K"] == "11x9x7"
    assert compatibility.loc[0, "decision"] == "REQUIRES_STATIC_VERIFICATION"
    assert not list((tmp_path / "work").rglob("POTCAR"))


def test_vasp_command_can_be_loaded_from_a_file(tmp_path: Path) -> None:
    command_file = tmp_path / "vasp_command.json"
    command_file.write_text('["/licensed/server/vasp_std"]\n', encoding="utf-8")

    assert load_vasp_command(command_json=None, command_file=command_file) == [
        "/licensed/server/vasp_std"
    ]
