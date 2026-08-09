from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from analysis.audit_current_potcars_remote import evaluate_potcar_inventory
from analysis.compute_self_consistent_fe import compute_formation_energy
from analysis.prospective_cr_current import (
    EXPECTED_CANDIDATE_IDS,
    build_candidate_preflight,
    build_input_package,
    kmesh_for_structure,
)
from analysis.recompute_self_consistent_fe_decimal import (
    recompute_formation_energy_decimal,
)
from analysis.run_prospective_cr_current_remote import choose_restart_geometry


REPO_ROOT = Path(__file__).resolve().parents[1]

FROZEN_REMOTE_FIXTURE = {
    "host": "connect.westb.seetacloud.com",
    "port": 41058,
    "host_key_sha256": "SHA256:liZ36vNCsNcNdXeWs4f+g5ZIhPM/ZihP834vxs8Ulqc",
    "license_scope": "USER_AUTHORIZED_LICENSED_VASP_SERVER",
    "library": {
        "path": "/root/software/potpaw_PBE",
        "exists": True,
        "source_class": "server-installed VASP PAW-PBE library",
    },
    "vasp": {
        "version": "6.5.1",
        "path": "/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std",
        "sha256": "2abdcfedd1c3e7962a56404bd14cc340dcb170867720921fcea8ec7058ef3d94",
        "exists": True,
        "executable": True,
    },
    "potcars": {
        "Li": {
            "titel": "PAW_PBE Li_sv 10Sep2004",
            "lexch": "PE",
            "zval": 3.0,
            "enmax_eV": 499.034,
            "sha256": (
                "201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b"
            ),
            "path": "/root/software/potpaw_PBE/Li_sv/POTCAR",
            "size_bytes": 144229,
            "functional": "PBE",
            "potential_type": "PAW",
        },
        "Cr": {
            "titel": "PAW_PBE Cr_pv 02Aug2007",
            "lexch": "PE",
            "zval": 12.0,
            "enmax_eV": 265.681,
            "sha256": (
                "836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b"
            ),
            "path": "/root/software/potpaw_PBE/Cr_pv/POTCAR",
            "size_bytes": 234590,
            "functional": "PBE",
            "potential_type": "PAW",
        },
        "O": {
            "titel": "PAW_PBE O 08Apr2002",
            "lexch": "PE",
            "zval": 6.0,
            "enmax_eV": 400.0,
            "sha256": (
                "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64"
            ),
            "path": "/root/software/potpaw_PBE/O/POTCAR",
            "size_bytes": 220583,
            "functional": "PBE",
            "potential_type": "PAW",
        },
    },
}


def test_preflight_accepts_exact_five_independent_cr_structures() -> None:
    frame, audit = build_candidate_preflight(REPO_ROOT)

    assert frame["candidate_id"].tolist() == EXPECTED_CANDIDATE_IDS
    assert audit["selected_pair_matches"] == []
    assert audit["historical_matches"] == []
    assert set(frame["formula"]) == {"Li1 Cr2 O4"}
    assert (frame["minimum_interatomic_distance_A"] > 1.2).all()
    assert (frame["lattice_determinant_A3"].abs() > 1.0).all()
    assert not frame["candidate_id"].str.contains("_Mg_").any()


def test_modern_potcar_gate_uses_pbe_metadata_and_649_ev_cutoff() -> None:
    audit = evaluate_potcar_inventory(FROZEN_REMOTE_FIXTURE)

    assert audit["status"] == "PASS"
    assert audit["encut_formula_eV"] == pytest.approx(648.7442)
    assert audit["encut_eV"] == 649
    assert {row["lexch"] for row in audit["potcars"].values()} == {"PE"}
    assert {row["functional"] for row in audit["potcars"].values()} == {"PBE"}
    assert {row["potential_type"] for row in audit["potcars"].values()} == {"PAW"}


def test_potcar_gate_fails_on_one_hash_drift() -> None:
    fixture = json.loads(json.dumps(FROZEN_REMOTE_FIXTURE))
    fixture["potcars"]["Cr"]["sha256"] = "0" * 64

    audit = evaluate_potcar_inventory(fixture)

    assert audit["status"] == "FAIL"
    assert "Cr:sha256_mismatch" in audit["failure_reasons"]


def test_package_has_references_then_exact_candidate_jobs_and_no_potcar(
    tmp_path: Path,
) -> None:
    audit = evaluate_potcar_inventory(FROZEN_REMOTE_FIXTURE)
    jobs = build_input_package(
        REPO_ROOT,
        tmp_path / "package",
        potcar_audit=audit,
    )

    assert set(jobs["job_kind"]) == {"reference", "candidate"}
    assert set(jobs["ENCUT_eV"]) == {649}
    assert set(jobs.loc[jobs["element"].eq("Cr"), "Ueff_eV"]) == {3.7}
    assert not any(path.name == "POTCAR" for path in tmp_path.rglob("*"))
    o2_incar = (
        tmp_path
        / "package"
        / "inputs"
        / "reference_O_relax"
        / "INCAR"
    ).read_text(encoding="utf-8")
    assert "MAGMOM = 2*1.0" in o2_incar
    assert jobs.loc[jobs["candidate_id"].ne(""), "candidate_id"].drop_duplicates().tolist() == (
        EXPECTED_CANDIDATE_IDS
    )
    assert set(
        jobs.loc[jobs["candidate_id"].eq(EXPECTED_CANDIDATE_IDS[0]), "stage"]
    ) == {"relax", "static_0p15", "static_0p10"}
    assert len(jobs) == 17


def test_kmesh_obeys_maximum_reciprocal_spacing() -> None:
    structure = Structure(
        Lattice.orthorhombic(4.0, 5.0, 6.0),
        ["Li"],
        [[0, 0, 0]],
    )

    mesh = kmesh_for_structure(structure, spacing_Ainv=0.15)
    reciprocal_lengths = structure.lattice.reciprocal_lattice.abc

    assert all(length / points <= 0.15 for length, points in zip(reciprocal_lengths, mesh))


def test_timeout_continuation_uses_latest_valid_contcar(tmp_path: Path) -> None:
    attempt_1 = tmp_path / "attempt_1"
    attempt_2 = tmp_path / "attempt_2"
    attempt_1.mkdir()
    attempt_2.mkdir()
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li"],
        [[0, 0, 0]],
    )
    structure.to(filename=attempt_1 / "CONTCAR", fmt="poscar")
    structure.to(filename=attempt_2 / "CONTCAR", fmt="poscar")

    chosen = choose_restart_geometry([attempt_1, attempt_2])

    assert chosen == attempt_2 / "CONTCAR"


def test_continuation_ignores_empty_latest_contcar(tmp_path: Path) -> None:
    attempt_1 = tmp_path / "attempt_1"
    attempt_2 = tmp_path / "attempt_2"
    attempt_1.mkdir()
    attempt_2.mkdir()
    structure = Structure(
        Lattice.cubic(4.0),
        ["Li"],
        [[0, 0, 0]],
    )
    structure.to(filename=attempt_1 / "CONTCAR", fmt="poscar")
    (attempt_2 / "CONTCAR").write_text("", encoding="utf-8")

    chosen = choose_restart_geometry([attempt_1, attempt_2])

    assert chosen == attempt_1 / "CONTCAR"


def test_independent_fe_implementations_match_hand_derived_case() -> None:
    row = {
        "candidate_total_energy_eV": -100.0,
        "Li_energy_per_atom_eV": -1.0,
        "Cr_energy_per_atom_eV": -5.0,
        "O2_energy_per_molecule_eV": -10.0,
    }

    first = compute_formation_energy(row)
    second = recompute_formation_energy_decimal(row)

    assert first == pytest.approx(-69.0 / 7.0)
    assert second == pytest.approx(-69.0 / 7.0)
    assert abs(first - second) <= 1e-12


def test_manifest_roundtrip_writes_no_old_energy_columns(tmp_path: Path) -> None:
    audit = evaluate_potcar_inventory(FROZEN_REMOTE_FIXTURE)
    jobs = build_input_package(
        REPO_ROOT,
        tmp_path / "package",
        potcar_audit=audit,
    )
    written = pd.read_csv(tmp_path / "package/job_manifest.csv")

    assert written["job_id"].tolist() == jobs["job_id"].tolist()
    assert "old_probe_energy_eV" not in written.columns
    assert set(written["same_scale_status"]) == {"UNRESOLVED"}
