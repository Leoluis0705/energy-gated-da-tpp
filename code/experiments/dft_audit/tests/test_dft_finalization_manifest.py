from __future__ import annotations

from pathlib import Path

from analysis.build_dft_finalization_manifest import (
    BASE_DFT_PROTOCOL_SHA256,
    build_manifest,
    build_report,
)


def test_manifest_has_exact_staged_vasp_scope() -> None:
    archive = Path(__file__).resolve().parents[1]
    manifest = build_manifest(archive)

    assert len(manifest) == 13
    assert manifest["job_id"].is_unique
    assert manifest["task_group"].value_counts().to_dict() == {
        "main_candidate_verification_static": 4,
        "main_candidate_verification_relaxation": 4,
        "alpha_mn_reference_sensitivity": 4,
        "alpha_mn_cost_probe": 1,
    }
    assert set(manifest["status"]) == {"PENDING"}
    assert set(manifest["base_protocol_sha256"]) == {BASE_DFT_PROTOCOL_SHA256}
    assert set(manifest["openblas_threads"]) == {8}


def test_alpha_mn_scope_is_source_backed_and_non_exhaustive() -> None:
    archive = Path(__file__).resolve().parents[1]
    alpha = build_manifest(archive).query("candidate_id == 'alpha_Mn'")

    assert len(alpha) == 5
    assert set(alpha["expected_mesh"]) == {"5x5x5"}
    assert set(alpha["atom_count"]) == {58}
    assert set(alpha["structure_source_id"]) == {"MAGNDATA_1.85"}
    assert set(alpha["structure_source_doi"]) == {"10.1063/1.358024"}
    assert set(alpha["magnetic_initialization"]) == {
        "magndata_1p85_noncollinear",
        "magndata_1p85_collinear_z_projection",
    }
    noncollinear = alpha.query("LNONCOLLINEAR == True")
    collinear = alpha.query("LNONCOLLINEAR == False")
    assert noncollinear["MAGMOM_definition"].str.contains("Mx,My,Mz", regex=False).all()
    assert collinear["MAGMOM_definition"].str.contains("Mz component", regex=False).all()
    assert not alpha["scientific_scope"].str.contains(
        "ground state|ground-state|exhaustive", case=False, regex=True
    ).any()


def test_probe_is_non_scientific_and_gates_full_alpha_runs() -> None:
    archive = Path(__file__).resolve().parents[1]
    manifest = build_manifest(archive)
    probe = manifest.query("task_group == 'alpha_mn_cost_probe'").iloc[0]
    full = manifest.query("task_group == 'alpha_mn_reference_sensitivity'")

    assert not bool(probe["scientific_result"])
    assert probe["NELM"] == 1
    assert probe["concurrency_cap"] == 1
    assert set(full["dependency_job_ids"]) == {probe["job_id"]}
    assert set(full["launch_gate"]) == {"MEASURED_ALPHA_COST_AND_MEMORY_ACCEPTED"}


def test_candidate_relaxations_use_archived_state_specific_structures() -> None:
    archive = Path(__file__).resolve().parents[1]
    relax = build_manifest(archive).query(
        "task_group == 'main_candidate_verification_relaxation'"
    )

    assert set(relax["candidate_id"]) == {"C120", "C214"}
    assert set(relax["magnetic_initialization"]) == {"state_fm", "state_afm"}
    assert set(relax["MAGMOM_definition"]) == {
        "1*0.6 2*5.0 4*0.6",
        "1*0.6 1*5.0 1*-5.0 4*0.6",
    }
    assert set(relax["calculation_type"]) == {"verification_relaxation"}
    assert set(relax["functional"]) == {"GGA+U"}
    assert relax["input_structure_path"].str.contains(
        "tight_magnetic_states", regex=False
    ).all()
    assert relax["input_structure_path"].str.endswith("CONTCAR").all()
    assert relax["input_structure_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    for raw_path in relax["input_structure_path"]:
        assert (archive / raw_path).is_file()


def test_candidate_statics_depend_on_matching_relaxations() -> None:
    archive = Path(__file__).resolve().parents[1]
    manifest = build_manifest(archive)
    relax_ids = set(
        manifest.query("task_group == 'main_candidate_verification_relaxation'")[
            "job_id"
        ]
    )
    statics = manifest.query("task_group == 'main_candidate_verification_static'")

    assert set(statics["dependency_job_ids"]) == relax_ids
    assert set(statics["input_structure_path"]) == {
        "DEPENDENCY_OUTPUT:CONTCAR"
    }
    assert set(statics["launch_gate"]) == {"RELAXATION_CONVERGED_WITHOUT_STOP_CONDITION"}


def test_manifest_never_contains_restricted_paw_content_or_credentials() -> None:
    archive = Path(__file__).resolve().parents[1]
    rendered = build_manifest(archive).to_csv(index=False).lower()

    assert "titel  =" not in rendered
    assert "begin_potcar" not in rendered
    assert "yet54si2vcku" not in rendered
    assert "23196" not in rendered


def test_report_declares_cost_uncertainty_and_stop_gate() -> None:
    archive = Path(__file__).resolve().parents[1]
    manifest = build_manifest(archive)
    report = build_report(manifest)

    assert "13 logical VASP tasks" in report
    assert "not a scientific result" in report
    assert "0.02 eV/atom" in report
    assert "two tested magnetic initializations" in report
    assert "server has not been contacted" in report
