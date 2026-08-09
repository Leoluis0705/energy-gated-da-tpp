from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analysis.extract_dft_audit import build_dft_audit_tables


@pytest.fixture(scope="module")
def tables():
    archive = Path(__file__).resolve().parents[2]
    return build_dft_audit_tables(archive)


def test_settings_cover_inputs_and_stage_specific_outputs(tables) -> None:
    settings = tables.settings
    required = {
        "candidate_id",
        "cohort",
        "calculation_id",
        "stage_role",
        "functional",
        "vasp_version",
        "paw_labels",
        "element_order",
        "ENCUT",
        "kpoints_mesh",
        "KSPACING",
        "EDIFF",
        "EDIFFG",
        "ISMEAR",
        "SIGMA",
        "ISPIN",
        "MAGMOM",
        "LDAU",
        "LDAUL",
        "LDAUU",
        "LDAUJ",
        "LASPH",
        "LMAXMIX",
        "NSW",
        "IBRION",
        "ISIF",
        "incar_path",
        "kpoints_path",
        "outcar_path",
        "vasprun_path",
    }
    assert required.issubset(settings.columns)
    assert len(settings) >= 80

    pbe = settings.query(
        "candidate_id == 'job_120_Cr_fe_-1.424_n4_generated_crystals_cif__gen_1' "
        "and calculation_id == '01_pbe_relax'"
    ).iloc[0]
    assert pbe["ENCUT"] == pytest.approx(520)
    assert pbe["EDIFFG"] == pytest.approx(-0.08)
    assert pbe["ISPIN"] == 2
    assert pbe["vasp_version"] == "6.5.1"
    assert pbe["LDAU"] is False or pbe["LDAU"] == False  # noqa: E712
    assert "Cr_pv" in pbe["paw_labels"]

    gga = settings.query(
        "candidate_id == 'job_120_Cr_fe_-1.424_n4_generated_crystals_cif__gen_1' "
        "and calculation_id == '03_gga_u_relax'"
    ).iloc[0]
    assert gga["LDAU"] is True or gga["LDAU"] == True  # noqa: E712
    assert "3.7" in str(gga["LDAUU"])
    assert gga["LMAXMIX"] == 4


def test_pilot_missing_relax_outputs_are_explicit(tables) -> None:
    settings = tables.settings
    pilot_relax = settings.query("cohort == 'pilot' and stage_role == 'relax'")
    primary = pilot_relax.query("functional == 'PBE'")
    assert len(primary) == 8
    assert not primary["outcar_available"].any()
    assert set(primary["output_provenance"]) == {
        "original_relaxation_artifact_unavailable; relax.log retained"
    }


def test_convergence_is_derived_per_stage_and_failures_are_retained(tables) -> None:
    convergence = tables.convergence
    failed_new = convergence.query(
        "candidate_id in ['job_148_Mn_fe_-0.904_n4_generated_crystals_cif__gen_2', "
        "'job_167_Mn_fe_-0.885_n4_generated_crystals_cif__gen_3'] "
        "and calculation_id == '02_pbe_static'"
    )
    assert len(failed_new) == 2
    assert not failed_new["electronic_converged"].any()

    failed_pilot = convergence.query(
        "candidate_id == 'job_182_Cr_fe_-1.464_n4_generated_crystals_cif__gen_2' "
        "and calculation_id == 'pilot_pbe_static'"
    ).iloc[0]
    assert not bool(failed_pilot["electronic_converged"])
    assert "electronic" in failed_pilot["failure_reason"]

    missing_new = convergence.query("cohort == 'new' and not outcar_available")
    assert len(missing_new) == 4
    assert set(missing_new["failure_reason"]) == {
        "stage_output_unavailable_in_archived_job_bundle"
    }
    assert set(missing_new["output_provenance"]) == {
        "stage inputs retained; output unavailable in archived job bundle"
    }


def test_structure_metrics_are_candidate_level_and_quantitative(tables) -> None:
    structures = tables.structures
    assert len(structures) == 20
    assert structures["candidate_id"].is_unique
    assert structures["initial_volume_A3"].notna().all()
    assert structures["final_volume_A3"].notna().all()
    assert structures["relative_volume_change_percent"].notna().all()
    assert (structures["minimum_interatomic_distance_A"] > 0).all()
    assert structures["minimum_M_O_distance_A"].notna().all()
    assert structures["Fmax_eV_A"].notna().all()
    assert structures["final_total_energy_eV"].notna().all()
    assert structures["final_space_group"].notna().all()
    assert set(structures["metric_definition"]) == {
        "periodic minimum-image distances; Fmax from relaxation output when available, "
        "otherwise final static output; energy from final static output"
    }
    new = structures.query("cohort == 'new'")
    assert set(new["Fmax_source"]) == {"final relaxation OUTCAR"}
    pilots = structures.query("cohort == 'pilot'")
    assert set(pilots["Fmax_source"]) == {
        "final static OUTCAR; original relaxation OUTCAR unavailable"
    }


def test_magnetic_table_has_two_raw_initializations_per_candidate(tables) -> None:
    magnetic = tables.magnetic
    assert len(magnetic) == 8
    assert set(magnetic.groupby("candidate_id").size()) == {2}
    assert set(magnetic["scope_statement"]) == {"two tested magnetic initializations"}
    selected = magnetic[magnetic["selected_lower_energy_among_two"]]
    assert len(selected) == 4
    assert set(selected["selection_statement"]) == {
        "lower-energy configuration among the two tested initializations"
    }
    for _, group in magnetic.groupby("candidate_id"):
        assert group.loc[group["selected_lower_energy_among_two"], "total_energy_eV"].iloc[0] == pytest.approx(
            group["total_energy_eV"].min()
        )


def test_elemental_references_are_reextracted_from_raw_outputs(tables) -> None:
    references = tables.references
    assert len(references) == 9
    assert set(references.query("functional == 'PBE'")["element"]) == {"Li", "Cr", "Mn", "Mg", "O"}
    assert set(references.query("functional == 'GGA+U'")["element"]) == {"Li", "Cr", "Mn", "O"}
    assert references["raw_output_path"].str.endswith("OUTCAR").all()
    assert references["total_energy_eV"].notna().all()
    assert references["atoms_per_cell"].gt(0).all()
    assert references["energy_per_atom_eV"].notna().all()
    cr = references.query("functional == 'PBE' and element == 'Cr'").iloc[0]
    assert cr["total_energy_eV"] == pytest.approx(-19.0297743, abs=1e-8)
    assert cr["energy_per_atom_eV"] == pytest.approx(-9.51488715, abs=1e-8)


def test_no_table_contains_prohibited_magnetic_claims(tables) -> None:
    for frame in (
        tables.settings,
        tables.convergence,
        tables.structures,
        tables.magnetic,
        tables.references,
    ):
        text = frame.to_csv(index=False).lower()
        assert "ground state" not in text
        assert "exhaustive magnetic" not in text
