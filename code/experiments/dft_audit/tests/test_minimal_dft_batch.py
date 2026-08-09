from __future__ import annotations

import math

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from analysis.minimal_dft_batch import (
    audit_reference_compatibility,
    build_input_bundle,
    candidate_probe_passes,
    classify_same_scale,
    formation_energy_per_atom,
    kmesh_for_structure,
    render_incar,
    select_frozen_candidates,
)


def _pool() -> pd.DataFrame:
    rows = [
        {
            "candidate_id": "core_mg_1",
            "m_element": "Mg",
            "formula": "Li1 Mg2 O4",
            "alignn_formation_energy_eV_atom": -2.101,
            "alignn_is_core_middle": True,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Mg_1",
            "fingerprint_cluster": "FP_Mg_1",
        },
        {
            "candidate_id": "core_cr_1",
            "m_element": "Cr",
            "formula": "Li1 Cr2 O4",
            "alignn_formation_energy_eV_atom": -2.107,
            "alignn_is_core_middle": True,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Cr_1",
            "fingerprint_cluster": "FP_Cr_1",
        },
        {
            "candidate_id": "core_mg_2",
            "m_element": "Mg",
            "formula": "Li1 Mg2 O4",
            "alignn_formation_energy_eV_atom": -2.114,
            "alignn_is_core_middle": True,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Mg_2",
            "fingerprint_cluster": "FP_Mg_2",
        },
        {
            "candidate_id": "disagreement_mg",
            "m_element": "Mg",
            "formula": "Li1 Mg2 O4",
            "alignn_formation_energy_eV_atom": -1.972,
            "alignn_is_core_middle": False,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Mg_3",
            "fingerprint_cluster": "FP_Mg_3",
        },
        {
            "candidate_id": "random_co_missing_reference",
            "m_element": "Co",
            "formula": "Li1 Co2 O4",
            "alignn_formation_energy_eV_atom": -1.40,
            "alignn_is_core_middle": False,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Co_1",
            "fingerprint_cluster": "FP_Co_1",
        },
        {
            "candidate_id": "random_cr_supported",
            "m_element": "Cr",
            "formula": "Li1 Cr2 O4",
            "alignn_formation_energy_eV_atom": -2.23,
            "alignn_is_core_middle": False,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Cr_2",
            "fingerprint_cluster": "FP_Cr_2",
        },
        {
            "candidate_id": "random_mn_supported",
            "m_element": "Mn",
            "formula": "Li1 Mn2 O4",
            "alignn_formation_energy_eV_atom": -1.93,
            "alignn_is_core_middle": False,
            "historical_dft_candidate": False,
            "cif_exists": True,
            "structure_matcher_cluster": "SM_Mn_1",
            "fingerprint_cluster": "FP_Mn_1",
        },
    ]
    return pd.DataFrame(rows)


def _proposed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"proposal_order": 1, "candidate_id": "core_mg_1", "candidate_stratum": "A_core_ALIGNN"},
            {"proposal_order": 2, "candidate_id": "core_cr_1", "candidate_stratum": "A_core_ALIGNN"},
            {"proposal_order": 3, "candidate_id": "core_mg_2", "candidate_stratum": "A_core_ALIGNN"},
            {
                "proposal_order": 4,
                "candidate_id": "disagreement_mg",
                "candidate_stratum": "D_high_model_disagreement",
            },
            {
                "proposal_order": 5,
                "candidate_id": "random_co_missing_reference",
                "candidate_stratum": "E_random_composition_cluster_control",
            },
        ]
    )


def _references() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"element": "Li", "functional": "PBE", "energy_per_atom_eV": -1.9},
            {"element": "Mg", "functional": "PBE", "energy_per_atom_eV": -1.5},
            {"element": "O", "functional": "PBE", "energy_per_atom_eV": -4.9},
            {"element": "Li", "functional": "GGA+U", "energy_per_atom_eV": -1.9},
            {"element": "Cr", "functional": "GGA+U", "energy_per_atom_eV": -5.8},
            {"element": "Mn", "functional": "GGA+U", "energy_per_atom_eV": -6.5},
            {"element": "O", "functional": "GGA+U", "energy_per_atom_eV": -4.9},
        ]
    )


def test_select_frozen_candidates_keeps_strata_and_replaces_uncomputable_random_control() -> None:
    selected = select_frozen_candidates(_pool(), _proposed(), _references(), random_seed=20260722)

    assert len(selected) == 5
    assert selected["candidate_id"].is_unique
    assert selected["candidate_stratum"].value_counts().to_dict() == {
        "A_core_ALIGNN": 3,
        "D_high_model_disagreement": 1,
        "E_random_composition_cluster_control": 1,
    }
    random_row = selected.loc[
        selected["candidate_stratum"].eq("E_random_composition_cluster_control")
    ].iloc[0]
    assert random_row["m_element"] in {"Cr", "Mn"}
    assert random_row["candidate_id"] != "random_co_missing_reference"
    assert random_row["random_selection_seed"] == 20260722
    assert selected["structure_matcher_cluster"].is_unique


def test_select_frozen_candidates_is_deterministic() -> None:
    first = select_frozen_candidates(_pool(), _proposed(), _references(), random_seed=20260722)
    second = select_frozen_candidates(_pool(), _proposed(), _references(), random_seed=20260722)

    assert first["candidate_id"].tolist() == second["candidate_id"].tolist()


def test_classify_same_scale_remains_unresolved_without_exact_label_reference_convention() -> None:
    metadata = {"dataset": "megnet", "target": "e_form", "checkpoint_sha256": "abc"}

    assert classify_same_scale(metadata) == "UNRESOLVED"


def test_classify_same_scale_requires_explicit_compatible_reference_hash() -> None:
    metadata = {
        "dataset": "megnet",
        "target": "e_form",
        "checkpoint_sha256": "abc",
        "label_reference_convention": "internal-v1",
        "dft_reference_convention": "internal-v1",
        "compatibility_transform_sha256": "def",
    }

    assert classify_same_scale(metadata) == "SAME_SCALE_CONFIRMED"


def test_kmesh_uses_ceiling_of_reciprocal_lengths_over_spacing() -> None:
    structure = Structure(
        Lattice.orthorhombic(5.0, 7.0, 9.0),
        ["Li"],
        [[0, 0, 0]],
    )
    expected = tuple(
        max(1, math.ceil(length / 0.15))
        for length in structure.lattice.reciprocal_lattice.abc
    )

    assert kmesh_for_structure(structure, spacing=0.15) == expected


@pytest.mark.parametrize(
    ("element", "state", "expected"),
    [
        ("Cr", "FM", ("LDAU = .TRUE.", "LDAUU = 0.0 3.7 0.0", "2*5.0")),
        ("Mn", "AFM_or_ferri", ("LDAU = .TRUE.", "LDAUU = 0.0 3.9 0.0", "1*5.0 1*-5.0")),
        ("Mg", "AFM_or_ferri", ("LDAU = .FALSE.", "LORBIT = 11", "2*0.6 2*-0.6")),
    ],
)
def test_render_incar_encodes_frozen_functional_and_magnetic_state(
    element: str, state: str, expected: tuple[str, ...]
) -> None:
    structure = Structure(
        Lattice.cubic(8.0),
        ["Li", element, element, "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75],
            [0.1, 0.1, 0.1],
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
        ],
    )

    text = render_incar(structure, element=element, state=state, stage="relax")

    for token in expected:
        assert token in text
    assert "ENCUT = 520" in text
    assert "EDIFF = 1E-6" in text
    assert "EDIFFG = -0.05" in text
    assert "PREC = Normal" in text
    assert "LREAL = .FALSE." in text
    assert "ISMEAR = 0" in text
    assert "SIGMA = 0.05" in text


def test_formation_energy_recomputes_from_total_and_matching_references() -> None:
    value = formation_energy_per_atom(
        total_energy_eV=-50.0,
        counts={"Li": 1, "Cr": 2, "O": 4},
        references_eV_atom={"Li": -1.9, "Cr": -5.8, "O": -4.9},
    )

    assert value == pytest.approx((-50.0 - (-1.9 - 11.6 - 19.6)) / 7)


def test_formation_energy_rejects_missing_reference() -> None:
    with pytest.raises(ValueError, match="missing elemental references"):
        formation_energy_per_atom(
            total_energy_eV=-50.0,
            counts={"Li": 1, "Co": 2, "O": 4},
            references_eV_atom={"Li": -1.9, "O": -4.9},
        )


def test_build_input_bundle_writes_four_potcar_free_stages_per_candidate(tmp_path) -> None:
    structures = []
    for index, element in enumerate(["Mg", "Cr", "Mg", "Mg", "Cr"], start=1):
        structure = Structure(
            Lattice.cubic(8.0 + 0.1 * index),
            ["Li", element, element, "O", "O", "O", "O"],
            [
                [0, 0, 0],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [0.1, 0.1, 0.1],
                [0.1, 0.9, 0.9],
                [0.9, 0.1, 0.9],
                [0.9, 0.9, 0.1],
            ],
        )
        cif = tmp_path / f"candidate_{index}.cif"
        structure.to(filename=cif, fmt="cif", symprec=None)
        structures.append((element, cif))
    selected = pd.DataFrame(
        [
            {
                "frozen_order": index,
                "candidate_id": f"candidate_{index}",
                "candidate_stratum": (
                    "A_core_ALIGNN"
                    if index <= 3
                    else "D_high_model_disagreement"
                    if index == 4
                    else "E_random_composition_cluster_control"
                ),
                "m_element": element,
                "formula": f"Li1 {element}2 O4",
                "cif_path": str(cif),
                "cif_sha256": "",
                "alignn_formation_energy_eV_atom": -2.1,
            }
            for index, (element, cif) in enumerate(structures, start=1)
        ]
    )
    output = tmp_path / "bundle"

    stage_manifest = build_input_bundle(
        selected,
        output,
        same_scale_status="UNRESOLVED",
    )

    assert len(stage_manifest) == 20
    assert stage_manifest[["candidate_id", "stage", "magnetic_state"]].duplicated().sum() == 0
    assert not list(output.rglob("POTCAR"))
    for row in stage_manifest.itertuples(index=False):
        stage = output / row.relative_stage_dir
        assert (stage / "POSCAR").is_file()
        assert (stage / "INCAR").is_file()
        assert (stage / "KPOINTS").is_file()
        assert (stage / "metadata.json").is_file()
        assert (stage / "README.md").is_file()
        assert "POTCAR" not in (stage / "README.md").read_text(encoding="utf-8").splitlines()[0]
    static_rows = stage_manifest.loc[stage_manifest["stage"].eq("static")]
    assert static_rows["poscar_dependency"].str.endswith("/CONTCAR").all()


def test_candidate_probe_passes_only_when_both_magnetic_branches_complete() -> None:
    good = pd.DataFrame(
        [
            {
                "magnetic_state": state,
                "stage": stage,
                "exit_code": 0,
                "electronic_converged": True,
                "ionic_converged": stage == "relax",
                "nsw_limit_reached": False,
                "structure_collapsed": False,
                "formation_energy_recomputed": stage == "static",
            }
            for state in ("FM", "AFM_or_ferri")
            for stage in ("relax", "static")
        ]
    )

    assert candidate_probe_passes(good)
    broken = good.copy()
    broken.loc[
        broken["magnetic_state"].eq("AFM_or_ferri") & broken["stage"].eq("static"),
        "electronic_converged",
    ] = False
    assert not candidate_probe_passes(broken)


def test_candidate_probe_accepts_documented_nsw_limit_without_hiding_it() -> None:
    rows = pd.DataFrame(
        [
            {
                "magnetic_state": state,
                "stage": stage,
                "exit_code": 0,
                "electronic_converged": True,
                "ionic_converged": False,
                "nsw_limit_reached": stage == "relax",
                "structure_collapsed": False,
                "formation_energy_recomputed": stage == "static",
            }
            for state in ("FM", "AFM_or_ferri")
            for stage in ("relax", "static")
        ]
    )

    assert candidate_probe_passes(rows)


def test_reference_audit_passes_complete_internal_channels() -> None:
    selected = pd.DataFrame(
        [
            {"candidate_id": "cr", "m_element": "Cr"},
            {"candidate_id": "mg", "m_element": "Mg"},
        ]
    )
    references = _references().assign(
        electronic_converged=True,
        Ueff_eV=lambda frame: frame["element"].map({"Cr": 3.7, "Mn": 3.9}).fillna(0.0),
        paw_label=lambda frame: frame["element"].map(
            {
                "Li": "PAW_PBE Li_sv 10Sep2004",
                "Cr": "PAW_PBE Cr_pv 02Aug2007",
                "Mn": "PAW_PBE Mn_pv 02Aug2007",
                "Mg": "PAW_PBE Mg_pv 13Apr2007",
                "O": "PAW_PBE O 08Apr2002",
            }
        ),
    )

    result = audit_reference_compatibility(selected, references)

    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["reference_convention"] == "INTERNAL_SELF_CONSISTENT_PBE_GGA_U"


def test_reference_audit_fails_candidate_without_complete_channel() -> None:
    selected = pd.DataFrame([{"candidate_id": "co", "m_element": "Co"}])

    result = audit_reference_compatibility(selected, _references())

    assert result["status"] == "FAIL"
    assert any("Co" in error for error in result["errors"])
