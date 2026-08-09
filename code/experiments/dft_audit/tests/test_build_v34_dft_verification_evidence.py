import hashlib
from pathlib import Path

import pandas as pd
import pytest

from analysis.build_v34_dft_verification_evidence import (
    _archive_cifs,
    merge_verification_evidence,
)


def _inputs():
    historical_main = pd.DataFrame(
        [
            {
                "candidate_label": "C044",
                "candidate_id": "job_044_Mn",
                "selected_magnetic_initialization": "state_fm",
                "recomputed_formation_energy_eV_per_atom": -1.87,
            },
            {
                "candidate_label": "C120",
                "candidate_id": "job_120_Cr",
                "selected_magnetic_initialization": "state_fm",
                "recomputed_formation_energy_eV_per_atom": -2.2560,
            },
            {
                "candidate_label": "C214",
                "candidate_id": "job_214_Cr",
                "selected_magnetic_initialization": "state_fm",
                "recomputed_formation_energy_eV_per_atom": -2.2489,
            },
        ]
    )
    historical_energy = pd.DataFrame(
        [
            {
                "job_id": "old120",
                "candidate_id": "job_120_Cr",
                "formula": "LiCr2O4",
                "functional": "GGA+U",
                "magnetic_initialization": "state_fm",
                "selected_for_formation_energy": True,
                "final_total_energy_eV": -49.109,
                "formation_energy_eV_per_atom": -2.2560,
                "source_output_path": "old/120",
                "outcar_sha256": "a" * 64,
            },
            {
                "job_id": "old214",
                "candidate_id": "job_214_Cr",
                "formula": "LiCr2O4",
                "functional": "GGA+U",
                "magnetic_initialization": "state_fm",
                "selected_for_formation_energy": True,
                "final_total_energy_eV": -49.059,
                "formation_energy_eV_per_atom": -2.2489,
                "source_output_path": "old/214",
                "outcar_sha256": "b" * 64,
            },
        ]
    )
    historical_structure = pd.DataFrame(
        [
            {
                "candidate_id": candidate,
                "formula": "LiCr2O4",
                "functional": "GGA+U",
                "magnetic_initialization": state,
                "minimum_interatomic_distance_A": 1.8,
            }
            for candidate in ("job_120_Cr", "job_214_Cr")
            for state in ("state_afm", "state_fm")
        ]
    )
    selected = pd.DataFrame(
        [
            {
                "candidate_id": "C120",
                "new_selected_initialization": "state_fm",
                "new_selected_total_energy_eV": -49.110,
                "new_selected_formation_energy_eV_per_atom": -2.2562,
                "selected_formation_energy_shift_eV_per_atom": -0.0002,
            },
            {
                "candidate_id": "C214",
                "new_selected_initialization": "state_afm",
                "new_selected_total_energy_eV": -49.061,
                "new_selected_formation_energy_eV_per_atom": -2.2492,
                "selected_formation_energy_shift_eV_per_atom": -0.0003,
            },
        ]
    )
    statics = pd.DataFrame(
        [
            {
                "job_id": f"new_{candidate}_{state}",
                "candidate_id": candidate,
                "magnetic_initialization": state,
                "final_total_energy_eV": energy,
                "Fmax_eV_A_static_diagnostic": 0.02,
                "minimum_interatomic_distance_A": 1.9,
                "minimum_M_O_distance_A": 1.9,
                "final_space_group": "P1 (1)",
                "final_total_magnetic_moment": 5.0,
                "source_output_path": f"new/{candidate}/{state}",
                "outcar_sha256": (candidate[-1] + state[-1]) * 32,
                "final_cif_path": f"new/{candidate}/{state}.cif",
                "final_cif_sha256": "c" * 64,
            }
            for candidate, values in {
                "C120": {"state_afm": -49.05, "state_fm": -49.110},
                "C214": {"state_afm": -49.061, "state_fm": -49.06},
            }.items()
            for state, energy in values.items()
        ]
    )
    relaxations = pd.DataFrame(
        [
            {
                "candidate_id": candidate,
                "magnetic_initialization": state,
                "initial_volume_A3": 69.0,
                "final_volume_A3": 69.1,
                "relative_volume_change_percent": 0.145,
                "maximum_internal_displacement_A": 0.03,
                "Fmax_eV_A": 0.04,
                "initial_space_group": "P1 (1)",
                "final_space_group": "P1 (1)",
                "source_output_path": f"relax/{candidate}/{state}",
            }
            for candidate in ("C120", "C214")
            for state in ("state_afm", "state_fm")
        ]
    )
    review = {"paper_conclusion_update_authorized": True, "pause_reasons": []}
    return historical_main, historical_energy, historical_structure, selected, statics, relaxations, review


def test_merge_updates_only_verified_candidates_and_preserves_c044():
    outputs = merge_verification_evidence(*_inputs())
    main = outputs["main_text"].set_index("candidate_label")
    energies = outputs["formation_energies"]

    assert main.loc["C044", "recomputed_formation_energy_eV_per_atom"] == pytest.approx(-1.87)
    assert main.loc["C120", "recomputed_formation_energy_eV_per_atom"] == pytest.approx(-2.2562)
    assert main.loc["C214", "selected_magnetic_initialization"] == "state_afm"
    row = energies.loc[energies["candidate_id"].eq("job_214_Cr")].iloc[0]
    assert row["final_total_energy_eV"] == pytest.approx(-49.061)
    assert row["source_output_path"] == "new/C214/state_afm"


def test_merge_maps_quantitative_relaxation_and_static_metrics():
    outputs = merge_verification_evidence(*_inputs())
    structures = outputs["structure_metrics"]
    row = structures.loc[
        structures["candidate_id"].eq("job_120_Cr")
        & structures["magnetic_initialization"].eq("state_fm")
    ].iloc[0]

    assert row["verification_relaxation_final_volume_A3"] == pytest.approx(69.1)
    assert row["verification_maximum_internal_displacement_A"] == pytest.approx(0.03)
    assert row["minimum_interatomic_distance_A"] == pytest.approx(1.9)
    assert row["Fmax_eV_A_static_diagnostic"] == pytest.approx(0.02)


def test_merge_refuses_a_paused_conclusion_gate():
    inputs = list(_inputs())
    inputs[-1] = {
        "paper_conclusion_update_authorized": False,
        "pause_reasons": ["C214: structural branch"],
    }
    with pytest.raises(ValueError, match="structural branch"):
        merge_verification_evidence(*inputs)


def test_archive_cifs_rewrites_remote_paths_to_verified_local_copies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "recovered"
    source.mkdir()
    cif = source / "c120_state_fm.cif"
    cif.write_text("data_C120\n", encoding="utf-8")
    digest = hashlib.sha256(cif.read_bytes()).hexdigest()
    frame = pd.DataFrame(
        [
            {
                "final_cif_path": "/remote/final_cifs/c120_state_fm.cif",
                "final_cif_sha256": digest,
            }
        ]
    )

    archived = _archive_cifs(
        frame,
        source_root=source,
        output_root=tmp_path / "bundle_cifs",
    )

    local = Path(archived.loc[0, "final_cif_path"])
    assert local.is_file()
    assert local.read_bytes() == cif.read_bytes()
    assert local.parent == (tmp_path / "bundle_cifs").resolve()
