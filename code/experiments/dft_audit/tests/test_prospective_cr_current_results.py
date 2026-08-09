from __future__ import annotations

import pandas as pd

from analysis.collect_prospective_cr_current import build_result_tables


def _stage(energy: float, fmax: float, moment: float) -> dict:
    return {
        "status": "PASS",
        "attempt_count": 1,
        "parsed": {
            "energy_eV": energy,
            "energy_per_atom_eV": energy / 7,
            "fmax_eV_A": fmax,
            "stress_kbar": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "total_magnetic_moment_muB": moment,
            "local_magnetic_moments_muB": [0.0, 2.0, 2.0, 0, 0, 0, 0],
            "band_gap_eV": 0.4,
            "metallic": False,
            "minimum_interatomic_distance_A": 1.8,
            "volume_change_percent": -3.0,
            "structure_collapse": False,
            "electronic_converged": True,
            "ionic_converged": True,
            "entropy_term_eV_cell": 0.001,
        },
    }


def test_complete_candidate_is_dft_evaluated_without_scale_overclaim() -> None:
    candidate_id = "candidate-a"
    state = {
        "entities": {
            candidate_id: {
                "status": "PASS",
                "relax": _stage(-99, 0.04, 4.1),
                "static_0p15": _stage(-100, 0.01, 4.0),
            }
        }
    }
    manifest = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "alignn_formation_energy_eV_atom": -2.1,
                "gate_round": 2,
                "greedy_round": 7,
                "cif_sha256": "abc",
            }
        ]
    )
    checks = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "formation_energy_primary_eV_atom": -2.3,
                "formation_energy_decimal_eV_atom": -2.3,
                "absolute_difference_eV_atom": 0.0,
                "roundtrip_pass": True,
            }
        ]
    )

    results, magnetic = build_result_tables(state, manifest, checks)

    assert results.iloc[0]["result_level"] == "DFT_EVALUATED"
    assert (
        results.iloc[0]["self_consistent_fe_target_status"]
        == "NOT_ASSESSED_SAME_SCALE_UNRESOLVED"
    )
    assert bool(results.iloc[0]["gate_precedes_greedy"])
    assert magnetic.iloc[0]["magnetic_branch"] == "MP_STANDARD_INITIALIZATION_ONLY"


def test_failed_candidate_is_preserved_without_energy_claim() -> None:
    state = {
        "entities": {
            "candidate-b": {
                "status": "FAIL",
                "failure_reason": "PERSISTENT_NUMERICAL_OR_CONVERGENCE_FAILURE",
            }
        }
    }
    manifest = pd.DataFrame(
        [
            {
                "candidate_id": "candidate-b",
                "alignn_formation_energy_eV_atom": -2.1,
                "gate_round": 8,
                "greedy_round": 1,
                "cif_sha256": "def",
            }
        ]
    )

    results, magnetic = build_result_tables(state, manifest, pd.DataFrame())

    assert results.iloc[0]["result_level"] == "NOT_DFT_EVALUATED"
    assert pd.isna(results.iloc[0]["formation_energy_eV_atom"])
    assert len(magnetic) == 0
