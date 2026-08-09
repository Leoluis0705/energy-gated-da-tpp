from __future__ import annotations

from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

from analysis.run_minimal_dft_remote import (
    candidate_gate_passes,
    formation_energy_eV_atom,
    parse_oszicar_text,
    parse_outcar_text,
    structure_metrics,
)


def test_parse_outcar_collects_completion_convergence_forces_and_moments() -> None:
    text = """
 vasp.6.5.1 10Mar25
 aborting loop because EDIFF is reached
 reached required accuracy - stopping structural energy minimisation
 free  energy   TOTEN  =       -42.500000 eV
 entropy T*S    EENTRO =         0.001400 eV
 TOTAL-FORCE (eV/Angst)
 -------------------------------------------------------------------
  0 0 0  0.01 0.02 0.02
  0 0 0  0.00 0.00 0.04
 -------------------------------------------------------------------
 magnetization (x)
 # of ion       s       p       d       tot
 --------------------------------------------------
     1        0.0     0.0     0.1     0.1
     2        0.0     0.0     2.2     2.2
 --------------------------------------------------
 General timing and accounting informations for this job:
 """
    result = parse_outcar_text(text)
    assert result["vasp_version"] == "6.5.1"
    assert result["electronic_converged"] is True
    assert result["ionic_marker_present"] is True
    assert result["timing_footer_present"] is True
    assert result["final_toten_eV"] == pytest.approx(-42.5)
    assert result["entropy_term_eV"] == pytest.approx(0.0014)
    assert result["Fmax_eV_A"] == pytest.approx(0.04)
    assert result["local_moments_muB"] == pytest.approx([0.1, 2.2])


def test_parse_oszicar_identifies_final_ionic_step_and_total_moment() -> None:
    text = """
 DAV:  1  -1.0
   1 F= -.100 E0= -.100  d E =-.1  mag= 3.2
 DAV:  1  -1.1
 160 F= -.200 E0= -.200  d E =-.1  mag= 2.8
 """
    result = parse_oszicar_text(text)
    assert result["last_ionic_step"] == 160
    assert result["final_total_moment_muB"] == pytest.approx(2.8)


def test_structure_metrics_detects_severe_overlap_and_large_volume_change() -> None:
    initial = Structure(
        Lattice.cubic(5),
        ["Li", "Mg", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]],
    )
    final = Structure(
        Lattice.cubic(3),
        ["Li", "Mg", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.01, 0.01, 0.01]],
    )
    metrics = structure_metrics(initial, final)
    assert metrics["minimum_interatomic_distance_A"] < 1.2
    assert abs(metrics["relative_volume_change_percent"]) > 30
    assert metrics["structure_collapsed"] is True


def test_formation_energy_uses_matching_internal_references() -> None:
    value = formation_energy_eV_atom(
        total_energy_eV=-50.0,
        composition={"Li": 1.0, "Cr": 2.0, "O": 4.0},
        references_eV_atom={"Li": -1.9, "Cr": -5.8, "O": -4.9},
    )
    assert value == pytest.approx((-50.0 + 1.9 + 11.6 + 19.6) / 7)


def test_probe_candidate_gate_requires_both_complete_magnetic_branches() -> None:
    rows = []
    for state in ("FM", "AFM_or_ferri"):
        for stage in ("relax", "static"):
            rows.append(
                {
                    "magnetic_state": state,
                    "stage": stage,
                    "exit_code": 0,
                    "electronic_converged": True,
                    "timing_footer_present": True,
                    "structure_collapsed": False,
                    "ionic_converged": stage == "relax",
                    "nsw_limit_reached": False,
                    "formation_energy_recomputed": stage == "static",
                    "potcar_retained": False,
                    "peak_rss_kb": 1_000_000,
                    "elapsed_seconds": 100.0,
                }
            )
    passed, reasons = candidate_gate_passes(
        rows, memory_limit_kb=50_000_000, wall_limit_seconds=21_600
    )
    assert passed is True
    assert reasons == []

    rows[-1]["electronic_converged"] = False
    passed, reasons = candidate_gate_passes(
        rows, memory_limit_kb=50_000_000, wall_limit_seconds=21_600
    )
    assert passed is False
    assert any("electronic" in reason for reason in reasons)
