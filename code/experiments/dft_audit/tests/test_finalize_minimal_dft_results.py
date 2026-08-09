from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.finalize_minimal_dft_results import finalize_results


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stage(
    candidate_id: str,
    magnetic_state: str,
    stage: str,
    *,
    total_energy: float,
    formation_energy: float | None = None,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "candidate_stratum": "A_core_ALIGNN",
        "formula": "Li1 Cr2 O4",
        "element": "Cr",
        "functional": "GGA+U",
        "magnetic_state": magnetic_state,
        "stage": stage,
        "exit_code": 0,
        "timed_out": False,
        "electronic_converged": True,
        "ionic_converged": stage == "relax",
        "nsw_limit_reached": False,
        "Fmax_eV_A": 0.03 if stage == "relax" else "",
        "timing_footer_present": True,
        "final_toten_eV": total_energy,
        "formation_energy_eV_atom": (
            formation_energy if formation_energy is not None else ""
        ),
        "formation_energy_recomputed": stage == "static",
        "structure_collapsed": False,
        "collapse_reasons": "[]",
        "volume_change_fraction": 0.01,
        "minimum_pair_distance_A": 1.8,
        "final_total_moment_muB": 2.0 if magnetic_state == "FM" else 0.2,
        "local_moments_muB_json": "[]",
        "band_gap_eV": 0.5,
        "entropy_term_abs_eV_atom": 0.0002,
        "smearing_review_required": False,
        "elapsed_seconds": 100.0,
        "peak_rss_kb": 1000,
        "potcar_retained": False,
    }


def test_unresolved_scale_never_becomes_strict_target_claim(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    artifact = tmp_path / "artifact"
    (repo / "manifests").mkdir(parents=True)
    (artifact / "execution").mkdir(parents=True)
    (repo / "manifests" / "dft_protocol_frozen.json").write_text(
        json.dumps(
            {
                "same_scale_status": "UNRESOLVED",
                "overall_status": "PASS",
                "target_interval_eV_atom": [-2.18, -2.02],
                "result_classification_ceiling": "INTERNAL_PROTOCOL_DIAGNOSTIC",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        repo / "manifests" / "minimal_dft_5_candidates.csv",
        [
            {
                "frozen_order": 1,
                "candidate_id": "candidate-a",
                "formula": "Li1 Cr2 O4",
                "m_element": "Cr",
                "candidate_stratum": "A_core_ALIGNN",
                "alignn_formation_energy_eV_atom": -2.10,
            },
            {
                "frozen_order": 2,
                "candidate_id": "candidate-b",
                "formula": "Li1 Mg2 O4",
                "m_element": "Mg",
                "candidate_stratum": "D_high_model_disagreement",
                "alignn_formation_energy_eV_atom": -1.97,
            },
        ],
    )
    rows = [
        _stage("candidate-a", "FM", "relax", total_energy=-100.0),
        _stage(
            "candidate-a",
            "FM",
            "static",
            total_energy=-101.0,
            formation_energy=-2.10,
        ),
        _stage("candidate-a", "AFM_or_ferri", "relax", total_energy=-100.5),
        _stage(
            "candidate-a",
            "AFM_or_ferri",
            "static",
            total_energy=-101.5,
            formation_energy=-2.11,
        ),
    ]
    _write_csv(artifact / "execution" / "stage_results.csv", rows)
    (artifact / "execution" / "final_status.json").write_text(
        json.dumps(
            {
                "status": "STOPPED_BY_PROBE_GATE",
                "probe_candidates": ["candidate-a", "candidate-b"],
                "remaining_candidates": [],
            }
        ),
        encoding="utf-8",
    )
    (artifact / "execution" / "probe_gate.json").write_text(
        json.dumps(
            {
                "candidate-a": {"pass": True, "reasons": []},
                "candidate-b": {"pass": False, "reasons": ["not_run"]},
            }
        ),
        encoding="utf-8",
    )

    output = finalize_results(repo_root=repo, artifact_root=artifact)

    with output.results_csv.open(encoding="utf-8", newline="") as handle:
        by_id = {row["candidate_id"]: row for row in csv.DictReader(handle)}
    evaluated = by_id["candidate-a"]
    assert evaluated["selected_magnetic_state"] == "AFM_or_ferri"
    assert evaluated["internal_interval_membership"] == "True"
    assert evaluated["same_scale_status"] == "UNRESOLVED"
    assert evaluated["strict_original_interval_claim_allowed"] == "False"
    assert evaluated["result_scope"] == "INTERNAL_PROTOCOL_DIAGNOSTIC"
    assert evaluated["dft_evaluation_status"] == "DFT_EVALUATED_CANDIDATE"
    assert evaluated["stability_supported"] == "False"
    assert by_id["candidate-b"]["dft_evaluation_status"] == "NOT_RUN"
    report = output.final_report.read_text(encoding="utf-8")
    assert "DFT_CONFIRMED_TARGET" not in report
    assert "SAME_SCALE_STATUS: UNRESOLVED" in report
