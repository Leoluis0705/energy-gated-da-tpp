from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.prospective_cr_v2 import (
    FROZEN_CANDIDATE_IDS,
    build_candidate_manifest,
    build_protocol_audit,
    write_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_candidate_manifest_is_exact_five() -> None:
    frame = build_candidate_manifest(REPO_ROOT)

    assert frame["candidate_id"].tolist() == FROZEN_CANDIDATE_IDS
    assert len(frame) == 5
    assert set(frame["formula"]) == {"Li1 Cr2 O4"}
    assert not frame["candidate_id"].str.contains("_Mg_").any()
    assert (
        frame.loc[frame["candidate_id"].str.startswith("job_126"), "candidate_id"]
        .item()
        == "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0"
    )


def test_candidate_manifest_hashes_and_structure_fields() -> None:
    frame = build_candidate_manifest(REPO_ROOT)

    assert frame["cif_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (frame["atom_count"] == 7).all()
    assert (frame["minimum_interatomic_distance_A"] > 1.2).all()
    assert frame[
        [
            "lattice_a_A",
            "lattice_b_A",
            "lattice_c_A",
            "lattice_alpha_deg",
            "lattice_beta_deg",
            "lattice_gamma_deg",
        ]
    ].notna().all().all()
    assert set(frame["historical_dft_duplicate"]) == {False}


def test_protocol_uses_one_frozen_generator() -> None:
    audit = build_protocol_audit(REPO_ROOT)

    assert audit["generator"]["pymatgen_version"] == "2026.5.4"
    assert audit["generator"]["relax_class"] == "MPRelaxSet"
    assert audit["generator"]["static_class"] == "MPStaticSet"
    assert audit["settings"]["Cr_Ueff_eV"] == 3.7
    assert audit["settings"]["ENCUT_eV"] == 520.0
    assert audit["settings"]["LASPH"] is True
    assert audit["settings"]["LMAXMIX"] == 4
    assert audit["compatibility_smoke"]["success"] is True


def test_protocol_fails_closed_on_potcar_and_reference_entries() -> None:
    audit = build_protocol_audit(REPO_ROOT)

    assert audit["checks"]["potcar_titles_match"] is False
    assert audit["checks"]["compatible_reference_entries_available"] is False
    assert audit["checks"]["remote_environment_live_reverified"] is False
    assert audit["submission_allowed"] is False
    assert audit["submission_status"] == "STOP_SUBMISSION"
    assert {
        "POTCAR_TITLE_MISMATCH",
        "COMPATIBLE_REFERENCE_ENTRIES_UNAVAILABLE",
        "REMOTE_ENVIRONMENT_NOT_LIVE_REVERIFIED",
    }.issubset(audit["stop_reasons"])


def test_stopped_results_never_contain_formal_energy(tmp_path: Path) -> None:
    write_artifacts(REPO_ROOT, output_root=tmp_path)

    results = pd.read_csv(tmp_path / "results/prospective_cr_v2_results.csv")
    assert set(results["execution_status"]) == {"NOT_SUBMITTED_PROTOCOL_STOP"}
    assert results["mp_formation_energy_eV_atom"].isna().all()
    assert set(results["result_level"]) == {"NOT_EVALUATED"}
    assert not results["warm_start_geometry_used"].any()

    roundtrip = pd.read_csv(
        tmp_path / "results/prospective_cr_v2_roundtrip_check.csv"
    )
    assert set(roundtrip["roundtrip_status"]) == {"NOT_RUN"}

    adjustments = pd.read_csv(
        tmp_path / "results/prospective_cr_v2_compatibility_adjustments.csv"
    )
    assert set(adjustments["adjustment_status"]) == {"NOT_RUN"}


def test_all_requested_artifacts_are_written(tmp_path: Path) -> None:
    write_artifacts(REPO_ROOT, output_root=tmp_path)

    expected = [
        "manifests/prospective_dft_discovery_cr_v2.csv",
        "manifests/mp_dft_protocol_frozen.json",
        "reports/PROSPECTIVE_CR_V2_SELECTION_PROTOCOL.md",
        "reports/PROSPECTIVE_CR_V2_FREEZE_RECORD.md",
        "reports/MP_DFT_PROTOCOL_AUDIT.md",
        "reports/MP_DFT_PROTOCOL_FREEZE.md",
        "results/prospective_cr_v2_results.csv",
        "results/prospective_cr_v2_roundtrip_check.csv",
        "results/prospective_cr_v2_compatibility_adjustments.csv",
        "reports/JOB_092_COMPLETE_CHAIN_REPORT.md",
        "reports/PROSPECTIVE_CR_V2_RUNTIME_REPORT.md",
        "reports/PROSPECTIVE_CR_V2_FINAL_REPORT.md",
        "reports/PROSPECTIVE_CR_V2_FAILURES.md",
        "reports/PROSPECTIVE_CR_V2_PAPER_USABILITY.md",
    ]
    assert all((tmp_path / relative).is_file() for relative in expected)

    protocol = json.loads(
        (tmp_path / "manifests/mp_dft_protocol_frozen.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["batch_id"] == "prospective_dft_discovery_cr_v2"
    assert protocol["submission_allowed"] is False

    final_report = (
        tmp_path / "reports/PROSPECTIVE_CR_V2_FINAL_REPORT.md"
    ).read_text(encoding="utf-8")
    assert "0/5" in final_report
    assert "NOT_SUBMITTED_PROTOCOL_STOP" in final_report
    assert "MP_DFT_EVALUATED" not in final_report
