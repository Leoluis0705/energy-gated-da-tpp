from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.build_dft_manifest import (
    REQUIRED_MANIFEST_COLUMNS,
    build_dft_manifest,
    build_pilot_artifact_search,
    build_timeline_report,
)


def test_manifest_has_exact_declared_cohorts_and_main_text_subset() -> None:
    archive = Path(__file__).resolve().parents[2]
    manifest = build_dft_manifest(archive)

    assert len(manifest) == 20
    assert manifest["candidate_id"].is_unique
    assert set(REQUIRED_MANIFEST_COLUMNS).issubset(manifest.columns)
    assert manifest["pilot_or_new"].value_counts().to_dict() == {"new": 12, "pilot": 8}

    main_text = manifest[manifest["main_text_selected"]]
    assert len(main_text) == 3
    assert set(main_text["candidate_id"]) == {
        "job_120_Cr_fe_-1.424_n4_generated_crystals_cif__gen_1",
        "job_214_Cr_fe_-0.857_n4_generated_crystals_cif__gen_0",
        "job_044_Mn_fe_-0.936_n4_generated_crystals_cif__gen_1",
    }
    assert set(main_text["pilot_or_new"]) == {"new"}


def test_manifest_does_not_promote_unverified_timestamps_or_rounds() -> None:
    archive = Path(__file__).resolve().parents[2]
    manifest = build_dft_manifest(archive)
    pilots = manifest.query("pilot_or_new == 'pilot'")
    new = manifest.query("pilot_or_new == 'new'")

    assert pilots["freeze_timestamp"].isna().all()
    assert set(pilots["result_known_at_freeze"]) == {"unknown"}
    assert pilots["Gate_round"].isna().all()
    assert pilots["Greedy_round"].isna().all()
    assert pilots["pilot_selection_iteration"].notna().all()

    assert new["freeze_timestamp"].notna().all()
    assert set(new["result_known_at_freeze"]) == {"false"}
    assert new["Gate_round"].notna().all()
    assert new["Greedy_round"].notna().all()
    assert new["timestamp_source"].str.contains("filesystem", case=False).all()


def test_manifest_marks_pilot_relaxation_artifacts_as_unavailable() -> None:
    archive = Path(__file__).resolve().parents[2]
    manifest = build_dft_manifest(archive)
    pilots = manifest.query("pilot_or_new == 'pilot'")

    assert set(pilots["relaxation_output_available"]) == {False}
    assert set(pilots["relaxation_evidence_available"]) == {True}
    assert set(pilots["static_output_available"]) == {True}
    failed = pilots[pilots["candidate_id"].str.startswith("job_182_Cr")].iloc[0]
    assert failed["DFT_status"] == "failed_static_electronic_nonconvergence"
    assert failed["failure_stage"] == "static"


def test_manifest_paths_and_hash_subjects_exist() -> None:
    archive = Path(__file__).resolve().parents[2]
    manifest = build_dft_manifest(archive)

    for row in manifest.itertuples(index=False):
        assert (archive / row.raw_job_path).is_dir()
        assert (archive / row.final_cif_path).is_file()
        assert row.sha256_subject == row.final_cif_path
        assert len(row.sha256) == 64


def test_pilot_artifact_search_records_all_requested_local_scopes() -> None:
    archive = Path(__file__).resolve().parents[2]
    search = build_pilot_artifact_search(archive)

    assert set(search["scope"]) >= {
        "archive_candidate_evidence",
        "archive_server_mirror",
        "archive_bundles",
        "historical_repository",
        "recycle_bin_current_user",
        "recycle_bin_other_sids",
        "baidu_sync",
        "onedrive",
        "wps_sync",
    }
    assert not search["original_relaxation_outcar_or_oszicar_found"].any()
    remote = search[search["scope"] == "historical_remote_server"].iloc[0]
    assert "authenticated read-only" in remote["search_method"]
    assert "all eight original relaxation artifacts remain unavailable" in remote["result"]


def test_timeline_report_uses_only_permitted_magnetic_wording() -> None:
    archive = Path(__file__).resolve().parents[2]
    manifest = build_dft_manifest(archive)
    search = build_pilot_artifact_search(archive)
    report = build_timeline_report(manifest, search)

    lowered = report.lower()
    assert "two tested magnetic initializations" in report
    assert "lower-energy configuration among the two tested initializations" in report
    assert "ground state" not in lowered
    assert "exhaustive magnetic" not in lowered
    assert "original_relaxation_artifact_unavailable" in report
    assert "reconstructed rerun" in report


def test_required_manifest_columns_are_stable() -> None:
    assert REQUIRED_MANIFEST_COLUMNS == (
        "candidate_id",
        "formula",
        "pilot_or_new",
        "selection_rule",
        "freeze_timestamp",
        "timestamp_source",
        "result_known_at_freeze",
        "Gate_round",
        "Greedy_round",
        "DFT_status",
        "relaxation_output_available",
        "static_output_available",
        "failure_stage",
        "failure_reason",
        "main_text_selected",
        "selection_reason",
        "raw_job_path",
        "final_cif_path",
        "sha256",
    )
