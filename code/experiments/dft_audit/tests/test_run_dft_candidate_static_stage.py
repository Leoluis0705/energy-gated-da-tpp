from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.run_dft_candidate_static_stage import validate_stage_inputs


def test_candidate_static_stage_requires_exact_pending_and_reuse_counts(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    manifest = work / "jobs.csv"
    compatibility = work / "compatibility.csv"
    pd.DataFrame(
        [
            {"job_id": "a", "status": "PENDING", "config_hash": "h1", "output_path": str(work / "results/a")},
            {"job_id": "b", "status": "PENDING", "config_hash": "h2", "output_path": str(work / "results/b")},
        ]
    ).to_csv(manifest, index=False)
    pd.DataFrame(
        [
            {"decision": "REQUIRES_STATIC_VERIFICATION"},
            {"decision": "REQUIRES_STATIC_VERIFICATION"},
            {"decision": "REUSED_FROZEN_PROTOCOL_OUTPUT"},
        ]
    ).to_csv(compatibility, index=False)

    result = validate_stage_inputs(
        manifest_path=manifest,
        compatibility_path=compatibility,
        work_root=work,
        expected_jobs=2,
        expected_records=3,
        expected_reused=1,
    )

    assert result["pending_jobs"] == 2
    assert result["reused_records"] == 1


def test_candidate_static_stage_rejects_potcar_inside_work_root(tmp_path: Path) -> None:
    work = tmp_path / "work"
    work.mkdir()
    manifest = work / "jobs.csv"
    compatibility = work / "compatibility.csv"
    pd.DataFrame(
        [{"job_id": "a", "status": "PENDING", "config_hash": "h1", "output_path": str(work / "results/a")}]
    ).to_csv(manifest, index=False)
    pd.DataFrame([{"decision": "REQUIRES_STATIC_VERIFICATION"}]).to_csv(compatibility, index=False)
    (work / "POTCAR").write_text("forbidden", encoding="utf-8")

    try:
        validate_stage_inputs(
            manifest_path=manifest,
            compatibility_path=compatibility,
            work_root=work,
            expected_jobs=1,
            expected_records=1,
            expected_reused=0,
        )
    except ValueError as error:
        assert "POTCAR" in str(error)
    else:
        raise AssertionError("POTCAR copy was not rejected")
