from __future__ import annotations

from pathlib import Path

from analysis.audit_pilot_relaxation_artifacts import inventory_pilot_artifacts


def _write_outcar(path: Path, nsw: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f" INCAR: NSW = {nsw}\n", encoding="utf-8")


def test_static_overwrite_is_not_misclassified_as_original_relaxation(tmp_path: Path) -> None:
    candidate_id = "job_test"
    directory = tmp_path / f"candidate_001_{candidate_id}"
    _write_outcar(directory / "OUTCAR", nsw=0)
    (directory / "OSZICAR").write_text("static\n", encoding="utf-8")
    (directory / "relax.log").write_text("retained stdout\n", encoding="utf-8")

    summaries, files = inventory_pilot_artifacts(tmp_path, [candidate_id])

    assert summaries == [
        {
            "candidate_id": candidate_id,
            "candidate_directories_found": 1,
            "current_outcar_count": 1,
            "current_oszicar_count": 1,
            "relax_log_count": 1,
            "relaxation_outcar_count": 0,
            "relaxation_oszicar_count": 0,
            "original_relaxation_outcar_or_oszicar_found": False,
            "classification": "original_relaxation_artifact_unavailable",
        }
    ]
    assert {row["interpreted_role"] for row in files if row["name"] == "OUTCAR"} == {
        "static"
    }


def test_stage_separated_positive_nsw_outcar_is_detected(tmp_path: Path) -> None:
    candidate_id = "job_test"
    _write_outcar(tmp_path / f"candidate_001_{candidate_id}" / "relax" / "OUTCAR", nsw=80)

    summaries, _ = inventory_pilot_artifacts(tmp_path, [candidate_id])

    assert summaries[0]["original_relaxation_outcar_or_oszicar_found"] is True
    assert summaries[0]["relaxation_outcar_count"] == 1
    assert summaries[0]["classification"] == "relaxation_artifact_found"
