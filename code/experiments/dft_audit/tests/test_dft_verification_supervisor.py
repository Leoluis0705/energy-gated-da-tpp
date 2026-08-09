from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from analysis.run_dft_verification_supervisor import (
    assemble_server_potcar,
    normalize_relaxation_input_paths,
    require_relaxations_done,
    stage_directory,
)


def test_relaxation_supervisor_requires_exactly_four_successful_jobs() -> None:
    frame = pd.DataFrame(
        [
            {"job_id": f"job-{index}", "status": "DONE", "exit_code": "0"}
            for index in range(4)
        ]
    )
    require_relaxations_done(frame)

    frame.loc[2, "status"] = "FAILED"
    with pytest.raises(ValueError, match="not all DONE"):
        require_relaxations_done(frame)


def test_server_potcar_assembly_records_only_labels_hashes_and_order(tmp_path: Path) -> None:
    components = []
    for element, label in [("Li", "PAW_PBE Li_sv"), ("Cr", "PAW_PBE Cr_pv"), ("O", "PAW_PBE O")]:
        path = tmp_path / f"{element}.POTCAR"
        path.write_bytes(f"TITEL  = {label} fixture\nlicensed-{element}-payload\n".encode())
        components.append((element, path, label))
    output = tmp_path / "combined.POTCAR"

    provenance = assemble_server_potcar(components, output)

    assert provenance["element_order"] == ["Li", "Cr", "O"]
    assert provenance["combined_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert "licensed-Li-payload" not in str(provenance)
    assert [row["TITEL"] for row in provenance["components"]] == [
        "PAW_PBE Li_sv fixture",
        "PAW_PBE Cr_pv fixture",
        "PAW_PBE O fixture",
    ]
    with pytest.raises(FileExistsError):
        assemble_server_potcar(components, output)


def test_supervisor_remaps_windows_manifest_paths_to_server_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "candidate_relaxation" / "candidate_relaxation_jobs.csv"
    inputs = manifest.parent / "inputs"
    rows = []
    for index in range(4):
        job_id = f"job-{index}"
        directory = inputs / job_id
        directory.mkdir(parents=True)
        (directory / "initial.POSCAR").write_text("structure", encoding="utf-8")
        (directory / "initial.cif").write_text("structure", encoding="utf-8")
        rows.append({"job_id": job_id, "input_dir": rf"D:\old\run\{job_id}"})
    frame = pd.DataFrame(rows)

    normalized = normalize_relaxation_input_paths(frame, manifest)

    assert set(normalized["input_dir"]) == {
        str((inputs / f"job-{index}").resolve()) for index in range(4)
    }
    assert set(frame["input_dir"]).issubset({rf"D:\old\run\job-{index}" for index in range(4)})


def test_attempt_stage_directory_is_relative_and_prefix_scoped(tmp_path: Path) -> None:
    assert stage_directory(
        tmp_path, "candidate_pipeline_supervisor_attempt_2", "candidate_pipeline_supervisor"
    ) == tmp_path / "candidate_pipeline_supervisor_attempt_2"
    with pytest.raises(ValueError, match="unsafe stage directory"):
        stage_directory(tmp_path, "../escape", "candidate_pipeline_supervisor")
    assert stage_directory(
        tmp_path, "postprocess_attempt_2", "postprocess_attempt_"
    ) == tmp_path / "postprocess_attempt_2"
