"""Authorization gates for the approved DFT finalization stages.

This module deliberately separates job authorization from remote execution.  A
row being present in the planning manifest is not sufficient permission to run
it: only the three named phases below can return executable rows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd


_PHASE_GROUP = {
    "alpha_probe": "alpha_mn_cost_probe",
    "candidate_relax": "main_candidate_verification_relaxation",
    "candidate_static": "main_candidate_verification_static",
}


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _pending(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"].astype(str) == "PENDING"].copy()


def authorized_jobs(
    manifest: pd.DataFrame,
    *,
    phase: str,
    dependency_results: Mapping[str, Mapping[str, object]] | None = None,
    structural_review: Path | None = None,
) -> pd.DataFrame:
    """Return only rows authorized for one explicitly approved stage.

    Candidate statics require both successful dependency records and a
    script-generated structural review that contains no pause condition.
    """

    if phase not in _PHASE_GROUP:
        raise ValueError(f"unsupported phase: {phase}")
    required = {
        "job_id",
        "task_group",
        "candidate_id",
        "calculation_type",
        "NELM",
        "scientific_result",
        "dependency_job_ids",
        "status",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing required columns: {sorted(missing)}")

    group = _PHASE_GROUP[phase]
    stage = manifest[manifest["task_group"].astype(str) == group].copy()

    if phase == "alpha_probe":
        if len(stage) != 1:
            raise ValueError("alpha probe phase must contain exactly one planned job")
        row = stage.iloc[0]
        if str(row["NELM"]) != "1" or _as_bool(row["scientific_result"]):
            raise ValueError("alpha probe must be the non-scientific NELM=1 task")
        return _pending(stage).reset_index(drop=True)

    if len(stage) != 4 or set(stage["candidate_id"].astype(str)) != {"C120", "C214"}:
        raise ValueError(f"{phase} must contain exactly four C120/C214 jobs")

    if phase == "candidate_relax":
        if set(stage["calculation_type"].astype(str)) != {"verification_relaxation"}:
            raise ValueError("candidate relaxation phase contains a non-relaxation row")
        if stage["dependency_job_ids"].astype(str).str.strip().ne("").any():
            raise ValueError("candidate relaxation jobs must not have dependencies")
        return _pending(stage).reset_index(drop=True)

    if set(stage["calculation_type"].astype(str)) != {"verification_static"}:
        raise ValueError("candidate static phase contains a non-static row")
    if dependency_results is None:
        raise ValueError("candidate static phase requires dependency results")
    for dependency in stage["dependency_job_ids"].astype(str):
        result = dependency_results.get(dependency)
        if result is None or not (
            str(result.get("status")) == "DONE"
            and int(result.get("exit_code", 1)) == 0
            and result.get("electronic_converged") is True
            and result.get("ionic_converged") is True
        ):
            raise ValueError(f"dependency did not pass the static gate: {dependency}")

    if structural_review is None or not Path(structural_review).is_file():
        raise ValueError("candidate static phase requires a generated structural review")
    review = json.loads(Path(structural_review).read_text(encoding="utf-8"))
    if review.get("review_generated_from_outputs") is not True or review.get("static_launch_authorized") is not True:
        raise ValueError("structural review did not authorize frozen statics")
    if review.get("pause_reasons"):
        raise ValueError("structural review contains pause reasons")
    return _pending(stage).reset_index(drop=True)


def build_queue_manifest(
    inputs: pd.DataFrame,
    *,
    output_root: Path,
    log_root: Path,
    runner_path: Path,
    python_executable: str,
    vasp_executable: str,
    potcar_source: Path,
    openblas_threads: int,
) -> pd.DataFrame:
    """Convert immutable input records into the audited queue schema."""

    required = {"job_id", "config_hash", "input_dir", "git_commit"}
    missing = required.difference(inputs.columns)
    if missing:
        raise ValueError(f"input records missing columns: {sorted(missing)}")
    if inputs.empty or inputs["job_id"].astype(str).duplicated().any():
        raise ValueError("queue inputs must contain unique jobs")
    if int(openblas_threads) <= 0:
        raise ValueError("OPENBLAS thread count must be positive")

    records: list[dict[str, object]] = []
    for _, row in inputs.iterrows():
        job_id = str(row["job_id"])
        output = Path(output_root) / job_id / "attempt_1"
        log = Path(log_root) / f"{job_id}.log"
        vasp_command = [str(vasp_executable)]
        command = [
            str(python_executable),
            str(runner_path),
            "--input-dir",
            str(row["input_dir"]),
            "--output-dir",
            str(output),
            "--command-json",
            json.dumps(vasp_command, separators=(",", ":")),
            "--potcar-source",
            str(potcar_source),
        ]
        records.append(
            {
                "job_id": job_id,
                "config_hash": str(row["config_hash"]),
                "git_commit": str(row["git_commit"]),
                "status": "PENDING",
                "start_time": "",
                "end_time": "",
                "exit_code": "",
                "log_path": str(log),
                "output_path": str(output),
                "sha256": "",
                "command_json": json.dumps(command, separators=(",", ":")),
                "cwd": str(Path(runner_path).parent.parent),
                "attempt": 0,
                "pid": "",
                "failure_reason": "",
                "env_json": json.dumps(
                    {
                        "OPENBLAS_NUM_THREADS": str(openblas_threads),
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                    },
                    separators=(",", ":"),
                ),
            }
        )
    frame = pd.DataFrame(records)
    if frame["output_path"].duplicated().any() or frame["log_path"].duplicated().any():
        raise ValueError("queue output or log collision")
    return frame
