#!/usr/bin/env python3
"""Wait for gated statics and automatically generate the final verification analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis.run_dft_verification_supervisor import stage_directory


RESTRICTED = {"POTCAR", "WAVECAR", "CHGCAR", "CHG", "AECCAR0", "AECCAR1", "AECCAR2"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def supervisor_action(directory: Path) -> tuple[str, str]:
    failure = Path(directory) / "failure.json"
    if failure.is_file():
        payload = json.loads(failure.read_text(encoding="utf-8"))
        return "BLOCKED", str(payload.get("error", "supervisor failure"))
    final = Path(directory) / "final_status.json"
    if not final.is_file():
        return "WAIT", ""
    payload = json.loads(final.read_text(encoding="utf-8"))
    status = str(payload.get("status"))
    if status == "STATIC_TASKS_DONE_PENDING_POSTPROCESSING":
        return "RUN", ""
    return "BLOCKED", status


def static_postprocess_directory(run_root: Path, name: str) -> Path:
    return stage_directory(
        Path(run_root).resolve() / "candidate_static",
        name,
        "postprocess_attempt_",
    )


def postprocessor_entrypoint(python_executable: str) -> list[str]:
    return [
        python_executable,
        "-m",
        "analysis.postprocess_dft_verification_statics",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_hashes(root: Path, output: Path) -> None:
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.resolve() == output.resolve():
            continue
        if path.name.upper() in RESTRICTED or path.name.upper().startswith("POTCAR."):
            raise ValueError(f"restricted VASP payload in postprocess output: {path}")
        rows.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


def run_watcher(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    supervisor = stage_directory(
        run_root, args.supervisor_directory_name, "candidate_pipeline_supervisor"
    )
    watcher = stage_directory(
        run_root, args.watcher_directory_name, "candidate_postprocess_watcher"
    )
    watcher.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        watcher / "watcher_config.json",
        {
            "started_at_utc": _utc_now(),
            "git_commit": args.git_commit,
            "poll_seconds": args.poll_seconds,
            "heartbeat_seconds": args.heartbeat_seconds,
            "gpu_rerun": False,
            "formal_alpha_jobs": "not_launched",
            "static_postprocess_directory_name": args.static_postprocess_directory_name,
        },
    )
    last_heartbeat = 0.0
    while True:
        action, reason = supervisor_action(supervisor)
        if action == "RUN":
            break
        if action == "BLOCKED":
            _write_json_exclusive(
                watcher / "final_status.json",
                {
                    "status": "BLOCKED_BY_UPSTREAM_GATE",
                    "ended_at_utc": _utc_now(),
                    "reason": reason,
                },
            )
            return 2
        now = time.monotonic()
        if now - last_heartbeat >= float(args.heartbeat_seconds):
            token = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _write_json_exclusive(
                watcher / f"heartbeat_{token}.json",
                {"timestamp": _utc_now(), "status": "WAITING_FOR_GATED_STATIC_COMPLETION"},
            )
            last_heartbeat = now
        time.sleep(float(args.poll_seconds))

    static_root = run_root / "candidate_static"
    output = static_postprocess_directory(
        run_root, args.static_postprocess_directory_name
    )
    command = [
        *postprocessor_entrypoint(args.python_executable),
        "--static-input-manifest",
        str(static_root / "candidate_static_jobs.csv"),
        "--static-queue-manifest",
        str(static_root / "jobs_manifest.csv"),
        "--relaxation-metrics",
        str(
            stage_directory(
                run_root / "candidate_relaxation",
                args.relaxation_postprocess_directory_name,
                "postprocess_attempt_",
            )
            / "structure_metrics.csv"
        ),
        "--historical-magnetic",
        str(Path(args.historical_magnetic).resolve()),
        "--elemental-references",
        str(Path(args.elemental_references).resolve()),
        "--cif-root",
        str(output / "final_cifs"),
        "--static-metrics",
        str(output / "main_candidate_verification_statics.csv"),
        "--magnetic",
        str(output / "magnetic_initializations.csv"),
        "--formation",
        str(output / "recomputed_formation_energies.csv"),
        "--selected",
        str(output / "selected_candidate_comparison.csv"),
        "--review",
        str(output / "conclusion_update_review.json"),
        "--report",
        str(output / "MAIN_CANDIDATE_VERIFICATION_RELAXATION_REPORT.md"),
    ]
    _write_json_exclusive(
        watcher / "postprocess_command.json",
        {"argv": command, "git_commit": args.git_commit},
    )
    with (watcher / "postprocess.log").open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=Path(args.code_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if not (output / "conclusion_update_review.json").is_file():
        raise RuntimeError(f"static postprocessor exited {completed.returncode} without a review")
    review = json.loads((output / "conclusion_update_review.json").read_text(encoding="utf-8"))
    _write_hashes(output, output / "SHA256SUMS.csv")
    authorized = review.get("paper_conclusion_update_authorized") is True
    expected_code = 0 if authorized else 2
    if completed.returncode != expected_code:
        raise RuntimeError(
            f"static postprocessor exit/review mismatch: exit={completed.returncode}, authorized={authorized}"
        )
    _write_json_exclusive(
        watcher / "final_status.json",
        {
            "status": "POSTPROCESS_DONE" if authorized else "POSTPROCESS_DONE_PAPER_UPDATE_PAUSED",
            "ended_at_utc": _utc_now(),
            "paper_conclusion_update_authorized": authorized,
            "pause_reasons": review.get("pause_reasons", []),
            "output_path": str(output),
            "output_sha256_manifest": str(output / "SHA256SUMS.csv"),
        },
    )
    return 0 if authorized else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--code-root", required=True)
    parser.add_argument("--historical-magnetic", required=True)
    parser.add_argument("--elemental-references", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--supervisor-directory-name", default="candidate_pipeline_supervisor"
    )
    parser.add_argument(
        "--watcher-directory-name", default="candidate_postprocess_watcher"
    )
    parser.add_argument(
        "--relaxation-postprocess-directory-name", default="postprocess_attempt_1"
    )
    parser.add_argument(
        "--static-postprocess-directory-name", default="postprocess_attempt_1"
    )
    args = parser.parse_args()
    try:
        return run_watcher(args)
    except Exception as error:
        directory = stage_directory(
            Path(args.run_root).resolve(),
            args.watcher_directory_name,
            "candidate_postprocess_watcher",
        )
        if directory.is_dir() and not (directory / "failure.json").exists():
            _write_json_exclusive(
                directory / "failure.json",
                {
                    "status": "FAILED",
                    "ended_at_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
