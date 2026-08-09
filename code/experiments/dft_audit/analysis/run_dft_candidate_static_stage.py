"""Validate and run the approved candidate static-verification queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.update_active_gpu_manifest import update_active_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_stage_inputs(
    *,
    manifest_path: Path,
    compatibility_path: Path,
    work_root: Path,
    expected_jobs: int,
    expected_records: int,
    expected_reused: int,
) -> dict[str, object]:
    root = Path(work_root).resolve()
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    compatibility = pd.read_csv(compatibility_path, dtype=str, keep_default_na=False)
    if len(manifest) != expected_jobs or len(compatibility) != expected_records:
        raise ValueError("candidate static manifest or compatibility row count mismatch")
    if set(manifest["status"]) != {"PENDING"}:
        raise ValueError("candidate static jobs must all be PENDING before first launch")
    if manifest["job_id"].duplicated().any() or manifest["config_hash"].duplicated().any():
        raise ValueError("candidate static manifest contains duplicate job or config hashes")
    counts = compatibility["decision"].value_counts().to_dict()
    reused = int(counts.get("REUSED_FROZEN_PROTOCOL_OUTPUT", 0))
    required = int(counts.get("REQUIRES_STATIC_VERIFICATION", 0))
    if reused != expected_reused or required != expected_jobs or reused + required != expected_records:
        raise ValueError("candidate compatibility decisions do not match the approved batch")
    potcars = [str(path) for path in root.rglob("POTCAR") if path.is_file()]
    if potcars:
        raise ValueError(f"POTCAR content found inside generated work root: {potcars}")
    for raw in manifest["output_path"]:
        output = Path(raw).resolve()
        if not output.is_relative_to(root):
            raise ValueError(f"job output escapes work root: {output}")
        if output.exists():
            raise ValueError(f"candidate static output already exists before first launch: {output}")
    return {
        "pending_jobs": len(manifest),
        "compatibility_records": len(compatibility),
        "reused_records": reused,
        "manifest_sha256": _sha256(Path(manifest_path)),
        "compatibility_sha256": _sha256(Path(compatibility_path)),
    }


class StageAudit:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def write_state(self, state: str, **extra: object) -> None:
        record = {"checked_at": _utc_now(), "state": state, **extra}
        with (self.path / "stage_history.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        (self.path / "stage_state.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )

    def write_hashes(self) -> None:
        lines = []
        for path in sorted(item for item in self.path.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            lines.append(f"{_sha256(path)}  {path.relative_to(self.path).as_posix()}")
        (self.path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def run_stage(args: argparse.Namespace) -> int:
    work = Path(args.work_root).resolve()
    manifest = Path(args.manifest).resolve()
    compatibility = Path(args.compatibility).resolve()
    audit_path = Path(args.audit_dir).resolve()
    audit_path.mkdir(parents=True, exist_ok=False)
    audit = StageAudit(audit_path)
    (audit_path / "start_time_utc.txt").write_text(_utc_now() + "\n", encoding="utf-8", newline="\n")
    (audit_path / "stage_command.json").write_text(
        json.dumps([sys.executable, *sys.argv], indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    exit_code = 2
    try:
        validation = validate_stage_inputs(
            manifest_path=manifest,
            compatibility_path=compatibility,
            work_root=work,
            expected_jobs=args.expected_jobs,
            expected_records=args.expected_records,
            expected_reused=args.expected_reused,
        )
        audit.write_state("VALIDATED_PRELAUNCH", **validation)
        pointer_record = update_active_manifest(
            pointer=Path(args.pointer),
            history=Path(args.pointer_history),
            manifest=manifest,
        )
        (audit_path / "pointer_transition.json").write_text(
            json.dumps(pointer_record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
        )
        queue_audit = audit_path / "queue"
        queue_audit.mkdir()
        command = [
            args.python_executable,
            str(Path(args.queue_script).resolve()),
            "--manifest", str(manifest),
            "--audit-dir", str(queue_audit),
            "--output-root", str(work),
            "--concurrency", str(args.concurrency),
            "--resource-kind", "cpu",
            "--sample-interval-seconds", "5",
            "--progress-interval-seconds", "7200",
            "--minimum-free-bytes", str(10 * 1024**3),
            "--maximum-root-bytes", str(10 * 1024**3),
            "--baseline-throughput-per-hour", "1.0",
            "--throughput-floor-fraction", "0.70",
        ]
        (audit_path / "queue_command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(args.snapshot).resolve())
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        audit.write_state("RUNNING_CANDIDATE_STATIC_QUEUE", manifest_sha256=validation["manifest_sha256"])
        with (audit_path / "queue_stdout.json").open("x", encoding="utf-8") as stdout, (
            audit_path / "queue_stderr.log"
        ).open("x", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command,
                cwd=Path(args.snapshot).resolve(),
                env=environment,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        queue_frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        status_counts = queue_frame["status"].value_counts().to_dict()
        exit_code = int(completed.returncode)
        audit.write_state(
            "CANDIDATE_STATIC_QUEUE_COMPLETE" if exit_code == 0 else "BLOCKED_CANDIDATE_STATIC_QUEUE",
            queue_exit_code=exit_code,
            status_counts={str(key): int(value) for key, value in status_counts.items()},
        )
    except Exception as error:
        (audit_path / "stage_exception.log").write_text(traceback.format_exc(), encoding="utf-8", newline="\n")
        audit.write_state("BLOCKED_STAGE_EXCEPTION", error=f"{type(error).__name__}: {error}")
        exit_code = 2
    finally:
        (audit_path / "stage_exit_code.txt").write_text(str(exit_code) + "\n", encoding="utf-8", newline="\n")
        (audit_path / "end_time_utc.txt").write_text(_utc_now() + "\n", encoding="utf-8", newline="\n")
        audit.write_hashes()
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--compatibility", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--pointer-history", type=Path, required=True)
    parser.add_argument("--queue-script", type=Path, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--expected-jobs", type=int, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--expected-reused", type=int, required=True)
    return run_stage(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
