"""Build a non-overwriting common-spacing extension of a completed k-point batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.prepare_dft_kpoint_jobs import build_kpoint_bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_kpoint_extension(
    *,
    base_manifest_path: Path,
    system_sources: Mapping[str, Path],
    spacing: float,
    work_root: Path,
    manifest_path: Path,
    git_commit: str,
    python_executable: str,
    runner_path: Path,
    vasp_command: Sequence[str],
) -> pd.DataFrame:
    """Append one pre-registered denser spacing without modifying the base manifest."""

    base_manifest = Path(base_manifest_path).resolve()
    base_sha256 = _sha256(base_manifest)
    base = pd.read_csv(base_manifest, dtype=str, keep_default_na=False)
    if base.empty or set(base["status"]) != {"DONE"}:
        raise ValueError("base k-point manifest must contain only DONE jobs")
    if base["system"].nunique() != 3 or set(base["system"]) != set(system_sources):
        raise ValueError("extension systems must exactly match the three base systems")
    if any(int(value) != 0 for value in base["exit_code"]):
        raise ValueError("base k-point manifest contains a nonzero exit code")

    spacing_value = float(spacing)
    existing_spacings = set(base["kpoint_spacing_Ainv"].astype(float))
    if spacing_value in existing_spacings:
        raise ValueError("extension spacing already exists in the base manifest")
    if spacing_value >= min(existing_spacings):
        raise ValueError("extension spacing must be denser than every base spacing")

    root = Path(work_root).resolve()
    extension_only_manifest = root / "jobs" / "extension_only_manifest.csv"
    extension = build_kpoint_bundle(
        system_sources=system_sources,
        spacings=(spacing_value,),
        work_root=root,
        manifest_path=extension_only_manifest,
        git_commit=git_commit,
        python_executable=python_executable,
        runner_path=runner_path,
        vasp_command=vasp_command,
    )
    if list(extension.columns) != list(base.columns):
        raise ValueError("base and extension manifest schemas differ")

    combined = pd.concat([base, extension.astype(str)], ignore_index=True)
    for column in ("job_id", "log_path", "output_path"):
        if not combined[column].is_unique:
            raise ValueError(f"combined manifest has duplicate {column}")

    destination = Path(manifest_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        combined.to_csv(handle, index=False, lineterminator="\n")

    plan = {
        "protocol_version": "egdatpp_dft_kpoint_extension_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str(base_manifest),
        "base_manifest_sha256": base_sha256,
        "combined_manifest": str(destination),
        "extension_spacing_Ainv": spacing_value,
        "selection_basis": "one common denser spacing after no common passing adjacent pair",
        "spacing_sequence_rule": "continue the preregistered 0.05 A^-1 decrement",
        "result_used_to_choose_spacing": False,
        "base_job_count": int(len(base)),
        "extension_job_count": int(len(extension)),
        "combined_job_count": int(len(combined)),
        "systems": sorted(system_sources),
        "git_commit": git_commit,
    }
    with (root / "extension_plan.json").open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--system", action="append", required=True, help="NAME=/server/source/directory")
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--runner-path", type=Path, required=True)
    parser.add_argument("--vasp-command-json", required=True)
    args = parser.parse_args()
    systems: dict[str, Path] = {}
    for value in args.system:
        if "=" not in value:
            raise ValueError("--system must use NAME=PATH")
        name, path = value.split("=", 1)
        systems[name] = Path(path)
    command = json.loads(args.vasp_command_json)
    if not isinstance(command, list) or not command:
        raise ValueError("VASP command JSON must be a non-empty list")
    combined = build_kpoint_extension(
        base_manifest_path=args.base_manifest,
        system_sources=systems,
        spacing=args.spacing,
        work_root=args.work_root,
        manifest_path=args.manifest,
        git_commit=args.git_commit,
        python_executable=args.python_executable,
        runner_path=args.runner_path,
        vasp_command=command,
    )
    print(
        json.dumps(
            {
                "combined_jobs": int(len(combined)),
                "pending_jobs": int(combined["status"].eq("PENDING").sum()),
                "manifest": str(Path(args.manifest).resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
