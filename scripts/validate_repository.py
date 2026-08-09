"""Validate the public Energy-Gated DA-TPP research artifact."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "manuscript/Energy_Gated_DA_TPP_v71_manuscript.tex",
    "manuscript/manuscript_body_v71.tex",
    "manuscript/Energy_Gated_DA_TPP_v71_supplementary.tex",
    "manuscript/references.tex",
    "manuscript/SourceData/paired_cgcnn_histories.csv",
    "manuscript/SourceData/Gamma005HoldoutAnalysis/v60_gamma005_holdout_per_seed.csv",
    "code/experiments/active_learning/checkpoint_formation_clean.pth.tar",
    "code/experiments/active_learning/configs/frozen_final_protocol.yaml",
    "code/experiments/hidden_evaluability/THREE_SYSTEM_PROTOCOL_FREEZE.yaml",
    "code/experiments/dft_audit/candidate_pool_master.csv",
)

RESTRICTED_NAMES = {"POTCAR", "WAVECAR", "CHGCAR", "OUTCAR", "vasprun.xml"}


def run(command: list[str], cwd: Path) -> None:
    print(f"[run] ({cwd.relative_to(ROOT)}) {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def check_files() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))
    restricted = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.name in RESTRICTED_NAMES
    ]
    if restricted:
        raise SystemExit("Restricted VASP files present:\n" + "\n".join(restricted))
    print(f"[ok] {len(REQUIRED)} required files found; no restricted VASP files detected")


def check_supplementary_paths() -> None:
    manuscript = ROOT / "manuscript"
    supplementary = manuscript / "Energy_Gated_DA_TPP_v71_supplementary.tex"
    referenced = sorted(set(re.findall(r"\\path\{([^}]+)\}", supplementary.read_text(encoding="utf-8"))))
    missing: list[str] = []
    for relative in referenced:
        if not ((manuscript / relative).exists() or (ROOT / relative).exists()):
            missing.append(relative)
    if missing:
        raise SystemExit(
            "Supplementary path references without packaged files:\n"
            + "\n".join(missing)
        )
    print(f"[ok] {len(referenced)} Supplementary path references resolve inside the package")


def run_tests() -> None:
    active = ROOT / "code/experiments/active_learning"
    hidden = ROOT / "code/experiments/hidden_evaluability"
    dft = ROOT / "code/experiments/dft_audit"
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_formal_protocol.py",
            "tests/test_mc_dropout_seed_policy.py",
            "tests/test_uncertainty_units.py",
            "tests/test_run_paired_dataset_job.py",
        ],
        active,
    )
    run([sys.executable, "-m", "pytest", "-q", "tests/three_system"], hidden)
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_prospective_cr_current.py",
            "tests/test_prospective_cr_current_results.py",
        ],
        dft,
    )


def rebuild_figures() -> None:
    manuscript = ROOT / "manuscript"
    scripts = manuscript / "Scripts"
    for name in (
        "build_v50_recovery_figures.py",
        "build_v50_mlip_figures.py",
        "build_v50_relaxed_structures.py",
        "rebuild_v63_discussion_figures.py",
        "rebuild_v69_holdout_figure.py",
    ):
        run([sys.executable, str(scripts / name)], manuscript)


def manifest() -> None:
    output = ROOT / "provenance/public_manifest_sha256.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = ["relative_path,bytes,sha256"]
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path == output:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(ROOT).as_posix().replace('"', '""')
        rows.append(f'"{relative}",{path.stat().st_size},{digest}')
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"[ok] wrote {output.relative_to(ROOT)} with {len(rows) - 1} entries")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tests", action="store_true", help="run focused scientific tests")
    parser.add_argument("--figures", action="store_true", help="rebuild manuscript figures")
    parser.add_argument("--manifest", action="store_true", help="write SHA-256 manifest")
    args = parser.parse_args()

    check_files()
    check_supplementary_paths()
    if args.tests:
        run_tests()
    if args.figures:
        rebuild_figures()
    if args.manifest:
        manifest()
    print("[ok] repository validation completed")


if __name__ == "__main__":
    main()
