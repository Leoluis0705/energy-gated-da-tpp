from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure

from analysis.prepare_dft_kpoint_extension import build_kpoint_extension
from analysis.prepare_dft_kpoint_jobs import build_kpoint_bundle


def make_source(path: Path, element: str) -> Path:
    path.mkdir(parents=True)
    structure = Structure(Lattice.cubic(3.45), [element], [[0, 0, 0]])
    structure.to(filename=path / "POSCAR", fmt="poscar")
    (path / "INCAR").write_text("ENCUT = 520\nNSW = 0\n", encoding="utf-8")
    (path / "POTCAR").write_text(f"licensed {element} fixture\n", encoding="utf-8")
    return path


def test_extension_preserves_base_manifest_and_adds_three_pending_jobs(tmp_path):
    sources = {
        "LiCr2O4_C214": make_source(tmp_path / "sources" / "c214", "Li"),
        "LiMn2O4_C044": make_source(tmp_path / "sources" / "c044", "Mn"),
        "Li_reference": make_source(tmp_path / "sources" / "li", "Li"),
    }
    base_root = tmp_path / "base"
    base_manifest = base_root / "jobs" / "dft_jobs_manifest.csv"
    base = build_kpoint_bundle(
        system_sources=sources,
        spacings=(0.35, 0.30, 0.25),
        work_root=base_root,
        manifest_path=base_manifest,
        git_commit="old",
        python_executable=sys.executable,
        runner_path=Path("/remote/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/vasp_std"],
    )
    base["status"] = "DONE"
    base["exit_code"] = "0"
    base.to_csv(base_manifest, index=False, lineterminator="\n")
    before = hashlib.sha256(base_manifest.read_bytes()).hexdigest()
    extension_root = tmp_path / "extension"
    combined_manifest = extension_root / "jobs" / "dft_jobs_manifest.csv"

    combined = build_kpoint_extension(
        base_manifest_path=base_manifest,
        system_sources=sources,
        spacing=0.20,
        work_root=extension_root,
        manifest_path=combined_manifest,
        git_commit="new",
        python_executable=sys.executable,
        runner_path=Path("/remote/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/vasp_std"],
    )

    assert hashlib.sha256(base_manifest.read_bytes()).hexdigest() == before
    assert len(combined) == 12
    assert combined["status"].value_counts().to_dict() == {"DONE": 9, "PENDING": 3}
    assert combined["job_id"].is_unique
    assert combined["output_path"].is_unique
    pending_spacings = combined.loc[
        combined["status"].eq("PENDING"), "kpoint_spacing_Ainv"
    ].astype(float)
    assert set(pending_spacings) == {0.20}
    assert not list((extension_root / "inputs").rglob("POTCAR"))
    plan = json.loads((extension_root / "extension_plan.json").read_text(encoding="utf-8"))
    assert plan["base_manifest_sha256"] == before
    assert plan["extension_spacing_Ainv"] == 0.20
