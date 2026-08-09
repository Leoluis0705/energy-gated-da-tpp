import json
import sys
from pathlib import Path

from pymatgen.core import Lattice, Structure

from analysis.prepare_dft_kpoint_jobs import build_kpoint_bundle, explicit_mesh


def make_source(path: Path, element: str) -> Path:
    path.mkdir(parents=True)
    structure = Structure(Lattice.orthorhombic(2.95, 5.05, 5.40), [element], [[0, 0, 0]])
    structure.to(filename=path / "POSCAR", fmt="poscar")
    (path / "INCAR").write_text("ENCUT = 520\nNSW = 0\n", encoding="utf-8")
    (path / "POTCAR").write_text(f"licensed {element} fixture\n", encoding="utf-8")
    return path


def test_explicit_mesh_uses_one_monotone_reciprocal_spacing_rule():
    lengths = (2.13, 1.25, 1.17)
    assert explicit_mesh(lengths, 0.35) == (7, 4, 4)
    assert explicit_mesh(lengths, 0.30) == (8, 5, 4)
    assert explicit_mesh(lengths, 0.25) == (9, 5, 5)


def test_kpoint_bundle_generates_nine_jobs_without_copying_potcar(tmp_path):
    sources = {
        "C214_LiCr2O4": make_source(tmp_path / "sources" / "c214", "Li"),
        "C044_LiMn2O4": make_source(tmp_path / "sources" / "c044", "Mn"),
        "Li_metal": make_source(tmp_path / "sources" / "li", "Li"),
    }
    work = tmp_path / "formal"
    manifest = work / "jobs" / "dft_jobs_manifest.csv"

    frame = build_kpoint_bundle(
        system_sources=sources,
        spacings=(0.35, 0.30, 0.25),
        work_root=work,
        manifest_path=manifest,
        git_commit="abc123",
        python_executable=sys.executable,
        runner_path=Path("/remote/project/analysis/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/server/vasp_std"],
    )

    assert len(frame) == 9
    assert set(frame["system"]) == set(sources)
    assert set(frame["kpoint_spacing_Ainv"]) == {0.35, 0.30, 0.25}
    assert frame["status"].eq("PENDING").all()
    assert frame["output_path"].is_unique
    assert frame["config_hash"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not list((work / "inputs").rglob("POTCAR"))
    for command_json in frame["command_json"]:
        command = json.loads(command_json)
        assert "--potcar-source" in command
    for env_json in frame["env_json"]:
        assert json.loads(env_json)["OPENBLAS_NUM_THREADS"] == "8"


def test_kpoint_bundle_accepts_one_common_preregistered_extension_spacing(tmp_path):
    sources = {
        "C214_LiCr2O4": make_source(tmp_path / "sources" / "c214", "Li"),
        "C044_LiMn2O4": make_source(tmp_path / "sources" / "c044", "Mn"),
        "Li_metal": make_source(tmp_path / "sources" / "li", "Li"),
    }
    work = tmp_path / "extension"

    frame = build_kpoint_bundle(
        system_sources=sources,
        spacings=(0.20,),
        work_root=work,
        manifest_path=work / "jobs" / "extension_only.csv",
        git_commit="abc123",
        python_executable=sys.executable,
        runner_path=Path("/remote/project/analysis/run_vasp_benchmark_task.py"),
        vasp_command=["/licensed/server/vasp_std"],
    )

    assert len(frame) == 3
    assert set(frame["kpoint_spacing_Ainv"]) == {0.20}
    assert frame["status"].eq("PENDING").all()
    assert not list((work / "inputs").rglob("POTCAR"))
