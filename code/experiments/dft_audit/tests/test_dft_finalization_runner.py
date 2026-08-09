from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from analysis.dft_finalization_runner import authorized_jobs, build_queue_manifest
from analysis.prepare_dft_finalization_inputs import (
    build_alpha_probe_input,
    build_candidate_relaxation_inputs,
    expand_vesta_magnetic_structure,
)


def _manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "job_id": "alpha_probe",
            "task_group": "alpha_mn_cost_probe",
            "candidate_id": "alpha_Mn",
            "calculation_type": "reference_static",
            "NELM": "1",
            "scientific_result": "False",
            "dependency_job_ids": "",
            "status": "PENDING",
        }
    ]
    for suffix in ("pbe_collinear", "pbe_noncollinear", "u_collinear", "u_noncollinear"):
        rows.append(
            {
                "job_id": f"alpha_formal_{suffix}",
                "task_group": "alpha_mn_reference_sensitivity",
                "candidate_id": "alpha_Mn",
                "calculation_type": "reference_static",
                "NELM": "120",
                "scientific_result": "True",
                "dependency_job_ids": "alpha_probe",
                "status": "PENDING",
            }
        )
    for candidate in ("C120", "C214"):
        for state in ("state_fm", "state_afm"):
            relax = f"{candidate}_{state}_relax"
            rows.extend(
                [
                    {
                        "job_id": relax,
                        "task_group": "main_candidate_verification_relaxation",
                        "candidate_id": candidate,
                        "calculation_type": "verification_relaxation",
                        "NELM": "160",
                        "scientific_result": "True",
                        "dependency_job_ids": "",
                        "status": "PENDING",
                    },
                    {
                        "job_id": f"{candidate}_{state}_static",
                        "task_group": "main_candidate_verification_static",
                        "candidate_id": candidate,
                        "calculation_type": "verification_static",
                        "NELM": "160",
                        "scientific_result": "True",
                        "dependency_job_ids": relax,
                        "status": "PENDING",
                    },
                ]
            )
    return pd.DataFrame(rows)


def _write_review(path: Path, *, safe: bool = True) -> None:
    payload = {
        "static_launch_authorized": safe,
        "review_generated_from_outputs": True,
        "pause_reasons": [] if safe else ["structural_difference_threshold_exceeded"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_alpha_phase_authorizes_only_non_scientific_nelm1_probe() -> None:
    selected = authorized_jobs(_manifest(), phase="alpha_probe")

    assert selected["job_id"].tolist() == ["alpha_probe"]
    assert selected.iloc[0]["NELM"] == "1"
    assert selected.iloc[0]["scientific_result"] == "False"


def test_relax_phase_authorizes_exactly_four_candidate_relaxations() -> None:
    selected = authorized_jobs(_manifest(), phase="candidate_relax")

    assert len(selected) == 4
    assert set(selected["candidate_id"]) == {"C120", "C214"}
    assert set(selected["calculation_type"]) == {"verification_relaxation"}
    assert not selected["job_id"].str.contains("alpha", case=False).any()


def test_static_phase_requires_complete_dependencies_and_safe_generated_review(tmp_path: Path) -> None:
    review = tmp_path / "review.json"
    _write_review(review)
    dependencies = {
        row["dependency_job_ids"]: {
            "status": "DONE",
            "exit_code": 0,
            "electronic_converged": True,
            "ionic_converged": True,
        }
        for _, row in _manifest().query("task_group == 'main_candidate_verification_static'").iterrows()
    }

    selected = authorized_jobs(
        _manifest(), phase="candidate_static", dependency_results=dependencies, structural_review=review
    )

    assert len(selected) == 4
    assert set(selected["calculation_type"]) == {"verification_static"}


def test_static_phase_stops_on_failed_dependency_or_structural_pause(tmp_path: Path) -> None:
    dependencies = {
        row["dependency_job_ids"]: {
            "status": "DONE",
            "exit_code": 0,
            "electronic_converged": True,
            "ionic_converged": True,
        }
        for _, row in _manifest().query("task_group == 'main_candidate_verification_static'").iterrows()
    }
    first = next(iter(dependencies))
    dependencies[first]["ionic_converged"] = False
    review = tmp_path / "review.json"
    _write_review(review)

    with pytest.raises(ValueError, match="dependency"):
        authorized_jobs(
            _manifest(), phase="candidate_static", dependency_results=dependencies, structural_review=review
        )

    dependencies[first]["ionic_converged"] = True
    _write_review(review, safe=False)
    with pytest.raises(ValueError, match="structural review"):
        authorized_jobs(
            _manifest(), phase="candidate_static", dependency_results=dependencies, structural_review=review
        )


def test_unknown_phase_cannot_authorize_jobs() -> None:
    with pytest.raises(ValueError, match="unsupported phase"):
        authorized_jobs(_manifest(), phase="all")


def test_vesta_expansion_applies_magnetic_time_reversal() -> None:
    text = """#VESTA_FORMAT_VERSION 3.1.9
MAGNETIC_CRYSTAL
GROUP
1 1 Custom
SYMOP
 0.0 0.0 0.0  1 0 0  0 1 0  0 0 1   1
 0.5 0.5 0.5  1 0 0  0 1 0  0 0 1  -1
 -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0
CELLP
 8.0 8.0 8.0 90.0 90.0 90.0
 0 0 0 0 0 0
STRUC
 1 Mn Mn1 1.0 0.0 0.0 0.0 2 -
                  0.0 0.0 0.0 0.0
 0 0 0 0 0 0 0
THERI 0
VECTR
 1 0.0 0.0 2.0 0
   1 0 0 0 0
 0 0 0 0 0
 0 0 0 0 0
VECTT
"""

    structure, moments = expand_vesta_magnetic_structure(text)

    assert len(structure) == 2
    assert structure.frac_coords.tolist() == [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    assert moments.tolist() == [[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]]


def test_candidate_input_builder_uses_frozen_manifest_values_and_never_writes_potcar(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    frame = _manifest()
    relax = frame.query("task_group == 'main_candidate_verification_relaxation'").copy()
    structure = Structure(Lattice.cubic(8.0), ["Li", "Cr", "Cr", "O", "O", "O", "O"],
                          [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
                           [0.1, 0.1, 0.1], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]])
    for index in relax.index:
        source = repo / f"source_{index}.vasp"
        structure.to(filename=source, fmt="poscar")
        relax.loc[index, "input_structure_path"] = source.name
        relax.loc[index, "input_structure_sha256"] = __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        relax.loc[index, "candidate_long_id"] = relax.loc[index, "candidate_id"]
        relax.loc[index, "magnetic_initialization"] = "state_fm" if "state_fm" in relax.loc[index, "job_id"] else "state_afm"
        relax.loc[index, "expected_mesh"] = "15x9x9" if relax.loc[index, "candidate_id"] == "C120" else "15x10x9"
        relax.loc[index, "ENCUT_eV"] = "520"
        relax.loc[index, "EDIFF"] = "1e-06"
        relax.loc[index, "EDIFFG"] = "-0.05"
        relax.loc[index, "ISMEAR"] = "0"
        relax.loc[index, "SIGMA"] = "0.05"
        relax.loc[index, "IBRION"] = "2"
        relax.loc[index, "ISIF"] = "3"
        relax.loc[index, "NSW"] = "160"
        relax.loc[index, "ALGO"] = "Normal"
        relax.loc[index, "LDAUL"] = "-1 2 -1"
        relax.loc[index, "LDAUU"] = "0.0 3.7 0.0"
        relax.loc[index, "LDAUJ"] = "0.0 0.0 0.0"
        relax.loc[index, "MAGMOM_definition"] = (
            "1*0.6 2*5.0 4*0.6" if relax.loc[index, "magnetic_initialization"] == "state_fm"
            else "1*0.6 1*5.0 1*-5.0 4*0.6"
        )

    output = tmp_path / "bundle"
    built = build_candidate_relaxation_inputs(relax, repo_root=repo, output_root=output, git_commit="abc123")

    assert len(built) == 4
    for job_id in built["job_id"]:
        directory = output / "inputs" / job_id
        assert {"INCAR", "KPOINTS", "POSCAR", "initial.cif", "input_provenance.json"}.issubset(
            {path.name for path in directory.iterdir()}
        )
        assert not (directory / "POTCAR").exists()
        incar = (directory / "INCAR").read_text(encoding="utf-8")
        assert "EDIFF = 1E-6" in incar
        assert "EDIFFG = -0.05" in incar
        assert "LREAL = .FALSE." in incar
        assert "LDAUU = 0.0 3.7 0.0" in incar


def test_queue_manifest_uses_audited_resource_sampling_and_unique_outputs(tmp_path: Path) -> None:
    inputs = pd.DataFrame(
        [
            {"job_id": "a", "config_hash": "ha", "input_dir": "/remote/inputs/a", "git_commit": "abc"},
            {"job_id": "b", "config_hash": "hb", "input_dir": "/remote/inputs/b", "git_commit": "abc"},
        ]
    )

    queue = build_queue_manifest(
        inputs,
        output_root=Path("/remote/outputs"),
        log_root=Path("/remote/logs"),
        runner_path=Path("/remote/code/run_vasp_benchmark_task.py"),
        python_executable="/usr/bin/python3",
        vasp_executable="/licensed/vasp_std",
        potcar_source=Path("/server-only/combined.POTCAR"),
        openblas_threads=8,
    )

    assert queue["output_path"].is_unique
    assert queue["log_path"].is_unique
    assert set(queue["status"]) == {"PENDING"}
    command = json.loads(queue.iloc[0]["command_json"])
    assert command[0] == "/usr/bin/python3"
    vasp_command = json.loads(command[-3])
    assert vasp_command == ["/licensed/vasp_std"]
    assert "/usr/bin/time" not in command[-3]
    assert json.loads(queue.iloc[0]["env_json"])["OPENBLAS_NUM_THREADS"] == "8"


def test_alpha_probe_builder_records_config_hash_and_nelm1_without_potcar(tmp_path: Path) -> None:
    operations = "\n".join(
        f" {index / 58:.12f} 0 0  1 0 0  0 1 0  0 0 1  {1 if index % 2 == 0 else -1}"
        for index in range(58)
    )
    source = tmp_path / "alpha.vesta"
    source.write_text(
        f"""#VESTA_FORMAT_VERSION 3.1.9
MAGNETIC_CRYSTAL
SYMOP
{operations}
 -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0
TRANM
CELLP
 58.0 8.0 8.0 90.0 90.0 90.0
 0 0 0 0 0 0
STRUC
 1 Mn Mn1 1.0 0.0 0.0 0.0 58 -
                  0.0 0.0 0.0 0.0
 0 0 0 0 0 0 0
THERI 0
VECTR
 1 0.0 0.0 2.0 0
   1 0 0 0 0
 0 0 0 0 0
VECTT
""",
        encoding="utf-8",
    )
    output = tmp_path / "probe"

    result = build_alpha_probe_input(vesta_path=source, output_root=output, git_commit="abc")

    provenance = json.loads((output / "input_provenance.json").read_text(encoding="utf-8"))
    assert provenance["config_hash"] == result["config_hash"]
    assert "NELM = 1" in (output / "INCAR").read_text(encoding="utf-8")
    assert not (output / "POTCAR").exists()
