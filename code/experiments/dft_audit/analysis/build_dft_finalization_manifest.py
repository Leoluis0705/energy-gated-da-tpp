#!/usr/bin/env python3
"""Build the staged DFT-finalization plan from fixed local evidence.

This module does not contact a server or run VASP.  It records the proposed
alpha-Mn sensitivity scope, the C120/C214 verification dependency graph, and
an intentionally broad pre-probe cost envelope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DFT_PROTOCOL_SHA256 = (
    "4544f2a9a4685399ee4aaa34de4368f17febe846cca9b1fb8564e9a8306e5b7e"
)
BASE_DFT_PROTOCOL_PATH = (
    "artifacts/dft_server/completed_formal_results/d36d9cf09be426a6/attempt_1/"
    "payload/project/configs/dft_frozen_protocol.yaml"
)
PLAN_PROTOCOL_VERSION = "egdatpp_dft_finalization_v1"
REMOTE_ROOT_TEMPLATE = (
    "/root/autodl-tmp/Energy_Gated_DA_TPP_DFT_Audit/"
    f"{PLAN_PROTOCOL_VERSION}_{{launch_timestamp}}"
)
MAGNDATA_URL = "https://www.cryst.ehu.es/magndata/index.php?index=1.85"
MAGNDATA_DOI = "10.1063/1.358024"
COD_URL = "https://www.crystallography.net/cod/9011068.html"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _planned_hash(row: dict[str, Any]) -> str:
    excluded = {
        "status",
        "launch_gate",
        "estimated_wall_hours_low",
        "estimated_wall_hours_central",
        "estimated_wall_hours_high",
        "estimate_basis",
        "planned_config_hash",
    }
    payload = {key: row[key] for key in sorted(row) if key not in excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _common_row() -> dict[str, Any]:
    return {
        "protocol_version": PLAN_PROTOCOL_VERSION,
        "base_protocol_path": BASE_DFT_PROTOCOL_PATH,
        "base_protocol_sha256": BASE_DFT_PROTOCOL_SHA256,
        "status": "PENDING",
        "attempt": 0,
        "openblas_threads": 8,
        "kpoint_rule": "explicit_Gamma_mesh_ceil_reciprocal_length_over_0.15_Ainv",
        "kpoint_spacing_Ainv": 0.15,
        "ENCUT_eV": 520,
        "EDIFF": 1e-6,
        "ALGO": "Normal",
        "LASPH": True,
        "ADDGRID": True,
        "LREAL": False,
        "LWAVE": False,
        "LCHARG": False,
        "paw_label": "PAW_PBE Mn_pv 02Aug2007",
        "git_commit": "TO_BE_REPLACED_BY_LAUNCH_COMMIT",
        "command_template": (
            "python analysis/run_vasp_benchmark_task.py --input-dir {input_dir} "
            "--output-dir {output_dir}/attempt_1 --command-json "
            "'[\"/root/software/vasp_serial_src/vasp.6.5.1/bin/vasp_std\"]' "
            "--potcar-source {licensed_server_paw_path}"
        ),
    }


def _alpha_rows() -> list[dict[str, Any]]:
    common = _common_row()
    common.update(
        {
            "candidate_id": "alpha_Mn",
            "formula": "Mn58",
            "atom_count": 58,
            "element_order": "Mn",
            "input_structure_path": "REMOTE_SOURCE_PENDING_HASH:MAGNDATA_1.85.mcif",
            "input_structure_sha256": "UNAVAILABLE_UNTIL_OFFICIAL_SOURCE_FETCH",
            "structure_source_id": "MAGNDATA_1.85",
            "structure_source_url": MAGNDATA_URL,
            "structure_source_doi": MAGNDATA_DOI,
            "structure_crosscheck_url": COD_URL,
            "expected_mesh": "5x5x5",
            "calculation_type": "reference_static",
            "IBRION": -1,
            "ISIF": "",
            "NSW": 0,
            "EDIFFG": "",
            "NELM": 120,
            "ISMEAR": 1,
            "SIGMA": 0.2,
            "ISYM": 0,
            "SAXIS": "0 0 1",
            "scientific_scope": (
                "alpha-Mn reference sensitivity; two tested magnetic initializations"
            ),
            "stop_condition": (
                "electronic nonconvergence; memory exceeds safe two-task envelope; "
                "Mn-candidate qualitative conclusion changes"
            ),
        }
    )
    probe_id = "dft_alpha_mn_cost_probe_pbe_noncollinear"
    probe = {
        **common,
        "job_id": probe_id,
        "task_group": "alpha_mn_cost_probe",
        "functional": "PBE",
        "magnetic_initialization": "magndata_1p85_noncollinear",
        "MAGMOM_definition": (
            "all 58 reported (Mx,My,Mz) vectors from MAGNDATA 1.85 in generated POSCAR order"
        ),
        "ISPIN": "not_set_with_LNONCOLLINEAR",
        "LNONCOLLINEAR": True,
        "LDAU": False,
        "LDAUTYPE": "",
        "LDAUL": "",
        "LDAUU": "",
        "LDAUJ": "",
        "LMAXMIX": "",
        "Ueff_eV": "",
        "NELM": 1,
        "scientific_result": False,
        "dependency_job_ids": "",
        "launch_gate": "USER_CONFIRMATION",
        "concurrency_cap": 1,
        "output_path_template": f"{REMOTE_ROOT_TEMPLATE}/alpha_mn/cost_probe",
        "estimated_wall_hours_low": 0.5,
        "estimated_wall_hours_central": 2.0,
        "estimated_wall_hours_high": 4.0,
        "estimate_basis": (
            "one electronic iteration; quadratic-to-cubic atom/band scaling from "
            "formal 7-atom static jobs; must replace this estimate before full alpha-Mn launch"
        ),
    }
    rows = [probe]
    settings = [
        ("PBE", "magndata_1p85_collinear_z_projection", False, False, "", 5.0, 14.0, 40.0),
        ("PBE", "magndata_1p85_noncollinear", True, False, "", 10.0, 28.0, 80.0),
        ("GGA+U", "magndata_1p85_collinear_z_projection", False, True, 3.9, 6.0, 17.0, 48.0),
        ("GGA+U", "magndata_1p85_noncollinear", True, True, 3.9, 12.0, 34.0, 96.0),
    ]
    for functional, initialization, noncollinear, ldau, ueff, low, central, high in settings:
        slug = functional.lower().replace("+", "_").replace(" ", "_")
        rows.append(
            {
                **common,
                "job_id": f"dft_alpha_mn_{slug}_{initialization}",
                "task_group": "alpha_mn_reference_sensitivity",
                "functional": functional,
                "magnetic_initialization": initialization,
                "MAGMOM_definition": (
                    "all 58 reported (Mx,My,Mz) vectors from MAGNDATA 1.85 in generated POSCAR order"
                    if noncollinear
                    else "reported Mz component for all 58 MAGNDATA 1.85 atoms in generated POSCAR order; Mx=My=0"
                ),
                "ISPIN": "not_set_with_LNONCOLLINEAR" if noncollinear else 2,
                "LNONCOLLINEAR": noncollinear,
                "LDAU": ldau,
                "LDAUTYPE": 2 if ldau else "",
                "LDAUL": "2" if ldau else "",
                "LDAUU": "3.9" if ldau else "",
                "LDAUJ": "0.0" if ldau else "",
                "LMAXMIX": 4 if ldau else "",
                "Ueff_eV": ueff,
                "scientific_result": True,
                "dependency_job_ids": probe_id,
                "launch_gate": "MEASURED_ALPHA_COST_AND_MEMORY_ACCEPTED",
                "concurrency_cap": 2,
                "output_path_template": (
                    f"{REMOTE_ROOT_TEMPLATE}/alpha_mn/{slug}/{initialization}"
                ),
                "estimated_wall_hours_low": low,
                "estimated_wall_hours_central": central,
                "estimated_wall_hours_high": high,
                "estimate_basis": (
                    "pre-probe quadratic-to-cubic atom/band scaling envelope; replace with "
                    "measured NELM=1 throughput and memory before launch"
                ),
            }
        )
    return rows


def _candidate_rows(archive: Path) -> list[dict[str, Any]]:
    specifications = [
        {
            "candidate_id": "C120",
            "candidate_long_id": "job_120_Cr_fe_-1.424_n4_generated_crystals_cif__gen_1",
            "folder": (
                "new12_dft_final/candidate_outputs/"
                "candidate_001_Cr_job_120_Cr_fe_-1.424_n4_generated_crystals_cif__gen_1"
            ),
            "mesh": "15x9x9",
            "state_estimates": {
                "state_fm": (2.5, 4.6, 7.0),
                "state_afm": (2.0, 3.7, 5.5),
            },
        },
        {
            "candidate_id": "C214",
            "candidate_long_id": "job_214_Cr_fe_-0.857_n4_generated_crystals_cif__gen_0",
            "folder": (
                "new12_dft_final/candidate_outputs/"
                "candidate_002_Cr_job_214_Cr_fe_-0.857_n4_generated_crystals_cif__gen_0"
            ),
            "mesh": "15x10x9",
            "state_estimates": {
                "state_fm": (1.0, 1.5, 2.5),
                "state_afm": (1.2, 2.0, 3.2),
            },
        },
    ]
    rows: list[dict[str, Any]] = []
    for specification in specifications:
        for state, estimates in specification["state_estimates"].items():
            source = (
                Path(specification["folder"])
                / "tight_magnetic_states"
                / state
                / "01_tight_relax"
                / "CONTCAR"
            )
            absolute_source = archive / source
            if not absolute_source.is_file():
                raise FileNotFoundError(absolute_source)
            relax_id = f"dft_{specification['candidate_id'].lower()}_{state}_verification_relax"
            common = _common_row()
            common.update(
                {
                    "candidate_id": specification["candidate_id"],
                    "candidate_long_id": specification["candidate_long_id"],
                    "formula": "LiCr2O4",
                    "atom_count": 7,
                    "element_order": "Li Cr O",
                    "paw_label": (
                        "PAW_PBE Li_sv 10Sep2004 | PAW_PBE Cr_pv 02Aug2007 | "
                        "PAW_PBE O 08Apr2002"
                    ),
                    "functional": "GGA+U",
                    "magnetic_initialization": state,
                    "MAGMOM_definition": (
                        "1*0.6 2*5.0 4*0.6"
                        if state == "state_fm"
                        else "1*0.6 1*5.0 1*-5.0 4*0.6"
                    ),
                    "ISPIN": 2,
                    "LNONCOLLINEAR": False,
                    "LDAU": True,
                    "LDAUTYPE": 2,
                    "LDAUL": "-1 2 -1",
                    "LDAUU": "0.0 3.7 0.0",
                    "LDAUJ": "0.0 0.0 0.0",
                    "LMAXMIX": 4,
                    "ISYM": "VASP_default",
                    "SAXIS": "",
                    "Ueff_eV": 3.7,
                    "expected_mesh": specification["mesh"],
                    "ISMEAR": 0,
                    "SIGMA": 0.05,
                    "scientific_result": True,
                    "scientific_scope": (
                        "verification calculation; two tested magnetic initializations; "
                        "not a historical result"
                    ),
                    "concurrency_cap": 4,
                    "stop_condition": (
                        "electronic or ionic nonconvergence; visible structural reconstruction; "
                        "distinct magnetic structural branches; formation-energy shift >0.02 eV/atom; "
                        "current structural conclusion no longer holds"
                    ),
                }
            )
            low, central, high = estimates
            rows.append(
                {
                    **common,
                    "job_id": relax_id,
                    "task_group": "main_candidate_verification_relaxation",
                    "calculation_type": "verification_relaxation",
                    "input_structure_path": source.as_posix(),
                    "input_structure_sha256": _sha256(absolute_source),
                    "structure_source_id": "archived_historical_tight_relaxation_final_structure",
                    "structure_source_url": "",
                    "structure_source_doi": "",
                    "structure_crosscheck_url": "",
                    "IBRION": 2,
                    "ISIF": 3,
                    "NSW": 160,
                    "EDIFFG": -0.05,
                    "NELM": 160,
                    "dependency_job_ids": "",
                    "launch_gate": "USER_CONFIRMATION",
                    "output_path_template": (
                        f"{REMOTE_ROOT_TEMPLATE}/main_candidate_verification/"
                        f"{specification['candidate_id']}/{state}/01_verification_relax"
                    ),
                    "estimated_wall_hours_low": low,
                    "estimated_wall_hours_central": central,
                    "estimated_wall_hours_high": high,
                    "estimate_basis": (
                        "historical state-specific ionic/electronic iteration count scaled by "
                        "formal 0.15-A^-1 static iteration timing"
                    ),
                }
            )
            rows.append(
                {
                    **common,
                    "job_id": f"dft_{specification['candidate_id'].lower()}_{state}_verification_static",
                    "task_group": "main_candidate_verification_static",
                    "calculation_type": "verification_static",
                    "input_structure_path": "DEPENDENCY_OUTPUT:CONTCAR",
                    "input_structure_sha256": "DEPENDENCY_OUTPUT:SHA256",
                    "structure_source_id": relax_id,
                    "structure_source_url": "",
                    "structure_source_doi": "",
                    "structure_crosscheck_url": "",
                    "IBRION": -1,
                    "ISIF": "",
                    "NSW": 0,
                    "EDIFFG": "",
                    "NELM": 160,
                    "dependency_job_ids": relax_id,
                    "launch_gate": "RELAXATION_CONVERGED_WITHOUT_STOP_CONDITION",
                    "output_path_template": (
                        f"{REMOTE_ROOT_TEMPLATE}/main_candidate_verification/"
                        f"{specification['candidate_id']}/{state}/02_frozen_static"
                    ),
                    "estimated_wall_hours_low": 0.65,
                    "estimated_wall_hours_central": 0.80,
                    "estimated_wall_hours_high": 1.10,
                    "estimate_basis": (
                        "measured formal C120/C214 frozen-static wall times with conservative tail"
                    ),
                }
            )
    return rows


def build_manifest(archive: str | Path) -> pd.DataFrame:
    root = Path(archive).resolve()
    rows = [*_alpha_rows(), *_candidate_rows(root)]
    for row in rows:
        row["planned_config_hash"] = _planned_hash(row)
    columns = [
        "job_id",
        "task_group",
        "protocol_version",
        "candidate_id",
        "candidate_long_id",
        "formula",
        "functional",
        "magnetic_initialization",
        "calculation_type",
        "scientific_result",
        "scientific_scope",
        "dependency_job_ids",
        "launch_gate",
        "status",
        "attempt",
        "input_structure_path",
        "input_structure_sha256",
        "structure_source_id",
        "structure_source_url",
        "structure_source_doi",
        "structure_crosscheck_url",
        "atom_count",
        "element_order",
        "paw_label",
        "base_protocol_path",
        "base_protocol_sha256",
        "kpoint_rule",
        "kpoint_spacing_Ainv",
        "expected_mesh",
        "ENCUT_eV",
        "EDIFF",
        "EDIFFG",
        "NELM",
        "ALGO",
        "ISMEAR",
        "SIGMA",
        "ISPIN",
        "MAGMOM_definition",
        "ISYM",
        "SAXIS",
        "LNONCOLLINEAR",
        "LDAU",
        "LDAUTYPE",
        "LDAUL",
        "LDAUU",
        "LDAUJ",
        "LMAXMIX",
        "Ueff_eV",
        "LASPH",
        "ADDGRID",
        "LREAL",
        "IBRION",
        "ISIF",
        "NSW",
        "LWAVE",
        "LCHARG",
        "openblas_threads",
        "concurrency_cap",
        "estimated_wall_hours_low",
        "estimated_wall_hours_central",
        "estimated_wall_hours_high",
        "estimate_basis",
        "stop_condition",
        "output_path_template",
        "command_template",
        "git_commit",
        "planned_config_hash",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = ""
    return frame.loc[:, columns]


def _sum_cost(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="raise").sum())


def build_report(manifest: pd.DataFrame) -> str:
    probe = manifest.query("task_group == 'alpha_mn_cost_probe'")
    alpha = manifest.query("task_group == 'alpha_mn_reference_sensitivity'")
    relax = manifest.query("task_group == 'main_candidate_verification_relaxation'")
    static = manifest.query("task_group == 'main_candidate_verification_static'")
    cost_rows = []
    for label, frame in [
        ("Alpha-Mn cost probe", probe),
        ("Alpha-Mn scientific static calculations", alpha),
        ("C120/C214 verification relaxations", relax),
        ("C120/C214 frozen statics", static),
        ("Total planned VASP task-work", manifest),
    ]:
        values = [
            _sum_cost(frame, "estimated_wall_hours_low"),
            _sum_cost(frame, "estimated_wall_hours_central"),
            _sum_cost(frame, "estimated_wall_hours_high"),
        ]
        cost_rows.append(
            f"| {label} | {len(frame)} | {values[0]:.2f} | {values[1]:.2f} | "
            f"{values[2]:.2f} | {values[0] * 8:.1f} / {values[1] * 8:.1f} / "
            f"{values[2] * 8:.1f} |"
        )
    return f"""# DFT finalization launch plan and pre-probe budget

## State

- This plan contains **{len(manifest)} logical VASP tasks**: one alpha-Mn cost probe,
  four conditional alpha-Mn reference calculations, four C120/C214 verification
  relaxations, and four dependency-gated frozen statics.
- The server has not been contacted and no VASP task has been started in this stage.
- Every row is `PENDING`. The cost probe is **not a scientific result** and must not
  enter the formation-energy analysis.
- Base frozen DFT protocol SHA-256: `{BASE_DFT_PROTOCOL_SHA256}`.

## Proposed alpha-Mn scope requiring confirmation

The actual alpha-Mn input is proposed to use [MAGNDATA #1.85]({MAGNDATA_URL}), whose
atomic positions and magnetic structure trace to Lawson et al. (1994), DOI
[`{MAGNDATA_DOI}`](https://doi.org/{MAGNDATA_DOI}). The crystallographic cross-check
is [COD 9011068]({COD_URL}). The two tested magnetic initializations
are (i) the reported non-collinear moment vectors and (ii) their explicitly labeled
collinear z projection. The scope is limited to these two tested magnetic
initializations.

PBE and GGA+U (`Ueff(Mn)=3.9 eV`) use the retained elemental-reference settings:
`ENCUT=520 eV`, `EDIFF=1e-6`, `ISMEAR=1`, `SIGMA=0.2 eV`, the same Mn PAW label,
and an explicit Gamma `5x5x5` mesh generated by the frozen 0.15-A^-1 rule.

## Candidate verification scope

C120 and C214 each start from the archived state-specific historical tight-relaxation
`CONTCAR`. `state_fm` and `state_afm` are separately relaxed with GGA+U and the frozen
0.15-A^-1 mesh. Each converged final structure receives its own frozen static. Only
then is the lower-energy configuration among the two tested initializations selected.
These are verification calculations and must not be represented as historical runs.

## Pre-probe compute envelope

The table reports task-wall hours summed over jobs. CPU-hours use eight OpenBLAS
threads per job. The alpha-Mn envelope is deliberately broad because prior formal
timings cover seven-atom cells, whereas the proposed magnetic cell has 58 atoms.

| scope | jobs | task-wall low (h) | central (h) | high (h) | CPU-hours low / central / high |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost_rows)}

The first launch gate is therefore staged: run the alpha-Mn `NELM=1` probe alone,
record elapsed time and peak memory, and replace the four alpha-Mn estimates before
launching them. Candidate verification may then run at concurrency four. Alpha-Mn
concurrency remains one until the probe shows enough memory headroom; it may rise to
two, never four, without another representative concurrency test.

Using the current central envelope, two-way alpha-Mn concurrency gives about 48 h for
the four scientific alpha-Mn jobs. Adding the isolated 2 h probe and fitting the
candidate verification inside the remaining CPU slots gives roughly 50 h before
retry reserve, or 60 h with 20% reserve. The pre-probe range is approximately
21--168 h with reserve and is not a formal server budget. A measured post-probe
estimate is mandatory.

## Stop conditions

Stop before dependent statics or manuscript work if any relaxation fails electronic
or ionic convergence, the two initializations enter distinct structural branches,
visible reconstruction occurs, the frozen-static formation energy moves by more than
**0.02 eV/atom**, or a current C120/C214 structural conclusion no longer holds. Stop
the Mn sensitivity analysis if the qualitative Mn-candidate conclusion changes.

No pilot candidate and no C044 relaxation is included.
"""


def write_outputs(archive: str | Path) -> tuple[Path, Path]:
    root = Path(archive).resolve()
    manifest = build_manifest(root)
    manifest_path = root / "jobs" / "dft_finalization_jobs_manifest.csv"
    report_path = root / "docs" / "DFT_FINALIZATION_LAUNCH_PLAN.md"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    report_path.write_text(build_report(manifest), encoding="utf-8", newline="\n")
    return manifest_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    manifest, report = write_outputs(args.archive)
    print(manifest)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
