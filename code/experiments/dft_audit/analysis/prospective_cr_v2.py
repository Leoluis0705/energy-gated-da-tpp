"""Freeze and fail-closed audit the prospective LiCr2O4 MP-DFT batch.

This module never invokes VASP or writes to a remote system.  Its submission
gate is deliberately false while the audited POTCAR titles and compatible
reference-entry prerequisites do not meet the frozen Materials Project
protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet


BATCH_ID = "prospective_dft_discovery_cr_v2"
TARGET_INTERVAL_EV_ATOM = (-2.18, -2.02)
MARGIN_AUDIT_DECISION = "REMOVE_FROM_CORE_CLAIM"
METHOD_MAINLINE = "Group-concentration-triggered diversity correction"

FROZEN_CANDIDATE_IDS = [
    "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_3",
    "job_196_Cr_fe_-0.819_n4_generated_crystals_cif__gen_1",
    "job_234_Cr_fe_-1.123_n4_generated_crystals_cif__gen_3",
    "job_079_Cr_fe_-0.854_n4_generated_crystals_cif__gen_1",
    "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0",
]

SELECTION_REASONS = {
    FROZEN_CANDIDATE_IDS[0]: (
        "original ALIGNN core-middle Cr candidate; first formal-chain gate"
    ),
    FROZEN_CANDIDATE_IDS[1]: "original ALIGNN core-middle Cr candidate",
    FROZEN_CANDIDATE_IDS[2]: (
        "pre-result empirical-rank proximity plus independent "
        "StructureMatcher cluster"
    ),
    FROZEN_CANDIDATE_IDS[3]: (
        "pre-result empirical-rank proximity plus independent "
        "StructureMatcher cluster"
    ),
    FROZEN_CANDIDATE_IDS[4]: (
        "pre-frozen cross-structure-cluster Cr control from the prior manifest"
    ),
}

OFFICIAL_MP_POTCAR_TITLES = {
    "Li": "PAW_PBE Li_sv 23Jan2001",
    "Cr": "PAW_PBE Cr_pv 07Sep2000",
    "O": "PAW_PBE O 08Apr2002",
}

LAST_AUDITED_SERVER_POTCAR_TITLES = {
    "Li": "PAW_PBE Li_sv 10Sep2004",
    "Cr": "PAW_PBE Cr_pv 02Aug2007",
    "O": "PAW_PBE O 08Apr2002",
}

LAST_AUDITED_SERVER_POTCAR_SHA256 = {
    "Li": "201875120238865c2f235e24081bce20639c4ae21bc4e97e31f9e3b7cc8fb95b",
    "Cr": "836672959fc86f3b167531577dbf63d7fb0b8d96aaf8b40fb3c4265879bd744b",
    "O": "8a74b9a1f5fdb3d0c3e0183c7873177abdbef07d407b310b7edcd9ed0a3eea64",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def minimum_distance(structure: Structure) -> float:
    values = [
        float(structure.distance_matrix[i, j])
        for i in range(len(structure))
        for j in range(i + 1, len(structure))
    ]
    return min(values)


def build_candidate_manifest(repo_root: Path) -> pd.DataFrame:
    repo_root = Path(repo_root).resolve()
    source_path = repo_root / "candidate_pool_master.csv"
    source = pd.read_csv(source_path)
    by_id = source.set_index("candidate_id", drop=False)

    missing = [value for value in FROZEN_CANDIDATE_IDS if value not in by_id.index]
    if missing:
        raise ValueError(f"frozen candidates missing from material-pool audit: {missing}")

    rows: list[dict[str, Any]] = []
    for order, candidate_id in enumerate(FROZEN_CANDIDATE_IDS, 1):
        source_row = by_id.loc[candidate_id]
        if isinstance(source_row, pd.DataFrame):
            raise ValueError(f"duplicate material-pool rows for {candidate_id}")
        cif_path = Path(str(source_row["cif_path"])).resolve()
        if not cif_path.is_file():
            raise FileNotFoundError(cif_path)
        observed_hash = sha256_file(cif_path)
        expected_hash = str(source_row["cif_sha256"])
        if observed_hash != expected_hash:
            raise ValueError(f"CIF SHA-256 mismatch for {candidate_id}")

        structure = Structure.from_file(cif_path)
        lattice = structure.lattice
        rows.append(
            {
                "frozen_order": order,
                "batch_id": BATCH_ID,
                "candidate_id": candidate_id,
                "formula": structure.composition.formula,
                "cif_path": str(cif_path),
                "cif_sha256": observed_hash,
                "alignn_formation_energy_eV_atom": float(
                    source_row["alignn_formation_energy_eV_atom"]
                ),
                "target_interval_lower_eV_atom": TARGET_INTERVAL_EV_ATOM[0],
                "target_interval_upper_eV_atom": TARGET_INTERVAL_EV_ATOM[1],
                "target_normalized_position": float(source_row["alignn_u"]),
                "gate_round": int(source_row["gate_round"]),
                "greedy_round": int(source_row["greedy_round"]),
                "gate_rank": int(source_row["gate_rank"]),
                "greedy_rank": int(source_row["greedy_rank"]),
                "structure_matcher_cluster": str(
                    source_row["structure_matcher_cluster"]
                ),
                "structure_matcher_cluster_size": int(
                    source_row["structure_matcher_cluster_size"]
                ),
                "fingerprint_cluster": str(source_row["fingerprint_cluster"]),
                "fingerprint_cluster_size": int(
                    source_row["fingerprint_cluster_size"]
                ),
                "fingerprint_cluster_kind": str(
                    source_row["fingerprint_cluster_kind"]
                ),
                "historical_dft_duplicate": bool(
                    source_row["historical_dft_candidate"]
                ),
                "historical_dft_duplicate_status": (
                    "NO_EXACT_OR_SHARED_CLUSTER_DUPLICATE"
                    if not bool(source_row["historical_dft_candidate"])
                    and str(source_row["previous_dft_nearby_status"])
                    == "no_shared_cluster"
                    else str(source_row["previous_dft_nearby_status"])
                ),
                "minimum_interatomic_distance_A": minimum_distance(structure),
                "lattice_a_A": float(lattice.a),
                "lattice_b_A": float(lattice.b),
                "lattice_c_A": float(lattice.c),
                "lattice_alpha_deg": float(lattice.alpha),
                "lattice_beta_deg": float(lattice.beta),
                "lattice_gamma_deg": float(lattice.gamma),
                "cell_volume_A3": float(structure.volume),
                "atom_count": len(structure),
                "selection_reason": SELECTION_REASONS[candidate_id],
                "selection_energy_source": (
                    "material-pool pre-audit only; excludes job_092/job_239 "
                    "probe intermediate energies"
                ),
            }
        )

    frame = pd.DataFrame(rows)
    if frame["candidate_id"].tolist() != FROZEN_CANDIDATE_IDS:
        raise AssertionError("frozen candidate order changed")
    if len(frame) != 5 or frame["candidate_id"].str.contains("_Mg_").any():
        raise AssertionError("candidate set is not the exact five-Cr freeze")
    return frame


def _serializable_incar(incar: Any) -> dict[str, Any]:
    payload = dict(incar)
    return {
        key: value.tolist() if hasattr(value, "tolist") else value
        for key, value in payload.items()
    }


def build_protocol_audit(repo_root: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    manifest = build_candidate_manifest(repo_root)
    structure = Structure.from_file(Path(manifest.iloc[0]["cif_path"]))
    relax_set = MPRelaxSet(structure, validate_magmom=True)
    static_set = MPStaticSet(structure, validate_magmom=True)
    relax_incar = _serializable_incar(relax_set.incar)
    static_incar = _serializable_incar(static_set.incar)

    smoke_entry = ComputedStructureEntry(
        structure=structure,
        energy=-100.0,
        entry_id="compatibility-smoke-job_092",
        parameters={
            "run_type": "GGA+U",
            "is_hubbard": True,
            "hubbards": {"Li": 0.0, "Cr": 3.7, "O": 0.0},
            "potcar_symbols": [
                LAST_AUDITED_SERVER_POTCAR_TITLES["Li"],
                LAST_AUDITED_SERVER_POTCAR_TITLES["Cr"],
                LAST_AUDITED_SERVER_POTCAR_TITLES["O"],
            ],
        },
    )
    compatibility = MaterialsProject2020Compatibility(check_potcar_hash=False)
    processed = compatibility.process_entry(smoke_entry, clean=True)
    smoke_adjustments = (
        [
            {
                "name": adjustment.name,
                "value_eV": float(adjustment.value),
                "uncertainty_eV": float(adjustment.uncertainty),
                "description": adjustment.description,
            }
            for adjustment in processed.energy_adjustments
        ]
        if processed is not None
        else []
    )

    potcar_matches = {
        element: (
            LAST_AUDITED_SERVER_POTCAR_TITLES[element]
            == OFFICIAL_MP_POTCAR_TITLES[element]
        )
        for element in OFFICIAL_MP_POTCAR_TITLES
    }
    checks = {
        "single_generator_family": True,
        "input_generator_version_fixed": True,
        "Cr_U_source_explicit": True,
        "compatibility_processing_smoke_success": processed is not None,
        "potcar_titles_match": all(potcar_matches.values()),
        "compatible_reference_entries_available": False,
        "formation_energy_from_output_recomputable": False,
        "candidate_and_competing_entries_same_compatibility": False,
        "remote_environment_live_reverified": False,
    }
    stop_reasons = [
        "POTCAR_TITLE_MISMATCH",
        "COMPATIBLE_REFERENCE_ENTRIES_UNAVAILABLE",
        "FORMATION_ENERGY_ROUNDTRIP_UNAVAILABLE",
        "CANDIDATE_COMPETITOR_COMPATIBILITY_UNRESOLVED",
        "REMOTE_ENVIRONMENT_NOT_LIVE_REVERIFIED",
    ]

    static_signature = inspect.signature(MPStaticSet.__init__)
    static_density = static_signature.parameters["reciprocal_density"].default
    protocol = {
        "schema": "PROSPECTIVE_CR_V2_MP_DFT_PROTOCOL_V1",
        "batch_id": BATCH_ID,
        "frozen_on": "2026-07-25",
        "submission_status": "STOP_SUBMISSION",
        "submission_allowed": False,
        "stop_reasons": stop_reasons,
        "margin_audit_decision": MARGIN_AUDIT_DECISION,
        "method_mainline": METHOD_MAINLINE,
        "generator": {
            "family": "pymatgen VASP input sets",
            "pymatgen_version": metadata.version("pymatgen"),
            "python_version_source": "runtime sys.version; recorded in runtime report",
            "relax_class": MPRelaxSet.__name__,
            "static_class": MPStaticSet.__name__,
            "atomate2_used": False,
            "workflow": [
                "MPRelaxSet relaxation 1 from frozen CIF",
                "MPRelaxSet relaxation 2 from relaxation-1 CONTCAR",
                "independent MPStaticSet from relaxation-2 CONTCAR",
            ],
            "MPRelaxSet_config_sha256": hashlib.sha256(
                json.dumps(MPRelaxSet.CONFIG, sort_keys=True).encode()
            ).hexdigest(),
            "MPStaticSet_config_sha256": hashlib.sha256(
                json.dumps(MPStaticSet.CONFIG, sort_keys=True).encode()
            ).hexdigest(),
        },
        "vasp": {
            "required_version": "6.5.1",
            "last_audited_binary_sha256": (
                "2abdcfedd1c3e7962a56404bd14cc340dcb170867720921fcea8ec7058ef3d94"
            ),
            "live_reverification": "FAILED_CONNECTION_RESET_BEFORE_AUTH",
            "scheduler": "none; wall segments controlled by runner when unblocked",
        },
        "potcar": {
            "functional": relax_set.potcar_functional,
            "symbols": relax_set.potcar_symbols,
            "official_mp_titles": OFFICIAL_MP_POTCAR_TITLES,
            "last_audited_server_titles": LAST_AUDITED_SERVER_POTCAR_TITLES,
            "last_audited_server_raw_sha256": LAST_AUDITED_SERVER_POTCAR_SHA256,
            "title_matches": potcar_matches,
            "bodies_copied_or_downloaded": False,
        },
        "settings": {
            "ENCUT_eV": float(relax_incar["ENCUT"]),
            "kpoint_generation": {
                "relax_reciprocal_density": MPRelaxSet.CONFIG["KPOINTS"][
                    "reciprocal_density"
                ],
                "static_reciprocal_density": static_density,
                "job_092_relax_mesh": list(relax_set.kpoints.kpts[0]),
                "job_092_static_mesh": list(static_set.kpoints.kpts[0]),
                "style": str(relax_set.kpoints.style.name),
            },
            "GGA_GGA_U": "PBE GGA+U for Cr oxide",
            "Cr_Ueff_eV": float(relax_incar["LDAUU"][1]),
            "Cr_U_source": (
                "pymatgen MPRelaxSet configuration and Materials Project "
                "Hubbard-U documentation"
            ),
            "LDAUTYPE": int(relax_incar["LDAUTYPE"]),
            "LDAUL_species_order_Li_Cr_O": relax_incar["LDAUL"],
            "LDAUJ_species_order_Li_Cr_O": relax_incar["LDAUJ"],
            "LASPH": bool(relax_incar["LASPH"]),
            "LMAXMIX": int(relax_incar["LMAXMIX"]),
            "ISPIN": int(relax_incar["ISPIN"]),
            "initial_MAGMOM_rule": {
                "Li_muB": 0.6,
                "Cr_muB": 5.0,
                "O_muB": 0.6,
                "job_092_site_vector_muB": relax_incar["MAGMOM"],
            },
            "EDIFF_eV_job_092": float(relax_incar["EDIFF"]),
            "EDIFF_rule": "5e-5 eV per atom from MPRelaxSet",
            "EDIFFG": None,
            "EDIFFG_rule": (
                "not written by MPRelaxSet; VASP positive energy-difference "
                "default follows EDIFF"
            ),
            "IBRION_relax": int(relax_incar["IBRION"]),
            "IBRION_static_effective": -1,
            "ISIF_relax": int(relax_incar["ISIF"]),
            "ISIF_static": None,
            "NSW_relax": int(relax_incar["NSW"]),
            "NSW_static": int(static_incar["NSW"]),
            "LREAL_relax": str(relax_incar["LREAL"]),
            "LREAL_static": bool(static_incar["LREAL"]),
            "PREC": str(relax_incar["PREC"]),
            "ALGO": str(relax_incar["ALGO"]),
            "NELM": int(relax_incar["NELM"]),
            "relax_smearing": {
                "ISMEAR": int(relax_incar["ISMEAR"]),
                "SIGMA_eV": float(relax_incar["SIGMA"]),
            },
            "static_smearing": {
                "ISMEAR": int(static_incar["ISMEAR"]),
                "SIGMA_eV": float(static_incar["SIGMA"]),
                "scheme": "tetrahedron with Bloechl corrections",
            },
        },
        "compatibility": {
            "class": "MaterialsProject2020Compatibility",
            "pymatgen_version": metadata.version("pymatgen"),
            "check_potcar_hash": False,
            "potcar_validation": (
                "exact server-side TITEL plus raw SHA-256 audit; POTCAR bodies "
                "remain server-side"
            ),
            "required_entry_parameters": [
                "run_type",
                "is_hubbard",
                "hubbards",
                "potcar_symbols",
            ],
            "real_output_status": "NOT_RUN",
        },
        "compatibility_smoke": {
            "success": processed is not None,
            "synthetic_energy_eV": -100.0,
            "not_a_scientific_result": True,
            "energy_adjustments": smoke_adjustments,
        },
        "formation_energy_and_phase_diagram": {
            "local_compatible_entry_cache": "NOT_FOUND",
            "mp_api_query": "HTTP_401_NO_API_KEY",
            "formation_energy_recomputable": False,
            "Li_Cr_O_phase_diagram_available": False,
            "policy": (
                "Do not combine prior internal elemental references or candidate "
                "energies with the frozen MP-compatible workflow."
            ),
        },
        "old_probe_policy": {
            "status": "RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY",
            "job_092_energy_reused": False,
            "job_239_continued": False,
            "warm_start_geometry_allowed_if_exact_match": True,
            "warm_start_geometry_used": False,
            "formal_job_092_start": "frozen original CIF",
        },
        "remote_reverification_attempts": [
            {
                "host": "connect.westb.seetacloud.com",
                "port": 35662,
                "result": "connection_reset_before_auth",
            },
            {
                "host": "connect.westb.seetacloud.com",
                "port": 53416,
                "result": "connection_reset_before_auth",
            },
        ],
        "checks": checks,
        "sources": {
            "mp_parameters": (
                "https://docs.materialsproject.org/methodology/"
                "materials-methodology/calculation-details/"
                "gga%2Bu-calculations/parameters-and-convergence"
            ),
            "mp_pseudopotentials": (
                "https://docs.materialsproject.org/methodology/"
                "materials-methodology/calculation-details/"
                "gga%2Bu-calculations/pseudopotentials"
            ),
            "mp_hubbard_u": (
                "https://docs.materialsproject.org/methodology/"
                "materials-methodology/calculation-details/"
                "gga%2Bu-calculations/hubbard-u-values"
            ),
            "mp_energy_corrections": (
                "https://docs.materialsproject.org/methodology/"
                "materials-methodology/thermodynamic-stability/"
                "thermodynamic-stability/anion-and-gga-gga%2Bu-mixing"
            ),
        },
    }
    return protocol


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        float_format="%.12g",
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _selection_protocol(manifest: pd.DataFrame) -> str:
    lines = [
        "# Prospective Cr v2 Selection Protocol",
        "",
        f"- Batch: `{BATCH_ID}`",
        "- Candidate count: exactly five independent Cr structures.",
        f"- Frozen ALIGNN target interval: `{TARGET_INTERVAL_EV_ATOM}` eV/atom.",
        "- Selection evidence: material-pool pre-audit only.",
        "- job_092/job_239 probe energies were not read into the selection rule.",
        "- No Mg candidate, replacement candidate, or sixth candidate is allowed.",
        f"- Margin audit decision: `{MARGIN_AUDIT_DECISION}`.",
        f"- Current method mainline: `{METHOD_MAINLINE}`.",
        "",
        "## Frozen candidates",
        "",
    ]
    for row in manifest.itertuples(index=False):
        lines.append(
            f"- `{row.candidate_id}` — {row.selection_reason}; "
            f"ALIGNN={row.alignn_formation_energy_eV_atom:.12g} eV/atom; "
            f"u={row.target_normalized_position:.12g}; "
            f"Gate round={row.gate_round}; Greedy round={row.greedy_round}; "
            f"SM={row.structure_matcher_cluster}; "
            f"fingerprint={row.fingerprint_cluster}."
        )
    return "\n".join(lines)


def _freeze_record(manifest: pd.DataFrame, protocol: dict[str, Any]) -> str:
    manifest_hash = hashlib.sha256(
        manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    return f"""# Prospective Cr v2 Freeze Record

- Freeze date: `2026-07-25`
- Batch ID: `{BATCH_ID}`
- Candidate manifest logical SHA-256: `{manifest_hash}`
- Exact candidate IDs: `{len(manifest)}`
- Candidate substitutions allowed: `no`
- Sixth candidate allowed: `no`
- Old probe status: `RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY`
- Formal job_092 warm start used: `no`
- Protocol submission state: `{protocol["submission_status"]}`
- Submission allowed: `no`

The candidate freeze is independent of the protocol stop. The Git tag
`prospective-cr-v2-frozen` records this selection and its fail-closed protocol
audit; it does not claim that any formal DFT calculation ran.
"""


def _protocol_audit_report(protocol: dict[str, Any]) -> str:
    potcar = protocol["potcar"]
    rows = []
    for element in ("Li", "Cr", "O"):
        rows.append(
            f"| {element} | `{potcar['official_mp_titles'][element]}` | "
            f"`{potcar['last_audited_server_titles'][element]}` | "
            f"{'PASS' if potcar['title_matches'][element] else 'FAIL'} |"
        )
    return f"""# MP DFT Protocol Audit

Decision: **STOP_SUBMISSION**

No VASP job was submitted. This is a protocol/environment stop, not a
candidate numerical failure.

## Generator

- Installed pymatgen: `{protocol["generator"]["pymatgen_version"]}`
- Workflow: `MPRelaxSet -> MPRelaxSet -> MPStaticSet`
- atomate2 used: `no`
- Compatibility processor: `MaterialsProject2020Compatibility`
- Compatibility smoke: `PASS` (synthetic entry only; not a scientific energy)

## Mandatory POTCAR title gate

| Element | Official MP GGA/GGA+U title | Last audited server title | Gate |
|---|---|---|---|
{chr(10).join(rows)}

Li and Cr fail exact-title matching. Symbols alone (`Li_sv`, `Cr_pv`, `O`)
are insufficient for the strict frozen thermodynamic口径.

## Formation-energy and phase-diagram gate

- Local same口径 Li-Cr-O entries: `not found`
- MP API query: `HTTP 401 — no API key`
- Candidate formation energy from a real output: `not recomputable`
- Candidate and competitors on one compatibility basis: `not demonstrated`

Existing internal elemental references and old probe energies are excluded.
They cannot be combined with this MP-compatible batch.

## Remote live re-verification

Both recorded DFT endpoints reset the SSH connection before authentication.
No remote write occurred and no POTCAR body was read, copied, or downloaded.

## Authoritative sources

- [MP parameters and convergence]({protocol["sources"]["mp_parameters"]})
- [MP pseudopotential titles]({protocol["sources"]["mp_pseudopotentials"]})
- [MP Hubbard U values]({protocol["sources"]["mp_hubbard_u"]})
- [MP energy corrections]({protocol["sources"]["mp_energy_corrections"]})
"""


def _protocol_freeze_report(protocol: dict[str, Any]) -> str:
    settings = protocol["settings"]
    return f"""# MP DFT Protocol Freeze

Status: `{protocol["submission_status"]}`; frozen for audit, not cleared to run.

## Common double-relax-static settings

- VASP required: `{protocol["vasp"]["required_version"]}`
- POTCAR functional: `{protocol["potcar"]["functional"]}`
- POTCAR symbols: `{", ".join(protocol["potcar"]["symbols"])}`
- ENCUT: `{settings["ENCUT_eV"]}` eV
- Relax k-point rule: reciprocal density `{settings["kpoint_generation"]["relax_reciprocal_density"]}`
- Static k-point rule: reciprocal density `{settings["kpoint_generation"]["static_reciprocal_density"]}`
- job_092 preview meshes: relax `{settings["kpoint_generation"]["job_092_relax_mesh"]}`, static `{settings["kpoint_generation"]["job_092_static_mesh"]}`
- Functional: PBE GGA+U for Cr oxide
- Cr Ueff: `{settings["Cr_Ueff_eV"]}` eV
- LASPH / LMAXMIX / ISPIN: `{settings["LASPH"]}` / `{settings["LMAXMIX"]}` / `{settings["ISPIN"]}`
- Initial MAGMOM: Li 0.6, Cr 5.0, O 0.6 μB
- EDIFF: `5e-5 eV/atom` (`{settings["EDIFF_eV_job_092"]}` eV for 7 atoms)
- EDIFFG: not written by MPRelaxSet; VASP positive energy default follows EDIFF
- IBRION: relax `{settings["IBRION_relax"]}`, static effective `{settings["IBRION_static_effective"]}`
- ISIF: relax `{settings["ISIF_relax"]}`, static not applicable
- NSW: relax `{settings["NSW_relax"]}`, static `{settings["NSW_static"]}`
- LREAL: relax `{settings["LREAL_relax"]}`, static `{settings["LREAL_static"]}`
- PREC / ALGO / NELM: `{settings["PREC"]}` / `{settings["ALGO"]}` / `{settings["NELM"]}`
- Relax smearing: ISMEAR `{settings["relax_smearing"]["ISMEAR"]}`, SIGMA `{settings["relax_smearing"]["SIGMA_eV"]}` eV
- Static smearing: ISMEAR `{settings["static_smearing"]["ISMEAR"]}`, SIGMA `{settings["static_smearing"]["SIGMA_eV"]}` eV; tetrahedron with Blöchl corrections

All five candidates would use this identical generator version and rule set.
Submission remains forbidden until every stop reason is cleared and the freeze
is reissued under a new version.
"""


def _empty_result_frames(manifest: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    results_rows = []
    roundtrip_rows = []
    adjustment_rows = []
    for row in manifest.itertuples(index=False):
        gate_earlier = row.gate_round < row.greedy_round
        results_rows.append(
            {
                "batch_id": BATCH_ID,
                "candidate_id": row.candidate_id,
                "execution_status": "NOT_SUBMITTED_PROTOCOL_STOP",
                "result_level": "NOT_EVALUATED",
                "formal_mp_workflow_complete": False,
                "warm_start_geometry_used": False,
                "relaxation_1_complete": False,
                "relaxation_2_complete": False,
                "static_complete": False,
                "electronic_converged": False,
                "ionic_converged": False,
                "alignn_formation_energy_eV_atom": (
                    row.alignn_formation_energy_eV_atom
                ),
                "mp_formation_energy_eV_atom": None,
                "alignn_minus_mp_eV_atom": None,
                "formation_energy_target_met": False,
                "final_space_group": None,
                "relative_volume_change_percent": None,
                "minimum_interatomic_distance_A": None,
                "Fmax_eV_A": None,
                "stress_kbar": None,
                "total_magnetic_moment_muB": None,
                "local_magnetic_moments_muB": None,
                "band_gap_eV": None,
                "electronic_character": None,
                "energy_above_hull_eV_atom": None,
                "convex_hull_complete": False,
                "magnetic_followup_required": None,
                "compatibility_processed": False,
                "compatibility_adjustment_total_eV": None,
                "gate_round": row.gate_round,
                "greedy_round": row.greedy_round,
                "gate_earlier_than_greedy": gate_earlier,
                "failure_class": "PROTOCOL_ENVIRONMENT_STOP",
                "failure_reason": (
                    "POTCAR title mismatch; compatible references unavailable; "
                    "remote environment not live-reverified"
                ),
            }
        )
        roundtrip_rows.append(
            {
                "candidate_id": row.candidate_id,
                "roundtrip_status": "NOT_RUN",
                "vasp_total_energy_eV": None,
                "compatibility_adjusted_energy_eV": None,
                "formation_energy_primary_eV_atom": None,
                "formation_energy_recomputed_eV_atom": None,
                "absolute_difference_eV_atom": None,
                "reason": "formal static output does not exist",
            }
        )
        adjustment_rows.append(
            {
                "candidate_id": row.candidate_id,
                "adjustment_status": "NOT_RUN",
                "adjustment_name": None,
                "adjustment_value_eV": None,
                "adjustment_uncertainty_eV": None,
                "compatibility_class": "MaterialsProject2020Compatibility",
                "metadata_status": "NO_REAL_ENTRY",
            }
        )
    return (
        pd.DataFrame(results_rows),
        pd.DataFrame(roundtrip_rows),
        pd.DataFrame(adjustment_rows),
    )


def _paper_usability(manifest: pd.DataFrame) -> str:
    lines = [
        "# Prospective Cr v2 Paper Usability",
        "",
        "No candidate is paper-usable as a formal MP DFT result in this stopped batch.",
        "A formation-energy hit must never be described as stability or novelty.",
        "",
    ]
    for row in manifest.itertuples(index=False):
        delta = row.greedy_round - row.gate_round
        timing = (
            f"yes, by {delta} rounds"
            if delta > 0
            else f"no, Gate was {-delta} rounds later"
            if delta < 0
            else "tie"
        )
        lines.extend(
            [
                f"## {row.candidate_id}",
                "",
                "- Formal MP workflow complete: `no`",
                "- Formation energy reportable: `no`",
                "- Predefined target satisfied: `not evaluated`",
                "- Magnetic follow-up needed: `not assessed`",
                "- Convex-hull analysis complete: `no`",
                "- Materials-discovery narrative value: `selection-only; no DFT claim`",
                f"- Gate earlier than Greedy: `{timing}`",
                "",
            ]
        )
    return "\n".join(lines)


def write_artifacts(repo_root: Path, output_root: Path | None = None) -> None:
    repo_root = Path(repo_root).resolve()
    destination = Path(output_root).resolve() if output_root else repo_root
    manifest = build_candidate_manifest(repo_root)
    protocol = build_protocol_audit(repo_root)
    results, roundtrip, adjustments = _empty_result_frames(manifest)

    _write_csv(
        manifest,
        destination / "manifests/prospective_dft_discovery_cr_v2.csv",
    )
    protocol_path = destination / "manifests/mp_dft_protocol_frozen.json"
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_SELECTION_PROTOCOL.md",
        _selection_protocol(manifest),
    )
    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_FREEZE_RECORD.md",
        _freeze_record(manifest, protocol),
    )
    _write_text(
        destination / "reports/MP_DFT_PROTOCOL_AUDIT.md",
        _protocol_audit_report(protocol),
    )
    _write_text(
        destination / "reports/MP_DFT_PROTOCOL_FREEZE.md",
        _protocol_freeze_report(protocol),
    )

    _write_csv(results, destination / "results/prospective_cr_v2_results.csv")
    _write_csv(
        roundtrip,
        destination / "results/prospective_cr_v2_roundtrip_check.csv",
    )
    _write_csv(
        adjustments,
        destination
        / "results/prospective_cr_v2_compatibility_adjustments.csv",
    )

    _write_text(
        destination / "reports/JOB_092_COMPLETE_CHAIN_REPORT.md",
        """# job_092 Complete Chain Report

Status: `NOT_SUBMITTED_PROTOCOL_STOP`

- Relaxation 1: not run
- Relaxation 2: not run
- Independent static: not run
- ComputedStructureEntry: not generated from a real output
- MP compatibility processing: synthetic smoke only; no real entry
- Formation energy and roundtrip: unavailable
- Final force, stress, structure, and magnetism: unavailable
- Warm-start geometry: not used
- Formal starting geometry when unblocked: frozen original CIF

The prior job_092/job_239 probe is retained only as
`RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY`. Its energy is excluded.
""",
    )
    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_RUNTIME_REPORT.md",
        """# Prospective Cr v2 Runtime Report

- Submitted VASP jobs: `0`
- Completed formal chains: `0/5`
- Scheduler: none
- Remote writes: `0`
- Continuations: `0`
- POTCAR bodies copied/downloaded: `0`
- Port 35662 live check: connection reset before authentication
- Port 53416 live check: connection reset before authentication

No wall-clock stop was interpreted as a material failure. The batch stopped
before submission on protocol and compatibility prerequisites.
""",
    )
    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_FINAL_REPORT.md",
        """# Prospective Cr v2 Final Report

Batch state: `NOT_SUBMITTED_PROTOCOL_STOP`

- Formal double-relax-static chains completed: `0/5`
- Reportable MP-compatible formation energies: `0/5`
- Target hits: `0` (not evaluated)
- Convex-hull results: `0`
- Stability-supported candidates: `0`
- Novel-material candidates: `0`

The candidate set is frozen, but the calculation batch is not cleared to run.
No old internal energy, probe energy, or synthetic compatibility-smoke energy
appears in the scientific result tables.
""",
    )
    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_FAILURES.md",
        """# Prospective Cr v2 Failures and Stops

## Batch-level protocol stops

1. `POTCAR_TITLE_MISMATCH`: audited Li and Cr POTCAR titles differ from the
   official MP GGA/GGA+U titles required by the frozen protocol.
2. `COMPATIBLE_REFERENCE_ENTRIES_UNAVAILABLE`: no local same口径 Li-Cr-O
   entry set is available; the MP API request returned HTTP 401 without a key.
3. `FORMATION_ENERGY_ROUNDTRIP_UNAVAILABLE`: a candidate output could not be
   converted into a reproducible compatible formation energy.
4. `REMOTE_ENVIRONMENT_NOT_LIVE_REVERIFIED`: recorded endpoints reset before
   authentication.

## Candidate failures

None. No candidate was submitted, so no electronic, ionic, structural, or
materials failure is recorded. All five candidates remain frozen; none was
deleted or replaced.
""",
    )
    _write_text(
        destination / "reports/PROSPECTIVE_CR_V2_PAPER_USABILITY.md",
        _paper_usability(manifest),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-root")
    args = parser.parse_args()
    write_artifacts(
        Path(args.repo_root),
        Path(args.output_root) if args.output_root else None,
    )
    print(
        json.dumps(
            {
                "batch_id": BATCH_ID,
                "candidate_count": 5,
                "submission_status": "STOP_SUBMISSION",
                "submitted_jobs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
