"""Freeze and generate the current self-consistent PAW-PBE+U Cr batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar


BATCH_ID = "prospective_cr_discovery_v2"
PROTOCOL_NAME = "CURRENT_SELF_CONSISTENT_PAW_PBE_U"
EXPECTED_CANDIDATE_IDS = [
    "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_3",
    "job_196_Cr_fe_-0.819_n4_generated_crystals_cif__gen_1",
    "job_234_Cr_fe_-1.123_n4_generated_crystals_cif__gen_3",
    "job_079_Cr_fe_-0.854_n4_generated_crystals_cif__gen_1",
    "job_126_Cr_fe_-0.901_n4_generated_crystals_cif__gen_0",
]
TARGET_INTERVAL = (-2.18, -2.02)
SELECTION_REASONS = {
    EXPECTED_CANDIDATE_IDS[0]: "ALIGNN core-middle; first execution gate",
    EXPECTED_CANDIDATE_IDS[1]: "ALIGNN core-middle",
    EXPECTED_CANDIDATE_IDS[2]: "pre-result rank plus independent structure cluster",
    EXPECTED_CANDIDATE_IDS[3]: "pre-result rank plus independent structure cluster",
    EXPECTED_CANDIDATE_IDS[4]: "pre-frozen cross-cluster Cr control",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _minimum_distance(structure: Structure) -> float:
    matrix = structure.distance_matrix
    return min(
        float(matrix[i, j])
        for i in range(len(structure))
        for j in range(i + 1, len(structure))
    )


def build_candidate_preflight(repo_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = Path(repo_root).resolve()
    pool = pd.read_csv(root / "candidate_pool_master.csv")
    indexed = pool.set_index("candidate_id", drop=False)
    missing = [value for value in EXPECTED_CANDIDATE_IDS if value not in indexed.index]
    if missing:
        raise ValueError(f"missing frozen candidates: {missing}")

    structures: dict[str, Structure] = {}
    rows: list[dict[str, Any]] = []
    for order, candidate_id in enumerate(EXPECTED_CANDIDATE_IDS, 1):
        source = indexed.loc[candidate_id]
        if isinstance(source, pd.DataFrame):
            raise ValueError(f"duplicate candidate row: {candidate_id}")
        cif_path = Path(str(source["cif_path"])).resolve()
        if not cif_path.is_file():
            raise FileNotFoundError(cif_path)
        observed_hash = _sha256(cif_path)
        if observed_hash != str(source["cif_sha256"]):
            raise ValueError(f"CIF hash drift: {candidate_id}")
        structure = Structure.from_file(cif_path)
        structures[candidate_id] = structure
        rows.append(
            {
                "frozen_order": order,
                "batch_id": BATCH_ID,
                "candidate_id": candidate_id,
                "formula": structure.composition.formula,
                "cif_path": str(cif_path),
                "cif_sha256": observed_hash,
                "alignn_formation_energy_eV_atom": float(
                    source["alignn_formation_energy_eV_atom"]
                ),
                "target_lower_eV_atom": TARGET_INTERVAL[0],
                "target_upper_eV_atom": TARGET_INTERVAL[1],
                "target_normalized_position": float(source["alignn_u"]),
                "gate_round": int(source["gate_round"]),
                "greedy_round": int(source["greedy_round"]),
                "gate_rank": int(source["gate_rank"]),
                "greedy_rank": int(source["greedy_rank"]),
                "structure_matcher_cluster": str(
                    source["structure_matcher_cluster"]
                ),
                "fingerprint_cluster": str(source["fingerprint_cluster"]),
                "historical_dft_candidate": bool(
                    source["historical_dft_candidate"]
                ),
                "historical_dft_nearby_status": str(
                    source["previous_dft_nearby_status"]
                ),
                "minimum_interatomic_distance_A": _minimum_distance(structure),
                "lattice_a_A": float(structure.lattice.a),
                "lattice_b_A": float(structure.lattice.b),
                "lattice_c_A": float(structure.lattice.c),
                "lattice_alpha_deg": float(structure.lattice.alpha),
                "lattice_beta_deg": float(structure.lattice.beta),
                "lattice_gamma_deg": float(structure.lattice.gamma),
                "lattice_determinant_A3": float(
                    np.linalg.det(structure.lattice.matrix)
                ),
                "cell_volume_A3": float(structure.volume),
                "atom_count": len(structure),
                "selection_reason": SELECTION_REASONS[candidate_id],
                "selection_evidence": (
                    "material-pool pre-audit only; probe energies excluded"
                ),
            }
        )

    matcher = StructureMatcher(primitive_cell=False, scale=True)
    selected_matches: list[list[str]] = []
    for left_index, left in enumerate(EXPECTED_CANDIDATE_IDS):
        for right in EXPECTED_CANDIDATE_IDS[left_index + 1 :]:
            if matcher.fit(structures[left], structures[right]):
                selected_matches.append([left, right])

    historical_matches: list[dict[str, str]] = []
    history_path = root / "dft" / "audit" / "dft_candidate_manifest.csv"
    if history_path.is_file():
        history = pd.read_csv(history_path)
        for historical in history.itertuples(index=False):
            final_value = str(getattr(historical, "final_cif_path", ""))
            if not final_value or final_value.lower() == "nan":
                continue
            final_path = Path(final_value)
            if not final_path.is_absolute():
                final_path = root / final_path
            if not final_path.is_file():
                continue
            try:
                prior = Structure.from_file(final_path)
            except Exception:
                continue
            if prior.composition.reduced_formula != "LiCr2O4":
                continue
            for candidate_id, structure in structures.items():
                if matcher.fit(structure, prior):
                    historical_matches.append(
                        {
                            "candidate_id": candidate_id,
                            "historical_candidate_id": str(
                                getattr(historical, "candidate_id", "")
                            ),
                            "historical_cif_path": str(final_path.resolve()),
                        }
                    )

    frame = pd.DataFrame(rows)
    audit = {
        "schema": "PROSPECTIVE_CR_DISCOVERY_V2_PREFLIGHT_V1",
        "candidate_ids": EXPECTED_CANDIDATE_IDS,
        "candidate_count": len(frame),
        "all_cifs_parse": True,
        "all_lattices_nondegenerate": bool(
            (frame["lattice_determinant_A3"].abs() > 1.0).all()
        ),
        "all_minimum_distances_above_1p2_A": bool(
            (frame["minimum_interatomic_distance_A"] > 1.2).all()
        ),
        "all_formula_LiCr2O4": set(frame["formula"]) == {"Li1 Cr2 O4"},
        "selected_pair_matches": selected_matches,
        "historical_matches": historical_matches,
        "status": "PASS",
    }
    if (
        selected_matches
        or historical_matches
        or not audit["all_lattices_nondegenerate"]
        or not audit["all_minimum_distances_above_1p2_A"]
        or not audit["all_formula_LiCr2O4"]
    ):
        audit["status"] = "FAIL"
    return frame, audit


def kmesh_for_structure(
    structure: Structure, *, spacing_Ainv: float
) -> tuple[int, int, int]:
    reciprocal_lengths = structure.lattice.reciprocal_lattice.abc
    return tuple(max(1, math.ceil(value / spacing_Ainv)) for value in reciprocal_lengths)


def _sorted_structure(structure: Structure) -> Structure:
    order = {"Li": 0, "Cr": 1, "O": 2}
    return structure.get_sorted_structure(
        key=lambda site: order.get(site.specie.symbol, 99)
    )


def _magmom(structure: Structure) -> list[float]:
    values = {"Li": 0.6, "Cr": 5.0, "O": 0.6}
    return [values[site.specie.symbol] for site in structure]


def _incar(
    structure: Structure,
    *,
    stage: str,
    reference_element: str,
) -> Incar:
    symbols: list[str] = []
    for site in structure:
        if site.specie.symbol not in symbols:
            symbols.append(site.specie.symbol)
    settings: dict[str, Any] = {
        "SYSTEM": f"{PROTOCOL_NAME}_{stage}",
        "GGA": "PE",
        "PREC": "Accurate",
        "ENCUT": 649,
        "EDIFF": 1e-6,
        "ALGO": "Normal",
        "NELM": 200,
        "ISPIN": 2,
        "MAGMOM": (
            [1.0 for _ in structure]
            if reference_element == "O"
            else _magmom(structure)
        ),
        "LORBIT": 11,
        "LREAL": False,
        "LASPH": True,
        "LMAXMIX": 4,
        "ISMEAR": 0,
        "SIGMA": 0.05,
        "LWAVE": False,
        "LCHARG": True,
    }
    if "Cr" in symbols:
        settings.update(
            {
                "LDAU": True,
                "LDAUTYPE": 2,
                "LDAUL": [2 if symbol == "Cr" else -1 for symbol in symbols],
                "LDAUU": [3.7 if symbol == "Cr" else 0.0 for symbol in symbols],
                "LDAUJ": [0.0 for _ in symbols],
            }
        )
    if stage == "relax":
        settings.update(
            {
                "IBRION": 2,
                "ISIF": 2 if reference_element == "O" else 3,
                "NSW": 200,
                "EDIFFG": -0.05,
            }
        )
    else:
        settings.update({"IBRION": -1, "NSW": 0})
    return Incar(settings)


def _reference_structures() -> dict[str, Structure]:
    lithium = Structure(
        Lattice.cubic(3.49),
        ["Li", "Li"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    chromium = Structure(
        Lattice.cubic(2.88),
        ["Cr", "Cr"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    oxygen = Structure(
        Lattice.cubic(15.0),
        ["O", "O"],
        [[0.5, 0.5, (7.5 - 0.605) / 15.0], [0.5, 0.5, (7.5 + 0.605) / 15.0]],
    )
    return {"Li": lithium, "Cr": chromium, "O": oxygen}


def build_input_package(
    repo_root: Path,
    output_dir: Path,
    *,
    potcar_audit: dict[str, Any],
) -> pd.DataFrame:
    if potcar_audit.get("status") != "PASS":
        raise ValueError("POTCAR audit is not PASS")
    root = Path(repo_root).resolve()
    package = Path(output_dir).resolve()
    package.mkdir(parents=True, exist_ok=True)
    candidates, preflight = build_candidate_preflight(root)
    if preflight["status"] != "PASS":
        raise ValueError(f"candidate preflight failed: {preflight}")

    specs: list[dict[str, Any]] = []
    for element, structure in _reference_structures().items():
        for stage in ("relax", "static_0p15"):
            specs.append(
                {
                    "job_id": f"reference_{element}_{stage}",
                    "job_kind": "reference",
                    "element": element,
                    "candidate_id": "",
                    "stage": stage,
                    "structure": structure,
                    "spacing": 0.15,
                }
            )
    for row in candidates.itertuples(index=False):
        stages = ["relax", "static_0p15"]
        if row.candidate_id == EXPECTED_CANDIDATE_IDS[0]:
            stages.append("static_0p10")
        for stage in stages:
            specs.append(
                {
                    "job_id": f"{row.candidate_id}__{stage}",
                    "job_kind": "candidate",
                    "element": "",
                    "candidate_id": row.candidate_id,
                    "stage": stage,
                    "structure": Structure.from_file(row.cif_path),
                    "spacing": 0.10 if stage == "static_0p10" else 0.15,
                }
            )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        structure = _sorted_structure(spec["structure"])
        job_dir = package / "inputs" / spec["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        stage_class = "relax" if spec["stage"] == "relax" else "static"
        incar = _incar(
            structure,
            stage=stage_class,
            reference_element=spec["element"],
        )
        if spec["element"] == "O":
            kpoints = Kpoints.gamma_automatic((1, 1, 1))
            mesh = (1, 1, 1)
        else:
            mesh = kmesh_for_structure(
                structure, spacing_Ainv=float(spec["spacing"])
            )
            kpoints = Kpoints.gamma_automatic(mesh)
        Poscar(structure).write_file(job_dir / "POSCAR")
        incar.write_file(job_dir / "INCAR")
        kpoints.write_file(job_dir / "KPOINTS")
        metadata = {
            "protocol": PROTOCOL_NAME,
            "batch_id": BATCH_ID,
            "job_id": spec["job_id"],
            "stage": spec["stage"],
            "source_is_placeholder_for_static": stage_class == "static",
            "static_geometry_source": (
                f"{spec['candidate_id'] or 'reference_' + spec['element']} relax CONTCAR"
                if stage_class == "static"
                else None
            ),
            "potcar_hashes": {
                element: row["sha256"]
                for element, row in potcar_audit["potcars"].items()
            },
        }
        (job_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rows.append(
            {
                "job_id": spec["job_id"],
                "job_kind": spec["job_kind"],
                "element": spec["element"],
                "candidate_id": spec["candidate_id"],
                "stage": spec["stage"],
                "ENCUT_eV": 649,
                "Ueff_eV": 3.7 if "Cr" in structure.symbol_set else 0.0,
                "reciprocal_spacing_Ainv": spec["spacing"],
                "kmesh": " ".join(str(value) for value in mesh),
                "same_scale_status": "UNRESOLVED",
                "protocol_name": PROTOCOL_NAME,
                "input_path": str(job_dir),
            }
        )

    jobs = pd.DataFrame(rows)
    jobs.to_csv(package / "job_manifest.csv", index=False, lineterminator="\n")
    package_manifest = {
        "schema": "CURRENT_SELF_CONSISTENT_DFT_INPUT_PACKAGE_V1",
        "batch_id": BATCH_ID,
        "protocol_name": PROTOCOL_NAME,
        "candidate_preflight": preflight,
        "potcar_manifest_status": potcar_audit["status"],
        "potcar_hashes": {
            element: row["sha256"]
            for element, row in potcar_audit["potcars"].items()
        },
        "contains_potcar_bodies": False,
        "exact_mp_compatibility_claimed": False,
        "same_scale_status": "UNRESOLVED",
        "job_count": len(jobs),
    }
    (package / "package_manifest.json").write_text(
        json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return jobs


def build_protocol_manifest(potcar_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "CURRENT_SELF_CONSISTENT_DFT_PROTOCOL_V1",
        "protocol_name": PROTOCOL_NAME,
        "batch_id": BATCH_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "exact_mp_compatibility_claimed": False,
            "energy_name": "self-consistent PAW-PBE+U formation energy",
            "same_scale_status": "UNRESOLVED",
            "alignn_target_interval_eV_atom": list(TARGET_INTERVAL),
            "alignn_target_is_not_a_dft_acceptance_interval": True,
            "margin_audit_decision": "REMOVE_FROM_CORE_CLAIM",
            "method_mainline": (
                "Group-concentration-triggered diversity correction"
            ),
        },
        "software": {
            "vasp_version": potcar_audit["vasp"]["version"],
            "vasp_binary_path": potcar_audit["vasp"]["path"],
            "vasp_binary_sha256": potcar_audit["vasp"]["sha256"],
            "python_local": sys.version.split()[0],
            "platform_local": platform.platform(),
            "pymatgen_local": metadata.version("pymatgen"),
            "remote_python_and_pymatgen": potcar_audit["remote_python"],
        },
        "potcars": {
            "functional": "PBE",
            "family": "PAW",
            "bodies_in_reproducibility_package": False,
            "metadata_manifest": "manifests/POTCAR_HASH_MANIFEST.json",
            "hashes": {
                element: row["sha256"]
                for element, row in potcar_audit["potcars"].items()
            },
            "titles": {
                element: row["titel"]
                for element, row in potcar_audit["potcars"].items()
            },
        },
        "electronic": {
            "exchange_correlation": "PBE",
            "hubbard_scheme": "Dudarev",
            "Cr_Ueff_eV": 3.7,
            "Cr_elemental_reference_Ueff_eV": 3.7,
            "Cr_reference_note": (
                "U is applied to elemental Cr to preserve the frozen internal "
                "calculation rule. The resulting scale is protocol-specific "
                "and is not Materials Project legacy thermochemistry."
            ),
            "PREC": "Accurate",
            "ENCUT_eV": potcar_audit["encut_eV"],
            "ENCUT_formula_eV": potcar_audit["encut_formula_eV"],
            "LASPH": True,
            "LMAXMIX": 4,
            "ISPIN": 2,
            "LORBIT": 11,
            "LREAL": False,
            "EDIFF_eV": 1e-6,
            "ALGO": "Normal",
            "NELM": 200,
            "initial_MAGMOM_muB": {"Li": 0.6, "Cr": 5.0, "O": 0.6},
        },
        "kpoints": {
            "style": "Gamma-centered regular mesh",
            "rule": "ceil(|b_i| / reciprocal_spacing)",
            "primary_reciprocal_spacing_Ainv": 0.15,
            "representative_check_reciprocal_spacing_Ainv": 0.10,
            "representative": EXPECTED_CANDIDATE_IDS[0],
            "energy_tolerance_meV_atom": 2.0,
        },
        "relax": {
            "workflow": (
                "continuous relaxation; each wall-time segment restarts from "
                "the newest valid CONTCAR; no duplicate formal relax"
            ),
            "IBRION": 2,
            "ISIF_candidates_and_solids": 3,
            "ISIF_O2": 2,
            "NSW_per_segment": 200,
            "EDIFFG_eV_A": -0.05,
            "primary_Fmax_threshold_eV_A": 0.05,
            "strict_followup_Fmax_threshold_eV_A": [0.02, 0.03],
            "ISMEAR": 0,
            "SIGMA_eV": 0.05,
            "walltime_timeout_is_scientific_failure": False,
        },
        "static": {
            "independent_from_final_relax_CONTCAR": True,
            "IBRION": -1,
            "NSW": 0,
            "ISMEAR_primary": 0,
            "SIGMA_primary_eV": 0.05,
            "insulator_optional_ISMEAR_check": -5,
            "entropy_term_checked": True,
        },
        "reference_states": {
            "Li": {
                "structure": "bcc conventional cell",
                "initial_a_A": 3.49,
                "energy_normalization": "per atom",
            },
            "Cr": {
                "structure": "bcc conventional cell",
                "initial_a_A": 2.88,
                "energy_normalization": "per atom",
                "Ueff_eV": 3.7,
            },
            "O2": {
                "cell": "15 A cubic",
                "initial_bond_length_A": 1.21,
                "spin_initialization": "triplet-like, 1.0 muB per O",
                "cell_fixed": True,
                "energy_normalization": "per molecule",
            },
        },
        "formation_energy": {
            "formula": (
                "(E[LiCr2O4] - E[Li] - 2 E[Cr] - 2 E[O2]) / 7"
            ),
            "independent_implementations": [
                "analysis/compute_self_consistent_fe.py",
                "analysis/recompute_self_consistent_fe_decimal.py",
            ],
            "roundtrip_tolerance_eV_atom": 1e-6,
        },
        "execution_gate": {
            "first_candidate": EXPECTED_CANDIDATE_IDS[0],
            "remaining_candidates": EXPECTED_CANDIDATE_IDS[1:],
            "remaining_concurrency_initial": 2,
            "remaining_launch_only_after_first_pass": True,
        },
        "old_probe_policy": {
            "status": "RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY",
            "old_energies_reused": False,
            "job_239_or_Mg_continued": False,
            "job_092_warm_start_allowed_only_if_all_critical_settings_match": True,
            "formal_default": "start from frozen original CIF",
        },
        "result_levels": [
            "DFT_EVALUATED",
            "SELF_CONSISTENT_FE_CANDIDATE",
            "STABILITY_SUPPORTED_CANDIDATE",
            "PREVIOUSLY_UNREPORTED_COMPUTATIONAL_CANDIDATE",
        ],
    }


def _write_freeze_record(
    frame: pd.DataFrame,
    preflight: dict[str, Any],
    protocol: dict[str, Any],
    jobs: pd.DataFrame,
    path: Path,
) -> None:
    candidate_hash = hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    job_hash = hashlib.sha256(
        jobs.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()
    lines = [
        "# Prospective Cr v2 Freeze Record",
        "",
        f"- Batch: `{BATCH_ID}`",
        f"- Protocol: `{PROTOCOL_NAME}`",
        f"- Candidate count: `{len(frame)}` (no substitutions or sixth candidate)",
        f"- Candidate manifest logical SHA-256: `{candidate_hash}`",
        f"- Input job manifest logical SHA-256: `{job_hash}`",
        f"- Candidate preflight: **{preflight['status']}**",
        f"- POTCAR audit: **{protocol['potcars'] and 'PASS'}**",
        "- Exact MP compatibility claimed: `no`",
        "- SAME_SCALE_STATUS: `UNRESOLVED`",
        "- Old job_092/job_239 status: `RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY`",
        "- Warm-start geometry for formal job_092: `not used; original frozen CIF`",
        "- Selection evidence excludes all probe intermediate energies.",
        "- Requested tag name previously dereferenced to "
        "`8210326f286b31770995fe540742384bbccdf6e5`; the modern protocol freeze "
        "supersedes that legacy-MP stop record and preserves the prior commit.",
        "",
        "## Frozen candidates",
        "",
        "| Order | Candidate | CIF SHA-256 | ALIGNN (eV/atom) | u | Gate | Greedy | SM | FP | dmin (Å) | Reason |",
        "|---:|---|---|---:|---:|---:|---:|---|---|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.frozen_order} | `{row.candidate_id}` | `{row.cif_sha256}` | "
            f"{row.alignn_formation_energy_eV_atom:.6f} | "
            f"{row.target_normalized_position:.6f} | {row.gate_round} | "
            f"{row.greedy_round} | `{row.structure_matcher_cluster}` | "
            f"`{row.fingerprint_cluster}` | "
            f"{row.minimum_interatomic_distance_A:.6f} | {row.selection_reason} |"
        )
    lines.extend(
        [
            "",
            "## Preflight checks",
            "",
            f"- CIF parse: `{preflight['all_cifs_parse']}`",
            f"- Nondegenerate lattice: `{preflight['all_lattices_nondegenerate']}`",
            f"- Minimum distance > 1.2 Å: `{preflight['all_minimum_distances_above_1p2_A']}`",
            f"- Formula LiCr2O4: `{preflight['all_formula_LiCr2O4']}`",
            f"- Selected StructureMatcher duplicates: `{preflight['selected_pair_matches']}`",
            f"- Historical DFT StructureMatcher duplicates: `{preflight['historical_matches']}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol_report(protocol: dict[str, Any], path: Path) -> None:
    text = f"""# Self-Consistent DFT Protocol

- Frozen protocol: `{protocol["protocol_name"]}`
- VASP: `{protocol["software"]["vasp_version"]}`
- VASP binary SHA-256: `{protocol["software"]["vasp_binary_sha256"]}`
- Energy label: `self-consistent PAW-PBE+U formation energy`
- Exact Materials Project compatibility: `not claimed`
- SAME_SCALE_STATUS against ALIGNN: `UNRESOLVED`

## Frozen electronic settings

PBE PAW, Dudarev Cr Ueff = 3.7 eV, ENCUT = 649 eV, PREC=Accurate,
LASPH=.TRUE., LMAXMIX=4, ISPIN=2, LORBIT=11, LREAL=.FALSE.,
EDIFF=1e-6 eV, ALGO=Normal, and NELM=200. Initial moments are
Li=0.6, Cr=5.0, and O=0.6 μB per site.

Elemental Cr also uses Ueff=3.7 eV. This makes the formation-energy scale
an explicitly protocol-dependent internal PAW-PBE+U scale; it is not MP
legacy thermochemistry.

## Relaxation and static

Each structure undergoes continuous relaxation with IBRION=2, ISIF=3,
NSW=200 per wall-time segment, ISMEAR=0, SIGMA=0.05 eV, and
EDIFFG=-0.05 eV/Å. A wall-time stop resumes from the newest valid CONTCAR.
After Fmax <= 0.05 eV/Å, an independent static calculation is generated from
that CONTCAR. Static uses fixed ions/cell, IBRION=-1, NSW=0, and the same
electronic settings. A promising 1–2 candidates may later receive a stricter
0.02–0.03 eV/Å relaxation and magnetic/U/k-point robustness checks.

## K points

Gamma-centered meshes use `N_i = ceil(|b_i| / spacing)`. The primary spacing
is 0.15 Å^-1. job_092 also receives a 0.10 Å^-1 static check; the required
energy difference is <=2 meV/atom.

## Reference states and formation energy

Li and Cr use bcc conventional cells initialized at 3.49 and 2.88 Å.
O2 uses a 15 Å cubic cell, 1.21 Å initial bond, fixed cell, and triplet-like
1 μB per O initialization. The formula is
`(E[LiCr2O4] - E[Li] - 2 E[Cr] - 2 E[O2]) / 7`.
Two independent scripts must agree within 1e-6 eV/atom.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--potcar-manifest", default="manifests/POTCAR_HASH_MANIFEST.json"
    )
    parser.add_argument(
        "--package", default="dft/prospective_cr_discovery_v2/input_package"
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    potcar_audit = json.loads(
        (root / args.potcar_manifest).read_text(encoding="utf-8")
    )
    frame, preflight = build_candidate_preflight(root)
    if preflight["status"] != "PASS":
        raise SystemExit(f"candidate preflight failed: {preflight}")
    manifest_path = root / "manifests/prospective_cr_discovery_v2.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest_path, index=False, lineterminator="\n")
    package_path = root / args.package
    jobs = build_input_package(root, package_path, potcar_audit=potcar_audit)
    protocol = build_protocol_manifest(potcar_audit)
    protocol_path = root / "manifests/self_consistent_dft_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_freeze_record(
        frame,
        preflight,
        protocol,
        jobs,
        root / "reports/PROSPECTIVE_CR_V2_FREEZE_RECORD.md",
    )
    _write_protocol_report(
        protocol, root / "reports/SELF_CONSISTENT_DFT_PROTOCOL.md"
    )
    print(
        json.dumps(
            {
                "preflight": preflight["status"],
                "candidate_count": len(frame),
                "job_count": len(jobs),
                "package": str(package_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
