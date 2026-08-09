"""Audit and prepare the frozen five-candidate DFT diagnostic batch."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar


TARGET_LOW = -2.18
TARGET_HIGH = -2.02
CORE_LOW = -2.12
CORE_HIGH = -2.08

FUNCTIONAL_BY_ELEMENT = {
    "Cr": "GGA+U",
    "Mn": "GGA+U",
    "Mg": "PBE",
}

U_EFF_BY_ELEMENT = {"Cr": 3.7, "Mn": 3.9}

PAW_LABEL_BY_ELEMENT = {
    "Li": "PAW_PBE Li_sv 10Sep2004",
    "Cr": "PAW_PBE Cr_pv 02Aug2007",
    "Mn": "PAW_PBE Mn_pv 02Aug2007",
    "Mg": "PAW_PBE Mg_pv 13Apr2007",
    "O": "PAW_PBE O 08Apr2002",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _reference_supported_elements(references: pd.DataFrame) -> set[str]:
    supported: set[str] = set()
    for element, functional in FUNCTIONAL_BY_ELEMENT.items():
        available = set(
            references.loc[
                references["functional"].astype(str).eq(functional),
                "element",
            ].astype(str)
        )
        if {"Li", element, "O"}.issubset(available):
            supported.add(element)
    return supported


def select_frozen_candidates(
    pool: pd.DataFrame,
    proposed: pd.DataFrame,
    references: pd.DataFrame,
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Select the fixed 3+1+1 design before any new DFT result exists."""

    pool_indexed = pool.set_index("candidate_id", drop=False)
    ordered = proposed.sort_values("proposal_order")
    core_ids = ordered.loc[
        ordered["candidate_stratum"].eq("A_core_ALIGNN"), "candidate_id"
    ].astype(str).tolist()
    disagreement_ids = ordered.loc[
        ordered["candidate_stratum"].eq("D_high_model_disagreement"), "candidate_id"
    ].astype(str).tolist()
    if len(core_ids) < 3 or len(disagreement_ids) < 1:
        raise ValueError("proposal does not contain the required core/disagreement strata")
    selected_ids = core_ids[:3] + disagreement_ids[:1]
    if any(candidate_id not in pool_indexed.index for candidate_id in selected_ids):
        raise ValueError("proposed candidate is absent from the frozen pool")

    selected = pool_indexed.loc[selected_ids].copy()
    selected["candidate_stratum"] = [
        "A_core_ALIGNN",
        "A_core_ALIGNN",
        "A_core_ALIGNN",
        "D_high_model_disagreement",
    ]
    core = selected.loc[selected["candidate_stratum"].eq("A_core_ALIGNN")]
    if not core["alignn_formation_energy_eV_atom"].astype(float).between(
        CORE_LOW, CORE_HIGH, inclusive="both"
    ).all():
        raise ValueError("core candidate lies outside the frozen core-middle interval")
    if core["historical_dft_candidate"].map(_as_bool).any():
        raise ValueError("core candidate overlaps historical DFT")

    supported = _reference_supported_elements(references)
    used_clusters = set(selected["structure_matcher_cluster"].astype(str))
    eligible = pool.loc[
        ~pool["candidate_id"].astype(str).isin(selected_ids)
        & ~pool["historical_dft_candidate"].map(_as_bool)
        & pool["cif_exists"].map(_as_bool)
        & pool["m_element"].astype(str).isin(supported)
        & ~pool["structure_matcher_cluster"].astype(str).isin(used_clusters)
    ].copy()
    if eligible.empty:
        raise ValueError("no reference-supported cross-cluster random control is available")
    eligible["_random_key"] = eligible["candidate_id"].astype(str).map(
        lambda candidate_id: hashlib.sha256(
            f"{int(random_seed)}|{candidate_id}".encode("utf-8")
        ).hexdigest()
    )
    random_row = eligible.sort_values(["_random_key", "candidate_id"]).iloc[[0]].copy()
    random_row["candidate_stratum"] = "E_random_composition_cluster_control"
    random_row["random_selection_seed"] = int(random_seed)
    random_row["random_selection_key"] = random_row["_random_key"]
    random_row = random_row.drop(columns=["_random_key"])

    selected["random_selection_seed"] = pd.NA
    selected["random_selection_key"] = pd.NA
    result = pd.concat([selected, random_row], ignore_index=True, sort=False)
    if len(result) != 5 or result["candidate_id"].duplicated().any():
        raise ValueError("frozen selection is not exactly five independent candidates")
    if result["structure_matcher_cluster"].astype(str).duplicated().any():
        raise ValueError("frozen candidates must occupy distinct structure-matcher clusters")
    return result


def classify_same_scale(metadata: Mapping[str, object]) -> str:
    """Classify strict ALIGNN/DFT scale compatibility from explicit provenance."""

    required = {
        "label_reference_convention",
        "dft_reference_convention",
        "compatibility_transform_sha256",
    }
    if not required.issubset(metadata):
        return "UNRESOLVED"
    if (
        str(metadata["label_reference_convention"])
        == str(metadata["dft_reference_convention"])
        and str(metadata["compatibility_transform_sha256"]).strip()
    ):
        return "SAME_SCALE_CONFIRMED"
    return "UNRESOLVED"


def audit_reference_compatibility(
    selected: pd.DataFrame,
    references: pd.DataFrame,
) -> dict[str, object]:
    """Verify that every selected composition has a complete internal reference channel."""

    errors: list[str] = []
    for candidate in selected.itertuples(index=False):
        element = str(candidate.m_element)
        functional = FUNCTIONAL_BY_ELEMENT.get(element)
        if functional is None:
            errors.append(f"{element}: no frozen candidate functional/reference rule")
            continue
        channel = references.loc[references["functional"].astype(str).eq(functional)]
        for required_element in ("Li", element, "O"):
            rows = channel.loc[channel["element"].astype(str).eq(required_element)]
            if len(rows) != 1:
                errors.append(
                    f"{element}/{functional}: expected one {required_element} reference, found {len(rows)}"
                )
                continue
            row = rows.iloc[0]
            energy = pd.to_numeric(pd.Series([row.get("energy_per_atom_eV")]), errors="coerce").iloc[0]
            if pd.isna(energy):
                errors.append(f"{element}/{functional}: {required_element} reference energy missing")
            if "electronic_converged" in rows.columns and not _as_bool(
                row.get("electronic_converged")
            ):
                errors.append(f"{element}/{functional}: {required_element} reference not converged")
            label = str(row.get("paw_label", ""))
            if label and PAW_LABEL_BY_ELEMENT[required_element] not in label:
                errors.append(
                    f"{element}/{functional}: {required_element} PAW label mismatch ({label})"
                )
            if required_element == element and functional == "GGA+U":
                actual_u = pd.to_numeric(
                    pd.Series([row.get("Ueff_eV", row.get("Ueff_eV_atom", 0.0))]),
                    errors="coerce",
                ).fillna(0.0).iloc[0]
                if not math.isclose(
                    float(actual_u),
                    U_EFF_BY_ELEMENT[element],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    errors.append(
                        f"{element}/{functional}: Ueff {actual_u} does not match "
                        f"{U_EFF_BY_ELEMENT[element]}"
                    )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "reference_convention": "INTERNAL_SELF_CONSISTENT_PBE_GGA_U",
        "materials_project_compatible": False,
    }


def kmesh_for_structure(
    structure: Structure,
    *,
    spacing: float,
) -> tuple[int, int, int]:
    """Return a Gamma mesh with reciprocal-vector spacing no larger than requested."""

    if spacing <= 0:
        raise ValueError("reciprocal spacing must be positive")
    return tuple(
        max(1, math.ceil(float(length) / spacing))
        for length in structure.lattice.reciprocal_lattice.abc
    )


def _magmom(structure: Structure, *, element: str, state: str) -> str:
    if state not in {"FM", "AFM_or_ferri"}:
        raise ValueError(f"unsupported magnetic state: {state}")
    counts = structure.composition.get_el_amt_dict()
    if counts.get("Li") != 1 or counts.get(element) != 2 or counts.get("O") != 4:
        raise ValueError("expected ordered Li1 M2 O4 composition")
    if state == "FM":
        return f"1*0.6 2*{5.0 if element in U_EFF_BY_ELEMENT else 0.0:.1f} 4*0.6"
    if element in U_EFF_BY_ELEMENT:
        return "1*0.6 1*5.0 1*-5.0 4*0.6"
    return "1*0.6 2*0.0 2*0.6 2*-0.6"


def render_incar(
    structure: Structure,
    *,
    element: str,
    state: str,
    stage: str,
) -> str:
    """Render the frozen internal-protocol INCAR for one candidate stage."""

    if element not in FUNCTIONAL_BY_ELEMENT:
        raise ValueError(f"unsupported candidate element: {element}")
    if stage not in {"relax", "static"}:
        raise ValueError(f"unsupported DFT stage: {stage}")
    lines = [
        f"SYSTEM = minimal-dft-5 {element} {state} {stage}",
        "ENCUT = 520",
        "PREC = Normal",
        "EDIFF = 1E-6",
        "NELM = 160",
        "ALGO = Normal",
        "ISMEAR = 0",
        "SIGMA = 0.05",
        "ISPIN = 2",
        f"MAGMOM = {_magmom(structure, element=element, state=state)}",
        "LORBIT = 11",
        "LREAL = .FALSE.",
        "LASPH = .TRUE.",
        "ADDGRID = .TRUE.",
    ]
    if stage == "relax":
        lines.extend(["EDIFFG = -0.05", "IBRION = 2", "ISIF = 3", "NSW = 160"])
    else:
        lines.extend(["IBRION = -1", "NSW = 0"])
    if element in U_EFF_BY_ELEMENT:
        lines.extend(
            [
                "LDAU = .TRUE.",
                "LDAUTYPE = 2",
                "LDAUL = -1 2 -1",
                f"LDAUU = 0.0 {U_EFF_BY_ELEMENT[element]:.1f} 0.0",
                "LDAUJ = 0.0 0.0 0.0",
                "LMAXMIX = 4",
            ]
        )
    else:
        lines.append("LDAU = .FALSE.")
    lines.extend(["LWAVE = .FALSE.", "LCHARG = .FALSE.", ""])
    return "\n".join(lines)


def formation_energy_per_atom(
    *,
    total_energy_eV: float,
    counts: Mapping[str, float],
    references_eV_atom: Mapping[str, float],
) -> float:
    """Recompute the internal-reference formation energy in eV/atom."""

    missing = sorted(set(counts) - set(references_eV_atom))
    if missing:
        raise ValueError(f"missing elemental references: {missing}")
    atom_count = sum(float(value) for value in counts.values())
    if atom_count <= 0:
        raise ValueError("atom count must be positive")
    reference_total = sum(
        float(amount) * float(references_eV_atom[element])
        for element, amount in counts.items()
    )
    return (float(total_energy_eV) - reference_total) / atom_count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_structure(structure: Structure, element: str) -> Structure:
    order = {"Li": 0, element: 1, "O": 2}
    if set(structure.composition.get_el_amt_dict()) != set(order):
        raise ValueError(f"candidate composition is not Li-{element}-O")
    sites = sorted(structure.sites, key=lambda site: order[str(site.specie)])
    return Structure.from_sites(sites)


def _write_kpoints(path: Path, mesh: tuple[int, int, int]) -> None:
    path.write_text(
        "Gamma mesh from frozen reciprocal spacing <= 0.15 A^-1\n"
        "0\n"
        "Gamma\n"
        f"{mesh[0]} {mesh[1]} {mesh[2]}\n"
        "0 0 0\n",
        encoding="utf-8",
        newline="\n",
    )


def build_input_bundle(
    selected: pd.DataFrame,
    output_root: str | Path,
    *,
    same_scale_status: str,
) -> pd.DataFrame:
    """Write four POTCAR-free stage templates for each frozen candidate."""

    if len(selected) != 5 or selected["candidate_id"].astype(str).duplicated().any():
        raise ValueError("input bundle requires exactly five unique candidates")
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for position, row in enumerate(selected.itertuples(index=False), start=1):
        candidate_id = str(row.candidate_id)
        element = str(row.m_element)
        source = Path(str(row.cif_path))
        if not source.is_file():
            raise FileNotFoundError(source)
        expected_hash = str(getattr(row, "cif_sha256", "") or "").lower()
        actual_hash = _sha256(source)
        if expected_hash and expected_hash != actual_hash:
            raise ValueError(f"CIF hash mismatch: {candidate_id}")
        structure = _ordered_structure(Structure.from_file(source), element)
        composition = structure.composition.get_el_amt_dict()
        if composition != {"Li": 1.0, element: 2.0, "O": 4.0}:
            raise ValueError(f"unexpected candidate composition: {candidate_id}")
        mesh = kmesh_for_structure(structure, spacing=0.15)
        candidate_dir = f"candidate_{position:02d}_{candidate_id}"
        for stage in ("relax", "static"):
            for state in ("FM", "AFM_or_ferri"):
                relative = PurePosixPath(candidate_dir, stage, state)
                directory = root.joinpath(*relative.parts)
                directory.mkdir(parents=True)
                Poscar(structure).write_file(directory / "POSCAR")
                (directory / "INCAR").write_text(
                    render_incar(structure, element=element, state=state, stage=stage),
                    encoding="utf-8",
                    newline="\n",
                )
                _write_kpoints(directory / "KPOINTS", mesh)
                dependency = (
                    str(PurePosixPath(candidate_dir, "relax", state, "CONTCAR"))
                    if stage == "static"
                    else ""
                )
                metadata = {
                    "candidate_id": candidate_id,
                    "candidate_stratum": str(row.candidate_stratum),
                    "element": element,
                    "formula": str(row.formula),
                    "functional": FUNCTIONAL_BY_ELEMENT[element],
                    "Ueff_eV": U_EFF_BY_ELEMENT.get(element, 0.0),
                    "stage": stage,
                    "magnetic_state": state,
                    "same_scale_status": same_scale_status,
                    "result_classification_ceiling": (
                        "INTERNAL_PROTOCOL_DIAGNOSTIC"
                        if same_scale_status != "SAME_SCALE_CONFIRMED"
                        else "DFT_EVALUATED_CANDIDATE"
                    ),
                    "source_cif_path": str(source),
                    "source_cif_sha256": actual_hash,
                    "element_order": ["Li", element, "O"],
                    "paw_labels": [
                        PAW_LABEL_BY_ELEMENT["Li"],
                        PAW_LABEL_BY_ELEMENT[element],
                        PAW_LABEL_BY_ELEMENT["O"],
                    ],
                    "kpoint_spacing_Ainv": 0.15,
                    "kpoints_mesh": list(mesh),
                    "poscar_dependency": dependency or None,
                    "static_poscar_must_be_replaced_by_relax_contcar": stage == "static",
                    "potcar_policy": "server-side assembly only; never archive or transfer",
                }
                (directory / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
                (directory / "README.md").write_text(
                    f"# {candidate_id}: {stage}/{state}\n\n"
                    "This is a frozen input template. Licensed PAW data are assembled only on "
                    "the authorized server and removed after execution.\n\n"
                    + (
                        f"The POSCAR must be replaced by `{dependency}` after that relaxation "
                        "passes its dependency gate.\n"
                        if dependency
                        else "This relaxation starts from the frozen source CIF.\n"
                    ),
                    encoding="utf-8",
                    newline="\n",
                )
                input_hashes = {
                    name: _sha256(directory / name)
                    for name in ("POSCAR", "INCAR", "KPOINTS", "metadata.json", "README.md")
                }
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_stratum": str(row.candidate_stratum),
                        "element": element,
                        "functional": FUNCTIONAL_BY_ELEMENT[element],
                        "stage": stage,
                        "magnetic_state": state,
                        "relative_stage_dir": str(relative),
                        "poscar_dependency": dependency,
                        "kpoints_mesh": "x".join(str(value) for value in mesh),
                        "config_sha256": hashlib.sha256(
                            json.dumps(input_hashes, sort_keys=True).encode("utf-8")
                        ).hexdigest(),
                        "input_hashes_json": json.dumps(input_hashes, sort_keys=True),
                    }
                )
    frame = pd.DataFrame(records)
    frame.to_csv(root / "stage_manifest.csv", index=False, lineterminator="\n")
    if any(path.name == "POTCAR" for path in root.rglob("*")):
        raise ValueError("POTCAR content must not exist in the local input bundle")
    return frame


def candidate_probe_passes(rows: pd.DataFrame) -> bool:
    """Apply the fixed operational probe gate to both magnetic branches."""

    required_columns = {
        "magnetic_state",
        "stage",
        "exit_code",
        "electronic_converged",
        "ionic_converged",
        "nsw_limit_reached",
        "structure_collapsed",
        "formation_energy_recomputed",
    }
    if not required_columns.issubset(rows.columns):
        return False
    expected = {
        (state, stage)
        for state in ("FM", "AFM_or_ferri")
        for stage in ("relax", "static")
    }
    observed = set(zip(rows["magnetic_state"].astype(str), rows["stage"].astype(str)))
    if observed != expected or len(rows) != 4:
        return False
    for row in rows.itertuples(index=False):
        if int(row.exit_code) != 0 or not _as_bool(row.electronic_converged):
            return False
        if _as_bool(row.structure_collapsed):
            return False
        if row.stage == "relax" and not (
            _as_bool(row.ionic_converged) or _as_bool(row.nsw_limit_reached)
        ):
            return False
        if row.stage == "static" and not _as_bool(row.formation_energy_recomputed):
            return False
    return True
