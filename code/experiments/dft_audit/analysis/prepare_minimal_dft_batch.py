#!/usr/bin/env python3
"""Prepare and audit the frozen five-candidate internal-protocol DFT batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from analysis.minimal_dft_batch import (
    CORE_HIGH,
    CORE_LOW,
    FUNCTIONAL_BY_ELEMENT,
    PAW_LABEL_BY_ELEMENT,
    TARGET_HIGH,
    TARGET_LOW,
    U_EFF_BY_ELEMENT,
    audit_reference_compatibility,
    build_input_bundle,
    classify_same_scale,
    formation_energy_per_atom,
    select_frozen_candidates,
)


RANDOM_SEED = 20260722
PROTOCOL_VERSION = "minimal_dft_5_internal_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
    ).strip()


def read_alignn_metadata(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as handle:
        config = json.loads(handle.read("mp_e_form_alignnn/config.json"))
        checkpoint = handle.read("mp_e_form_alignnn/checkpoint_300.pt")
    return {
        "dataset": config.get("dataset"),
        "target": config.get("target"),
        "model_version": config.get("version"),
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "recovered_fields": ["dataset", "target", "model_version", "checkpoint_sha256"],
        "unrecovered_fields": [
            "exact elemental reference states",
            "oxygen correction",
            "GGA/GGA+U compatibility transform",
            "database snapshot and correction scheme",
        ],
    }


def reference_smoke_test(repo_root: Path, references: pd.DataFrame) -> dict[str, Any]:
    source = (
        repo_root
        / "results"
        / "post_submission_analysis"
        / "egdatpp_psfix_v1_20260719T031102Z"
        / "dft"
        / "recomputed_formation_energies.csv"
    )
    rows = pd.read_csv(source)
    ref_index = references.set_index("reference_id")
    differences: list[float] = []
    checked = 0
    for row in rows.loc[rows["selected_for_formation_energy"].astype(bool)].itertuples(
        index=False
    ):
        counts = json.loads(row.composition_json)
        ids = json.loads(row.elemental_reference_ids_json)
        channel = {
            element: float(ref_index.loc[ids[element], "energy_per_atom_eV"])
            for element in counts
        }
        recomputed = formation_energy_per_atom(
            total_energy_eV=float(row.final_total_energy_eV),
            counts=counts,
            references_eV_atom=channel,
        )
        differences.append(abs(recomputed - float(row.formation_energy_eV_per_atom)))
        checked += 1
    maximum = max(differences, default=float("inf"))
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "rows_checked": checked,
        "maximum_absolute_difference_eV_atom": maximum,
        "pass": checked > 0 and maximum <= 1e-9,
    }


def enrich_selection(selected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected.to_dict(orient="records"), start=1):
        structure = Structure.from_file(row["cif_path"])
        distances = structure.distance_matrix.copy()
        distances[distances == 0] = float("inf")
        element = str(row["m_element"])
        status = str(row.get("previous_dft_nearby_status", "UNKNOWN"))
        record = {
            **row,
            "frozen_order": index,
            "formula": structure.composition.formula,
            "candidate_stratum": row["candidate_stratum"],
            "functional": FUNCTIONAL_BY_ELEMENT[element],
            "Ueff_eV": U_EFF_BY_ELEMENT.get(element, 0.0),
            "normalized_interval_position": (
                float(row["alignn_formation_energy_eV_atom"]) - TARGET_LOW
            )
            / (TARGET_HIGH - TARGET_LOW),
            "cif_sha256": sha256(Path(row["cif_path"])),
            "atom_count": len(structure),
            "cell_volume_A3": structure.volume,
            "volume_per_atom_A3": structure.volume / len(structure),
            "minimum_pair_distance_A": float(distances.min()),
            "composition_validity": (
                "PASS"
                if structure.composition.get_el_amt_dict()
                == {"Li": 1.0, element: 2.0, "O": 4.0}
                else "FAIL"
            ),
            "historical_dft_duplicate": bool(row.get("historical_dft_candidate", False)),
            "historical_structure_cluster_status": status,
            "training_set_overlap_status": "UNKNOWN_NOT_RETESTED",
            "paw_labels": " | ".join(
                [
                    PAW_LABEL_BY_ELEMENT["Li"],
                    PAW_LABEL_BY_ELEMENT[element],
                    PAW_LABEL_BY_ELEMENT["O"],
                ]
            ),
            "selection_reason": (
                "fixed audited core-middle proposal"
                if row["candidate_stratum"] == "A_core_ALIGNN"
                else "highest-information audited model-disagreement proposal"
                if row["candidate_stratum"] == "D_high_model_disagreement"
                else "fixed-seed random control after pre-result reference-coverage and "
                "cross-cluster eligibility filters"
            ),
        }
        rows.append(record)
    return pd.DataFrame(rows)


def duplicate_audit(repo_root: Path, selected: pd.DataFrame) -> dict[str, Any]:
    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )
    structures = {
        row.candidate_id: Structure.from_file(row.cif_path)
        for row in selected.itertuples(index=False)
    }
    pair_matches: list[list[str]] = []
    ids = list(structures)
    for index, first in enumerate(ids):
        for second in ids[index + 1 :]:
            if matcher.fit(structures[first], structures[second]):
                pair_matches.append([first, second])

    historical = pd.read_csv(repo_root / "dft" / "audit" / "dft_candidate_manifest.csv")
    historical_matches: list[list[str]] = []
    for history in historical.itertuples(index=False):
        path = repo_root / str(history.final_cif_path)
        if not path.is_file():
            continue
        old = Structure.from_file(path)
        for candidate_id, structure in structures.items():
            if matcher.fit(structure, old):
                historical_matches.append([candidate_id, str(history.candidate_id)])
    return {
        "matcher": {
            "ltol": 0.2,
            "stol": 0.3,
            "angle_tol": 5,
            "primitive_cell": True,
            "scale": True,
            "attempt_supercell": False,
        },
        "selected_pair_matches": pair_matches,
        "historical_matches": historical_matches,
        "pass": not pair_matches and not historical_matches,
    }


def protocol_payload(
    *,
    selection_code_commit: str,
    same_scale_status: str,
    reference_audit: dict[str, Any],
    formation_smoke: dict[str, Any],
    duplicate_result: dict[str, Any],
    remote_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    remote_pass = bool(remote_audit and remote_audit.get("pass"))
    local_pass = (
        reference_audit["status"] == "PASS"
        and formation_smoke["pass"]
        and duplicate_result["pass"]
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_commit": selection_code_commit,
        "target_interval_eV_atom": [TARGET_LOW, TARGET_HIGH],
        "core_middle_interval_eV_atom": [CORE_LOW, CORE_HIGH],
        "same_scale_status": same_scale_status,
        "result_classification_ceiling": (
            "DFT_EVALUATED_CANDIDATE"
            if same_scale_status == "SAME_SCALE_CONFIRMED"
            else "INTERNAL_PROTOCOL_DIAGNOSTIC"
        ),
        "reference_convention": "INTERNAL_SELF_CONSISTENT_PBE_GGA_U",
        "materials_project_compatible": False,
        "functional_rule": FUNCTIONAL_BY_ELEMENT,
        "Ueff_eV": U_EFF_BY_ELEMENT,
        "shared_candidate_settings": {
            "VASP": "6.5.1",
            "ENCUT_eV": 520,
            "PREC": "Normal (explicitly frozen historical default)",
            "reciprocal_spacing_Ainv": 0.15,
            "EDIFF_eV": 1e-6,
            "EDIFFG_eV_A": -0.05,
            "IBRION_relax": 2,
            "ISIF_relax": 3,
            "NSW_relax": 160,
            "IBRION_static": -1,
            "NSW_static": 0,
            "LASPH": True,
            "LMAXMIX_d_electrons": 4,
            "ISMEAR": 0,
            "SIGMA_eV": 0.05,
            "LREAL": False,
            "ALGO": "Normal",
            "NELM": 160,
            "ISPIN": 2,
            "LORBIT": 11,
            "ISYM": "historical default; magnetic pattern initialized explicitly",
            "parallel": "serial VASP; OPENBLAS_NUM_THREADS=8; no NCORE/KPAR",
        },
        "smearing_rule": (
            "Unknown pre-run electronic character: primary static uses ISMEAR=0, SIGMA=0.05; "
            "entropy term and electronic character are audited after execution. "
            "No blind tetrahedron assignment."
        ),
        "magnetism_scope": "two tested magnetic initializations; not exhaustive",
        "reference_audit": reference_audit,
        "formation_energy_recomputation_smoke": formation_smoke,
        "duplicate_audit": duplicate_result,
        "remote_audit": remote_audit or {"status": "PENDING"},
        "local_pass": local_pass,
        "remote_pass": remote_pass,
        "overall_status": "PASS" if local_pass and remote_pass else "REMOTE_PENDING"
        if local_pass
        else "FAIL",
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    clean = frame[columns].copy().fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in clean.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def prepare(repo_root: Path, alignn_archive: Path, remote_audit_path: Path | None) -> int:
    reports = repo_root / "reports"
    manifests = repo_root / "manifests"
    batch_root = repo_root / "dft" / "minimal_dft_5_frozen"
    reports.mkdir(exist_ok=True)
    manifests.mkdir(exist_ok=True)

    pool = pd.read_csv(repo_root / "candidate_pool_master.csv")
    proposed = pd.read_csv(repo_root / "manifests" / "proposed_dft_audit_candidates.csv")
    references = pd.read_csv(repo_root / "dft" / "results" / "elemental_references.csv")
    selected = select_frozen_candidates(
        pool,
        proposed,
        references,
        random_seed=RANDOM_SEED,
    )
    selected = enrich_selection(selected)
    selected["frozen_order"] = range(1, 6)
    selected = selected.sort_values("frozen_order").reset_index(drop=True)

    label_metadata = read_alignn_metadata(alignn_archive)
    same_scale_status = classify_same_scale(label_metadata)
    reference_audit = audit_reference_compatibility(selected, references)
    formation_smoke = reference_smoke_test(repo_root, references)
    duplicates = duplicate_audit(repo_root, selected)
    remote_audit = (
        json.loads(remote_audit_path.read_text(encoding="utf-8"))
        if remote_audit_path is not None
        else None
    )
    code_commit = git_head(repo_root)
    payload = protocol_payload(
        selection_code_commit=code_commit,
        same_scale_status=same_scale_status,
        reference_audit=reference_audit,
        formation_smoke=formation_smoke,
        duplicate_result=duplicates,
        remote_audit=remote_audit,
    )
    payload["alignn_label_metadata"] = label_metadata

    manifest_columns = [
        "frozen_order",
        "candidate_id",
        "formula",
        "m_element",
        "candidate_stratum",
        "alignn_formation_energy_eV_atom",
        "second_model_energy_eV_atom",
        "absolute_model_disagreement_eV_atom",
        "structure_matcher_cluster",
        "fingerprint_cluster",
        "gate_round",
        "greedy_round",
        "cif_path",
        "cif_sha256",
        "minimum_pair_distance_A",
        "cell_volume_A3",
        "volume_per_atom_A3",
        "atom_count",
        "composition_validity",
        "historical_dft_duplicate",
        "historical_structure_cluster_status",
        "functional",
        "Ueff_eV",
        "paw_labels",
        "selection_reason",
        "random_selection_seed",
        "random_selection_key",
    ]
    selected_manifest = selected.reindex(columns=manifest_columns)
    csv_path = manifests / "minimal_dft_5_candidates.csv"
    json_path = manifests / "minimal_dft_5_candidates.json"
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        if old["candidate_id"].astype(str).tolist() != selected_manifest[
            "candidate_id"
        ].astype(str).tolist():
            raise RuntimeError("frozen candidate manifest already exists with different candidates")
    else:
        selected_manifest.to_csv(csv_path, index=False, lineterminator="\n")
        write_json(json_path, selected_manifest.where(pd.notna(selected_manifest), None).to_dict("records"))

    input_root = batch_root / "inputs"
    if input_root.exists():
        stage_manifest = pd.read_csv(input_root / "stage_manifest.csv")
        if set(stage_manifest["candidate_id"]) != set(selected_manifest["candidate_id"]):
            raise RuntimeError("existing input bundle does not match the frozen candidates")
        if any(path.name == "POTCAR" for path in input_root.rglob("*")):
            raise RuntimeError("local input bundle contains forbidden POTCAR content")
    else:
        stage_manifest = build_input_bundle(
            selected_manifest,
            input_root,
            same_scale_status=same_scale_status,
        )

    write_json(manifests / "dft_protocol_frozen.json", payload)
    candidate_table = markdown_table(
        selected_manifest,
        [
            "frozen_order",
            "candidate_id",
            "formula",
            "candidate_stratum",
            "alignn_formation_energy_eV_atom",
            "second_model_energy_eV_atom",
            "structure_matcher_cluster",
            "functional",
        ],
    )
    reports.joinpath("DFT_PROTOCOL_COMPATIBILITY_AUDIT.md").write_text(
        "# DFT Protocol Compatibility Audit\n\n"
        f"- Protocol: `{PROTOCOL_VERSION}`\n"
        f"- Local audit: **{'PASS' if payload['local_pass'] else 'FAIL'}**\n"
        f"- Remote environment: **{payload['remote_audit'].get('status', 'PENDING')}**\n"
        f"- SAME_SCALE_STATUS: **{same_scale_status}**\n"
        "- Reference convention: **internal self-consistent PBE/PBE+U**; not "
        "Materials Project compatible.\n"
        "- New energies are capped at `INTERNAL_PROTOCOL_DIAGNOSTIC` while the exact "
        "MEGNET/ALIGNN label reference and compatibility transform remain unrecovered.\n\n"
        "## Protocol findings\n\n"
        "- VASP 6.5.1, PAW-PBE, ENCUT=520 eV, explicit Gamma meshes at <=0.15 "
        "A^-1, EDIFF=1e-6 eV, EDIFFG=-0.05 eV/A, LASPH, and LREAL=.FALSE. are frozen.\n"
        "- Cr uses Dudarev PBE+U with Ueff=3.7 eV; Mg uses PBE. Every selected "
        "composition has Li/M/O references in its matching internal channel.\n"
        "- FM and one AFM/ferrimagnetic initialization are compared. This is not an "
        "exhaustive magnetic-state search.\n"
        "- The primary static uses ISMEAR=0, SIGMA=0.05 because electronic character "
        "is unknown before calculation; entropy and electronic character are checked "
        "afterward.\n"
        f"- Formation-energy recomputation smoke: {formation_smoke['rows_checked']} rows, "
        f"max difference {formation_smoke['maximum_absolute_difference_eV_atom']:.3e} eV/atom.\n"
        f"- Candidate duplicate audit: selected pairs={len(duplicates['selected_pair_matches'])}, "
        f"historical matches={len(duplicates['historical_matches'])}.\n\n"
        "## Frozen candidates\n\n"
        f"{candidate_table}\n",
        encoding="utf-8",
        newline="\n",
    )
    reports.joinpath("DFT_PROTOCOL_PASS_FAIL.md").write_text(
        "# DFT Protocol PASS/FAIL\n\n"
        f"**{payload['overall_status']}**\n\n"
        f"- local_pass: `{payload['local_pass']}`\n"
        f"- remote_pass: `{payload['remote_pass']}`\n"
        f"- SAME_SCALE_STATUS: `{same_scale_status}`\n"
        f"- reference_errors: `{json.dumps(reference_audit['errors'])}`\n"
        f"- duplicate_errors: `{json.dumps(duplicates)}`\n\n"
        "Submission is authorized only when this result is `PASS`. `UNRESOLVED` "
        "same-scale status does not block the internal diagnostic calculation, but "
        "strict ALIGNN-interval hit language remains prohibited.\n",
        encoding="utf-8",
        newline="\n",
    )
    freeze_record = reports / "MINIMAL_DFT_FREEZE_RECORD.md"
    if not freeze_record.exists():
        freeze_record.write_text(
            "# Minimal DFT 5 Freeze Record\n\n"
            f"- Frozen before new DFT output: `{datetime.now(timezone.utc).isoformat()}`\n"
            f"- Selection-code Git commit: `{code_commit}`\n"
            f"- Target interval: `[{TARGET_LOW}, {TARGET_HIGH}] eV/atom`\n"
            f"- Core interval: `[{CORE_LOW}, {CORE_HIGH}] eV/atom`\n"
            f"- Fixed random seed: `{RANDOM_SEED}`\n"
            f"- Candidate CSV SHA-256: `{sha256(csv_path)}`\n"
            f"- Candidate JSON SHA-256: `{sha256(json_path)}`\n"
            "- Candidate substitution after freeze: prohibited.\n"
            "- Failed calculations must be retained.\n"
            "- Licensed POTCAR content is excluded from this repository and package.\n\n"
            f"{candidate_table}\n",
            encoding="utf-8",
            newline="\n",
        )
    reports.joinpath("MINIMAL_DFT_CANDIDATE_REVIEW.md").write_text(
        "# Minimal DFT Candidate Review\n\n"
        "The three core candidates come from the previously untested core-middle "
        "stratum. The disagreement candidate is an adjacent, high-disagreement Mg "
        "case. The original unfrozen Co random control was excluded before any new "
        "result because no complete Co elemental-reference channel exists. The final "
        "random control was selected deterministically from reference-supported, "
        "uncomputed, cross-cluster candidates.\n\n"
        f"{candidate_table}\n",
        encoding="utf-8",
        newline="\n",
    )
    reports.joinpath("MINIMAL_DFT_DUPLICATE_AUDIT.md").write_text(
        "# Minimal DFT Duplicate Audit\n\n"
        f"- Selected-pair StructureMatcher matches: `{duplicates['selected_pair_matches']}`\n"
        f"- Historical-final-structure matches: `{duplicates['historical_matches']}`\n"
        f"- Result: **{'PASS' if duplicates['pass'] else 'FAIL'}**\n",
        encoding="utf-8",
        newline="\n",
    )
    reports.joinpath("MINIMAL_DFT_STRUCTURE_AUDIT.md").write_text(
        "# Minimal DFT Structure Audit\n\n"
        + markdown_table(
            selected_manifest,
            [
                "candidate_id",
                "composition_validity",
                "minimum_pair_distance_A",
                "cell_volume_A3",
                "atom_count",
                "historical_structure_cluster_status",
            ],
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    reports.joinpath("MINIMAL_DFT_INPUT_PREFLIGHT.md").write_text(
        "# Minimal DFT Input Preflight\n\n"
        f"- Stage templates: `{len(stage_manifest)}` (5 candidates x 2 states x 2 stages)\n"
        "- Local POTCAR files: `0`\n"
        "- Static POSCAR dependency: each frozen template is replaced by its matching "
        "relaxation CONTCAR on the authorized server.\n"
        "- Remote submission remains blocked until `DFT_PROTOCOL_PASS_FAIL.md` is PASS.\n",
        encoding="utf-8",
        newline="\n",
    )
    reports.joinpath("MINIMAL_DFT_RESOURCE_REQUEST_DRAFT.md").write_text(
        "# Minimal DFT Resource Request\n\n"
        "- Execution: licensed VASP 6.5.1 serial binary, eight OpenBLAS threads per job.\n"
        "- Probe phase: 2 candidates x 2 magnetic relaxations, followed by dependent "
        "statics; start at concurrency 1 and increase only from observed headroom.\n"
        "- Expected wall time: approximately 1-3 hours for the probe phase under the "
        "historical serial timing envelope; exact time depends on relaxation steps.\n"
        "- Remaining phase: 3 candidates only if both probes pass.\n"
        "- No scheduler is assumed; durable manifest and controller logs are required.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": payload["overall_status"], "candidates": selected_manifest["candidate_id"].tolist()}))
    return 0 if payload["local_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--alignn-archive",
        type=Path,
        default=Path.home() / "Downloads" / "mp_e_form_alignnn.zip",
    )
    parser.add_argument("--remote-audit-json", type=Path)
    args = parser.parse_args()
    return prepare(args.repo_root.resolve(), args.alignn_archive, args.remote_audit_json)


if __name__ == "__main__":
    raise SystemExit(main())
