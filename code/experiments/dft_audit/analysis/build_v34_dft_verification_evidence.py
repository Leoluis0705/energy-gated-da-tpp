#!/usr/bin/env python3
"""Merge gated C120/C214 verification results into a v34 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


VERIFIED_CANDIDATES = ("C120", "C214")


def _required_unique(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if frame.duplicated(columns).any():
        raise ValueError(f"duplicate {description}: {columns}")


def _long_ids(main_text: pd.DataFrame) -> dict[str, str]:
    indexed = main_text.set_index("candidate_label", drop=False)
    missing = [candidate for candidate in VERIFIED_CANDIDATES if candidate not in indexed.index]
    if missing:
        raise ValueError(f"historical main-text table is missing: {missing}")
    return {candidate: str(indexed.loc[candidate, "candidate_id"]) for candidate in VERIFIED_CANDIDATES}


def merge_verification_evidence(
    historical_main: pd.DataFrame,
    historical_energy: pd.DataFrame,
    historical_structure: pd.DataFrame,
    selected: pd.DataFrame,
    statics: pd.DataFrame,
    relaxations: pd.DataFrame,
    review: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Return updated tables only when the generated scientific gate authorizes it."""

    if review.get("paper_conclusion_update_authorized") is not True:
        reasons = "; ".join(str(item) for item in review.get("pause_reasons", []))
        raise ValueError(f"paper conclusion update is paused: {reasons or 'unspecified reason'}")
    selected = selected.copy()
    statics = statics.copy()
    relaxations = relaxations.copy()
    if set(selected["candidate_id"].astype(str)) != set(VERIFIED_CANDIDATES):
        raise ValueError("selected verification rows must be exactly C120 and C214")
    expected_states = {
        (candidate, state)
        for candidate in VERIFIED_CANDIDATES
        for state in ("state_afm", "state_fm")
    }
    if set(zip(statics["candidate_id"].astype(str), statics["magnetic_initialization"].astype(str))) != expected_states:
        raise ValueError("static metrics must contain both tested initializations for C120 and C214")
    if set(zip(relaxations["candidate_id"].astype(str), relaxations["magnetic_initialization"].astype(str))) != expected_states:
        raise ValueError("relaxation metrics must contain both tested initializations for C120 and C214")
    _required_unique(selected, ["candidate_id"], "selected candidate")
    _required_unique(statics, ["candidate_id", "magnetic_initialization"], "static state")
    _required_unique(relaxations, ["candidate_id", "magnetic_initialization"], "relaxation state")

    main = historical_main.copy(deep=True)
    energies = historical_energy.copy(deep=True)
    structures = historical_structure.copy(deep=True)
    long_ids = _long_ids(main)
    selected_index = selected.set_index("candidate_id", drop=False)
    static_index = statics.set_index(["candidate_id", "magnetic_initialization"], drop=False)
    relax_index = relaxations.set_index(["candidate_id", "magnetic_initialization"], drop=False)

    for candidate in VERIFIED_CANDIDATES:
        long_id = long_ids[candidate]
        selection = selected_index.loc[candidate]
        state = str(selection["new_selected_initialization"])
        static = static_index.loc[(candidate, state)]
        main_mask = main["candidate_label"].astype(str).eq(candidate)
        main.loc[main_mask, "selected_magnetic_initialization"] = state
        main.loc[main_mask, "recomputed_formation_energy_eV_per_atom"] = float(
            selection["new_selected_formation_energy_eV_per_atom"]
        )
        main.loc[main_mask, "verification_formation_energy_shift_eV_per_atom"] = float(
            selection["selected_formation_energy_shift_eV_per_atom"]
        )
        main.loc[main_mask, "verification_status"] = "completed_frozen_protocol_relaxation_and_static"

        energy_mask = (
            energies["candidate_id"].astype(str).eq(long_id)
            & energies["functional"].astype(str).eq("GGA+U")
            & energies["selected_for_formation_energy"].astype(str).str.lower().isin(["true", "1"])
        )
        if int(energy_mask.sum()) != 1:
            raise ValueError(f"historical selected GGA+U energy row is not unique: {candidate}")
        updates = {
            "job_id": static["job_id"],
            "magnetic_initialization": state,
            "final_total_energy_eV": float(static["final_total_energy_eV"]),
            "formation_energy_eV_per_atom": float(selection["new_selected_formation_energy_eV_per_atom"]),
            "source_output_path": static["source_output_path"],
            "outcar_sha256": static["outcar_sha256"],
            "verification_decision": "COMPLETED_RECONSTRUCTED_VERIFICATION",
        }
        for column, value in updates.items():
            energies.loc[energy_mask, column] = value

        for tested_state in ("state_afm", "state_fm"):
            static_row = static_index.loc[(candidate, tested_state)]
            relax_row = relax_index.loc[(candidate, tested_state)]
            structure_mask = (
                structures["candidate_id"].astype(str).eq(long_id)
                & structures["functional"].astype(str).eq("GGA+U")
                & structures["magnetic_initialization"].astype(str).eq(tested_state)
            )
            if int(structure_mask.sum()) != 1:
                raise ValueError(f"historical structure row is not unique: {candidate}/{tested_state}")
            structure_updates = {
                "verification_decision": "COMPLETED_RECONSTRUCTED_VERIFICATION",
                "source_output_path": static_row["source_output_path"],
                "outcar_sha256": static_row["outcar_sha256"],
                "minimum_interatomic_distance_A": float(static_row["minimum_interatomic_distance_A"]),
                "minimum_M_O_distance_A": float(static_row["minimum_M_O_distance_A"]),
                "Fmax_eV_A_static_diagnostic": float(static_row["Fmax_eV_A_static_diagnostic"]),
                "final_total_energy_eV": float(static_row["final_total_energy_eV"]),
                "final_total_magnetic_moment": static_row["final_total_magnetic_moment"],
                "final_space_group": static_row["final_space_group"],
                "verification_cif_path": static_row["final_cif_path"],
                "verification_cif_sha256": static_row["final_cif_sha256"],
                "verification_relaxation_initial_volume_A3": float(relax_row["initial_volume_A3"]),
                "verification_relaxation_final_volume_A3": float(relax_row["final_volume_A3"]),
                "verification_relative_volume_change_percent": float(
                    relax_row["relative_volume_change_percent"]
                ),
                "verification_maximum_internal_displacement_A": float(
                    relax_row["maximum_internal_displacement_A"]
                ),
                "verification_relaxation_Fmax_eV_A": float(relax_row["Fmax_eV_A"]),
                "verification_initial_space_group": relax_row["initial_space_group"],
                "verification_final_space_group": relax_row["final_space_group"],
                "verification_relaxation_output_path": relax_row["source_output_path"],
            }
            for column, value in structure_updates.items():
                structures.loc[structure_mask, column] = value

    return {
        "main_text": main,
        "formation_energies": energies,
        "structure_metrics": structures,
        "selected_comparison": selected.reset_index(drop=True),
        "verification_statics": statics.reset_index(drop=True),
        "verification_relaxations": relaxations.reset_index(drop=True),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_cifs(
    frame: pd.DataFrame,
    *,
    source_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    archived = frame.copy()
    output_root.mkdir(parents=True, exist_ok=False)
    for index, row in archived.iterrows():
        filename = Path(str(row["final_cif_path"])).name
        source = Path(source_root) / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing recovered verification CIF: {source}")
        expected = str(row["final_cif_sha256"])
        observed = _sha256(source)
        if observed != expected:
            raise ValueError(
                f"verification CIF SHA-256 mismatch: {source}; "
                f"expected={expected}; observed={observed}"
            )
        target = output_root / filename
        if target.exists():
            raise FileExistsError(f"duplicate verification CIF target: {target}")
        shutil.copy2(source, target)
        archived.loc[index, "final_cif_path"] = str(target.resolve())
    return archived


def _write_outputs(
    outputs: dict[str, pd.DataFrame],
    output: Path,
    sources: list[Path],
    *,
    static_cif_root: Path,
    relaxation_cif_root: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    static = _archive_cifs(
        outputs["verification_statics"],
        source_root=static_cif_root,
        output_root=output / "verification_static_cifs",
    )
    relaxations = _archive_cifs(
        outputs["verification_relaxations"],
        source_root=relaxation_cif_root,
        output_root=output / "verification_relaxation_cifs",
    )
    structures = outputs["structure_metrics"].copy()
    for row in static.itertuples(index=False):
        mask = structures["outcar_sha256"].astype(str).eq(str(row.outcar_sha256))
        if int(mask.sum()) != 1:
            raise ValueError(
                f"verification structure/CIF row is not unique: {row.job_id}"
            )
        structures.loc[mask, "verification_cif_path"] = row.final_cif_path
    outputs = {
        **outputs,
        "verification_statics": static,
        "verification_relaxations": relaxations,
        "structure_metrics": structures,
    }
    names = {
        "main_text": "main_text_table7_comparison.csv",
        "formation_energies": "recomputed_formation_energies.csv",
        "structure_metrics": "structure_metrics.csv",
        "selected_comparison": "selected_candidate_comparison.csv",
        "verification_statics": "verification_static_metrics.csv",
        "verification_relaxations": "verification_relaxation_metrics.csv",
    }
    for key, name in names.items():
        outputs[key].to_csv(output / name, index=False, lineterminator="\n")
    manifest = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sources
    ]
    (output / "source_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-dft", type=Path, required=True)
    parser.add_argument("--verification-relaxation-root", type=Path, required=True)
    parser.add_argument("--verification-static-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    historical = args.historical_dft
    relaxation = args.verification_relaxation_root
    static = args.verification_static_root
    sources = [
        historical / "main_text_table7_comparison.csv",
        historical / "recomputed_formation_energies.csv",
        historical / "structure_metrics.csv",
        relaxation / "structure_metrics.csv",
        static / "main_candidate_verification_statics.csv",
        static / "selected_candidate_comparison.csv",
        static / "conclusion_update_review.json",
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing v34 verification evidence: " + "; ".join(missing))
    review = json.loads((static / "conclusion_update_review.json").read_text(encoding="utf-8"))
    outputs = merge_verification_evidence(
        pd.read_csv(sources[0]),
        pd.read_csv(sources[1]),
        pd.read_csv(sources[2]),
        pd.read_csv(static / "selected_candidate_comparison.csv"),
        pd.read_csv(static / "main_candidate_verification_statics.csv"),
        pd.read_csv(relaxation / "structure_metrics.csv"),
        review,
    )
    _write_outputs(
        outputs,
        args.output,
        sources,
        static_cif_root=static / "final_cifs",
        relaxation_cif_root=relaxation / "final_cifs",
    )
    print(json.dumps({"status": "PASS", "output": str(args.output), "files": len(outputs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
