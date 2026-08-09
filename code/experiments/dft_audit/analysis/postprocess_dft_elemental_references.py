"""Select and report frozen-protocol elemental-reference calculations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from analysis.prepare_dft_elemental_reference_jobs import EXPECTED_REFERENCES


DIAGNOSTIC_REFERENCE_IDS = {"GGA_U_Cr_metal", "GGA_U_Mn_metal"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_result(artifact_root: Path, reference_id: str) -> tuple[dict[str, object], Path]:
    output = Path(artifact_root) / "results" / reference_id / "attempt_1"
    result_path = output / "task_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if (
        result.get("status") != "DONE"
        or int(result.get("exit_code", -1)) != 0
        or result.get("electronic_converged") is not True
        or result.get("timing_footer_present") is not True
    ):
        raise ValueError(f"reference task evidence is incomplete for {reference_id}")
    return result, output


def _encut(path: Path) -> float:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"(?im)^\s*ENCUT\s*=\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"ENCUT is missing from {path}")
    return float(match.group(1))


def _manifest(path: Path, expected_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if set(frame["reference_id"].astype(str)) != expected_ids:
        raise ValueError(f"unexpected reference IDs in {path}")
    if not frame["status"].astype(str).eq("DONE").all():
        raise ValueError(f"manifest contains unfinished jobs: {path}")
    if not frame["exit_code"].astype(str).eq("0").all():
        raise ValueError(f"manifest contains nonzero exits: {path}")
    return frame


def build_elemental_reference_tables(
    *,
    base_manifest_path: Path,
    base_artifact_root: Path,
    diagnostic_manifest_path: Path,
    diagnostic_artifact_root: Path,
    legacy_reference_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the selected nine-reference table and a non-destructive comparison."""

    base = _manifest(Path(base_manifest_path), set(EXPECTED_REFERENCES)).set_index("reference_id")
    diagnostic = _manifest(
        Path(diagnostic_manifest_path), DIAGNOSTIC_REFERENCE_IDS
    ).set_index("reference_id")
    legacy = pd.read_csv(legacy_reference_path, keep_default_na=False).set_index(
        ["reference_name", "functional"]
    )
    selected_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []

    for reference_id in base.index:
        base_row = base.loc[reference_id]
        base_result, _ = _task_result(Path(base_artifact_root), reference_id)
        use_diagnostic = reference_id in DIAGNOSTIC_REFERENCE_IDS
        selected_row = diagnostic.loc[reference_id] if use_diagnostic else base_row
        selected_root = Path(diagnostic_artifact_root) if use_diagnostic else Path(base_artifact_root)
        selected_result, output = _task_result(selected_root, reference_id)
        input_dir = selected_root / "inputs" / reference_id
        provenance = json.loads((input_dir / "input_provenance.json").read_text(encoding="utf-8"))
        outcar = output / "OUTCAR"
        total_energy = float(selected_result["final_toten_ev"])
        atoms = int(selected_row["atom_count"])
        legacy_row = legacy.loc[(str(base_row["reference_name"]), str(base_row["functional"]))]
        legacy_total = float(legacy_row["total_energy_eV"])
        diagnostic_total = (
            float(selected_result["final_toten_ev"]) if use_diagnostic else None
        )
        structure_filename = str(
            selected_row.get("structure_source_filename", "")
            or provenance.get("source_structure_filename", "POSCAR")
        )
        structure_path = str(
            selected_row.get("structure_source_path", "")
            or provenance.get(
                "source_structure_path",
                Path(str(selected_row.get("source_dir", ""))) / structure_filename,
            )
        )
        structure_sha = str(
            selected_row.get("structure_source_sha256", "")
            or provenance.get("source_structure_sha256", provenance.get("POSCAR_sha256", ""))
        )
        risk = ""
        if str(selected_row["element"]) == "Mn":
            risk = (
                "retained bcc Mn screening reference; not alpha-Mn and not a unique "
                "materials ground-state determination"
            )
        selected_records.append(
            {
                "reference_id": reference_id,
                "reference_name": str(selected_row["reference_name"]),
                "element": str(selected_row["element"]),
                "structure": str(selected_row["structure"]),
                "magnetic_setup": str(selected_row["magnetic_setup"]),
                "paw_label": str(selected_row["paw_label"]),
                "functional": str(selected_row["functional"]),
                "Ueff_eV": selected_row["Ueff_eV"],
                "ENCUT_eV": _encut(input_dir / "INCAR"),
                "kpoints_mesh": str(selected_row["mesh"]),
                "kpoint_spacing_Ainv": float(selected_row["kpoint_spacing_Ainv"]),
                "total_energy_eV": total_energy,
                "atoms_per_cell": atoms,
                "energy_per_atom_eV": total_energy / atoms,
                "electronic_converged": True,
                "timing_footer_present": True,
                "exit_code": int(selected_result["exit_code"]),
                "elapsed_seconds": float(selected_result["elapsed_seconds"]),
                "selection_source": "structure_diagnostic" if use_diagnostic else "base_verification",
                "selection_reason": (
                    "historical CONTCAR reproduces the structure recorded by the legacy OUTCAR"
                    if use_diagnostic
                    else "base frozen-protocol verification"
                ),
                "structure_source_filename": structure_filename,
                "structure_source_path": structure_path,
                "structure_source_sha256": structure_sha,
                "raw_output_path": str(Path(str(selected_row["output_path"])) / "OUTCAR"),
                "recovered_output_path": str(outcar.resolve()),
                "raw_output_sha256": _sha256(outcar),
                "result_tree_sha256": str(selected_row["sha256"]),
                "config_hash": str(selected_row["config_hash"]),
                "git_commit": str(selected_row["git_commit"]),
                "frozen_protocol_sha256": str(selected_row["frozen_protocol_sha256"]),
                "reference_risk": risk,
            }
        )
        comparison_records.append(
            {
                "reference_id": reference_id,
                "reference_name": str(base_row["reference_name"]),
                "functional": str(base_row["functional"]),
                "atoms_per_cell": atoms,
                "legacy_total_energy_eV": legacy_total,
                "compact_poscar_total_energy_eV": float(base_result["final_toten_ev"]),
                "diagnostic_contcar_total_energy_eV": diagnostic_total,
                "selected_total_energy_eV": total_energy,
                "selected_minus_legacy_meV_per_atom": round(
                    (total_energy - legacy_total) * 1000.0 / atoms, 12
                ),
                "compact_minus_legacy_meV_per_atom": round(
                    (float(base_result["final_toten_ev"]) - legacy_total) * 1000.0 / atoms,
                    12,
                ),
                "selected_structure_source_filename": structure_filename,
                "selected_kpoints_mesh": str(selected_row["mesh"]),
                "selection_source": "structure_diagnostic" if use_diagnostic else "base_verification",
            }
        )

    references = pd.DataFrame(selected_records)
    comparison = pd.DataFrame(comparison_records)
    if len(references) != 9 or references["reference_id"].duplicated().any():
        raise ValueError("selected elemental-reference table is incomplete")
    return references, comparison


def _markdown_rows(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, rule]
    for record in frame[columns].to_dict(orient="records"):
        rows.append("| " + " | ".join(str(record[column]) for column in columns) + " |")
    return rows


def write_elemental_reference_outputs(
    *,
    references: pd.DataFrame,
    comparison: pd.DataFrame,
    reference_csv_path: Path,
    comparison_csv_path: Path,
    report_path: Path,
) -> None:
    for path in (reference_csv_path, comparison_csv_path, report_path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    references.to_csv(reference_csv_path, index=False, lineterminator="\n")
    comparison.to_csv(comparison_csv_path, index=False, lineterminator="\n")
    diagnostics = comparison[comparison["reference_id"].isin(DIAGNOSTIC_REFERENCE_IDS)].copy()
    material_differences = comparison[
        comparison["selected_minus_legacy_meV_per_atom"].abs() >= 2.0
    ].copy()
    report_lines = [
        "# Elemental reference report",
        "",
        "## Validation outcome",
        "",
        f"The selected table contains {len(references)} converged frozen-protocol reference calculations.",
        "All selected jobs exited zero and contain the VASP timing footer. The two compact-POSCAR",
        "Cr/Mn results remain in the comparison CSV but are excluded from the selected reference table.",
        "",
        "## Structure-source diagnostic",
        "",
        *_markdown_rows(
            diagnostics,
            [
                "reference_id",
                "legacy_total_energy_eV",
                "compact_poscar_total_energy_eV",
                "diagnostic_contcar_total_energy_eV",
                "selected_minus_legacy_meV_per_atom",
            ],
        ),
        "",
        "## Differences at or above 2 meV/atom",
        "",
        *(
            _markdown_rows(
                material_differences,
                ["reference_id", "selected_minus_legacy_meV_per_atom"],
            )
            if not material_differences.empty
            else ["None."]
        ),
        "",
        "## Mn limitation",
        "",
        "The retained Mn reference is a bcc Mn screening reference. It is not an alpha-Mn sensitivity calculation,",
        "does not establish a unique materials ground state, and must be labelled as a reference-protocol limitation",
        "when absolute Mn-containing formation energies are interpreted.",
        "",
        "## Machine-readable outputs",
        "",
        f"- Selected references: `{Path(reference_csv_path).as_posix()}`",
        f"- Energy comparison: `{Path(comparison_csv_path).as_posix()}`",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--base-artifact-root", type=Path, required=True)
    parser.add_argument("--diagnostic-manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-artifact-root", type=Path, required=True)
    parser.add_argument("--legacy-reference-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--comparison-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    references, comparison = build_elemental_reference_tables(
        base_manifest_path=args.base_manifest,
        base_artifact_root=args.base_artifact_root,
        diagnostic_manifest_path=args.diagnostic_manifest,
        diagnostic_artifact_root=args.diagnostic_artifact_root,
        legacy_reference_path=args.legacy_reference_csv,
    )
    write_elemental_reference_outputs(
        references=references,
        comparison=comparison,
        reference_csv_path=args.reference_csv,
        comparison_csv_path=args.comparison_csv,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "references": len(references),
                "reference_csv": str(args.reference_csv),
                "comparison_csv": str(args.comparison_csv),
                "report": str(args.report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
