from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.postprocess_dft_elemental_references import (
    build_elemental_reference_tables,
    write_elemental_reference_outputs,
)


REFERENCES = [
    ("PBE_Li_metal", "Li_metal", "Li", "PBE", ""),
    ("PBE_Cr_metal", "Cr_metal", "Cr", "PBE", ""),
    ("PBE_Mn_metal", "Mn_metal", "Mn", "PBE", ""),
    ("PBE_Mg_metal", "Mg_metal", "Mg", "PBE", ""),
    ("PBE_O2_molecule", "O2_molecule", "O", "PBE", ""),
    ("GGA_U_Li_metal", "Li_metal", "Li", "GGA+U", 0.0),
    ("GGA_U_Cr_metal", "Cr_metal", "Cr", "GGA+U", 3.7),
    ("GGA_U_Mn_metal", "Mn_metal", "Mn", "GGA+U", 3.9),
    ("GGA_U_O2_molecule", "O2_molecule", "O", "GGA+U", 0.0),
]


def _write_batch(root: Path, rows: list[dict[str, object]]) -> Path:
    for row in rows:
        reference_id = str(row["reference_id"])
        output = root / "results" / reference_id / "attempt_1"
        input_dir = root / "inputs" / reference_id
        output.mkdir(parents=True)
        input_dir.mkdir(parents=True)
        (output / "task_result.json").write_text(
            json.dumps(
                {
                    "status": "DONE",
                    "exit_code": 0,
                    "final_toten_ev": row["energy"],
                    "electronic_converged": True,
                    "timing_footer_present": True,
                    "elapsed_seconds": 12.0,
                }
            ),
            encoding="utf-8",
        )
        (output / "OUTCAR").write_text(
            "vasp.6.5.1 test build\nGeneral timing and accounting informations for this job:\n",
            encoding="utf-8",
        )
        (input_dir / "INCAR").write_text("ENCUT = 520\n", encoding="utf-8")
        (input_dir / "KPOINTS").write_text("test\n0\nGamma\n3 3 3\n", encoding="utf-8")
        (input_dir / "input_provenance.json").write_text(
            json.dumps(
                {
                    "source_structure_filename": row.get("structure_source_filename", "POSCAR"),
                    "source_structure_path": f"/source/{reference_id}/{row.get('structure_source_filename', 'POSCAR')}",
                    "source_structure_sha256": str(row.get("structure_source_sha256", "p" * 64)),
                }
            ),
            encoding="utf-8",
        )
    manifest_rows = []
    for row in rows:
        manifest_rows.append(
            {
                **row,
                "status": "DONE",
                "exit_code": 0,
                "output_path": f"/remote/results/{row['reference_id']}/attempt_1",
                "sha256": "a" * 64,
                "config_hash": "b" * 64,
                "git_commit": "abc123",
                "mesh": "3x3x3",
                "atom_count": 2,
                "structure": "Im-3m (229)",
                "magnetic_setup": "ISPIN=2; MAGMOM=2*5.0",
                "paw_label": f"PAW_PBE {row['element']}",
                "kpoint_spacing_Ainv": 0.15,
                "frozen_protocol_sha256": "f" * 64,
            }
        )
    manifest = root / "jobs.csv"
    pd.DataFrame(manifest_rows).drop(columns=["energy"]).to_csv(manifest, index=False)
    return manifest


def test_diagnostic_contcar_rows_replace_compact_poscar_results(tmp_path: Path) -> None:
    base_rows = []
    for index, (reference_id, name, element, functional, ueff) in enumerate(REFERENCES):
        base_rows.append(
            {
                "reference_id": reference_id,
                "reference_name": name,
                "element": element,
                "functional": functional,
                "Ueff_eV": ueff,
                "energy": -10.0 - index,
                "structure_source_filename": "POSCAR",
                "structure_source_sha256": "p" * 64,
            }
        )
    diagnostic_rows = [
        {
            **next(row for row in base_rows if row["reference_id"] == reference_id),
            "energy": corrected,
            "structure_source_filename": "CONTCAR",
            "structure_source_sha256": "c" * 64,
        }
        for reference_id, corrected in [
            ("GGA_U_Cr_metal", -11.64),
            ("GGA_U_Mn_metal", -13.08),
        ]
    ]
    base_root = tmp_path / "base"
    diagnostic_root = tmp_path / "diagnostic"
    base_manifest = _write_batch(base_root, base_rows)
    diagnostic_manifest = _write_batch(diagnostic_root, diagnostic_rows)
    legacy = pd.DataFrame(
        [
            {
                "reference_name": row["reference_name"],
                "functional": row["functional"],
                "total_energy_eV": row["energy"] - 0.002,
                "energy_per_atom_eV": (row["energy"] - 0.002) / 2,
            }
            for row in base_rows
        ]
    )
    legacy_path = tmp_path / "legacy.csv"
    legacy.to_csv(legacy_path, index=False)

    references, comparison = build_elemental_reference_tables(
        base_manifest_path=base_manifest,
        base_artifact_root=base_root,
        diagnostic_manifest_path=diagnostic_manifest,
        diagnostic_artifact_root=diagnostic_root,
        legacy_reference_path=legacy_path,
    )

    selected = references.set_index("reference_id")
    compared = comparison.set_index("reference_id")
    assert len(references) == 9
    assert selected.loc["GGA_U_Cr_metal", "total_energy_eV"] == -11.64
    assert selected.loc["GGA_U_Cr_metal", "structure_source_filename"] == "CONTCAR"
    assert selected.loc["GGA_U_Cr_metal", "selection_source"] == "structure_diagnostic"
    assert compared.loc["GGA_U_Cr_metal", "compact_poscar_total_energy_eV"] == -16.0
    assert compared.loc["GGA_U_Cr_metal", "selected_total_energy_eV"] == -11.64
    assert compared.loc[
        "GGA_U_Mn_metal", "selected_minus_legacy_meV_per_atom"
    ] == pytest.approx(1961.0)


def test_writes_machine_readable_tables_and_mn_limitation(tmp_path: Path) -> None:
    references = pd.DataFrame(
        [
            {
                "reference_id": "GGA_U_Mn_metal",
                "reference_name": "Mn_metal",
                "functional": "GGA+U",
                "total_energy_eV": -13.08,
                "energy_per_atom_eV": -6.54,
                "electronic_converged": True,
                "selection_source": "structure_diagnostic",
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {
                "reference_id": "GGA_U_Mn_metal",
                "legacy_total_energy_eV": -13.075,
                "compact_poscar_total_energy_eV": -10.35,
                "diagnostic_contcar_total_energy_eV": -13.08,
                "selected_total_energy_eV": -13.08,
                "selected_minus_legacy_meV_per_atom": -2.5,
            }
        ]
    )

    write_elemental_reference_outputs(
        references=references,
        comparison=comparison,
        reference_csv_path=tmp_path / "elemental_references.csv",
        comparison_csv_path=tmp_path / "comparison.csv",
        report_path=tmp_path / "report.md",
    )

    assert (tmp_path / "elemental_references.csv").is_file()
    assert (tmp_path / "comparison.csv").is_file()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "bcc Mn" in report
    assert "not an alpha-Mn sensitivity calculation" in report
    assert "compact-POSCAR" in report
    assert "Differences at or above 2 meV/atom" in report
    assert "GGA_U_Mn_metal" in report
