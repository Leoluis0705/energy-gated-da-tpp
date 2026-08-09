import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.analyze_dft_kpoint_convergence import (
    analyze_kpoint_convergence,
    parse_total_magnetization,
)


SPACINGS = (0.35, 0.30, 0.25)


def _write_run(path: Path, energy: float, magnetic_moment: float | None, *, converged: bool = True) -> None:
    path.mkdir(parents=True)
    (path / "task_result.json").write_text(
        json.dumps(
            {
                "status": "DONE",
                "exit_code": 0,
                "final_toten_ev": energy,
                "electronic_converged": converged,
                "timing_footer_present": True,
                "elapsed_seconds": 12.5,
            }
        ),
        encoding="utf-8",
    )
    (path / "INCAR").write_text(f"ISPIN = {2 if magnetic_moment is not None else 1}\n", encoding="utf-8")
    oszicar = "  1 F= -1.0 E0= -1.0 d E = 0"
    if magnetic_moment is not None:
        oszicar += f" mag= {magnetic_moment:.6f}"
    (path / "OSZICAR").write_text(oszicar + "\n", encoding="utf-8")
    (path / "OUTCAR").write_text("General timing and accounting informations for this job:\n", encoding="utf-8")


def _manifest(
    tmp_path: Path,
    per_atom_energies: dict[str, tuple[float, ...]],
    *,
    spacings: tuple[float, ...] = SPACINGS,
) -> Path:
    rows = []
    for system, energies in per_atom_energies.items():
        for spacing, energy_per_atom in zip(spacings, energies, strict=True):
            token = f"{spacing:.2f}".replace(".", "p")
            output = tmp_path / "runs" / system / token
            moment = None if system == "Li_reference" else 5.0
            _write_run(output, energy_per_atom * 2, moment)
            rows.append(
                {
                    "job_id": f"{system}_{token}",
                    "status": "DONE",
                    "exit_code": "0",
                    "output_path": str(output),
                    "system": system,
                    "kpoint_spacing_Ainv": spacing,
                    "mesh": f"mesh_{token}",
                    "atom_count": 2,
                    "config_hash": "a" * 64,
                    "sha256": "b" * 64,
                    "potcar_sha256": "c" * 64,
                }
            )
    manifest = tmp_path / "dft_jobs_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


def test_parse_total_magnetization_uses_last_ionic_record(tmp_path):
    path = tmp_path / "OSZICAR"
    path.write_text(
        " 1 F= -1 E0= -1 mag= 4.9000\n 2 F= -2 E0= -2 mag= -5.1250\n",
        encoding="utf-8",
    )
    assert parse_total_magnetization(path) == pytest.approx(-5.125)


def test_analysis_selects_coarsest_common_passing_spacing_and_writes_frozen_protocol(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "LiCr2O4_C214": (-7.0000, -7.0008, -7.0010),
            "LiMn2O4_C044": (-6.0000, -6.0010, -6.0012),
            "Li_reference": (-2.0000, -2.0015, -2.0018),
        },
    )
    protocol = tmp_path / "dft_frozen_protocol.yaml"

    result = analyze_kpoint_convergence(
        manifest_path=manifest,
        details_path=tmp_path / "details.csv",
        adjacent_path=tmp_path / "adjacent.csv",
        report_path=tmp_path / "report.md",
        frozen_protocol_path=protocol,
    )

    assert result["decision"] == "FROZEN"
    assert result["selected_spacing_Ainv"] == pytest.approx(0.35)
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    assert payload["frozen"] is True
    assert payload["kpoint_spacing_Ainv"] == pytest.approx(0.35)
    adjacent = pd.read_csv(tmp_path / "adjacent.csv")
    assert adjacent["energy_pass_2meV_per_atom"].all()


def test_analysis_blocks_without_writing_protocol_when_no_common_adjacent_pair_passes(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "LiCr2O4_C214": (-7.0000, -7.0001, -7.0002),
            "LiMn2O4_C044": (-6.0000, -6.0001, -6.0002),
            "Li_reference": (-2.0000, -2.0100, -2.0210),
        },
    )
    protocol = tmp_path / "dft_frozen_protocol.yaml"

    result = analyze_kpoint_convergence(
        manifest_path=manifest,
        details_path=tmp_path / "details.csv",
        adjacent_path=tmp_path / "adjacent.csv",
        report_path=tmp_path / "report.md",
        frozen_protocol_path=protocol,
    )

    assert result["decision"] == "BLOCKED_NO_COMMON_CONVERGED_SPACING"
    assert result["selected_spacing_Ainv"] is None
    assert not protocol.exists()
    assert "must not be frozen" in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_analysis_rejects_incomplete_or_failed_manifest(tmp_path):
    manifest = _manifest(
        tmp_path,
        {
            "LiCr2O4_C214": (-7.0, -7.0, -7.0),
            "LiMn2O4_C044": (-6.0, -6.0, -6.0),
            "Li_reference": (-2.0, -2.0, -2.0),
        },
    )
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    rows[0]["status"] = "FAILED"
    pd.DataFrame(rows).to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="nine DONE jobs"):
        analyze_kpoint_convergence(
            manifest_path=manifest,
            details_path=tmp_path / "details.csv",
            adjacent_path=tmp_path / "adjacent.csv",
            report_path=tmp_path / "report.md",
            frozen_protocol_path=tmp_path / "protocol.yaml",
        )


def test_analysis_accepts_one_common_denser_spacing_and_freezes_from_all_twelve_jobs(tmp_path):
    spacings = (0.35, 0.30, 0.25, 0.20)
    manifest = _manifest(
        tmp_path,
        {
            "LiCr2O4_C214": (-7.0000, -7.0002, -7.0004, -7.0005),
            "LiMn2O4_C044": (-6.0000, -6.0002, -6.0004, -6.0005),
            "Li_reference": (-2.0000, -2.0100, -2.0210, -2.0215),
        },
        spacings=spacings,
    )
    protocol = tmp_path / "dft_frozen_protocol.yaml"

    result = analyze_kpoint_convergence(
        manifest_path=manifest,
        details_path=tmp_path / "details.csv",
        adjacent_path=tmp_path / "adjacent.csv",
        report_path=tmp_path / "report.md",
        frozen_protocol_path=protocol,
    )

    assert result["decision"] == "FROZEN"
    assert result["selected_spacing_Ainv"] == pytest.approx(0.25)
    details = pd.read_csv(tmp_path / "details.csv")
    assert len(details) == 12
    payload = json.loads(protocol.read_text(encoding="utf-8"))
    assert payload["source_job_count"] == 12
