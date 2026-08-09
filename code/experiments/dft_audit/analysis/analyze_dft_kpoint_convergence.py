"""Analyze the audited three-system VASP k-point convergence batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import pandas as pd


ENERGY_TOLERANCE_MEV_PER_ATOM = 2.0
MAGNETIC_TOLERANCE_MUB_PER_ATOM = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_total_magnetization(path: Path) -> float | None:
    """Return the final OSZICAR total magnetic moment, if VASP wrote one."""

    matches = re.findall(r"\bmag\s*=\s*([-+0-9.Ee]+)", Path(path).read_text(encoding="utf-8", errors="replace"))
    return float(matches[-1]) if matches else None


def _parse_ispin(path: Path) -> int:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        clean = line.split("!", 1)[0].split("#", 1)[0]
        match = re.match(r"\s*ISPIN\s*=\s*(\d+)", clean, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1


def _write_csv_exclusive(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")


def _collect_details(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    if len(manifest) < 9 or set(manifest["status"]) != {"DONE"}:
        raise ValueError("k-point analysis requires at least nine DONE jobs")
    if set(manifest["system"]).issubset({""}) or manifest["system"].nunique() != 3:
        raise ValueError("k-point analysis requires exactly three systems")
    spacings = sorted(manifest["kpoint_spacing_Ainv"].astype(float).unique())
    if len(spacings) < 3:
        raise ValueError("k-point analysis requires at least three spacings")
    expected_jobs = manifest["system"].nunique() * len(spacings)
    if len(manifest) != expected_jobs:
        raise ValueError("k-point analysis requires a complete system-by-spacing grid")
    for _, block in manifest.groupby("system"):
        if sorted(block["kpoint_spacing_Ainv"].astype(float).tolist()) != spacings:
            raise ValueError("k-point analysis requires the same spacings for every system")

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        if int(record["exit_code"]) != 0:
            raise ValueError("k-point analysis requires exactly nine DONE jobs with exit code zero")
        output = Path(record["output_path"])
        if (output / "POTCAR").exists():
            raise ValueError(f"completed output retained POTCAR: {output}")
        required = [output / "task_result.json", output / "INCAR", output / "OSZICAR", output / "OUTCAR"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing completed VASP evidence: " + "; ".join(missing))
        result = json.loads((output / "task_result.json").read_text(encoding="utf-8"))
        if (
            result.get("status") != "DONE"
            or int(result.get("exit_code", -1)) != 0
            or not bool(result.get("electronic_converged"))
            or not bool(result.get("timing_footer_present"))
        ):
            raise ValueError(f"completed VASP evidence is not electronically converged: {output}")
        atom_count = int(record["atom_count"])
        energy = float(result["final_toten_ev"])
        ispin = _parse_ispin(output / "INCAR")
        moment = parse_total_magnetization(output / "OSZICAR")
        if ispin == 2 and moment is None:
            raise ValueError(f"ISPIN=2 output has no retained total magnetic moment: {output}")
        rows.append(
            {
                "job_id": record["job_id"],
                "system": record["system"],
                "kpoint_spacing_Ainv": float(record["kpoint_spacing_Ainv"]),
                "mesh": record["mesh"],
                "atom_count": atom_count,
                "final_toten_eV": energy,
                "energy_eV_per_atom": energy / atom_count,
                "ISPIN": ispin,
                "final_total_magnetic_moment_muB": moment,
                "elapsed_seconds": float(result["elapsed_seconds"]),
                "electronic_converged": True,
                "timing_footer_present": True,
                "output_path": str(output),
                "OUTCAR_sha256": _sha256(output / "OUTCAR"),
                "job_tree_sha256": record["sha256"],
                "config_hash": record["config_hash"],
                "POTCAR_sha256": record["potcar_sha256"],
                "POTCAR_retained": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["system", "kpoint_spacing_Ainv"], ascending=[True, False])


def _adjacent_differences(details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system, block in details.groupby("system", sort=True):
        ordered = block.sort_values("kpoint_spacing_Ainv", ascending=False).reset_index(drop=True)
        for index in range(len(ordered) - 1):
            coarse = ordered.iloc[index]
            fine = ordered.iloc[index + 1]
            energy_delta = abs(float(coarse["energy_eV_per_atom"]) - float(fine["energy_eV_per_atom"])) * 1000
            if int(coarse["ISPIN"]) == 1 and int(fine["ISPIN"]) == 1:
                moment_delta = math.nan
                magnetic_pass = True
                magnetic_basis = "not_applicable_ISPIN1"
            else:
                coarse_moment = float(coarse["final_total_magnetic_moment_muB"])
                fine_moment = float(fine["final_total_magnetic_moment_muB"])
                moment_delta = abs(coarse_moment - fine_moment) / int(coarse["atom_count"])
                magnetic_pass = moment_delta <= MAGNETIC_TOLERANCE_MUB_PER_ATOM
                magnetic_basis = "total_moment_change_per_atom"
            energy_pass = energy_delta < ENERGY_TOLERANCE_MEV_PER_ATOM
            rows.append(
                {
                    "system": system,
                    "coarse_spacing_Ainv": float(coarse["kpoint_spacing_Ainv"]),
                    "fine_spacing_Ainv": float(fine["kpoint_spacing_Ainv"]),
                    "coarse_mesh": coarse["mesh"],
                    "fine_mesh": fine["mesh"],
                    "absolute_energy_difference_meV_per_atom": energy_delta,
                    "energy_pass_2meV_per_atom": energy_pass,
                    "absolute_total_moment_difference_muB_per_atom": moment_delta,
                    "magnetic_basis": magnetic_basis,
                    "magnetic_pass_0p05muB_per_atom": magnetic_pass,
                    "pair_pass": energy_pass and magnetic_pass,
                }
            )
    return pd.DataFrame(rows)


def _render_report(
    *, manifest_path: Path, details: pd.DataFrame, adjacent: pd.DataFrame, decision: str, selected: float | None
) -> str:
    detail_columns = [
        "system",
        "kpoint_spacing_Ainv",
        "mesh",
        "final_toten_eV",
        "energy_eV_per_atom",
        "final_total_magnetic_moment_muB",
        "elapsed_seconds",
    ]
    adjacent_columns = [
        "system",
        "coarse_spacing_Ainv",
        "fine_spacing_Ainv",
        "absolute_energy_difference_meV_per_atom",
        "absolute_total_moment_difference_muB_per_atom",
        "pair_pass",
    ]
    lines = [
        "# K-point convergence report",
        "",
        f"- Source manifest: `{manifest_path.resolve()}`",
        f"- Source manifest SHA-256: `{_sha256(manifest_path)}`",
        f"- Jobs: {len(details)} DONE / 0 FAILED",
        f"- Energy criterion: adjacent absolute difference < {ENERGY_TOLERANCE_MEV_PER_ATOM:g} meV/atom",
        (
            "- Magnetic criterion: for ISPIN=2 representatives, adjacent absolute total-moment change "
            f"<= {MAGNETIC_TOLERANCE_MUB_PER_ATOM:g} muB/atom; ISPIN=1 is not applicable"
        ),
        "- POTCAR retained in completed or recovered outputs: no",
        "",
        "## Per-job results",
        "",
        details[detail_columns].to_markdown(index=False, floatfmt=".10g"),
        "",
        "## Adjacent-density checks",
        "",
        adjacent[adjacent_columns].to_markdown(index=False, floatfmt=".10g"),
        "",
        "## Decision",
        "",
        f"Decision: `{decision}`.",
    ]
    if selected is None:
        lines.extend(
            [
                "",
                "No common tested spacing meets the declared adjacent-density criteria for all three systems. "
                "The DFT protocol must not be frozen from this batch, and elemental-reference or candidate "
                "verification calculations must not start under an inferred k-point rule.",
                "",
                "The minimum defensible next action is one additional, common denser spacing for all three "
                "representatives, followed by the same adjacent-pair test. This report does not select that "
                "spacing or silently relax the 2 meV/atom threshold.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"The coarsest common passing spacing is `{selected:.2f} A^-1`. The frozen protocol records "
                "this choice and the exact source-manifest hash.",
            ]
        )
    return "\n".join(lines) + "\n"


def analyze_kpoint_convergence(
    *,
    manifest_path: Path,
    details_path: Path,
    adjacent_path: Path,
    report_path: Path,
    frozen_protocol_path: Path,
) -> dict[str, object]:
    manifest = Path(manifest_path)
    details = _collect_details(manifest)
    adjacent = _adjacent_differences(details)
    required_systems = details["system"].nunique()
    candidates: list[float] = []
    for spacing, block in adjacent.groupby("coarse_spacing_Ainv"):
        if len(block) == required_systems and bool(block["pair_pass"].all()):
            candidates.append(float(spacing))
    selected = max(candidates) if candidates else None
    decision = "FROZEN" if selected is not None else "BLOCKED_NO_COMMON_CONVERGED_SPACING"

    _write_csv_exclusive(details, Path(details_path))
    _write_csv_exclusive(adjacent, Path(adjacent_path))
    report = _render_report(
        manifest_path=manifest, details=details, adjacent=adjacent, decision=decision, selected=selected
    )
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report)

    protocol_file = Path(frozen_protocol_path)
    if selected is not None:
        payload = {
            "protocol_version": "egdatpp_dft_psfix_v1",
            "frozen": True,
            "kpoint_rule": "explicit_Gamma_mesh_ceil_reciprocal_length_over_spacing",
            "kpoint_spacing_Ainv": selected,
            "energy_tolerance_meV_per_atom_strictly_less_than": ENERGY_TOLERANCE_MEV_PER_ATOM,
            "magnetic_tolerance_muB_per_atom_maximum": MAGNETIC_TOLERANCE_MUB_PER_ATOM,
            "representative_systems": sorted(details["system"].unique().tolist()),
            "source_manifest": str(manifest.resolve()),
            "source_manifest_sha256": _sha256(manifest),
            "source_config_hashes": sorted(details["config_hash"].unique().tolist()),
            "source_job_count": int(len(details)),
        }
        protocol_file.parent.mkdir(parents=True, exist_ok=True)
        with protocol_file.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return {"decision": decision, "selected_spacing_Ainv": selected}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--adjacent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frozen-protocol", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_kpoint_convergence(
        manifest_path=args.manifest,
        details_path=args.details,
        adjacent_path=args.adjacent,
        report_path=args.report,
        frozen_protocol_path=args.frozen_protocol,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
