from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FinalizationOutputs:
    results_csv: Path
    probe_results: Path
    probe_go_no_go: Path
    final_report: Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty minimal-DFT result table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stage_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "not run"
    parts = []
    for row in sorted(
        rows, key=lambda item: (item["magnetic_state"], item["stage"])
    ):
        parts.append(
            f"{row['magnetic_state']}/{row['stage']}: "
            f"exit={row.get('exit_code')}, "
            f"electronic={row.get('electronic_converged')}, "
            f"ionic={row.get('ionic_converged')}"
        )
    return "; ".join(parts)


def finalize_results(*, repo_root: Path, artifact_root: Path) -> FinalizationOutputs:
    repo_root = Path(repo_root).resolve()
    artifact_root = Path(artifact_root).resolve()
    protocol = _read_json(
        repo_root / "manifests" / "dft_protocol_frozen.json", {}
    )
    if protocol.get("overall_status") != "PASS":
        raise RuntimeError("frozen DFT protocol is not PASS")
    same_scale = str(protocol.get("same_scale_status", "UNKNOWN"))
    lower, upper = [
        float(value) for value in protocol["target_interval_eV_atom"]
    ]
    candidates = _read_csv(
        repo_root / "manifests" / "minimal_dft_5_candidates.csv"
    )
    if not candidates:
        raise FileNotFoundError("frozen candidate manifest is missing or empty")

    execution = artifact_root / "execution"
    stages = _read_csv(execution / "stage_results.csv")
    final_status = _read_json(execution / "final_status.json", {})
    probe_gate = _read_json(execution / "probe_gate.json", {})
    candidate_gates = _read_json(execution / "candidate_gates.json", {})
    gates = {**probe_gate, **candidate_gates}
    probe_ids = set(final_status.get("probe_candidates", []))

    result_rows: list[dict[str, Any]] = []
    stage_details: list[str] = []
    for candidate in sorted(candidates, key=lambda row: int(row["frozen_order"])):
        candidate_id = candidate["candidate_id"]
        rows = [row for row in stages if row["candidate_id"] == candidate_id]
        gate = gates.get(candidate_id, {})
        gate_pass = bool(gate.get("pass", False))
        gate_reasons = list(gate.get("reasons", []))
        static_rows = [
            row
            for row in rows
            if row.get("stage") == "static"
            and _bool(row.get("formation_energy_recomputed"))
            and _float(row.get("final_toten_eV")) is not None
        ]
        selected_static = (
            min(static_rows, key=lambda row: float(row["final_toten_eV"]))
            if static_rows
            else None
        )
        selected_state = (
            selected_static["magnetic_state"] if selected_static else ""
        )
        selected_relax = next(
            (
                row
                for row in rows
                if row.get("stage") == "relax"
                and row.get("magnetic_state") == selected_state
            ),
            {},
        )
        formation = (
            _float(selected_static.get("formation_energy_eV_atom"))
            if selected_static
            else None
        )
        alignn = _float(candidate.get("alignn_formation_energy_eV_atom"))
        membership = (
            lower <= formation <= upper if formation is not None else False
        )
        all_electronic = bool(rows) and all(
            _bool(row.get("electronic_converged")) for row in rows
        )
        all_relax_ionic = bool(rows) and all(
            _bool(row.get("ionic_converged"))
            or _bool(row.get("nsw_limit_reached"))
            for row in rows
            if row.get("stage") == "relax"
        )
        any_collapsed = any(
            _bool(row.get("structure_collapsed")) for row in rows
        )
        if not rows:
            evaluation = "NOT_RUN"
        elif gate_pass:
            evaluation = "DFT_EVALUATED_CANDIDATE"
        else:
            evaluation = "FAILED_OR_INCOMPLETE"
        strict_claim = bool(same_scale == "CONFIRMED" and gate_pass and membership)
        result_rows.append(
            {
                "frozen_order": candidate["frozen_order"],
                "candidate_id": candidate_id,
                "formula": candidate["formula"],
                "candidate_stratum": candidate["candidate_stratum"],
                "probe_candidate": candidate_id in probe_ids,
                "terminal_batch_status": final_status.get("status", "UNKNOWN"),
                "candidate_gate_pass": gate_pass,
                "candidate_gate_reasons": json.dumps(gate_reasons),
                "stage_count": len(rows),
                "dft_evaluation_status": evaluation,
                "selected_magnetic_state": selected_state,
                "selected_internal_formation_energy_eV_atom": (
                    formation if formation is not None else ""
                ),
                "alignn_formation_energy_eV_atom": (
                    alignn if alignn is not None else ""
                ),
                "dft_minus_alignn_eV_atom": (
                    formation - alignn
                    if formation is not None and alignn is not None
                    else ""
                ),
                "internal_interval_membership": membership,
                "same_scale_status": same_scale,
                "result_scope": protocol.get(
                    "result_classification_ceiling",
                    "INTERNAL_PROTOCOL_DIAGNOSTIC",
                ),
                "strict_original_interval_claim_allowed": strict_claim,
                "all_completed_stages_electronically_converged": all_electronic,
                "all_relaxations_converged_or_at_nsw_limit": all_relax_ionic,
                "structure_collapsed": any_collapsed,
                "selected_Fmax_eV_A": selected_relax.get("Fmax_eV_A", ""),
                "selected_volume_change_fraction": selected_relax.get(
                    "volume_change_fraction", ""
                ),
                "selected_minimum_pair_distance_A": selected_relax.get(
                    "minimum_pair_distance_A", ""
                ),
                "selected_final_total_moment_muB": (
                    selected_static.get("final_total_moment_muB", "")
                    if selected_static
                    else ""
                ),
                "selected_band_gap_eV": (
                    selected_static.get("band_gap_eV", "")
                    if selected_static
                    else ""
                ),
                "selected_smearing_review_required": (
                    selected_static.get("smearing_review_required", "")
                    if selected_static
                    else ""
                ),
                "stability_supported": False,
            }
        )
        stage_details.append(
            f"- `{candidate_id}`: {_stage_summary(rows)}"
        )

    results_csv = repo_root / "results" / "minimal_dft_5_results.csv"
    probe_results = repo_root / "reports" / "DFT_PROBE_RESULTS.md"
    probe_go_no_go = repo_root / "reports" / "DFT_PROBE_GO_NO_GO.md"
    final_report = repo_root / "reports" / "MINIMAL_DFT_5_FINAL_REPORT.md"
    _write_csv(results_csv, result_rows)
    probe_results.parent.mkdir(parents=True, exist_ok=True)

    probe_rows = [row for row in result_rows if row["probe_candidate"]]
    probe_pass = bool(probe_rows) and all(
        bool(row["candidate_gate_pass"]) for row in probe_rows
    )
    probe_results.write_text(
        "# DFT Probe Results\n\n"
        + f"- Terminal batch status: `{final_status.get('status', 'UNKNOWN')}`\n"
        + f"- SAME_SCALE_STATUS: `{same_scale}`\n"
        + f"- Probe gate: `{'PASS' if probe_pass else 'NO-GO'}`\n\n"
        + "## Stage evidence\n\n"
        + "\n".join(
            detail
            for detail in stage_details
            if any(row["candidate_id"] in detail for row in probe_rows)
        )
        + "\n",
        encoding="utf-8",
    )
    probe_go_no_go.write_text(
        "# DFT Probe GO/NO-GO\n\n"
        + f"Decision: `{'GO' if probe_pass else 'NO-GO'}`\n\n"
        + (
            "Both frozen probes passed the fixed execution gate; the remaining "
            "three candidates were eligible for automatic submission.\n"
            if probe_pass
            else "At least one frozen probe did not pass the fixed execution "
            "gate; the remaining batch was not eligible for automatic submission.\n"
        ),
        encoding="utf-8",
    )
    evaluated_count = sum(
        row["dft_evaluation_status"] == "DFT_EVALUATED_CANDIDATE"
        for row in result_rows
    )
    internal_members = sum(
        bool(row["internal_interval_membership"])
        and row["dft_evaluation_status"] == "DFT_EVALUATED_CANDIDATE"
        for row in result_rows
    )
    final_report.write_text(
        "# Minimal DFT-5 Final Report\n\n"
        + f"- Protocol status: `{protocol.get('overall_status', 'UNKNOWN')}`\n"
        + f"- SAME_SCALE_STATUS: {same_scale}\n"
        + f"- Remote terminal status: `{final_status.get('status', 'UNKNOWN')}`\n"
        + f"- DFT-evaluated candidates: `{evaluated_count}/5`\n"
        + f"- Internal-protocol interval members: `{internal_members}`\n"
        + "- Strict original-ALIGNN interval claims allowed: "
        + f"`{'yes' if same_scale == 'CONFIRMED' else 'no'}`\n"
        + "- Stability claims allowed: `no` (no compatible convex-hull analysis)\n\n"
        + "The energy-scale relationship remains unresolved. Numerical membership "
        + "under the internal PBE/GGA+U reference convention is diagnostic only "
        + "and cannot be promoted to a strict original-target claim.\n\n"
        + "## Candidate-stage evidence\n\n"
        + "\n".join(stage_details)
        + "\n",
        encoding="utf-8",
    )
    return FinalizationOutputs(
        results_csv=results_csv,
        probe_results=probe_results,
        probe_go_no_go=probe_go_no_go,
        final_report=final_report,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--artifact-root", required=True)
    args = parser.parse_args()
    outputs = finalize_results(
        repo_root=Path(args.repo_root), artifact_root=Path(args.artifact_root)
    )
    print(json.dumps({key: str(value) for key, value in outputs.__dict__.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
