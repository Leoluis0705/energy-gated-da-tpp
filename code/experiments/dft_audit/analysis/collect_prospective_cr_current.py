"""Collect and report the self-consistent prospective Cr batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
from pathlib import Path
from typing import Any

import pandas as pd
import paramiko
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from analysis.audit_current_potcars_remote import EXPECTED_HOST_KEY, host_key_sha256


def _json_cell(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_result_tables(
    state: dict[str, Any],
    manifest: pd.DataFrame,
    formation_checks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checks_by_id: dict[str, dict[str, Any]] = {}
    if not formation_checks.empty:
        checks_by_id = {
            str(row["candidate_id"]): row.to_dict()
            for _, row in formation_checks.iterrows()
        }
    results: list[dict[str, Any]] = []
    magnetic: list[dict[str, Any]] = []
    for source in manifest.itertuples(index=False):
        candidate_id = str(source.candidate_id)
        entity = state.get("entities", {}).get(candidate_id, {})
        completed = entity.get("status") == "PASS"
        relax = entity.get("relax", {}).get("parsed", {}) if completed else {}
        static = (
            entity.get("static_0p15", {}).get("parsed", {}) if completed else {}
        )
        check = checks_by_id.get(candidate_id, {})
        formation = (
            float(check["formation_energy_primary_eV_atom"])
            if check and bool(check.get("roundtrip_pass"))
            else math.nan
        )
        alignn = float(source.alignn_formation_energy_eV_atom)
        result_level = "DFT_EVALUATED" if completed else "NOT_DFT_EVALUATED"
        row = {
            "candidate_id": candidate_id,
            "formal_workflow_status": "PASS" if completed else "FAIL_OR_NOT_RUN",
            "result_level": result_level,
            "failure_reason": entity.get("failure_reason", ""),
            "cif_sha256": str(source.cif_sha256),
            "relax_attempt_count": entity.get("relax", {}).get(
                "attempt_count", math.nan
            ),
            "static_attempt_count": entity.get("static_0p15", {}).get(
                "attempt_count", math.nan
            ),
            "electronic_converged": static.get(
                "electronic_converged", math.nan
            ),
            "ionic_converged": relax.get("ionic_converged", math.nan),
            "Fmax_eV_A": relax.get("fmax_eV_A", math.nan),
            "static_energy_eV": static.get("energy_eV", math.nan),
            "static_energy_per_atom_eV": static.get(
                "energy_per_atom_eV", math.nan
            ),
            "formation_energy_eV_atom": formation,
            "formation_energy_roundtrip_difference_eV_atom": check.get(
                "absolute_difference_eV_atom", math.nan
            ),
            "formation_energy_roundtrip_pass": check.get(
                "roundtrip_pass", False
            ),
            "formation_energy_scale": "self-consistent PAW-PBE+U",
            "alignn_formation_energy_eV_atom": alignn,
            "dft_minus_alignn_eV_atom": (
                formation - alignn if math.isfinite(formation) else math.nan
            ),
            "same_scale_status": "UNRESOLVED",
            "self_consistent_fe_target_status": (
                "NOT_ASSESSED_SAME_SCALE_UNRESOLVED"
            ),
            "minimum_interatomic_distance_A": static.get(
                "minimum_interatomic_distance_A", math.nan
            ),
            "volume_change_percent": relax.get(
                "volume_change_percent", math.nan
            ),
            "stress_kbar": _json_cell(static.get("stress_kbar", [])),
            "entropy_term_eV_cell": static.get(
                "entropy_term_eV_cell", math.nan
            ),
            "total_magnetic_moment_muB": static.get(
                "total_magnetic_moment_muB", math.nan
            ),
            "local_magnetic_moments_muB": _json_cell(
                static.get("local_magnetic_moments_muB", [])
            ),
            "band_gap_eV": static.get("band_gap_eV", math.nan),
            "electronic_state": (
                "metal_or_small_gap"
                if isinstance(static.get("band_gap_eV"), (int, float))
                and float(static["band_gap_eV"]) <= 0.05
                else "insulator_candidate"
                if isinstance(static.get("band_gap_eV"), (int, float))
                else "not_evaluated"
            ),
            "structure_status": (
                "NO_COLLAPSE"
                if completed and not static.get("structure_collapse", True)
                else "FAILED_OR_NOT_EVALUATED"
            ),
            "final_space_group_symbol": static.get(
                "final_space_group_symbol", ""
            ),
            "final_space_group_number": static.get(
                "final_space_group_number", math.nan
            ),
            "gate_round": int(source.gate_round),
            "greedy_round": int(source.greedy_round),
            "gate_precedes_greedy": int(source.gate_round)
            < int(source.greedy_round),
            "energy_above_hull_eV_atom": math.nan,
            "hull_status": "NOT_COMPUTED_NO_SAME_PROTOCOL_COMPETITOR_SET",
            "database_duplicate_status": "NOT_YET_CHECKED",
        }
        results.append(row)
        if completed:
            magnetic.append(
                {
                    "candidate_id": candidate_id,
                    "magnetic_branch": "MP_STANDARD_INITIALIZATION_ONLY",
                    "initialization": "Li=0.6, Cr=5.0, O=0.6 muB",
                    "total_magnetic_moment_muB": static.get(
                        "total_magnetic_moment_muB", math.nan
                    ),
                    "local_magnetic_moments_muB": _json_cell(
                        static.get("local_magnetic_moments_muB", [])
                    ),
                    "additional_magnetic_state_required": "NOT_YET_ASSESSED",
                }
            )
    return pd.DataFrame(results), pd.DataFrame(magnetic)


def augment_state_with_local_structures(
    state: dict[str, Any], local_root: Path
) -> None:
    for entity in state.get("entities", {}).values():
        for stage_key in ("relax", "static_0p15", "static_0p10"):
            stage = entity.get(stage_key, {})
            final_attempt = str(stage.get("final_attempt", ""))
            if "/runs/" not in final_attempt or "parsed" not in stage:
                continue
            suffix = final_attempt.split("/runs/", 1)[1]
            cif_path = local_root / "runs" / suffix / "final.cif"
            if not cif_path.is_file():
                continue
            structure = Structure.from_file(cif_path)
            analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
            stage["parsed"]["final_space_group_symbol"] = (
                analyzer.get_space_group_symbol()
            )
            stage["parsed"]["final_space_group_number"] = (
                analyzer.get_space_group_number()
            )
            total_seconds = 0.0
            for runtime_path in (local_root / "runs" / suffix).parent.glob(
                "attempt_*/runtime.json"
            ):
                runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
                total_seconds += float(runtime.get("elapsed_seconds", 0.0))
            stage["total_elapsed_seconds"] = total_seconds


def _display(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value) if value not in (None, "") else "—"
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def write_reports(
    *,
    state: dict[str, Any],
    results: pd.DataFrame,
    magnetic: pd.DataFrame,
    report_root: Path,
) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    first_id = (
        "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_3"
    )
    first = state.get("entities", {}).get(first_id, {})
    first_row = (
        results.loc[results["candidate_id"].eq(first_id)].iloc[0]
        if not results.empty
        else None
    )
    dense = first.get("static_0p10", {})
    job_text = [
        "# job_092 Complete Chain Report",
        "",
        f"- Formal chain status: `{first.get('status', 'NOT_RUN')}`",
        "- Warm-start geometry: `no`; formal chain started from frozen CIF.",
        "- Old probe status: `RESOURCE_AND_RELAXATION_DIAGNOSTIC_ONLY`.",
        "- Old probe energies used: `no`.",
        f"- Relax attempts: `{first.get('relax', {}).get('attempt_count', 0)}`",
        f"- 0.15 Å^-1 static attempts: `{first.get('static_0p15', {}).get('attempt_count', 0)}`",
        f"- 0.10 Å^-1 static attempts: `{dense.get('attempt_count', 0)}`",
        f"- k-point energy difference: `{_display(first.get('kpoint_energy_difference_eV_atom'))}` eV/atom",
        f"- <=2 meV/atom check: `{first.get('kpoint_converged_2meV_atom', False)}`",
    ]
    if first_row is not None:
        job_text.extend(
            [
                f"- Fmax: `{_display(first_row['Fmax_eV_A'])}` eV/Å",
                f"- Static energy: `{_display(first_row['static_energy_eV'])}` eV/cell",
                f"- Self-consistent formation energy: `{_display(first_row['formation_energy_eV_atom'])}` eV/atom",
                f"- Formation-energy roundtrip: `{first_row['formation_energy_roundtrip_pass']}`",
                f"- Total moment: `{_display(first_row['total_magnetic_moment_muB'])}` μB/cell",
                f"- Band gap: `{_display(first_row['band_gap_eV'])}` eV",
                f"- Structure: `{first_row['structure_status']}`",
            ]
        )
    job_text.extend(
        [
            "",
            "A chain is reportable only if the status is PASS. Formation-energy "
            "agreement does not establish equivalence to the ALIGNN label scale, "
            "phase stability, novelty, or experimental discovery.",
        ]
    )
    (report_root / "JOB_092_COMPLETE_CHAIN_REPORT.md").write_text(
        "\n".join(job_text) + "\n", encoding="utf-8"
    )

    runtime_lines = [
        "# Prospective Cr v2 Runtime Report",
        "",
        f"- Controller status: `{state.get('status')}`",
        f"- Controller phase: `{state.get('phase')}`",
        f"- Started: `{state.get('started_at_utc', '')}`",
        f"- Completed: `{state.get('completed_at_utc', '')}`",
        "- Initial candidate concurrency: `1` (job_092 gate).",
        "- Remaining-candidate concurrency after gate: `2`.",
        "- Wall-time segment: `21000 s` (5 h 50 min).",
        "- Timeout policy: resume relaxation from newest valid CONTCAR; timeout "
        "alone is not a scientific failure.",
        "",
        "## Entity runtime",
        "",
        "| Entity | Status | Relax attempts | Relax seconds | Static attempts | Static seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for entity, payload in state.get("entities", {}).items():
        relax = payload.get("relax", {})
        static = payload.get("static_0p15", {})
        runtime_lines.append(
            f"| `{entity}` | {payload.get('status')} | "
            f"{relax.get('attempt_count', 0)} | "
            f"{_display(relax.get('total_elapsed_seconds', relax.get('runtime', {}).get('elapsed_seconds')), 1)} | "
            f"{static.get('attempt_count', 0)} | "
            f"{_display(static.get('total_elapsed_seconds', static.get('runtime', {}).get('elapsed_seconds')), 1)} |"
        )
    runtime_lines.extend(
        [
            "",
            "## Recorded controller repair",
            "",
            "One script-only repair (`repair_001`) corrected JSON serialization "
            "of NumPy stress arrays. The completed/partial CONTCAR was preserved "
            "and the next segment resumed from it. No scientific INCAR, KPOINTS, "
            "POTCAR, U, ENCUT, or magnetic initialization changed.",
        ]
    )
    (report_root / "PROSPECTIVE_CR_V2_RUNTIME_REPORT.md").write_text(
        "\n".join(runtime_lines) + "\n", encoding="utf-8"
    )

    final_lines = [
        "# Prospective Cr v2 Final Report",
        "",
        "- Energy definition: `self-consistent PAW-PBE+U formation energy`.",
        "- Exact Materials Project compatibility: `not claimed`.",
        "- SAME_SCALE_STATUS versus ALIGNN: `UNRESOLVED`.",
        "- Therefore, DFT-minus-ALIGNN differences are listed diagnostically and "
        "are not interpreted as strict target hits.",
        "",
        "| Candidate | Workflow | Fmax | Static E | Formation E | Moment | Gap | SG | Structure | Gate/Greedy | Level |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|---|",
    ]
    for row in results.itertuples(index=False):
        final_lines.append(
            f"| `{row.candidate_id}` | {row.formal_workflow_status} | "
            f"{_display(row.Fmax_eV_A)} | {_display(row.static_energy_eV)} | "
            f"{_display(row.formation_energy_eV_atom)} | "
            f"{_display(row.total_magnetic_moment_muB)} | "
            f"{_display(row.band_gap_eV)} | "
            f"{row.final_space_group_symbol} "
            f"({_display(row.final_space_group_number, 0)}) | "
            f"{row.structure_status} | "
            f"{row.gate_round}/{row.greedy_round} | {row.result_level} |"
        )
    final_lines.extend(
        [
            "",
            "No energy above hull is reported unless every Li–Cr–O competitor "
            "has been recalculated under this exact protocol. A formation-energy "
            "result alone is not a stability or novelty claim.",
        ]
    )
    (report_root / "PROSPECTIVE_CR_V2_FINAL_REPORT.md").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )

    usability_lines = [
        "# Prospective Cr v2 Paper Usability",
        "",
        "| Candidate | Formal DFT | FE reportable | Target | Magnetic follow-up | Hull | Discovery narrative | Gate before Greedy |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in results.itertuples(index=False):
        formal = row.formal_workflow_status == "PASS"
        fe_reportable = formal and bool(row.formation_energy_roundtrip_pass)
        usability_lines.append(
            f"| `{row.candidate_id}` | {'yes' if formal else 'no'} | "
            f"{'yes, on internal scale' if fe_reportable else 'no'} | "
            "not assessed (scale unresolved) | "
            f"{'assess after five-candidate ranking' if formal else 'not yet'} | "
            "not computed without same-protocol competitors | "
            f"{'relative screening only' if formal else 'none yet'} | "
            f"{'yes' if row.gate_precedes_greedy else 'no'} |"
        )
    usability_lines.extend(
        [
            "",
            "Results may support a paper only as a modern, internally consistent "
            "PAW-PBE+U prospective evaluation with the stated hashes and "
            "limitations. They cannot yet support claims of exact MP "
            "compatibility, strict entry into the original ALIGNN interval, "
            "thermodynamic stability, novelty, or experimental discovery.",
        ]
    )
    (report_root / "PROSPECTIVE_CR_V2_PAPER_USABILITY.md").write_text(
        "\n".join(usability_lines) + "\n", encoding="utf-8"
    )


def _connect(
    host: str, port: int, user: str, key_path: Path
) -> tuple[paramiko.Transport, socket.socket]:
    key = paramiko.Ed25519Key.from_private_key_file(str(key_path.expanduser()))
    sock = socket.create_connection((host, port), timeout=20)
    transport = paramiko.Transport(sock)
    transport.start_client(timeout=20)
    observed = host_key_sha256(transport.get_remote_server_key())
    if observed != EXPECTED_HOST_KEY:
        transport.close()
        sock.close()
        raise RuntimeError(f"host-key mismatch: {observed}")
    transport.auth_publickey(user, key)
    return transport, sock


def _download_if_exists(
    sftp: paramiko.SFTPClient, remote: str, local: Path
) -> bool:
    try:
        sftp.stat(remote)
    except OSError:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    return True


def collect_metadata(
    *,
    host: str,
    port: int,
    user: str,
    key_path: Path,
    remote_root: str,
    local_root: Path,
) -> dict[str, Any]:
    transport, sock = _connect(host, port, user, key_path)
    downloaded: list[str] = []
    try:
        sftp = paramiko.SFTPClient.from_transport(transport)
        try:
            fixed = [
                "controller_state.json",
                "controller.log",
                "deployment.json",
                "results/formation_energy_checks.csv",
            ]
            for relative in fixed:
                if _download_if_exists(
                    sftp,
                    remote_root.rstrip("/") + "/" + relative,
                    local_root / relative,
                ):
                    downloaded.append(relative)

            command = (
                f"find {remote_root}/runs -type f "
                "\\( -name stage_summary.json -o -name runtime.json "
                "-o -name input_hashes.json -o -name parsed.json "
                "-o -name final.cif -o -name OUTCAR -o -name OSZICAR "
                "-o -name vasprun.xml -o -name CONTCAR -o -name INCAR "
                "-o -name KPOINTS -o -name POSCAR -o -name vasp.stdout "
                "-o -name vasp.stderr -o -name parse_error.txt \\) -print"
            )
            channel = transport.open_session(timeout=20)
            try:
                channel.exec_command(command)
                output = channel.makefile("rb").read().decode(
                    "utf-8", "replace"
                )
                status = channel.recv_exit_status()
            finally:
                channel.close()
            if status:
                raise RuntimeError("remote result enumeration failed")
            for remote_file in output.splitlines():
                if not remote_file.startswith(remote_root.rstrip("/") + "/"):
                    raise RuntimeError(f"unexpected remote path: {remote_file}")
                relative = remote_file[len(remote_root.rstrip("/") + "/") :]
                if Path(relative).name in {"POTCAR", "WAVECAR", "CHGCAR"}:
                    raise RuntimeError("prohibited file appeared in allowlist")
                if _download_if_exists(
                    sftp, remote_file, local_root / relative
                ):
                    downloaded.append(relative)
        finally:
            sftp.close()
    finally:
        transport.close()
        sock.close()
    return {
        "downloaded_file_count": len(downloaded),
        "files": [
            {
                "relative_path": relative,
                "sha256": _sha256(local_root / relative),
                "size_bytes": (local_root / relative).stat().st_size,
            }
            for relative in downloaded
        ],
        "potcar_downloaded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--user", default="root")
    parser.add_argument("--key", required=True)
    parser.add_argument(
        "--remote-root",
        default="/root/autodl-tmp/prospective_cr_discovery_v2",
    )
    parser.add_argument(
        "--local-root", default="dft/prospective_cr_discovery_v2/remote_results"
    )
    parser.add_argument(
        "--manifest", default="manifests/prospective_cr_discovery_v2.csv"
    )
    args = parser.parse_args()
    local_root = Path(args.local_root).resolve()
    collection = collect_metadata(
        host=args.host,
        port=args.port,
        user=args.user,
        key_path=Path(args.key),
        remote_root=args.remote_root,
        local_root=local_root,
    )
    state = json.loads(
        (local_root / "controller_state.json").read_text(encoding="utf-8")
    )
    augment_state_with_local_structures(state, local_root)
    checks_path = local_root / "results/formation_energy_checks.csv"
    checks = (
        pd.read_csv(checks_path)
        if checks_path.is_file() and checks_path.stat().st_size
        else pd.DataFrame()
    )
    manifest = pd.read_csv(args.manifest)
    results, magnetic = build_result_tables(state, manifest, checks)
    Path("results").mkdir(exist_ok=True)
    results.to_csv(
        "results/prospective_cr_v2_results.csv",
        index=False,
        lineterminator="\n",
    )
    checks.to_csv(
        "results/prospective_cr_v2_formation_energy_checks.csv",
        index=False,
        lineterminator="\n",
    )
    magnetic.to_csv(
        "results/prospective_cr_v2_magnetic_results.csv",
        index=False,
        lineterminator="\n",
    )
    write_reports(
        state=state,
        results=results,
        magnetic=magnetic,
        report_root=Path("reports"),
    )
    (local_root / "collection_manifest.json").write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "controller_status": state.get("status"),
                "controller_phase": state.get("phase"),
                "downloaded_file_count": collection["downloaded_file_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
