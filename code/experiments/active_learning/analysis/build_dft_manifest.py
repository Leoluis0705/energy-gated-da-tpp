from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file, write_bytes_protected


REQUIRED_MANIFEST_COLUMNS = (
    "candidate_id",
    "formula",
    "pilot_or_new",
    "selection_rule",
    "freeze_timestamp",
    "timestamp_source",
    "result_known_at_freeze",
    "Gate_round",
    "Greedy_round",
    "DFT_status",
    "relaxation_output_available",
    "static_output_available",
    "failure_stage",
    "failure_reason",
    "main_text_selected",
    "selection_reason",
    "raw_job_path",
    "final_cif_path",
    "sha256",
)


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _candidate_directory(root: Path, parent: str, candidate_id: str) -> Path:
    candidates = [path for path in (root / parent).iterdir() if path.is_dir() and candidate_id in path.name]
    if len(candidates) != 1:
        raise ValueError(f"expected one directory for {candidate_id} under {parent}; found {candidates}")
    return candidates[0]


def _validate_prelaunch_freeze(root: Path) -> tuple[str, str]:
    manifest_path = root / "new12_dft_prelaunch/NEW12_DFT_CANDIDATE_MANIFEST.csv"
    status_path = root / "new12_dft_prelaunch/NEW12_DFT_STATUS.csv"
    protocol_path = root / "new12_dft_prelaunch/NEW12_DFT_PROTOCOL_AUDIT.md"
    status = pd.read_csv(status_path)
    if len(status) != 12 or set(status["status"]) != {"pending"}:
        raise ValueError("new12 prelaunch status is not an all-pending 12-candidate freeze")
    if set(status["review_gate"]) != {"awaiting_user_review"}:
        raise ValueError("new12 prelaunch review gate does not show launch was paused")
    protocol = protocol_path.read_text(encoding="utf-8")
    if "VASP launch authorization: **FALSE**" not in protocol:
        raise ValueError("new12 prelaunch protocol does not explicitly deny launch authorization")
    timestamp = _mtime_utc(manifest_path)
    source = (
        "filesystem_last_write_time_utc_of_new12_dft_prelaunch/"
        "NEW12_DFT_CANDIDATE_MANIFEST.csv; not an independently signed clock"
    )
    return timestamp, source


def _pilot_rows(root: Path) -> list[dict]:
    selection_path = root / "outputs/dft_candidate_selection/selected_dft_candidates.csv"
    selected = pd.read_csv(selection_path)
    summary = pd.read_csv(root / "materials_maintext_candidates/candidate_summary.csv")
    joined = selected.merge(summary, on="candidate_id", validate="one_to_one")
    if len(joined) != 8:
        raise ValueError(f"expected 8 pilot candidates, found {len(joined)}")

    rows: list[dict] = []
    for record in joined.sort_values("selection_number").to_dict("records"):
        candidate_number = int(record["selection_number"])
        candidate_dir = root / "materials_maintext_candidates" / f"candidate_{candidate_number:03d}"
        final_cif = candidate_dir / "final.cif"
        static_outcar = candidate_dir / "OUTCAR"
        static_oszicar = candidate_dir / "OSZICAR"
        relax_log = candidate_dir / "relax.log"
        if not all(path.is_file() for path in (final_cif, static_outcar, static_oszicar, relax_log)):
            raise FileNotFoundError(f"pilot evidence is incomplete in {candidate_dir}")
        outcar_header = static_outcar.read_text(encoding="utf-8", errors="ignore")[:80_000]
        if "NSW    =      0" not in outcar_header and "NSW = 0" not in outcar_header:
            raise ValueError(f"retained pilot OUTCAR is not demonstrably static: {static_outcar}")

        electronic_converged = bool(record["electronic_converged"])
        status = "static_finished" if electronic_converged else "failed_static_electronic_nonconvergence"
        failure_reason = "" if electronic_converged else "static electronic convergence marker absent"
        rows.append(
            {
                "candidate_id": record["candidate_id"],
                "formula": record["reduced_formula"],
                "pilot_or_new": "pilot",
                "selection_rule": (
                    "post-screening multi-criterion score: early acquisition, ALIGNN interval evidence, "
                    "structural sanity, and diversity"
                ),
                "freeze_timestamp": pd.NA,
                "timestamp_source": "MISSING_no_contemporaneous_result_blind_freeze_record",
                "result_known_at_freeze": "unknown",
                "Gate_round": pd.NA,
                "Greedy_round": pd.NA,
                "DFT_status": status,
                "relaxation_output_available": False,
                "static_output_available": True,
                "failure_stage": "" if electronic_converged else "static",
                "failure_reason": failure_reason,
                "main_text_selected": False,
                "selection_reason": (
                    f"retained pilot selection #{candidate_number}; recorded score={float(record['score']):.12g}"
                ),
                "raw_job_path": _relative(candidate_dir, root),
                "final_cif_path": _relative(final_cif, root),
                "sha256": sha256_file(final_cif),
                "sha256_subject": _relative(final_cif, root),
                "relaxation_evidence_available": True,
                "relaxation_evidence_type": "stdout_relax.log_only; original OUTCAR/OSZICAR unavailable",
                "pilot_selection_query_rank": int(record["query_rank"]),
                "pilot_selection_iteration": int(record["iteration"]),
                "pilot_selection_round_source": (
                    "outputs/dft_candidate_selection/selected_dft_candidates.csv; not relabelled as Gate_round"
                ),
                "selection_file_mtime_utc": _mtime_utc(selection_path),
                "selection_source_path": _relative(selection_path, root),
                "selection_source_sha256": sha256_file(selection_path),
                "historical_remote_job_path": record["calculation_path"],
            }
        )
    return rows


def _new_rows(root: Path) -> list[dict]:
    selection_path = root / "new12_dft_final/NEW12_DFT_CANDIDATE_MANIFEST.csv"
    selected = pd.read_csv(selection_path)
    prelaunch_path = root / "new12_dft_prelaunch/NEW12_DFT_CANDIDATE_MANIFEST.csv"
    prelaunch = pd.read_csv(prelaunch_path)
    status = pd.read_csv(root / "new12_dft_final/NEW12_DFT_STATUS.csv")
    results = pd.read_csv(root / "new12_dft_final/NEW12_DFT_RESULTS.csv")
    main = pd.read_csv(root / "new12_dft_final/NEW12_DFT_MAIN_TEXT_CANDIDATES.csv")
    if len(selected) != 12 or set(selected["candidate_id"]) != set(prelaunch["candidate_id"]):
        raise ValueError("new12 final and prelaunch candidate sets differ")
    if set(main["candidate_id"]) - set(selected["candidate_id"]):
        raise ValueError("main-text shortlist is not a subset of new12")
    freeze_timestamp, timestamp_source = _validate_prelaunch_freeze(root)
    main_reasons = main.set_index("candidate_id")["reason"].to_dict()

    merged = selected.merge(status, on=["candidate_id", "formula"], validate="one_to_one")
    merged = merged.merge(
        results[["candidate_id", "final_recommendation"]], on="candidate_id", validate="one_to_one"
    )
    rows: list[dict] = []
    for record in merged.to_dict("records"):
        candidate_id = record["candidate_id"]
        candidate_dir = _candidate_directory(root, "new12_dft_final/candidate_outputs", candidate_id)
        relax_dir = candidate_dir / "stages/01_pbe_relax"
        static_dir = candidate_dir / "stages/02_pbe_static"
        final_cif = candidate_dir / "final.cif"
        relaxation_available = all((relax_dir / name).is_file() for name in ("OUTCAR", "OSZICAR"))
        static_available = all((static_dir / name).is_file() for name in ("OUTCAR", "OSZICAR"))
        is_failed = str(record["status"]) == "failed"
        main_selected = candidate_id in main_reasons
        rows.append(
            {
                "candidate_id": candidate_id,
                "formula": record["formula"],
                "pilot_or_new": "new",
                "selection_rule": str(record["selection_category"]),
                "freeze_timestamp": freeze_timestamp,
                "timestamp_source": timestamp_source,
                "result_known_at_freeze": "false",
                "Gate_round": int(record["gate_first_query_round"]),
                "Greedy_round": int(record["greedy_first_query_round"]),
                "DFT_status": str(record["status"]),
                "relaxation_output_available": relaxation_available,
                "static_output_available": static_available,
                "failure_stage": str(record["current_stage"]) if is_failed else "",
                "failure_reason": str(record["notes"]) if is_failed else "",
                "main_text_selected": main_selected,
                "selection_reason": main_reasons.get(candidate_id, str(record["reason_for_inclusion"])),
                "raw_job_path": _relative(candidate_dir, root),
                "final_cif_path": _relative(final_cif, root),
                "sha256": sha256_file(final_cif),
                "sha256_subject": _relative(final_cif, root),
                "relaxation_evidence_available": relaxation_available,
                "relaxation_evidence_type": "stage-specific original OUTCAR and OSZICAR",
                "pilot_selection_query_rank": pd.NA,
                "pilot_selection_iteration": pd.NA,
                "pilot_selection_round_source": "not_applicable",
                "selection_file_mtime_utc": _mtime_utc(prelaunch_path),
                "selection_source_path": _relative(prelaunch_path, root),
                "selection_source_sha256": sha256_file(prelaunch_path),
                "historical_remote_job_path": str(record["remote_path"]),
            }
        )
    return rows


def build_dft_manifest(root: Path) -> pd.DataFrame:
    archive_root = Path(root).resolve()
    manifest = pd.DataFrame(_pilot_rows(archive_root) + _new_rows(archive_root))
    if len(manifest) != 20 or not manifest["candidate_id"].is_unique:
        raise ValueError("unified DFT manifest must contain exactly 20 unique candidates")
    if int(manifest["main_text_selected"].sum()) != 3:
        raise ValueError("unified DFT manifest must flag exactly three main-text candidates")
    for column in ("Gate_round", "Greedy_round", "pilot_selection_query_rank", "pilot_selection_iteration"):
        manifest[column] = pd.array(manifest[column], dtype="Int64")
    return manifest.sort_values(["pilot_or_new", "candidate_id"]).reset_index(drop=True)


def build_pilot_artifact_search(root: Path) -> pd.DataFrame:
    archive_root = Path(root).resolve()
    mirror = archive_root / "tmp/dft_candidate_inventory/remote/vasp_inputs"
    bundles = [
        archive_root / "outputs/dft_candidate_selection/dft_candidate_selection_bundle.tar.gz",
        archive_root / "outputs/dft_candidate_selection/final_dft_values_bundle.tar.gz",
        archive_root / "outputs/dft_candidate_selection/gga_u_values_bundle.tar.gz",
    ]
    for path in bundles:
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = [
        {
            "scope": "archive_candidate_evidence",
            "location": "materials_maintext_candidates/candidate_001..008",
            "search_method": "file inventory plus OUTCAR NSW check",
            "result": "retained OUTCAR/OSZICAR are static (NSW=0); eight relax.log files retained",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "archive_server_mirror",
            "location": _relative(mirror, archive_root),
            "search_method": "recursive OUTCAR/OSZICAR inventory and comparison with retained copy",
            "result": "server mirror contains the same post-static candidate outputs, not pre-static relaxation artifacts",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "archive_bundles",
            "location": ";".join(_relative(path, archive_root) for path in bundles),
            "search_method": "tar member-name inventory without extraction",
            "result": "input bundle has no outputs; result bundles retain summaries/logs but no pilot relaxation OUTCAR/OSZICAR",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "historical_repository",
            "location": "D:/CGCNN",
            "search_method": "rg --files with OUTCAR, OSZICAR, relaxation-log, scheduler-log globs and candidate-ID text search",
            "result": "no pilot VASP OUTCAR/OSZICAR or candidate scheduler records found",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "historical_remote_server",
            "location": "/root/dft_limo/outputs/dft_candidate_selection/vasp_inputs",
            "search_method": "local mirrored evidence only; no authenticated remote filesystem was mounted",
            "result": "live historical server directory not accessible in this workspace",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "recycle_bin_current_user",
            "location": "D:/$Recycle.Bin/current-user-SID",
            "search_method": "recursive file inventory and binary metadata string search for candidate IDs and dft_limo",
            "result": "no relevant payload or original-path metadata hit",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "recycle_bin_other_sids",
            "location": "D:/$Recycle.Bin/other-SIDs",
            "search_method": "read-only inventory attempt",
            "result": "access denied for three other SID directories; not searched",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "baidu_sync",
            "location": "D:/BaiduSyncdisk",
            "search_method": "rg --files with OUTCAR, OSZICAR, relaxation-log, and archive globs",
            "result": "no relevant file found",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "onedrive",
            "location": "C:/Users/leoluis0705/OneDrive",
            "search_method": "rg --files with OUTCAR, OSZICAR, relaxation-log, and archive globs",
            "result": "no relevant file found",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
        {
            "scope": "wps_sync",
            "location": "C:/Users/leoluis0705/WPS Cloud Files; C:/Users/leoluis0705/WPSDrive",
            "search_method": "rg --files with OUTCAR, OSZICAR, relaxation-log, and archive globs",
            "result": "no relevant file found",
            "original_relaxation_outcar_or_oszicar_found": False,
        },
    ]
    return pd.DataFrame(rows)


def _source_line(path: Path, root: Path) -> str:
    return f"- `{_relative(path, root)}`: SHA-256 `{sha256_file(path)}`"


def build_timeline_report(manifest: pd.DataFrame, artifact_search: pd.DataFrame) -> str:
    root = Path(__file__).resolve().parents[1]
    new_freeze = manifest.loc[manifest["pilot_or_new"] == "new", "freeze_timestamp"].iloc[0]
    pilot_missing = int((manifest["pilot_or_new"] == "pilot").sum())
    lines = [
        "# DFT Selection Timeline",
        "",
        "## Scope and set identity",
        "",
        "The unified manifest contains 20 unique candidates: eight pilot candidates and twelve new candidates. The three main-text candidates are flags within the new-candidate set; they are not a third candidate set.",
        "",
        "## Evidence-backed timeline",
        "",
        "1. The eight-pilot selection table records query rank, iteration, an ALIGNN interval score, structural checks, and diversity terms. No contemporaneous result-blind freeze record or independently verifiable freeze timestamp was found. Their `freeze_timestamp` is therefore blank and `result_known_at_freeze=unknown`.",
        "2. The pilot `query_rank/iteration` values do not agree with the retained v14 illustrative Gate/Greedy histories for the same IDs. They remain in dedicated `pilot_selection_*` columns and are not relabelled as `Gate_round` or `Greedy_round`.",
        f"3. The twelve-new prelaunch manifest has filesystem UTC mtime `{new_freeze}`. The adjacent prelaunch status contains twelve `pending/awaiting_user_review` rows, and the protocol audit explicitly records launch authorization as false. This supports `result_known_at_freeze=false`, while the timestamp itself is explicitly only filesystem metadata.",
        "4. A later launch-authorization note records authorization on 2026-07-13. The first retained per-candidate completion update follows the prelaunch filesystem timestamp. No more precise signed authorization time was found.",
        "5. The final new12 results identify three main-text candidates: job_120 (Cr), job_214 (Cr), and job_044 (Mn). Their selection reasons are retained verbatim from the main-text shortlist CSV.",
        "",
        "## Magnetic-sampling boundary",
        "",
        "For candidates with strict magnetic reruns, the evidence covers two tested magnetic initializations. A selected state is described only as the lower-energy configuration among the two tested initializations.",
        "",
        "## Pilot relaxation-artifact search",
        "",
        f"All {pilot_missing} pilot rows are marked `original_relaxation_artifact_unavailable`: the retained top-level OUTCAR/OSZICAR files are static outputs (`NSW=0`), while relaxation evidence is limited to `relax.log` plus the final relaxed structure.",
        "",
        "| Scope | Location | Result |",
        "|---|---|---|",
    ]
    for row in artifact_search.itertuples(index=False):
        lines.append(f"| {row.scope} | `{row.location}` | {row.result} |")
    lines.extend(
        [
            "",
            "A reconstructed rerun may be useful for protocol verification or to regenerate stage-separated outputs, but it would be a new calculation and must never be represented as the original historical relaxation. It is not required to interpret the retained static energies, but the absent original relaxation force/stress trajectory remains a declared provenance limitation.",
            "",
            "## Main-text dependency check",
            "",
            "The current v33 comparison PDF uses the three new12 main-text candidates rather than the eight pilot candidates for its main DFT shortlist. The pilot calculations nevertheless support historical DFT tables and methodological claims elsewhere in the archive, so their missing relaxation-stage artifacts remain relevant to full reproducibility.",
            "",
            "## Primary source hashes",
            "",
            _source_line(root / "outputs/dft_candidate_selection/selected_dft_candidates.csv", root),
            _source_line(root / "materials_maintext_candidates/candidate_summary.csv", root),
            _source_line(root / "new12_dft_prelaunch/NEW12_DFT_CANDIDATE_MANIFEST.csv", root),
            _source_line(root / "new12_dft_prelaunch/NEW12_DFT_STATUS.csv", root),
            _source_line(root / "new12_dft_prelaunch/NEW12_DFT_PROTOCOL_AUDIT.md", root),
            _source_line(root / "new12_dft_screening_snapshot/extracted/LAUNCH_AUTHORIZATION.md", root),
            _source_line(root / "new12_dft_final/NEW12_DFT_RESULTS.csv", root),
            _source_line(root / "new12_dft_final/NEW12_DFT_MAIN_TEXT_CANDIDATES.csv", root),
            "",
        ]
    )
    return "\n".join(lines)


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    root = args.archive_root.resolve()
    manifest = build_dft_manifest(root)
    artifact_search = build_pilot_artifact_search(root)
    report = build_timeline_report(manifest, artifact_search)
    outputs = {
        root / "dft/audit/dft_candidate_manifest.csv": _csv_bytes(manifest),
        root / "dft/audit/pilot_relaxation_artifact_search.csv": _csv_bytes(artifact_search),
        root / "docs/DFT_SELECTION_TIMELINE.md": report.encode("utf-8"),
    }
    statuses = {
        str(path.relative_to(root)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
