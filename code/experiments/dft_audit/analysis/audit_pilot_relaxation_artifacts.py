"""Inventory historical pilot DFT relaxation evidence without reading POTCAR."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PILOT_IDS = (
    "job_029_Cr_fe_-1.337_n4_generated_crystals_cif__gen_2",
    "job_058_Cr_fe_-1.423_n4_generated_crystals_cif__gen_0",
    "job_073_Cr_fe_-1.375_n4_generated_crystals_cif__gen_0",
    "job_092_Cr_fe_-1.075_n4_generated_crystals_cif__gen_1",
    "job_148_Mn_fe_-0.904_n4_generated_crystals_cif__gen_1",
    "job_182_Cr_fe_-1.464_n4_generated_crystals_cif__gen_2",
    "job_201_Cr_fe_-1.216_n4_generated_crystals_cif__gen_2",
    "job_249_Cr_fe_-1.500_n4_generated_crystals_cif__gen_1",
)

_EVIDENCE_NAMES = {"OUTCAR", "OSZICAR", "relax.log", "static.log"}
_NSW_PATTERN = re.compile(r"\bNSW\s*=\s*(-?\d+)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _outcar_nsw(path: Path) -> int | None:
    text = path.read_text(encoding="utf-8", errors="ignore")
    matches = _NSW_PATTERN.findall(text)
    return int(matches[-1]) if matches else None


def inventory_pilot_artifacts(
    server_root: Path,
    candidate_ids: Iterable[str] = PILOT_IDS,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Return candidate summaries and retained evidence-file records."""

    root = Path(server_root).resolve()
    summaries: list[dict[str, object]] = []
    files: list[dict[str, object]] = []
    all_files = [path for path in root.rglob("*") if path.is_file()]

    for candidate_id in candidate_ids:
        candidate_files = [path for path in all_files if candidate_id in str(path)]
        evidence_files = [path for path in candidate_files if path.name in _EVIDENCE_NAMES]
        outcars = [path for path in evidence_files if path.name == "OUTCAR"]
        oszicars = [path for path in evidence_files if path.name == "OSZICAR"]
        relax_logs = [path for path in evidence_files if path.name == "relax.log"]
        outcar_nsw = {path: _outcar_nsw(path) for path in outcars}
        relaxation_outcars = [path for path, nsw in outcar_nsw.items() if nsw is not None and nsw > 0]
        relaxation_oszicars = [
            path
            for path in oszicars
            if outcar_nsw.get(path.with_name("OUTCAR"), 0) > 0
        ]
        relaxation_artifacts = relaxation_outcars + relaxation_oszicars

        for path in sorted(evidence_files):
            stat = path.stat()
            nsw = outcar_nsw.get(path)
            files.append(
                {
                    "candidate_id": candidate_id,
                    "path": str(path),
                    "name": path.name,
                    "size_bytes": stat.st_size,
                    "mtime_utc": datetime.fromtimestamp(
                        stat.st_mtime, timezone.utc
                    ).isoformat(),
                    "sha256": _sha256(path),
                    "NSW": nsw if nsw is not None else "",
                    "interpreted_role": (
                        "static"
                        if nsw == 0
                        else "relaxation"
                        if nsw and nsw > 0
                        else "log_or_unknown"
                    ),
                }
            )

        summaries.append(
            {
                "candidate_id": candidate_id,
                "candidate_directories_found": len({str(path.parent) for path in candidate_files}),
                "current_outcar_count": len(outcars),
                "current_oszicar_count": len(oszicars),
                "relax_log_count": len(relax_logs),
                "relaxation_outcar_count": len(relaxation_outcars),
                "relaxation_oszicar_count": len(relaxation_oszicars),
                "original_relaxation_outcar_or_oszicar_found": bool(relaxation_artifacts),
                "classification": (
                    "relaxation_artifact_found"
                    if relaxation_artifacts
                    else "original_relaxation_artifact_unavailable"
                ),
            }
        )
    return summaries, files


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty evidence table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries, files = inventory_pilot_artifacts(args.server_root)
    _write_csv(output_dir / "candidate_summary.csv", summaries)
    _write_csv(output_dir / "evidence_files.csv", files)
    metadata = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "server_root": str(args.server_root.resolve()),
        "python_version": platform.python_version(),
        "candidate_count": len(summaries),
        "all_original_relaxation_artifacts_unavailable": not any(
            row["original_relaxation_outcar_or_oszicar_found"] for row in summaries
        ),
        "potcar_read_or_copied": False,
        "outputs": ["candidate_summary.csv", "evidence_files.csv"],
    }
    (output_dir / "search_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
