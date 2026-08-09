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


AUDIT_ROOTS = (
    Path("analysis"),
    Path("tests/audit"),
    Path("docs"),
    Path("results"),
    Path("dft/audit"),
)
SELF_PATH = Path("docs/audit_output_sha256.csv")


def _category(relative: Path) -> str:
    if relative in {Path(".gitignore"), Path(".gitattributes")}:
        return "protection_config"
    if relative.parts[:2] == ("tests", "audit"):
        return "audit_test"
    return {
        "analysis": "audit_code",
        "docs": "audit_documentation",
        "results": "audit_result",
        "dft": "dft_audit_result",
    }[relative.parts[0]]


def collect_audit_files(root: Path) -> pd.DataFrame:
    root = Path(root).resolve()
    candidates: set[Path] = set()
    for config_name in (".gitignore", ".gitattributes"):
        config_path = root / config_name
        if config_path.is_file():
            candidates.add(config_path)
    for relative_root in AUDIT_ROOTS:
        directory = root / relative_root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.resolve().relative_to(root)
            if relative == SELF_PATH:
                continue
            if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
                continue
            candidates.add(path.resolve())
    rows = []
    for path in sorted(candidates, key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        stat = path.stat()
        rows.append(
            {
                "path": relative.as_posix(),
                "category": _category(relative),
                "size_bytes": int(stat.st_size),
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows, columns=["path", "category", "size_bytes", "mtime_utc", "sha256"])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    archive = args.archive.resolve()
    manifest = collect_audit_files(archive)
    content = manifest.to_csv(index=False, lineterminator="\n").encode("utf-8")
    output = archive / SELF_PATH
    state = write_bytes_protected(output, content, args.check_existing)
    print(
        json.dumps(
            {
                "path": str(output.relative_to(archive)),
                "state": state,
                "rows": len(manifest),
                "total_bytes": int(manifest["size_bytes"].sum()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
