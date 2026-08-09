"""Atomically point the watchdog at a validated GPU stage manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def update_active_manifest(
    *,
    pointer: Path,
    history: Path,
    manifest: Path,
    changed_at: datetime | None = None,
) -> dict[str, object]:
    active = Path(manifest).resolve()
    if not active.is_file():
        raise FileNotFoundError(active)
    when = changed_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        raise ValueError("changed_at must be timezone-aware")
    pointer = Path(pointer)
    history = Path(history)
    pointer.parent.mkdir(parents=True, exist_ok=True)
    history.parent.mkdir(parents=True, exist_ok=True)
    previous = pointer.read_text(encoding="utf-8").strip() if pointer.exists() else None
    temporary = pointer.with_suffix(".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(str(active) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, pointer)
    record: dict[str, object] = {
        "changed_at": when.astimezone(timezone.utc).isoformat(),
        "previous_manifest": previous,
        "active_manifest": str(active),
        "manifest_sha256": hashlib.sha256(active.read_bytes()).hexdigest(),
    }
    with history.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            update_active_manifest(
                pointer=args.pointer,
                history=args.history,
                manifest=args.manifest,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
