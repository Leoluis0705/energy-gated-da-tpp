import json
from datetime import datetime, timezone
from pathlib import Path

from analysis.update_active_gpu_manifest import update_active_manifest


def test_pointer_update_is_atomic_and_history_is_append_only(tmp_path: Path) -> None:
    pointer = tmp_path / "ACTIVE_GPU_MANIFEST.txt"
    history = tmp_path / "ACTIVE_GPU_MANIFEST.history.jsonl"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("job_id,status\na,DONE\n", encoding="utf-8")
    second.write_text("job_id,status\nb,PENDING\n", encoding="utf-8")

    one = update_active_manifest(
        pointer=pointer,
        history=history,
        manifest=first,
        changed_at=datetime(2026, 7, 17, 1, 2, 3, tzinfo=timezone.utc),
    )
    two = update_active_manifest(
        pointer=pointer,
        history=history,
        manifest=second,
        changed_at=datetime(2026, 7, 17, 1, 3, 4, tzinfo=timezone.utc),
    )

    assert pointer.read_text(encoding="utf-8") == str(second.resolve()) + "\n"
    records = [json.loads(line) for line in history.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert one["previous_manifest"] is None
    assert two["previous_manifest"] == str(first.resolve())
    assert records[1]["active_manifest"] == str(second.resolve())
    assert not pointer.with_suffix(".tmp").exists()

