from __future__ import annotations

import json
from pathlib import Path

from analysis.watch_dft_verification_postprocess import (
    postprocessor_entrypoint,
    static_postprocess_directory,
    supervisor_action,
)


def test_followup_watcher_runs_only_after_static_success(tmp_path: Path) -> None:
    assert supervisor_action(tmp_path) == ("WAIT", "")
    (tmp_path / "final_status.json").write_text(
        json.dumps({"status": "PAUSED_BY_STRUCTURAL_REVIEW"}), encoding="utf-8"
    )
    assert supervisor_action(tmp_path)[0] == "BLOCKED"
    (tmp_path / "final_status.json").write_text(
        json.dumps({"status": "STATIC_TASKS_DONE_PENDING_POSTPROCESSING"}), encoding="utf-8"
    )
    assert supervisor_action(tmp_path) == ("RUN", "")


def test_followup_watcher_propagates_supervisor_failure(tmp_path: Path) -> None:
    (tmp_path / "failure.json").write_text(
        json.dumps({"status": "FAILED", "error": "controller stopped"}), encoding="utf-8"
    )
    action, reason = supervisor_action(tmp_path)
    assert action == "BLOCKED"
    assert "controller stopped" in reason


def test_recovery_watcher_uses_an_exclusive_static_postprocess_attempt(
    tmp_path: Path,
) -> None:
    output = static_postprocess_directory(tmp_path, "postprocess_attempt_2")
    assert output == (tmp_path / "candidate_static" / "postprocess_attempt_2").resolve()


def test_watcher_launches_postprocessor_as_a_package_module() -> None:
    assert postprocessor_entrypoint("/python") == [
        "/python",
        "-m",
        "analysis.postprocess_dft_verification_statics",
    ]
