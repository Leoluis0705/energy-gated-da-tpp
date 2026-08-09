import json
from pathlib import Path
import subprocess
import sys

from analysis.build_structural_gate_feasibility_manifest import build_manifest_rows


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_builder_runs_as_a_direct_server_script():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "analysis" / "build_structural_gate_feasibility_manifest.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_manifest_has_complete_collision_safe_50_job_grid():
    rows = build_manifest_rows(
        project_root="/project",
        execution_root="/execution",
        python="/venv/bin/python",
        gpu_ids=(0, 1),
    )

    assert len(rows) == 50
    assert len({row["job_id"] for row in rows}) == 50
    assert len({row["output_path"] for row in rows}) == 50
    assert {row["seed"] for row in rows} == {str(value) for value in range(111, 116)}
    assert {row["gpu_id"] for row in rows} == {"0", "1"}
    assert {row["status"] for row in rows} == {"PENDING"}
    assert {row["method"] for row in rows} == {
        "predicted_target_greedy",
        "energy_gated_da_tpp",
        "structural_group_gate",
        "structural_group_gate_q95",
        "gradient_norm_hybrid",
    }
    assert all("structural_group_feasibility_v1" in row["output_path"] for row in rows)
    assert all("formal_w0p2" not in row["output_path"] for row in rows)


def test_manifest_uses_method_specific_frozen_protocols():
    rows = build_manifest_rows(
        project_root="/project",
        execution_root="/execution",
        python="/venv/bin/python",
        gpu_ids=(0, 1),
    )
    protocols = {}
    for row in rows:
        command = json.loads(row["command_json"])
        protocol = command[command.index("--protocol-config") + 1]
        protocols.setdefault(row["method"], set()).add(protocol)

    assert protocols["energy_gated_da_tpp"] == {
        "/project/configs/structural_group_feasibility/legacy_protocol.json"
    }
    assert protocols["structural_group_gate"] == {
        "/project/configs/structural_group_feasibility/structural_protocol.json"
    }
    assert protocols["structural_group_gate_q95"] == {
        "/project/configs/structural_group_feasibility/structural_q95_protocol.json"
    }


def test_manifest_rejects_empty_or_duplicate_gpu_ids():
    for gpu_ids in ((), (0, 0)):
        try:
            build_manifest_rows(
                project_root="/project",
                execution_root="/execution",
                python="/venv/bin/python",
                gpu_ids=gpu_ids,
            )
        except ValueError as error:
            assert "GPU IDs" in str(error)
        else:
            raise AssertionError("invalid GPU IDs were accepted")


def test_manifest_config_hash_tracks_local_frozen_file_content(tmp_path):
    for name in {
        "legacy_protocol.json",
        "structural_protocol.json",
        "structural_q95_protocol.json",
        "mn_task.json",
        "mg_task.json",
    }:
        (tmp_path / name).write_text(name, encoding="utf-8")
    before = build_manifest_rows(
        project_root="/project",
        execution_root="/execution",
        python="/venv/bin/python",
        gpu_ids=(0, 1),
        config_source_root=tmp_path,
    )
    (tmp_path / "mn_task.json").write_text("changed", encoding="utf-8")
    after = build_manifest_rows(
        project_root="/project",
        execution_root="/execution",
        python="/venv/bin/python",
        gpu_ids=(0, 1),
        config_source_root=tmp_path,
    )

    assert before[0]["config_hash"] != after[0]["config_hash"]
    assert before[25]["config_hash"] == after[25]["config_hash"]
