import hashlib
import json
from pathlib import Path

from analysis.prepare_gpu_threshold_calibration import (
    M0_VALUES,
    G0_VALUES,
    build_threshold_screen_bundle,
    protocol_payloads,
)


def test_threshold_payloads_cover_preregistered_grid_with_frozen_k() -> None:
    payloads = protocol_payloads(mc_passes=30)

    assert {(item["M0"], item["G0"]) for item in payloads} == {
        (m0, g0) for m0 in M0_VALUES for g0 in G0_VALUES
    }
    assert len(payloads) == 9
    assert all(item["phase"] == "threshold_calibration" for item in payloads)
    assert all(item["allowed_seeds"] == [0, 1, 2, 3, 4] for item in payloads)
    assert all(item["allowed_methods"] == ["energy_gated_da_tpp"] for item in payloads)
    assert all(item["mc_passes"] == 30 for item in payloads)
    assert all((item["alpha"], item["beta"], item["gamma"]) == (0.1, 0.2, 0.1) for item in payloads)


def test_screen_bundle_contains_only_nine_seed_zero_jobs(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    manifest = tmp_path / "jobs.csv"
    frame = build_threshold_screen_bundle(
        project_root="/remote/project",
        output_root="/remote/results/parameter_calibration/threshold_screen",
        local_configs_root=configs,
        remote_configs_root="/remote/project/configs/parameter_calibration/threshold",
        manifest_path=manifest,
        git_commit="abc123",
        mc_passes=30,
    )

    assert len(frame) == 9
    assert set(frame["seed"]) == {0}
    assert set(frame["method"]) == {"energy_gated_da_tpp"}
    assert set(frame["K"]) == {30}
    assert frame["job_id"].is_unique
    assert frame["output_path"].is_unique
    assert frame["log_path"].is_unique
    assert set(frame["status"]) == {"PENDING"}
    for row in frame.to_dict(orient="records"):
        command = json.loads(row["command_json"])
        assert command[command.index("--seed") + 1] == "0"
        config = configs / Path(command[command.index("--protocol-config") + 1]).name
        assert hashlib.sha256(config.read_bytes()).hexdigest() == row["config_hash"]
        payload = json.loads(config.read_text(encoding="utf-8"))
        assert payload["allowed_seeds"] == [0, 1, 2, 3, 4]
