import hashlib
import json
from pathlib import Path

from analysis.prepare_gpu_weight_calibration import (
    WEIGHT_VARIANTS,
    build_weight_screen_bundle,
    weight_protocol_payloads,
)
from experiments.reproducibility.formal_protocol import load_formal_protocol


def test_weight_payloads_cover_preregistered_local_variants() -> None:
    payloads = weight_protocol_payloads(mc_passes=30, m0=1.0, g0=0.5)

    assert len(payloads) == 7
    assert {item["variant_id"] for item in payloads} == set(WEIGHT_VARIANTS)
    assert all(item["allowed_seeds"] == [0, 1, 2, 3, 4] for item in payloads)
    assert all(item["mc_passes"] == 30 for item in payloads)
    assert all((item["M0"], item["G0"]) == (1.0, 0.5) for item in payloads)
    observed = {
        item["variant_id"]: (item["alpha"], item["beta"], item["gamma"])
        for item in payloads
    }
    assert observed == WEIGHT_VARIANTS


def test_weight_screen_contains_only_seven_seed_zero_jobs(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    manifest = tmp_path / "jobs.csv"
    frame = build_weight_screen_bundle(
        project_root="/remote/project",
        output_root="/remote/results/weight_screen",
        local_configs_root=configs,
        remote_configs_root="/remote/project/configs/weight",
        manifest_path=manifest,
        git_commit="abc123",
        mc_passes=30,
        m0=1.25,
        g0=0.4,
    )

    assert len(frame) == 7
    assert set(frame["seed"]) == {0}
    assert frame["job_id"].is_unique
    assert frame["output_path"].is_unique
    assert set(frame["status"]) == {"PENDING"}
    for row in frame.to_dict(orient="records"):
        command = json.loads(row["command_json"])
        config = configs / Path(command[command.index("--protocol-config") + 1]).name
        assert hashlib.sha256(config.read_bytes()).hexdigest() == row["config_hash"]
        payload = json.loads(config.read_text(encoding="utf-8"))
        assert payload["M0"] == 1.25
        assert payload["G0"] == 0.4
        protocol = load_formal_protocol(config)
        assert protocol.phase == "weight_calibration"
        assert command[command.index("--seed") + 1] == "0"
