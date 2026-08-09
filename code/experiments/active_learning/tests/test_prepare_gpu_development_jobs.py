import hashlib
import json
from pathlib import Path

from analysis.prepare_gpu_development_jobs import (
    build_development_bundle,
    protocol_payloads,
)


def test_development_bundle_has_exact_30_jobs_and_three_k_protocols(tmp_path):
    configs = tmp_path / "configs"
    manifest = tmp_path / "jobs" / "gpu_jobs_manifest.csv"
    frame = build_development_bundle(
        project_root="/remote/formal/project",
        output_root="/remote/formal/results/development",
        local_configs_root=configs,
        remote_configs_root="/remote/formal/project/configs/gpu_development",
        manifest_path=manifest,
        git_commit="abc123",
    )

    assert len(frame) == 30
    assert set(frame["seed"]) == {0, 1, 2, 3, 4}
    assert set(frame["K"]) == {3, 10, 30}
    assert set(frame["method"]) == {"interval_hit_greedy", "energy_gated_da_tpp"}
    assert frame["status"].eq("PENDING").all()
    assert frame["output_path"].is_unique
    assert frame["log_path"].is_unique
    assert frame["job_id"].is_unique
    assert len(list(configs.glob("*.yaml"))) == 3
    for path in configs.glob("*.yaml"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["allowed_seeds"] == [0, 1, 2, 3, 4]
        assert payload["frozen"] is False
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in set(frame["config_hash"])


def test_development_protocols_change_only_k(tmp_path):
    payloads = protocol_payloads()
    assert [value["mc_passes"] for value in payloads] == [3, 10, 30]
    without_k = [{key: value for key, value in payload.items() if key != "mc_passes"} for payload in payloads]
    assert without_k[0] == without_k[1] == without_k[2]

