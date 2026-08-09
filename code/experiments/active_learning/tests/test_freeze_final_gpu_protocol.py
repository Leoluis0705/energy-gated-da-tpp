import hashlib
import json
from pathlib import Path

from analysis.freeze_final_gpu_protocol import build_frozen_protocol_bundle
from experiments.reproducibility.formal_protocol import load_formal_protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_bundle_separates_final_and_mc_cohorts(tmp_path: Path) -> None:
    outputs = build_frozen_protocol_bundle(
        configs_root=tmp_path,
        mc_passes=30,
        m0=1.0,
        g0=0.5,
        alpha=0.1,
        beta=0.2,
        gamma=0.1,
        git_commit="abc123",
        source_evidence={"weight_ranking_sha256": "a" * 64},
    )

    primary = tmp_path / "frozen_final_protocol.yaml"
    assert outputs["primary"] == primary
    limo = load_formal_protocol(primary)
    assert limo.phase == "formal_evaluation"
    assert limo.dataset == "limo"
    assert limo.allowed_seeds == tuple(range(15, 25))
    assert set(limo.allowed_methods) == {
        "interval_hit_greedy",
        "always_da_tpp",
        "margin_only_gate",
        "group_only_gate",
        "energy_gated_da_tpp",
    }
    assert limo.mc_passes == 30
    assert limo.frozen is True

    protocols = tmp_path / "frozen_protocols" / "egdatpp_psfix_v1"
    mn_block = load_formal_protocol(protocols / "mnoxide_block.yaml")
    assert mn_block.allowed_seeds == tuple(range(15, 25))
    assert mn_block.allowed_methods == ("energy_gated_da_tpp",)
    assert mn_block.group_key_mode == "coelement_block_multiset"
    assert mn_block.group_key_map_relative_path.endswith("mnoxide_coelement_block_multiset.csv")

    for k in (3, 10, 30):
        sensitivity = load_formal_protocol(protocols / f"limo_mc_k{k}.yaml")
        assert sensitivity.phase == "mc_dropout_sensitivity"
        assert sensitivity.allowed_seeds == tuple(range(25, 30))
        assert sensitivity.mc_passes == k
        assert set(sensitivity.allowed_methods) == {
            "interval_hit_greedy",
            "energy_gated_da_tpp",
        }

    manifest_path = outputs["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_evidence"] == {"weight_ranking_sha256": "a" * 64}
    for record in manifest["protocols"]:
        path = tmp_path / record["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_runtime_protocol_and_group_map_bytes_are_pinned_to_lf() -> None:
    attributes = (REPOSITORY_ROOT / ".gitattributes").read_text(encoding="utf-8")

    assert "/configs/frozen_final_protocol.yaml text eol=lf" in attributes
    assert "/configs/frozen_protocols/egdatpp_psfix_v1/* text eol=lf" in attributes
    assert "/configs/group_keys/egdatpp_psfix_v1/*.csv text eol=lf" in attributes
