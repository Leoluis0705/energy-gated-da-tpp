from pathlib import Path

from analysis.freeze_final_gpu_protocol import build_frozen_protocol_bundle
from analysis.prepare_gpu_formal_jobs import build_formal_job_manifests


def test_formal_manifests_cover_only_approved_jobs(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    frozen = build_frozen_protocol_bundle(
        configs_root=configs,
        mc_passes=30,
        m0=1.0,
        g0=0.5,
        alpha=0.1,
        beta=0.2,
        gamma=0.1,
        git_commit="freezecommit",
        source_evidence={"calibration": "a" * 64},
    )
    protocol_root = configs / "frozen_protocols" / "egdatpp_psfix_v1"
    frames = build_formal_job_manifests(
        project_root=str(tmp_path / "project"),
        output_root=str(tmp_path / "formal_outputs"),
        manifest_directory=tmp_path / "jobs",
        git_commit="runcodecommit",
        limo_protocol=frozen["primary"],
        mn_original_protocol=protocol_root / "mnoxide_original.yaml",
        mn_block_protocol=protocol_root / "mnoxide_block.yaml",
        mn_iupac_protocol=protocol_root / "mnoxide_iupac.yaml",
        mc_protocols={k: protocol_root / f"limo_mc_k{k}.yaml" for k in (3, 10, 30)},
    )

    limo = frames["li_m_o_ablation"]
    mn = frames["mn_group_key"]
    mc = frames["mc_dropout_sensitivity"]
    assert (len(limo), len(mn), len(mc)) == (50, 50, 30)
    assert set(limo["seed"]) == set(range(15, 25))
    assert set(limo["method"]) == {
        "interval_hit_greedy",
        "always_da_tpp",
        "margin_only_gate",
        "group_only_gate",
        "energy_gated_da_tpp",
    }
    observed_mn = set(zip(mn["method"], mn["group_key"], strict=True))
    assert observed_mn == {
        ("interval_hit_greedy", "element_system_current"),
        ("always_da_tpp", "element_system_current"),
        ("energy_gated_da_tpp", "element_system_current"),
        ("energy_gated_da_tpp", "coelement_block_multiset"),
        ("energy_gated_da_tpp", "coelement_iupac_group_set"),
    }
    assert set(mc["seed"]) == set(range(25, 30))
    assert set(mc["K"]) == {3, 10, 30}
    assert set(mc["method"]) == {"interval_hit_greedy", "energy_gated_da_tpp"}

    combined = frames["combined"]
    assert len(combined) == 130
    assert combined["job_id"].is_unique
    assert combined["output_path"].is_unique
    assert set(combined["status"]) == {"PENDING"}
    assert not set(combined["seed"]).intersection(range(5, 15))
    assert set(combined["git_commit"]) == {"runcodecommit"}

