import json
from dataclasses import replace

import pandas as pd
import pytest

from experiments.reproducibility.formal_protocol import (
    FormalProtocolError,
    load_formal_protocol,
    resolve_group_keys_from_map,
)
from experiments.reproducibility.two_dataset_paired_protocol import dataset_configs


def write_protocol(tmp_path, **overrides):
    payload = {
        "protocol_version": "egdatpp_psfix_v1",
        "phase": "mc_dropout_development",
        "dataset": "limo",
        "allowed_seeds": [0, 1, 2, 3, 4],
        "allowed_methods": ["interval_hit_greedy", "energy_gated_da_tpp"],
        "mc_passes": 10,
        "M0": 1.0,
        "G0": 0.50,
        "alpha": 0.10,
        "beta": 0.20,
        "gamma": 0.10,
        "group_key_mode": "element_system_current",
        "group_key_map_relative_path": None,
        "frozen": False,
    }
    payload.update(overrides)
    path = tmp_path / "protocol.yaml"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_protocol_overrides_only_declared_acquisition_parameters(tmp_path):
    protocol = load_formal_protocol(write_protocol(tmp_path))
    base = dataset_configs()["limo"]

    resolved = protocol.resolve_dataset_config(base, method="energy_gated_da_tpp", seed=4)

    assert resolved == replace(
        base,
        mc_passes=10,
        group_key_construction="element_system_current",
    )
    assert protocol.sha256


@pytest.mark.parametrize(
    ("phase", "seed", "frozen", "message"),
    [
        ("mc_dropout_development", 15, False, "not allowed"),
        ("formal_evaluation", 15, False, "must be frozen"),
        ("formal_evaluation", 4, True, "cohort"),
        ("mc_dropout_sensitivity", 15, True, "cohort"),
    ],
)
def test_protocol_enforces_cohort_and_freeze_gates(tmp_path, phase, seed, frozen, message):
    allowed_seeds = {
        "mc_dropout_development": [0, 1, 2, 3, 4],
        "formal_evaluation": list(range(15, 25)),
        "mc_dropout_sensitivity": list(range(25, 30)),
    }[phase]
    path = write_protocol(
        tmp_path,
        phase=phase,
        allowed_seeds=allowed_seeds,
        frozen=frozen,
    )
    protocol = load_formal_protocol(path)

    with pytest.raises(FormalProtocolError, match=message):
        protocol.resolve_dataset_config(dataset_configs()["limo"], method="energy_gated_da_tpp", seed=seed)


def test_noncurrent_group_key_requires_label_blind_map(tmp_path):
    path = write_protocol(
        tmp_path,
        dataset="mnoxide",
        group_key_mode="coelement_block_multiset",
        group_key_map_relative_path="configs/group_keys/mnoxide_coelement_block_multiset.csv",
    )
    protocol = load_formal_protocol(path)

    assert protocol.group_key_map_relative_path == "configs/group_keys/mnoxide_coelement_block_multiset.csv"

    bad = write_protocol(
        tmp_path,
        dataset="mnoxide",
        group_key_mode="coelement_block_multiset",
        group_key_map_relative_path=None,
    )
    with pytest.raises(FormalProtocolError, match="group-key map"):
        load_formal_protocol(bad)


def test_protocol_rejects_unknown_fields(tmp_path):
    path = write_protocol(tmp_path, target_label_column="target_label")
    with pytest.raises(FormalProtocolError, match="unknown fields"):
        load_formal_protocol(path)


def test_group_key_map_requires_exact_label_blind_columns_and_candidate_ids(tmp_path):
    path = tmp_path / "groups.csv"
    pd.DataFrame(
        {"candidate_id": ["a", "b"], "group_key": ["s1", "d1"]}
    ).to_csv(path, index=False)

    assert resolve_group_keys_from_map(["b", "a"], path) == ["d1", "s1"]

    pd.DataFrame(
        {"candidate_id": ["a", "b"], "group_key": ["s1", "d1"], "target_label": [0, 1]}
    ).to_csv(path, index=False)
    with pytest.raises(FormalProtocolError, match="exactly candidate_id and group_key"):
        resolve_group_keys_from_map(["a"], path)

    pd.DataFrame(
        {"candidate_id": ["a"], "group_key": ["s1"]}
    ).to_csv(path, index=False)
    with pytest.raises(FormalProtocolError, match="missing candidate IDs"):
        resolve_group_keys_from_map(["b"], path)
