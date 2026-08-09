from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.build_group_key_inventory import (
    DESIGN_ORDER,
    build_group_keys,
    group_inventory,
    load_formal_top_b_history,
)


def test_group_key_designs_use_only_prequery_composition_columns() -> None:
    metadata = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "chemsys": ["Ca-Mn-O", "Mg-Mn-O", "Bi-La-Mn-O"],
            "nelements": [3, 3, 4],
            "target_label": [0, 1, 1],
        }
    )
    changed_targets = metadata.assign(target_label=[1, 0, 0])
    first = build_group_keys(metadata)
    second = build_group_keys(changed_targets)
    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == ["candidate_id", *DESIGN_ORDER]


def test_candidate_designs_are_coarser_than_element_system_for_fixture() -> None:
    metadata = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c", "d"],
            "chemsys": ["Ca-Mn-O", "Mg-Mn-O", "Fe-Mn-O", "Co-Mn-O"],
            "nelements": [3, 3, 3, 3],
        }
    )
    keys = build_group_keys(metadata)
    baseline = keys["element_system_current"].nunique()
    assert baseline == 4
    for design in DESIGN_ORDER[1:]:
        assert keys[design].nunique() < baseline


def test_inventory_defines_both_singleton_denominators() -> None:
    keys = pd.Series(["A", "B", "B", "C", "C", "C"])
    inventory = group_inventory(keys)
    assert inventory["group_count"] == 3
    assert inventory["singleton_group_count"] == 1
    assert inventory["singleton_group_fraction"] == 1 / 3
    assert inventory["singleton_candidate_fraction"] == 1 / 6
    assert inventory["maximum_group_size"] == 3
    assert inventory["group_size_distribution_json"] == '{"1":1,"2":1,"3":1}'


def test_formal_mn_pool_reproduces_reported_element_system_inventory() -> None:
    archive = Path(__file__).resolve().parents[2]
    oracle = Path(
        r"D:\CGCNN\NON_GEN_INTERVAL_POOLS_20260618\Mn_NON_GEN_HARD640_M2P59_M2P47_111_20260709\oracle.csv"
    )
    metadata = pd.read_csv(
        oracle,
        usecols=["candidate_id", "chemsys", "nelements"],
        dtype={"candidate_id": str},
    )
    keys = build_group_keys(metadata)
    inventory = group_inventory(keys["element_system_current"])
    assert len(keys) == 640
    assert inventory["group_count"] == 614
    assert inventory["singleton_group_count"] == 588
    history = load_formal_top_b_history(archive)
    assert len(history) == 400
    assert history["candidate_ids"].map(len).eq(16).all()
