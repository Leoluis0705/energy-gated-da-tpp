from pathlib import Path
import subprocess
import sys

import pandas as pd
from pandas.testing import assert_frame_equal

from analysis.build_structural_gate_feasibility_assets import (
    build_group_map,
    build_initial_set_table,
    deterministic_initial_ids,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_MASTER = (
    PROJECT_ROOT.parent
    / "hidden_evaluability"
    / "inputs"
    / "three_system"
    / "candidate_pool_master.csv"
)


def test_asset_builder_runs_as_a_direct_server_script():
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "analysis" / "build_structural_gate_feasibility_assets.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_deterministic_initial_ids_has_literal_hash_order():
    assert deterministic_initial_ids(["a", "b", "c", "d", "e"], 111) == [
        "b",
        "d",
        "e",
        "c",
    ]


def test_group_map_is_exact_and_label_blind():
    pool = pd.read_csv(POOL_MASTER)
    group_map = build_group_map(pool)

    assert group_map.columns.tolist() == ["candidate_id", "group_key"]
    assert group_map.shape == (640, 2)
    assert group_map["candidate_id"].is_unique
    assert group_map["group_key"].nunique() == 148
    assert int(group_map["group_key"].value_counts().max()) == 41
    assert int((group_map["group_key"].value_counts() == 1).sum()) == 84

    permuted = pool.copy()
    permuted["target_label"] = list(reversed(permuted["target_label"].tolist()))
    assert_frame_equal(group_map, build_group_map(permuted))


def test_heldout_initial_sets_are_distinct_reproducible_and_complete():
    pool_ids = pd.read_csv(POOL_MASTER)["candidate_id"].astype(str).tolist()

    first = build_initial_set_table(pool_ids, seeds=range(111, 116))
    second = build_initial_set_table(pool_ids, seeds=range(111, 116))

    assert_frame_equal(first, second)
    assert first.shape == (20, 4)
    assert first.groupby("seed")["candidate_id"].nunique().eq(4).all()
    assert first.groupby("seed")["initial_set_sha256"].nunique().eq(1).all()
    assert first.groupby("seed")["initial_set_sha256"].first().nunique() == 5
