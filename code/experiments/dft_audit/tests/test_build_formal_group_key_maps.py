from pathlib import Path

import pandas as pd

from analysis.build_formal_group_key_maps import (
    FORMAL_GROUP_KEY_MODES,
    build_formal_group_maps,
    write_formal_group_maps,
)


def test_formal_maps_use_only_candidate_id_and_prequery_composition() -> None:
    metadata = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "chemsys": ["Ca-Mn-O", "Mg-Mn-O", "Bi-La-Mn-O"],
            "nelements": [3, 3, 4],
            "target_label": [0, 1, 1],
        }
    )
    changed = metadata.assign(target_label=[1, 0, 0])

    first = build_formal_group_maps(metadata)
    second = build_formal_group_maps(changed)

    assert set(first) == set(FORMAL_GROUP_KEY_MODES)
    for mode in FORMAL_GROUP_KEY_MODES:
        assert first[mode].columns.tolist() == ["candidate_id", "group_key"]
        pd.testing.assert_frame_equal(first[mode], second[mode])


def test_writer_creates_one_protected_map_per_formal_mode(tmp_path: Path) -> None:
    metadata = pd.DataFrame(
        {
            "candidate_id": ["a", "b"],
            "chemsys": ["Ca-Mn-O", "Mg-Mn-O"],
            "nelements": [3, 3],
        }
    )
    paths = write_formal_group_maps(metadata, tmp_path)

    assert set(paths) == set(FORMAL_GROUP_KEY_MODES)
    for mode, path in paths.items():
        assert path == tmp_path / f"mnoxide_{mode}.csv"
        frame = pd.read_csv(path, dtype=str)
        assert frame.columns.tolist() == ["candidate_id", "group_key"]
        assert len(frame) == 2

