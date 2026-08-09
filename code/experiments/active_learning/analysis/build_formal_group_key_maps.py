"""Build frozen, label-blind Mn-oxide group-key maps for formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from analysis.build_group_key_inventory import build_group_keys


FORMAL_GROUP_KEY_MODES = (
    "element_system_current",
    "coelement_block_multiset",
    "coelement_iupac_group_set",
)


def build_formal_group_maps(metadata: pd.DataFrame) -> dict[str, pd.DataFrame]:
    keys = build_group_keys(metadata)
    maps: dict[str, pd.DataFrame] = {}
    for mode in FORMAL_GROUP_KEY_MODES:
        frame = keys.loc[:, ["candidate_id", mode]].rename(columns={mode: "group_key"})
        frame = frame.sort_values("candidate_id").reset_index(drop=True)
        if frame["candidate_id"].duplicated().any() or frame["group_key"].astype(str).eq("").any():
            raise ValueError(f"invalid formal group-key map for {mode}")
        maps[mode] = frame
    return maps


def write_formal_group_maps(
    metadata: pd.DataFrame, output_directory: Path
) -> dict[str, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for mode, frame in build_formal_group_maps(metadata).items():
        path = output / f"mnoxide_{mode}.csv"
        with path.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
        paths[mode] = path
    return paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(
        args.metadata,
        usecols=["candidate_id", "chemsys", "nelements"],
        dtype={"candidate_id": str},
    )
    paths = write_formal_group_maps(metadata, args.output_directory)
    print(
        json.dumps(
            {
                mode: {"path": str(path), "sha256": _sha256(path)}
                for mode, path in paths.items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
