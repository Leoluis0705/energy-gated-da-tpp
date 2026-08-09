"""Build immutable assets for the held-out structural-group feasibility run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.reproducibility.two_dataset_paired_protocol import (
    candidate_order_digest,
    clean_id,
)


INITIAL_NAMESPACE = "structural_gate_holdout_v1"
PROTOCOL_VERSION = "egdatpp_structgate_feas_v1"
SEEDS = tuple(range(111, 116))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_initial_ids(pool_ids: Sequence[str], seed: int) -> list[str]:
    cleaned = [clean_id(value) for value in pool_ids]
    if len(cleaned) != len(set(cleaned)) or len(cleaned) < 4:
        raise ValueError("pool IDs must contain at least four unique candidates")

    def key(candidate_id: str) -> str:
        payload = f"{INITIAL_NAMESPACE}:{int(seed)}:{candidate_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return sorted(cleaned, key=key)[:4]


def build_group_map(pool: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "structure_matcher_cluster"}
    if not required.issubset(pool.columns):
        raise ValueError(f"pool master is missing columns: {sorted(required - set(pool))}")
    result = pool.loc[:, ["candidate_id", "structure_matcher_cluster"]].copy()
    result.columns = ["candidate_id", "group_key"]
    result["candidate_id"] = result["candidate_id"].map(clean_id)
    result["group_key"] = result["group_key"].astype(str).str.strip()
    if result.shape[0] != 640:
        raise ValueError(f"expected 640 candidates, found {result.shape[0]}")
    if result["candidate_id"].duplicated().any() or result["group_key"].eq("").any():
        raise ValueError("candidate IDs must be unique and structural group keys non-empty")
    return result


def build_initial_set_table(
    pool_ids: Sequence[str], *, seeds: Iterable[int] = SEEDS
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        selected = deterministic_initial_ids(pool_ids, int(seed))
        membership_hash = candidate_order_digest(sorted(selected))
        for rank, candidate_id in enumerate(selected):
            rows.append(
                {
                    "seed": int(seed),
                    "candidate_id": candidate_id,
                    "selection_rank": rank,
                    "initial_set_sha256": membership_hash,
                }
            )
    return pd.DataFrame(rows)


def _protocol_payload(*, methods: list[str], group_mode: str, group_map: str | None) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "phase": "structural_group_feasibility",
        "dataset": "limo",
        "allowed_seeds": list(SEEDS),
        "allowed_methods": methods,
        "mc_passes": 30,
        "M0": 0.75,
        "G0": 0.50,
        "alpha": 0.10,
        "beta": 0.20,
        "gamma": 0.05,
        "group_key_mode": group_mode,
        "group_key_map_relative_path": group_map,
        "frozen": True,
    }


def _task_payload(*, task: str) -> dict:
    if task == "mn":
        low, high, target_count = -2.1, -1.9, 126
        task_id = "mn_structural_gate_feasibility_m2p1_m1p9"
        label_source = "frozen ALIGNN proxy formation energy; Mn DFT-anchored interval"
    elif task == "mg":
        low, high, target_count = -2.3, -2.1, 139
        task_id = "mg_structural_gate_feasibility_m2p3_m2p1"
        label_source = "frozen ALIGNN proxy formation energy; Mg DFT-anchored interval"
    else:
        raise ValueError(f"unsupported task: {task}")
    return {
        "task_version": "structural_group_gate_feasibility_v1",
        "task_id": task_id,
        "base_dataset": "limo",
        "target_low": low,
        "target_high": high,
        "target_count": target_count,
        "pool_size": 640,
        "budget": 320,
        "batch_size": 4,
        "rounds": 79,
        "initial_set_size": 4,
        "initial_sets_relative_path": (
            "configs/structural_group_feasibility/heldout_initial_sets.csv"
        ),
        "checkpoints": [80, 160, 240, 320],
        "label_source": label_source,
        "hidden_evaluability_role": "post_selection_only",
        "frozen": True,
    }


def write_assets(pool_master: Path, output_dir: Path) -> list[Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty asset directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.read_csv(pool_master)
    group_map = build_group_map(pool)
    initial_sets = build_initial_set_table(group_map["candidate_id"].tolist())

    written: list[Path] = []
    group_path = output_dir / "structure_matcher_group_map.csv"
    group_map.to_csv(group_path, index=False, lineterminator="\n")
    written.append(group_path)
    initial_path = output_dir / "heldout_initial_sets.csv"
    initial_sets.to_csv(initial_path, index=False, lineterminator="\n")
    written.append(initial_path)

    group_relative = "configs/structural_group_feasibility/structure_matcher_group_map.csv"
    payloads = {
        "legacy_protocol.json": _protocol_payload(
            methods=[
                "predicted_target_greedy",
                "energy_gated_da_tpp",
                "gradient_norm_hybrid",
            ],
            group_mode="element_system_current",
            group_map=None,
        ),
        "structural_protocol.json": _protocol_payload(
            methods=["structural_group_gate"],
            group_mode="structure_matcher_cluster",
            group_map=group_relative,
        ),
        "structural_q95_protocol.json": _protocol_payload(
            methods=["structural_group_gate_q95"],
            group_mode="structure_matcher_cluster",
            group_map=group_relative,
        ),
        "mn_task.json": _task_payload(task="mn"),
        "mg_task.json": _task_payload(task="mg"),
    }
    for filename, payload in payloads.items():
        path = output_dir / filename
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)

    manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(output_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(written)
        ]
    )
    manifest_path = output_dir / "SHA256SUMS.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    written.append(manifest_path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-master", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    written = write_assets(args.pool_master.resolve(), args.output_dir.resolve())
    print(f"WROTE {len(written)} frozen assets to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
