#!/usr/bin/env python3
"""Frozen protocol helpers for the two-dataset paired confirmation."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


EQUIVALENCE_MARGIN = 0.01
METHODS = ("energy_gated_da_tpp", "predicted_distance_greedy")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    pool_relative_path: str
    oracle_relative_path: str
    label_source: str
    target_low: float
    target_high: float
    target_count: int
    budget: int
    batch_size: int
    rounds: int
    M0: float
    G0: float
    alpha: float
    beta: float
    gamma: float
    refit_epochs: int = 10
    learning_rate: float = 0.001
    cgcnn_batch_size: int = 256
    mc_passes: int = 3
    dropout_rate: float = 0.30
    prediction_loader_shuffle: bool = False
    explicit_candidate_id_reindex: bool = True
    group_key_construction: str = "sorted_element_system_from_candidate_cif"
    initial_labeled_set: str = "empty"

    @property
    def selector_tuple(self) -> tuple[float, float, float, float, float]:
        return self.M0, self.G0, self.alpha, self.beta, self.gamma

    @property
    def target_interval(self) -> tuple[float, float]:
        return self.target_low, self.target_high

    def to_dict(self) -> dict:
        return asdict(self)


def dataset_configs() -> dict[str, DatasetConfig]:
    return {
        "limo": DatasetConfig(
            name="limo",
            pool_relative_path="EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_FLAT_20260617",
            oracle_relative_path="EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_20260617/oracle.csv",
            label_source="ALIGNN-derived formation energy",
            target_low=-2.18,
            target_high=-2.02,
            target_count=78,
            budget=640,
            batch_size=16,
            rounds=40,
            M0=1.0,
            G0=0.50,
            alpha=0.10,
            beta=0.20,
            gamma=0.10,
        ),
        "mnoxide": DatasetConfig(
            name="mnoxide",
            pool_relative_path="NON_GEN_INTERVAL_POOLS_20260618/Mn_NON_GEN_HARD640_FLAT_20260619",
            oracle_relative_path="NON_GEN_INTERVAL_POOLS_20260618/Mn_NON_GEN_HARD640_M2P59_M2P47_111_20260709/oracle.csv",
            label_source="Materials Project formation energy",
            target_low=-2.59,
            target_high=-2.47,
            target_count=111,
            budget=320,
            batch_size=16,
            rounds=20,
            M0=1.0,
            G0=0.75,
            alpha=0.05,
            beta=0.20,
            gamma=0.10,
        ),
    }


def build_planned_runs(seeds: Iterable[int]) -> list[dict]:
    rows: list[dict] = []
    for dataset in ("limo", "mnoxide"):
        for seed in seeds:
            for method in METHODS:
                rows.append({"dataset": dataset, "method": method, "seed": int(seed)})
    return rows


def paired_round_seeds(seed: int, round_index: int) -> dict[str, int]:
    base = int(seed) * 1_000_000
    return {
        "python_seed": int(seed),
        "numpy_seed": int(seed),
        "torch_seed": int(seed),
        "cuda_seed": int(seed),
        "dataloader_seed": int(seed),
        "inference_seed": base + 100_000 + int(round_index),
        "training_seed": base + 200_000 + int(round_index),
    }


def clean_id(value: object) -> str:
    return str(value).replace(".cif", "").strip()


def reindex_predictions(frame: pd.DataFrame, candidate_ids: Sequence[str]) -> pd.DataFrame:
    if "id" not in frame.columns:
        raise ValueError("prediction table has no id column")
    result = frame.copy()
    result["id"] = result["id"].map(clean_id)
    expected = [clean_id(value) for value in candidate_ids]
    if result["id"].duplicated().any():
        raise ValueError("duplicate candidate IDs in prediction table")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate candidate IDs in eligible pool")
    actual_set = set(result["id"])
    expected_set = set(expected)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f"candidate-ID mismatch: missing={missing[:10]}, extra={extra[:10]}")
    return result.set_index("id").loc[expected].reset_index()


def candidate_order_digest(candidate_ids: Sequence[str]) -> str:
    payload = "\n".join(clean_id(value) for value in candidate_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(sha256_file(path))
        digest.update(file_hash)
    return digest.hexdigest()


def repetition_rate(group_keys: Sequence[str]) -> float:
    if not group_keys:
        return 0.0
    return (len(group_keys) - len(set(group_keys))) / float(len(group_keys))


def protocol_compatible(
    left: Mapping[str, object], right: Mapping[str, object], fields: Iterable[str] | None = None
) -> tuple[bool, dict[str, tuple[object, object]]]:
    compare_fields = list(fields) if fields is not None else sorted(set(left) | set(right))
    differences = {
        field: (left.get(field), right.get(field))
        for field in compare_fields
        if left.get(field) != right.get(field)
    }
    return not differences, differences

