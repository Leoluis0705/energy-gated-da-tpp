from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def left_continuous_autc(
    query_counts: Sequence[int] | np.ndarray,
    recoveries: Sequence[int] | np.ndarray,
    total_targets: int,
    budget: int,
) -> float:
    queries = np.asarray(query_counts, dtype=int)
    hits = np.asarray(recoveries, dtype=int)
    if queries.ndim != 1 or hits.ndim != 1 or len(queries) != len(hits):
        raise ValueError("query counts and recoveries must be equal-length one-dimensional arrays")
    if total_targets <= 0 or budget <= 0:
        raise ValueError("total_targets and budget must be positive")
    if len(queries) and (np.any(np.diff(queries) <= 0) or np.any(np.diff(hits) < 0)):
        raise ValueError("query counts must increase and recoveries must be nondecreasing")
    if len(queries) and (queries[0] <= 0 or hits[0] < 0):
        raise ValueError("query counts must be positive and recoveries nonnegative")

    area = 0
    previous_queries = 0
    previous_hits = 0
    for query_count, recovery in zip(queries, hits, strict=True):
        clipped_queries = min(int(query_count), int(budget))
        area += max(0, clipped_queries - previous_queries) * previous_hits
        previous_queries = clipped_queries
        previous_hits = int(recovery)
        if previous_queries >= budget:
            break
    if previous_queries < budget:
        area += (budget - previous_queries) * previous_hits
    return float(area / (total_targets * budget))


def recovery_at(
    query_counts: Sequence[int] | np.ndarray,
    recoveries: Sequence[int] | np.ndarray,
    checkpoint: int,
) -> int:
    queries = np.asarray(query_counts, dtype=int)
    hits = np.asarray(recoveries, dtype=int)
    if len(queries) != len(hits):
        raise ValueError("query counts and recoveries must have equal length")
    available = np.flatnonzero(queries <= int(checkpoint))
    return int(hits[available[-1]]) if len(available) else 0


def candidate_sequence_hash(candidate_ids: Iterable[str]) -> str:
    payload = "".join(f"{candidate_id}\n" for candidate_id in candidate_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def first_attainment_positions(target_labels: Iterable[int]) -> dict[int, int]:
    positions: dict[int, int] = {}
    cumulative = 0
    for position, value in enumerate(target_labels, start=1):
        label = int(value)
        if label not in (0, 1):
            raise ValueError("target labels must be binary")
        if label:
            cumulative += 1
            positions[cumulative] = position
    return positions


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) < 1:
        raise ValueError("bootstrap values must be a non-empty one-dimensional array")
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(int(samples), len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def write_bytes_protected(
    path: Path,
    content: bytes,
    check_existing: bool,
    create_if_missing_during_check: bool = False,
) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not check_existing:
            raise FileExistsError(f"refusing to overwrite existing audit output: {target}")
        existing = target.read_bytes()
        if existing != content:
            raise RuntimeError(f"existing audit output differs from recomputation: {target}")
        return "verified_identical"
    if check_existing and not create_if_missing_during_check:
        raise FileNotFoundError(f"cannot verify missing audit output: {target}")
    with target.open("xb") as handle:
        handle.write(content)
    if check_existing:
        return "created_missing_during_verified_resume"
    return "created"
