"""Frozen task overlays for the Mn/Mg interval-robustness GPU runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import pandas as pd

try:
    from .two_dataset_paired_protocol import candidate_order_digest, clean_id
except ImportError:  # Direct execution by the server runner.
    from two_dataset_paired_protocol import candidate_order_digest, clean_id


TASK_FIELDS = {
    "task_version",
    "task_id",
    "base_dataset",
    "target_low",
    "target_high",
    "target_count",
    "pool_size",
    "budget",
    "batch_size",
    "rounds",
    "initial_set_size",
    "initial_sets_relative_path",
    "checkpoints",
    "label_source",
    "hidden_evaluability_role",
    "frozen",
}


class IntervalTaskError(ValueError):
    """Raised when an interval task or its frozen initial set is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IntervalTaskError("task root must be a mapping")
    return value


def _safe_relative_path(value: object) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise IntervalTaskError("initial-set path must be safe and project-relative")
    return path.as_posix()


@dataclass(frozen=True)
class IntervalTask:
    task_version: str
    task_id: str
    base_dataset: str
    target_low: float
    target_high: float
    target_count: int
    pool_size: int
    budget: int
    batch_size: int
    rounds: int
    initial_set_size: int
    initial_sets_relative_path: str
    checkpoints: tuple[int, ...]
    label_source: str
    hidden_evaluability_role: str
    frozen: bool
    source_path: Path
    sha256: str

    def apply(self, base):
        if base.name != self.base_dataset:
            raise IntervalTaskError(
                f"task base dataset {self.base_dataset!r} does not match {base.name!r}"
            )
        return replace(
            base,
            target_low=self.target_low,
            target_high=self.target_high,
            target_count=self.target_count,
            budget=self.budget,
            batch_size=self.batch_size,
            rounds=self.rounds,
            label_source=self.label_source,
            initial_labeled_set=f"frozen_seed_specific_{self.initial_set_size}",
        )


def load_interval_task(path: str | Path) -> IntervalTask:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _load_mapping(source)
    unknown = set(payload).difference(TASK_FIELDS)
    missing = TASK_FIELDS.difference(payload)
    if unknown:
        raise IntervalTaskError(f"task has unknown fields: {sorted(unknown)}")
    if missing:
        raise IntervalTaskError(f"task is missing fields: {sorted(missing)}")
    if not bool(payload["frozen"]):
        raise IntervalTaskError("interval task must be frozen before GPU execution")
    low = float(payload["target_low"])
    high = float(payload["target_high"])
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise IntervalTaskError("target_low must be finite and less than target_high")
    pool_size = int(payload["pool_size"])
    target_count = int(payload["target_count"])
    budget = int(payload["budget"])
    batch_size = int(payload["batch_size"])
    rounds = int(payload["rounds"])
    initial_size = int(payload["initial_set_size"])
    if pool_size <= 0 or target_count <= 0 or target_count > pool_size:
        raise IntervalTaskError("pool and target counts are invalid")
    if initial_size <= 0 or initial_size >= budget:
        raise IntervalTaskError("initial_set_size must be positive and below the budget")
    if batch_size <= 0 or budget > pool_size or (budget - initial_size) % batch_size:
        raise IntervalTaskError("budget must accommodate complete batches after initialization")
    expected_rounds = (budget - initial_size) // batch_size
    if rounds != expected_rounds:
        raise IntervalTaskError(f"rounds must equal {expected_rounds} for this budget")
    checkpoints = tuple(int(value) for value in payload["checkpoints"])
    if not checkpoints or tuple(sorted(set(checkpoints))) != checkpoints:
        raise IntervalTaskError("checkpoints must be non-empty, unique, and increasing")
    if checkpoints[-1] > budget or checkpoints[0] < initial_size:
        raise IntervalTaskError("checkpoints must lie within the evaluated budget")
    if str(payload["hidden_evaluability_role"]) != "post_selection_only":
        raise IntervalTaskError("hidden evaluability is restricted to post_selection_only")
    return IntervalTask(
        task_version=str(payload["task_version"]),
        task_id=str(payload["task_id"]),
        base_dataset=str(payload["base_dataset"]),
        target_low=low,
        target_high=high,
        target_count=target_count,
        pool_size=pool_size,
        budget=budget,
        batch_size=batch_size,
        rounds=rounds,
        initial_set_size=initial_size,
        initial_sets_relative_path=_safe_relative_path(payload["initial_sets_relative_path"]),
        checkpoints=checkpoints,
        label_source=str(payload["label_source"]),
        hidden_evaluability_role=str(payload["hidden_evaluability_role"]),
        frozen=True,
        source_path=source,
        sha256=_sha256(source),
    )


def load_initial_set_ids(
    path: str | Path,
    *,
    seed: int,
    pool_ids: Sequence[str],
    expected_size: int,
) -> list[str]:
    source = Path(path)
    frame = pd.read_csv(source, dtype=str)
    required = {"seed", "candidate_id", "initial_set_sha256"}
    if not required.issubset(frame.columns):
        raise IntervalTaskError(f"initial-set table is missing columns: {sorted(required - set(frame))}")
    selected = frame[pd.to_numeric(frame["seed"], errors="coerce") == int(seed)].copy()
    ids = [clean_id(value) for value in selected["candidate_id"].tolist()]
    if len(ids) != int(expected_size) or len(ids) != len(set(ids)):
        raise IntervalTaskError(f"seed {seed} must have {expected_size} unique initial candidates")
    pool = {clean_id(value) for value in pool_ids}
    missing = sorted(set(ids) - pool)
    if missing:
        raise IntervalTaskError(f"initial candidates are not in the frozen pool: {missing}")
    hashes = selected["initial_set_sha256"].dropna().unique().tolist()
    # The frozen CPU replay stored the membership hash in sorted-ID order.
    expected_hash = candidate_order_digest(sorted(ids))
    if hashes != [expected_hash]:
        raise IntervalTaskError(
            f"initial-set hash mismatch for seed {seed}: expected {expected_hash}, found {hashes}"
        )
    return ids
