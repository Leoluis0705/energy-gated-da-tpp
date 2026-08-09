"""Validated protocol files and label-blind group maps for formal runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import pandas as pd


PROTOCOL_FIELDS = {
    "protocol_version",
    "phase",
    "dataset",
    "allowed_seeds",
    "allowed_methods",
    "mc_passes",
    "M0",
    "G0",
    "alpha",
    "beta",
    "gamma",
    "group_key_mode",
    "group_key_map_relative_path",
    "frozen",
}
PHASE_COHORTS = {
    "mc_dropout_development": range(0, 5),
    "threshold_calibration": range(0, 5),
    "weight_calibration": range(0, 5),
    "formal_evaluation": range(15, 25),
    "mc_dropout_sensitivity": range(25, 30),
    "interval_robustness": range(101, 111),
    "structural_group_feasibility": range(111, 116),
}
FORMAL_METHODS = {
    "interval_hit_greedy",
    "always_da_tpp",
    "margin_only_gate",
    "group_only_gate",
    "energy_gated_da_tpp",
    "predicted_target_greedy",
    "explore",
    "mc_dropout",
    "gradient_norm_hybrid",
    "random_sampling",
    "structural_group_gate",
    "structural_group_gate_q95",
}
GROUP_KEY_MODES = {
    "element_system_current",
    "coelement_block_multiset",
    "coelement_iupac_group_set",
    "structure_matcher_cluster",
}


class FormalProtocolError(ValueError):
    """Raised when a formal protocol violates a preregistered boundary."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - formal files are JSON-compatible YAML.
            raise FormalProtocolError("protocol is not JSON-compatible YAML and PyYAML is unavailable") from error
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise FormalProtocolError("protocol root must be a mapping")
    return value


def _safe_relative_path(value: object) -> str | None:
    if value is None:
        return None
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise FormalProtocolError("group-key map must be a safe project-relative path")
    return path.as_posix()


@dataclass(frozen=True)
class FormalProtocol:
    protocol_version: str
    phase: str
    dataset: str
    allowed_seeds: tuple[int, ...]
    allowed_methods: tuple[str, ...]
    mc_passes: int
    M0: float
    G0: float
    alpha: float
    beta: float
    gamma: float
    group_key_mode: str
    group_key_map_relative_path: str | None
    frozen: bool
    source_path: Path
    sha256: str

    def resolve_dataset_config(self, base, *, method: str, seed: int):
        if base.name != self.dataset:
            raise FormalProtocolError(
                f"protocol dataset {self.dataset!r} does not match requested dataset {base.name!r}"
            )
        if method not in self.allowed_methods:
            raise FormalProtocolError(f"method {method!r} is not allowed by protocol")
        cohort = PHASE_COHORTS[self.phase]
        if int(seed) not in cohort or int(seed) not in self.allowed_seeds:
            raise FormalProtocolError(
                f"seed {seed} is not allowed by protocol cohort {cohort.start}..{cohort.stop - 1}"
            )
        if self.phase in {
            "formal_evaluation",
            "mc_dropout_sensitivity",
            "interval_robustness",
            "structural_group_feasibility",
        } and not self.frozen:
            raise FormalProtocolError(f"phase {self.phase} must be frozen before execution")
        return replace(
            base,
            mc_passes=self.mc_passes,
            M0=self.M0,
            G0=self.G0,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            group_key_construction=self.group_key_mode,
        )


def load_formal_protocol(path: str | Path) -> FormalProtocol:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = _load_yaml_mapping(source)
    unknown = set(payload).difference(PROTOCOL_FIELDS)
    missing = PROTOCOL_FIELDS.difference(payload)
    if unknown:
        raise FormalProtocolError(f"protocol has unknown fields: {sorted(unknown)}")
    if missing:
        raise FormalProtocolError(f"protocol is missing fields: {sorted(missing)}")
    phase = str(payload["phase"])
    if phase not in PHASE_COHORTS:
        raise FormalProtocolError(f"unsupported protocol phase: {phase}")
    dataset = str(payload["dataset"])
    if dataset not in {"limo", "mnoxide"}:
        raise FormalProtocolError(f"unsupported dataset: {dataset}")
    seeds = tuple(int(value) for value in payload["allowed_seeds"])
    if len(seeds) != len(set(seeds)) or not seeds:
        raise FormalProtocolError("allowed_seeds must be non-empty and unique")
    methods = tuple(str(value) for value in payload["allowed_methods"])
    if len(methods) != len(set(methods)) or not methods:
        raise FormalProtocolError("allowed_methods must be non-empty and unique")
    unsupported_methods = set(methods).difference(FORMAL_METHODS)
    if unsupported_methods:
        raise FormalProtocolError(f"unsupported formal methods: {sorted(unsupported_methods)}")
    mc_passes = int(payload["mc_passes"])
    if mc_passes not in {3, 10, 30}:
        raise FormalProtocolError("mc_passes must be one of 3, 10, 30")
    numeric = {
        name: float(payload[name])
        for name in ("M0", "G0", "alpha", "beta", "gamma")
    }
    if not all(math.isfinite(value) for value in numeric.values()):
        raise FormalProtocolError("all gate parameters must be finite")
    group_key_mode = str(payload["group_key_mode"])
    if group_key_mode not in GROUP_KEY_MODES:
        raise FormalProtocolError(f"unsupported group-key mode: {group_key_mode}")
    group_map = _safe_relative_path(payload["group_key_map_relative_path"])
    if group_key_mode != "element_system_current" and not group_map:
        raise FormalProtocolError("noncurrent group-key mode requires a label-blind group-key map")
    return FormalProtocol(
        protocol_version=str(payload["protocol_version"]),
        phase=phase,
        dataset=dataset,
        allowed_seeds=seeds,
        allowed_methods=methods,
        mc_passes=mc_passes,
        M0=numeric["M0"],
        G0=numeric["G0"],
        alpha=numeric["alpha"],
        beta=numeric["beta"],
        gamma=numeric["gamma"],
        group_key_mode=group_key_mode,
        group_key_map_relative_path=group_map,
        frozen=bool(payload["frozen"]),
        source_path=source,
        sha256=_sha256(source),
    )


def resolve_group_keys_from_map(candidate_ids: Sequence[str], path: str | Path) -> list[str]:
    frame = pd.read_csv(path, dtype=str)
    if list(frame.columns) != ["candidate_id", "group_key"]:
        raise FormalProtocolError("group-key map must contain exactly candidate_id and group_key columns")
    frame["candidate_id"] = frame["candidate_id"].astype(str).str.replace(".cif", "", regex=False).str.strip()
    frame["group_key"] = frame["group_key"].astype(str).str.strip()
    if frame["candidate_id"].duplicated().any() or frame["group_key"].eq("").any():
        raise FormalProtocolError("group-key map candidate IDs must be unique and keys non-empty")
    mapping = dict(zip(frame["candidate_id"], frame["group_key"], strict=True))
    cleaned = [str(value).replace(".cif", "").strip() for value in candidate_ids]
    missing = [value for value in cleaned if value not in mapping]
    if missing:
        raise FormalProtocolError(f"group-key map is missing candidate IDs: {missing[:10]}")
    return [mapping[value] for value in cleaned]

