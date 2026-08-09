"""Method-independent deterministic seed policy for paired MC dropout."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence


SEED_POLICY_VERSION = "egdatpp_mc_mask_v1"


def _non_negative_integer(name: str, value: int) -> int:
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted


def deterministic_mask_seed(
    experiment_seed: int,
    acquisition_round: int,
    pass_index: int,
    model_refit_index: int,
) -> int:
    """Derive one positive-63-bit torch seed from acquisition state only."""

    payload = {
        "acquisition_round": _non_negative_integer("acquisition_round", acquisition_round),
        "experiment_seed": _non_negative_integer("experiment_seed", experiment_seed),
        "model_refit_index": _non_negative_integer("model_refit_index", model_refit_index),
        "pass_index": _non_negative_integer("pass_index", pass_index),
        "policy_version": SEED_POLICY_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def mc_mask_seeds(
    k: int,
    *,
    experiment_seed: int,
    acquisition_round: int,
    model_refit_index: int,
) -> list[int]:
    """Return the reproducible nested MC seed prefix of length K."""

    count = int(k)
    if count < 1:
        raise ValueError("K must be positive")
    return [
        deterministic_mask_seed(
            experiment_seed,
            acquisition_round,
            pass_index,
            model_refit_index,
        )
        for pass_index in range(count)
    ]


def mask_sequence_sha256(
    k: int,
    *,
    experiment_seed: int,
    acquisition_round: int,
    model_refit_index: int,
) -> str:
    """Hash the exact ordered seed sequence stored in experiment metadata."""

    seeds: Sequence[int] = mc_mask_seeds(
        k,
        experiment_seed=experiment_seed,
        acquisition_round=acquisition_round,
        model_refit_index=model_refit_index,
    )
    payload = {
        "policy_version": SEED_POLICY_VERSION,
        "seeds": list(seeds),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
