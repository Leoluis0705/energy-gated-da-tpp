"""Protocol-versioned paths and collision-safe tabular writes."""

from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


PROTOCOL_VERSION = "egdatpp_psfix_v1"
STRUCTURAL_GATE_PROTOCOL_VERSION = "egdatpp_structgate_feas_v1"
SUPPORTED_PROTOCOL_VERSIONS = {
    PROTOCOL_VERSION,
    STRUCTURAL_GATE_PROTOCOL_VERSION,
}


def validate_protocol_version(value: str) -> str:
    version = str(value)
    if version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise ValueError(f"unsupported protocol version: {version}")
    return version


def _safe_token(label: str, value: str) -> str:
    token = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", token):
        raise ValueError(f"{label} contains unsupported path characters: {token!r}")
    return token


def score_artifact_path(
    output_dir: str | Path,
    *,
    method_name: str,
    iteration: int,
    protocol_version: str = PROTOCOL_VERSION,
) -> Path:
    method = _safe_token("method_name", method_name)
    protocol = _safe_token("protocol_version", validate_protocol_version(protocol_version))
    round_index = int(iteration)
    if round_index < 1:
        raise ValueError("iteration must be positive")
    return Path(output_dir) / f"{method}_scores_{protocol}_iter_{round_index}.csv"


def trace_artifact_path(
    output_dir: str | Path,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> Path:
    protocol = _safe_token("protocol_version", validate_protocol_version(protocol_version))
    return Path(output_dir) / f"mode_trace_{protocol}.csv"


def write_dataframe_exclusive(frame: pd.DataFrame, path: str | Path) -> None:
    """Create a CSV once; refuse to replace any existing artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
