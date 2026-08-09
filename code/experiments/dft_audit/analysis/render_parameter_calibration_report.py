"""Render the development-only parameter search history and freeze report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PARAMETER_COLUMNS = ["M0", "G0", "alpha", "beta", "gamma", "mc_passes"]


def build_search_history(
    *,
    threshold_seed0: pd.DataFrame,
    threshold_full: pd.DataFrame,
    weight_seed0: pd.DataFrame,
    weight_full: pd.DataFrame,
) -> pd.DataFrame:
    stages = [
        ("threshold_seed0", threshold_seed0, "seed0"),
        ("threshold_seeds0_4", threshold_full, "seeds0_4"),
        ("weight_seed0", weight_seed0, "seed0"),
        ("weight_seeds0_4", weight_full, "seeds0_4"),
    ]
    blocks: list[pd.DataFrame] = []
    for stage, frame, cohort in stages:
        block = frame.copy()
        block.insert(0, "stage", stage)
        block.insert(1, "cohort", cohort)
        blocks.append(block)
    return pd.concat(blocks, ignore_index=True, sort=False)


def _table(frame: pd.DataFrame, *, aggregate: bool) -> str:
    metric_columns = (
        ["mean_AUTC", "sample_sd_AUTC", "mean_correction_rounds"]
        if aggregate
        else ["AUTC", "correction_rounds"]
    )
    columns = ["selection_rank", "config_id", *PARAMETER_COLUMNS, *metric_columns]
    return frame.loc[:, columns].to_markdown(index=False)


def render_parameter_calibration_report(
    *,
    threshold_seed0: pd.DataFrame,
    threshold_full: pd.DataFrame,
    weight_seed0: pd.DataFrame,
    weight_full: pd.DataFrame,
    selected_mc_passes: int,
    source_sha256: dict[str, str],
) -> str:
    if selected_mc_passes not in {3, 10, 30}:
        raise ValueError("selected MC passes must be 3, 10, or 30")
    if weight_full.empty or int(weight_full.iloc[0]["selection_rank"]) != 1:
        raise ValueError("weight full-cohort ranking must begin with selection rank 1")
    selected = weight_full.iloc[0]
    source_lines = [f"- {name}: `{digest}`" for name, digest in sorted(source_sha256.items())]
    return "\n".join(
        [
            "# Parameter calibration report",
            "",
            "## Information boundary",
            "",
            "All rankings in this report use Li-M-O development seeds 0-4 only. Seeds "
            "5-14, 15-24, and 25-29 were not used to choose MC passes or gate parameters.",
            "",
            f"The independently recorded MC-dropout development analysis selected `K = {selected_mc_passes}` "
            "before threshold calibration.",
            "",
            "Ranking uses higher AUTC first. Absolute AUTC differences strictly below "
            "`1e-4` are tied; fewer correction rounds then wins, followed by distance "
            "from the original center under the fixed distance definitions.",
            "",
            "## Threshold seed-0 screen",
            "",
            _table(threshold_seed0, aggregate=False),
            "",
            "## Threshold top-three on seeds 0-4",
            "",
            _table(threshold_full, aggregate=True),
            "",
            "## Weight seed-0 screen",
            "",
            _table(weight_seed0, aggregate=False),
            "",
            "## Weight top-three on seeds 0-4",
            "",
            _table(weight_full, aggregate=True),
            "",
            "## Final frozen choice",
            "",
            f"- `K = {selected_mc_passes}`",
            f"- `M0 = {float(selected['M0']):g}`",
            f"- `G0 = {float(selected['G0']):g}`",
            f"- `alpha = {float(selected['alpha']):g}`",
            f"- `beta = {float(selected['beta']):g}`",
            f"- `gamma = {float(selected['gamma']):g}`",
            "",
            "This choice must be frozen before any seed 15-24 result is generated and "
            "must not be changed in response to final-cohort outcomes.",
            "",
            "## Source SHA-256",
            "",
            *source_lines,
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold-seed0", type=Path, required=True)
    parser.add_argument("--threshold-full", type=Path, required=True)
    parser.add_argument("--weight-seed0", type=Path, required=True)
    parser.add_argument("--weight-full", type=Path, required=True)
    parser.add_argument("--selected-mc-passes", type=int, required=True)
    parser.add_argument("--source-sha256-json", type=Path, required=True)
    parser.add_argument("--search-history", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frames = {
        "threshold_seed0": pd.read_csv(args.threshold_seed0),
        "threshold_full": pd.read_csv(args.threshold_full),
        "weight_seed0": pd.read_csv(args.weight_seed0),
        "weight_full": pd.read_csv(args.weight_full),
    }
    source_sha256 = json.loads(args.source_sha256_json.read_text(encoding="utf-8"))
    history = build_search_history(**frames)
    report = render_parameter_calibration_report(
        **frames,
        selected_mc_passes=args.selected_mc_passes,
        source_sha256=source_sha256,
    )
    args.search_history.parent.mkdir(parents=True, exist_ok=True)
    with args.search_history.open("x", encoding="utf-8", newline="") as handle:
        history.to_csv(handle, index=False, lineterminator="\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
