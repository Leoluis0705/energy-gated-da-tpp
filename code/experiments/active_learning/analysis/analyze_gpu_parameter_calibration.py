"""Analyze completed GPU calibration runs and prepare threshold promotion jobs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path, PurePosixPath

import pandas as pd


AUTC_TIE_TOLERANCE = 1e-4
CENTER = {"M0": 1.00, "G0": 0.50}
GRID_STEP = {"M0": 0.25, "G0": 0.10}
PROMOTION_SEEDS = (1, 2, 3, 4)


def _config_id(job_id: str) -> str:
    match = re.fullmatch(r"gpu_cal_(?:threshold|weight)_(.+)_seed\d+", str(job_id))
    if match is None:
        raise ValueError(f"unrecognized threshold-screen job ID: {job_id}")
    return match.group(1)


def _single_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected exactly one row in {path}")
    return frame.iloc[0]


def _center_distance(record: dict[str, object]) -> float:
    stage = str(record["calibration_stage"])
    if stage.startswith("threshold_"):
        m0 = float(record["M0"])
        g0 = float(record["G0"])
        return math.sqrt(
            ((m0 - CENTER["M0"]) / GRID_STEP["M0"]) ** 2
            + ((g0 - CENTER["G0"]) / GRID_STEP["G0"]) ** 2
        )
    if stage.startswith("weight_"):
        return math.sqrt(
            (float(record["alpha"]) - 0.10) ** 2
            + (float(record["beta"]) - 0.20) ** 2
            + (float(record["gamma"]) - 0.10) ** 2
        )
    raise ValueError(f"unsupported calibration stage: {stage}")


def summarize_completed_calibration(manifest_path: Path) -> pd.DataFrame:
    """Read run-level evidence for a completed calibration manifest."""

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    incomplete = manifest.loc[
        (manifest["status"] != "DONE") | (manifest["exit_code"].astype(str) != "0")
    ]
    if not incomplete.empty:
        raise ValueError(
            "calibration manifest is not complete: "
            + ", ".join(incomplete["job_id"].astype(str).tolist())
        )

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        run = Path(str(record["output_path"]))
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        if status.get("status") != "DONE":
            raise ValueError(f"run status is not DONE: {run}")
        metrics = _single_row(run / "run_metrics.csv")
        m0 = float(record["M0"])
        g0 = float(record["G0"])
        rows.append(
            {
                "config_id": _config_id(str(record["job_id"])),
                "seed": int(record["seed"]),
                "M0": m0,
                "G0": g0,
                "alpha": float(record["alpha"]),
                "beta": float(record["beta"]),
                "gamma": float(record["gamma"]),
                "mc_passes": int(record["K"]),
                "AUTC": float(metrics["AUTC"]),
                "correction_rounds": int(metrics["correction_rounds"]),
                "center_distance_grid_units": _center_distance(record),
                "runtime_seconds": float(status["elapsed_seconds"]),
                "candidate_sequence_sha256": str(metrics["candidate_sequence_sha256"]),
                "config_hash": str(record["config_hash"]),
                "git_commit": str(record["git_commit"]),
                "output_path": str(record["output_path"]),
                "output_sha256": str(record["sha256"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["config_id", "seed"]).reset_index(drop=True)


def _rank_rows(
    frame: pd.DataFrame, *, autc_column: str, correction_column: str
) -> pd.DataFrame:
    ordered = frame.sort_values([autc_column, "config_id"], ascending=[False, True]).copy()
    tie_groups: list[int] = []
    group = 0
    anchor: float | None = None
    for autc in ordered[autc_column].astype(float):
        if anchor is None or anchor - autc >= AUTC_TIE_TOLERANCE:
            group += 1
            anchor = autc
        tie_groups.append(group)
    ordered["AUTC_tie_group"] = tie_groups
    ordered = ordered.sort_values(
        ["AUTC_tie_group", correction_column, "center_distance_grid_units", "config_id"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    ordered.insert(0, "selection_rank", range(1, len(ordered) + 1))
    return ordered


def rank_calibration_configs(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank one result per configuration using the declared AUTC tie rule."""

    counts = summary.groupby("config_id")["seed"].size()
    if not (counts == 1).all():
        raise ValueError("threshold seed-0 ranking requires exactly one run per configuration")
    return _rank_rows(summary, autc_column="AUTC", correction_column="correction_rounds")


def aggregate_calibration_configs(
    summary: pd.DataFrame, *, expected_seeds: range | tuple[int, ...]
) -> pd.DataFrame:
    """Aggregate a complete matched development cohort by configuration."""

    expected = tuple(int(seed) for seed in expected_seeds)
    rows: list[dict[str, object]] = []
    fixed_columns = (
        "M0",
        "G0",
        "alpha",
        "beta",
        "gamma",
        "mc_passes",
        "center_distance_grid_units",
        "config_hash",
        "git_commit",
    )
    for config_id, block in summary.groupby("config_id", sort=True):
        seeds = tuple(sorted(block["seed"].astype(int).tolist()))
        if seeds != expected:
            raise ValueError(
                f"configuration {config_id} has seeds {seeds}; expected {expected}"
            )
        varying = [column for column in fixed_columns if block[column].nunique(dropna=False) != 1]
        if varying:
            raise ValueError(f"configuration {config_id} varies in fixed fields: {varying}")
        row: dict[str, object] = {
            "config_id": config_id,
            "seed_count": len(block),
            "seeds": ";".join(str(seed) for seed in seeds),
            "mean_AUTC": float(block["AUTC"].mean()),
            "sample_sd_AUTC": float(block["AUTC"].std(ddof=1)),
            "mean_correction_rounds": float(block["correction_rounds"].mean()),
            "sample_sd_correction_rounds": float(block["correction_rounds"].std(ddof=1)),
            "mean_runtime_seconds": float(block["runtime_seconds"].mean()),
            "candidate_sequence_sha256s": ";".join(
                sorted(block["candidate_sequence_sha256"].astype(str).unique())
            ),
        }
        row.update({column: block.iloc[0][column] for column in fixed_columns})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config_id").reset_index(drop=True)


def rank_aggregated_configs(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Rank matched-cohort means with the declared correction/distance tiebreaks."""

    return _rank_rows(
        aggregated,
        autc_column="mean_AUTC",
        correction_column="mean_correction_rounds",
    )


def _replace_argument(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise ValueError(f"command is missing {option}") from exc
    command[index + 1] = value


def _build_promotion_bundle(
    *,
    screen_manifest_path: Path,
    ranked_summary: pd.DataFrame,
    manifest_path: Path,
    remote_output_root: str,
    calibration_kind: str,
) -> pd.DataFrame:
    if calibration_kind not in {"threshold", "weight"}:
        raise ValueError(f"unsupported calibration kind: {calibration_kind}")

    screen = pd.read_csv(screen_manifest_path, keep_default_na=False)
    screen = screen.assign(config_id=screen["job_id"].map(_config_id)).set_index("config_id")
    selected = ranked_summary.loc[ranked_summary["selection_rank"] <= 3].sort_values(
        "selection_rank"
    )
    if len(selected) != 3:
        raise ValueError(f"exactly three {calibration_kind} configurations must be promoted")

    root = PurePosixPath(remote_output_root)
    rows: list[dict[str, object]] = []
    for selected_record in selected.to_dict(orient="records"):
        config_id = str(selected_record["config_id"])
        if config_id not in screen.index:
            raise ValueError(f"selected configuration is absent from screen manifest: {config_id}")
        source = screen.loc[config_id].to_dict()
        for seed in PROMOTION_SEEDS:
            record = dict(source)
            job_id = f"gpu_cal_{calibration_kind}_{config_id}_seed{seed:02d}"
            output_path = str(root / config_id / f"seed_{seed}" / "attempt_1")
            command = json.loads(str(record["command_json"]))
            _replace_argument(command, "--seed", str(seed))
            _replace_argument(command, "--run-dir", output_path)
            record.update(
                {
                    "job_id": job_id,
                    "seed": seed,
                    "status": "PENDING",
                    "start_time": "",
                    "end_time": "",
                    "exit_code": "",
                    "log_path": str(root / "logs" / f"{job_id}.log"),
                    "output_path": output_path,
                    "sha256": "",
                    "command_json": json.dumps(command, separators=(",", ":")),
                    "attempt": 1,
                    "pid": "",
                    "failure_reason": "",
                    "calibration_stage": f"{calibration_kind}_top3_seeds1_4",
                    "config_id": config_id,
                    "seed0_selection_rank": int(selected_record["selection_rank"]),
                }
            )
            rows.append(record)

    frame = pd.DataFrame(rows)
    if not frame["job_id"].is_unique or not frame["output_path"].is_unique:
        raise ValueError("promotion jobs must have unique IDs and output paths")
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n")
    return frame


def build_threshold_promotion_bundle(
    *,
    screen_manifest_path: Path,
    ranked_summary: pd.DataFrame,
    manifest_path: Path,
    remote_output_root: str,
) -> pd.DataFrame:
    """Create seeds 1-4 jobs for the top three seed-0 threshold configs."""

    return _build_promotion_bundle(
        screen_manifest_path=screen_manifest_path,
        ranked_summary=ranked_summary,
        manifest_path=manifest_path,
        remote_output_root=remote_output_root,
        calibration_kind="threshold",
    )


def build_weight_promotion_bundle(
    *,
    screen_manifest_path: Path,
    ranked_summary: pd.DataFrame,
    manifest_path: Path,
    remote_output_root: str,
) -> pd.DataFrame:
    """Create seeds 1-4 jobs for the top three seed-0 weight configs."""

    return _build_promotion_bundle(
        screen_manifest_path=screen_manifest_path,
        ranked_summary=ranked_summary,
        manifest_path=manifest_path,
        remote_output_root=remote_output_root,
        calibration_kind="weight",
    )


def build_combined_calibration_manifest(
    *,
    screen_manifest_path: Path,
    promotion_manifest_path: Path,
    ranked_seed0: pd.DataFrame,
    manifest_path: Path,
) -> pd.DataFrame:
    """Combine retained seed-0 rows with completed seeds 1-4 promotion rows."""

    selected = set(
        ranked_seed0.loc[ranked_seed0["selection_rank"] <= 3, "config_id"].astype(str)
    )
    if len(selected) != 3:
        raise ValueError("combined calibration manifest requires exactly three selected configs")
    screen = pd.read_csv(screen_manifest_path, keep_default_na=False)
    screen["config_id"] = screen["job_id"].map(_config_id)
    screen = screen.loc[screen["config_id"].isin(selected)].copy()
    promotion = pd.read_csv(promotion_manifest_path, keep_default_na=False)
    derived = promotion["job_id"].map(_config_id)
    if "config_id" in promotion and not promotion["config_id"].astype(str).equals(derived):
        raise ValueError("promotion config_id disagrees with job_id")
    promotion["config_id"] = derived
    combined = pd.concat([screen, promotion], ignore_index=True, sort=False)
    incomplete = combined.loc[
        (combined["status"] != "DONE") | (combined["exit_code"].astype(str) != "0")
    ]
    if not incomplete.empty:
        raise ValueError(
            "combined calibration manifest contains incomplete jobs: "
            + ", ".join(incomplete["job_id"].astype(str).tolist())
        )
    if not combined["job_id"].is_unique or not combined["output_path"].is_unique:
        raise ValueError("combined calibration job IDs and output paths must be unique")
    for config_id, block in combined.groupby("config_id"):
        seeds = sorted(block["seed"].astype(int).tolist())
        if seeds != list(range(5)):
            raise ValueError(f"configuration {config_id} has development seeds {seeds}")
    combined = combined.sort_values(["config_id", "seed"]).reset_index(drop=True)
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        combined.to_csv(handle, index=False, lineterminator="\n")
    return combined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--promotion-manifest", type=Path, required=True)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--calibration-kind", choices=("threshold", "weight"), default="threshold")
    args = parser.parse_args()

    summary = summarize_completed_calibration(args.screen_manifest)
    ranked = rank_calibration_configs(summary)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("x", encoding="utf-8", newline="") as handle:
        ranked.to_csv(handle, index=False, lineterminator="\n")
    promotion_builder = (
        build_threshold_promotion_bundle
        if args.calibration_kind == "threshold"
        else build_weight_promotion_bundle
    )
    promoted = promotion_builder(
        screen_manifest_path=args.screen_manifest,
        ranked_summary=ranked,
        manifest_path=args.promotion_manifest,
        remote_output_root=args.remote_output_root,
    )
    print(
        json.dumps(
            {
                "selected_config_ids": ranked.iloc[:3]["config_id"].tolist(),
                "promotion_jobs": len(promoted),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
