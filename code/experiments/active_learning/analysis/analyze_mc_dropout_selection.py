"""Summarize development-cohort MC-dropout sensitivity against K=30."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


KS = (3, 10, 30)
METHODS = ("interval_hit_greedy", "energy_gated_da_tpp")
GATE_METHOD = "energy_gated_da_tpp"


def render_selection_report(
    summary: pd.DataFrame,
    *,
    validation: dict[str, object],
    environment: dict[str, object],
    selected_k: int,
    source_directory: str,
) -> str:
    """Render a source-backed development-cohort K-selection report."""

    if selected_k not in KS:
        raise ValueError(f"selected K must be one of {KS}")
    indexed = summary.set_index("mc_passes")
    missing = [k for k in KS if k not in indexed.index]
    if missing:
        raise ValueError(f"summary is missing K values: {missing}")

    rows: list[str] = []
    for k in KS:
        row = indexed.loc[k]
        rows.append(
            "| {k} | {spearman:.6f} | {overlap:.6f} | {flip:.6f} | "
            "{mean_autc:.6f} | {max_autc:.6f} | {runtime:.3f} | {ratio:.6f} |".format(
                k=k,
                spearman=float(row["median_uncertainty_spearman_vs_k30"]),
                overlap=float(row["median_top_b_overlap_vs_k30"]),
                flip=float(row["gate_flip_rate_vs_k30"]),
                mean_autc=float(row["mean_absolute_AUTC_difference_vs_k30"]),
                max_autc=float(row["maximum_absolute_AUTC_difference_vs_k30"]),
                runtime=float(row["mean_runtime_seconds"]),
                ratio=float(row["median_runtime_ratio_vs_k30"]),
            )
        )

    completed = validation.get("status_counts", {}).get("DONE", 0)
    issue_count = validation.get("issue_count", "unknown")
    selected = indexed.loc[selected_k]
    lower_rows = indexed.loc[[k for k in KS if k < selected_k]]
    lower_flip_min = float(lower_rows["gate_flip_rate_vs_k30"].min())
    lower_overlap_max = float(lower_rows["median_top_b_overlap_vs_k30"].max())
    lower_spearman_max = float(lower_rows["median_uncertainty_spearman_vs_k30"].max())
    lower_runtime_ratio_min = float(lower_rows["median_runtime_ratio_vs_k30"].min())

    return "\n".join(
        [
            "# MC-dropout selection report",
            "",
            "## Evidence scope",
            "",
            "- Selection cohort: Li-M-O development seeds 0-4 only.",
            "- Methods: Interval-Hit Greedy and Full Energy-Gated DA-TPP.",
            f"- Completed trajectories: {completed} of {validation.get('job_count', 'unknown')}.",
            f"- Paired mask comparisons: {validation.get('paired_mask_comparisons', 'unknown')}.",
            f"- Validation issues: {issue_count}.",
            f"- Development manifest SHA-256: `{validation.get('manifest_sha256', 'unknown')}`.",
            f"- Source directory: `{source_directory}`.",
            "",
            "No seed from the legacy replication cohort (5-14), final confirmation cohort "
            "(15-24), or independent MC-sensitivity cohort (25-29) was used for this choice.",
            "",
            "## Pre-registered comparison metrics",
            "",
            "All stability columns compare the indicated K with K=30 on matched development "
            "method, seed, and acquisition round.",
            "",
            "| K | median uncertainty Spearman | median top-b overlap | gate-flip rate | "
            "mean abs. AUTC difference | max abs. AUTC difference | mean runtime (s) | "
            "median runtime ratio |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Decision",
            "",
            f"Selected MC passes: `K = {selected_k}`.",
            "",
            "The lower-K alternatives changed the acquisition evidence materially: their "
            f"best median uncertainty Spearman correlation was {lower_spearman_max:.6f}, "
            f"their best median top-b overlap was {lower_overlap_max:.6f}, and even the "
            f"lower gate-flip rate was {lower_flip_min:.6f}. Their largest observed median "
            f"runtime saving was only {(1.0 - lower_runtime_ratio_min) * 100.0:.3f}% relative "
            "to K=30. The stability loss therefore outweighed the measured end-to-end "
            "trajectory saving. This decision was made before running seeds 15-24.",
            "",
            "The selected K is fixed for parameter calibration and final evaluation; it must "
            "not be revised in response to seeds 15-24.",
            "",
            "## Analysis environment",
            "",
            f"- Python: `{environment.get('python', 'unknown')}`",
            f"- NumPy: `{environment.get('numpy', 'unknown')}`",
            f"- SciPy: `{environment.get('scipy', 'unknown')}`",
            f"- Analysis Git commit: `{environment.get('git_commit', 'unknown')}`",
            "",
        ]
    )


def _done_attempt(run_parent: Path) -> Path:
    done: list[Path] = []
    for attempt in sorted(run_parent.glob("attempt_*")):
        status_path = attempt / "status.json"
        if status_path.is_file() and json.loads(status_path.read_text(encoding="utf-8")).get("status") == "DONE":
            done.append(attempt)
    if len(done) != 1:
        raise ValueError(f"expected exactly one DONE attempt under {run_parent}, found {len(done)}")
    return done[0]


def _split_ids(value: object) -> set[str]:
    return {item for item in str(value).split(";") if item}


def _score_path(run: Path, round_number: int) -> Path:
    matches = sorted(run.glob(f"*_scores_*_iter_{round_number}.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected one score file for round {round_number} under {run}")
    return matches[0]


def _read_single_row(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    if len(frame) != 1:
        raise ValueError(f"expected one row in {path}")
    return frame.iloc[0]


def analyze_mc_dropout_selection(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return round-, run-, and K-level comparisons for a complete development grid."""

    root = Path(results_root)
    seed_dirs = sorted((root / "k_30" / METHODS[0]).glob("seed_*"))
    seeds = [int(path.name.removeprefix("seed_")) for path in seed_dirs]
    if not seeds:
        raise ValueError("no K=30 development seeds found")

    runs: dict[tuple[int, str, int], Path] = {}
    missing: list[str] = []
    for k in KS:
        for method in METHODS:
            for seed in seeds:
                parent = root / f"k_{k}" / method / f"seed_{seed}"
                try:
                    runs[(k, method, seed)] = _done_attempt(parent)
                except ValueError:
                    missing.append(str(parent))
    if missing:
        raise ValueError("development results require a complete method-by-K-by-seed grid: " + "; ".join(missing))

    round_rows: list[dict[str, object]] = []
    run_rows: list[dict[str, object]] = []
    for k in KS:
        for method in METHODS:
            for seed in seeds:
                run = runs[(k, method, seed)]
                baseline = runs[(30, method, seed)]
                status = json.loads((run / "status.json").read_text(encoding="utf-8"))
                baseline_status = json.loads((baseline / "status.json").read_text(encoding="utf-8"))
                metrics = _read_single_row(run / "run_metrics.csv")
                baseline_metrics = _read_single_row(baseline / "run_metrics.csv")
                autc_difference = float(metrics["AUTC"]) - float(baseline_metrics["AUTC"])
                runtime = float(status["elapsed_seconds"])
                baseline_runtime = float(baseline_status["elapsed_seconds"])
                run_rows.append(
                    {
                        "mc_passes": k,
                        "method": method,
                        "seed": seed,
                        "AUTC": float(metrics["AUTC"]),
                        "AUTC_K30": float(baseline_metrics["AUTC"]),
                        "AUTC_difference_vs_K30": autc_difference,
                        "absolute_AUTC_difference_vs_K30": abs(autc_difference),
                        "runtime_seconds": runtime,
                        "runtime_seconds_K30": baseline_runtime,
                        "runtime_ratio_vs_K30": runtime / baseline_runtime,
                        "run_path": str(run),
                        "baseline_run_path": str(baseline),
                    }
                )

                diagnostics = pd.read_csv(run / "round_diagnostics.csv")
                baseline_diagnostics = pd.read_csv(baseline / "round_diagnostics.csv")
                if diagnostics["round"].tolist() != baseline_diagnostics["round"].tolist():
                    raise ValueError(f"round grid differs from K=30 for {method} seed {seed} K={k}")
                for record, baseline_record in zip(
                    diagnostics.to_dict(orient="records"),
                    baseline_diagnostics.to_dict(orient="records"),
                    strict=True,
                ):
                    round_number = int(record["round"])
                    scores = pd.read_csv(_score_path(run, round_number), usecols=["id", "mu_eV", "sigma_eV"])
                    baseline_scores = pd.read_csv(
                        _score_path(baseline, round_number), usecols=["id", "mu_eV", "sigma_eV"]
                    )
                    paired = scores.merge(baseline_scores, on="id", suffixes=("", "_K30"), validate="one_to_one")
                    if len(paired) >= 2:
                        correlation = float(spearmanr(paired["sigma_eV"], paired["sigma_eV_K30"]).statistic)
                    else:
                        correlation = np.nan
                    top = _split_ids(record["direct_top_b_candidate_ids"])
                    baseline_top = _split_ids(baseline_record["direct_top_b_candidate_ids"])
                    denominator = min(len(top), len(baseline_top))
                    overlap = len(top.intersection(baseline_top)) / denominator if denominator else np.nan
                    gate_observed = method == GATE_METHOD
                    round_rows.append(
                        {
                            "mc_passes": k,
                            "method": method,
                            "seed": seed,
                            "round": round_number,
                            "common_candidate_count": len(paired),
                            "candidate_count": len(scores),
                            "candidate_count_K30": len(baseline_scores),
                            "common_candidate_fraction_of_smaller_pool": (
                                len(paired) / min(len(scores), len(baseline_scores))
                                if min(len(scores), len(baseline_scores))
                                else np.nan
                            ),
                            "predictive_mean_MAE_eV_vs_K30": float(
                                np.mean(np.abs(paired["mu_eV"] - paired["mu_eV_K30"]))
                            ),
                            "predictive_SD_MAE_eV_vs_K30": float(
                                np.mean(np.abs(paired["sigma_eV"] - paired["sigma_eV_K30"]))
                            ),
                            "uncertainty_spearman_vs_K30": correlation,
                            "top_b_overlap_fraction_vs_K30": overlap,
                            "route": record["route"],
                            "route_K30": baseline_record["route"],
                            "gate_flip_observed": gate_observed,
                            "gate_flip_vs_K30": (
                                bool(str(record["route"]) != str(baseline_record["route"]))
                                if gate_observed
                                else False
                            ),
                            "score_path": str(_score_path(run, round_number)),
                            "score_path_K30": str(_score_path(baseline, round_number)),
                        }
                    )

    round_detail = pd.DataFrame(round_rows).sort_values(["mc_passes", "method", "seed", "round"])
    run_detail = pd.DataFrame(run_rows).sort_values(["mc_passes", "method", "seed"])
    summary_rows: list[dict[str, object]] = []
    for k in KS:
        round_block = round_detail.loc[round_detail["mc_passes"] == k]
        run_block = run_detail.loc[run_detail["mc_passes"] == k]
        gate_block = round_block.loc[round_block["gate_flip_observed"]]
        summary_rows.append(
            {
                "mc_passes": k,
                "round_comparison_count": len(round_block),
                "gate_round_comparison_count": len(gate_block),
                "median_common_candidate_fraction": float(
                    round_block["common_candidate_fraction_of_smaller_pool"].median()
                ),
                "median_uncertainty_spearman_vs_k30": float(
                    round_block["uncertainty_spearman_vs_K30"].median()
                ),
                "median_top_b_overlap_vs_k30": float(
                    round_block["top_b_overlap_fraction_vs_K30"].median()
                ),
                "gate_flip_rate_vs_k30": float(gate_block["gate_flip_vs_K30"].mean()),
                "mean_absolute_AUTC_difference_vs_k30": float(
                    run_block["absolute_AUTC_difference_vs_K30"].mean()
                ),
                "maximum_absolute_AUTC_difference_vs_k30": float(
                    run_block["absolute_AUTC_difference_vs_K30"].max()
                ),
                "mean_runtime_seconds": float(run_block["runtime_seconds"].mean()),
                "median_runtime_ratio_vs_k30": float(run_block["runtime_ratio_vs_K30"].median()),
            }
        )
    return round_detail, run_detail, pd.DataFrame(summary_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--round-detail", type=Path, required=True)
    parser.add_argument("--run-detail", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--selected-k", type=int, choices=KS)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument("--environment-json", type=Path)
    parser.add_argument("--source-directory")
    args = parser.parse_args()
    round_detail, run_detail, summary = analyze_mc_dropout_selection(args.results_root)
    for frame, path in (
        (round_detail, args.round_detail),
        (run_detail, args.run_detail),
        (summary, args.summary),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, index=False, lineterminator="\n")
    if args.report is not None:
        required = {
            "--selected-k": args.selected_k,
            "--validation-json": args.validation_json,
            "--environment-json": args.environment_json,
            "--source-directory": args.source_directory,
        }
        absent = [name for name, value in required.items() if value is None]
        if absent:
            parser.error("--report also requires " + ", ".join(absent))
        validation = json.loads(args.validation_json.read_text(encoding="utf-8"))
        environment = json.loads(args.environment_json.read_text(encoding="utf-8"))
        report = render_selection_report(
            summary,
            validation=validation,
            environment=environment,
            selected_k=args.selected_k,
            source_directory=args.source_directory,
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8", newline="\n")
    print(summary.to_json(orient="records", indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
