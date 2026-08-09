"""Audit archived formal cosine values and exact/near score ties."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "id",
    "P_hit",
    "U_i",
    "group_key",
    "mode",
    "similarity_mode",
    "selection_score",
    "selected_max_similarity",
    "selected",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_score(
    *,
    p_hit: float,
    uncertainty: float,
    max_similarity: float,
    group_penalty: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    return float(
        p_hit
        + alpha * uncertainty
        - beta * max_similarity
        - gamma * group_penalty
    )


def summarize_similarity_values(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "n": 0,
            "minimum": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "maximum": np.nan,
            "mean": np.nan,
            "negative_count": 0,
            "zero_count": 0,
            "positive_count": 0,
        }
    quantiles = np.quantile(array, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "n": int(array.size),
        "minimum": float(array.min()),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "negative_count": int(np.count_nonzero(array < 0.0)),
        "zero_count": int(np.count_nonzero(array == 0.0)),
        "positive_count": int(np.count_nonzero(array > 0.0)),
    }


def _tie_summary(values: np.ndarray) -> dict[str, int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"groups": 0, "candidates": 0, "largest_group": 0}
    _, counts = np.unique(finite, return_counts=True)
    tied = counts[counts > 1]
    return {
        "groups": int(tied.size),
        "candidates": int(tied.sum()) if tied.size else 0,
        "largest_group": int(tied.max()) if tied.size else 0,
    }


def _minimum_positive_gap(values: np.ndarray) -> float:
    unique = np.unique(values[np.isfinite(values)])
    if unique.size < 2:
        return np.nan
    gaps = np.diff(np.sort(unique))
    positive = gaps[gaps > 0]
    return float(positive.min()) if positive.size else np.nan


def _explicit_order(ids: list[str], scores: np.ndarray) -> list[int]:
    if not np.isfinite(scores).all():
        raise ValueError("ranking scores must be finite")
    return sorted(range(len(ids)), key=lambda index: (-float(scores[index]), ids[index]))


def _boundary_tie(scores: np.ndarray, boundary: int) -> tuple[int, float]:
    if boundary <= 0 or boundary >= len(scores):
        return 0, np.nan
    ordered_values = np.sort(scores)[::-1]
    value = float(ordered_values[boundary - 1])
    greater = int(np.count_nonzero(scores > value))
    equal = int(np.count_nonzero(scores == value))
    crosses = greater < boundary < greater + equal
    next_value = float(ordered_values[boundary])
    return (equal if crosses else 0), float(value - next_value)


def _round_from_score_name(path: Path) -> int:
    match = re.search(r"_iter_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"cannot parse round from {path}")
    return int(match.group(1))


def audit_formal_results(formal_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cosine_rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    aggregate_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    run_configs = sorted(formal_root.rglob("run_config.json"))
    for config_path in run_configs:
        run_dir = config_path.parent
        config = json.loads(config_path.read_text(encoding="utf-8"))
        trace_paths = sorted(run_dir.glob("mode_trace_*.csv"))
        if len(trace_paths) != 1:
            raise ValueError(f"expected one mode trace in {run_dir}, found {len(trace_paths)}")
        trace = pd.read_csv(trace_paths[0]).set_index("iteration")
        dataset = str(config["name"])
        method = str(config["method"])
        group_key_mode = str(config.get("group_key_mode", config.get("group_key_construction")))
        seed = int(config["seed"])
        score_paths = sorted(run_dir.glob("*scores_*_iter_*.csv"), key=_round_from_score_name)
        for score_path in score_paths:
            round_index = _round_from_score_name(score_path)
            trace_row = trace.loc[round_index]
            score = pd.read_csv(score_path, usecols=lambda column: column in SCORE_COLUMNS)
            ids = score["id"].astype(str).tolist()
            p_hit = pd.to_numeric(score["P_hit"], errors="raise").to_numpy(float)
            uncertainty = pd.to_numeric(score["U_i"], errors="raise").to_numpy(float)
            alpha = float(trace_row["alpha"])
            quality = p_hit + alpha * uncertainty
            batch_size = int(trace_row["batch_size"])
            prefilter_size = int(trace_row["prefilter_size"])
            mode = str(trace_row["mode"])
            similarity_mode = str(trace_row["similarity_mode"])
            source_relative = score_path.relative_to(formal_root).as_posix()
            source_sha = sha256_file(score_path)

            selected_similarity = pd.to_numeric(
                score["selected_max_similarity"], errors="coerce"
            ).dropna().to_numpy(float)
            evidence_status = (
                "available_selected_candidates"
                if selected_similarity.size
                else ("not_applicable_direct_route" if mode == "threshold_greedy" else "missing")
            )
            common = {
                "dataset": dataset,
                "method": method,
                "group_key_mode": group_key_mode,
                "seed": seed,
                "round": round_index,
                "mode": mode,
                "similarity_mode": similarity_mode,
                "source_file": source_relative,
                "source_sha256": source_sha,
            }
            cosine_rows.append(
                {
                    **common,
                    "scope": "selected_correction_candidates",
                    "evidence_status": evidence_status,
                    **summarize_similarity_values(selected_similarity),
                }
            )
            if mode == "diversity_aware":
                cosine_rows.append(
                    {
                        **common,
                        "scope": "all_considered_correction_candidates",
                        "evidence_status": "not_archived_full_similarity_matrix",
                        **summarize_similarity_values([]),
                    }
                )
                aggregate_values[(dataset, method, group_key_mode, similarity_mode)].extend(
                    selected_similarity.tolist()
                )

            legacy_p = np.argsort(-p_hit).tolist()
            explicit_p = _explicit_order(ids, p_hit)
            legacy_q = np.argsort(-quality).tolist()
            explicit_q = _explicit_order(ids, quality)
            p_ties = _tie_summary(p_hit)
            q_ties = _tie_summary(quality)
            selected_scores = pd.to_numeric(score["selection_score"], errors="coerce").dropna().to_numpy(float)
            selected_ties = _tie_summary(selected_scores)
            p_boundary_count, p_boundary_gap = _boundary_tie(p_hit, batch_size)
            q_boundary_count, q_boundary_gap = _boundary_tie(quality, prefilter_size)
            selected_ids = set(score.loc[pd.to_numeric(score["selected"], errors="raise").eq(1), "id"].astype(str))
            direct_explicit_membership_matches = (
                selected_ids == {ids[index] for index in explicit_p[:batch_size]}
                if mode == "threshold_greedy"
                else None
            )
            group_counts = score["group_key"].astype(str).value_counts()
            tie_rows.append(
                {
                    **common,
                    "candidate_count": len(score),
                    "candidate_ids_unique": len(set(ids)) == len(ids),
                    "p_hit_exact_tie_groups": p_ties["groups"],
                    "p_hit_tied_candidates": p_ties["candidates"],
                    "p_hit_largest_tie_group": p_ties["largest_group"],
                    "p_hit_min_positive_gap": _minimum_positive_gap(p_hit),
                    "quality_exact_tie_groups": q_ties["groups"],
                    "quality_tied_candidates": q_ties["candidates"],
                    "quality_largest_tie_group": q_ties["largest_group"],
                    "quality_min_positive_gap": _minimum_positive_gap(quality),
                    "top_b_boundary_tied_candidates": p_boundary_count,
                    "top_b_boundary_gap": p_boundary_gap,
                    "top_b_membership_changed_by_explicit_rule": (
                        set(legacy_p[:batch_size]) != set(explicit_p[:batch_size])
                    ),
                    "top_b_order_changed_by_explicit_rule": legacy_p[:batch_size] != explicit_p[:batch_size],
                    "prefilter_boundary_tied_candidates": q_boundary_count,
                    "prefilter_boundary_gap": q_boundary_gap,
                    "prefilter_membership_changed_by_explicit_rule": (
                        set(legacy_q[:prefilter_size]) != set(explicit_q[:prefilter_size])
                    ),
                    "prefilter_order_changed_by_explicit_rule": legacy_q[:prefilter_size] != explicit_q[:prefilter_size],
                    "selected_score_exact_tie_groups": selected_ties["groups"],
                    "selected_score_tied_candidates": selected_ties["candidates"],
                    "selected_score_min_positive_gap": _minimum_positive_gap(selected_scores),
                    "group_key_group_count": int(group_counts.size),
                    "group_key_repeated_candidate_count": int(group_counts[group_counts > 1].sum()),
                    "dynamic_competitor_scores_archived": False,
                    "direct_selected_membership_matches_explicit_rule": direct_explicit_membership_matches,
                }
            )

    for (dataset, method, group_key_mode, similarity_mode), values in sorted(aggregate_values.items()):
        cosine_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "group_key_mode": group_key_mode,
                "seed": "ALL",
                "round": "ALL",
                "mode": "diversity_aware",
                "similarity_mode": similarity_mode,
                "scope": "selected_correction_candidates_aggregate",
                "evidence_status": "available_selected_candidates",
                "source_file": "multiple_formal_score_files",
                "source_sha256": "see_per_round_rows",
                **summarize_similarity_values(values),
            }
        )
    return pd.DataFrame(cosine_rows), pd.DataFrame(tie_rows)


def _write_without_overwriting_different_content(frame: pd.DataFrame, output: Path) -> None:
    rendered = frame.to_csv(index=False, lineterminator="\n")
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to overwrite non-identical evidence: {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8", newline="")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=root / "artifacts/gpu_server/completed_formal_results/63729a5a4bea44b3/attempt_1/payload/results/final",
    )
    parser.add_argument(
        "--cosine-output",
        type=Path,
        default=root / "results/audit/cosine_similarity_distribution.csv",
    )
    parser.add_argument(
        "--tie-output", type=Path, default=root / "results/audit/tie_inventory.csv"
    )
    args = parser.parse_args()
    cosine, ties = audit_formal_results(args.formal_root)
    _write_without_overwriting_different_content(cosine, args.cosine_output)
    _write_without_overwriting_different_content(ties, args.tie_output)
    print(
        json.dumps(
            {
                "cosine_rows": len(cosine),
                "tie_rows": len(ties),
                "negative_selected_similarity_count": int(cosine["negative_count"].sum()),
                "top_b_membership_changes": int(ties["top_b_membership_changed_by_explicit_rule"].sum()),
                "prefilter_membership_changes": int(ties["prefilter_membership_changed_by_explicit_rule"].sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
