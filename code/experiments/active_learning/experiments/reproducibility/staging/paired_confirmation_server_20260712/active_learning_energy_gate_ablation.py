#!/usr/bin/env python3
"""Parameterized Energy-Gated DA-TPP ablation selector.

This selector keeps the same information boundary as active_learning_gated_ta_dpp:
only pseudo predictions, model-derived uncertainty, candidate embeddings and
composition/group keys are used for acquisition. Oracle labels are appended only
after selection by the outer active-learning runner.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from active_learning_etdg_tage import (
    EPS,
    clean_id,
    extract_features_gradients_dropout,
    group_key,
    normalize01,
)


def normal_cdf(values):
    values = np.asarray(values, dtype=float)
    return np.asarray([0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0))) for x in values])


def composition_features(group_keys):
    elements = sorted({token for key in group_keys for token in str(key).split("-") if token and token != "UNKNOWN"})
    index = {element: idx for idx, element in enumerate(elements)}
    matrix = np.zeros((len(group_keys), max(1, len(elements))), dtype=float)
    for row, key in enumerate(group_keys):
        for token in str(key).split("-"):
            if token in index:
                matrix[row, index[token]] = 1.0
    return matrix


def normalize_rows(matrix):
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, EPS)


def choose_similarity(ids, features, groups):
    if features and all(sample_id in features for sample_id in ids):
        matrix = np.vstack([features[sample_id] for sample_id in ids])
        return normalize_rows(matrix), "cgcnn_embedding"
    return normalize_rows(composition_features(groups)), "composition"


def pairwise_stats(selected_idx, similarity, groups):
    if len(selected_idx) < 2:
        avg_sim = 0.0
        max_sim = 0.0
    else:
        vals = []
        for i, a in enumerate(selected_idx):
            for b in selected_idx[i + 1:]:
                vals.append(float(np.dot(similarity[a], similarity[b])))
        avg_sim = float(np.mean(vals)) if vals else 0.0
        max_sim = float(np.max(vals)) if vals else 0.0
    counts = Counter(groups[idx] for idx in selected_idx)
    n = max(1, len(selected_idx))
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log(p + 1e-12)
    return {
        "selected_batch_avg_pairwise_similarity": avg_sim,
        "selected_batch_max_pairwise_similarity": max_sim,
        "selected_batch_unique_group_count": len(counts),
        "selected_batch_largest_group_fraction": max(counts.values(), default=0) / n,
        "selected_batch_group_entropy": entropy,
    }


def diversity_select(ids, p_hit, uncertainty, groups, similarity, batch_size, prefilter_multiplier, alpha, beta, gamma):
    prefilter_size = min(len(ids), int(prefilter_multiplier * batch_size))
    quality = p_hit + alpha * uncertainty
    prefilter_idx = np.argsort(-quality)[:prefilter_size]
    selected = []
    trace_scores = {}
    selected_group_counts = Counter()
    selected_max_sim = {}
    selected_group_penalty = {}
    while len(selected) < min(batch_size, prefilter_size):
        best_idx = None
        best_score = -float("inf")
        best_sim = 0.0
        best_group_penalty = 0.0
        for idx in prefilter_idx:
            idx = int(idx)
            if idx in selected:
                continue
            max_sim = max((float(np.dot(similarity[idx], similarity[chosen])) for chosen in selected), default=0.0)
            group_penalty = selected_group_counts[groups[idx]] / max(1, len(selected))
            score = float(p_hit[idx] + alpha * uncertainty[idx] - beta * max_sim - gamma * group_penalty)
            if score > best_score:
                best_idx = idx
                best_score = score
                best_sim = max_sim
                best_group_penalty = group_penalty
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_group_counts[groups[best_idx]] += 1
        trace_scores[ids[best_idx]] = best_score
        selected_max_sim[ids[best_idx]] = best_sim
        selected_group_penalty[ids[best_idx]] = best_group_penalty
    return selected, trace_scores, selected_max_sim, selected_group_penalty, prefilter_size


def gate_mode(ablation_mode, margin, concentration, margin_threshold, concentration_threshold):
    if ablation_mode == "p_hit_greedy":
        return "threshold_greedy"
    if ablation_mode == "always_diversity":
        return "diversity_aware"
    if ablation_mode in ("gate_no_margin", "concentration_only_gate"):
        return "threshold_greedy" if concentration <= concentration_threshold else "diversity_aware"
    if ablation_mode in ("gate_no_concentration", "margin_only_gate"):
        return "threshold_greedy" if margin >= margin_threshold else "diversity_aware"
    return "threshold_greedy" if (margin >= margin_threshold and concentration <= concentration_threshold) else "diversity_aware"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--test-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-feature", type=float, required=True)
    parser.add_argument("--target-low", type=float, required=True)
    parser.add_argument("--target-high", type=float, required=True)
    parser.add_argument("--selection-size", type=int, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mc-passes", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.30)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--score-log-dir", default=None)
    parser.add_argument("--selection-method-name", default="energy_gate_full")
    parser.add_argument(
        "--ablation-mode",
        choices=[
            "full",
            "p_hit_greedy",
            "always_diversity",
            "gate_no_margin",
            "gate_no_concentration",
            "margin_only_gate",
            "concentration_only_gate",
        ],
        default="full",
    )
    parser.add_argument("--margin-threshold", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=0.50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.10)
    args = parser.parse_args()

    pred = pd.read_csv(args.test_results, header=None, names=["id", "actual_feature", "pseudo_feature"])
    pred["id"] = pred["id"].map(clean_id)
    pred["pseudo_feature"] = pd.to_numeric(pred["pseudo_feature"], errors="coerce")
    pred = pred.dropna(subset=["pseudo_feature"]).copy()
    if pred.empty:
        raise ValueError("No valid pseudo predictions found")

    ids = pred["id"].tolist()
    pseudo = pred["pseudo_feature"].to_numpy(dtype=float)
    groups = [group_key(sample_id, args.pseudo_dir, args.original_dir) for sample_id in ids]
    features, _, dropout_vars, warnings = extract_features_gradients_dropout(
        args.model, args.pseudo_dir, args.mc_passes, args.dropout_rate, args.target_feature
    )
    for warning in warnings[:30]:
        print(f"WARNING: {warning}")

    raw_sigma = np.sqrt(np.maximum([dropout_vars.get(sample_id, 0.0) for sample_id in ids], 0.0))
    uncertainty = normalize01(raw_sigma)
    if np.max(raw_sigma) > EPS:
        sigma = np.maximum(raw_sigma, float(args.sigma_min))
        low = float(args.target_low)
        high = float(args.target_high)
        p_hit = normal_cdf((high - pseudo) / sigma) - normal_cdf((low - pseudo) / sigma)
        p_hit_mode = "mc_dropout_interval_cdf"
    else:
        p_hit = normalize01(-np.abs(pseudo - args.target_feature))
        p_hit_mode = "rank_normalized_interval_center_fallback"
        print("WARNING: reliable MC-dropout uncertainty unavailable; using center-distance fallback.")
    p_hit = np.clip(p_hit, 0.0, 1.0)

    similarity, similarity_mode = choose_similarity(ids, features, groups)
    order = np.argsort(-p_hit)
    batch_size = min(args.selection_size, len(ids))
    top_b = order[:batch_size]
    next_b = order[batch_size: min(2 * batch_size, len(ids))]
    top_2b = order[: min(2 * batch_size, len(ids))]
    margin = float((np.mean(p_hit[top_b]) - np.mean(p_hit[next_b])) / (np.std(p_hit[top_2b]) + 1e-8)) if len(next_b) else float("inf")
    top_groups = Counter(groups[int(idx)] for idx in top_b)
    concentration = max(top_groups.values(), default=0) / max(1, batch_size)
    mode = gate_mode(args.ablation_mode, margin, concentration, args.margin_threshold, args.concentration_threshold)

    selected_max_sim = {}
    selected_group_penalty = {}
    if mode == "threshold_greedy":
        selected_idx = [int(idx) for idx in top_b]
        prefilter_size = batch_size
        selection_scores = {ids[idx]: float(p_hit[idx]) for idx in selected_idx}
    else:
        multiplier = {16: 12, 32: 10, 64: 8}.get(int(args.selection_size), 10)
        selected_idx, selection_scores, selected_max_sim, selected_group_penalty, prefilter_size = diversity_select(
            ids, p_hit, uncertainty, groups, similarity, batch_size, multiplier,
            args.alpha, args.beta, args.gamma
        )

    selected_ids = [ids[idx] for idx in selected_idx]
    selected_set = set(selected_ids)
    score_dir = Path(args.score_log_dir or args.output_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_df = pred.copy()
    score_df["P_hit"] = p_hit
    score_df["U_i"] = uncertainty
    score_df["sigma_i"] = raw_sigma
    score_df["group_key"] = groups
    score_df["mode"] = mode
    score_df["ablation_mode"] = args.ablation_mode
    score_df["similarity_mode"] = similarity_mode
    score_df["selection_score"] = score_df["id"].map(selection_scores)
    score_df["selected_max_similarity"] = score_df["id"].map(selected_max_sim)
    score_df["selected_group_penalty"] = score_df["id"].map(selected_group_penalty)
    score_df["selected"] = score_df["id"].isin(selected_set).astype(int)
    score_df.to_csv(score_dir / f"{args.selection_method_name}_scores_iter_{args.iteration}.csv", index=False)

    original_df = pd.read_csv(Path(args.original_dir) / "id_prop.csv", header=None, names=["id", "feature"])
    original_df["id"] = original_df["id"].map(clean_id)
    selected_df = original_df[original_df["id"].isin(selected_set)].copy()
    remaining_df = original_df[~original_df["id"].isin(selected_set)].copy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_dir / "id_prop.csv", mode="a", header=False, index=False)
    for sample_id in selected_df["id"]:
        src = Path(args.original_dir) / f"{sample_id}.cif"
        if src.exists():
            shutil.copy(src, output_dir / src.name)
    remaining_df.to_csv(Path(args.original_dir) / "id_prop.csv", header=False, index=False)

    history_path = Path(os.getenv("ACTIVE_HISTORY", "active_learning_history.csv"))
    by_id = pred.set_index("id")
    hist = []
    for idx in selected_idx:
        sample_id = ids[idx]
        row = by_id.loc[sample_id]
        hist.append({
            "id": sample_id,
            "iteration": args.iteration,
            "target_feature": args.target_feature,
            "pseudo_feature": row["pseudo_feature"],
            "actual_feature": row["actual_feature"],
            "distance": row["pseudo_feature"] - args.target_feature,
            "selection_method": args.selection_method_name,
            "P_hit": p_hit[idx],
            "U_i": uncertainty[idx],
            "group_key": groups[idx],
            "mode": mode,
            "ablation_mode": args.ablation_mode,
        })
    pd.DataFrame(hist).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)

    selected_groups = Counter(groups[idx] for idx in selected_idx)
    stats = pairwise_stats(selected_idx, similarity, groups)
    trace = pd.DataFrame([{
        "iteration": args.iteration,
        "batch_size": args.selection_size,
        "selection_method": args.selection_method_name,
        "ablation_mode": args.ablation_mode,
        "mode": mode,
        "margin_score": margin,
        "group_concentration": concentration,
        "margin_threshold": args.margin_threshold,
        "concentration_threshold": args.concentration_threshold,
        "alpha": args.alpha,
        "beta": args.beta,
        "gamma": args.gamma,
        "prefilter_size": prefilter_size,
        "mean_selected_P_hit": float(np.mean(p_hit[selected_idx])) if selected_idx else 0.0,
        "mean_selected_uncertainty": float(np.mean(uncertainty[selected_idx])) if selected_idx else 0.0,
        "mean_pool_P_hit": float(np.mean(p_hit)),
        "selected_group_key_count": len(selected_groups),
        "top_group_ratio_in_selected_batch": max(selected_groups.values(), default=0) / max(1, len(selected_idx)),
        "p_hit_mode": p_hit_mode,
        "similarity_mode": similarity_mode,
        **stats,
    }])
    trace_path = score_dir / "mode_trace.csv"
    trace.to_csv(trace_path, mode="a", header=not trace_path.exists(), index=False)
    print(trace.to_string(index=False))


if __name__ == "__main__":
    main()
