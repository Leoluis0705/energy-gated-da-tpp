#!/usr/bin/env python3
"""Parameterized Energy-Gated DA-TPP ablation selector.

This selector keeps the same information boundary as active_learning_gated_ta_dpp:
only pseudo predictions, model-derived uncertainty, candidate embeddings and
composition/group keys are used for acquisition. Oracle labels are appended only
after selection by the outer active-learning runner.
"""
from __future__ import annotations

import argparse
import json
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
from experiments.reproducibility.protocol_artifacts import (
    PROTOCOL_VERSION,
    SUPPORTED_PROTOCOL_VERSIONS,
    score_artifact_path,
    trace_artifact_path,
    write_dataframe_exclusive,
)
from mc_dropout_protocol import extract_seeded_mc_dropout, prepare_selector_uncertainty
from experiments.reproducibility.formal_protocol import resolve_group_keys_from_map


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


def apply_quality_safeguard(direct_idx, proposed_idx, p_hit, min_fraction):
    """Fall back to the direct batch when a proposal loses too much P_hit."""
    fraction = float(min_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("quality safeguard fraction must be finite and in (0, 1]")
    direct = [int(value) for value in direct_idx]
    proposed = [int(value) for value in proposed_idx]
    values = np.asarray(p_hit, dtype=float)
    direct_sum = float(values[direct].sum())
    proposed_sum = float(values[proposed].sum())
    accepted = proposed_sum + EPS >= fraction * direct_sum
    return (proposed if accepted else direct), (not accepted), direct_sum, proposed_sum


def _stable_descending(values, ids):
    values = np.asarray(values, dtype=float)
    labels = np.asarray([str(value) for value in ids], dtype=object)
    return [int(index) for index in np.lexsort((labels, -values))]


def _selection_seed(experiment_seed, iteration):
    payload = f"mn_mg_interval_gpu_v1:{int(experiment_seed)}:{int(iteration)}".encode("utf-8")
    import hashlib

    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _kmeans_plus_plus(frontier, feature_matrix, count, rng):
    frontier = [int(index) for index in frontier]
    if count <= 0 or not frontier:
        return []
    if count >= len(frontier):
        return frontier
    matrix = np.asarray(feature_matrix, dtype=float)[frontier]
    selected_local = [int(rng.integers(len(frontier)))]
    while len(selected_local) < count:
        centers = matrix[selected_local]
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2).min(axis=1)
        distances[selected_local] = 0.0
        total = float(distances.sum())
        remaining = [index for index in range(len(frontier)) if index not in selected_local]
        if not math.isfinite(total) or total <= EPS:
            selected_local.append(int(rng.choice(remaining)))
        else:
            selected_local.append(int(rng.choice(len(frontier), p=distances / total)))
    return [frontier[index] for index in selected_local]


def select_baseline_indices(
    mode,
    ids,
    p_hit,
    uncertainty,
    feature_matrix,
    gradient_proxy,
    *,
    batch_size,
    experiment_seed,
    iteration,
):
    """Select corrected historical baselines without access to oracle labels."""
    count = min(int(batch_size), len(ids))
    quality_order = _stable_descending(p_hit, ids)
    rng = np.random.default_rng(_selection_seed(experiment_seed, iteration))
    if mode == "mc_dropout":
        selected = _stable_descending(uncertainty, ids)[:count]
        return selected, {ids[index]: float(uncertainty[index]) for index in selected}, "mc_dropout"
    if mode == "random_sampling":
        selected = [int(index) for index in rng.choice(len(ids), size=count, replace=False)]
        return selected, {ids[index]: 0.0 for index in selected}, "random_sampling"
    if mode == "gradient_norm_hybrid":
        gradient_count = count // 2
        gradient_order = _stable_descending(gradient_proxy, ids)
        selected = gradient_order[:gradient_count]
        selected.extend(index for index in quality_order if index not in selected)
        selected = selected[:count]
        scores = {ids[index]: float(gradient_proxy[index]) for index in selected[:gradient_count]}
        scores.update({ids[index]: float(p_hit[index]) for index in selected[gradient_count:]})
        return selected, scores, "gradient_norm_hybrid"
    if mode == "explore":
        explore_count = count // 2
        frontier_size = min(len(ids), max(count * 3, int(math.ceil(0.20 * len(ids)))))
        frontier = quality_order[:frontier_size]
        selected = _kmeans_plus_plus(frontier, feature_matrix, explore_count, rng)
        selected.extend(index for index in quality_order if index not in selected)
        selected = selected[:count]
        return selected, {ids[index]: float(p_hit[index]) for index in selected}, "explore_kmeanspp_plus_greedy"
    raise ValueError(f"unsupported standalone baseline mode: {mode}")


def gate_mode(ablation_mode, margin, concentration, margin_threshold, concentration_threshold):
    if ablation_mode == "p_hit_greedy":
        return "threshold_greedy"
    if ablation_mode == "always_diversity":
        return "diversity_aware"
    if ablation_mode == "gate_no_margin":
        return "threshold_greedy" if concentration <= concentration_threshold else "diversity_aware"
    if ablation_mode == "gate_no_concentration":
        return "threshold_greedy" if margin >= margin_threshold else "diversity_aware"
    return "threshold_greedy" if (margin >= margin_threshold and concentration <= concentration_threshold) else "diversity_aware"


def build_parser():
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
    parser.add_argument("--experiment-seed", type=int, required=True)
    parser.add_argument("--model-refit-index", type=int, required=True)
    parser.add_argument("--mc-passes", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.30)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--score-log-dir", default=None)
    parser.add_argument("--group-key-map", default=None)
    parser.add_argument("--selection-method-name", default="energy_gate_full")
    parser.add_argument(
        "--ablation-mode",
        choices=[
            "full",
            "p_hit_greedy",
            "always_diversity",
            "gate_no_margin",
            "gate_no_concentration",
            "explore",
            "mc_dropout",
            "gradient_norm_hybrid",
            "random_sampling",
        ],
        default="full",
    )
    parser.add_argument("--margin-threshold", type=float, default=1.0)
    parser.add_argument("--concentration-threshold", type=float, default=0.50)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.10)
    parser.add_argument("--quality-safeguard-fraction", type=float, default=None)
    parser.add_argument(
        "--protocol-version",
        choices=sorted(SUPPORTED_PROTOCOL_VERSIONS),
        default=PROTOCOL_VERSION,
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    pred = pd.read_csv(args.test_results, header=None, names=["id", "actual_feature", "pseudo_feature"])
    pred["id"] = pred["id"].map(clean_id)
    pred["pseudo_feature"] = pd.to_numeric(pred["pseudo_feature"], errors="coerce")
    pred = pred.dropna(subset=["pseudo_feature"]).copy()
    if pred.empty:
        raise ValueError("No valid pseudo predictions found")

    ids = pred["id"].tolist()
    pseudo = pred["pseudo_feature"].to_numpy(dtype=float)
    groups = (
        resolve_group_keys_from_map(ids, args.group_key_map)
        if args.group_key_map
        else [group_key(sample_id, args.pseudo_dir, args.original_dir) for sample_id in ids]
    )
    mc_state = extract_seeded_mc_dropout(
        model_path=args.model,
        candidates_dir=args.pseudo_dir,
        mc_passes=args.mc_passes,
        dropout_rate=args.dropout_rate,
        experiment_seed=args.experiment_seed,
        acquisition_round=args.iteration,
        model_refit_index=args.model_refit_index,
    )
    features = mc_state.features
    warnings = list(mc_state.warnings)
    for warning in warnings[:30]:
        print(f"WARNING: {warning}")

    sigma_normalized = np.asarray(
        [mc_state.sigma_normalized.get(sample_id, 0.0) for sample_id in ids],
        dtype=float,
    )
    corrected = prepare_selector_uncertainty(
        deterministic_mean_ev=pseudo,
        mc_sigma_normalized=sigma_normalized,
        normalizer_location=mc_state.normalizer_location,
        normalizer_scale=mc_state.normalizer_scale,
        target_low_ev=float(args.target_low),
        target_high_ev=float(args.target_high),
        sigma_floor_ev=float(args.sigma_min),
    )
    sigma_ev = corrected.sigma_ev
    uncertainty = normalize01(sigma_ev)
    p_hit = corrected.interval_hit_probability
    p_hit_mode = "mc_dropout_interval_cdf_ev_psfix_v1"

    similarity, similarity_mode = choose_similarity(ids, features, groups)
    if features and all(sample_id in features for sample_id in ids):
        raw_feature_matrix = np.vstack([features[sample_id] for sample_id in ids]).astype(float)
    else:
        raw_feature_matrix = composition_features(groups)
    embedding_norm = np.linalg.norm(raw_feature_matrix, axis=1)
    gradient_proxy = p_hit * (1.0 - p_hit) * np.maximum(embedding_norm, EPS)
    order = np.argsort(-p_hit)
    batch_size = min(args.selection_size, len(ids))
    top_b = order[:batch_size]
    next_b = order[batch_size: min(2 * batch_size, len(ids))]
    top_2b = order[: min(2 * batch_size, len(ids))]
    margin = float((np.mean(p_hit[top_b]) - np.mean(p_hit[next_b])) / (np.std(p_hit[top_2b]) + 1e-8)) if len(next_b) else float("inf")
    top_groups = Counter(groups[int(idx)] for idx in top_b)
    concentration = max(top_groups.values(), default=0) / max(1, batch_size)
    standalone_modes = {"explore", "mc_dropout", "gradient_norm_hybrid", "random_sampling"}
    mode = (
        args.ablation_mode
        if args.ablation_mode in standalone_modes
        else gate_mode(args.ablation_mode, margin, concentration, args.margin_threshold, args.concentration_threshold)
    )

    selected_max_sim = {}
    selected_group_penalty = {}
    quality_fallback = False
    direct_p_hit_sum = float(np.sum(p_hit[top_b]))
    proposed_p_hit_sum = direct_p_hit_sum
    if args.ablation_mode in standalone_modes:
        selected_idx, selection_scores, mode = select_baseline_indices(
            args.ablation_mode,
            ids,
            p_hit,
            uncertainty,
            raw_feature_matrix,
            gradient_proxy,
            batch_size=batch_size,
            experiment_seed=args.experiment_seed,
            iteration=args.iteration,
        )
        prefilter_size = min(len(ids), max(batch_size * 3, int(math.ceil(0.20 * len(ids)))))
    elif mode == "threshold_greedy":
        selected_idx = [int(idx) for idx in top_b]
        prefilter_size = batch_size
        selection_scores = {ids[idx]: float(p_hit[idx]) for idx in selected_idx}
    else:
        multiplier = {16: 12, 32: 10, 64: 8}.get(int(args.selection_size), 10)
        selected_idx, selection_scores, selected_max_sim, selected_group_penalty, prefilter_size = diversity_select(
            ids, p_hit, uncertainty, groups, similarity, batch_size, multiplier,
            args.alpha, args.beta, args.gamma
        )
        proposed_idx = list(selected_idx)
        proposed_p_hit_sum = float(np.sum(p_hit[proposed_idx]))
        if args.quality_safeguard_fraction is not None:
            selected_idx, quality_fallback, direct_p_hit_sum, proposed_p_hit_sum = apply_quality_safeguard(
                [int(idx) for idx in top_b],
                proposed_idx,
                p_hit,
                args.quality_safeguard_fraction,
            )
            if quality_fallback:
                selection_scores = {ids[idx]: float(p_hit[idx]) for idx in selected_idx}
                selected_max_sim = {}
                selected_group_penalty = {}

    selected_ids = [ids[idx] for idx in selected_idx]
    selected_set = set(selected_ids)
    score_dir = Path(args.score_log_dir or args.output_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_df = pred.copy()
    score_df["P_hit"] = p_hit
    score_df["U_i"] = uncertainty
    score_df["mu_eV"] = corrected.mean_ev
    score_df["sigma_i"] = sigma_ev
    score_df["sigma_normalized"] = sigma_normalized
    score_df["sigma_eV"] = sigma_ev
    score_df["mc_mean_normalized"] = [
        mc_state.mean_normalized.get(sample_id, float("nan")) for sample_id in ids
    ]
    score_df["mc_mean_eV"] = [
        mc_state.mean_ev.get(sample_id, float("nan")) for sample_id in ids
    ]
    score_df["mean_input_space"] = corrected.mean_input_space
    score_df["sigma_input_space"] = corrected.sigma_input_space
    score_df["normalizer_location"] = mc_state.normalizer_location
    score_df["normalizer_scale"] = mc_state.normalizer_scale
    score_df["mc_mask_sequence_sha256"] = mc_state.mask_sequence_sha256
    score_df["mc_seed_policy_version"] = mc_state.seed_policy_version
    score_df["protocol_version"] = args.protocol_version
    score_df["group_key"] = groups
    score_df["mode"] = mode
    score_df["ablation_mode"] = args.ablation_mode
    score_df["similarity_mode"] = similarity_mode
    score_df["selection_score"] = score_df["id"].map(selection_scores)
    score_df["selected_max_similarity"] = score_df["id"].map(selected_max_sim)
    score_df["selected_group_penalty"] = score_df["id"].map(selected_group_penalty)
    score_df["selected"] = score_df["id"].isin(selected_set).astype(int)
    score_path = score_artifact_path(
        score_dir,
        method_name=args.selection_method_name,
        iteration=args.iteration,
        protocol_version=args.protocol_version,
    )
    write_dataframe_exclusive(score_df, score_path)

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
            "protocol_version": args.protocol_version,
            "experiment_seed": args.experiment_seed,
            "model_refit_index": args.model_refit_index,
            "mc_passes": args.mc_passes,
            "mc_mask_seeds_json": json.dumps(list(mc_state.mask_seeds), separators=(",", ":")),
            "mc_mask_sequence_sha256": mc_state.mask_sequence_sha256,
            "mc_seed_policy_version": mc_state.seed_policy_version,
            "normalizer_location": mc_state.normalizer_location,
            "normalizer_scale": mc_state.normalizer_scale,
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
        "protocol_version": args.protocol_version,
        "experiment_seed": args.experiment_seed,
        "model_refit_index": args.model_refit_index,
        "mc_passes": args.mc_passes,
        "mc_mask_seeds_json": json.dumps(list(mc_state.mask_seeds), separators=(",", ":")),
        "mc_mask_sequence_sha256": mc_state.mask_sequence_sha256,
        "mc_seed_policy_version": mc_state.seed_policy_version,
        "normalizer_location": mc_state.normalizer_location,
        "normalizer_scale": mc_state.normalizer_scale,
        "mean_unit": "eV atom^-1",
        "sigma_unit": "eV atom^-1",
        "prefilter_size": prefilter_size,
        "mean_selected_P_hit": float(np.mean(p_hit[selected_idx])) if selected_idx else 0.0,
        "mean_selected_uncertainty": float(np.mean(uncertainty[selected_idx])) if selected_idx else 0.0,
        "mean_pool_P_hit": float(np.mean(p_hit)),
        "selected_group_key_count": len(selected_groups),
        "top_group_ratio_in_selected_batch": max(selected_groups.values(), default=0) / max(1, len(selected_idx)),
        "p_hit_mode": p_hit_mode,
        "similarity_mode": similarity_mode,
        "quality_safeguard_fraction": args.quality_safeguard_fraction,
        "quality_safeguard_fallback": int(quality_fallback),
        "direct_batch_p_hit_sum": direct_p_hit_sum,
        "proposed_batch_p_hit_sum": proposed_p_hit_sum,
        "selected_batch_p_hit_sum": float(np.sum(p_hit[selected_idx])) if selected_idx else 0.0,
        **stats,
    }])
    trace_path = trace_artifact_path(score_dir, protocol_version=args.protocol_version)
    if trace_path.exists():
        existing_trace = pd.read_csv(trace_path)
        if "iteration" in existing_trace.columns and int(args.iteration) in set(
            pd.to_numeric(existing_trace["iteration"], errors="raise").astype(int)
        ):
            raise FileExistsError(f"trace already contains iteration {args.iteration}: {trace_path}")
        trace.to_csv(trace_path, mode="a", header=False, index=False, lineterminator="\n")
    else:
        write_dataframe_exclusive(trace, trace_path)
    print(trace.to_string(index=False))


if __name__ == "__main__":
    main()
