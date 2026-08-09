#!/usr/bin/env python3
"""
ETDG-TAGE: Expected Target Discovery Gain based Target-Aware Gradient Exploration.

Quick one-round validation after a prediction file exists:

python active_learning_etdg_tage.py --model D:\\CGCNN\\path\\current.pth.tar --original-dir D:\\CGCNN\\path\\cifs --pseudo-dir D:\\CGCNN\\path\\pseudo_dir --test-results D:\\CGCNN\\path\\test_results.csv --output-dir D:\\CGCNN\\path\\train_dir --target-feature -2.0 --selection-size 64 --iteration 1

Runner-level 1-3 iteration validation:

python scripts\\run_formation_active_learning.py --pool-dir <POOL> --oracle-csv <ORACLE.csv> --checkpoint <CLEAN_CHECKPOINT.pth.tar> --output-root <OUT> --batch-sizes 64 --strategies etdg_tage --max-iterations 3 --initialization-mode pretrained --target-mode threshold --formation-energy-threshold -2.0

Selection uses pseudo predictions, CGCNN features, target-anchor gradient sensitivity,
MC-dropout variance, labeled history, and candidate structure chemistry.
The test_results actual column is never used to compute ETDG scores.
"""

import argparse
import math
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.append(str(PROJECT_ROOT.parent))

from cgcnn.data import CIFData, collate_pool
from cgcnn.model import CrystalGraphConvNet

EPS = 1e-12


def clean_id(value):
    return str(value).split(".cif")[0].strip()


def normalize01(values):
    arr = np.asarray(values, dtype=float)
    out = np.zeros_like(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = np.nanmin(arr[finite])
    hi = np.nanmax(arr[finite])
    if hi - lo <= EPS:
        return out
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def sigmoid(values):
    values = np.clip(values, -60, 60)
    return 1.0 / (1.0 + np.exp(-values))


def parse_elements_from_cif(cif_path):
    try:
        text = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ["UNKNOWN"]

    elements = set()
    for pattern in (r"_chemical_formula_sum\s+['\"]?([^'\"]+)['\"]?", r"_chemical_formula_structural\s+['\"]?([^'\"]+)['\"]?"):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            elements.update(re.findall(r"\b([A-Z][a-z]?)\s*[0-9.]*", match.group(1)))

    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "_atom_site_type_symbol" not in line:
            continue
        header = []
        j = idx
        while j < len(lines) and lines[j].strip().startswith("_atom_site"):
            header.append(lines[j].strip())
            j += 1
        try:
            type_idx = header.index("_atom_site_type_symbol")
        except ValueError:
            type_idx = None
        while j < len(lines):
            row = lines[j].strip()
            if not row or row.startswith("_") or row.lower().startswith("loop_") or row.startswith("#"):
                break
            parts = row.split()
            token = None
            if type_idx is not None and type_idx < len(parts):
                token = parts[type_idx]
            elif parts:
                token = parts[0]
            if token:
                token = re.sub(r"[^A-Za-z]", "", token)
                match = re.match(r"^([A-Z][a-z]?)", token)
                if match:
                    elements.add(match.group(1))
            j += 1

    elements = sorted(e for e in elements if e and e != "D")
    return elements or ["UNKNOWN"]


def group_key(sample_id, *dirs):
    for directory in dirs:
        if not directory:
            continue
        cif_path = Path(directory) / f"{clean_id(sample_id)}.cif"
        if cif_path.exists():
            return "-".join(parse_elements_from_cif(cif_path))
    return "UNKNOWN"


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    if hasattr(model_args, "__dict__"):
        args_dict = vars(model_args)
    elif isinstance(model_args, dict):
        args_dict = model_args
    else:
        args_dict = {}

    model = CrystalGraphConvNet(
        orig_atom_fea_len=92,
        nbr_fea_len=41,
        atom_fea_len=args_dict.get("atom_fea_len", 64),
        h_fea_len=args_dict.get("h_fea_len", 128),
        n_conv=args_dict.get("n_conv", 3),
        n_h=args_dict.get("n_h", 1),
        classification=False,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def penultimate_forward(model, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
    atom_fea = model.embedding(atom_fea)
    for conv in model.convs:
        atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)
    crys_fea = model.pooling(atom_fea, crystal_atom_idx)
    crys_fea = model.conv_to_fc(model.conv_to_fc_softplus(crys_fea))
    crys_fea = model.conv_to_fc_softplus(crys_fea)
    if hasattr(model, "fcs") and hasattr(model, "softpluses"):
        for fc, softplus in zip(model.fcs, model.softpluses):
            crys_fea = softplus(fc(crys_fea))
    return crys_fea


def extract_features_gradients_dropout(model_path, candidates_dir, mc_passes, dropout_rate, target_anchor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    dataset = CIFData(candidates_dir)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_pool, shuffle=False)

    features = {}
    grad_norms = {}
    dropout_vars = {}
    warnings = []

    for batch in dataloader:
        (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), _, batch_ids = batch
        batch_ids = [clean_id(x) for x in batch_ids]
        atom_fea = atom_fea.to(device)
        nbr_fea = nbr_fea.to(device)
        nbr_fea_idx = nbr_fea_idx.to(device)
        crystal_atom_idx = [idx.to(device) for idx in crystal_atom_idx]

        try:
            fc_input = penultimate_forward(model, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
            fc_input = fc_input.detach().requires_grad_(True)
            output = model.fc_out(fc_input).view(-1)

            for idx, sample_id in enumerate(batch_ids):
                features[sample_id] = fc_input[idx].detach().cpu().numpy().astype(float)
                try:
                    loss_i = (output[idx] - float(target_anchor)) ** 2
                    grad_full = torch.autograd.grad(loss_i, fc_input, retain_graph=True, allow_unused=True)[0]
                    if grad_full is None:
                        grad_norms[sample_id] = 0.0
                        warnings.append(f"gradient is None for {sample_id}")
                    else:
                        grad_norms[sample_id] = float(torch.linalg.norm(grad_full[idx]).detach().cpu().item())
                except Exception as exc:
                    grad_norms[sample_id] = 0.0
                    warnings.append(f"gradient failed for {sample_id}: {exc}")

            try:
                preds = []
                keep_prob = max(EPS, 1.0 - float(dropout_rate))
                passes = max(1, int(mc_passes))
                with torch.no_grad():
                    for pass_idx in range(passes):
                        torch.manual_seed(1000003 + pass_idx)
                        mask = (torch.rand_like(fc_input) < keep_prob).float() / keep_prob
                        preds.append(model.fc_out(fc_input * mask).view(-1).detach().cpu().numpy())
                pred_matrix = np.vstack(preds)
                variances = np.var(pred_matrix, axis=0) if pred_matrix.shape[0] >= 2 else np.zeros(len(batch_ids))
                for sample_id, var in zip(batch_ids, variances):
                    dropout_vars[sample_id] = float(var)
            except Exception as exc:
                warnings.append(f"mc dropout failed for batch {batch_ids[:3]}: {exc}")
                for sample_id in batch_ids:
                    dropout_vars[sample_id] = 0.0
        except Exception as exc:
            warnings.append(f"feature extraction failed for batch {batch_ids[:3]}: {exc}")
            for sample_id in batch_ids:
                grad_norms[sample_id] = 0.0
                dropout_vars[sample_id] = 0.0

    return features, grad_norms, dropout_vars, warnings


def compute_neighbor_reward(ids, features, p_hit, neighbor_k):
    if len(ids) <= 1 or not features:
        return {sample_id: 0.0 for sample_id in ids}, "neighbor reward unavailable; pool too small or features missing"

    usable_ids = [sample_id for sample_id in ids if sample_id in features]
    if len(usable_ids) <= 1:
        return {sample_id: 0.0 for sample_id in ids}, "neighbor reward unavailable; too few usable features"

    matrix = np.vstack([features[sample_id] for sample_id in usable_ids]).astype(float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, EPS)
    sims = matrix @ matrix.T
    distances = 1.0 - sims
    np.fill_diagonal(distances, np.inf)

    k = min(int(neighbor_k), len(usable_ids) - 1)
    p_hit_map = dict(zip(ids, p_hit))
    rewards = {sample_id: 0.0 for sample_id in ids}
    for idx, sample_id in enumerate(usable_ids):
        nn_idx = np.argpartition(distances[idx], k - 1)[:k]
        rewards[sample_id] = float(np.mean([p_hit_map[usable_ids[j]] for j in nn_idx]))
    return rewards, None


def compute_history_bonus(ids, group_by_id, history_path):
    history_path = Path(history_path)
    if not history_path.exists() or history_path.stat().st_size == 0:
        return {sample_id: 0.0 for sample_id in ids}

    try:
        hist = pd.read_csv(history_path)
    except Exception:
        return {sample_id: 0.0 for sample_id in ids}
    if "id" not in hist.columns:
        return {sample_id: 0.0 for sample_id in ids}

    hist["id"] = hist["id"].apply(clean_id)
    if "group_key" in hist.columns:
        hist_groups = hist["group_key"].fillna("UNKNOWN").astype(str)
    else:
        hist_groups = hist["id"].map(group_by_id).fillna("UNKNOWN")
    n_by_group = hist_groups.value_counts().to_dict()
    total = int(len(hist))

    hit_by_group = defaultdict(int)
    can_use_hits = "is_valid" in hist.columns
    if can_use_hits:
        valid_series = pd.to_numeric(hist["is_valid"], errors="coerce").fillna(0).astype(int)
        for group, is_valid in zip(hist_groups, valid_series):
            hit_by_group[group] += int(is_valid > 0)

    raw = {}
    alpha = 1.0
    beta = 1.0
    c = 0.10
    for sample_id in ids:
        group = group_by_id.get(sample_id, "UNKNOWN")
        n_g = float(n_by_group.get(group, 0))
        coverage = c * math.sqrt(math.log(total + 2.0) / (n_g + 1.0)) if total >= 0 else 0.0
        if can_use_hits:
            hit_g = float(hit_by_group.get(group, 0))
            raw[sample_id] = (hit_g + alpha) / (n_g + alpha + beta) + coverage
        else:
            raw[sample_id] = coverage
    norm = normalize01([raw[sample_id] for sample_id in ids])
    return dict(zip(ids, norm))


def choose_stage(remaining_count):
    total_env = os.getenv("TOTAL_POOL_SIZE") or os.getenv("INITIAL_POOL_SIZE")
    try:
        initial = float(total_env) if total_env else 0.0
    except ValueError:
        initial = 0.0
    if initial <= 0:
        return "early"
    ratio = remaining_count / initial
    if ratio > 2.0 / 3.0:
        return "early"
    if ratio > 1.0 / 3.0:
        return "middle"
    return "late"


def scheduled_params(stage, lambda_u, lambda_r, mmr_beta):
    if stage == "early":
        return 0.8, 0.5, 0.12
    if stage == "middle":
        return 0.6, 0.4, 0.08
    if stage == "late":
        return 0.3, 0.2, 0.04
    return lambda_u, lambda_r, mmr_beta


def mmr_select(ids, score_base, features, selection_size, beta):
    if not features or any(sample_id not in features for sample_id in ids):
        ordered = sorted(ids, key=lambda sample_id: score_base.get(sample_id, 0.0), reverse=True)
        return ordered[:selection_size], {sample_id: score_base.get(sample_id, 0.0) for sample_id in ordered[:selection_size]}, True

    feature_matrix = {sample_id: np.asarray(features[sample_id], dtype=float) for sample_id in ids}
    normed = {}
    for sample_id, vec in feature_matrix.items():
        normed[sample_id] = vec / max(float(np.linalg.norm(vec)), EPS)

    selected = []
    mmr_scores = {}
    remaining = set(ids)
    while remaining and len(selected) < selection_size:
        best_id = None
        best_score = -float("inf")
        for sample_id in remaining:
            if not selected:
                score = score_base.get(sample_id, 0.0)
            else:
                sim = max(float(np.dot(normed[sample_id], normed[chosen])) for chosen in selected)
                score = score_base.get(sample_id, 0.0) - beta * sim
            if score > best_score:
                best_score = score
                best_id = sample_id
        selected.append(best_id)
        mmr_scores[best_id] = best_score
        remaining.remove(best_id)
    return selected, mmr_scores, False


def main():
    parser = argparse.ArgumentParser(description="ETDG-TAGE active-learning selection.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--original-dir", required=True)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--test-results", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-feature", type=float, required=True)
    parser.add_argument("--target-mode", choices=["threshold", "quantile"], default="threshold")
    parser.add_argument("--formation-energy-threshold", type=float, default=-2.0)
    parser.add_argument("--target-quantile", type=float, default=0.10)
    parser.add_argument("--selection-size", type=int, default=50)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--mc-passes", type=int, default=3)
    parser.add_argument("--dropout-rate", type=float, default=0.30)
    parser.add_argument("--neighbor-k", type=int, default=10)
    parser.add_argument("--alpha-g", type=float, default=0.6)
    parser.add_argument("--lambda-u", type=float, default=0.6)
    parser.add_argument("--lambda-r", type=float, default=0.4)
    parser.add_argument("--lambda-c", type=float, default=0.01)
    parser.add_argument("--mmr-beta", type=float, default=0.15)
    parser.add_argument("--hit-gamma", type=float, default=1.5)
    parser.add_argument("--score-log-dir", default=None)
    parser.add_argument("--selection-method-name", default="etdg_tage")
    args = parser.parse_args()

    pred = pd.read_csv(args.test_results, header=None, names=["id", "actual_feature", "pseudo_feature"])
    pred["id"] = pred["id"].apply(clean_id)
    pred["pseudo_feature"] = pd.to_numeric(pred["pseudo_feature"], errors="coerce")
    pred = pred.dropna(subset=["pseudo_feature"]).copy()
    if pred.empty:
        raise ValueError("No valid pseudo predictions found in test_results.csv")

    ids = pred["id"].tolist()
    pseudo = pred["pseudo_feature"].to_numpy(dtype=float)
    pseudo_by_id = dict(zip(ids, pseudo))

    if args.target_mode == "threshold":
        tau_target = float(args.formation_energy_threshold)
    else:
        tau_target = float(np.quantile(pseudo, args.target_quantile))
    temperature = max(float(np.std(pseudo) * 0.25), 1e-6)
    p_hit = np.clip(sigmoid((tau_target - pseudo) / temperature), 0.0, 1.0)

    features, grad_norms, dropout_vars, warnings = extract_features_gradients_dropout(
        args.model,
        args.pseudo_dir,
        args.mc_passes,
        args.dropout_rate,
        tau_target,
    )
    if warnings:
        for warning in warnings[:50]:
            print(f"WARNING: {warning}")

    g_norm = normalize01([grad_norms.get(sample_id, 0.0) for sample_id in ids])
    d_norm = normalize01([dropout_vars.get(sample_id, 0.0) for sample_id in ids])
    g_norm_by_id = dict(zip(ids, g_norm))
    d_norm_by_id = dict(zip(ids, d_norm))
    u_val = p_hit * (float(args.alpha_g) * g_norm + (1.0 - float(args.alpha_g)) * d_norm)

    neighbor_raw, neighbor_warning = compute_neighbor_reward(ids, features, p_hit, args.neighbor_k)
    if neighbor_warning:
        print(f"WARNING: {neighbor_warning}")
    r_val = normalize01([neighbor_raw.get(sample_id, 0.0) for sample_id in ids])
    r_norm_by_id = dict(zip(ids, r_val))

    group_by_id = {sample_id: group_key(sample_id, args.pseudo_dir, args.original_dir, args.output_dir) for sample_id in ids}
    history_path = os.getenv("ACTIVE_HISTORY", "active_learning_history.csv")
    c_map = compute_history_bonus(ids, group_by_id, history_path)
    c_val = np.asarray([c_map.get(sample_id, 0.0) for sample_id in ids], dtype=float)

    stage = choose_stage(len(ids))
    lambda_u, lambda_r, mmr_beta = scheduled_params(stage, args.lambda_u, args.lambda_r, args.mmr_beta)
    lambda_c = float(args.lambda_c)
    hit_gamma = float(args.hit_gamma)

    p_main = np.power(np.clip(p_hit, 0.0, 1.0), hit_gamma)
    c_bonus_raw = lambda_c * c_val
    c_bonus_cap = 0.10 * p_main
    c_bonus_capped = np.minimum(c_bonus_raw, c_bonus_cap)
    c_contrib_ratio = c_bonus_capped / np.maximum(p_main, 1e-12)
    score_base = p_main * (1.0 + lambda_u * u_val) * (1.0 + lambda_r * r_val) + c_bonus_capped
    score_by_id = dict(zip(ids, score_base))
    p_hit_by_id = dict(zip(ids, p_hit))

    selected_ids, mmr_scores, used_score_fallback = mmr_select(
        ids,
        score_by_id,
        features,
        min(args.selection_size, len(ids)),
        mmr_beta,
    )
    if used_score_fallback:
        print("WARNING: MMR feature similarity unavailable; selected by Score_base ranking.")

    selected_set = set(selected_ids)

    log_dir = Path(args.score_log_dir or args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    score_df = pred.copy()
    score_df["P_hit"] = p_hit
    score_df["G_i"] = g_norm
    score_df["D_i"] = d_norm
    score_df["U_i"] = u_val
    score_df["R_i"] = r_val
    score_df["group_key"] = score_df["id"].map(group_by_id)
    score_df["C_i"] = c_val
    score_df["P_main"] = p_main
    score_df["C_bonus_raw"] = c_bonus_raw
    score_df["C_bonus_cap"] = c_bonus_cap
    score_df["C_bonus_capped"] = c_bonus_capped
    score_df["C_contrib_ratio"] = c_contrib_ratio
    score_df["P_hit_main"] = p_main
    score_df["C_contrib"] = c_bonus_capped
    score_df["Score_base"] = score_base
    score_df["MMR_score"] = score_df["id"].map(mmr_scores)
    score_df["selected"] = score_df["id"].isin(selected_set).astype(int)
    score_df["stage"] = stage
    score_df["lambda_u"] = lambda_u
    score_df["lambda_r"] = lambda_r
    score_df["lambda_c"] = lambda_c
    score_df["mmr_beta"] = mmr_beta
    score_df["hit_gamma"] = hit_gamma
    score_df["target_mode"] = args.target_mode
    score_df["tau_target"] = tau_target
    score_df["formation_energy_threshold"] = args.formation_energy_threshold if args.target_mode == "threshold" else np.nan
    score_df["target_quantile"] = args.target_quantile if args.target_mode == "quantile" else np.nan
    score_df.to_csv(log_dir / f"etdg_tage_scores_iter_{args.iteration}.csv", index=False)

    original_df = pd.read_csv(Path(args.original_dir) / "id_prop.csv", header=None, names=["id", "feature"])
    original_df["id"] = original_df["id"].apply(clean_id)
    selected_df = original_df[original_df["id"].isin(selected_set)].copy()
    remaining_df = original_df[~original_df["id"].isin(selected_set)].copy()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_df.to_csv(output_dir / "id_prop.csv", mode="a", header=False, index=False)
    for sample_id in selected_df["id"]:
        src = Path(args.original_dir) / f"{sample_id}.cif"
        dst = output_dir / f"{sample_id}.cif"
        if src.exists():
            shutil.copy(src, dst)
    remaining_df.to_csv(Path(args.original_dir) / "id_prop.csv", header=False, index=False)

    hist_rows = []
    pred_by_id = pred.set_index("id")
    for sample_id in selected_ids:
        if sample_id not in pred_by_id.index:
            continue
        row = pred_by_id.loc[sample_id]
        hist_rows.append(
            {
                "id": sample_id,
                "iteration": args.iteration,
                "target_feature": args.target_feature,
                "pseudo_feature": row["pseudo_feature"],
                "actual_feature": row["actual_feature"],
                "distance": row["pseudo_feature"] - args.target_feature,
                "selection_method": args.selection_method_name,
                "P_hit": p_hit_by_id.get(sample_id, np.nan),
                "G_i": g_norm_by_id.get(sample_id, 0.0),
                "D_i": d_norm_by_id.get(sample_id, 0.0),
                "R_i": r_norm_by_id.get(sample_id, 0.0),
                "group_key": group_by_id.get(sample_id, "UNKNOWN"),
                "Score_base": score_by_id.get(sample_id, np.nan),
            }
        )
    history_path = Path(history_path)
    pd.DataFrame(hist_rows).to_csv(history_path, mode="a", header=not history_path.exists(), index=False)

    selected_scores = score_df[score_df["selected"] == 1]
    print("\nETDG-TAGE Selection Summary:")
    print(f"candidate pool size: {len(pred)}")
    print(f"selection size: {len(selected_ids)}")
    print(f"stage: {stage}")
    print(
        "selected pseudo min / median / max: "
        f"{selected_scores['pseudo_feature'].min():.6f} / {selected_scores['pseudo_feature'].median():.6f} / {selected_scores['pseudo_feature'].max():.6f}"
    )
    print(f"selected P_hit mean: {selected_scores['P_hit'].mean():.6f}")
    print(f"pool P_hit mean: {score_df['P_hit'].mean():.6f}")
    print(f"selected G_i mean: {selected_scores['G_i'].mean():.6f}")
    print(f"selected D_i mean: {selected_scores['D_i'].mean():.6f}")
    print(f"selected R_i mean: {selected_scores['R_i'].mean():.6f}")
    print(f"selected group counts: {dict(Counter(selected_scores['group_key']))}")
    print(f"target_mode: {args.target_mode}")
    print(f"tau_target: {tau_target:.6f}")
    print(f"hit_gamma: {hit_gamma:.6f}")
    if selected_scores["P_hit"].mean() < score_df["P_hit"].mean():
        print("WARNING: selected samples have lower P_hit than pool average.")
    if (score_df["C_contrib_ratio"] > 0.10 + 1e-12).any():
        print("WARNING: C_i contribution exceeds bounded auxiliary limit.")


if __name__ == "__main__":
    main()
