from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file, write_bytes_protected


METHOD_ORDER = [
    "Interval-Hit Greedy",
    "Always-DA-TPP",
    "Margin-only Gate",
    "Group-only Gate",
    "Full Energy-Gated DA-TPP",
]
EPS = 1e-12
LIMO_FORMAL_PARAMS = {
    "M0": 1.0,
    "G0": 0.50,
    "alpha": 0.10,
    "beta": 0.20,
    "gamma": 0.10,
    "mc_passes": 3,
    "dropout_rate": 0.30,
}


def route_for_method(
    method: str,
    margin: float,
    concentration: float,
    margin_threshold: float,
    concentration_threshold: float,
) -> str:
    if method == "Interval-Hit Greedy":
        return "direct"
    if method == "Always-DA-TPP":
        return "correction"
    if method == "Margin-only Gate":
        return "direct" if margin >= margin_threshold else "correction"
    if method == "Group-only Gate":
        return "direct" if concentration <= concentration_threshold else "correction"
    if method == "Full Energy-Gated DA-TPP":
        return (
            "direct"
            if margin >= margin_threshold and concentration <= concentration_threshold
            else "correction"
        )
    raise ValueError(f"unknown ablation method: {method}")


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in values],
        dtype=float,
    )


def _normalize01(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    result = np.zeros_like(array)
    finite = np.isfinite(array)
    if not finite.any():
        return result
    low = float(np.nanmin(array[finite]))
    high = float(np.nanmax(array[finite]))
    if high - low <= EPS:
        return result
    result[finite] = (array[finite] - low) / (high - low)
    return np.clip(result, 0.0, 1.0)


def _diversity_select(
    ids: Sequence[str],
    p_hit: np.ndarray,
    uncertainty: np.ndarray,
    groups: Sequence[str],
    similarity: np.ndarray,
    batch_size: int,
    prefilter_multiplier: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> list[int]:
    prefilter_size = min(len(ids), int(prefilter_multiplier * batch_size))
    quality = p_hit + alpha * uncertainty
    prefilter_idx = np.argsort(-quality)[:prefilter_size]
    selected: list[int] = []
    selected_group_counts: Counter[str] = Counter()
    while len(selected) < min(batch_size, prefilter_size):
        best_idx: int | None = None
        best_score = -float("inf")
        for raw_idx in prefilter_idx:
            idx = int(raw_idx)
            if idx in selected:
                continue
            max_similarity = max(
                (float(np.dot(similarity[idx], similarity[chosen])) for chosen in selected),
                default=0.0,
            )
            group_penalty = selected_group_counts[groups[idx]] / max(1, len(selected))
            score = float(
                p_hit[idx]
                + alpha * uncertainty[idx]
                - beta * max_similarity
                - gamma * group_penalty
            )
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is None:
            break
        selected.append(best_idx)
        selected_group_counts[groups[best_idx]] += 1
    return selected


def _shared_state_hash(
    ids: Sequence[str], pseudo: np.ndarray, raw_sigma: np.ndarray, groups: Sequence[str]
) -> str:
    frame = pd.DataFrame(
        {
            "id": list(ids),
            "pseudo": np.asarray(pseudo, dtype=float),
            "raw_sigma": np.asarray(raw_sigma, dtype=float),
            "group": list(groups),
        }
    )
    return hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def evaluate_gate_methods(
    *,
    ids: Sequence[str],
    pseudo: np.ndarray,
    raw_sigma: np.ndarray,
    groups: Sequence[str],
    similarity: np.ndarray,
    target_low: float,
    target_high: float,
    batch_size: int,
    margin_threshold: float,
    concentration_threshold: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> pd.DataFrame:
    ids = list(ids)
    pseudo = np.asarray(pseudo, dtype=float)
    raw_sigma = np.asarray(raw_sigma, dtype=float)
    similarity = np.asarray(similarity, dtype=float)
    if not (len(ids) == len(pseudo) == len(raw_sigma) == len(groups) == len(similarity)):
        raise ValueError("all score-state arrays must have one row per candidate")
    if similarity.ndim != 2:
        raise ValueError("similarity features must be a two-dimensional normalized feature matrix")

    uncertainty = _normalize01(raw_sigma)
    if np.max(raw_sigma) > EPS:
        sigma = np.maximum(raw_sigma, 0.05)
        p_hit = _normal_cdf((target_high - pseudo) / sigma) - _normal_cdf(
            (target_low - pseudo) / sigma
        )
        p_hit_mode = "mc_dropout_interval_cdf"
    else:
        center = (target_low + target_high) / 2.0
        p_hit = _normalize01(-np.abs(pseudo - center))
        p_hit_mode = "rank_normalized_interval_center_fallback"
    p_hit = np.clip(p_hit, 0.0, 1.0)

    order = np.argsort(-p_hit)
    actual_batch_size = min(int(batch_size), len(ids))
    top_b = [int(value) for value in order[:actual_batch_size]]
    next_b = order[actual_batch_size : min(2 * actual_batch_size, len(ids))]
    top_2b = order[: min(2 * actual_batch_size, len(ids))]
    margin = (
        float(
            (np.mean(p_hit[top_b]) - np.mean(p_hit[next_b]))
            / (np.std(p_hit[top_2b]) + 1e-8)
        )
        if len(next_b)
        else float("inf")
    )
    top_groups = Counter(groups[idx] for idx in top_b)
    concentration = max(top_groups.values(), default=0) / max(1, actual_batch_size)
    shared_hash = _shared_state_hash(ids, pseudo, raw_sigma, groups)

    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        route = route_for_method(
            method,
            margin,
            concentration,
            margin_threshold,
            concentration_threshold,
        )
        if route == "direct":
            selected = top_b
        else:
            multiplier = {16: 12, 32: 10, 64: 8}.get(int(batch_size), 10)
            selected = _diversity_select(
                ids,
                p_hit,
                uncertainty,
                groups,
                similarity,
                actual_batch_size,
                multiplier,
                alpha,
                beta,
                gamma,
            )
        selected_ids = [ids[index] for index in selected]
        selected_groups = [groups[index] for index in selected]
        unique_groups = len(set(selected_groups))
        replacements = actual_batch_size - len(set(selected).intersection(top_b))
        rows.append(
            {
                "method": method,
                "route": route,
                "direct_rounds": int(route == "direct"),
                "correction_rounds": int(route == "correction"),
                "selection_size": len(selected),
                "effective_replacements": replacements,
                "unique_groups": unique_groups,
                "repetition_rate": 1.0 - unique_groups / max(1, len(selected)),
                "top_group_fraction_selected": (
                    max(Counter(selected_groups).values(), default=0) / max(1, len(selected))
                ),
                "mean_selected_p_hit": float(np.mean(p_hit[selected])) if selected else 0.0,
                "margin_score": margin,
                "group_concentration_top_b": concentration,
                "p_hit_mode": p_hit_mode,
                "selected_candidate_ids": "|".join(selected_ids),
                "selected_sequence_sha256": hashlib.sha256(
                    "\n".join(selected_ids).encode("utf-8")
                ).hexdigest(),
                "shared_score_state_sha256": shared_hash,
                "autc": "not_computed_single_round_smoke",
            }
        )
    return pd.DataFrame(rows)


def load_formal_gate_module(formal_path: Path, support_root: Path) -> ModuleType:
    formal_path = Path(formal_path).resolve()
    support_root = Path(support_root).resolve()
    if not formal_path.is_file():
        raise FileNotFoundError(formal_path)
    if not (support_root / "active_learning_etdg_tage.py").is_file():
        raise FileNotFoundError(support_root / "active_learning_etdg_tage.py")
    root_text = str(support_root)
    sys.path.insert(0, root_text)
    try:
        module_name = f"formal_energy_gate_{sha256_file(formal_path)[:12]}"
        spec = importlib.util.spec_from_file_location(module_name, formal_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load formal selector: {formal_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _run_prediction(
    archive: Path,
    checkpoint: Path,
    pool: Path,
    work_dir: Path,
    inference_seed: int,
) -> tuple[Path, float]:
    output = work_dir / "test_results.csv"
    if output.exists():
        return output, 0.0
    wrapper = archive / "experiments/reproducibility/seeded_runpy_torch_compat.py"
    predictor = archive / "experiments/reproducibility/paired_predict_no_shuffle.py"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(archive) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        str(wrapper),
        "--seed",
        str(inference_seed),
        str(predictor),
        str(checkpoint),
        str(pool),
        "--batch-size",
        "256",
        "--workers",
        "0",
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=work_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log = work_dir / "prediction.log"
    with log.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(completed.stdout)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"prediction failed with exit code {completed.returncode}; see {log}")
    return output, elapsed


def _load_or_compute_score_state(
    formal: ModuleType,
    checkpoint: Path,
    pool: Path,
    prediction_path: Path,
    work_dir: Path,
    inference_seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], str, float, float, list[str]]:
    cache = work_dir / "score_state.npz"
    prediction = pd.read_csv(
        prediction_path,
        header=None,
        names=["id", "actual_feature_ignored", "pseudo_feature"],
    )
    prediction["id"] = prediction["id"].map(formal.clean_id)
    prediction["pseudo_feature"] = pd.to_numeric(prediction["pseudo_feature"], errors="coerce")
    prediction = prediction.dropna(subset=["pseudo_feature"]).copy()
    ids = prediction["id"].tolist()
    if cache.exists():
        cached = np.load(cache, allow_pickle=False)
        if cached["ids"].astype(str).tolist() != ids:
            raise RuntimeError("cached score-state IDs do not match current prediction IDs")
        return (
            prediction,
            cached["raw_sigma"].astype(float),
            cached["similarity"].astype(float),
            cached["groups"].astype(str).tolist(),
            str(cached["similarity_mode"].item()),
            0.0,
            float(cached["gpu_peak_mib"].item()),
            cached["warnings"].astype(str).tolist(),
        )

    _set_seed(inference_seed)
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    features, _, dropout_vars, warnings = formal.extract_features_gradients_dropout(
        str(checkpoint), str(pool), 3, 0.30, -2.10
    )
    elapsed = time.perf_counter() - started
    peak_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    )
    groups = [formal.group_key(candidate_id, str(pool)) for candidate_id in ids]
    raw_sigma = np.sqrt(
        np.maximum([dropout_vars.get(candidate_id, 0.0) for candidate_id in ids], 0.0)
    )
    similarity, similarity_mode = formal.choose_similarity(ids, features, groups)
    with cache.open("xb") as handle:
        np.savez_compressed(
            handle,
            ids=np.asarray(ids, dtype=str),
            raw_sigma=np.asarray(raw_sigma, dtype=float),
            similarity=np.asarray(similarity, dtype=float),
            groups=np.asarray(groups, dtype=str),
            similarity_mode=np.asarray(similarity_mode),
            gpu_peak_mib=np.asarray(peak_mib),
            warnings=np.asarray(warnings, dtype=str),
        )
    return prediction, raw_sigma, similarity, groups, similarity_mode, elapsed, peak_mib, warnings


def _verify_invariants(formal: ModuleType) -> dict[str, bool]:
    grid = [(margin, concentration) for margin in (0.0, 1.0, 2.0) for concentration in (0.5, 0.75, 0.9)]
    m0 = LIMO_FORMAL_PARAMS["M0"]
    g0 = LIMO_FORMAL_PARAMS["G0"]
    return {
        "greedy_always_direct": all(
            route_for_method("Interval-Hit Greedy", margin, group, m0, g0) == "direct"
            for margin, group in grid
        ),
        "always_datpp_always_correction": all(
            route_for_method("Always-DA-TPP", margin, group, m0, g0) == "correction"
            for margin, group in grid
        ),
        "margin_only_ignores_group": all(
            route_for_method("Margin-only Gate", margin, 0.5, m0, g0)
            == route_for_method("Margin-only Gate", margin, 0.9, m0, g0)
            for margin in (0.0, 1.0, 2.0)
        ),
        "group_only_ignores_margin": all(
            route_for_method("Group-only Gate", 0.0, group, m0, g0)
            == route_for_method("Group-only Gate", 2.0, group, m0, g0)
            for group in (0.5, 0.75, 0.9)
        ),
        "full_matches_formal_gate_mode": all(
            route_for_method("Full Energy-Gated DA-TPP", margin, group, m0, g0)
            == {
                "threshold_greedy": "direct",
                "diversity_aware": "correction",
            }[formal.gate_mode("full", margin, group, m0, g0)]
            for margin, group in grid
        ),
    }


def _restore_cached_runtime(observed: float, prior: pd.DataFrame, column: str) -> float:
    if float(observed) > 0.0:
        return float(observed)
    if column not in prior.columns:
        raise ValueError(f"prior smoke output is missing runtime column {column}")
    values = pd.to_numeric(prior[column], errors="raise").dropna().unique()
    if len(values) != 1:
        raise ValueError(f"prior smoke output must contain one shared value for {column}")
    return float(values[0])


def _restore_cached_boolean(prior: pd.DataFrame, column: str) -> bool:
    if column not in prior.columns:
        raise ValueError(f"prior smoke output is missing boolean column {column}")
    values = prior[column].astype(str).str.strip().str.lower().unique()
    if len(values) != 1 or values[0] not in {"true", "false"}:
        raise ValueError(f"prior smoke output must contain one boolean value for {column}")
    return values[0] == "true"


def _render_report(
    result: pd.DataFrame,
    invariants: dict[str, bool],
    inputs: dict[str, str],
    prediction_seconds: float,
    mc_seconds: float,
    gpu_peak_mib: float,
    warnings: Sequence[str],
    runtime_restored_from_preserved_v1: bool,
) -> str:
    lines = [
        "# Gate Ablation Smoke Report",
        "",
        "## Scope",
        "",
        "This is a one-round Li–M–O smoke test on development seed 0. It uses one shared prediction, MC-dropout score state, candidate pool and initial checkpoint for all five selection rules. It does not train a model, close the active-learning loop, estimate AUTC, or use seeds 5–14 for parameter selection.",
        "",
        "The gate and correction parameters are copied from the retained Li–M–O corrected-seed run configuration: M0=1.0, G0=0.50, alpha=0.10, beta=0.20, gamma=0.10, MC passes=3 and dropout rate=0.30. They are not reselected here.",
        "",
        "## Automated behavior checks",
        "",
    ]
    for name, passed in invariants.items():
        lines.append(f"- `{name}`: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "The formal-equivalence check calls `gate_mode` from the retained formal selector source. Selection scoring in this audit is an independent, line-for-line behavioral reimplementation and is covered by unit tests.",
            "",
            "## Single-round observations",
            "",
            result[
                [
                    "method",
                    "route",
                    "effective_replacements",
                    "unique_groups",
                    "repetition_rate",
                    "target_hits_after_selection",
                ]
            ].to_markdown(index=False),
            "",
            f"Shared margin score: `{result['margin_score'].iloc[0]:.12g}`; shared top-b group concentration: `{result['group_concentration_top_b'].iloc[0]:.12g}`.",
            "",
            "`target_hits_after_selection` is joined from the oracle only after selection and is not part of routing or batch construction.",
            "",
            "## Runtime and provenance",
            "",
            f"- Deterministic base prediction: {prediction_seconds:.3f} s",
            f"- Feature/MC3 extraction: {mc_seconds:.3f} s",
            f"- Peak allocated GPU memory observed in feature/MC3 extraction: {gpu_peak_mib:.3f} MiB",
            f"- Extraction warnings: {len(warnings)}",
            "- Runtime source: "
            + (
                "preserved initial shared-inference measurement; changing G0 from 0.75 to the retained formal 0.50 does not change prediction or MC extraction"
                if runtime_restored_from_preserved_v1
                else "current execution"
            ),
        ]
    )
    for label, value in inputs.items():
        lines.append(f"- {label}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A PASS establishes route semantics and executable compatibility for one acquisition round only. It is not evidence for ten-seed AUTC, checkpoint recovery, or superiority of any ablation. Those claims remain pending a separately authorized batch experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--historical-root", type=Path, default=Path(r"D:\CGCNN"))
    parser.add_argument("--development-seed", type=int, default=0)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()

    archive = args.archive.resolve()
    historical_root = args.historical_root.resolve()
    pool = historical_root / "EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_FLAT_20260617"
    oracle_path = (
        historical_root / "EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_20260617/oracle.csv"
    )
    checkpoint = historical_root / "checkpoint_formation_clean.pth.tar"
    formal_path = archive / "active_learning_energy_gate_ablation.py"
    support_root = archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712"
    formal_config_path = (
        archive
        / "baseline_snapshot/archive/experiments/reproducibility/results"
        / "paired_two_dataset_confirmation_20260712/runs/limo"
        / "energy_gated_da_tpp/seed_5/run_config.json"
    )
    required = [
        pool / "id_prop.csv",
        pool / "atom_init.json",
        oracle_path,
        checkpoint,
        formal_path,
        formal_config_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required smoke-test evidence missing: {missing}")
    if args.development_seed in range(5, 15):
        raise ValueError("seeds 5–14 are reserved for formal evaluation, not this development smoke test")

    work_dir = archive / f"tmp/gate_ablation_smoke_seed{args.development_seed}"
    work_dir.mkdir(parents=True, exist_ok=True)
    inference_seed = args.development_seed * 1_000_000 + 100_000 + 1
    prediction_path, prediction_seconds = _run_prediction(
        archive, checkpoint, pool, work_dir, inference_seed
    )
    formal = load_formal_gate_module(formal_path, support_root)
    (
        prediction,
        raw_sigma,
        similarity,
        groups,
        similarity_mode,
        mc_seconds,
        gpu_peak_mib,
        warnings,
    ) = _load_or_compute_score_state(
        formal, checkpoint, pool, prediction_path, work_dir, inference_seed
    )
    prior_output_path = archive / "results/smoke_tests/gate_ablation_smoke.csv"
    preserved_v1_path = (
        archive / "results/smoke_tests/superseded_v1_g0_075/gate_ablation_smoke.csv"
    )
    runtime_restored_from_preserved_v1 = False
    prior_runtime_path: Path | None = None
    if args.check_existing and prior_output_path.is_file():
        prior_runtime_path = prior_output_path
    elif (prediction_seconds <= 0.0 or mc_seconds <= 0.0) and preserved_v1_path.is_file():
        prior_runtime_path = preserved_v1_path
        runtime_restored_from_preserved_v1 = True
    if prior_runtime_path is not None:
        prior_output = pd.read_csv(prior_runtime_path)
        prediction_seconds = _restore_cached_runtime(
            prediction_seconds, prior_output, "prediction_runtime_seconds_shared"
        )
        mc_seconds = _restore_cached_runtime(
            mc_seconds, prior_output, "mc3_runtime_seconds_shared"
        )
        if prior_runtime_path == prior_output_path and "runtime_restored_from_preserved_v1" in prior_output:
            runtime_restored_from_preserved_v1 = _restore_cached_boolean(
                prior_output, "runtime_restored_from_preserved_v1"
            )

    result = evaluate_gate_methods(
        ids=prediction["id"].tolist(),
        pseudo=prediction["pseudo_feature"].to_numpy(dtype=float),
        raw_sigma=raw_sigma,
        groups=groups,
        similarity=similarity,
        target_low=-2.18,
        target_high=-2.02,
        batch_size=16,
        margin_threshold=LIMO_FORMAL_PARAMS["M0"],
        concentration_threshold=LIMO_FORMAL_PARAMS["G0"],
        alpha=LIMO_FORMAL_PARAMS["alpha"],
        beta=LIMO_FORMAL_PARAMS["beta"],
        gamma=LIMO_FORMAL_PARAMS["gamma"],
    )
    oracle = pd.read_csv(oracle_path)
    oracle["candidate_id"] = oracle["candidate_id"].map(formal.clean_id)
    target_by_id = oracle.set_index("candidate_id")["target_label"].astype(int).to_dict()
    result["target_hits_after_selection"] = result["selected_candidate_ids"].map(
        lambda value: sum(target_by_id.get(candidate_id, 0) for candidate_id in value.split("|"))
    )
    result.insert(0, "dataset", "Li-M-O")
    result.insert(1, "development_seed", args.development_seed)
    result.insert(2, "round", 1)
    result["inference_seed"] = inference_seed
    result["mc_passes"] = LIMO_FORMAL_PARAMS["mc_passes"]
    result["dropout_rate"] = LIMO_FORMAL_PARAMS["dropout_rate"]
    result["M0"] = LIMO_FORMAL_PARAMS["M0"]
    result["G0"] = LIMO_FORMAL_PARAMS["G0"]
    result["alpha"] = LIMO_FORMAL_PARAMS["alpha"]
    result["beta"] = LIMO_FORMAL_PARAMS["beta"]
    result["gamma"] = LIMO_FORMAL_PARAMS["gamma"]
    result["similarity_mode"] = similarity_mode
    result["prediction_runtime_seconds_shared"] = prediction_seconds
    result["mc3_runtime_seconds_shared"] = mc_seconds
    result["gpu_peak_allocated_mib_mc3"] = gpu_peak_mib
    result["runtime_restored_from_preserved_v1"] = runtime_restored_from_preserved_v1

    invariants = _verify_invariants(formal)
    if not all(invariants.values()):
        raise AssertionError(f"gate smoke behavior check failed: {invariants}")
    inputs = {
        "checkpoint path / SHA-256": f"{checkpoint} / {sha256_file(checkpoint)}",
        "pool id_prop path / SHA-256": f"{pool / 'id_prop.csv'} / {sha256_file(pool / 'id_prop.csv')}",
        "oracle path / SHA-256": f"{oracle_path} / {sha256_file(oracle_path)}",
        "formal selector path / SHA-256": f"{formal_path} / {sha256_file(formal_path)}",
        "formal support path / SHA-256": (
            f"{support_root / 'active_learning_etdg_tage.py'} / "
            f"{sha256_file(support_root / 'active_learning_etdg_tage.py')}"
        ),
        "retained Li-M-O formal run config / SHA-256": (
            f"{formal_config_path} / {sha256_file(formal_config_path)}"
        ),
        "shared prediction path / SHA-256": (
            f"{prediction_path} / {sha256_file(prediction_path)}"
        ),
        "inference seed": str(inference_seed),
        "manual MC mask seeds": "1000003, 1000004, 1000005 (retained implementation)",
    }
    csv_buffer = io.StringIO()
    result.to_csv(csv_buffer, index=False, lineterminator="\n")
    report = _render_report(
        result,
        invariants,
        inputs,
        prediction_seconds,
        mc_seconds,
        gpu_peak_mib,
        warnings,
        runtime_restored_from_preserved_v1,
    )
    states = {
        archive / "results/smoke_tests/gate_ablation_smoke.csv": write_bytes_protected(
            archive / "results/smoke_tests/gate_ablation_smoke.csv",
            csv_buffer.getvalue().encode("utf-8"),
            args.check_existing,
        ),
        archive / "docs/GATE_ABLATION_SMOKE_REPORT.md": write_bytes_protected(
            archive / "docs/GATE_ABLATION_SMOKE_REPORT.md",
            report.encode("utf-8"),
            args.check_existing,
        ),
    }
    print(json.dumps({str(path): state for path, state in states.items()}, indent=2))


if __name__ == "__main__":
    main()
