from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.audit_common import sha256_file, write_bytes_protected
from analysis.run_gate_ablation_smoke import (
    LIMO_FORMAL_PARAMS,
    evaluate_gate_methods,
    load_formal_gate_module,
)


K_VALUES = (3, 10, 30)


def loaded_formal_support_module():
    module = sys.modules.get("active_learning_etdg_tage")
    if module is None:
        raise RuntimeError("formal gate module must be loaded before requesting its support module")
    return module


def mc_mask_seeds(k: int) -> list[int]:
    if int(k) < 1:
        raise ValueError("MC pass count must be positive")
    return [1_000_003 + index for index in range(int(k))]


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.array_equal(left, right):
        return 1.0
    result = spearmanr(left, right)
    return float(result.statistic)


def _overlap(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    denominator = max(1, len(left_set))
    return len(left_set.intersection(right_set)) / denominator


def _sequence_hash(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def summarize_mc_sensitivity(
    states: dict[int, pd.DataFrame],
    selections: dict[int, dict[str, object]],
    *,
    runtimes: dict[int, float],
    peak_memory_mib: dict[int, float],
    embedding_runtime_seconds: float,
    embedding_peak_memory_mib: float,
) -> pd.DataFrame:
    if set(states) != set(K_VALUES) or set(selections) != set(K_VALUES):
        raise ValueError(f"states and selections must contain exactly K={K_VALUES}")
    baseline = states[3].copy().set_index("id")
    baseline_ids = baseline.index.astype(str).tolist()
    baseline_top = list(selections[3]["top_b_ids"])
    baseline_full = list(selections[3]["full_ids"])
    rows = []
    for k in K_VALUES:
        state = states[k].copy().set_index("id")
        if state.index.astype(str).tolist() != baseline_ids:
            raise ValueError("candidate IDs or order differ across MC states")
        mean_difference = np.abs(
            state["mean_ev"].to_numpy(dtype=float) - baseline["mean_ev"].to_numpy(dtype=float)
        )
        sd_difference = np.abs(
            state["sd_ev"].to_numpy(dtype=float) - baseline["sd_ev"].to_numpy(dtype=float)
        )
        top_ids = list(selections[k]["top_b_ids"])
        full_ids = list(selections[k]["full_ids"])
        rows.append(
            {
                "mc_passes": k,
                "candidate_count": len(state),
                "mean_predictive_mean_ev": float(state["mean_ev"].mean()),
                "mean_predictive_sd_ev": float(state["sd_ev"].mean()),
                "mean_abs_predictive_mean_difference_ev_vs_k3": float(mean_difference.mean()),
                "max_abs_predictive_mean_difference_ev_vs_k3": float(mean_difference.max()),
                "mean_abs_predictive_sd_difference_ev_vs_k3": float(sd_difference.mean()),
                "max_abs_predictive_sd_difference_ev_vs_k3": float(sd_difference.max()),
                "predictive_mean_rank_spearman_vs_k3": _rank_correlation(
                    state["mean_ev"].to_numpy(dtype=float), baseline["mean_ev"].to_numpy(dtype=float)
                ),
                "uncertainty_rank_spearman_vs_k3": _rank_correlation(
                    state["sd_ev"].to_numpy(dtype=float), baseline["sd_ev"].to_numpy(dtype=float)
                ),
                "top_b_overlap_fraction_vs_k3": _overlap(top_ids, baseline_top),
                "full_batch_overlap_fraction_vs_k3": _overlap(full_ids, baseline_full),
                "gate_route": str(selections[k]["route"]),
                "gate_flip_vs_k3": str(selections[k]["route"]) != str(selections[3]["route"]),
                "gate_margin_score": float(selections[k]["margin"]),
                "gate_group_concentration": float(selections[k]["concentration"]),
                "top_b_candidate_ids": "|".join(top_ids),
                "top_b_sequence_sha256": _sequence_hash(top_ids),
                "full_batch_candidate_ids": "|".join(full_ids),
                "full_batch_sequence_sha256": _sequence_hash(full_ids),
                "embedding_runtime_seconds_shared": float(embedding_runtime_seconds),
                "mc_forward_runtime_seconds": float(runtimes[k]),
                "total_runtime_seconds_with_shared_embedding": float(
                    embedding_runtime_seconds + runtimes[k]
                ),
                "embedding_peak_allocated_mib": float(embedding_peak_memory_mib),
                "mc_peak_allocated_mib": float(peak_memory_mib[k]),
                "peak_allocated_mib": float(
                    max(embedding_peak_memory_mib, peak_memory_mib[k])
                ),
                "mask_seed_first": mc_mask_seeds(k)[0],
                "mask_seed_last": mc_mask_seeds(k)[-1],
            }
        )
    return pd.DataFrame(rows)


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


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _extract_embedding_batches(support, checkpoint: Path, pool: Path):
    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_started = time.perf_counter()
    model = support.load_model(str(checkpoint), device)
    _synchronize()
    model_load_seconds = time.perf_counter() - load_started
    dataset = support.CIFData(str(pool))
    loader = DataLoader(dataset, batch_size=32, collate_fn=support.collate_pool, shuffle=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _synchronize()
    started = time.perf_counter()
    batches: list[tuple[list[str], torch.Tensor]] = []
    with torch.no_grad():
        for batch in loader:
            (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), _, batch_ids = batch
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [index.to(device) for index in crystal_atom_idx]
            embedding = support.penultimate_forward(
                model, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx
            ).detach()
            batches.append(
                ([support.clean_id(value) for value in batch_ids], embedding.cpu())
            )
    _synchronize()
    elapsed = time.perf_counter() - started
    peak_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    )
    return model, device, batches, model_load_seconds, elapsed, peak_mib


def _run_mc_prefix(model, device, batches, k: int, normalizer_mean: float, normalizer_std: float):
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _synchronize()
    started = time.perf_counter()
    rows: list[dict[str, float | str]] = []
    keep_probability = 0.70
    with torch.no_grad():
        for batch_ids, embedding_cpu in batches:
            embedding = embedding_cpu.to(device)
            predictions = []
            for mask_seed in mc_mask_seeds(k):
                torch.manual_seed(mask_seed)
                mask = (torch.rand_like(embedding) < keep_probability).float() / keep_probability
                predictions.append(model.fc_out(embedding * mask).view(-1).detach().cpu().numpy())
            matrix = np.vstack(predictions)
            means = np.mean(matrix, axis=0)
            standard_deviations = np.std(matrix, axis=0, ddof=0)
            for candidate_id, mean_normalized, sd_normalized in zip(
                batch_ids, means, standard_deviations, strict=True
            ):
                rows.append(
                    {
                        "id": candidate_id,
                        "mean_normalized": float(mean_normalized),
                        "sd_normalized": float(sd_normalized),
                        "mean_ev": float(mean_normalized * normalizer_std + normalizer_mean),
                        "sd_ev": float(sd_normalized * abs(normalizer_std)),
                    }
                )
    _synchronize()
    elapsed = time.perf_counter() - started
    peak_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
    )
    return pd.DataFrame(rows), elapsed, peak_mib


def _build_or_load_states(
    formal,
    support,
    checkpoint: Path,
    pool: Path,
    prediction_path: Path,
    cache_path: Path,
    inference_seed: int,
):
    input_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint),
        "pool_id_prop_sha256": sha256_file(pool / "id_prop.csv"),
        "prediction_sha256": sha256_file(prediction_path),
    }
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        for key, expected in input_hashes.items():
            if str(cached[key].item()) != expected:
                raise RuntimeError(f"MC microbenchmark cache input mismatch for {key}")
        ids = cached["ids"].astype(str).tolist()
        states = {
            k: pd.DataFrame(
                {
                    "id": ids,
                    "mean_normalized": cached[f"mean_normalized_k{k}"].astype(float),
                    "sd_normalized": cached[f"sd_normalized_k{k}"].astype(float),
                    "mean_ev": cached[f"mean_ev_k{k}"].astype(float),
                    "sd_ev": cached[f"sd_ev_k{k}"].astype(float),
                }
            )
            for k in K_VALUES
        }
        runtimes = {k: float(cached[f"runtime_k{k}"].item()) for k in K_VALUES}
        peaks = {k: float(cached[f"peak_mib_k{k}"].item()) for k in K_VALUES}
        features = cached["normalized_features"].astype(float)
        groups = cached["groups"].astype(str).tolist()
        return (
            states,
            runtimes,
            peaks,
            features,
            groups,
            float(cached["model_load_seconds"].item()),
            float(cached["embedding_runtime_seconds"].item()),
            float(cached["embedding_peak_mib"].item()),
            float(cached["normalizer_mean"].item()),
            float(cached["normalizer_std"].item()),
            str(cached["device"].item()),
            str(cached["gpu_name"].item()),
        )

    _set_seed(inference_seed)
    import torch

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalizer_mean = float(checkpoint_data["normalizer"]["mean"].item())
    normalizer_std = float(checkpoint_data["normalizer"]["std"].item())
    model, device, batches, model_load_seconds, embedding_seconds, embedding_peak = (
        _extract_embedding_batches(support, checkpoint, pool)
    )
    feature_by_id = {
        candidate_id: vector.numpy().astype(float)
        for batch_ids, embedding in batches
        for candidate_id, vector in zip(batch_ids, embedding, strict=True)
    }
    prediction = pd.read_csv(
        prediction_path,
        header=None,
        names=["id", "actual_feature_ignored", "pseudo_feature"],
        dtype={"id": str},
    )
    ids = prediction["id"].map(formal.clean_id).tolist()
    groups = [formal.group_key(candidate_id, str(pool)) for candidate_id in ids]
    normalized_features, similarity_mode = formal.choose_similarity(ids, feature_by_id, groups)
    if similarity_mode != "cgcnn_embedding":
        raise RuntimeError(f"expected CGCNN embeddings, observed {similarity_mode}")
    states = {}
    runtimes = {}
    peaks = {}
    for k in K_VALUES:
        state, runtime, peak = _run_mc_prefix(
            model, device, batches, k, normalizer_mean, normalizer_std
        )
        state = state.set_index("id").loc[ids].reset_index()
        states[k] = state
        runtimes[k] = runtime
        peaks[k] = peak
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    payload: dict[str, object] = {
        **{key: np.asarray(value) for key, value in input_hashes.items()},
        "ids": np.asarray(ids, dtype=str),
        "normalized_features": np.asarray(normalized_features, dtype=float),
        "groups": np.asarray(groups, dtype=str),
        "model_load_seconds": np.asarray(model_load_seconds),
        "embedding_runtime_seconds": np.asarray(embedding_seconds),
        "embedding_peak_mib": np.asarray(embedding_peak),
        "normalizer_mean": np.asarray(normalizer_mean),
        "normalizer_std": np.asarray(normalizer_std),
        "device": np.asarray(str(device)),
        "gpu_name": np.asarray(gpu_name),
    }
    for k in K_VALUES:
        for column in ("mean_normalized", "sd_normalized", "mean_ev", "sd_ev"):
            payload[f"{column}_k{k}"] = states[k][column].to_numpy(dtype=float)
        payload[f"runtime_k{k}"] = np.asarray(runtimes[k])
        payload[f"peak_mib_k{k}"] = np.asarray(peaks[k])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return (
        states,
        runtimes,
        peaks,
        normalized_features,
        groups,
        model_load_seconds,
        embedding_seconds,
        embedding_peak,
        normalizer_mean,
        normalizer_std,
        str(device),
        gpu_name,
    )


def _render_report(
    result: pd.DataFrame,
    *,
    checkpoint: Path,
    pool: Path,
    prediction_path: Path,
    inference_seed: int,
    model_load_seconds: float,
    normalizer_mean: float,
    normalizer_std: float,
    device: str,
    gpu_name: str,
    formal_consistency_max_abs_sd: float,
    formal_config_path: Path,
) -> str:
    display = result[
        [
            "mc_passes",
            "mean_abs_predictive_mean_difference_ev_vs_k3",
            "mean_abs_predictive_sd_difference_ev_vs_k3",
            "uncertainty_rank_spearman_vs_k3",
            "top_b_overlap_fraction_vs_k3",
            "full_batch_overlap_fraction_vs_k3",
            "gate_route",
            "gate_flip_vs_k3",
            "mc_forward_runtime_seconds",
            "peak_allocated_mib",
        ]
    ]
    lines = [
        "# MC-dropout Microbenchmark",
        "",
        "## Scope",
        "",
        "This is a single-round Li–M–O microbenchmark on development seed 0. K=3, 10 and 30 reuse one frozen checkpoint, one deterministic prediction, one candidate pool and one shared embedding extraction. It does not train, update the pool, close the active-learning loop, compute AUTC, or use seeds 5–14 for tuning.",
        "",
        "Gate parameters are copied from the retained Li–M–O corrected-seed run configuration (M0=1.0, G0=0.50, alpha=0.10, beta=0.20, gamma=0.10); only K is varied.",
        "",
        "## Results",
        "",
        display.to_markdown(index=False),
        "",
        "Predictive mean and SD differences are candidate-wise absolute differences relative to K=3 after applying the checkpoint normalizer, in eV atom⁻¹. Rank correlation is Spearman correlation of predictive SD. Top-b and full-batch overlaps use 16 candidates as the denominator.",
        "",
        "The retained gate uses the deterministic conventional prediction as its mean; the MC predictive mean is reported here only as a sensitivity diagnostic. K-specific SD is the quantity that enters the retained interval-hit score.",
        "",
        "## Unit audit note",
        "",
        "The retained selector computes dropout SD directly from `model.fc_out` without applying the checkpoint normalizer, while its deterministic pseudo prediction and target interval are in eV atom⁻¹. Therefore the retained gate combines normalized-output SD with physical-unit means/thresholds. This microbenchmark preserves that behavior for exact route/batch comparability and separately reports denormalized mean/SD. The unit mismatch is a methodological issue that must be resolved before a full K-sensitivity experiment is interpreted.",
        "",
        "## Runtime protocol",
        "",
        "The expensive crystal-graph embedding is measured once and shared. Each K row times only its nested manual-mask `fc_out` forwards; `total_runtime_seconds_with_shared_embedding` in the CSV adds the shared embedding time. Peak allocated memory is PyTorch allocated memory, not total process or driver memory.",
        "",
        f"- Model load: {model_load_seconds:.6f} s",
        f"- Device: `{device}`",
        f"- GPU: `{gpu_name}`",
        f"- Checkpoint normalizer mean/std: `{normalizer_mean:.12g}` / `{normalizer_std:.12g}`",
        f"- K=3 formal-extractor SD consistency max absolute difference: `{formal_consistency_max_abs_sd:.12g}` normalized-output units",
        "",
        "## Seed and provenance",
        "",
        f"- Development seed: `0`; inference seed: `{inference_seed}`",
        "- Nested manual mask seeds: K=3 uses 1000003–1000005; K=10 extends through 1000012; K=30 extends through 1000032.",
        f"- Checkpoint: `{checkpoint.resolve()}` / `{sha256_file(checkpoint)}`",
        f"- Pool id_prop: `{(pool / 'id_prop.csv').resolve()}` / `{sha256_file(pool / 'id_prop.csv')}`",
        f"- Shared prediction: `{prediction_path.resolve()}` / `{sha256_file(prediction_path)}`",
        f"- Retained Li–M–O formal run config: `{formal_config_path.resolve()}` / `{sha256_file(formal_config_path)}`",
        f"- Analysis script: `{Path(__file__).resolve()}` / `{sha256_file(Path(__file__).resolve())}`",
        "",
        "## Interpretation boundary",
        "",
        "This microbenchmark can identify immediate rank, batch or route instability, but it cannot establish closed-loop AUTC/recovery robustness. A full experiment remains pending authorization after the unit-handling decision is fixed and documented.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--historical-root", type=Path, default=Path(r"D:\CGCNN"))
    parser.add_argument("--development-seed", type=int, default=0)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args(argv)
    if args.development_seed in range(5, 15):
        raise ValueError("seeds 5–14 are reserved for formal evaluation, not this microbenchmark")
    if args.development_seed != 0:
        raise ValueError("this frozen microbenchmark protocol is defined only for development seed 0")

    archive = args.archive.resolve()
    historical_root = args.historical_root.resolve()
    checkpoint = historical_root / "checkpoint_formation_clean.pth.tar"
    pool = historical_root / "EARLY_LiM2O4_ALIGNN_HARD640_m2p18_m2p02_FLAT_20260617"
    prediction_path = archive / "tmp/gate_ablation_smoke_seed0/test_results.csv"
    formal_path = archive / "active_learning_energy_gate_ablation.py"
    support_root = archive / "experiments/reproducibility/staging/paired_confirmation_server_20260712"
    formal_config_path = (
        archive
        / "baseline_snapshot/archive/experiments/reproducibility/results"
        / "paired_two_dataset_confirmation_20260712/runs/limo"
        / "energy_gated_da_tpp/seed_5/run_config.json"
    )
    for required in (
        checkpoint,
        pool / "id_prop.csv",
        prediction_path,
        formal_path,
        formal_config_path,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    formal = load_formal_gate_module(formal_path, support_root)
    support = loaded_formal_support_module()
    inference_seed = 100_001
    cache_path = archive / "tmp/mc_dropout_microbenchmark_seed0/microbenchmark_state.npz"
    (
        states,
        runtimes,
        peaks,
        normalized_features,
        groups,
        model_load_seconds,
        embedding_seconds,
        embedding_peak,
        normalizer_mean,
        normalizer_std,
        device,
        gpu_name,
    ) = _build_or_load_states(
        formal, support, checkpoint, pool, prediction_path, cache_path, inference_seed
    )
    prediction = pd.read_csv(
        prediction_path,
        header=None,
        names=["id", "actual_feature_ignored", "pseudo_feature"],
        dtype={"id": str},
    )
    prediction["id"] = prediction["id"].map(formal.clean_id)
    prediction["pseudo_feature"] = pd.to_numeric(prediction["pseudo_feature"], errors="raise")
    ids = prediction["id"].tolist()
    selections: dict[int, dict[str, object]] = {}
    for k in K_VALUES:
        aligned = states[k].set_index("id").loc[ids]
        evaluated = evaluate_gate_methods(
            ids=ids,
            pseudo=prediction["pseudo_feature"].to_numpy(dtype=float),
            raw_sigma=aligned["sd_normalized"].to_numpy(dtype=float),
            groups=groups,
            similarity=normalized_features,
            target_low=-2.18,
            target_high=-2.02,
            batch_size=16,
            margin_threshold=LIMO_FORMAL_PARAMS["M0"],
            concentration_threshold=LIMO_FORMAL_PARAMS["G0"],
            alpha=LIMO_FORMAL_PARAMS["alpha"],
            beta=LIMO_FORMAL_PARAMS["beta"],
            gamma=LIMO_FORMAL_PARAMS["gamma"],
        )
        greedy = evaluated[evaluated["method"] == "Interval-Hit Greedy"].iloc[0]
        full = evaluated[evaluated["method"] == "Full Energy-Gated DA-TPP"].iloc[0]
        selections[k] = {
            "top_b_ids": str(greedy["selected_candidate_ids"]).split("|"),
            "full_ids": str(full["selected_candidate_ids"]).split("|"),
            "route": str(full["route"]),
            "margin": float(full["margin_score"]),
            "concentration": float(full["group_concentration_top_b"]),
        }
    result = summarize_mc_sensitivity(
        states,
        selections,
        runtimes=runtimes,
        peak_memory_mib=peaks,
        embedding_runtime_seconds=embedding_seconds,
        embedding_peak_memory_mib=embedding_peak,
    )
    result.insert(0, "dataset", "Li-M-O")
    result.insert(1, "development_seed", 0)
    result.insert(2, "round", 1)
    result["inference_seed"] = inference_seed
    result["dropout_rate"] = 0.30
    result["M0"] = LIMO_FORMAL_PARAMS["M0"]
    result["G0"] = LIMO_FORMAL_PARAMS["G0"]
    result["alpha"] = LIMO_FORMAL_PARAMS["alpha"]
    result["beta"] = LIMO_FORMAL_PARAMS["beta"]
    result["gamma"] = LIMO_FORMAL_PARAMS["gamma"]
    result["model_load_seconds"] = model_load_seconds
    result["device"] = device
    result["gpu_name"] = gpu_name
    result["normalizer_mean"] = normalizer_mean
    result["normalizer_std"] = normalizer_std

    smoke_cache = archive / "tmp/gate_ablation_smoke_seed0/score_state.npz"
    if not smoke_cache.is_file():
        raise FileNotFoundError(smoke_cache)
    retained = np.load(smoke_cache, allow_pickle=False)
    retained_ids = retained["ids"].astype(str).tolist()
    retained_sigma = pd.Series(retained["raw_sigma"].astype(float), index=retained_ids).loc[ids]
    k3_sigma = states[3].set_index("id").loc[ids, "sd_normalized"]
    formal_consistency = float(np.max(np.abs(retained_sigma.to_numpy() - k3_sigma.to_numpy())))
    if formal_consistency > 1e-6:
        raise RuntimeError(
            f"K=3 implementation differs from retained formal extractor by {formal_consistency}"
        )
    report = _render_report(
        result,
        checkpoint=checkpoint,
        pool=pool,
        prediction_path=prediction_path,
        inference_seed=inference_seed,
        model_load_seconds=model_load_seconds,
        normalizer_mean=normalizer_mean,
        normalizer_std=normalizer_std,
        device=device,
        gpu_name=gpu_name,
        formal_consistency_max_abs_sd=formal_consistency,
        formal_config_path=formal_config_path,
    )
    csv_buffer = io.StringIO()
    result.to_csv(csv_buffer, index=False, lineterminator="\n")
    outputs = {
        archive / "results/mc_dropout/microbenchmark.csv": csv_buffer.getvalue().encode("utf-8"),
        archive / "docs/MC_DROPOUT_MICROBENCHMARK.md": report.encode("utf-8"),
    }
    statuses = {
        str(path.relative_to(archive)): write_bytes_protected(path, content, args.check_existing)
        for path, content in outputs.items()
    }
    print(json.dumps(statuses, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
