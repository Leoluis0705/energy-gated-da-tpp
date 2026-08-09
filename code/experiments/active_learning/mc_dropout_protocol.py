"""Corrected physical-unit, paired-seed MC-dropout inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mc_dropout_seed_policy import (
    SEED_POLICY_VERSION,
    mask_sequence_sha256,
    mc_mask_seeds,
)
from uncertainty_units import PredictiveMoments, interval_hit_probability_ev


@dataclass(frozen=True)
class SelectorUncertaintyState:
    mean_ev: np.ndarray
    sigma_normalized: np.ndarray
    sigma_ev: np.ndarray
    interval_hit_probability: np.ndarray
    mean_input_space: str = "ev_per_atom"
    sigma_input_space: str = "normalized"


@dataclass(frozen=True)
class MCDropoutExtraction:
    features: dict[str, np.ndarray]
    mean_normalized: dict[str, float]
    sigma_normalized: dict[str, float]
    mean_ev: dict[str, float]
    sigma_ev: dict[str, float]
    normalizer_location: float
    normalizer_scale: float
    mask_seeds: tuple[int, ...]
    mask_sequence_sha256: str
    seed_policy_version: str
    warnings: tuple[str, ...]


def prepare_selector_uncertainty(
    *,
    deterministic_mean_ev: np.ndarray,
    mc_sigma_normalized: np.ndarray,
    normalizer_location: float,
    normalizer_scale: float,
    target_low_ev: float,
    target_high_ev: float,
    sigma_floor_ev: float,
) -> SelectorUncertaintyState:
    """Combine an already-physical deterministic mean with corrected MC sigma."""

    mean_ev = np.asarray(deterministic_mean_ev, dtype=float)
    sigma_normalized = np.asarray(mc_sigma_normalized, dtype=float)
    if mean_ev.shape != sigma_normalized.shape:
        raise ValueError("deterministic mean and MC sigma must have the same shape")
    sigma_conversion = PredictiveMoments(
        mean=np.zeros_like(sigma_normalized),
        sigma=sigma_normalized,
        space="normalized",
    ).to_ev(location=normalizer_location, scale=normalizer_scale)
    physical = PredictiveMoments(
        mean=mean_ev,
        sigma=sigma_conversion.sigma,
        space="ev_per_atom",
    )
    probability = interval_hit_probability_ev(
        physical.mean,
        physical.sigma,
        target_low_ev,
        target_high_ev,
        sigma_floor_ev=sigma_floor_ev,
    )
    return SelectorUncertaintyState(
        mean_ev=physical.mean,
        sigma_normalized=sigma_normalized,
        sigma_ev=physical.sigma,
        interval_hit_probability=probability,
    )


def _checkpoint_normalizer(checkpoint: dict[str, Any]) -> tuple[float, float]:
    normalizer = checkpoint.get("normalizer")
    if not isinstance(normalizer, dict) or "mean" not in normalizer or "std" not in normalizer:
        raise ValueError("checkpoint is missing normalizer mean/std")

    def scalar(value: Any) -> float:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        return float(value)

    location = scalar(normalizer["mean"])
    scale = scalar(normalizer["std"])
    PredictiveMoments(
        mean=np.array([0.0]),
        sigma=np.array([0.0]),
        space="normalized",
    ).to_ev(location=location, scale=scale)
    return location, scale


def extract_seeded_mc_dropout(
    *,
    model_path: str | Path,
    candidates_dir: str | Path,
    mc_passes: int,
    dropout_rate: float,
    experiment_seed: int,
    acquisition_round: int,
    model_refit_index: int,
) -> MCDropoutExtraction:
    """Extract embeddings and MC moments without modifying the legacy support file."""

    import torch
    from torch.utils.data import DataLoader

    from active_learning_etdg_tage import (
        CIFData,
        clean_id,
        collate_pool,
        load_model,
        penultimate_forward,
    )

    rate = float(dropout_rate)
    if not 0.0 <= rate < 1.0:
        raise ValueError("dropout_rate must satisfy 0 <= rate < 1")
    keep_probability = 1.0 - rate
    seeds = mc_mask_seeds(
        mc_passes,
        experiment_seed=experiment_seed,
        acquisition_round=acquisition_round,
        model_refit_index=model_refit_index,
    )
    sequence_hash = mask_sequence_sha256(
        mc_passes,
        experiment_seed=experiment_seed,
        acquisition_round=acquisition_round,
        model_refit_index=model_refit_index,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    location, scale = _checkpoint_normalizer(checkpoint)
    model = load_model(str(model_path), device)
    dataset = CIFData(str(candidates_dir))
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=collate_pool,
        shuffle=False,
        num_workers=0,
    )

    features: dict[str, np.ndarray] = {}
    mean_normalized: dict[str, float] = {}
    sigma_normalized: dict[str, float] = {}
    mean_ev: dict[str, float] = {}
    sigma_ev: dict[str, float] = {}
    warnings: list[str] = []

    with torch.no_grad():
        for batch in dataloader:
            (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), _, batch_ids = batch
            ids = [clean_id(value) for value in batch_ids]
            atom_fea = atom_fea.to(device)
            nbr_fea = nbr_fea.to(device)
            nbr_fea_idx = nbr_fea_idx.to(device)
            crystal_atom_idx = [index.to(device) for index in crystal_atom_idx]
            try:
                embedding = penultimate_forward(
                    model,
                    atom_fea,
                    nbr_fea,
                    nbr_fea_idx,
                    crystal_atom_idx,
                ).detach()
                for index, candidate_id in enumerate(ids):
                    features[candidate_id] = embedding[index].cpu().numpy().astype(float)
                predictions = []
                for mask_seed in seeds:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(mask_seed)
                    random_values = torch.rand(
                        embedding.shape,
                        dtype=embedding.dtype,
                        device=device,
                        generator=generator,
                    )
                    mask = (random_values < keep_probability).to(embedding.dtype) / keep_probability
                    predictions.append(model.fc_out(embedding * mask).view(-1).detach().cpu().numpy())
                matrix = np.vstack(predictions)
                batch_mean = np.mean(matrix, axis=0)
                batch_sigma = np.std(matrix, axis=0, ddof=0)
                physical = PredictiveMoments(
                    mean=batch_mean,
                    sigma=batch_sigma,
                    space="normalized",
                ).to_ev(location=location, scale=scale)
                for candidate_id, mu_norm, sd_norm, mu_ev, sd_ev in zip(
                    ids,
                    batch_mean,
                    batch_sigma,
                    physical.mean,
                    physical.sigma,
                    strict=True,
                ):
                    mean_normalized[candidate_id] = float(mu_norm)
                    sigma_normalized[candidate_id] = float(sd_norm)
                    mean_ev[candidate_id] = float(mu_ev)
                    sigma_ev[candidate_id] = float(sd_ev)
            except Exception as exc:
                warnings.append(f"seeded MC dropout failed for batch {ids[:3]}: {exc}")

    return MCDropoutExtraction(
        features=features,
        mean_normalized=mean_normalized,
        sigma_normalized=sigma_normalized,
        mean_ev=mean_ev,
        sigma_ev=sigma_ev,
        normalizer_location=location,
        normalizer_scale=scale,
        mask_seeds=tuple(seeds),
        mask_sequence_sha256=sequence_hash,
        seed_policy_version=SEED_POLICY_VERSION,
        warnings=tuple(warnings),
    )
