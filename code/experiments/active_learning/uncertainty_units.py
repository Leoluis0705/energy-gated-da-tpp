"""Physical-unit handling for Energy-Gated DA-TPP predictive moments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


MomentSpace = Literal["normalized", "ev_per_atom"]


@dataclass(frozen=True)
class PredictiveMoments:
    """Predictive mean and standard deviation with an explicit unit space."""

    mean: np.ndarray
    sigma: np.ndarray
    space: MomentSpace

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float)
        sigma = np.asarray(self.sigma, dtype=float)
        if mean.shape != sigma.shape:
            raise ValueError("predictive mean and sigma must have the same shape")
        if self.space not in ("normalized", "ev_per_atom"):
            raise ValueError(f"unsupported predictive-moment space: {self.space}")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sigma)):
            raise ValueError("predictive mean and sigma must be finite")
        if np.any(sigma < 0.0):
            raise ValueError("predictive sigma must be non-negative")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "sigma", sigma)

    def to_ev(
        self,
        *,
        location: float,
        scale: float,
        epsilon: float = 1e-12,
    ) -> "PredictiveMoments":
        """Return moments in eV atom^-1 without applying conversion twice."""

        if self.space == "ev_per_atom":
            return self
        location = float(location)
        scale = float(scale)
        epsilon = float(epsilon)
        if not math.isfinite(location):
            raise ValueError("normalizer location must be finite")
        if not math.isfinite(scale) or abs(scale) <= epsilon:
            raise ValueError("normalizer scale must be finite and larger than epsilon")
        return PredictiveMoments(
            mean=self.mean * scale + location,
            sigma=self.sigma * abs(scale),
            space="ev_per_atom",
        )


def _normal_cdf(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=float).ravel()
    result = np.fromiter(
        (0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0))) for value in flat),
        dtype=float,
        count=flat.size,
    )
    return result.reshape(np.asarray(values).shape)


def interval_hit_probability_ev(
    mean_ev: np.ndarray,
    sigma_ev: np.ndarray,
    low_ev: float,
    high_ev: float,
    *,
    sigma_floor_ev: float = 0.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Calculate Gaussian interval-hit probabilities wholly in eV atom^-1."""

    mean = np.asarray(mean_ev, dtype=float)
    sigma = np.asarray(sigma_ev, dtype=float)
    if mean.shape != sigma.shape:
        raise ValueError("predictive mean and sigma must have the same shape")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(sigma)):
        raise ValueError("predictive mean and sigma must be finite")
    if np.any(sigma < 0.0):
        raise ValueError("predictive sigma must be non-negative")
    low = float(low_ev)
    high = float(high_ev)
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("target interval must be finite and ordered low <= high")
    floor = float(sigma_floor_ev)
    epsilon = float(epsilon)
    if not math.isfinite(floor) or floor < 0.0:
        raise ValueError("sigma floor must be finite and non-negative")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    effective_sigma = np.maximum(sigma, floor)
    deterministic = effective_sigma <= epsilon
    probability = np.empty(mean.shape, dtype=float)
    probability[deterministic] = (
        (mean[deterministic] >= low) & (mean[deterministic] <= high)
    ).astype(float)
    stochastic = ~deterministic
    if np.any(stochastic):
        stochastic_mean = mean[stochastic]
        stochastic_sigma = effective_sigma[stochastic]
        probability[stochastic] = _normal_cdf(
            (high - stochastic_mean) / stochastic_sigma
        ) - _normal_cdf((low - stochastic_mean) / stochastic_sigma)
    return np.clip(probability, 0.0, 1.0)
