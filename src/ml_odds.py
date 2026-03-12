"""Utilities for model-implied moneyline probabilities and fair odds."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

MlSigmaMode = Literal["raw", "cap14", "cap17", "const14"]


def normal_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF using erf."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))
    x_arr = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x_arr / math.sqrt(2.0)))


def stabilize_sigma_for_ml(
    sigma: np.ndarray | float,
    mode: MlSigmaMode = "cap14",
) -> np.ndarray | float:
    """Apply the chosen sigma stabilization for ML odds."""
    sigma_arr = np.asarray(sigma, dtype=float)
    if mode == "raw":
        out = sigma_arr
    elif mode == "cap14":
        out = np.minimum(sigma_arr, 14.0)
    elif mode == "cap17":
        out = np.minimum(sigma_arr, 17.0)
    elif mode == "const14":
        out = np.full_like(sigma_arr, 14.0, dtype=float)
    else:
        raise ValueError(f"Unsupported ML sigma mode: {mode}")
    out = np.maximum(out, 0.5)
    if np.isscalar(sigma):
        return float(out)
    return out


def mu_sigma_home_win_prob(
    mu_home: np.ndarray | float,
    sigma: np.ndarray | float,
    *,
    sigma_mode: MlSigmaMode = "cap14",
) -> np.ndarray | float:
    """Convert margin mean/sigma to home win probability."""
    sigma_used = stabilize_sigma_for_ml(sigma, mode=sigma_mode)
    z = np.asarray(mu_home, dtype=float) / np.asarray(sigma_used, dtype=float)
    p = normal_cdf(z)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if np.isscalar(mu_home) and np.isscalar(sigma):
        return float(p)
    return p


def fair_american_odds(prob: float | np.ndarray) -> float | np.ndarray:
    """Convert probability to fair American odds."""
    p = np.clip(np.asarray(prob, dtype=float), 1e-9, 1 - 1e-9)
    out = np.full_like(p, np.nan, dtype=float)
    fav = p >= 0.5
    dog = ~fav
    out[fav] = -100.0 * (p[fav] / (1.0 - p[fav]))
    out[dog] = 100.0 * ((1.0 - p[dog]) / p[dog])
    if np.isscalar(prob):
        return float(out)
    return out
