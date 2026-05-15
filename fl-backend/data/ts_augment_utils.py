"""Time-series-aware augmentations for windowed anomaly detection.

All helpers operate on arrays of shape ``(N, T, F)`` where N is the number of
synthetic windows, T the window length, and F the number of features. Inputs
are assumed standardized (e.g., StandardScaler-applied), so ``sigma`` values
are interpreted in units of feature standard deviations.
"""

from __future__ import annotations

import numpy as np


def jitter(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return X
    noise = rng.normal(0.0, sigma, size=X.shape).astype(X.dtype, copy=False)
    return X + noise


def scaling(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return X
    N, _, F = X.shape
    scale = rng.normal(1.0, sigma, size=(N, 1, F)).astype(X.dtype, copy=False)
    return X * scale


def magnitude_warp(
    X: np.ndarray,
    sigma: float,
    n_knots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if sigma <= 0 or n_knots <= 0:
        return X

    from scipy.interpolate import CubicSpline

    N, T, F = X.shape
    K = n_knots + 2
    knot_t = np.linspace(0, T - 1, K)
    knot_vals = 1.0 + rng.normal(0.0, sigma, size=(N, K, F))
    # Anchor both endpoints to 1.0 so the synthetic window matches the original
    # at the boundaries; only the interior knots vary.
    knot_vals[:, 0, :] = 1.0
    knot_vals[:, -1, :] = 1.0
    cs = CubicSpline(knot_t, knot_vals, axis=1)
    warp = cs(np.arange(T)).astype(X.dtype, copy=False)
    return X * warp
