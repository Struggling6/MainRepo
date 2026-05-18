"""Time-series-aware augmentations for windowed anomaly detection.

All helpers operate on arrays of shape ``(N, T, F)`` where N is the number of
synthetic windows, T the window length, and F the number of features. Inputs
are assumed standardized (e.g., StandardScaler-applied), so ``sigma`` values
are interpreted in units of feature standard deviations.
"""

from __future__ import annotations

import numpy as np

# Hardcoded params for time_warp / window_slicing / mixup. No config knobs by design.
TSAUG_TIMEWARP_SIGMA = 0.2
TSAUG_TIMEWARP_KNOTS = 4
TSAUG_SLICE_RATIO    = 0.9
TSAUG_MIXUP_ALPHA    = 0.2


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


def time_warp(
    X: np.ndarray,
    sigma: float = TSAUG_TIMEWARP_SIGMA,
    n_knots: int = TSAUG_TIMEWARP_KNOTS,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if sigma <= 0 or n_knots <= 0:
        return X
    if rng is None:
        rng = np.random.default_rng()

    from scipy.interpolate import CubicSpline

    N, T, F = X.shape
    K = n_knots + 2
    knot_t = np.linspace(0, T - 1, K)
    # Random per-window time deltas with anchored endpoints; cumulative-sum yields
    # a monotone-ish warped index that we then rescale to [0, T-1].
    deltas = 1.0 + rng.normal(0.0, sigma, size=(N, K))
    deltas[:, 0] = 1.0
    deltas[:, -1] = 1.0
    warped_knots = np.cumsum(deltas, axis=1)
    warped_knots -= warped_knots[:, :1]
    warped_knots /= warped_knots[:, -1:] / (T - 1)

    out = np.empty_like(X)
    grid = np.arange(T)
    for i in range(N):
        cs = CubicSpline(knot_t, warped_knots[i])
        new_idx = np.clip(cs(grid), 0, T - 1)
        for f in range(F):
            out[i, :, f] = np.interp(new_idx, grid, X[i, :, f])
    return out


def window_slicing(
    X: np.ndarray,
    slice_ratio: float = TSAUG_SLICE_RATIO,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if not (0 < slice_ratio < 1):
        return X
    if rng is None:
        rng = np.random.default_rng()

    N, T, F = X.shape
    L = max(2, int(round(T * slice_ratio)))
    if L >= T:
        return X

    out = np.empty_like(X)
    starts = rng.integers(0, T - L + 1, size=N)
    src_grid = np.linspace(0.0, L - 1, T)
    base = np.arange(L)
    for i in range(N):
        s = int(starts[i])
        for f in range(F):
            out[i, :, f] = np.interp(src_grid, base, X[i, s:s + L, f])
    return out


def mixup(
    X: np.ndarray,
    alpha: float = TSAUG_MIXUP_ALPHA,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if alpha <= 0 or X.shape[0] < 2:
        return X
    if rng is None:
        rng = np.random.default_rng()

    N = X.shape[0]
    partner = rng.integers(0, N, size=N)
    # Symmetric Beta(alpha, alpha); keep lam closer to 1 so synth stays anchored
    # to the row's original anomaly shape.
    lam = rng.beta(alpha, alpha, size=N).astype(X.dtype, copy=False)
    lam = np.maximum(lam, 1.0 - lam)
    lam = lam[:, None, None]
    return lam * X + (1.0 - lam) * X[partner]
