"""From marginal quantile forecasts to joint demand scenarios (Gaussian copula).

Marginals: the quantile function of each series is interpolated piecewise-linearly in
normal-score space z = Phi^-1(alpha) through the forecast quantiles and extrapolated
linearly beyond the outermost levels (a locally Gaussian tail), truncated at zero.

Dependence: the probability integral transforms (PIT) of past outcomes under past
forecasts are mapped to normal scores; their correlation (Ledoit-Wolf shrinkage, since
there are 260 series and ~120 days) defines a Gaussian copula. Ignoring this dependence
(independent copula) makes national and lane-level totals far too certain, which is
exactly the quantity capacity planning is sensitive to.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.covariance import LedoitWolf

from .models import QUANTILES

Z_LEVELS = norm.ppf(QUANTILES)


def quantile_function(q: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate the interpolated quantile function.

    q: (..., nQ) forecast quantiles; z: (N, ...) normal scores (broadcast over leading N).
    Returns values with shape of z.
    """
    zl = Z_LEVELS
    nQ = len(zl)
    seg = np.clip(np.searchsorted(zl, z) - 1, 0, nQ - 2)          # segment index, end segments extrapolate
    q_lo = np.take_along_axis(np.broadcast_to(q, z.shape + (nQ,)), seg[..., None], -1)[..., 0]
    q_hi = np.take_along_axis(np.broadcast_to(q, z.shape + (nQ,)), seg[..., None] + 1, -1)[..., 0]
    z_lo, z_hi = zl[seg], zl[seg + 1]
    val = q_lo + (q_hi - q_lo) * (z - z_lo) / (z_hi - z_lo)
    return np.maximum(val, 0.0)


def pit_scores(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Normal score z such that quantile_function(q, z) = y (inverse of the above), per element.

    q: (n, S, nQ), y: (n, S). Points beyond the outer quantiles use the linear tail. Ties at
    zero (possible for tiny series) are resolved at the middle of the flat region.
    """
    zl = Z_LEVELS
    nQ = len(zl)
    # locate segment of y among the forecast quantiles
    seg = (y[..., None] > q).sum(-1) - 1
    seg = np.clip(seg, 0, nQ - 2)
    q_lo = np.take_along_axis(q, seg[..., None], -1)[..., 0]
    q_hi = np.take_along_axis(q, seg[..., None] + 1, -1)[..., 0]
    slope = np.maximum(q_hi - q_lo, 1e-9) / (zl[seg + 1] - zl[seg])
    z = zl[seg] + (y - q_lo) / slope
    return np.clip(z, -4.0, 4.0)


def fit_copula(Z: np.ndarray) -> np.ndarray:
    """Shrunk correlation matrix from a (n_obs, d) matrix of normal scores."""
    Z = Z[np.all(np.isfinite(Z), axis=1)]
    lw = LedoitWolf().fit(Z)
    C = lw.covariance_
    sd = np.sqrt(np.diag(C))
    C = C / np.outer(sd, sd)
    return C


def sample_scenarios(q: np.ndarray, n: int, rng: np.random.Generator, corr: np.ndarray | None = None,
                     antithetic: bool = False) -> np.ndarray:
    """Draw n joint scenarios from marginal quantiles q (S, nQ) and copula correlation corr (S, S).

    corr=None gives the independence copula. Returns (n, S).
    """
    S = q.shape[0]
    m = (n + 1) // 2 if antithetic else n
    eps = rng.standard_normal((m, S))
    if corr is not None:
        L = np.linalg.cholesky(corr + 1e-9 * np.eye(S))
        eps = eps @ L.T
    if antithetic:
        eps = np.vstack([eps, -eps])[:n]
    return quantile_function(q[None], eps)
