"""Point and probabilistic forecast accuracy metrics (vectorised over a trailing quantile axis)."""
from __future__ import annotations

import numpy as np


def pinball(q: np.ndarray, y: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Quantile (pinball) loss, shape of q; y broadcast against q[..., 0]."""
    d = y[..., None] - q
    return np.maximum(levels * d, (levels - 1) * d)


def scaled_quantile_loss(q, y, levels) -> float:
    """sum pinball / sum y, averaged over levels; 2 x this approximates scaled CRPS."""
    return float(pinball(q, y, levels).sum() / y.sum() / len(levels))


def wmape(point, y) -> float:
    return float(np.abs(point - y).sum() / y.sum())


def bias(point, y) -> float:
    return float((point - y).sum() / y.sum())


def coverage(q, y, levels, lo: float, hi: float) -> float:
    a, b = int(np.argmin(np.abs(levels - lo))), int(np.argmin(np.abs(levels - hi)))
    return float(((y >= q[..., a]) & (y <= q[..., b])).mean())


def interval_width(q, y, levels, lo, hi) -> float:
    a, b = int(np.argmin(np.abs(levels - lo))), int(np.argmin(np.abs(levels - hi)))
    return float((q[..., b] - q[..., a]).sum() / y.sum())


def quantile_hits(q, y) -> np.ndarray:
    """Empirical P(y <= q_a) per level (reliability diagram)."""
    return (y[..., None] <= q).reshape(-1, q.shape[-1]).mean(axis=0)


def energy_score(samples: np.ndarray, y: np.ndarray, n_pairs: int = 2000, rng=None) -> float:
    """Multivariate energy score (Gneiting & Raftery 2007) of scenario set `samples` (N,d) for y (d,).

    ES = E||X - y|| - 0.5 E||X - X'||, estimated with all N draws for the first term and
    n_pairs random pairs for the second. Lower is better.
    """
    rng = rng or np.random.default_rng(0)
    t1 = np.linalg.norm(samples - y[None, :], axis=1).mean()
    a = rng.integers(0, len(samples), n_pairs)
    b = rng.integers(0, len(samples), n_pairs)
    t2 = np.linalg.norm(samples[a] - samples[b], axis=1).mean()
    return float(t1 - 0.5 * t2)


def variogram_score(samples: np.ndarray, y: np.ndarray, p: float = 0.5, w: np.ndarray | None = None) -> float:
    """Variogram score of order p (Scheuerer & Hamill 2015): sensitive to the dependence structure."""
    d = samples.shape[1]
    Ey = np.abs(y[:, None] - y[None, :]) ** p
    Ex = (np.abs(samples[:, :, None] - samples[:, None, :]) ** p).mean(axis=0)
    w = np.ones((d, d)) if w is None else w
    return float((w * (Ey - Ex) ** 2).sum())
