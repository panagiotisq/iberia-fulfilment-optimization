"""Glue between forecasting and planning: per-day context and scenario sets."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .forecast.scenarios import fit_copula, pit_scores, sample_scenarios
from .planning.model import position_inventory
from .world import world

COPULA_WINDOW = 120

# Event days of the test year (always included) + two random days per month
EVENT_DAYS = ["2025-10-07", "2025-11-28", "2025-12-01", "2025-12-16", "2026-01-02", "2026-07-14", "2026-08-12"]


def test_days(n_per_month: int = 2, seed: int = 3) -> list[pd.Timestamp]:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2025-10-01", "2026-09-30", freq="D")
    ev = set(pd.to_datetime(EVENT_DAYS))
    out = list(ev)
    for _, grp in pd.Series(days, index=days).groupby(days.to_period("M")):
        cand = [d for d in grp if d not in ev]
        out += list(rng.choice(cand, n_per_month, replace=False))
    return sorted(pd.to_datetime(out))


@lru_cache(maxsize=None)
def forecasts():
    f = np.load(config.CACHE / "forecasts.npz")
    return {k: f[k] for k in f.files}


@lru_cache(maxsize=None)
def _pit(method: str):
    net, hist = world()
    f = forecasts()
    Y = hist.y.reshape(hist.T, -1)
    o = f["origins"]
    return pit_scores(f[method][:, 0].astype(float), Y[o + 1])


def day_quantiles(method: str, t: int) -> np.ndarray:
    """h=1 quantile forecast for target day t (made at origin t-1), (S, nQ)."""
    f = forecasts()
    i = int(np.searchsorted(f["origins"], t - 1))
    assert f["origins"][i] == t - 1
    return f[method][i, 0].astype(float)


def day_copula(method: str, t: int) -> np.ndarray:
    f = forecasts()
    o = f["origins"]
    m = (o + 1 <= t - 1) & (o + 1 > t - 1 - COPULA_WINDOW)
    return fit_copula(_pit(method)[m])


def day_inventory(t: int, net=None) -> np.ndarray:
    _net, hist = world()
    net = net or _net
    return position_inventory(net, hist.oracle_mean(t))


def scenarios(method: str, t: int, n: int, seed: int = 0, copula: bool = True) -> np.ndarray:
    """Demand scenarios (n, J, K) for target day t from a forecaster ('oracle' = true distribution)."""
    net, hist = world()
    rng = np.random.default_rng(seed)
    if method == "oracle":
        return hist.oracle_samples(t, n, rng)
    q = day_quantiles(method, t)
    corr = day_copula(method, t) if copula else None
    return sample_scenarios(q, n, rng, corr).reshape(n, net.J, net.K)


def point_forecast(method: str, t: int, level: float = 0.5) -> np.ndarray:
    net, _ = world()
    f = forecasts()
    a = int(np.argmin(np.abs(f["quantiles"] - level)))
    return day_quantiles(method, t)[:, a].reshape(net.J, net.K)
