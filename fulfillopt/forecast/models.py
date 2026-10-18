"""Probabilistic forecasters producing quantiles for every (origin, horizon, series).

All forecasters return an array Q[n_origin, n_horizon, S, n_quantile] in units.

* SeasonalNaive     y[t-7] with empirical multiplicative error quantiles (rolling, per series & horizon)
* LGBMQuantile      global LightGBM, one booster per quantile level, scale-free target
* conformalize()    rolling split-conformal recalibration of any quantile forecaster
* oracle_quantiles  true conditional distribution (Monte Carlo from the DGP)
"""
from __future__ import annotations

import numpy as np
import lightgbm as lgb

from ..demand import DemandHistory
from .features import HORIZONS, FeatureBuilder, feature_columns

QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

LGB_PARAMS = dict(
    objective="quantile", learning_rate=0.05, num_leaves=63, min_data_in_leaf=400,
    feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
    max_bin=255, verbose=-1, num_threads=0,
)
NUM_ROUNDS = 600
CATEGORICAL = ["zone", "cat", "dow", "h", "hol", "origin_hol", "month"]


def sort_quantiles(q: np.ndarray) -> np.ndarray:
    """Monotone rearrangement (Chernozhukov, Fernandez-Val & Galichon, 2010)."""
    return np.sort(q, axis=-1)


class SeasonalNaive:
    """y_hat[t] = y[t-7*ceil(h/7)]; quantiles from the trailing distribution of log(y/y_hat)."""

    name = "snaive"

    def __init__(self, hist: DemandHistory, window: int = 56):
        self.Y = hist.y.reshape(hist.T, -1)
        self.window = window

    def point(self, t, h):
        return self.Y[t - 7 * int(np.ceil(h / 7))]

    def predict(self, origins, horizons=HORIZONS, quantiles=QUANTILES):
        Y = np.maximum(self.Y, 1.0)
        out = np.empty((len(origins), len(horizons), Y.shape[1], len(quantiles)))
        for a, h in enumerate(horizons):
            lag = 7 * int(np.ceil(h / 7))
            for n, o in enumerate(origins):
                ts = np.arange(o - self.window + 1, o + 1)               # past target days, all known at o
                err = np.log(Y[ts] / Y[ts - lag])                        # (W,S)
                qe = np.quantile(err, quantiles, axis=0).T               # (S,Q)
                out[n, a] = Y[o + h - lag][:, None] * np.exp(qe)
        return out


class LGBMQuantile:
    name = "lgbm"

    def __init__(self, fb: FeatureBuilder, quantiles=QUANTILES, num_rounds=NUM_ROUNDS, params=None):
        self.fb = fb
        self.quantiles = quantiles
        self.num_rounds = num_rounds
        self.params = dict(LGB_PARAMS, **(params or {}))
        self.models = None
        self.cols = None

    def fit(self, train_origins, max_target: int):
        df = self.fb.build(train_origins)
        df = df[(df.target <= max_target) & np.isfinite(df.ratio)]
        self.cols = feature_columns(df)
        X = df[self.cols]
        self.models = []
        ds = lgb.Dataset(X, df.ratio.values, categorical_feature=CATEGORICAL, free_raw_data=False)
        for q in self.quantiles:
            p = dict(self.params, alpha=float(q))
            self.models.append(lgb.train(p, ds, self.num_rounds))
        return self

    def predict(self, origins, horizons=HORIZONS):
        df = self.fb.build(origins, horizons, with_target=False)
        X = df[self.cols]
        r = np.column_stack([m.predict(X) for m in self.models])
        q = sort_quantiles(np.maximum(r, 0)) * np.exp(df.log_scale.values)[:, None]
        return q.reshape(len(horizons), len(origins), self.fb.S, len(self.quantiles)).transpose(1, 0, 2, 3)

    def importance(self, q_index=3):
        m = self.models[q_index]
        return dict(zip(self.cols, m.feature_importance("gain")))


def conformalize(Q: np.ndarray, origins: np.ndarray, Y: np.ndarray, horizons=HORIZONS, quantiles=QUANTILES,
                 window: int = 56, groups: np.ndarray | None = None) -> np.ndarray:
    """Rolling split-conformal recalibration, separately for every quantile level.

    For level a, origin o and horizon h the corrected quantile is
        q~ = q + s * Quantile_a( (y - q)/s  over calibration pairs (o', h) with o' + h <= o )
    with s the forecast median (scale-free scores) and the calibration window the last
    `window` origins whose outcome is already observed at o. Pooled within `groups`
    (default: product category). Corrections at different levels are applied to the same
    sorted vector and re-sorted, which keeps the quantile function monotone.
    """
    nO, nH, S, nQ = Q.shape
    groups = np.zeros(S, dtype=int) if groups is None else groups
    G = groups.max() + 1
    out = Q.copy()
    for a, h in enumerate(horizons):
        med = np.maximum(Q[:, a, :, nQ // 2], 1.0)
        tgt = origins + h
        obs = np.where(tgt[:, None] < len(Y), Y[np.minimum(tgt, len(Y) - 1)], np.nan)
        score = (obs[:, :, None] - Q[:, a]) / med[:, :, None]           # (nO,S,nQ)
        for n, o in enumerate(origins):
            cal = np.where((origins + h <= o) & (origins > o - window - h))[0]
            if len(cal) < 14:
                continue
            for g in range(G):
                m = groups == g
                sc = score[cal][:, m, :].reshape(-1, nQ)
                corr = np.array([np.nanquantile(sc[:, i], quantiles[i]) for i in range(nQ)])
                out[n, a, m] = Q[n, a, m] + med[n, m][:, None] * corr[None, :]
    return sort_quantiles(np.maximum(out, 0))


def oracle_quantiles(hist: DemandHistory, origins, horizons=HORIZONS, quantiles=QUANTILES, n=2000, seed=0):
    """Quantiles of the true conditional distribution of y[o+h] given information at o."""
    rng = np.random.default_rng(seed)
    p = hist.params
    out = np.empty((len(origins), len(horizons), hist.net.J * hist.net.K, len(quantiles)))
    for a, h in enumerate(horizons):
        for i, o in enumerate(origins):
            t = o + h
            samples = _oracle_h(hist, o, h, n, rng)
            out[i, a] = np.quantile(samples.reshape(n, -1), quantiles, axis=0).T
    return out


def _oracle_h(hist: DemandHistory, o: int, h: int, n: int, rng) -> np.ndarray:
    """h-step-ahead sampling of the latent AR(1) states, then NB draws."""
    p = hist.params
    t = o + h

    def ar(phi, sd, x0, shape):
        m = phi ** h * x0
        v = sd ** 2 * (1 - phi ** (2 * h)) / (1 - phi ** 2)
        return m + np.sqrt(v) * rng.standard_normal(shape)

    eta = ar(p.phi_nat, p.sd_nat, hist.eta[o], (n,))
    xi = ar(p.phi_reg, p.sd_reg, hist.xi[o][None, :], (n, hist.xi.shape[1]))
    zeta = ar(p.phi_cat, p.sd_cat, hist.zeta[o][None, :], (n, hist.zeta.shape[1]))
    log_mu = hist.det_log[t][None] + eta[:, None, None] + xi[:, hist.region_idx][:, :, None] + zeta[:, None, :]
    r = hist.net.cat_dispersion[None, None, :]
    lam = rng.gamma(np.broadcast_to(r, log_mu.shape), np.exp(log_mu) / r)
    return rng.poisson(lam).astype(float)
