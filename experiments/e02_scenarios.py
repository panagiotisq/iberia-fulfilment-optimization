"""E2 - Joint scenario quality: independence vs Gaussian copula vs oracle.

Scenarios are generated from the conformalised LightGBM quantiles (h=1). Scores are computed
on the aggregates that drive capacity decisions: sort-centre x category totals (48 dims,
energy + variogram score) and national totals (interval coverage). Also stores the rolling
copula correlations used by the planning experiments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fulfillopt import config
from fulfillopt.forecast import metrics as M
from fulfillopt.forecast.scenarios import fit_copula, pit_scores, sample_scenarios
from fulfillopt.world import world

N_SCEN = 1000
COPULA_WINDOW = 120
STRIDE = 3


def load_forecasts():
    f = np.load(config.CACHE / "forecasts.npz")
    return f


def rolling_pit(Q, origins, hist):
    """Normal scores of realised h=1 outcomes, (n_origins, S)."""
    Y = hist.y.reshape(hist.T, -1)
    return pit_scores(Q[:, 0], Y[origins + 1])


def copula_at(Z, origins, o, window=COPULA_WINDOW):
    """Correlation from PITs whose outcome (origin+1) is observed at o."""
    m = (origins + 1 <= o) & (origins + 1 > o - window)
    return fit_copula(Z[m])


def aggregate(net, scen):
    """(N, J*K) -> SC x cat totals (N, H*K) and national (N,)."""
    N = scen.shape[0]
    s = scen.reshape(N, net.J, net.K)
    agg = np.zeros((N, net.H, net.K))
    np.add.at(agg.transpose(1, 0, 2), net.zone_sc, s.transpose(1, 0, 2))
    return agg.reshape(N, -1), s.sum(axis=(1, 2))


def run():
    net, hist = world()
    f = load_forecasts()
    origins = f["origins"]
    Y = hist.y.reshape(hist.T, -1)
    test_o = np.arange(hist.index_of(config.TEST_START), hist.index_of(config.TEST_END) + 1, STRIDE)
    rows = []
    rng = np.random.default_rng(42)
    for method in ["lgbm_cqr", "snaive_cqr"]:
        Q = f[method]
        Z = rolling_pit(Q, origins, hist)
        C = None
        for n, o in enumerate(test_o):
            if C is None or n % 3 == 0:
                C = copula_at(Z, origins, o)
            qo = Q[np.searchsorted(origins, o), 0]
            y = Y[o + 1]
            ya, yn = aggregate(net, y[None])
            for cop, corr in [("indep", None), ("copula", C)]:
                sc = sample_scenarios(qo, N_SCEN, rng, corr)
                a, nat = aggregate(net, sc)
                rows.append(dict(method=method, copula=cop, origin=o, date=hist.dates[o + 1],
                                 es=M.energy_score(a / 1e3, ya[0] / 1e3, rng=rng),
                                 vs=M.variogram_score(a[:400] / 1e3, ya[0] / 1e3),
                                 nat_in90=float(np.quantile(nat, .05) <= yn[0] <= np.quantile(nat, .95)),
                                 nat_in50=float(np.quantile(nat, .25) <= yn[0] <= np.quantile(nat, .75)),
                                 nat_sd=float(nat.std() / nat.mean())))
        if method == "lgbm_cqr":
            np.savez_compressed(config.CACHE / "pit_lgbm_cqr.npz", Z=Z, origins=origins)
    # oracle reference
    for o in test_o:
        sc = hist.oracle_samples(o + 1, N_SCEN, rng).reshape(N_SCEN, -1)
        a, nat = aggregate(net, sc)
        ya, yn = aggregate(net, Y[o + 1][None])
        rows.append(dict(method="oracle", copula="true", origin=o, date=hist.dates[o + 1],
                         es=M.energy_score(a / 1e3, ya[0] / 1e3, rng=rng),
                         vs=M.variogram_score(a[:400] / 1e3, ya[0] / 1e3),
                         nat_in90=float(np.quantile(nat, .05) <= yn[0] <= np.quantile(nat, .95)),
                         nat_in50=float(np.quantile(nat, .25) <= yn[0] <= np.quantile(nat, .75)),
                         nat_sd=float(nat.std() / nat.mean())))
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "scenario_scores.csv", index=False)
    print(df.groupby(["method", "copula"])[["es", "vs", "nat_in50", "nat_in90", "nat_sd"]].mean().round(4))


if __name__ == "__main__":
    run()
