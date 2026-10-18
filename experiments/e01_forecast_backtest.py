"""E1 - Rolling-origin backtest of probabilistic forecasters.

Produces results/cache/forecasts.npz with quantile forecasts Q[method][origin, h, series, q]
for every origin from FORECAST_START to TEST_END, and results/forecast_metrics.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fulfillopt import config
from fulfillopt.forecast import metrics as M
from fulfillopt.forecast.features import HORIZONS, FeatureBuilder
from fulfillopt.forecast.models import QUANTILES, LGBMQuantile, SeasonalNaive, conformalize, oracle_quantiles
from fulfillopt.world import world


def run():
    config.ensure_dirs()
    net, hist = world()
    fb = FeatureBuilder(hist)
    o0, o1 = hist.index_of(config.FORECAST_START), hist.index_of(config.TEST_END)
    origins = np.arange(o0, o1 + 1)
    retrain = [hist.index_of(d) for d in config.RETRAIN_DATES] + [o1 + 1]
    Y = hist.y.reshape(hist.T, -1)

    Q = {}
    t0 = time.time()
    Q["snaive"] = SeasonalNaive(hist).predict(origins)
    print(f"snaive {time.time() - t0:.0f}s", flush=True)

    blocks, importances = [], []
    for a, b in zip(retrain[:-1], retrain[1:]):
        t0 = time.time()
        m = LGBMQuantile(fb).fit(np.arange(60, a), max_target=a)
        blk = np.arange(a, b)
        blocks.append(m.predict(blk))
        importances.append(pd.Series(m.importance(3), name=str(hist.dates[a].date())))
        print(f"lgbm block {hist.dates[a].date()} {time.time() - t0:.0f}s", flush=True)
    Q["lgbm"] = np.concatenate(blocks)
    pd.concat(importances, axis=1).to_csv(config.RESULTS / "lgbm_importance.csv")

    t0 = time.time()
    Q["lgbm_cqr"] = conformalize(Q["lgbm"], origins, Y, groups=fb.cat_of)
    print(f"conformal {time.time() - t0:.0f}s", flush=True)
    Q["snaive_cqr"] = conformalize(Q["snaive"], origins, Y, groups=fb.cat_of)

    t0 = time.time()
    Q["oracle"] = oracle_quantiles(hist, origins, n=1000)
    print(f"oracle {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(config.CACHE / "forecasts.npz", origins=origins, quantiles=QUANTILES,
                        **{k: v.astype(np.float32) for k, v in Q.items()})
    evaluate(Q, origins, hist)


def evaluate(Q, origins, hist):
    Y = hist.y.reshape(hist.T, -1)
    test = (origins >= hist.index_of(config.TEST_START))
    rows = []
    for name, q in Q.items():
        for split, mask in [("validation", ~test), ("test", test)]:
            for a, h in enumerate(HORIZONS):
                o = origins[mask]
                ok = o + h < hist.T
                qq, yy = q[mask][ok, a], Y[o[ok] + h]
                rows.append(dict(method=name, split=split, h=h,
                                 wmape=M.wmape(qq[..., 3], yy), bias=M.bias(qq[..., 3], yy),
                                 sql=M.scaled_quantile_loss(qq, yy, QUANTILES),
                                 cov50=M.coverage(qq, yy, QUANTILES, .25, .75),
                                 cov80=M.coverage(qq, yy, QUANTILES, .10, .90),
                                 cov90=M.coverage(qq, yy, QUANTILES, .05, .95),
                                 width80=M.interval_width(qq, yy, QUANTILES, .10, .90)))
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "forecast_metrics.csv", index=False)
    print(df[df.h.isin([1, 7])].round(4).to_string())


if __name__ == "__main__":
    run()
