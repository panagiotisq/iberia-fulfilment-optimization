"""E7 - Online order assignment under the day-ahead plan.

For every test day, the capacity plan from E5 (SP with LightGBM + copula scenarios) is fixed;
demand realisations are drawn from the true distribution and arrive over 96 intraday slots.
Policies: greedy, static bid prices, re-solved bid prices (hourly, with demand learning),
fluid LP tracking, and the clairvoyant hindsight LP. A second study varies the re-solve
frequency and the demand-learning switch.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.online.simulator import OnlineSimulator, arrivals
from fulfillopt.world import world

POLICIES = ["greedy", "bid_static", "bid_resolve", "lp_tracking"]
REPS = 4


def run():
    net, hist = world()
    rows = []
    for day in pipeline.test_days():
        t = hist.index_of(day)
        key = str(day.date())
        f = np.load(config.CACHE / "plans" / f"{key}_sp_lgbm_cop.npz")
        inv = pipeline.day_inventory(t)
        sim = OnlineSimulator(net, inv, f["y"], f["w"])
        # expected demand known to the online policy: mean of the planning scenarios
        fc = pipeline.scenarios("lgbm_cqr", t, 100, seed=t, copula=True).mean(axis=0)
        for rep in range(REPS):
            rng = np.random.default_rng(50_000 + 97 * t + rep)
            d = hist.oracle_samples(t, 1, rng)[0]
            arr = arrivals(d, rng)
            hs = sim.hindsight(d)
            rows.append(dict(date=key, rep=rep, policy="hindsight", cost=hs.cost, units=hs.units, unmet=hs.unmet,
                             late=hs.late, overtime=hs.overtime, spot=hs.spot, lp_solves=1, time=0.0))
            for pol in POLICIES:
                t0 = time.time()
                r = sim.run(arr, fc, pol, rng=np.random.default_rng(rep))
                rows.append(dict(date=key, rep=rep, policy=pol, cost=r.cost, units=r.units, unmet=r.unmet,
                                 late=r.late, overtime=r.overtime, spot=r.spot, lp_solves=r.lp_solves,
                                 time=time.time() - t0))
        g = pd.DataFrame(rows).query("date == @key").groupby("policy").cost.mean()
        print(key, (g / g["hindsight"] - 1).round(4).to_dict(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS / "online.csv", index=False)

    # sensitivity: re-solve frequency and demand learning (subset of days)
    sens = []
    days = pipeline.test_days()[::3]
    for day in days:
        t = hist.index_of(day)
        key = str(day.date())
        f = np.load(config.CACHE / "plans" / f"{key}_sp_lgbm_cop.npz")
        inv = pipeline.day_inventory(t)
        sim = OnlineSimulator(net, inv, f["y"], f["w"])
        fc = pipeline.scenarios("lgbm_cqr", t, 100, seed=t, copula=True).mean(axis=0)
        for rep in range(REPS):
            rng = np.random.default_rng(50_000 + 97 * t + rep)
            d = hist.oracle_samples(t, 1, rng)[0]
            arr = arrivals(d, rng)
            hs = sim.hindsight(d).cost
            for pol in ["bid_resolve", "lp_tracking"]:
                for every in [1, 2, 4, 8, 16, 32, 96]:
                    for learn in [True, False]:
                        r = sim.run(arr, fc, pol, resolve_every=every, learn=learn, rng=np.random.default_rng(rep))
                        sens.append(dict(date=key, rep=rep, policy=pol, every=every, learn=learn,
                                         regret=r.cost / hs - 1))
        print("sens", key, flush=True)
    pd.DataFrame(sens).to_csv(config.RESULTS / "online_sensitivity.csv", index=False)


if __name__ == "__main__":
    run()
