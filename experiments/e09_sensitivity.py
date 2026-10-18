"""E9 - When does stochastic planning pay? VSS and EVPI-style gaps vs. recourse flexibility.

The value of the stochastic solution depends on how expensive it is to fix a capacity
shortfall on the day. We vary the spot premium, the overtime cap and the unmet-demand
penalty, and for each setting compare on the true distribution:
  RP  - SP plan (N=100 true scenarios), and
  EEV - plan for the mean demand,
both evaluated on 1000 fresh true scenarios (common random numbers).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.planning.model import PlanningInstance, Recourse
from fulfillopt.planning.solve import evaluate_plan, solve_benders, solve_mean_value, summarize
from fulfillopt.world import world

DAYS = ["2025-10-25", "2026-02-17", "2026-05-12", "2026-09-09"]
GRID = dict(spot_premium=[1.4, 1.8, 2.5, 3.5], overtime_frac=[0.30, 0.10], unmet_penalty=[15.0])
EXTRA = [dict(spot_premium=1.8, overtime_frac=0.30, unmet_penalty=p) for p in (9.0, 25.0, 50.0)]
WORKERS = 24


def run():
    net0, hist = world()
    rows = []
    settings = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())] + EXTRA
    for day in DAYS:
        t = hist.index_of(day)
        inv = pipeline.day_inventory(t)
        scen = hist.oracle_samples(t, 100, np.random.default_rng(t))
        ev_scen = hist.oracle_samples(t, 1000, np.random.default_rng(30_000 + t))
        for st in settings:
            net = net0.with_params(**st)
            rec = Recourse(net, inv)
            rp = solve_benders(PlanningInstance(net, inv, scen), rec, workers=WORKERS)
            mv = solve_mean_value(PlanningInstance(net, inv, scen), rec, demand=ev_scen.mean(axis=0))
            e_rp = evaluate_plan(net, inv, rp.y, rp.w, ev_scen, workers=WORKERS, rec=rec)
            e_mv = evaluate_plan(net, inv, mv.y, mv.w, ev_scen, workers=WORKERS, rec=rec)
            diff = e_mv.total.values - e_rp.total.values
            s_rp, s_mv = summarize(e_rp), summarize(e_mv)
            rows.append(dict(day=day, **st, RP=s_rp["exp_total"], EEV=s_mv["exp_total"], VSS=diff.mean(),
                             VSS_se=diff.std(ddof=1) / np.sqrt(len(diff)), VSS_pct=100 * diff.mean() / s_rp["exp_total"],
                             rp_trucks=s_rp["trucks"], mv_trucks=s_mv["trucks"], rp_crews=s_rp["crews"],
                             mv_crews=s_mv["crews"], rp_unmet=s_rp["unmet_rate"], mv_unmet=s_mv["unmet_rate"],
                             rp_cvar=s_rp["cvar95"], mv_cvar=s_mv["cvar95"]))
            print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(config.RESULTS / "sensitivity.csv", index=False)


def run_forecast_regimes():
    """Do the copula and the stochastic program matter more when recourse is inflexible?

    Same comparison as E5 (forecast-based plans scored on the true distribution) under the base
    recourse and under an inflexible one (spot 3.5x, overtime cap 10%), on 10 regular test days.
    """
    from fulfillopt.planning.solve import solve_mean_value
    net0, hist = world()
    days = [d for d in pipeline.test_days() if str(d.date()) not in pipeline.EVENT_DAYS][::2][:10]
    regimes = {"base": dict(), "inflexible": dict(spot_premium=3.5, overtime_frac=0.10)}
    rows = []
    for day in days:
        t = hist.index_of(day)
        inv = pipeline.day_inventory(t)
        ev_scen = hist.oracle_samples(t, 1000, np.random.default_rng(40_000 + t))
        for reg, kw in regimes.items():
            net = net0.with_params(**kw)
            rec = Recourse(net, inv)
            plans = {}
            d_med = pipeline.point_forecast("lgbm_cqr", t, 0.5)
            plans["det_point"] = solve_mean_value(PlanningInstance(net, inv, d_med[None]), rec, demand=d_med)
            for name, fc, cop in [("sp_lgbm_indep", "lgbm_cqr", False), ("sp_lgbm_cop", "lgbm_cqr", True),
                                  ("sp_oracle", "oracle", True)]:
                scen = pipeline.scenarios(fc, t, 100, seed=t, copula=cop)
                plans[name] = solve_benders(PlanningInstance(net, inv, scen), rec, workers=WORKERS)
            for name, pl in plans.items():
                e = evaluate_plan(net, inv, pl.y, pl.w, ev_scen, workers=WORKERS, rec=rec)
                s = summarize(e)
                rows.append(dict(day=str(day.date()), regime=reg, method=name, exp_total=s["exp_total"],
                                 unmet=s["unmet_rate"], trucks=s["trucks"], crews=s["crews"], cvar95=s["cvar95"]))
            print(day.date(), reg, {r["method"]: round(r["exp_total"]) for r in rows[-4:]}, flush=True)
            pd.DataFrame(rows).to_csv(config.RESULTS / "sensitivity_forecast_regimes.csv", index=False)


if __name__ == "__main__":
    run()
    run_forecast_regimes()
