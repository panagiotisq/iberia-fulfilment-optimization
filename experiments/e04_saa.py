"""E4 - Sample average approximation: statistical bounds, VSS and EVPI.

For a representative day and a peak day, with scenarios from the true distribution:
  * for N in N_GRID, solve M independent SAA replications -> lower-bound estimator
    (mean of SAA optimal values, Mak, Morton & Wood 1999) with a 95% CI;
  * evaluate every candidate solution on N_EVAL fresh scenarios -> upper bound;
  * VSS = EEV - RP (mean-value plan evaluated out-of-sample vs stochastic plan);
  * EVPI = RP - WS (wait-and-see: per-scenario deterministic optimum, perfect information).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fulfillopt import config, pipeline
from fulfillopt.planning.model import PlanningInstance, Recourse
from fulfillopt.planning.solve import evaluate_plan, solve_benders, solve_mean_value, summarize, wait_and_see
from fulfillopt.world import world

N_GRID = [10, 25, 50, 100, 200, 400]
M_REP = 10
N_EVAL = 5000
N_WS = 100
WORKERS = 24
DAYS = ["2026-03-05", "2025-11-28"]


def run():
    net, hist = world()
    rows, summary = [], []
    for day in DAYS:
        t = hist.index_of(day)
        inv = pipeline.day_inventory(t)
        rec = Recourse(net, inv)
        ev_scen = hist.oracle_samples(t, N_EVAL, np.random.default_rng(99))
        best = None
        for N in N_GRID:
            for m in range(M_REP):
                scen = hist.oracle_samples(t, N, np.random.default_rng(1000 * N + m))
                inst = PlanningInstance(net, inv, scen)
                plan = solve_benders(inst, rec, workers=WORKERS)
                ev = evaluate_plan(net, inv, plan.y, plan.w, ev_scen, workers=WORKERS, rec=rec)
                s = summarize(ev)
                rows.append(dict(day=day, N=N, rep=m, saa_obj=plan.objective, saa_bound=plan.bound,
                                 oos=s["exp_total"], oos_se=s["se_total"], time=plan.time,
                                 iterations=plan.info["iterations"], trucks=s["trucks"], crews=s["crews"],
                                 on_time=s["on_time"]))
                if best is None or s["exp_total"] < best[0]:
                    best = (s["exp_total"], plan, ev)
                print(f"{day} N={N:4d} rep={m} saa={plan.objective:,.0f} oos={s['exp_total']:,.0f} "
                      f"t={plan.time:.1f}s", flush=True)
            pd.DataFrame(rows).to_csv(config.RESULTS / "saa_runs.csv", index=False)

        # VSS: mean-value solution, with the mean of a large sample as point forecast
        mv = solve_mean_value(PlanningInstance(net, inv, ev_scen[:1]), rec, demand=ev_scen.mean(axis=0))
        ev_mv = evaluate_plan(net, inv, mv.y, mv.w, ev_scen, workers=WORKERS, rec=rec)
        # EVPI: wait-and-see on N_WS scenarios, compared with the RP solution on the same scenarios
        ws_scen = ev_scen[:N_WS]
        ws_vals = wait_and_see(PlanningInstance(net, inv, ws_scen), rec)
        ws = ws_vals[:, 1]            # incumbent values (within 0.05% of the optimum)
        rp_best, plan_best, ev_best = best
        diff = ev_mv.total.values - ev_best.total.values          # paired (common random numbers)
        rp_on_ws = ev_best.total.values[:N_WS]
        summary.append(dict(day=day, RP=rp_best, EEV=ev_mv.total.mean(), VSS=diff.mean(),
                            VSS_se=diff.std(ddof=1) / np.sqrt(len(diff)),
                            WS=ws.mean(), EVPI=(rp_on_ws - ws).mean(),
                            EVPI_se=(rp_on_ws - ws).std(ddof=1) / np.sqrt(N_WS),
                            RP_on_time=summarize(ev_best)["on_time"], EEV_on_time=summarize(ev_mv)["on_time"],
                            RP_unmet=summarize(ev_best)["unmet_rate"], EEV_unmet=summarize(ev_mv)["unmet_rate"],
                            RP_trucks=plan_best.y.sum(), EEV_trucks=mv.y.sum(),
                            RP_crews=plan_best.w.sum(), EEV_crews=mv.w.sum()))
        print(summary[-1], flush=True)
        pd.DataFrame(summary).to_csv(config.RESULTS / "saa_summary.csv", index=False)

    # Mak-Morton-Wood style bounds per (day, N)
    df = pd.DataFrame(rows)
    out = []
    for (day, N), g in df.groupby(["day", "N"]):
        lb = g.saa_obj.mean()
        lb_hw = stats.t.ppf(0.975, len(g) - 1) * g.saa_obj.std(ddof=1) / np.sqrt(len(g))
        ub = g.oos.min()
        out.append(dict(day=day, N=N, lb=lb, lb_hw=lb_hw, ub_best=ub, ub_mean=g.oos.mean(),
                        ub_sd=g.oos.std(ddof=1), gap_pct=100 * (ub - lb) / ub,
                        gap_ci_pct=100 * (ub - (lb - lb_hw)) / ub, time=g.time.mean()))
    pd.DataFrame(out).to_csv(config.RESULTS / "saa_bounds.csv", index=False)
    print(pd.DataFrame(out).round(2).to_string())


if __name__ == "__main__":
    run()
