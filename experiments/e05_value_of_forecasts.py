"""E5 - Decision-focused evaluation: what is a better forecast worth in the planning problem?

For each test day, first-stage plans (trucks, crews) are built from different forecasts and
planning paradigms, then evaluated on the same 2000 draws from the *true* conditional
demand distribution (common random numbers). Plans are cached for the online experiment.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.planning.model import PlanningInstance, Recourse
from fulfillopt.planning.solve import evaluate_plan, solve_benders, solve_mean_value, summarize
from fulfillopt.world import world

N_SCEN = 100
N_EVAL = 2000
WORKERS = 24

METHODS = {
    # name: (kind, forecaster, copula / quantile level)
    "det_point": ("det", "lgbm_cqr", 0.5),
    "det_p90": ("det", "lgbm_cqr", 0.9),
    "sp_snaive": ("sp", "snaive_cqr", True),
    "sp_lgbm_indep": ("sp", "lgbm_cqr", False),
    "sp_lgbm_cop": ("sp", "lgbm_cqr", True),
    "sp_oracle": ("sp", "oracle", True),
}


def run():
    net, hist = world()
    out_dir = config.CACHE / "plans"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_file = config.RESULTS / "value_of_forecasts.csv"
    done = pd.read_csv(res_file) if res_file.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    for day in pipeline.test_days():
        t = hist.index_of(day)
        key = str(day.date())
        if len(done) and (done.date == key).sum() >= len(METHODS):
            continue
        inv = pipeline.day_inventory(t)
        rec = Recourse(net, inv)
        eval_scen = hist.oracle_samples(t, N_EVAL, np.random.default_rng(10_000 + t))
        for name, (kind, fc, arg) in METHODS.items():
            t0 = time.time()
            if kind == "det":
                d = pipeline.point_forecast(fc, t, arg)
                inst = PlanningInstance(net, inv, d[None], label=f"{key}-{name}")
                plan = solve_mean_value(inst, rec, demand=d)
            else:
                scen = pipeline.scenarios(fc, t, N_SCEN, seed=t, copula=arg)
                inst = PlanningInstance(net, inv, scen, label=f"{key}-{name}")
                plan = solve_benders(inst, rec, workers=WORKERS)
            t_solve = time.time() - t0
            ev = evaluate_plan(net, inv, plan.y, plan.w, eval_scen, workers=WORKERS, rec=rec)
            s = summarize(ev)
            s.update(date=key, method=name, solve_time=t_solve, in_sample=plan.objective,
                     iterations=plan.info.get("iterations", 0), gap=plan.info.get("gap", np.nan),
                     demand=float(eval_scen.sum(axis=(1, 2)).mean()))
            rows.append(s)
            np.savez_compressed(out_dir / f"{key}_{name}.npz", y=plan.y, w=plan.w)
            print(f"{key} {name:14s} E[cost]={s['exp_total']:,.0f} ±{s['se_total']:,.0f} on-time={s['on_time']:.4f} "
                  f"unmet={s['unmet_rate']:.4f} trucks={s['trucks']:.0f} crews={s['crews']:.0f} solve={t_solve:.1f}s",
                  flush=True)
        pd.DataFrame(rows).to_csv(res_file, index=False)


if __name__ == "__main__":
    run()
