"""E8 - Strategic what-if: value of opening a fulfilment centre near Lisbon (Azambuja).

The candidate site is added to the network, inventory is re-positioned over the 11 FCs
(same national cover), and the day-ahead stochastic plan is re-optimised. The comparison
uses the true demand distribution (oracle scenarios) on the planning test days, so the
result isolates the network effect from forecast error. Results are reported for two
values of the price of speed (late penalty), because the business case depends on it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fulfillopt import config, geo, pipeline
from fulfillopt.planning.model import PlanningInstance, Recourse, position_inventory
from fulfillopt.planning.solve import evaluate_plan, solve_benders, summarize
from fulfillopt.world import world

WORKERS = 24
LAMBDAS = [1.0, 2.5]


def run():
    base, hist = world()
    cand = base.add_fcs(geo.CANDIDATE_FCS)
    rows = []
    for day in pipeline.test_days():
        t = hist.index_of(day)
        scen = hist.oracle_samples(t, 100, np.random.default_rng(t))
        eval_scen = hist.oracle_samples(t, 1000, np.random.default_rng(20_000 + t))
        mean = hist.oracle_mean(t)
        for lam in LAMBDAS:
            for name, net in [("base", base), ("with_LIS1", cand)]:
                net_l = net.with_params(late_penalty=lam)
                inv = position_inventory(net_l, mean)
                rec = Recourse(net_l, inv)
                plan = solve_benders(PlanningInstance(net_l, inv, scen), rec, workers=WORKERS)
                s = summarize(evaluate_plan(net_l, inv, plan.y, plan.w, eval_scen, workers=WORKERS, rec=rec))
                s.update(date=str(day.date()), network=name, lam=lam)
                rows.append(s)
                print(day.date(), lam, name, round(s["exp_total"]), round(s["exp_cost"]), round(s["on_time"], 4),
                      round(s["on_time_pt"], 4), flush=True)
        pd.DataFrame(rows).to_csv(config.RESULTS / "network_design.csv", index=False)


if __name__ == "__main__":
    # guard: skip if already complete or running in another process (see experiments/run_rest.sh)
    lock = config.RESULTS / "network_design.csv.lock"
    if lock.exists() or (config.RESULTS / "network_design.csv.done").exists():
        print("skipped (running elsewhere or done)")
    else:
        lock.touch()
        try:
            run()
            (config.RESULTS / "network_design.csv.done").touch()
        finally:
            lock.unlink()
