"""E6 - Multi-objective analysis: cost vs speed (on-time rate) vs CO2.

(a) Weighted-sum frontier: late penalty lam swept, SAA with 100 true scenarios (Benders).
(b) Exact comparison on a 10-scenario SAA solved as extensive form:
      weighted sum vs augmented epsilon-constraint (AUGMECON-style). With integer trucks the
      frontier is non-convex, so the weighted sum can only find *supported* points.
    The augmentation is a tiny late penalty (1e-3 EUR/unit) that breaks ties towards fewer late
    units, which removes weakly efficient points exactly like AUGMECON's slack reward.
(c) Carbon-price sweep: a price kappa (EUR/t CO2) added to line-haul truck-km, cost vs CO2.
All plans are evaluated out-of-sample on 2000 true scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.planning.model import PlanningInstance, Recourse
from fulfillopt.planning.solve import evaluate_plan, solve_benders, solve_extensive, summarize
from fulfillopt.world import world

DAY = "2026-03-05"
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0]
CARBON = [0, 50, 100, 200, 400, 800, 1600, 3200]
WORKERS = 24


def _eval(net, inv, plan, scen, lam_eval=0.0):
    ev = evaluate_plan(net, inv, plan.y, plan.w, scen, workers=WORKERS, late_penalty=lam_eval)
    s = summarize(ev)
    return s


def run():
    net0, hist = world()
    t = hist.index_of(DAY)
    inv = pipeline.day_inventory(t)
    eval_scen = hist.oracle_samples(t, 2000, np.random.default_rng(7))
    rows = []

    # (a) weighted sum, S=100, Benders
    scen100 = hist.oracle_samples(t, 100, np.random.default_rng(1))
    for lam in LAMBDAS:
        net = net0.with_params(late_penalty=lam)
        plan = solve_benders(PlanningInstance(net, inv, scen100), workers=WORKERS)
        s = _eval(net0, inv, plan, eval_scen, lam_eval=lam)   # operations use the same price of speed
        rows.append(dict(study="weighted_S100", param=lam, **s))
        print("ws100", lam, round(s["exp_cost"]), round(s["on_time"], 4), flush=True)

    # (b) exact comparison on S=10 extensive form
    scen10 = hist.oracle_samples(t, 10, np.random.default_rng(2))
    late_in = []
    for lam in LAMBDAS:
        net = net0.with_params(late_penalty=lam)
        inst = PlanningInstance(net, inv, scen10)
        plan = solve_extensive(inst, mip_gap=1e-4, time_limit=900)
        s_in = _eval(net0, inv, plan, scen10, lam_eval=lam)
        s = _eval(net0, inv, plan, eval_scen, lam_eval=lam)
        late_in.append(1 - s_in["on_time"])
        rows.append(dict(study="weighted_S10", param=lam, in_cost=s_in["exp_cost"], in_on_time=s_in["on_time"], **s))
        print("ws10", lam, round(s_in["exp_cost"]), round(s_in["on_time"], 4), flush=True)
    hi, lo = max(late_in), min(late_in)
    net_aug = net0.with_params(late_penalty=1e-3)
    for eps in np.linspace(hi, lo, 16):
        inst = PlanningInstance(net_aug, inv, scen10)
        plan = solve_extensive(inst, mip_gap=1e-4, time_limit=900, eps_late=float(eps) + 1e-7)
        if plan.y is None:
            continue
        # recourse operated with the Lagrange multiplier of the epsilon-constraint
        lam_eps = max(plan.info.get("lambda_eps", 0.0), 0.0) + 1e-3
        s_in = _eval(net0, inv, plan, scen10, lam_eval=lam_eps)
        s = _eval(net0, inv, plan, eval_scen, lam_eval=lam_eps)
        rows.append(dict(study="epsilon_S10", param=float(eps), lam_eps=lam_eps, in_cost=s_in["exp_cost"],
                         in_on_time=s_in["on_time"], ef_obj=plan.objective,
                         ef_on_time=1 - float(eps), **s))
        print("eps10", round(eps, 4), round(s_in["exp_cost"]), round(s_in["on_time"], 4), flush=True)

    # (c) carbon price
    for kappa in CARBON:
        p = net0.params
        net = net0.with_params(truck_per_km=p.truck_per_km + kappa * p.co2_truck_per_km / 1000)
        plan = solve_benders(PlanningInstance(net, inv, scen100), workers=WORKERS)
        s = _eval(net0, inv, plan, eval_scen, lam_eval=p.late_penalty)
        rows.append(dict(study="carbon", param=kappa, **s))
        print("carbon", kappa, round(s["exp_cost"]), round(s["co2_t"], 2), flush=True)

    pd.DataFrame(rows).to_csv(config.RESULTS / "pareto.csv", index=False)


if __name__ == "__main__":
    # guard: skip if already complete or running in another process (see experiments/run_rest.sh)
    lock = config.RESULTS / "pareto.csv.lock"
    if lock.exists() or (config.RESULTS / "pareto.csv.done").exists():
        print("skipped (running elsewhere or done)")
    else:
        lock.touch()
        try:
            run()
            (config.RESULTS / "pareto.csv.done").touch()
        finally:
            lock.unlink()
