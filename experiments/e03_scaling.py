"""E3 - Computational study of the decomposition.

(a) Extensive form (HiGHS MIP) vs Benders as the number of scenarios grows.
(b) Benders scaling to 1000 scenarios and parallel speed-up of the subproblem phase.
(c) Ablation of the algorithmic enhancements over several days.
(d) Larger synthetic networks (exploratory, not in the default run: the master MIP does not
    reach the target within an hour from 20 FCs on; see report Section 6).
Run alone on the machine: timings are wall-clock.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from fulfillopt import config, pipeline
from fulfillopt.demand import simulate
from fulfillopt.network import synthetic
from fulfillopt.planning.model import PlanningInstance, Recourse, position_inventory
from fulfillopt.planning.solve import solve_benders, solve_extensive
from fulfillopt.world import world

DAY = "2026-03-05"
TOL = 5e-4


def part_a(rows):
    net, hist = world()
    t = hist.index_of(DAY)
    inv = pipeline.day_inventory(t)
    rec = Recourse(net, inv)
    for S in [5, 10, 25, 50, 100]:
        inst = PlanningInstance(net, inv, hist.oracle_samples(t, S, np.random.default_rng(S)))
        ef = solve_extensive(inst, rec, time_limit=1800, mip_gap=TOL)
        rows.append(dict(part="a", S=S, method="extensive", time=ef.time, obj=ef.objective, bound=ef.bound,
                         gap=(ef.objective - ef.bound) / ef.objective if np.isfinite(ef.objective) else np.nan,
                         cols=ef.info.get("n_cols"), rows=ef.info.get("n_rows")))
        print(rows[-1], flush=True)
        for w in [1, 24]:
            bd = solve_benders(inst, rec, workers=w, tol=TOL)
            rows.append(dict(part="a", S=S, method=f"benders_w{w}", time=bd.time, obj=bd.objective, bound=bd.bound,
                             gap=bd.info["gap"], iterations=bd.info["iterations"]))
            print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(config.RESULTS / "scaling.csv", index=False)


def part_b(rows):
    net, hist = world()
    t = hist.index_of(DAY)
    inv = pipeline.day_inventory(t)
    rec = Recourse(net, inv)
    for S in [50, 100, 200, 500, 1000]:
        inst = PlanningInstance(net, inv, hist.oracle_samples(t, S, np.random.default_rng(S)))
        bd = solve_benders(inst, rec, workers=24, tol=TOL)
        rows.append(dict(part="b", S=S, method="benders_w24", time=bd.time, obj=bd.objective, gap=bd.info["gap"],
                         iterations=bd.info["iterations"], t_master=bd.info["time_master"], t_sub=bd.info["time_sub"]))
        print(rows[-1], flush=True)
    inst = PlanningInstance(net, inv, hist.oracle_samples(t, 500, np.random.default_rng(500)))
    for w in [1, 2, 4, 8, 16, 24]:
        bd = solve_benders(inst, rec, workers=w, tol=TOL)
        rows.append(dict(part="speedup", S=500, method=f"w{w}", workers=w, time=bd.time, t_sub=bd.info["time_sub"],
                         t_master=bd.info["time_master"], iterations=bd.info["iterations"], obj=bd.objective))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(config.RESULTS / "scaling.csv", index=False)


VARIANTS = {
    "classic": dict(lp_phase=False, inout=False, jensen_blocks=0),
    "+LP phase": dict(lp_phase=True, inout=False, jensen_blocks=0),
    "+in-out": dict(lp_phase=True, inout=True, jensen_blocks=0),
    "+Jensen (phase 1)": dict(),
    "+Jensen (both phases)": dict(jensen_in_mip=True),
    "+Jensen x4 (phase 1)": dict(jensen_blocks=4),
    "single-cut (all)": dict(multicut=False),
}


def part_c(rows):
    net, hist = world()
    for day in ["2025-10-25", "2026-03-05", "2026-06-30"]:
        t = hist.index_of(day)
        inv = pipeline.day_inventory(t)
        rec = Recourse(net, inv)
        inst = PlanningInstance(net, inv, hist.oracle_samples(t, 50, np.random.default_rng(t)))
        for name, kw in VARIANTS.items():
            bd = solve_benders(inst, rec, workers=24, tol=TOL, time_limit=900, **kw)
            rows.append(dict(part="ablation", day=day, method=name, time=bd.time, iterations=bd.info["iterations"],
                             lp_iterations=bd.info["lp_iterations"], cuts=bd.info["cuts"], gap=bd.info["gap"],
                             t_master=bd.info["time_master"], t_sub=bd.info["time_sub"], obj=bd.objective))
            print(rows[-1], flush=True)
            hist_df = pd.DataFrame(bd.info["history"]).assign(day=day, method=name)
            hist_df.to_csv(config.CACHE / f"conv_{day}_{name.replace(' ', '_')}.csv", index=False)
        pd.DataFrame(rows).to_csv(config.RESULTS / "scaling.csv", index=False)


def part_d(rows):
    for (I, H, J, K) in [(10, 12, 65, 4), (20, 25, 150, 6), (30, 40, 300, 8), (50, 60, 600, 8)]:
        net = synthetic(I, H, J, K, seed=I)
        hist = simulate(net, "2024-01-01", "2024-04-30", seed=I)
        t = 100
        inv = position_inventory(net, hist.oracle_mean(t))
        inst = PlanningInstance(net, inv, hist.oracle_samples(t, 50, np.random.default_rng(t)))
        rec = Recourse(net, inv)
        bd = solve_benders(inst, rec, workers=24, tol=TOL, time_limit=3600)
        rows.append(dict(part="synthetic", I=I, H=H, J=J, K=K, S=50, method="benders_w24", time=bd.time,
                         iterations=bd.info["iterations"], gap=bd.info["gap"], int_vars=I * H + I,
                         sub_cols=rec.n_col, sub_rows=rec.n_row, ef_cols=50 * rec.n_col,
                         t_master=bd.info["time_master"], t_sub=bd.info["time_sub"]))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(config.RESULTS / "scaling.csv", index=False)


if __name__ == "__main__":
    f = config.RESULTS / "scaling.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    parts = sys.argv[1:] or ["a", "b", "c"]
    for p in parts:
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[p](rows)
