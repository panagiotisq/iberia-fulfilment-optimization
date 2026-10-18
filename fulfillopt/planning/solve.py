"""High-level planning API: solve the stochastic program, its deterministic variants, and
evaluate any first-stage plan out-of-sample."""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import highspy
import numpy as np
import pandas as pd

from .benders import BendersOptions, BendersSolver, SubproblemSolver
from .highs_util import make_highs, status_ok
from .model import PlanningInstance, Recourse, extensive_form, first_stage_co2, first_stage_cost


@dataclass
class Plan:
    y: np.ndarray
    w: np.ndarray
    objective: float                 # in-sample objective
    bound: float
    time: float
    method: str
    info: dict = field(default_factory=dict)


def solve_benders(inst: PlanningInstance, rec: Recourse | None = None, **kw) -> Plan:
    opts = BendersOptions(**kw)
    r = BendersSolver(inst, rec, opts).solve()
    return Plan(r.y, r.w, r.ub, r.lb, r.time, "benders",
                dict(iterations=r.iterations, lp_iterations=r.lp_iterations, cuts=r.cuts, gap=r.gap,
                     time_master=r.time_master, time_sub=r.time_sub, history=r.history))


def solve_extensive(inst: PlanningInstance, rec: Recourse | None = None, time_limit: float = 3600.0,
                    mip_gap: float = 5e-4, eps_late: float | None = None, threads: int | None = None) -> Plan:
    rec = rec or Recourse(inst.net, inst.inventory)
    m = extensive_form(inst, rec, eps_late=eps_late)
    opts = dict(mip_rel_gap=mip_gap, time_limit=time_limit)
    h = make_highs(*m, **opts)
    t0 = time.time()
    h.run()
    dt = time.time() - t0
    I, H = inst.net.I, inst.net.H
    info = h.getInfo()
    feasible = info.primal_solution_status == 2
    if not feasible:
        return Plan(None, None, np.inf, info.mip_dual_bound, dt, "extensive", dict(status=str(h.getModelStatus())))
    x = np.asarray(h.getSolution().col_value)
    y = np.round(x[:I * H]).reshape(I, H)
    w = np.round(x[I * H:I * H + I])
    extra = {}
    if eps_late is not None:
        # Lagrange multiplier of the expected-lateness constraint with the first stage fixed: the
        # operational "price of speed" that makes the per-scenario recourse reproduce the plan
        n1 = I * H + I
        idx = np.arange(n1, dtype=np.int32)
        h.changeColsIntegrality(n1, idx, np.array([highspy.HighsVarType.kContinuous] * n1))
        h.changeColsBounds(n1, idx, x[:n1].round(), x[:n1].round())
        h.run()
        extra["lambda_eps"] = float(-np.asarray(h.getSolution().row_dual)[-1])
    return Plan(y, w, info.objective_function_value, info.mip_dual_bound, dt, "extensive",
                dict(gap=info.mip_gap, status=str(h.getModelStatus()), nodes=info.mip_node_count,
                     n_rows=m[1].shape[0], n_cols=m[1].shape[1], **extra))


def solve_mean_value(inst: PlanningInstance, rec: Recourse | None = None, demand: np.ndarray | None = None,
                     time_limit: float = 300.0) -> Plan:
    """Deterministic (expected-value) problem: plan for a single point forecast."""
    d = inst.scenarios.mean(axis=0) if demand is None else demand
    ev = inst.with_scenarios(d[None], np.ones(1), label=inst.label + "-EV")
    p = solve_extensive(ev, rec, time_limit=time_limit, mip_gap=5e-4)
    p.method = "mean-value"
    return p


# ----------------------------------------------------------------------------- evaluation

_E = {}


def _eval_init(rec, scen):
    _E["sp"] = SubproblemSolver(rec)
    _E["scen"] = scen
    _E["rec"] = rec


def _eval_chunk(args):
    idx, y, w = args
    out = []
    for s in idx:
        obj, _, _, prim = _E["sp"].solve(y, w, _E["scen"][s], want_primal=True)
        k = _E["rec"].kpis(prim, _E["scen"][s])
        k["recourse"] = obj
        k["scenario"] = int(s)
        out.append(k)
    return out


def evaluate_plan(net, inventory, y, w, scenarios, workers: int = 1, rec: Recourse | None = None,
                  late_penalty: float | None = None) -> pd.DataFrame:
    """Out-of-sample evaluation: one recourse LP per scenario; returns per-scenario KPIs.

    total = first-stage cost + recourse (recourse includes the late penalty lam * late units).
    """
    rec = rec or Recourse(net, inventory)
    if late_penalty is not None:
        rec.set_late_penalty(late_penalty)
    S = len(scenarios)
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_eval_init, initargs=(rec, scenarios)) as pool:
            chunks = np.array_split(np.arange(S), workers * 4)
            res = pool.map(_eval_chunk, [(c, y, w) for c in chunks])
        rows = [r for chunk in res for r in chunk]
    else:
        _eval_init(rec, scenarios)
        rows = _eval_chunk((np.arange(S), y, w))
    df = pd.DataFrame(rows).sort_values("scenario").reset_index(drop=True)
    fc = first_stage_cost(net, y, w)
    df["first_stage"] = fc
    df["total"] = fc + df["recourse"]
    df["cost_excl_late"] = df["total"] - rec.lam * df["late_units"]
    df["trucks"] = y.sum()
    df["crews"] = w.sum()
    df["co2"] = first_stage_co2(net, y) + df["co2_spot"] + df["co2_lastmile"]
    return df


def summarize(df: pd.DataFrame) -> dict:
    n = len(df)
    tot = df["total"]
    return dict(
        exp_total=tot.mean(), se_total=tot.std(ddof=1) / np.sqrt(n),
        exp_cost=df["cost_excl_late"].mean(),
        on_time=1 - df["late_units"].sum() / df["units"].sum(),
        on_time_pt=1 - df["late_pt"].sum() / max(df["units_pt"].sum(), 1.0),
        on_time_es=1 - (df["late_units"].sum() - df["late_pt"].sum()) / (df["units"].sum() - df["units_pt"].sum()),
        unmet_rate=df["unmet"].sum() / df["units"].sum(),
        p95_total=tot.quantile(.95), cvar95=tot[tot >= tot.quantile(.95)].mean(),
        trucks=df["trucks"].iloc[0], crews=df["crews"].iloc[0],
        spot_pe=df["spot_pe"].mean(), overtime=df["overtime_lu"].mean(), co2_t=df["co2"].mean() / 1e3,
    )


def wait_and_see(inst: PlanningInstance, rec: Recourse | None = None, time_limit=60.0,
                 mip_gap: float = 5e-4) -> np.ndarray:
    """Value of each scenario's deterministic problem (perfect information).

    Returns an (S, 2) array: column 0 holds the MIP dual bounds, column 1 the incumbent values
    (both within the MIP tolerance of the deterministic optimum).
    """
    rec = rec or Recourse(inst.net, inst.inventory)
    vals = []
    for s in range(inst.S):
        ws = inst.with_scenarios(inst.scenarios[s:s + 1], np.ones(1))
        p = solve_extensive(ws, rec, time_limit=time_limit, mip_gap=mip_gap)
        vals.append((p.bound, p.objective))
    return np.array(vals)
