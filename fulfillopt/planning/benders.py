"""Benders (L-shaped) decomposition for the two-stage capacity planning problem.

Master:  min  F.y + L.w + sum_s p_s theta_s
         s.t. dock limits, crew limits, y, w integer,
              theta_s >= Q_s(y^k, w^k) + g^k_s.((y,w) - (y^k,w^k))   for all generated cuts.

Enhancements (each can be switched off for the ablation study):
  * multi-cut (one theta per scenario) vs single aggregated cut (Birge & Louveaux 1988);
  * phase 1 on the LP relaxation of the master, whose cuts remain valid for the MIP
    (McDaniel & Devine 1977);
  * in-out stabilisation of phase 1: cuts are separated at a convex combination of the
    master solution and a stability centre (Ben-Ameur & Neto 2007; Fischetti et al. 2017);
  * subproblems solved in parallel by persistent worker processes, each holding warm-started
    HiGHS models (dual simplex re-optimisation after right-hand-side changes);
  * trivial per-scenario lower bounds on theta_s.

Every master iterate is first-stage feasible (complete recourse), so evaluating it gives a
valid upper bound, and the algorithm stops when (UB - LB) / UB <= tol.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from dataclasses import dataclass, field

import highspy
import numpy as np
import scipy.sparse as sp

from .highs_util import INF, make_highs, set_row_bounds, status_ok
from .model import PlanningInstance, Recourse, first_stage_cost, linking_matrix

# ----------------------------------------------------------------------------- subproblems


class SubproblemSolver:
    """Warm-started HiGHS model of the recourse LP; scenarios are swapped through row bounds."""

    def __init__(self, rec: Recourse):
        self.rec = rec
        lo, hi = rec.rhs(np.zeros((rec.net.I, rec.net.H)), np.zeros(rec.net.I), np.zeros((rec.net.J, rec.net.K)))
        self.h = make_highs(rec.cost, rec.A, lo, hi, rec.col_lo, rec.col_hi,
                            simplex_strategy=1, presolve="off")   # dual simplex, keep basis between solves

    def set_cost(self, cost):
        n = len(cost)
        self.h.changeColsCost(n, np.arange(n, dtype=np.int32), np.asarray(cost, dtype=np.float64))

    def solve(self, y, w, d, want_primal=False):
        lo, hi = self.rec.rhs(y, w, d)
        set_row_bounds(self.h, lo, hi)
        self.h.run()
        if not status_ok(self.h):
            # very rarely the warm basis leads to numerical trouble: retry from scratch
            self.h.clearSolver()
            self.h.run()
            assert status_ok(self.h), self.h.getModelStatus()
        sol = self.h.getSolution()
        obj = self.h.getInfo().objective_function_value
        gy, gw = self.rec.cut_from_duals(np.asarray(sol.row_dual))
        prim = np.asarray(sol.col_value) if want_primal else None
        return obj, gy, gw, prim


_W = {}


def _worker_init(rec: Recourse, scenarios: np.ndarray):
    _W["sp"] = SubproblemSolver(rec)
    _W["scen"] = scenarios


def _worker_solve(args):
    idx, y, w, cost = args
    sp_ = _W["sp"]
    if cost is not None:
        sp_.set_cost(cost)
    out = []
    for s in idx:
        obj, gy, gw, _ = sp_.solve(y, w, _W["scen"][s])
        out.append((s, obj, gy, gw))
    return out


class SubproblemPool:
    """Evaluates Q_s(y, w) and cut coefficients for many scenarios, optionally in parallel."""

    def __init__(self, rec: Recourse, scenarios: np.ndarray, workers: int = 1):
        self.rec, self.scenarios, self.workers = rec, scenarios, workers
        self.S = len(scenarios)
        self.cost_override = None
        if workers > 1:
            ctx = mp.get_context("fork")
            self.pool = ctx.Pool(workers, initializer=_worker_init, initargs=(rec, scenarios))
            self.chunks = np.array_split(np.arange(self.S), workers)
        else:
            self.pool = None
            _worker_init(rec, scenarios)
            self.local = _W["sp"]

    def evaluate(self, y, w, cost=None):
        obj = np.empty(self.S)
        gy = np.empty((self.S,) + y.shape)
        gw = np.empty((self.S,) + w.shape)
        if self.pool is None:
            if cost is not None:
                self.local.set_cost(cost)
            res = [_worker_solve((np.arange(self.S), y, w, None))]
        else:
            res = self.pool.map(_worker_solve, [(c, y, w, cost) for c in self.chunks])
        for chunk in res:
            for s, o, a, b in chunk:
                obj[s], gy[s], gw[s] = o, a.reshape(y.shape), b
        return obj, gy, gw

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ----------------------------------------------------------------------------- master


@dataclass
class BendersOptions:
    multicut: bool = True
    lp_phase: bool = True
    inout: bool = True
    inout_alpha: float = 0.5          # weight of the master point in the separation point
    jensen_blocks: int = 1            # partial decomposition: # of mean-value recourse blocks kept in master
    jensen_in_mip: bool = False       # keep the blocks in the integer phase (stronger but heavier master)
    tol: float = 5e-4                 # relative optimality gap (UB - LB) / UB
    lp_tol: float = 1e-4
    lp_max_iter: int = 150            # budget of the LP phase (the integer phase always runs)
    max_iter: int = 300
    time_limit: float = 3600.0
    master_gap: float = 2e-3          # initial relative MIP gap of master solves; tightened adaptively
    master_time: float = 120.0        # time limit per master MIP solve
    workers: int = 1
    verbose: bool = False


@dataclass
class BendersResult:
    y: np.ndarray
    w: np.ndarray
    ub: float
    lb: float
    gap: float
    iterations: int
    lp_iterations: int
    cuts: int
    time: float
    time_master: float
    time_sub: float
    recourse: np.ndarray                     # Q_s at the returned solution
    history: list = field(default_factory=list)


class BendersSolver:
    def __init__(self, inst: PlanningInstance, rec: Recourse | None = None, opts: BendersOptions | None = None,
                 pool: SubproblemPool | None = None):
        self.inst = inst
        self.rec = rec or Recourse(inst.net, inst.inventory)
        self.opts = opts or BendersOptions()
        self._own_pool = pool is None
        self.pool = pool or SubproblemPool(self.rec, inst.scenarios, self.opts.workers)
        self._build_master()

    # master columns: y (I*H) | w (I) | theta (S or 1)
    def _build_master(self):
        net, p, o = self.inst.net, self.inst.net.params, self.opts
        I, H, S = net.I, net.H, self.inst.S
        self.n1 = I * H + I
        self.nt = S if o.multicut else 1
        lbs = np.array([self.rec.lower_bound(d) for d in self.inst.scenarios])
        theta_lo = lbs if o.multicut else np.array([float(self.inst.probs @ lbs)])
        theta_c = self.inst.probs if o.multicut else np.array([1.0])
        c = np.concatenate([net.truck_cost.reshape(-1), np.full(I, p.crew_cost), theta_c])
        col_lo = np.concatenate([np.zeros(self.n1), theta_lo])
        col_hi = np.concatenate([np.repeat(net.dock_max.astype(float), H), net.crew_max.astype(float),
                                 np.full(self.nt, INF)])
        D = sp.csc_matrix((np.ones(I * H), (np.repeat(np.arange(I), H), np.arange(I * H))),
                          shape=(I, self.n1 + self.nt))
        self.agg_rows = np.zeros(0, dtype=int)
        blocks_A, blocks_lo, blocks_hi = [D], [np.full(I, -INF)], [net.dock_max.astype(float)]
        extra_c, extra_lo, extra_hi = [], [], []
        nb = o.jensen_blocks
        if nb > 0:
            # Partial Benders decomposition (Crainic, Hewitt, Maggioni & Rei 2021). Scenarios are
            # grouped by national volume; for each group C the recourse LP of its mean demand is kept
            # in the master. Q_s is convex in d (LP right-hand side), so by Jensen
            #   sum_{s in C} p_s Q_s(y,w) >= P_C Q(y, w, mean_C d),
            # which is enforced through the linking row below.
            rec = self.rec
            m2, n2 = rec.A.shape
            T = linking_matrix(rec)
            order = np.argsort(self.inst.scenarios.sum(axis=(1, 2)))
            groups = np.array_split(order, nb)
            n_extra = nb * n2
            ncol = self.n1 + self.nt + n_extra
            blocks_A = [sp.hstack([D, sp.csc_matrix((I, n_extra))])]
            zy, zw = np.zeros((I, H)), np.zeros(I)
            agg_rows = []
            for b, g in enumerate(groups):
                P = self.inst.probs[g].sum()
                dbar = np.tensordot(self.inst.probs[g] / P, self.inst.scenarios[g], 1)
                left = sp.hstack([T, sp.csc_matrix((m2, self.nt + b * n2))])
                right = sp.csc_matrix((m2, (nb - b - 1) * n2))
                blocks_A.append(sp.hstack([left, rec.A, right]))
                lo_b, hi_b = rec.rhs(zy, zw, dbar)
                blocks_lo.append(lo_b); blocks_hi.append(hi_b)
                agg = np.zeros(ncol)
                if o.multicut:
                    agg[self.n1 + g] = self.inst.probs[g]
                    agg[self.n1 + self.nt + b * n2: self.n1 + self.nt + (b + 1) * n2] = -P * rec.cost
                    agg_rows.append(agg)
                else:
                    if b == 0:
                        agg_rows.append(np.zeros(ncol)); agg_rows[0][self.n1] = 1.0
                    agg_rows[0][self.n1 + self.nt + b * n2: self.n1 + self.nt + (b + 1) * n2] = -P * rec.cost
            self.agg_rows = I + nb * m2 + np.arange(len(agg_rows))
            blocks_A.append(sp.csr_matrix(np.vstack(agg_rows)))
            blocks_lo.append(np.zeros(len(agg_rows))); blocks_hi.append(np.full(len(agg_rows), INF))
            extra_c = [np.zeros(n_extra)]
            extra_lo = [np.tile(rec.col_lo, nb)]
            extra_hi = [np.tile(rec.col_hi, nb)]
        A = sp.vstack(blocks_A, format="csc")
        c = np.concatenate([c] + extra_c)
        col_lo = np.concatenate([col_lo] + extra_lo)
        col_hi = np.concatenate([col_hi] + extra_hi)
        self.n_master_cols = A.shape[1]
        self.master = make_highs(c, A, np.concatenate(blocks_lo), np.concatenate(blocks_hi), col_lo, col_hi,
                                 mip_rel_gap=o.master_gap)
        self.n_cuts = 0

    def _set_integer(self, flag: bool):
        n = self.n1
        vt = highspy.HighsVarType.kInteger if flag else highspy.HighsVarType.kContinuous
        self.master.changeColsIntegrality(n, np.arange(n, dtype=np.int32), np.array([vt] * n))

    def _solve_master(self):
        self.master.run()
        if not status_ok(self.master):
            # a master MIP stopped by its time limit is still usable: its incumbent is a valid
            # first-stage candidate and its dual bound a valid lower bound
            info = self.master.getInfo()
            if not (self._integer and info.primal_solution_status == 2):
                raise RuntimeError(f"master failed: {self.master.getModelStatus()}")
        x = np.asarray(self.master.getSolution().col_value)[:self.n1 + self.nt]
        info = self.master.getInfo()
        lb = info.mip_dual_bound if self._integer else info.objective_function_value
        if not np.isfinite(lb) or abs(lb) > 1e20:
            lb = info.objective_function_value
        return x, lb

    def _split(self, x):
        net = self.inst.net
        I, H = net.I, net.H
        return x[:I * H].reshape(I, H), x[I * H:self.n1], x[self.n1:]

    def _add_cuts(self, yk, wk, q, gy, gw, theta, force=False):
        """Add violated optimality cuts at (yk, wk). Returns number of cuts added."""
        p = self.inst.probs
        rows_idx, rows_val, lo = [], [], []
        if self.opts.multicut:
            const = q - (gy.reshape(len(q), -1) @ yk.reshape(-1)) - gw @ wk
            for s in range(len(q)):
                if force or q[s] > theta[s] + 1e-6 * max(1.0, abs(q[s])):
                    idx = np.r_[np.arange(self.n1), self.n1 + s]
                    val = np.r_[-gy[s].reshape(-1), -gw[s], 1.0]
                    rows_idx.append(idx); rows_val.append(val); lo.append(const[s])
        else:
            Q = p @ q
            if force or Q > theta[0] + 1e-6 * max(1.0, abs(Q)):
                gya = np.tensordot(p, gy, 1).reshape(-1)
                gwa = p @ gw
                const = Q - gya @ yk.reshape(-1) - gwa @ wk
                rows_idx.append(np.r_[np.arange(self.n1), self.n1])
                rows_val.append(np.r_[-gya, -gwa, 1.0]); lo.append(const)
        if not rows_idx:
            return 0
        starts = np.cumsum([0] + [len(r) for r in rows_idx[:-1]]).astype(np.int32)
        idx = np.concatenate(rows_idx).astype(np.int32)
        val = np.concatenate(rows_val)
        # drop numerically zero coefficients
        keep = np.abs(val) > 1e-9
        if not keep.all():
            new_idx, new_val, new_starts, pos = [], [], [], 0
            for r_i, r_v in zip(rows_idx, rows_val):
                k = np.abs(r_v) > 1e-9
                new_starts.append(pos); new_idx.append(r_i[k]); new_val.append(r_v[k]); pos += k.sum()
            starts = np.array(new_starts, dtype=np.int32)
            idx = np.concatenate(new_idx).astype(np.int32)
            val = np.concatenate(new_val)
        n = len(lo)
        self.master.addRows(n, np.array(lo, dtype=np.float64), np.full(n, INF), len(idx), starts, idx, val)
        self.n_cuts += n
        return n

    # ------------------------------------------------------------------
    def solve(self) -> BendersResult:
        o, inst, net = self.opts, self.inst, self.inst.net
        t_start = time.time()
        t_master = t_sub = 0.0
        hist = []
        best = (np.inf, None, None, None)
        lb = -np.inf
        it = lp_it = 0

        def evaluate(yk, wk):
            nonlocal t_sub
            t0 = time.time()
            q, gy, gw = self.pool.evaluate(yk, wk)
            t_sub += time.time() - t0
            return q, gy, gw

        # ---------------- phase 1: LP relaxation of the master (+ in-out stabilisation)
        self._integer = False
        if o.lp_phase:
            self._set_integer(False)
            core = None
            lp_ub = np.inf
            stall = 0
            while lp_it < min(o.lp_max_iter, o.max_iter) and time.time() - t_start < o.time_limit:
                t0 = time.time()
                x, lb_lp = self._solve_master()
                t_master += time.time() - t0
                yk, wk, theta = self._split(x)
                if core is None:
                    core = (yk.copy(), wk.copy())
                alpha = o.inout_alpha if (o.inout and stall == 0) else 1.0
                ys = alpha * yk + (1 - alpha) * core[0]
                ws = alpha * wk + (1 - alpha) * core[1]
                q, gy, gw = evaluate(ys, ws)
                # value of the separation point gives an LP upper bound
                lp_ub = min(lp_ub, first_stage_cost(net, ys, ws) + inst.probs @ q)
                theta_s = self._theta_at(ys, ws, x) if alpha < 1 else theta
                added = self._add_cuts(ys, ws, q, gy, gw, theta_s)
                it += 1; lp_it += 1
                gap = (lp_ub - lb_lp) / abs(lp_ub)
                hist.append(dict(it=it, phase="lp", lb=lb_lp, ub=lp_ub, cuts=self.n_cuts, t=time.time() - t_start))
                if o.verbose:
                    print(f"[LP {it:3d}] lb={lb_lp:,.0f} ub={lp_ub:,.0f} gap={gap:.2e} cuts={self.n_cuts} a={alpha}")
                if o.inout:
                    core = (ys, ws)
                if added == 0:
                    if alpha < 1:
                        stall = 1          # no cut at the separation point: separate at the master point
                        continue
                    break
                stall = 0
                if gap <= o.lp_tol:
                    break
            lb = lb_lp

        # ---------------- phase 2: integer master
        self._integer = True
        if len(self.agg_rows) and not o.jensen_in_mip:
            # deactivate the Jensen linking rows: the blocks become free and are removed by presolve;
            # cuts generated in phase 1 remain
            n = len(self.agg_rows)
            set_row_bounds(self.master, np.full(n, -INF), np.full(n, INF), self.agg_rows)
        self._set_integer(True)
        mgap = o.master_gap
        self.master.setOptionValue("time_limit", o.master_time)
        mip_it = 0
        while (mip_it < o.max_iter and time.time() - t_start < o.time_limit) or best[1] is None:
            mip_it += 1
            # the master only has to be solved as accurately as the current Benders gap requires;
            # its *dual bound* is used as LB, so a loose master gap never invalidates the bound
            self.master.setOptionValue("mip_rel_gap", mgap)
            if best[1] is not None:
                # incumbent with theta = its recourse values satisfies every cut (cuts underestimate Q_s)
                th = best[3] if o.multicut else np.array([inst.probs @ best[3]])
                self._hint(np.r_[best[1].reshape(-1), best[2], th])
            t0 = time.time()
            x, lb_k = self._solve_master()
            t_master += time.time() - t0
            lb = max(lb, lb_k)
            yk, wk, theta = self._split(x)
            yk, wk = np.round(yk), np.round(wk)
            q, gy, gw = evaluate(yk, wk)
            ub_k = first_stage_cost(net, yk, wk) + inst.probs @ q
            if ub_k < best[0]:
                best = (ub_k, yk, wk, q)
            added = self._add_cuts(yk, wk, q, gy, gw, theta)
            it += 1
            gap = (best[0] - lb) / abs(best[0])
            hist.append(dict(it=it, phase="mip", lb=lb, ub=best[0], cuts=self.n_cuts, t=time.time() - t_start))
            if o.verbose:
                print(f"[MIP {it:3d}] lb={lb:,.0f} ub={best[0]:,.0f} gap={gap:.2e} cuts={self.n_cuts}")
            if gap <= o.tol:
                break
            if added == 0:
                if mgap <= o.tol / 4:
                    break
                mgap = max(o.tol / 4, mgap / 4)   # no violated cut: the master must be solved tighter
            else:
                mgap = max(o.tol / 4, min(mgap, gap / 4))

        res = BendersResult(y=best[1], w=best[2], ub=best[0], lb=lb, gap=(best[0] - lb) / abs(best[0]),
                            iterations=it, lp_iterations=lp_it, cuts=self.n_cuts, time=time.time() - t_start,
                            time_master=t_master, time_sub=t_sub, recourse=best[3], history=hist)
        if self._own_pool:
            self.pool.close()
        return res

    def _theta_at(self, ys, ws, x):
        """Current master approximation of theta at the separation point (max over existing cuts).

        Using the master's theta (taken at the master point) for the violation test at a
        different point would be wrong; we conservatively force all cuts to be added by
        returning -inf, which is cheap because separated cuts are always tight there.
        """
        return np.full(self.nt, -np.inf)

    def _hint(self, xs):
        """Pass the incumbent (y, w, theta) as a partial MIP start; HiGHS completes the rest."""
        idx = np.arange(len(xs), dtype=np.int32)
        try:
            self.master.setSolution(len(xs), idx, np.asarray(xs, dtype=np.float64))
        except Exception:
            pass
