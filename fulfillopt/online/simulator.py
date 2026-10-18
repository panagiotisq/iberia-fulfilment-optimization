r"""Intraday online order assignment under a fixed day-ahead capacity plan.

Units arrive over N_SLOTS intraday slots. When a batch of units of category k for zone j
arrives it must be irrevocably assigned to an FC i (or left unmet). Resources are consumed
as in the recourse LP:
    inventory Inv[i,k];  regular labour pi*w_i, then overtime (premium, capped at rho*pi*w_i);
    scheduled lane capacity Q*y[i,h], then spot capacity (premium, unlimited).
The *immediate* marginal cost of a unit (i, j, k) is
    c_ijk = processing_i + lastmile_j + lam*late(i,j) + [overtime premium] + [spot premium],
where the bracketed terms apply once the regular resource is exhausted.

Policies differ only in how they score FCs:
    greedy          argmin immediate cost (myopic)
    bid_price       argmin immediate cost + sum_r a_r(i,k) mu_r, mu = shadow prices of the
                    fluid LP over the *expected remaining* demand (Acimovic & Graves 2015);
                    static: LP solved once at the start of the day; resolve: re-solved
                    every `resolve_every` slots with on-line demand learning
    lp_tracking     route each batch to FCs in proportion to the latest fluid-LP flows
                    (fluid-tracking / probabilistic routing, cf. Jasin & Sinha 2015)
The hindsight benchmark solves the recourse LP with the realised demand of the whole
day: no non-anticipative policy can do better with the same plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..network import Network
from ..planning.benders import SubproblemSolver
from ..planning.model import Recourse

N_SLOTS = 96


def intraday_profile(n_slots: int = N_SLOTS) -> np.ndarray:
    """Share of the day's units arriving in each slot (night trough, lunch and evening peaks)."""
    hr = (np.arange(n_slots) + 0.5) * 24 / n_slots
    prof = (0.15 + 0.9 * np.exp(-0.5 * ((hr - 12.5) / 2.2) ** 2) + 1.2 * np.exp(-0.5 * ((hr - 21.0) / 2.0) ** 2)
            + 0.5 * np.exp(-0.5 * ((hr - 9.5) / 1.5) ** 2))
    prof[hr < 6] *= 0.35
    return prof / prof.sum()


def arrivals(d: np.ndarray, rng: np.random.Generator, n_slots: int = N_SLOTS, conc: float = 300.0) -> np.ndarray:
    """Split realised daily demand d (J,K) into slots (n_slots, J, K); each series gets a
    Dirichlet-perturbed version of the national intraday profile."""
    prof = intraday_profile(n_slots)
    J, K = d.shape
    shares = rng.dirichlet(conc * prof, size=(J, K))                  # (J,K,n_slots)
    return np.moveaxis(shares * d[..., None], -1, 0)


@dataclass
class State:
    inv: np.ndarray     # (I,K)
    lab: np.ndarray     # (I,) regular labour units left
    ot: np.ndarray      # (I,) overtime labour units left
    lane: np.ndarray    # (I,H) scheduled package-equivalents left


@dataclass
class DayResult:
    policy: str
    cost: float
    units: float
    unmet: float
    late: float
    overtime: float
    spot: float
    lp_solves: int

    @property
    def on_time(self):
        return 1 - self.late / self.units


class OnlineSimulator:
    def __init__(self, net: Network, inventory: np.ndarray, y: np.ndarray, w: np.ndarray,
                 late_penalty: float | None = None):
        self.net = net
        self.inventory = inventory
        self.y, self.w = y, w
        p = net.params
        self.lam = p.late_penalty if late_penalty is None else late_penalty
        self.base = (net.fc_var_cost[:, None] + net.lastmile_cost[None, :]
                     + self.lam * net.late_ij)                                   # (I,J)
        self.unmet_cost = p.unmet_penalty + self.lam
        self.rec = Recourse(net, inventory)
        self.rec.set_late_penalty(self.lam)
        self.sp = SubproblemSolver(self.rec)

    def initial_state(self) -> State:
        p = self.net.params
        return State(self.inventory.astype(float).copy(), p.crew_capacity * self.w.astype(float),
                     p.overtime_frac * p.crew_capacity * self.w.astype(float),
                     p.truck_capacity * self.y.astype(float))

    # ------------------------------------------------------------------ fluid LP
    def fluid_lp(self, st: State, d_rem: np.ndarray):
        """Recourse LP over remaining expected demand with remaining capacities.

        Returns shadow prices (non-negative opportunity costs) and the flow matrix x[i,j,k].
        """
        rec, net, p = self.rec, self.net, self.net.params
        lo, hi = rec.row_lo0.copy(), rec.row_hi0.copy()
        hi[rec.r_inv] = st.inv[rec.inv_pairs[:, 0], rec.inv_pairs[:, 1]]
        hi[rec.r_lab] = st.lab
        hi[rec.r_ot] = st.ot
        hi[rec.r_lane] = st.lane.reshape(-1)
        lo[rec.r_dem] = hi[rec.r_dem] = d_rem.reshape(-1)
        from ..planning.highs_util import set_row_bounds
        set_row_bounds(self.sp.h, lo, hi)
        self.sp.h.run()
        sol = self.sp.h.getSolution()
        dual = -np.asarray(sol.row_dual)
        mu_inv = np.zeros((net.I, net.K))
        mu_inv[rec.inv_pairs[:, 0], rec.inv_pairs[:, 1]] = dual[rec.r_inv]
        mu = dict(inv=mu_inv, lab=dual[rec.r_lab], ot=dual[rec.r_ot],
                  lane=dual[rec.r_lane].reshape(net.I, net.H))
        x = np.zeros((net.I, net.J, net.K))
        x[rec.arc_i, rec.arc_j, rec.arc_k] = np.asarray(sol.col_value)[rec.c_x]
        return mu, x

    # ------------------------------------------------------------------ simulation
    def run(self, arr: np.ndarray, forecast: np.ndarray, policy: str, resolve_every: int = 4,
            learn: bool = True, rng: np.random.Generator | None = None) -> DayResult:
        """Simulate one day. arr: (n_slots, J, K) arriving units; forecast: (J,K) expected day demand."""
        net, p = self.net, self.net.params
        rng = rng or np.random.default_rng(0)
        n_slots = arr.shape[0]
        prof = intraday_profile(n_slots)
        cum_prof = np.concatenate([[0], np.cumsum(prof)])
        st = self.initial_state()
        lab_k, vol_k = net.cat_labour, net.cat_volume
        hz = net.zone_sc
        spot = net.spot_cost
        ot_c = net.overtime_cost
        mu, xflow = None, None
        cost = unmet = late = ot_used = spot_used = 0.0
        lp_solves = 0
        seen = np.zeros_like(forecast)
        uses_lp = policy in ("bid_static", "bid_resolve", "lp_tracking")

        for t in range(n_slots):
            if uses_lp and (mu is None or (policy != "bid_static" and t % resolve_every == 0)):
                rem_share = 1 - cum_prof[t]
                d_rem = forecast * rem_share
                if learn and policy != "bid_static" and t > 0:
                    # multiplicative demand learning per category, shrunk towards the forecast
                    exp_so_far = forecast * cum_prof[t]
                    k_ratio = (seen.sum(0) + 2000) / (exp_so_far.sum(0) + 2000)
                    d_rem = d_rem * k_ratio[None, :]
                mu, xflow = self.fluid_lp(st, np.maximum(d_rem, 0))
                lp_solves += 1
            batch = arr[t]
            seen += batch
            jj, kk = np.nonzero(batch > 1e-9)
            order = rng.permutation(len(jj))
            for idx in order:
                j, k = jj[idx], kk[idx]
                n = batch[j, k]
                h = hz[j]
                if policy == "lp_tracking":
                    flows = xflow[:, j, k]
                    tot = flows.sum()
                    if tot > 1e-9:
                        targets = n * flows / tot
                        n_left = 0.0
                        for i in np.nonzero(targets > 1e-9)[0]:
                            got, c_, l_, o_, s_ = self._assign_fixed(st, i, j, k, targets[i], h)
                            cost += c_; late += l_; ot_used += o_; spot_used += s_
                            n_left += targets[i] - got
                        n = n_left
                    if n <= 1e-9:
                        continue
                # score-based assignment (greedy/bid price, and fallback of lp_tracking)
                while n > 1e-9:
                    feas = st.inv[:, k] > 1e-9
                    lab_reg = st.lab >= lab_k[k] - 1e-9
                    lab_ok = lab_reg | (st.ot >= lab_k[k] - 1e-9)
                    lane_reg = st.lane[:, h] >= vol_k[k] - 1e-9
                    feas &= lab_ok & net.stocks[:, k]
                    imm = (self.base[:, j] + np.where(lab_reg, 0.0, ot_c * lab_k[k])
                           + np.where(lane_reg, 0.0, spot[:, h] * vol_k[k]))
                    score = imm.copy()
                    if policy in ("bid_static", "bid_resolve"):
                        score += (mu["inv"][:, k] + lab_k[k] * np.where(lab_reg, mu["lab"], mu["ot"])
                                  + vol_k[k] * np.where(lane_reg, mu["lane"][:, h], 0.0))
                    score = np.where(feas, score, np.inf)
                    i = int(np.argmin(score))
                    if not np.isfinite(score[i]) or imm[i] >= self.unmet_cost:
                        cost += self.unmet_cost * n; unmet += n; late += n
                        break
                    # largest amount that keeps the same cost regime at FC i
                    cap = st.inv[i, k]
                    cap = min(cap, st.lab[i] / lab_k[k] if lab_reg[i] else st.ot[i] / lab_k[k])
                    if lane_reg[i]:
                        cap = min(cap, st.lane[i, h] / vol_k[k])
                    amt = min(n, max(cap, 1e-6))
                    got, c_, l_, o_, s_ = self._assign_fixed(st, i, j, k, amt, h)
                    cost += c_; late += l_; ot_used += o_; spot_used += s_
                    n -= got
                    if got <= 1e-12:
                        cost += self.unmet_cost * n; unmet += n; late += n
                        break
        units = float(arr.sum())
        return DayResult(policy, cost, units, unmet, late, ot_used, spot_used, lp_solves)

    def _assign_fixed(self, st: State, i, j, k, amt, h):
        """Assign up to `amt` units of (j,k) to FC i, using regular capacity first, then
        overtime / spot. Returns (assigned, cost, late, overtime, spot)."""
        net = self.net
        lk, vk = net.cat_labour[k], net.cat_volume[k]
        amt = min(amt, st.inv[i, k])
        lab_units = amt * lk
        reg = min(lab_units, st.lab[i])
        ot = min(lab_units - reg, st.ot[i])
        amt = (reg + ot) / lk
        if amt <= 1e-12:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        lab_units = amt * lk
        reg = min(lab_units, st.lab[i])
        ot = lab_units - reg
        vol = amt * vk
        lane_reg = min(vol, st.lane[i, h])
        spot = vol - lane_reg
        st.inv[i, k] -= amt
        st.lab[i] -= reg
        st.ot[i] -= ot
        st.lane[i, h] -= lane_reg
        c = amt * self.base[i, j] + ot * net.overtime_cost + spot * net.spot_cost[i, h]
        late = amt if net.late_ij[i, j] else 0.0
        return amt, c, late, ot, spot

    def hindsight(self, d: np.ndarray) -> DayResult:
        obj, _, _, prim = self.sp.solve(self.y, self.w, d, want_primal=True)
        k = self.rec.kpis(prim, d)
        return DayResult("hindsight", obj, k["units"], k["unmet"], k["late_units"], k["overtime_lu"],
                         k["spot_pe"], 1)
