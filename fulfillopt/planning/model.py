r"""Two-stage stochastic capacity planning with order assignment as recourse.

First stage (decided the day before; integer):
    y[i,h] >= 0  scheduled line-haul trucks on lane FC i -> SC h        cost F[i,h]
    w[i]   >= 0  labour crews scheduled at FC i (w <= W_i)              cost L
    sum_h y[i,h] <= D_i                                                 (outbound docks)

Second stage, for every demand scenario d^s (LP, complete recourse):
    min  sum_a c_a x_a + c^ot sum_i o_i + sum_ih s_ih z_ih + sum_jk (p + lam) u_jk
    s.t. sum_{j} x[i,j,k]               <= Inv[i,k]          (inventory)        [rows R_inv]
         sum_{jk} l_k x[i,j,k] - o_i     <= pi w_i            (regular labour)   [rows R_lab]
         o_i                             <= rho pi w_i        (overtime cap)     [rows R_ot]
         sum_{j:h(j)=h,k} v_k x[i,j,k] - z_ih <= Q y[i,h]     (truck capacity)   [rows R_lane]
         sum_i x[i,j,k] + u_jk           =  d^s_jk            (demand)           [rows R_dem]
    with c_a = processing_i + last-mile_j + lam * late(i,j).

The recourse right-hand side is affine in (y, w, d):  rhs = r0 + T_y y + T_w w + E d, so the
Benders optimality cut of scenario s at (y^, w^) is
    theta_s >= Q_s(y^, w^) + g_y.(y - y^) + g_w.(w - w^),
    g_y[i,h] = Q * pi_lane[i,h],   g_w[i] = pi * (pi_lab[i] + rho * pi_ot[i]),
where pi are the (non-positive) optimal duals of the <= rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from ..network import Network


def position_inventory(net: Network, expected: np.ndarray, cover: float = 1.35, imbalance_sd: float = 0.25,
                       seed: int = 11) -> np.ndarray:
    """Available-to-ship inventory Inv[i,k] for one day.

    National cover of `cover` days of expected demand per category, split across the FCs
    that stock the category in proportion to site size, distorted by a persistent
    (seeded) imbalance so that inventory is not perfectly aligned with demand.
    """
    rng = np.random.default_rng(seed)
    size = net.crew_max.astype(float)[:, None] * net.stocks
    distort = np.exp(imbalance_sd * rng.standard_normal(size.shape))
    w = size * distort
    w /= w.sum(axis=0, keepdims=True)
    return cover * w * expected.sum(axis=0)[None, :]


@dataclass
class PlanningInstance:
    net: Network
    inventory: np.ndarray          # (I,K)
    scenarios: np.ndarray          # (S,J,K) demand scenarios
    probs: np.ndarray | None = None
    label: str = ""

    def __post_init__(self):
        S = self.scenarios.shape[0]
        if self.probs is None:
            self.probs = np.full(S, 1.0 / S)

    @property
    def S(self):
        return self.scenarios.shape[0]

    def with_scenarios(self, scen, probs=None, label=None) -> "PlanningInstance":
        return PlanningInstance(self.net, self.inventory, scen, probs, label or self.label)


@dataclass
class Recourse:
    """Sparse matrix form of the second-stage LP (shared by all scenarios)."""

    net: Network
    inventory: np.ndarray
    A: sp.csc_matrix = field(init=False)
    cost: np.ndarray = field(init=False)
    row_lo0: np.ndarray = field(init=False)
    row_hi0: np.ndarray = field(init=False)

    def __post_init__(self):
        net, p = self.net, self.net.params
        I, H, J, K = net.I, net.H, net.J, net.K
        ii, jj, kk = np.nonzero(np.broadcast_to(net.stocks[:, None, :], (I, J, K)))
        self.arc_i, self.arc_j, self.arc_k = ii, jj, kk
        self.arc_h = net.zone_sc[jj]
        nA = len(ii)
        late = net.late_ij[ii, jj]
        self.arc_late = late
        # column blocks
        self.c_x = slice(0, nA)
        self.c_o = slice(nA, nA + I)
        self.c_z = slice(nA + I, nA + I + I * H)
        self.c_u = slice(nA + I + I * H, nA + I + I * H + J * K)
        self.n_col = nA + I + I * H + J * K
        # row blocks
        inv_pairs = np.argwhere(net.stocks)
        self.inv_pairs = inv_pairs
        inv_id = -np.ones((I, K), dtype=int)
        inv_id[inv_pairs[:, 0], inv_pairs[:, 1]] = np.arange(len(inv_pairs))
        nInv = len(inv_pairs)
        self.r_inv = slice(0, nInv)
        self.r_lab = slice(nInv, nInv + I)
        self.r_ot = slice(nInv + I, nInv + 2 * I)
        self.r_lane = slice(nInv + 2 * I, nInv + 2 * I + I * H)
        self.r_dem = slice(nInv + 2 * I + I * H, nInv + 2 * I + I * H + J * K)
        self.n_row = self.r_dem.stop

        rows, cols, vals = [], [], []
        ax = np.arange(nA)
        # x columns
        rows += [inv_id[ii, kk], self.r_lab.start + ii, self.r_lane.start + ii * H + self.arc_h,
                 self.r_dem.start + jj * K + kk]
        cols += [ax] * 4
        vals += [np.ones(nA), net.cat_labour[kk], net.cat_volume[kk], np.ones(nA)]
        # o columns
        oi = np.arange(I)
        rows += [self.r_lab.start + oi, self.r_ot.start + oi]
        cols += [self.c_o.start + oi] * 2
        vals += [-np.ones(I), np.ones(I)]
        # z columns
        zi = np.arange(I * H)
        rows += [self.r_lane.start + zi]
        cols += [self.c_z.start + zi]
        vals += [-np.ones(I * H)]
        # u columns
        ui = np.arange(J * K)
        rows += [self.r_dem.start + ui]
        cols += [self.c_u.start + ui]
        vals += [np.ones(J * K)]
        self.A = sp.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                               shape=(self.n_row, self.n_col))
        self.A.sort_indices()

        self.cost_x_base = net.fc_var_cost[ii] + net.lastmile_cost[jj]
        self.set_late_penalty(p.late_penalty)

        inf = np.inf
        self.row_lo0 = np.full(self.n_row, -inf)
        self.row_hi0 = np.zeros(self.n_row)
        self.row_hi0[self.r_inv] = self.inventory[inv_pairs[:, 0], inv_pairs[:, 1]]
        self.col_lo = np.zeros(self.n_col)
        self.col_hi = np.full(self.n_col, inf)

    # ------------------------------------------------------------------
    def set_late_penalty(self, lam: float):
        net, p = self.net, self.net.params
        self.lam = lam
        c = np.empty(self.n_col)
        c[self.c_x] = self.cost_x_base + lam * self.arc_late
        c[self.c_o] = net.overtime_cost
        c[self.c_z] = net.spot_cost.reshape(-1)
        c[self.c_u] = p.unmet_penalty + lam
        self.cost = c

    def rhs(self, y: np.ndarray, w: np.ndarray, d: np.ndarray):
        """Row bounds for first-stage decision (y (I,H), w (I,)) and demand d (J,K)."""
        p = self.net.params
        lo, hi = self.row_lo0.copy(), self.row_hi0.copy()
        hi[self.r_lab] = p.crew_capacity * w
        hi[self.r_ot] = p.overtime_frac * p.crew_capacity * w
        hi[self.r_lane] = p.truck_capacity * y.reshape(-1)
        dd = d.reshape(-1)
        lo[self.r_dem] = dd
        hi[self.r_dem] = dd
        return lo, hi

    def cut_from_duals(self, row_dual: np.ndarray):
        """Subgradient of the recourse value w.r.t. (y, w) from row duals (HiGHS sign convention)."""
        p = self.net.params
        gy = p.truck_capacity * row_dual[self.r_lane]
        gw = p.crew_capacity * (row_dual[self.r_lab] + p.overtime_frac * row_dual[self.r_ot])
        return gy, gw

    def lower_bound(self, d: np.ndarray) -> float:
        """Trivial valid lower bound on Q_s: every unit is either shipped at least at its
        cheapest arc cost or left unmet."""
        net = self.net
        cmin = np.full((net.J, net.K), self.cost[self.c_u][0])
        np.minimum.at(cmin, (self.arc_j, self.arc_k), self.cost[self.c_x])
        return float((cmin * d).sum())

    # ------------------------------------------------------------------
    def kpis(self, sol: np.ndarray, d: np.ndarray) -> dict:
        """Decompose a second-stage primal solution into business KPIs."""
        net, p = self.net, self.net.params
        x, o, z, u = sol[self.c_x], sol[self.c_o], sol[self.c_z], sol[self.c_u]
        units = d.sum()
        late_units = float(x[self.arc_late].sum() + u.sum())
        spot_pe = z.reshape(net.I, net.H)
        co2_spot = float((spot_pe * net.lane_km / p.truck_capacity * p.co2_truck_per_km * p.co2_spot_factor).sum())
        pt_zone = net.zone_country == "PT"
        pt_arc = pt_zone[self.arc_j]
        late_pt = float(x[self.arc_late & pt_arc].sum() + u.reshape(net.J, net.K)[pt_zone].sum())
        units_pt = float(d[pt_zone].sum())
        return dict(
            units=float(units),
            shipped=float(x.sum()),
            unmet=float(u.sum()),
            late_units=late_units,
            on_time=1.0 - late_units / max(units, 1.0),
            overtime_lu=float(o.sum()),
            spot_pe=float(z.sum()),
            cost_var=float((self.cost_x_base * x).sum()),
            cost_overtime=float(net.overtime_cost * o.sum()),
            cost_spot=float((net.spot_cost.reshape(-1) * z).sum()),
            cost_unmet=float(p.unmet_penalty * u.sum()),
            co2_spot=co2_spot,
            co2_lastmile=float(p.co2_lastmile_per_unit * x.sum()),
            late_pt=late_pt,
            units_pt=units_pt,
        )


def first_stage_cost(net: Network, y: np.ndarray, w: np.ndarray) -> float:
    return float((net.truck_cost * y).sum() + net.params.crew_cost * w.sum())


def first_stage_co2(net: Network, y: np.ndarray) -> float:
    return float((net.lane_km * y).sum() * net.params.co2_truck_per_km)


def linking_matrix(rec: Recourse) -> sp.csr_matrix:
    """T such that the recourse rows read  A x - T (y, w) <= rhs(0, 0, d)  ->  A x + T(y,w) <= rhs0.

    Entries: -Q on lane rows (y), -pi on labour rows and -rho pi on overtime rows (w).
    """
    net, p = rec.net, rec.net.params
    I, H = net.I, net.H
    Tr = [np.arange(rec.r_lane.start, rec.r_lane.stop), np.arange(rec.r_lab.start, rec.r_lab.stop),
          np.arange(rec.r_ot.start, rec.r_ot.stop)]
    Tc = [np.arange(I * H), I * H + np.arange(I), I * H + np.arange(I)]
    Tv = [np.full(I * H, -p.truck_capacity), np.full(I, -p.crew_capacity),
          np.full(I, -p.overtime_frac * p.crew_capacity)]
    return sp.csr_matrix((np.concatenate(Tv), (np.concatenate(Tr), np.concatenate(Tc))),
                         shape=(rec.n_row, I * H + I))


def extensive_form(inst: PlanningInstance, rec: Recourse, integer: bool = True,
                   eps_late: float | None = None):
    """Deterministic equivalent as (c, A, row_lo, row_hi, col_lo, col_hi, integrality).

    Column order: y (I*H), w (I), then one second-stage block per scenario.
    If eps_late is given, adds the epsilon-constraint
        sum_s p_s (late_x^s + u^s) <= eps_late * sum_s p_s * total_demand^s.
    """
    net, p = inst.net, inst.net.params
    I, H = net.I, net.H
    n1 = I * H + I
    S = inst.S
    A2 = rec.A
    m2, n2 = A2.shape
    T = linking_matrix(rec)
    # first-stage dock rows
    D = sp.csr_matrix((np.ones(I * H), (np.repeat(np.arange(I), H), np.arange(I * H))), shape=(I, n1))
    blocks = [[D] + [None] * S]
    for s in range(S):
        row = [T] + [None] * S
        row[1 + s] = A2
        blocks.append(row)
    A = sp.bmat(blocks, format="csc")
    c = np.concatenate([net.truck_cost.reshape(-1), np.full(I, p.crew_cost)]
                       + [inst.probs[s] * rec.cost for s in range(S)])
    lo, hi = [np.full(I, -np.inf)], [net.dock_max.astype(float)]
    zeros_y, zeros_w = np.zeros((I, H)), np.zeros(I)
    for s in range(S):
        l, h = rec.rhs(zeros_y, zeros_w, inst.scenarios[s])
        lo.append(l); hi.append(h)
    row_lo, row_hi = np.concatenate(lo), np.concatenate(hi)
    col_lo = np.zeros(A.shape[1])
    col_hi = np.concatenate([np.repeat(net.dock_max.astype(float), H), net.crew_max.astype(float)]
                            + [rec.col_hi] * S)
    if eps_late is not None:
        late_vec = np.zeros(n2)
        late_vec[rec.c_x] = rec.arc_late
        late_vec[rec.c_u] = 1.0
        row = sp.hstack([sp.csr_matrix((1, n1))] + [sp.csr_matrix(inst.probs[s] * late_vec) for s in range(S)])
        A = sp.vstack([A, row], format="csc")
        tot = float((inst.probs * inst.scenarios.sum(axis=(1, 2))).sum())
        row_lo = np.append(row_lo, -np.inf)
        row_hi = np.append(row_hi, eps_late * tot)
    integrality = np.zeros(A.shape[1], dtype=np.int32)
    if integer:
        integrality[:n1] = 1
    return c, A, row_lo, row_hi, col_lo, col_hi, integrality
