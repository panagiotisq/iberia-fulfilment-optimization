import numpy as np
import pytest

from fulfillopt.demand import simulate
from fulfillopt.network import iberia, synthetic
from fulfillopt.planning.benders import SubproblemSolver
from fulfillopt.planning.model import PlanningInstance, Recourse, position_inventory
from fulfillopt.planning.solve import evaluate_plan, solve_benders, solve_extensive


@pytest.fixture(scope="module")
def small():
    net = synthetic(4, 5, 30, 3, seed=3)
    hist = simulate(net, "2024-01-01", "2024-03-31", seed=1)
    t = 60
    mean = hist.oracle_mean(t)
    inv = position_inventory(net, mean)
    scen = hist.oracle_samples(t, 6, np.random.default_rng(0))
    return PlanningInstance(net, inv, scen)


def test_cut_is_valid_underestimator(small):
    rec = Recourse(small.net, small.inventory)
    sp = SubproblemSolver(rec)
    rng = np.random.default_rng(1)
    I, H = small.net.I, small.net.H
    for _ in range(5):
        y0 = rng.integers(0, 8, (I, H)).astype(float)
        w0 = rng.integers(0, 10, I).astype(float)
        q0, gy, gw, _ = sp.solve(y0, w0, small.scenarios[0])
        gy = gy.reshape(I, H)
        for _ in range(5):
            y1 = rng.integers(0, 8, (I, H)).astype(float)
            w1 = rng.integers(0, 10, I).astype(float)
            q1, *_ = sp.solve(y1, w1, small.scenarios[0])
            assert q1 >= q0 + (gy * (y1 - y0)).sum() + gw @ (w1 - w0) - 1e-6 * abs(q0)


def test_benders_matches_extensive_form(small):
    ef = solve_extensive(small, mip_gap=1e-6, time_limit=300)
    for kw in [dict(), dict(jensen_in_mip=True), dict(lp_phase=False, jensen_blocks=0),
               dict(inout=False, jensen_blocks=2)]:
        bd = solve_benders(small, tol=1e-4, **kw)
        # both are within their tolerance of the same optimum
        assert abs(bd.objective - ef.objective) / ef.objective < 1.2e-4, kw
        assert bd.bound <= ef.objective * (1 + 1e-6)


def test_evaluation_reproduces_in_sample_objective(small):
    bd = solve_benders(small, tol=1e-3)
    df = evaluate_plan(small.net, small.inventory, bd.y, bd.w, small.scenarios)
    assert abs(df.total.mean() - bd.objective) / bd.objective < 1e-6


def test_iberia_recourse_dimensions():
    net = iberia()
    inv = np.ones((net.I, net.K)) * 1e4
    rec = Recourse(net, inv)
    assert rec.A.shape == (rec.n_row, rec.n_col)
    assert rec.n_row == net.stocks.sum() + 2 * net.I + net.I * net.H + net.J * net.K
