import numpy as np
import pytest

from fulfillopt.demand import simulate
from fulfillopt.network import synthetic
from fulfillopt.online.simulator import OnlineSimulator, arrivals, intraday_profile
from fulfillopt.planning.model import PlanningInstance, position_inventory
from fulfillopt.planning.solve import solve_benders


@pytest.fixture(scope="module")
def setup():
    net = synthetic(5, 6, 40, 3, seed=8)
    hist = simulate(net, "2024-01-01", "2024-03-31", seed=2)
    t = 70
    inv = position_inventory(net, hist.oracle_mean(t))
    plan = solve_benders(PlanningInstance(net, inv, hist.oracle_samples(t, 20, np.random.default_rng(0))), tol=1e-3)
    return net, hist, t, inv, plan


def test_profile_and_arrivals_conserve_units():
    assert abs(intraday_profile().sum() - 1) < 1e-12
    d = np.full((7, 3), 1000.0)
    arr = arrivals(d, np.random.default_rng(0))
    assert np.allclose(arr.sum(axis=0), d)


@pytest.mark.parametrize("policy", ["greedy", "bid_static", "bid_resolve", "lp_tracking"])
def test_policies_respect_capacity_and_hindsight_bound(setup, policy):
    net, hist, t, inv, plan = setup
    sim = OnlineSimulator(net, inv, plan.y, plan.w)
    rng = np.random.default_rng(3)
    d = hist.oracle_samples(t, 1, rng)[0]
    arr = arrivals(d, rng)
    hs = sim.hindsight(d)
    r = sim.run(arr, hist.oracle_mean(t), policy)
    assert r.cost >= hs.cost * (1 - 1e-9)
    assert abs(r.units - d.sum()) < 1e-6 * d.sum()
    # overtime never exceeds the cap
    p = net.params
    assert r.overtime <= p.overtime_frac * p.crew_capacity * plan.w.sum() + 1e-6
