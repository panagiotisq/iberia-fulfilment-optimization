import numpy as np

from fulfillopt.forecast.models import QUANTILES
from fulfillopt.forecast.scenarios import pit_scores, quantile_function, sample_scenarios
from fulfillopt.network import iberia
from fulfillopt.demand import simulate


def test_quantile_function_roundtrip():
    rng = np.random.default_rng(0)
    q = np.sort(rng.gamma(5, 100, (50, len(QUANTILES))), axis=1)
    z = rng.standard_normal((1, 50)) * 1.5
    y = quantile_function(q[None], z)[0]
    pos = y > 0
    assert np.allclose(pit_scores(q[None], y[None])[0][pos], np.clip(z[0][pos], -4, 4), atol=1e-6)


def test_quantile_function_hits_knots():
    q = np.array([[10, 20, 30, 40, 50, 60, 70.]])
    from scipy.stats import norm
    z = norm.ppf(QUANTILES)[:, None]
    assert np.allclose(quantile_function(q[None], z)[..., 0], q[0])


def test_copula_sampling_marginals():
    rng = np.random.default_rng(1)
    q = np.tile(np.array([80, 90, 95, 100, 105, 110, 120.]), (3, 1))
    C = np.array([[1, .8, 0], [.8, 1, 0], [0, 0, 1.]])
    s = sample_scenarios(q, 20000, rng, C)
    assert abs(np.median(s[:, 0]) - 100) < 1
    assert abs(np.corrcoef(s.T)[0, 1] - 0.8) < 0.05
    assert abs(np.corrcoef(s.T)[0, 2]) < 0.05


def test_simulation_is_reproducible_and_oracle_unbiased():
    net = iberia()
    h1 = simulate(net, "2024-01-01", "2024-02-29", seed=5)
    h2 = simulate(net, "2024-01-01", "2024-02-29", seed=5)
    assert np.array_equal(h1.y, h2.y)
    s = h1.oracle_samples(40, 4000, np.random.default_rng(0))
    assert abs(s.sum(axis=(1, 2)).mean() / h1.oracle_mean(40).sum() - 1) < 0.01
