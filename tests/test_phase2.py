import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

from ssr.risk import coverage_tests as ct, garchx, rolling


def simulate(n, mu, omega, alpha, gamma, beta, delta, nu, x, seed=0):
    rng = np.random.default_rng(seed)
    z = student_t.rvs(nu, size=n, random_state=rng) * np.sqrt((nu - 2) / nu)
    r = np.empty(n); s2 = np.empty(n)
    s2[0] = omega / (1 - alpha - beta - gamma / 2)
    r[0] = mu + np.sqrt(s2[0]) * z[0]
    for t in range(1, n):
        e = r[t - 1] - mu
        s2[t] = omega + (alpha + gamma * (e < 0)) * e * e + beta * s2[t - 1] + delta * x[t]
        r[t] = mu + np.sqrt(s2[t]) * z[t]
    return r


def test_recovers_parameters_on_simulated_data():
    n = 6000
    rng = np.random.default_rng(1)
    x = np.maximum(0, rng.standard_normal(n)) * 1e-4
    r = simulate(n, 5e-4, 2e-6, 0.03, 0.12, 0.85, 0.3, 6.0, x)
    f = garchx.fit(r, x[:, None])
    assert f.converged
    assert f.beta == pytest.approx(0.85, abs=0.05)
    assert f.gamma == pytest.approx(0.12, abs=0.06)
    assert f.delta[0] == pytest.approx(0.3, abs=0.15)
    assert f.nu == pytest.approx(6.0, abs=1.5)


def test_irrelevant_regressor_gets_small_delta():
    n = 4000
    rng = np.random.default_rng(2)
    x = np.maximum(0, rng.standard_normal(n)) * 1e-4
    r = simulate(n, 5e-4, 2e-6, 0.03, 0.12, 0.85, 0.0, 6.0, x, seed=3)
    f = garchx.fit(r, x[:, None])
    assert f.delta[0] < 0.05


def test_sigma_t_does_not_use_r_t():
    rng = np.random.default_rng(4)
    r = rng.standard_normal(300) * 0.01
    X = np.abs(rng.standard_normal((300, 1))) * 1e-4
    f = garchx.fit(r, X)
    s1 = garchx.sigma_path(f, r, X, r.var())
    r2 = r.copy(); r2[200:] *= 10; X2 = X.copy(); X2[201:] *= 10
    s2 = garchx.sigma_path(f, r2, X2, r.var())
    np.testing.assert_allclose(s1[:201], s2[:201])  # sigma_200 depends on r_199, X_200


def test_t_quantile_and_es_match_monte_carlo():
    nu = 5.0
    q, es = garchx.t_quantile_es(0.99, nu)
    z = student_t.rvs(nu, size=2_000_000, random_state=np.random.default_rng(5)) * np.sqrt((nu - 2) / nu)
    assert q == pytest.approx(np.quantile(z, 0.01), rel=0.02)
    assert es == pytest.approx(-z[z <= q].mean(), rel=0.02)


def test_rolling_forecasts_are_out_of_sample():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2015-01-01", periods=700)
    r = pd.Series(rng.standard_normal(700) * 0.01, index=idx)
    fc, fits = rolling.rolling_var(r, None, W=500, refit_every=50)
    assert fc.index[0] == idx[500] and len(fc) == 200 and len(fits) == 4
    r2 = r.copy(); r2.iloc[650:] = r2.iloc[650:] * 5
    fc2, _ = rolling.rolling_var(r2, None, W=500, refit_every=50)
    pd.testing.assert_series_equal(fc["var99"].iloc[:151], fc2["var99"].iloc[:151])
    assert ((fc["loss"] > fc["var99"]).astype(int) == fc["hit99"]).all()


def test_tick_loss_and_dm():
    r = pd.Series([-0.03, 0.01, 0.0])
    var = pd.Series([0.02, 0.02, 0.02])
    tl = rolling.tick_loss(r, var, 0.99)
    assert tl.iloc[0] == pytest.approx((0.01 - 1) * (-0.03 + 0.02))
    assert tl.iloc[1] == pytest.approx(0.01 * 0.03)
    rng = np.random.default_rng(7)
    a = pd.Series(rng.random(500)); b = a + 0.1
    assert rolling.diebold_mariano(a, b)["dm"] < -5


def test_coverage_tests_known_values():
    h = np.zeros(1000, int); h[:10] = 1
    assert ct.kupiec_pof(h, 0.01).statistic == pytest.approx(0.0, abs=1e-12)
    assert ct.christoffersen_independence(h).p_value < 0.01  # all hits clustered
