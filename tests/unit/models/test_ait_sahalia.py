"""Tests for ait_sahalia — closed-form CKLS likelihood expansion."""

import numpy as np
import pytest

from quant_fund.models.ait_sahalia import (
    as_log_density,
    bench_ait_sahalia,
    ckls_loglik,
    fit_ckls,
    synth_ckls,
    vasicek_exact_loglik,
)


def _ou_path(seed: int = 7, n: int = 300) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    x = np.empty(n)
    x[0] = 0.06
    for i in range(n - 1):
        x[i + 1] = abs(x[i] + (0.05 - 0.8 * x[i]) * dt + 0.2 * np.sqrt(dt) * rng.standard_normal())
    return np.abs(x) + 0.02, dt


def test_expansion_matches_exact_ou_density() -> None:
    x, dt = _ou_path()
    p = np.array([0.05, -0.8, 0.2, 0.0])
    ll_as = ckls_loglik(p, x, dt)
    ll_ex = vasicek_exact_loglik(p[:3], x, dt)
    assert abs(ll_as - ll_ex) / abs(ll_ex) < 1e-4


def test_density_pointwise_matches_exact_ou() -> None:
    from scipy.stats import norm

    x0, xt, dt = 0.05, 0.052, 1.0 / 252.0
    got = as_log_density(x0, xt, dt, 0.0, 0.05, -0.8, 0.0, 0.04, 0.0, 0.0, 0.0)
    m, e = 0.05 / 0.8, np.exp(-0.8 * dt)
    v = 0.04 * (np.exp(-1.6 * dt) - 1.0) / (-1.6)
    want = np.log(norm.pdf(xt, m + (x0 - m) * e, np.sqrt(v)))
    assert got == pytest.approx(want, abs=1e-6)


def test_fit_ckls_recovers_diffusion_params() -> None:
    # Bench seed: 400 daily obs; drift params are weakly identified
    # (documented in the module), so check the diffusion block.
    d = synth_ckls(seed=20261231 + 296)
    fit = fit_ckls(np.asarray(d["x"]), float(d["dt"]))
    assert fit["converged"] > 0.5
    assert abs(fit["sigma"] - float(d["sigma"])) < 0.15
    assert abs(fit["rho"] - float(d["rho"])) < 0.3
    assert fit["beta"] < 0.0


def test_loglik_deterministic_and_finite() -> None:
    d = synth_ckls(seed=5)
    x = np.asarray(d["x"])
    a = ckls_loglik(np.array([0.05, -0.5, 0.3, 0.5]), x, float(d["dt"]))
    b = ckls_loglik(np.array([0.05, -0.5, 0.3, 0.5]), x, float(d["dt"]))
    assert np.isfinite(a)
    assert a == b


def test_fail_closed_inputs() -> None:
    with pytest.raises(ValueError):
        ckls_loglik(np.array([0.05, -0.5, 0.3, 0.5]), np.array([0.1, -0.1] * 10), 0.01)
    with pytest.raises(ValueError):
        ckls_loglik(np.array([0.05, -0.5, -0.3, 0.5]), np.ones(20), 0.01)
    with pytest.raises(ValueError):
        fit_ckls(np.ones(5), 0.01)
    with pytest.raises(ValueError):
        fit_ckls(np.ones(20), -0.5)
    assert as_log_density(0.05, -0.1, 0.01, 0.0, 0.0, -0.5, 0.0, 0.04, 0.0, 0.0, 0.0) == -np.inf
    assert as_log_density(0.05, 0.06, -0.01, 0.0, 0.0, -0.5, 0.0, 0.04, 0.0, 0.0, 0.0) == -np.inf


def test_synth_deterministic_positive() -> None:
    a = synth_ckls(seed=3)
    b = synth_ckls(seed=3)
    np.testing.assert_array_equal(a["x"], b["x"])
    assert np.all(np.asarray(a["x"]) > 0.0)


def test_bench_schema_and_score() -> None:
    r = bench_ait_sahalia(seed=3)
    for k in (
        "synthetic_sigma",
        "synthetic_rho",
        "synthetic_beta_err",
        "synthetic_ll_gap_ou",
        "synthetic_score",
    ):
        assert np.isfinite(r[k])
    assert r["synthetic_score"] == 1.0, r
