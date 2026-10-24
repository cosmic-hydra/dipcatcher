"""CMA-ES evolution strategy tests."""

from __future__ import annotations

import numpy as np
import pytest

from quant_fund.models.cma_es import (
    bench_cma_es,
    cma_es,
    ellipsoid,
    rastrigin,
    sphere,
)


def test_sphere_converges():
    x0 = np.array([3.0, -2.0, 1.0, 0.5])
    r = cma_es(sphere, x0, sigma0=1.5, budget=1200, seed=1)
    assert r["best_f"] < 1e-4


def test_ellipsoid_handles_conditioning():
    x0 = np.ones(6)
    r = cma_es(ellipsoid, x0, sigma0=1.0, budget=3500, seed=2)
    assert r["best_f"] < 1e-6


def test_rastrigin_reasonable():
    x0 = np.full(6, 3.0)
    r = cma_es(rastrigin, x0, sigma0=1.5, budget=6000, seed=3)
    assert r["best_f"] < 20.0


def test_eigh_column_sign_is_canonical():
    """A BLAS sign flip must not survive into the covariance factor."""
    from quant_fund.models.cma_es import _canonical_eigh

    rng = np.random.default_rng(0)
    draw = rng.normal(size=(6, 6))
    cov = draw @ draw.T
    values, vectors = _canonical_eigh(cov)
    ref, _ = np.linalg.eigh(cov)
    assert np.allclose(values, ref)
    for i in range(vectors.shape[1]):
        col = vectors[:, i]
        pivot = int(np.argmax(np.abs(col)))
        assert col[pivot] > 0.0


def test_deterministic():
    x0 = np.ones(5)
    a = cma_es(sphere, x0, 1.0, 800, seed=9)
    b = cma_es(sphere, x0, 1.0, 800, seed=9)
    assert a == b


def test_fail_closed():
    with pytest.raises(ValueError):
        cma_es(sphere, np.array([0.0]), 1.0, 500)
    with pytest.raises(ValueError):
        cma_es(sphere, np.ones(4), -1.0, 500)
    with pytest.raises(ValueError):
        cma_es(lambda x: float("nan"), np.ones(4), 1.0, 500)
    with pytest.raises(ValueError):
        cma_es(sphere, np.ones(4), 1.0, 5)


def test_bench_passes():
    out = bench_cma_es()
    assert out["synthetic_sphere_final"] < 1e-4
