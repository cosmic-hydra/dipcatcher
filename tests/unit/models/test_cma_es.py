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
    # Rastrigin is multimodal: which basin a fixed-lambda run settles in is
    # chaotic in last-ulp rounding, and that rounding follows the OpenBLAS
    # kernel the CPU selects (eigh/gemm). A single seed is therefore a
    # per-machine coin flip (seed 3: 4.97 on SkylakeX, 27.86 on Haswell/Zen).
    # Judge the optimizer on a fixed seed panel instead: measured over seeds
    # 0..19 on 13 OpenBLAS kernels, >= 19/20 runs reach f < 20 and the median
    # is <= 7.96.
    x0 = np.full(6, 3.0)
    finals = [cma_es(rastrigin, x0, sigma0=1.5, budget=6000, seed=s)["best_f"] for s in range(20)]
    assert sum(f < 20.0 for f in finals) >= 18, finals
    assert float(np.median(finals)) < 10.0, finals


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
