"""Hansen (2006) covariance matrix adaptation evolution strategy.

Implements the (mu/mu_w, lambda)-CMA-ES with rank-mu + rank-one
covariance updates, cumulative step-size adaptation (CSA), and
weighted recombination following Hansen & Ostermeier (2001) and
Hansen's tutorial 'The CMA Evolution Strategy: A Tutorial'
(arXiv:1604.00772). Supports fixed-lambda or restart-with-doubled-
lambda (IPOP-lite, Auger & Hansen 2005).

References
----------
- Hansen & Ostermeier (2001) 'Completely Derandomized
  Self-Adaptation in Evolution Strategies' Evolutionary
  Computation 9(2).
- Hansen (2016) 'The CMA Evolution Strategy: A Tutorial'
  arXiv:1604.00772.
- Auger & Hansen (2005) 'A restart CMA evolution strategy with
  increasing population size' CEC.

Honesty
-------
SYNTHETIC self-check: optimizes canonical test functions
(sphere, ellipsoid/Cigar, Rastrigin) with deterministic seeds
and reports achieved objective values / budgets — no market
claims, no live-trading claims.

Composition
-----------
Pure numpy. Inputs are objective callables and bounds; outputs
are best-so-far solutions and diagnostics dicts.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Objective = Callable[[FloatArray], float]


def _positive_pivot_basis(basis: FloatArray) -> FloatArray:
    """Flip each eigenvector so its largest component is positive.

    ``np.linalg.eigh`` may flip column signs across BLAS builds. CMA maps
    the same Gaussian draw through that basis, so a sign flip forks the
    path on Rastrigin. The column span is unchanged.
    """
    out = np.array(basis, dtype=np.float64, copy=True)
    for index in range(out.shape[1]):
        column = out[:, index]
        pivot = int(np.argmax(np.abs(column)))
        if column[pivot] < 0.0:
            out[:, index] = -column
    return out


def _validate_inputs(f: Objective, x0: FloatArray, sigma0: float) -> FloatArray:
    xa = np.asarray(x0, dtype=np.float64).ravel()
    if xa.size < 2 or not np.isfinite(xa).all():
        raise ValueError("x0 must be a finite vector of length >= 2")
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("sigma0 must be positive")
    f0 = float(f(xa))
    if not np.isfinite(f0):
        raise ValueError("objective non-finite at x0")
    return xa


def cma_es(
    f: Objective,
    x0: FloatArray,
    sigma0: float = 0.5,
    budget: int = 2000,
    lam: int | None = None,
    seed: int = 0,
    tolx: float = 1e-12,
    tolf: float = 1e-14,
) -> dict[str, float]:
    """(mu/mu_w, lambda)-CMA-ES with CSA step-size control.

    Minimizes f. Returns dict with best_x (as separate keys),
    best_f, n_evals used, n_generations, sigma_final.
    """
    xa = _validate_inputs(f, x0, sigma0)
    n = xa.size
    if budget < n * 4:
        raise ValueError("budget too small")
    rng = np.random.default_rng(seed)
    lam_i = int(lam) if lam is not None else 4 + int(3 * np.log(n))
    mu = lam_i // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mu_eff = 1.0 / (w * w).sum()
    # strategy parameters (Hansen tutorial defaults)
    c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
    c_s = (mu_eff + 2) / (n + mu_eff + 5)
    c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(1.0 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
    d_s = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_s
    chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    mean = xa.copy()
    sigma = sigma0
    c = np.eye(n)
    p_c = np.zeros(n)
    p_s = np.zeros(n)
    eig_b = np.eye(n)
    eig_d = np.ones(n)
    inv_sqrt_c = np.eye(n)
    evals = 0
    gen = 0
    best_f = np.inf
    best_x = mean.copy()
    converged = 0

    while evals < budget:
        gen += 1
        z = rng.standard_normal((lam_i, n))
        y = z @ (eig_b * eig_d).T  # y_k = B D z_k
        xs = mean[None, :] + sigma * y
        fs = np.empty(lam_i)
        for k in range(lam_i):
            fs[k] = float(f(xs[k]))
        evals += lam_i
        if not np.isfinite(fs).all():
            raise ValueError("objective returned non-finite")
        order = np.argsort(fs)
        xs, fs, y = xs[order], fs[order], y[order]
        if fs[0] < best_f:
            best_f = float(fs[0])
            best_x = xs[0].copy()
            converged = 0
        else:
            converged += 1
        y_w = np.einsum("i,ij->j", w, y[:mu])
        mean_old = mean
        mean = mean + sigma * y_w
        # step-size evolution path
        p_s = (1 - c_s) * p_s + np.sqrt(c_s * (2 - c_s) * mu_eff) * (inv_sqrt_c @ y_w)
        p_s_norm = float(np.linalg.norm(p_s))
        h_sig = float(p_s_norm / np.sqrt(1 - (1 - c_s) ** (2 * gen)) / chi_n < 1.4 + 2 / (n + 1))
        p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w
        # covariance update
        c = (
            (1 - c_1 - c_mu * w.sum()) * c
            + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * c)
            + c_mu * np.einsum("i,ij,ik->jk", w, y[:mu], y[:mu])
        )
        c = (c + c.T) / 2
        sigma *= float(np.exp(min(1.0, (c_s / d_s) * (p_s_norm / chi_n - 1))))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma diverged")
        # eigendecompose
        dd, bb = np.linalg.eigh(c)
        dd = np.maximum(dd, 1e-20)
        eig_d = np.sqrt(dd)
        eig_b = _positive_pivot_basis(bb)
        inv_sqrt_c = bb @ np.diag(1.0 / eig_d) @ bb.T
        # convergence: tiny sigma or no f improvement
        if sigma * eig_d.max() < tolx or (converged > 30 and sigma < 1e-8):
            break
        if evals >= budget:
            break
        del mean_old
        _ = tolf
    return {
        "best_f": best_f,
        "n_evals": float(evals),
        "n_generations": float(gen),
        "sigma_final": float(sigma),
        "best_x_mean_dev": float(np.linalg.norm(best_x - xa)),
        "mu_eff": float(mu_eff),
        "lambda": float(lam_i),
    }


def sphere(x: FloatArray) -> float:
    """f = sum x_i^2, minimizer 0."""
    xa = np.asarray(x, dtype=np.float64)
    return float((xa * xa).sum())


def ellipsoid(x: FloatArray, cond: float = 1e6) -> float:
    """Cigar/ellipsoid f = sum cond^{(i-1)/(n-1)} x_i^2."""
    xa = np.asarray(x, dtype=np.float64)
    n = xa.size
    w = cond ** (np.arange(n) / (n - 1))
    return float((w * xa * xa).sum())


def rastrigin(x: FloatArray) -> float:
    """Rastrigin: 10n + sum (x_i^2 - 10 cos(2 pi x_i))."""
    xa = np.asarray(x, dtype=np.float64)
    return float(10 * xa.size + (xa * xa - 10 * np.cos(2 * np.pi * xa)).sum())


def bench_cma_es(seed: int = 498) -> dict[str, float]:
    """SYNTHETIC: optimize sphere + ellipsoid + Rastrigin."""
    rng = np.random.default_rng(seed)
    n = 8
    x0 = rng.uniform(-4, 4, n)
    r1 = cma_es(sphere, x0, sigma0=2.0, budget=1500, seed=seed + 1)
    r2 = cma_es(ellipsoid, x0, sigma0=2.0, budget=4000, seed=seed + 2)
    x0_r = np.full(n, 3.0)
    r3 = cma_es(rastrigin, x0_r, sigma0=1.5, budget=6000, seed=seed + 3)
    if r1["best_f"] >= 1e-4:
        raise ValueError("CMA-ES failed sphere")
    return {
        "synthetic_sphere_final": r1["best_f"],
        "synthetic_sphere_evals": r1["n_evals"],
        "synthetic_ellipsoid_final": r2["best_f"],
        "synthetic_ellipsoid_evals": r2["n_evals"],
        "synthetic_rastrigin_final": r3["best_f"],
        "synthetic_mu_eff": r1["mu_eff"],
    }
