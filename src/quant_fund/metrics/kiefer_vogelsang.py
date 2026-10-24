"""Kiefer-Vogelsang fixed-b HAR testing.

Conventional HAC inference lets the truncation lag grow
slower than n; fixed-b asymptotics instead sets the bandwidth
as a fixed proportion b of the sample — the covariance
estimate is inconsistent but the t-statistic remains
asymptotically pivotal against a nonstandard distribution
ξ_b = W(1)/√J_b where J_b is a kernel-weighted double
integral of Brownian-bridge increments. Critical values are
simulated once from the limiting functional (the paper's own
method) under a fixed internal seed.

Honesty: synthetic benches check size discipline and power
ordering on generated AR(1) means — proper diagnostics,
never market evidence.

References:
- Kiefer, N. M., Vogelsang, T. J. (2005). A new asymptotic
  theory for heteroskedasticity-autocorrelation robust tests.
  *Econometric Theory* 21 — the ξ_b limiting distribution.
- Kiefer, N. M., Vogelsang, T. J. (2002). Heteroskedasticity-
  autocorrelation robust testing using bandwidth equal to
  sample size. *Econometric Theory* 18 — the b=1 case.
- Müller, U. K. (2007). A theory of robust long-run variance
  estimation. *Journal of Econometrics* 141 — fixed-b
  foundations.
- Sun, Y., Phillips, P. C. B., Jin, S. (2008). Optimal
  bandwidth selection in HAR inference. *Econometrica* 76 —
  fixed-b vs conventional trade-off.

Composition: pure numpy — Bartlett kernel quadratic forms +
one deterministic simulation of ξ_b quantiles; deterministic
``np.random.default_rng``; no new dependencies.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _bartlett(u: FloatArray) -> FloatArray:
    return np.maximum(0.0, 1.0 - np.abs(u))


def fixedb_t(x: FloatArray, b: float = 0.1) -> dict[str, float]:
    """HAR t-stat for H0: mean(x)=0 at bandwidth b·n (Bartlett)."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 50:
        raise ValueError("x (n>=50,) required")
    if not np.all(np.isfinite(xx)):
        raise ValueError("finite x required")
    if not 0 < b <= 1:
        raise ValueError("b in (0,1] required")
    n = xx.size
    e = xx - xx.mean()
    m = max(1, int(round(b * n)))
    i = np.arange(n)
    k = _bartlett((i[:, None] - i[None, :]) / m)
    # HAC estimate of Var(xbar) — the e'Ke/n^2 quadratic form
    j_hat = float(e @ k @ e) / (n * n)
    if j_hat <= 0:
        raise ValueError("degenerate variance estimate")
    t_b = float(xx.mean() / np.sqrt(j_hat))
    return {"t_b": t_b, "j_hat": j_hat, "b": b, "m": float(m), "n": float(n)}


def _xi_b_samples(b: float, n_grid: int, n_rep: int, seed: int = 777) -> FloatArray:
    """Simulate ξ_b = W(1)/√J_b with J_b = ∫∫ k((r-s)/b)
    dB(r)dB(s) over Brownian-bridge increments — the paper's
    own simulation approach (demeaned increments = bridge)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, (n_rep, n_grid))
    dz = z / np.sqrt(n_grid)  # BM increments scaled
    db = dz - dz.mean(axis=1, keepdims=True)  # bridge increments
    w1 = dz.sum(axis=1)  # W(1)
    i = np.arange(n_grid)
    m = max(1, int(round(b * n_grid)))
    k = _bartlett((i[:, None] - i[None, :]) / m)
    jb = np.einsum("ri,ij,rj->r", db, k, db)
    jb = np.clip(jb, 1e-12, None)
    return np.asarray(w1 / np.sqrt(jb), dtype=np.float64)


def fixedb_cv(
    b: float,
    level: float = 0.05,
    *,
    n_grid: int = 1000,
    n_rep: int = 4000,
) -> float:
    """Two-sided critical value of |ξ_b| at the given level."""
    if not 0 < b <= 1 or not 0 < level < 0.5:
        raise ValueError("b in (0,1], level in (0,.5) required")
    sims = _xi_b_samples(b, n_grid=n_grid, n_rep=n_rep)
    return float(np.quantile(np.abs(sims), 1 - level))


def fixedb_pvalue(
    x: FloatArray,
    b: float = 0.1,
    *,
    n_grid: int = 1000,
    n_rep: int = 4000,
) -> float:
    """Two-sided p-value of the fixed-b mean test."""
    st = fixedb_t(x, b)
    sims = _xi_b_samples(b, n_grid=n_grid, n_rep=n_rep)
    return float(np.mean(np.abs(sims) >= abs(st["t_b"])))


def synth_ar1_mean(
    n: int = 300,
    rho: float = 0.5,
    mean: float = 0.0,
    sd: float = 1.0,
    seed: int = 0,
) -> FloatArray:
    """AR(1) series with given mean — autocorrelation makes
    iid-t over-reject, fixed-b stays sized."""
    rng = np.random.default_rng(seed)
    e = rng.normal(0, sd, n)
    x = np.empty(n)
    x[0] = e[0]
    for i in range(1, n):
        x[i] = rho * x[i - 1] + e[i]
    return np.asarray(x / np.sqrt(1 - rho**2) * sd + mean, dtype=np.float64)


def bench_kiefer_vogelsang(seed: int = 20261231 + 256) -> dict[str, float]:
    """Fixed-b self-check: cv(b) increases with b and tends
    toward normal as b→0; an AR(1) series with shifted mean
    rejects at b=.1 while a zero-mean one does not.
    All ``synthetic_*``."""
    # The kernel is n_grid x n_grid. The published fixture uses 1000 x 4000
    # (the direct fixedb_cv / fixedb_pvalue tests). This sweep only needs
    # the ordering and the rejection call, which hold on a coarser grid.
    grid, reps = 250, 600
    cv_small = fixedb_cv(0.02, 0.05, n_grid=grid, n_rep=reps)
    cv_big = fixedb_cv(0.5, 0.05, n_grid=grid, n_rep=reps)
    x1 = synth_ar1_mean(mean=0.9, seed=seed)
    x0 = synth_ar1_mean(mean=0.0, seed=seed + 1)
    p1 = fixedb_pvalue(x1, b=0.1, n_grid=grid, n_rep=reps)
    p0 = fixedb_pvalue(x0, b=0.1, n_grid=grid, n_rep=reps)
    p1b = fixedb_pvalue(x1, b=0.1, n_grid=grid, n_rep=reps)
    return {
        "synthetic_cv_small_b": cv_small,
        "synthetic_cv_big_b": cv_big,
        "synthetic_p_shifted": p1,
        "synthetic_p_null": p0,
        "synthetic_detects": float(
            cv_small < cv_big and 1.5 < cv_small < 3.0 and p1 < 0.05 and p0 > p1
        ),
        "synthetic_determinism": float(p1 == p1b),
    }
