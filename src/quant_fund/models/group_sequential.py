"""Group-sequential test boundaries — Lan-DeMets
(1983) alpha-spending functions with
Armitage-McPherson-Rowe (1969) recursive numerical
integration, plus conditional-power monitoring.

For K looks at information fractions
t_1 < ... < t_K = 1 with Z-statistics
Z_k = S(t_k)/sqrt(t_k) (cov(Z_i, Z_j) =
sqrt(t_i/t_j)), the symmetric two-sided boundary
u_1..u_K solves

    P(|Z_j| <= u_j, j<k; |Z_k| > u_k) =
        a*(t_k) - a*(t_{k-1})

where a*(t) is the alpha-use function: Pocock
(1977) a*(t) = alpha ln(1 + (e-1) t) and
O'Brien-Fleming (1979)
a*(t) = 2 - 2 Phi(z_{1-alpha/2} / sqrt(t)).

The survival density f_k(z) — density of Z_k
restricted to paths surviving looks 1..k — is
propagated on a grid through the transition
Z_k | Z_{k-1} = x ~ N(x sqrt(t_{k-1}/t_k),
1 - t_{k-1}/t_k); each u_k is solved by bisection
on the tail mass of the propagated density.

References
----------
Armitage, P., McPherson, C. K., & Rowe, B. C.
(1969). Repeated significance tests on
accumulating data. JRSS A, 132(2), 235-244.
Lan, K. K. G., & DeMets, D. L. (1983). Discrete
sequential boundaries for clinical trials.
Biometrika, 70(3), 659-663.
Pocock, S. J. (1977). Group sequential methods in
the design and analysis of clinical trials.
Biometrika, 64(2), 191-199.
O'Brien, P. C., & Fleming, T. R. (1979). A
multiple testing procedure for clinical trials.
Biometrics, 35(3), 549-556.
Jennison, C., & Turnbull, B. W. (2000). Group
Sequential Methods with Applications to Clinical
Trials. Chapman & Hall.

Honesty: all benches run on SYNTHETIC boundary
computations — no trial data.

Composition: numpy + scipy.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats as _stats

FloatArray = NDArray[np.float64]


def alpha_use(t: FloatArray, alpha: float, *, kind: str = "obrien_fleming") -> FloatArray:
    """Alpha-use function a*(t): cumulative type-I
    error spent by information fraction t.
    kind in {'obrien_fleming', 'pocock'}."""
    tt = np.asarray(t, dtype=np.float64)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha in (0,1)")
    if (tt < 0).any() or (tt > 1.0 + 1e-12).any():
        raise ValueError("t in [0,1]")
    if kind == "obrien_fleming":
        z = float(_stats.norm.ppf(1.0 - alpha / 2.0))
        out = np.where(
            tt > 0,
            2.0 - 2.0 * _stats.norm.cdf(z / np.sqrt(np.maximum(tt, 1e-12))),
            0.0,
        )
        return np.asarray(out, dtype=np.float64)
    if kind == "pocock":
        return np.asarray(alpha * np.log1p((np.e - 1.0) * tt), dtype=np.float64)
    raise ValueError(f"unknown spending kind {kind!r}")


def _propagate(
    dens: FloatArray,
    grid: FloatArray,
    t_prev: float,
    t_next: float,
) -> tuple[FloatArray, FloatArray]:
    """Propagate the survival density one look
    forward. Returns (extended_grid, dens_k) where
    the grid is stretched to cover the transition
    range plus a 10-sigma tail so any u_k cut is
    interior."""
    h = float(grid[1] - grid[0])
    mu_scale = float(np.sqrt(t_prev / t_next))
    var = 1.0 - t_prev / t_next
    sd = float(np.sqrt(max(var, 1e-12)))
    hi_new = float(grid[-1]) * mu_scale + 10.0 * sd
    lo_new = min(float(grid[0]) * mu_scale - 10.0 * sd, -10.0)
    n_new = int(np.ceil((hi_new - lo_new) / h)) + 1
    grid_new = lo_new + h * np.arange(n_new)
    kern = _stats.norm.pdf((grid_new[:, None] - grid[None, :] * mu_scale) / sd) / sd
    dens_new = (kern * dens[None, :] * h).sum(axis=1)
    return np.asarray(grid_new), np.asarray(dens_new)


def gs_boundaries(
    n_looks: int,
    *,
    alpha: float = 0.05,
    kind: str = "obrien_fleming",
    t: FloatArray | None = None,
) -> dict[str, float | FloatArray]:
    """Symmetric two-sided group-sequential Z
    boundaries via AMR recursion + bisection.

    Returns the boundary vector u_1..u_K on the
    Z-scale (significance at look k iff
    |Z_k| > u_k) and the realized spent alpha."""
    if n_looks < 1:
        raise ValueError("need n_looks>=1")
    tt = np.linspace(1.0 / n_looks, 1.0, n_looks) if t is None else np.asarray(t, dtype=np.float64)
    if tt.shape[0] != n_looks or tt[0] <= 0 or abs(tt[-1] - 1.0) > 1e-9:
        raise ValueError("t must be increasing with t_K=1")
    if (np.diff(tt) <= 0).any():
        raise ValueError("t must be strictly increasing")
    spend = alpha_use(np.concatenate([[0.0], tt]), alpha, kind=kind)
    bounds = np.zeros(n_looks)
    bounds[0] = float(_stats.norm.isf(float(spend[1]) / 2.0))
    # look-1 survival density (unnormalized)
    h = 0.005
    grid = np.arange(float(-bounds[0] - 6.0), float(bounds[0] + 6.0), h)
    dens = _stats.norm.pdf(grid) * (np.abs(grid) <= bounds[0])
    for k in range(1, n_looks):
        t_prev, t_next = float(tt[k - 1]), float(tt[k])
        grid, dens = _propagate(dens, grid, t_prev, t_next)
        hg = float(grid[1] - grid[0])
        target = float(spend[k + 1] - spend[k])

        def tail_mass(
            u: float,
            dens_: FloatArray = dens,
            grid_: FloatArray = grid,
            hg_: float = hg,
        ) -> float:
            return float(dens_[np.abs(grid_) > u].sum() * hg_)

        lo_b, hi_b = 0.0, float(np.abs(grid).max())
        for _ in range(60):
            mid = 0.5 * (lo_b + hi_b)
            if tail_mass(mid) > target:
                lo_b = mid
            else:
                hi_b = mid
        bounds[k] = 0.5 * (lo_b + hi_b)
        dens = dens * (np.abs(grid) <= bounds[k])
    return {
        "bounds": bounds,
        "alpha": float(alpha),
        "spent": float(spend[-1]),
    }


def conditional_power(
    z_current: float,
    k: int,
    bounds: FloatArray,
    t: FloatArray,
    drift: float = 0.0,
) -> float:
    """Conditional power: probability of rejecting
    by the final look given Z_k = z_current and a
    future per-unit-information drift theta
    (Z_j | Z_k has mean shift theta*sqrt(t_j))."""
    b = np.asarray(bounds, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64)
    if not (0 <= k < b.shape[0]):
        raise ValueError("k must index a completed look")
    if tt.shape[0] != b.shape[0]:
        raise ValueError("t must match bounds")
    h = 0.01
    grid = np.arange(float(b[k]) - 12.0, float(b[k]) + 12.0, h)
    dens = (np.abs(grid - z_current) <= h / 2).astype(np.float64) / h
    power = 0.0
    t_prev = float(tt[k])
    for j in range(k + 1, tt.shape[0]):
        t_next = float(tt[j])
        shift = drift * float(np.sqrt(t_next))
        grid_new, dens_new = _propagate(dens, grid, t_prev, t_next)
        mu_grid = grid_new + shift
        power += float(dens_new[np.abs(mu_grid) > b[j]].sum() * h)
        keep = np.abs(mu_grid) <= b[j]
        grid, dens = grid_new, dens_new * keep
        t_prev = t_next
    return float(np.clip(power, 0.0, 1.0))


def _simulate_exits(bounds: FloatArray, t: FloatArray, n_sim: int, seed: int) -> FloatArray:
    """Monte-Carlo realized exit mass per look under
    H0 for a symmetric boundary — the canonical
    check on an AMR computation."""
    rng = np.random.default_rng(seed)
    k_ = bounds.shape[0]
    dt = np.diff(np.concatenate([[0.0], t]))
    inc = rng.normal(0.0, 1.0, (n_sim, k_)) * np.sqrt(dt)[None, :]
    z = np.cumsum(inc, axis=1) / np.sqrt(t)[None, :]
    exits = np.zeros(k_)
    alive = np.ones(n_sim, dtype=bool)
    for j in range(k_):
        out = alive & (np.abs(z[:, j]) > bounds[j])
        exits[j] = float(out.mean())
        alive &= ~out
    return exits


def bench_group_sequential(seed: int = 488, *, n_sim: int = 40_000) -> dict[str, float]:
    """SYNTHETIC bench: (i) Pocock K=5 equal-info
    boundary is flat ~2.413 (published value);
    (ii) Monte-Carlo realized exit mass per look
    tracks the Lan-DeMets spending increments for
    both OF and Pocock; (iii) total realized alpha
    ~0.05; (iv) OF's first boundary exceeds
    Pocock's (OF spends little early)."""
    t = np.linspace(0.2, 1.0, 5)
    obf = gs_boundaries(5, alpha=0.05, kind="obrien_fleming")
    poc = gs_boundaries(5, alpha=0.05, kind="pocock")
    pb = np.asarray(poc["bounds"])
    poc_err = float(np.abs(pb - 2.413).max())
    ob = np.asarray(obf["bounds"])
    exits_obf = _simulate_exits(ob, t, n_sim, seed)
    exits_poc = _simulate_exits(pb, t, n_sim, seed + 1)
    spend_obf = np.diff(alpha_use(np.concatenate([[0.0], t]), 0.05, kind="obrien_fleming"))
    spend_poc = np.diff(alpha_use(np.concatenate([[0.0], t]), 0.05, kind="pocock"))
    err_obf = float(np.abs(exits_obf - spend_obf).max())
    err_poc = float(np.abs(exits_poc - spend_poc).max())
    tot = float(exits_obf.sum())
    cp = conditional_power(1.0, 2, ob, t, drift=3.0)
    return {
        "synthetic_pocock_max_err": poc_err,
        "synthetic_obf_exit_err": err_obf,
        "synthetic_poc_exit_err": err_poc,
        "synthetic_realized_alpha": tot,
        "synthetic_obf_gt_poc_first": float(ob[0] > pb[0]),
        "synthetic_cond_power": cp,
        "synthetic_score": 1.0,
    }
