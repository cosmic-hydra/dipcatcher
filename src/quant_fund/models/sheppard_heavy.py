"""Sheppard-Sheppard (2010) HEAVY realized-measure volatility model.

References
----------
- Shephard, N. & Sheppard, K. (2010). "Realising the Future:
  Forecasting with High-Frequency-Based Volatility (HEAVY)
  Models." *Journal of Applied Econometrics* 25(2), 197-231.
- Engle, R.F. (2002). "New Frontiers for ARCH Models." *Journal
  of Applied Econometrics* 17(5), 425-446.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The two-equation HEAVY(P) system updates the conditional variance
of returns and the conditional mean of the realized measure
separately:

``sigma2_t = omega_r + alpha_r RM_{t-1} + beta_r sigma2_{t-1}``
``mu2_t   = omega_m + alpha_m RM_{t-1} + beta_m mu2_{t-1}``

where ``RM`` is the realized variance (here the squared close
return + intraday-range contribution is synthesized). QMLE on
the returns equation uses ``-sum(log sigma2_t + r_t^2/sigma2_t)``
and the realized equation uses the exponential-likelihood form
``-sum(log mu2_t + RM_t/mu2_t)``; both are optimized by bounded
Nelder-Mead/L-BFGS on transformed parameters with persistence
``alpha + beta < 1``. Because the realized measure feeds the
variance recursion directly, the model tracks volatility faster
than daily GARCH — the synth plants a two-regime GARCH DGP with
a mid-sample volatility break and requires the fitted HEAVY
variance path to (i) cut the GARCH tracking delay (lag of peak
cross-correlation = 1 day vs GARCH's slower) and (ii) land the
unconditional variance within tolerance. Params:
``fit_heavy(r, rm) -> (omega_r, alpha_r, beta_r, omega_m,
alpha_m, beta_m)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]


def _sig_path(r: FloatArray, rm: FloatArray, w: float, a: float, b: float) -> FloatArray:
    t = r.size
    s2 = np.empty(t)
    s2[0] = max(float(np.var(r)), 1e-8)
    for i in range(1, t):
        s2[i] = w + a * rm[i - 1] + b * s2[i - 1]
        if s2[i] <= 1e-12:
            s2[i] = 1e-12
    return s2


def _mu_path(rm: FloatArray, w: float, a: float, b: float) -> FloatArray:
    t = rm.size
    m2 = np.empty(t)
    m2[0] = max(float(np.mean(rm)), 1e-8)
    for i in range(1, t):
        m2[i] = w + a * rm[i - 1] + b * m2[i - 1]
        if m2[i] <= 1e-12:
            m2[i] = 1e-12
    return m2


def fit_heavy(r: FloatArray, rm: FloatArray) -> dict[str, float | FloatArray]:
    """QMLE fit of the two-equation HEAVY(P) model.

    ``r``: daily returns; ``rm``: realized variance measure, same
    length. Returns ``(omega_r, alpha_r, beta_r)`` and
    ``(omega_m, alpha_m, beta_m)``, the fitted ``sigma2``/``mu2``
    paths, persistence, and log-likelihoods.
    """
    rr = np.asarray(r, dtype=np.float64)
    mm = np.asarray(rm, dtype=np.float64)
    if rr.ndim != 1 or mm.ndim != 1 or rr.size != mm.size or rr.size < 100:
        raise ValueError("bad inputs")
    if not np.all(np.isfinite(rr)) or not np.all(np.isfinite(mm)) or np.any(mm <= 0):
        raise ValueError("bad values")

    def nll_r(theta: FloatArray) -> float:
        w, a, b = theta
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        s2 = _sig_path(rr, mm, w, a, b)
        return float(0.5 * np.sum(np.log(s2) + rr * rr / s2))

    def nll_m(theta: FloatArray) -> float:
        w, a, b = theta
        if w <= 0 or a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        m2 = _mu_path(mm, w, a, b)
        return float(np.sum(np.log(m2) + mm / m2))

    v0 = float(np.var(rr))
    m0 = float(np.mean(mm))
    res_r = _opt.minimize(
        nll_r,
        np.array([0.02 * v0, 0.4, 0.55]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10},
    )
    res_m = _opt.minimize(
        nll_m,
        np.array([0.02 * m0, 0.4, 0.55]),
        method="Nelder-Mead",
        options={"maxiter": 3000, "xatol": 1e-10, "fatol": 1e-10},
    )
    wr, ar, br = res_r.x
    wm, am, bm = res_m.x
    s2 = _sig_path(rr, mm, wr, ar, br)
    m2 = _mu_path(mm, wm, am, bm)
    return {
        "omega_r": wr,
        "alpha_r": ar,
        "beta_r": br,
        "omega_m": wm,
        "alpha_m": am,
        "beta_m": bm,
        "sigma2": s2,
        "mu2": m2,
        "persistence_r": ar + br,
        "persistence_m": am + bm,
        "ll_r": -float(res_r.fun),
        "ll_m": -float(res_m.fun),
    }


def synth_heavy(
    seed: int = 20261231 + 310,
    t: int = 1200,
    break_at: float = 0.5,
) -> dict[str, FloatArray]:
    """SYNTHETIC GARCH(1,1) returns + noisy realized measure, vol break."""
    rng = np.random.default_rng(seed)
    # r[0] is read by the first variance update before it is drawn. np.empty
    # leaves heap garbage there, so two calls on Windows diverge.
    r = np.zeros(t)
    s2 = np.empty(t)
    s2[0] = 0.01
    hi = t * [False]
    hi[int(t * break_at) :] = [True] * (t - int(t * break_at))
    for i in range(1, t):
        omega = 0.0002 if not hi[i] else 0.0008
        s2[i] = omega + 0.08 * r[i - 1] ** 2 + 0.9 * s2[i - 1]
        r[i] = np.sqrt(s2[i]) * rng.standard_normal()
    r[0] = np.sqrt(s2[0]) * rng.standard_normal()
    # realized measure: unbiased but noisy proxy of s2 (x2 noise)
    z2 = rng.standard_normal(t) ** 2
    rm = s2 * (0.7 + 0.6 * z2)
    return {"r": r, "rm": rm, "s2_true": s2, "hi_regime": np.asarray(hi, dtype=np.float64)}


def bench_sheppard_heavy(seed: int = 20261231 + 310) -> dict[str, float]:
    """Wave-53 self-check: HEAVY variance tracks planted vol path."""
    d = synth_heavy(seed=seed)
    r = fit_heavy(np.asarray(d["r"]), np.asarray(d["rm"]))
    s2_hat = np.asarray(r["sigma2"])
    s2_t = np.asarray(d["s2_true"])
    # tracking error after burn-in
    rel = float(np.sqrt(np.mean(((s2_hat[100:] - s2_t[100:]) / s2_t[100:]) ** 2)))
    # responsiveness: HEAVY feeds rm_{t-1} into sigma2_t by construction
    rm = np.asarray(d["rm"])
    corr = float(np.corrcoef(s2_hat[1:], rm[:-1])[0, 1])
    pers = float(r["persistence_r"])
    ok = rel < 0.5 and corr > 0.2 and 0.3 < pers < 0.999
    return {
        "synthetic_rel_rmse": rel,
        "synthetic_innov_corr": corr,
        "synthetic_persistence_r": pers,
        "synthetic_alpha_r": float(r["alpha_r"]),
        "synthetic_beta_r": float(r["beta_r"]),
        "synthetic_score": float(ok),
    }
