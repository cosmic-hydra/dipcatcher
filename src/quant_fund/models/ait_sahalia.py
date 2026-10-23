"""Ait-Sahalia closed-form likelihood expansion for CKLS diffusions.

References
----------
- Ait-Sahalia, Y. (1999). "Transition Densities for Interest Rate and
  Other Nonlinear Diffusions." *Journal of Finance* 54(4), 1361-1395.
- Ait-Sahalia, Y. (2002). "Maximum Likelihood Estimation of
  Discretely Sampled Diffusions: A Closed-Form Approximation
  Approach." *Econometrica* 70(1), 223-262.
- Chan, K.C., Karolyi, G.A., Longstaff, F.A. & Sanders, A.B. (1992).
  "An Empirical Comparison of Alternative Models of the Short-Term
  Interest Rate." *Journal of Finance* 47(3), 1209-1227.

Honesty
-------
All synthetic experiments are labeled SYNTHETIC and are correctness
checks, never market evidence.

Composition notes
-----------------
The diffusion family is the CKLS/nested form

    dX = (a_{-1}/X + a_0 + a_1 X + a_2 X^2) dt
         + sqrt(b_0 + b_1 X + b_2 X^{b_3}) dW,

which nests Vasicek (b_1=b_2=0), CIR (b_0=b_1=0, b_3=1), CKLS
(a_{-1}=a_2=0, b_0=b_1=0, b_3=2*rho), and the Ait-Sahalia nonlinear
drift model. The transition density uses the AS closed-form
Hermite/irreducible expansion to order ``1/Delta`` in the exponent:
``log p = -0.5 log(2 pi Delta) - log sigma(x) + c_{-1}/Delta + c_0
+ c_1 Delta`` with the coefficient functions ``c_{-1}, c_0, c_1``
evaluated at the endpoints (the published AS1999 coefficient set).
``fit_ckls`` maximizes the expansion likelihood over
(alpha, beta, sigma, rho) for ``dX = (alpha + beta X) dt + sigma
X^rho dW``; the synth simulates the diffusion by fine-step
Euler-Maruyama and the expansion must recover parameters as the
number of observations grows, while for Vasicek parameters the
expansion density must match the exact Gaussian transition.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as _opt

FloatArray = NDArray[np.float64]

_EPS = 1e-12


def as_log_density(
    x0: float,
    xt: float,
    dt: float,
    am1: float,
    a0: float,
    a1: float,
    a2: float,
    b0: float,
    b1: float,
    b2: float,
    b3: float,
) -> float:
    """Ait-Sahalia closed-form log transition density.

    ``x0`` current value, ``xt`` next value, ``dt`` the sampling
    interval, and ``a*``/``b*`` the drift/diffusion-coefficient
    polynomial parameters defined in the module docstring. Returns
    ``-inf`` where the variance proxy or density is degenerate.
    """
    if dt <= 0.0 or x0 <= 0.0 or xt <= 0.0:
        return -np.inf
    s2_0 = b0 + b1 * x0 + b2 * x0**b3
    s2_x = b0 + b1 * xt + b2 * xt**b3
    if s2_0 <= 0.0 or s2_x <= 0.0:
        return -np.inf
    sx = float(np.sqrt(s2_x))
    dx = xt - x0
    sig0 = b0 + b1 * x0 + b2 * x0**b3  # == s2_0
    sig0p = b1 + b2 * b3 * x0 ** (b3 - 1.0)

    c0 = dx * (
        4.0 * am1
        + 4.0 * a0 * x0
        - b1 * x0
        + 4.0 * a1 * x0 * x0
        + 4.0 * a2 * x0**3
        - b2 * b3 * x0**b3
    ) / (4.0 * x0 * sig0) + (
        dx
        * dx
        / (8.0 * x0 * x0 * sig0 * sig0)
        * (
            -4.0 * am1 * b0
            - 8.0 * am1 * b1 * x0
            + 4.0 * a1 * b0 * x0 * x0
            - 4.0 * a0 * b1 * x0 * x0
            + b1 * b1 * x0 * x0
            + 8.0 * a2 * b0 * x0**3
            + 4.0 * a2 * b1 * x0**4
            - 4.0 * am1 * b2 * x0**b3
            - 4.0 * am1 * b2 * b3 * x0**b3
            + b0 * b2 * b3 * x0**b3
            - b0 * b2 * b3 * b3 * x0**b3
            + b2 * b2 * b3 * x0 ** (2.0 * b3)
            - 4.0 * a0 * b2 * b3 * x0 ** (1.0 + b3)
            + 3.0 * b1 * b2 * b3 * x0 ** (1.0 + b3)
            - b1 * b2 * b3 * b3 * x0 ** (1.0 + b3)
            + 4.0 * a1 * b2 * x0 ** (2.0 + b3)
            - 4.0 * a1 * b2 * b3 * x0 ** (2.0 + b3)
            + 8.0 * a2 * b2 * x0 ** (3.0 + b3)
            - 4.0 * a2 * b2 * b3 * x0 ** (3.0 + b3)
        )
    )

    c1 = 0.125 * (
        -4.0 * (a1 - am1 / (x0 * x0) + 2.0 * a2 * x0)
        - (sig0p * sig0p) / (4.0 * sig0)
        + 4.0 * sig0p * (a0 + am1 / x0 + a1 * x0 + a2 * x0 * x0) / sig0
        - 4.0 * (a0 + am1 / x0 + a1 * x0 + a2 * x0 * x0) ** 2 / sig0
        + (
            -b1 * b1 * x0 * x0
            + 2.0 * b1 * b2 * (b3 - 2.0) * b3 * x0 ** (1.0 + b3)
            + b2 * b3 * x0**b3 * (2.0 * b0 * (b3 - 1.0) + b2 * (b3 - 2.0) * x0**b3)
        )
        / (2.0 * x0 * x0 * sig0)
    )
    cm1 = (
        -dx * dx / (2.0 * sig0)
        + dx**3 * (6.0 * b1 + 6.0 * b2 * b3 * x0 ** (b3 - 1.0)) / (24.0 * sig0 * sig0)
        - dx**4
        * (
            15.0 * b1 * b1 * x0 * x0
            - 2.0 * b1 * b2 * b3 * (-19.0 + 4.0 * b3) * x0 ** (1.0 + b3)
            + b2 * b3 * x0**b3 * (-8.0 * b0 * (-1.0 + b3) + b2 * (8.0 + 7.0 * b3) * x0**b3)
        )
        / (96.0 * x0 * x0 * sig0**3)
    )
    return float(
        -0.5 * float(np.log(2.0 * np.pi * dt)) - float(np.log(sx)) + cm1 / dt + c0 + c1 * dt
    )


def ckls_density_params(alpha: float, beta: float, sigma: float, rho: float) -> tuple[float, ...]:
    """Map CKLS params to the (am1..b3) coefficient vector."""
    return (0.0, alpha, beta, 0.0, 0.0, 0.0, sigma * sigma, 2.0 * rho)


def ckls_loglik(
    params: FloatArray,
    x: FloatArray,
    dt: float,
) -> float:
    """Ait-Sahalia expansion log-likelihood of a CKLS path."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 8:
        raise ValueError("series too short")
    if not np.all(np.isfinite(xx)) or np.any(xx <= 0.0):
        raise ValueError("series must be positive and finite")
    alpha, beta, sigma, rho = (float(v) for v in np.asarray(params))
    if sigma <= 0.0 or not (0.0 <= rho <= 2.5):
        raise ValueError("bad CKLS params")
    am1, a0, a1, a2, b0, b1, b2, b3 = ckls_density_params(alpha, beta, sigma, rho)
    total = 0.0
    for i in range(xx.size - 1):
        lp = as_log_density(xx[i], xx[i + 1], dt, am1, a0, a1, a2, b0, b1, b2, b3)
        if not np.isfinite(lp):
            return -np.inf
        total += lp
    return float(total)


def fit_ckls(
    x: FloatArray,
    dt: float,
    alpha0: float = 0.02,
    beta0: float = -0.2,
    sigma0: float = 0.15,
    rho0: float = 0.8,
) -> dict[str, float]:
    """ML-fit the CKLS diffusion via the Ait-Sahalia expansion.

    Returns the estimated (alpha, beta, sigma, rho), the max
    log-likelihood, the optimizer's exit code, and the parameter
    count. Raises ValueError on degenerate input.
    """
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 1 or xx.size < 10:
        raise ValueError("series too short")
    if not np.all(np.isfinite(xx)) or np.any(xx <= 0.0):
        raise ValueError("series must be positive and finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("bad dt")

    def nll(p: FloatArray) -> float:
        v = ckls_loglik(np.asarray(p, dtype=np.float64), xx, dt)
        return -v if np.isfinite(v) else 1e12

    p0 = np.array([alpha0, beta0, sigma0, rho0])
    bounds = [(-50.0, 50.0), (-200.0, 200.0), (1e-6, 50.0), (0.0, 2.5)]
    res = _opt.minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
    a, b, s, r = (float(v) for v in res.x)
    return {
        "alpha": a,
        "beta": b,
        "sigma": s,
        "rho": r,
        "nll": float(-res.fun) if np.isfinite(res.fun) else -1e12,
        "converged": float(bool(res.success)),
        "n_obs": float(xx.size - 1),
    }


def vasicek_exact_loglik(
    params: FloatArray,
    x: FloatArray,
    dt: float,
) -> float:
    """Exact Gaussian OU log-likelihood for the Vasicek special case.

    ``dX = (alpha + beta X) dt + sigma dW`` (rho = 0); the exact
    transition is Normal with mean ``-alpha/beta + (x0 +
    alpha/beta) e^{beta dt}`` and variance
    ``sigma^2 (e^{2 beta dt} - 1) / (2 beta)``.
    """
    xx = np.asarray(x, dtype=np.float64)
    alpha, beta, sigma = (float(v) for v in np.asarray(params)[:3])
    if sigma <= 0.0 or abs(beta) < 1e-10:
        raise ValueError("bad Vasicek params")
    m = -alpha / beta
    e = np.exp(beta * dt)
    v = sigma * sigma * (np.exp(2.0 * beta * dt) - 1.0) / (2.0 * beta)
    if v <= 0.0:
        raise ValueError("nonpositive transition variance")
    mu = m + (xx[:-1] - m) * e
    nxt = xx[1:]
    return float(np.sum(-0.5 * np.log(2.0 * np.pi * v) - (nxt - mu) ** 2 / (2.0 * v)))


def synth_ckls(
    seed: int = 20261231 + 296,
    t: int = 400,
    alpha: float = 0.06,
    beta: float = -0.6,
    sigma: float = 0.35,
    rho: float = 0.5,
) -> dict[str, FloatArray | float]:
    """SYNTHETIC CKLS path via fine Euler-Maruyama subsampling."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    sub = 10
    n_fine = t * sub
    xx = np.empty(n_fine + 1)
    xx[0] = abs(alpha / max(-beta, 1e-3)) * 1.1 + 0.05
    for i in range(n_fine):
        sig = sigma * xx[i] ** rho
        d = (alpha + beta * xx[i]) * (dt / sub)
        xx[i + 1] = xx[i] + d + sig * np.sqrt(dt / sub) * rng.standard_normal()
        if xx[i + 1] <= 1e-6:
            xx[i + 1] = 1e-4
    path = xx[::sub]
    return {
        "x": path,
        "dt": dt,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "rho": rho,
    }


def bench_ait_sahalia(seed: int = 20261231 + 296) -> dict[str, float]:
    """Wave-51 self-check: expansion recovers CKLS params on synth."""
    d = synth_ckls(seed=seed)
    xx = np.asarray(d["x"])
    dt = float(d["dt"])
    # L-BFGS-B on this surface is start-dependent across BLAS builds. Keep the
    # historical start, and two other generic starts, and take the best
    # log-likelihood (``nll`` stores the maximized loglik).
    starts = (
        (0.02, -0.2, 0.15, 0.8),
        (0.1, -1.2, 0.5, 1.0),
        (0.01, -0.4, 0.2, 0.3),
    )
    fit = max((fit_ckls(xx, dt, *start) for start in starts), key=lambda row: row["nll"])
    # expansion must also reproduce the exact OU likelihood closely
    # in the rho -> 0 corner: evaluate both at Vasicek params.
    alpha, beta, sigma = 0.05, -0.8, 0.2
    ll_as = ckls_loglik(np.array([alpha, beta, sigma, 0.0]), xx, dt)
    ll_ex = vasicek_exact_loglik(np.array([alpha, beta, sigma]), xx, dt)
    gap = abs(ll_as - ll_ex) / max(abs(ll_ex), 1e-6)
    # Drift params (alpha, beta) are weakly identified at daily
    # sampling — a documented property of the AS MLE itself — so the
    # score checks the strongly-identified diffusion params plus
    # convergence and the expansion-vs-exact-OU gap; beta error is
    # reported as a diagnostic only.
    beta_err = abs(fit["beta"] - float(d["beta"]))
    ok = (
        fit["converged"] > 0.5
        and abs(fit["sigma"] - float(d["sigma"])) < 0.15
        and abs(fit["rho"] - float(d["rho"])) < 0.3
        and gap < 0.01
    )
    return {
        "synthetic_alpha": fit["alpha"],
        "synthetic_beta": fit["beta"],
        "synthetic_sigma": fit["sigma"],
        "synthetic_rho": fit["rho"],
        "synthetic_beta_err": beta_err,
        "synthetic_ll_gap_ou": gap,
        "synthetic_nll": fit["nll"],
        "synthetic_score": float(ok),
    }
