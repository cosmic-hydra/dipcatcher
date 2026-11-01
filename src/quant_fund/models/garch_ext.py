"""Extended GARCH-family estimators beyond the base suite (SYNTHETIC).

The repo's ``models/volatility.py`` covers EWMA, GARCH(1,1), HAR and
tree-based vol; this module adds the asymmetric-power and fractionally
integrated members of the canon, estimated by quasi-MLE.

References:
- Ding, Granger, Engle (1993) APARCH: sigma^delta = omega + alpha
  (|e| - gamma e)^delta + beta sigma_prev^delta.
- Baillie, Bollerslev, Mikkelsen (1996) FIGARCH: fractional integration
  via the binomial expansion of (1 - L)^d.
- Engle, Ng (1993) news-impact asymmetry (the gamma term).
- Bollerslev, Mikkelsen (1996) EGARCH is in volatility.py already.
- Zumbach (2004) historical-volatility prior for QMLE initialization.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt

Array = NDArray[np.float64]


def _v(x: Array, n: int = 50) -> Array:
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size < n or not np.all(np.isfinite(v)):
        raise ValueError(f"returns must be finite with length >= {n}")
    return v


def _fracdiff_weights(d: float, n: int) -> Array:
    """Binomial coefficients pi_k of (1 - L)^d: pi_0=1, pi_k=pi_{k-1}(k-1-d)/k."""
    w = np.empty(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = w[k - 1] * (k - 1.0 - d) / k
    return w


def figarch_variance(
    returns: Array,
    phi: float,
    d: float,
    beta: float,
    sigma2_0: float | None = None,
) -> Array:
    """Baillie–Bollerslev–Mikkelsen (1996) FIGARCH(1, d, 1) recursion.

    ``sigma2_t = omega/(1-beta) + [1 - (1 - phi L)(1 - L)^d / (1 - beta L)] e^2``
    implemented via the lambda weights:
    ``lambda_1 = phi - beta + d; lambda_k = lambda_{k-1} * (k - 1 - d)/k * ...``
    We use the standard truncated recursion on the pi-weights of
    (1-L)^d applied to (phi L - 1) e^2.
    """
    v = _v(returns)
    if not (0.0 <= d <= 1.0 and 0.0 <= phi < 1.0 and 0.0 <= beta < 1.0):
        raise ValueError("need 0 <= phi, beta < 1 and 0 <= d <= 1")
    if phi + beta >= 1.0 + 1e-8:
        raise ValueError("phi + beta must be < 1")
    n = v.size
    e2 = v * v
    s0 = float(np.mean(e2[: max(10, n // 10)])) if sigma2_0 is None else float(sigma2_0)
    pi = _fracdiff_weights(d, n)
    # lam[k-1] is the lag-k FIGARCH weight. Expanding
    # [1 - (1 - phi L)(1 - L)^d / (1 - beta L)] with pi_k the signed binomial
    # coefficients of (1 - L)^d (pi_1 = -d) gives the BBM (1996) recursion
    # lam_k - beta*lam_{k-1} = -(pi_k - phi*pi_{k-1}).
    lam = np.zeros(n)
    lam[0] = d + phi - beta
    for k in range(2, n):
        lam[k - 1] = beta * lam[k - 2] - pi[k] + phi * pi[k - 1]
    omega_bar = s0 * max(1.0 - beta, 1e-6)  # unconditional anchor
    sigma2 = np.empty(n)
    sigma2[0] = s0
    # sigma2[t] = omega_bar + sum_{j=0}^{t-1} lam[j] * e2[t-1-j], i.e. the
    # length-(n-1) prefix of convolve(e2, lam). Same weights as the nested
    # accumulation; the convolution is what QMLE evaluates hundreds of times.
    if n > 1:
        conv = np.convolve(e2, lam, mode="full")[: n - 1]
        sigma2[1:] = np.maximum(omega_bar + conv, 1e-12)
    return sigma2


def fit_figarch(
    returns: Array,
    max_iter: int = 100,
) -> dict[str, Array]:
    """QMLE for FIGARCH(1, d, 1): maximizes Gaussian quasi-likelihood over
    (phi, d, beta) with stationarity constraints. Returns params, sigma2
    path, log-likelihood, and diagnostics.
    """
    v = _v(returns)

    def nll(theta: Array) -> float:
        phi, d, beta = theta
        if phi < 0.0 or beta < 0.0 or phi + beta >= 0.999 or not (0.01 <= d <= 0.99):
            return 1e12
        try:
            s2 = figarch_variance(v, phi, d, beta)
        except (ValueError, FloatingPointError):
            return 1e12
        ll = -0.5 * float(np.sum(np.log(s2) + v * v / s2))
        return -ll if np.isfinite(ll) else 1e12

    best = None
    for s0 in ([0.05, 0.4, 0.4], [0.1, 0.5, 0.3], [0.2, 0.6, 0.2]):
        res = opt.minimize(
            nll, s0, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-4}
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun) or best.fun >= 1e12:
        raise ValueError("FIGARCH QMLE failed to converge")
    phi, d, beta = best.x
    sigma2 = figarch_variance(v, phi, d, beta)
    return {
        "phi": np.array([phi]),
        "d": np.array([d]),
        "beta": np.array([beta]),
        "sigma2": sigma2,
        "loglik": np.array([-best.fun]),
        "converged": np.array([float(best.success)]),
    }


def aparch_variance(
    returns: Array,
    omega: float,
    alpha: float,
    gamma: float,
    beta: float,
    delta: float,
    sigma2_0: float | None = None,
) -> Array:
    """Ding–Granger–Engle (1993) APARCH(1,1) recursion.

    ``sigma_t^delta = omega + alpha (|e| - gamma e)^delta + beta sigma_{t-1}^delta``.
    """
    v = _v(returns)
    if omega <= 0 or alpha < 0 or abs(gamma) > 1 or beta < 0 or delta <= 0:
        raise ValueError("invalid APARCH parameters")
    if alpha + beta >= 1.0 + 1e-8:
        raise ValueError("alpha + beta must be < 1")
    n = v.size
    s0 = float(np.mean(v[: max(10, n // 10)] ** 2)) if sigma2_0 is None else float(sigma2_0)
    sdelta = np.empty(n)
    sdelta[0] = s0 ** (delta / 2.0)
    for t in range(1, n):
        e = v[t - 1]
        news = (abs(e) - gamma * e) ** delta
        sdelta[t] = omega + alpha * news + beta * sdelta[t - 1]
        if not np.isfinite(sdelta[t]) or sdelta[t] <= 0:
            raise FloatingPointError("APARCH variance path diverged")
    return sdelta ** (2.0 / delta)


def fit_aparch(
    returns: Array,
    max_iter: int = 150,
) -> dict[str, Array]:
    """QMLE for APARCH(1,1) over (omega, alpha, gamma, beta, delta).

    Gaussian quasi-likelihood on sigma_t = (sigma^delta)^{1/delta}.
    """
    v = _v(returns)
    e2bar = float(np.mean(v * v))

    def nll(theta: Array) -> float:
        omega, alpha, gamma, beta, delta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or abs(gamma) > 1 or not (0.5 <= delta <= 3.5):
            return 1e12
        if alpha + beta >= 0.999:
            return 1e12
        try:
            s2 = aparch_variance(v, omega, alpha, gamma, beta, delta)
        except (ValueError, FloatingPointError):
            return 1e12
        ll = -0.5 * float(np.sum(np.log(s2) + v * v / s2))
        return -ll if np.isfinite(ll) else 1e12

    s0_om = 0.02 * e2bar
    best = None
    for start in (
        [s0_om, 0.05, 0.0, 0.85, 2.0],
        [s0_om, 0.08, 0.2, 0.80, 1.5],
        [s0_om, 0.04, -0.2, 0.9, 2.0],
    ):
        res = opt.minimize(
            nll, start, method="Nelder-Mead", options={"maxiter": max_iter, "xatol": 1e-5}
        )
        if best is None or res.fun < best.fun:
            best = res
    if best is None or not np.isfinite(best.fun) or best.fun >= 1e12:
        raise ValueError("APARCH QMLE failed")
    omega, alpha, gamma, beta, delta = best.x
    sigma2 = aparch_variance(v, omega, alpha, gamma, beta, delta)
    return {
        "omega": np.array([omega]),
        "alpha": np.array([alpha]),
        "gamma": np.array([gamma]),
        "beta": np.array([beta]),
        "delta": np.array([delta]),
        "sigma2": sigma2,
        "loglik": np.array([-best.fun]),
        "converged": np.array([float(best.success)]),
    }


def news_impact_curve(gamma: float, delta: float, e_grid: Array | None = None) -> dict[str, Array]:
    """Engle–Ng (1993) news-impact curve for the APARCH news function
    ``(|e| - gamma e)^delta`` — the asymmetric response surface.
    """
    if abs(gamma) > 1 or delta <= 0:
        raise ValueError("need |gamma| <= 1 and delta > 0")
    e = np.linspace(-3.0, 3.0, 121) if e_grid is None else np.asarray(e_grid, dtype=float)
    if not np.all(np.isfinite(e)):
        raise ValueError("e_grid must be finite")
    impact = (np.abs(e) - gamma * e) ** delta
    return {
        "e": e,
        "impact": impact,
        "asymmetry_ratio": np.array([float(impact[e < 0].mean() / impact[e > 0].mean())]),
    }
