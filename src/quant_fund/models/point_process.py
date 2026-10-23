"""Self-exciting point processes: univariate/bivariate exponential-kernel Hawkes (SYNTHETIC).

References:
- Hawkes (1971). Spectra of some self-exciting and mutually exciting point
  processes. *Biometrika* 58.
- Ogata (1981). On Lewis' simulation method for point processes + (1988)
  statistical models for earthquake occurrences (MLE loglik form).
- Bowsher (2007). Modelling security market events in continuous time.
- Filimonov, Sornette (2012). Quantifying reflexivity in financial markets.
- Daley, Vere-Jones. *An Introduction to the Theory of Point Processes* —
  compensator / random time-change theorem.
- Lewis, Shedler (1979) thinning simulation; Ogata (1981) variant used here.

Kernel convention: ``phi(t) = alpha * beta * exp(-beta t)`` so ``alpha`` is
directly the branching ratio (stationarity requires ``alpha < 1``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
from scipy import stats as sstats

Array = NDArray[np.float64]


def _as_events(t: Array, name: str = "events") -> Array:
    v = np.asarray(t, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 5:
        raise ValueError(f"{name} must contain at least 5 finite event times")
    if np.any(np.diff(v) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return v


def _kernel_sums(t: Array, beta: float) -> Array:
    """``A_i = sum_{j<i} exp(-beta (t_i - t_j))`` via the O(n) recursion."""
    n = t.size
    a = np.zeros(n)
    for i in range(1, n):
        a[i] = np.exp(-beta * (t[i] - t[i - 1])) * (1.0 + a[i - 1])
    return a


def hawkes_intensity(t: Array, mu: float, alpha: float, beta: float) -> Array:
    """Conditional intensity path ``lambda(t_i)`` at the event times."""
    v = _as_events(t)
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        raise ValueError("mu>0, alpha>=0, beta>0 required")
    a = _kernel_sums(v, beta)
    return mu + alpha * beta * a


def hawkes_loglik(t: Array, mu: float, alpha: float, beta: float) -> float:
    """Ogata loglik ``sum log lam(t_i) - int_0^T lam(s) ds`` (closed form)."""
    v = _as_events(t)
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        return -np.inf
    a = _kernel_sums(v, beta)
    lam = mu + alpha * beta * a
    if np.any(lam <= 0.0):
        return -np.inf
    horizon = v[-1] - v[0]
    if horizon <= 0.0:
        return -np.inf
    compensator = mu * horizon + alpha * float(np.sum(1.0 - np.exp(-beta * (v[-1] - v))))
    return float(np.sum(np.log(lam)) - compensator)


def hawkes_mle(t: Array) -> dict[str, float]:
    """Univariate Hawkes MLE over (mu, alpha, beta); Nelder-Mead on log-params.

    Returns fitted params, ``branching_ratio`` (= alpha), stationarity flag
    and the achieved loglik.  Fails closed when the optimizer cannot find a
    finite optimum.
    """
    v = _as_events(t)
    horizon = v[-1] - v[0]
    mu0 = max(v.size / horizon * 0.3, 1e-6)
    x0 = np.log([mu0, 0.5, 1.0])

    def _nll(theta: Array) -> float:
        mu, alpha, beta = np.exp(theta)
        return -hawkes_loglik(v, mu, alpha, beta)

    res = optimize.minimize(_nll, x0, method="Nelder-Mead")
    mu, alpha, beta = np.exp(res.x)
    ll = -float(res.fun)
    if not np.isfinite(ll):
        raise ValueError("hawkes_mle did not reach a finite optimum")
    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(alpha),
        "stationary": float(alpha < 1.0),
        "loglik": ll,
        "n_events": float(v.size),
        "converged": float(bool(res.success)),
    }


def hawkes_compensator(t: Array, mu: float, alpha: float, beta: float) -> Array:
    """Integrated intensity ``Lambda(t_i)`` (random time-change transform).

    Under the true model the transformed interarrivals
    ``Lambda(t_i) - Lambda(t_{i-1})`` are iid Exp(1) — the diagnostic anchor.
    """
    v = _as_events(t)
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0:
        raise ValueError("mu>0, alpha>=0, beta>0 required")
    n = v.size
    lam = np.empty(n)
    for i in range(n):
        hist = np.sum(1.0 - np.exp(-beta * (v[i] - v[:i]))) if i > 0 else 0.0
        lam[i] = mu * (v[i] - v[0]) + alpha * hist
    return lam


def hawkes_residuals(t: Array, mu: float, alpha: float, beta: float) -> dict[str, float]:
    """Residual diagnostics: transformed interarrival KS-uniformity + moments.

    ``tau_i = Lambda(t_i) - Lambda(t_{i-1})`` iid Exp(1) under the model; the
    reported ``ks_pvalue`` tests uniformity of ``1 - exp(-tau)``.
    """
    lam = hawkes_compensator(t, mu, alpha, beta)
    tau = np.diff(lam)
    if tau.size < 4:
        raise ValueError("too few events for residual diagnostics")
    u = 1.0 - np.exp(-tau)
    ks = sstats.kstest(u, "uniform")
    return {
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "mean_tau": float(tau.mean()),
        "var_tau": float(tau.var()),
    }


def hawkes_simulate(mu: float, alpha: float, beta: float, horizon: float, seed: int = 0) -> Array:
    """Ogata (1981) thinning simulation of a stationary Hawkes path.

    Upper bound ``lam*`` is recomputed after each event; candidate draws are
    accepted with probability ``lam(s)/lam*``.  Requires ``alpha < 1`` for a
    stationary simulation (fail closed otherwise).
    """
    if mu <= 0.0 or alpha < 0.0 or beta <= 0.0 or horizon <= 0.0:
        raise ValueError("mu>0, alpha>=0, beta>0, horizon>0 required")
    if alpha >= 1.0:
        raise ValueError("alpha >= 1 is explosive; refusing to simulate")
    rng = np.random.default_rng(seed)
    events: list[float] = []
    t = 0.0
    while t < horizon:
        lam_bar = mu + alpha * beta * _hawkes_kernel(events, t, beta)
        lam_bar = max(lam_bar, mu)
        t += rng.exponential(1.0 / lam_bar)
        if t >= horizon:
            break
        lam_t = mu + alpha * beta * _hawkes_kernel(events, t, beta)
        if rng.uniform() <= lam_t / lam_bar:
            events.append(t)
    return np.asarray(events)


def _hawkes_kernel(events: list[float], t: float, beta: float) -> float:
    """Sum of ``exp(-beta * (t - e))`` over the last 200 events.

    One vectorized ``exp`` plus a left-to-right add. The previous generator
    expression called ``np.exp`` once per event and dominated the research
    benches that resimulate this path.
    """
    window = events[-200:]
    if not window:
        return 0.0
    weights = np.exp(-beta * (t - np.asarray(window, dtype=np.float64)))
    total = 0.0
    for weight in weights:
        total += float(weight)
    return total


def hawkes2_mle(t1: Array, t2: Array) -> dict[str, float]:
    """Bivariate Hawkes MLE with shared decay ``beta`` (6 free params).

    ``lam_1(t) = mu_1 + beta (a11 * B11(t) + a12 * B12(t))`` where ``Bij`` is
    the kernel sum of process j at times of process i.  Cross terms ``a12``
    (2->1) and ``a21`` (1->2) measure mutual excitation; the 2x2 spectral
    radius is the joint branching ratio.
    """
    a = _as_events(t1, "t1")
    b = _as_events(t2, "t2")
    horizon = max(a[-1], b[-1]) - min(a[0], b[0])

    def _sums(events_i: Array, events_j: Array, beta: float) -> Array:
        # A_i for each time in events_i summing over earlier events_j.
        out = np.zeros(events_i.size)
        j_idx = 0
        acc = 0.0
        last = events_i[0]
        for i in range(events_i.size):
            ti = events_i[i]
            acc *= np.exp(-beta * (ti - last))
            while j_idx < events_j.size and events_j[j_idx] < ti:
                acc += np.exp(-beta * (ti - events_j[j_idx]))
                j_idx += 1
            out[i] = acc
            last = ti
        return out

    def _nll(theta: Array) -> float:
        mu1, mu2, a11, a12, a21, a22, beta = np.exp(theta)
        lam1 = mu1 + beta * (a11 * _sums(a, a, beta) + a12 * _sums(a, b, beta))
        lam2 = mu2 + beta * (a21 * _sums(b, a, beta) + a22 * _sums(b, b, beta))
        if np.any(lam1 <= 0.0) or np.any(lam2 <= 0.0):
            return np.inf
        end = max(a[-1], b[-1])
        comp = mu1 * horizon + mu2 * horizon
        comp += a11 * np.sum(1.0 - np.exp(-beta * (end - a)))
        comp += a12 * np.sum(1.0 - np.exp(-beta * (end - b))) if b.size else 0.0
        comp += a21 * np.sum(1.0 - np.exp(-beta * (end - a))) if a.size else 0.0
        comp += a22 * np.sum(1.0 - np.exp(-beta * (end - b)))
        return -float(np.sum(np.log(lam1)) + np.sum(np.log(lam2)) - comp)

    rate = (a.size + b.size) / max(horizon, 1e-9)
    x0 = np.log([rate * 0.2, rate * 0.2, 0.1, 0.1, 0.1, 0.1, 1.0])
    res = optimize.minimize(_nll, x0, method="Nelder-Mead")
    mu1, mu2, a11, a12, a21, a22, beta = np.exp(res.x)
    br = np.linalg.eigvals(np.array([[a11, a12], [a21, a22]]))
    rho = float(np.max(np.abs(br)))
    return {
        "mu1": float(mu1),
        "mu2": float(mu2),
        "alpha11": float(a11),
        "alpha12": float(a12),
        "alpha21": float(a21),
        "alpha22": float(a22),
        "beta": float(beta),
        "branching_ratio": rho,
        "stationary": float(rho < 1.0),
        "loglik": -float(res.fun),
        "converged": float(bool(res.success)),
    }
