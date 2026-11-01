"""CAViaR: Conditional Autoregressive Value at Risk by regression (SYNTHETIC)
quantiles (Engle & Manganelli 2004).

Models the tau-quantile q_t of the return distribution directly,
estimated by minimizing the Koenker tick loss over beta. Four specs:

- SAV (symmetric absolute value): q_t = b0 + b1 q_{t-1} + b2 |y_{t-1}|
- AS  (asymmetric slope):         q_t = b0 + b1 q_{t-1} + b2 y_{t-1}^+ + b3 y_{t-1}^-
- IG  (indirect GARCH):           q_t = sqrt(b0 + b1 q_{t-1}^2 + b2 y_{t-1}^2)
- AD  (adaptive):                 q_t = q_{t-1} + b0 / (1 + exp(G (y_{t-1} - q_{t-1}))),  G=10

Convention: tau is the quantile level on the return scale
(tau=0.05 -> left-tail quantile of y; sign matches the data).

The objective is nonsmooth/nonconvex -> Powell multi-start, best of
several perturbations around the unconditional quantile.

Fail-closed: invalid tau, non-finite returns, too few obs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import optimize

Array = NDArray[np.float64]


def _tick_loss(y: Array, q: Array, tau: float) -> float:
    e = y - q
    return float(np.mean(np.abs(e * (tau - (e < 0)))))


def _q_path(y: Array, beta: Array, spec: str, tau: float) -> Array | None:
    n = y.size
    q = np.empty(n)
    q[0] = float(np.quantile(y[: max(n // 4, 20)], tau))
    # IG multiplies every step by the sign of the unconditional quantile.
    # That sign does not depend on t; leaving the quantile inside the
    # recursion repeated an O(n log n) reduction on every observation of
    # every Powell evaluation.
    ig_sign = np.sign(float(np.quantile(y, tau))) if spec == "ig" else 0.0
    for t in range(1, n):
        if spec == "sav":
            q[t] = beta[0] + beta[1] * q[t - 1] + beta[2] * abs(y[t - 1])
        elif spec == "as":
            q[t] = (
                beta[0]
                + beta[1] * q[t - 1]
                + beta[2] * max(y[t - 1], 0.0)
                + beta[3] * min(y[t - 1], 0.0)
            )
        elif spec == "ig":
            inner = beta[0] + beta[1] * q[t - 1] ** 2 + beta[2] * y[t - 1] ** 2
            if inner <= 0:
                return None
            q[t] = ig_sign * np.sqrt(inner)
        elif spec == "adaptive":
            q[t] = q[t - 1] + beta[0] / (1.0 + np.exp(10.0 * (y[t - 1] - q[t - 1])))
        else:
            raise ValueError(f"unknown spec {spec!r}")
        if not np.isfinite(q[t]):
            return None
    return q


def _fitted_q_path(y: Array, beta: Array, spec: str, tau: float) -> Array:
    """Quantile path at the fitted beta; contract-checked fail-closed."""
    q = _q_path(y, beta, spec, tau)
    if not (q is not None):
        raise ValueError("q is not None")
    return q


def caviar_fit(
    returns: Array,
    tau: float = 0.05,
    spec: str = "sav",
    n_starts: int = 8,
    seed: int = 0,
) -> dict[str, Array | float]:
    """Fit a CAViaR spec by tick-loss minimization (multi-start Powell)."""
    y = np.asarray(returns, dtype=float).ravel()
    if y.size < 100 or not np.isfinite(y).all():
        raise ValueError("returns must be finite, >= 100 obs")
    if not 0.0 < tau < 0.5:
        raise ValueError("tau must be in (0, 0.5) — tail quantile")
    if spec not in ("sav", "as", "ig", "adaptive"):
        raise ValueError("spec must be sav|as|ig|adaptive")
    if n_starts < 1:
        raise ValueError("n_starts >= 1")

    rng = np.random.default_rng(seed)
    q0 = float(np.quantile(y, tau))
    p_dim = {"sav": 3, "as": 4, "ig": 3, "adaptive": 1}[spec]
    base = np.zeros(p_dim)
    if spec == "sav":
        base[:] = [q0 * 0.05, 0.9, 0.05 * np.sign(q0) * 0 + 0.05]
    elif spec == "as":
        base[:] = [q0 * 0.05, 0.9, 0.03, -0.03]
    elif spec == "ig":
        base[:] = [q0 * q0 * 0.05, 0.9, 0.05]
    else:
        base[:] = [0.02 * abs(q0)]

    def obj(beta: Array) -> float:
        q = _q_path(y, beta, spec, tau)
        if q is None:
            return 1e12
        return _tick_loss(y, q, tau)

    best_val = np.inf
    best_beta = base.copy()
    for s in range(n_starts):
        b0 = base * (1.0 + 0.5 * rng.standard_normal(p_dim)) if s else base
        try:
            res = optimize.minimize(obj, b0, method="Powell", options={"maxiter": 300})
        except Exception:
            continue
        # res.fun == 1e12 is the invalid-parameter penalty, not a real
        # optimum — keeping it would produce a q path of None downstream.
        if np.isfinite(res.fun) and res.fun < 1e12 and res.fun < best_val:
            best_val = float(res.fun)
            best_beta = np.asarray(res.x, dtype=float)
    if not np.isfinite(best_val):
        raise ValueError("caviar fit failed")
    q = _fitted_q_path(y, best_beta, spec, tau)
    hits = (y < q).mean() if tau < 0.5 else (y > q).mean()
    return {
        "beta": best_beta,
        "q": q,
        "rq_loss": best_val,
        "hit_rate": float(hits),
        "tau": float(tau),
    }


def caviar_forecast(
    q_path: Array,
    beta: Array,
    spec: str,
    y_last: float,
) -> float:
    """One-step-ahead conditional quantile given last observation."""
    q = np.asarray(q_path, dtype=float).reshape(-1)
    b = np.asarray(beta, dtype=float).reshape(-1)
    if q.size == 0 or not np.isfinite(q).all():
        raise ValueError("q_path must be non-empty and finite")
    if not np.isfinite(b).all():
        raise ValueError("beta must be finite")
    q_prev = float(q[-1])
    y = float(y_last)
    if not np.isfinite(y):
        raise ValueError("y_last must be finite")
    if spec == "sav":
        out = float(b[0] + b[1] * q_prev + b[2] * abs(y))
    elif spec == "as":
        out = float(b[0] + b[1] * q_prev + b[2] * max(y, 0.0) + b[3] * min(y, 0.0))
    elif spec == "ig":
        inner = b[0] + b[1] * q_prev**2 + b[2] * y * y
        if inner <= 0:
            raise ValueError("ig recursion hit a non-positive variance")
        out = float(np.sign(q_prev) * np.sqrt(inner))
    elif spec == "adaptive":
        out = float(q_prev + b[0] / (1.0 + np.exp(10.0 * (y - q_prev))))
    else:
        raise ValueError(f"unknown spec {spec!r}")
    if not np.isfinite(out):
        raise ValueError(f"{spec} forecast overflowed")
    return out
