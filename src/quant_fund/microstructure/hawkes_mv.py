"""hawkes_mv — multivariate marked Hawkes cross-excitation, real tape vs sim.

``hawkes_real`` fits a *univariate* exponential Hawkes to market-order arrivals
and reports a single branching ratio (eta = 0.536 on the AMZN 2012-06-21 tape).
That scalar compresses the actual structure: order flow is a marked process —
market orders, limit-order placements, and cancellations on each side excite
each other asymmetrically. This lane fits the multivariate exponential-kernel
Hawkes process

    lambda_m(t) = mu_m + sum_j sum_{t_i^j < t} alpha[m,j] * exp(-beta_j (t - t_i^j))

over the six marks {MO_buy, MO_sell, LO_bid_add, LO_ask_add, cancel_bid,
cancel_ask}. ``alpha`` here is a *rate* (events per unit time contributed per
parent); the branching/power matrix is B[m,j] = alpha[m,j] / beta_j and
stability requires spectral radius rho(B) < 1 — enforced by a soft interior
penalty during the fit and verified on the reported optimum.

Per excitor j the kernel state obeys the O(1)-per-event recursion
``R_j <- exp(-beta_j dt) (R_j + 1{prev event was j})``, so a full
loglikelihood + analytic gradient pass over the merged stream is O(n * M)
with M=6 marks. The beta-gradient state ``P_j = dR_j/dbeta_j`` follows the
same recursion, so L-BFGS-B on log-parameters uses exact derivatives — no
finite differences.

Composition / differentiation from existing modules (nothing here modifies
them):

- ``microstructure.hawkes_real`` — univariate eta fit on MO arrivals only;
  this lane generalizes it to the 6-mark mutually-exciting specification and
  reuses its LOBSTER mapping and sim-arm conventions.
- ``microstructure.lobster`` — message-file parser; this lane consumes
  ``parse_messages`` for the real-tape mark streams.
- ``microstructure.zi_lob_simulator`` + ``microstructure.split_flow`` — the
  iid / Markov-regime / metaorder-splitting MO arms. ``_MarkedZILob`` wraps
  the simulator (public counters only, no internal edits) to record a
  six-mark event stream from the same arms.
- ``models.point_process`` — the models-package univariate/bivariate MLE
  reference implementation; the multivariate compensator recursion and
  Ogata thinning sampler here are the lane-scaled counterparts.

Honesty: the LOBSTER leg measures a real NASDAQ tape when a tape dir is
present; sim arms and the committed fixture are SYNTHETIC correctness runs,
always labeled, never market evidence. No headline Sharpe/Sortino/Calmar/
PnL/NAV anywhere — branching ratios and likelihoods only. Fail-closed:
nonpositive intensities, subcritical-violating simulations, and missing
tapes raise / return ok=False rather than fabricate.

References:
- Hawkes (1971). Spectra of some self-exciting and mutually exciting point
  processes. *Biometrika* 58(1):83-90.
- Hawkes (1971). Point spectra of some mutually exciting point processes.
  *JRSS B* 33(3):438-443 — the multivariate branching-matrix spectral-radius
  stability condition used here.
- Ogata (1981). On Lewis' simulation method for point processes. *IEEE IT*
  27(1):23-31 — thinning sampler; and (1988) *JASA* 83:9-27 — the MLE
  loglik / compensator form.
- Bowsher (2007). Modelling security market events in continuous time.
  *J. Econometrics* 141:876-912 — marked Hawkes on order-flow events.
- Filimonov, Sornette (2012). Quantifying reflexivity in financial markets.
  *PRE* 85:056108 — endo/exo share via the branching matrix.
- Hardiman, Bouchaud (2014). Branching-ratio approximation for the
  self-exciting Hawkes process. *PRE* 90:062807.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quant_fund.compute.numba_guard import call_kernel

try:
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - numba is a required dependency
    _HAVE_NUMBA = False

    def njit(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _deco(fn: Any) -> Any:
            return fn

        if len(_args) == 1 and callable(_args[0]) and not _kwargs:
            return _args[0]
        return _deco


from quant_fund.microstructure.lobster import (
    CANCEL_PARTIAL,
    DELETE,
    EXECUTION,
    EXECUTION_HIDDEN,
    SUBMISSION,
    parse_messages,
)
from quant_fund.microstructure.split_flow import SplitFlow
from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    MOFlow,
    RegimeState,
    ZILobConfig,
    ZILobSimulator,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

HAWKES_MV_SCHEMA = "hawkes_mv.v1"

# Mark order is part of the receipt contract — do not reorder.
MARKS: tuple[str, ...] = (
    "MO_buy",
    "MO_sell",
    "LO_bid_add",
    "LO_ask_add",
    "cancel_bid",
    "cancel_ask",
)
N_MARKS = len(MARKS)
_MARK_IDX = {name: i for i, name in enumerate(MARKS)}

Array = NDArray[np.float64]
MarkStreams = dict[str, Array]

# Soft barrier keeps the reported branching matrix inside rho(B) < 1.
_RHO_CAP = 0.999
_RHO_PENALTY_W = 1.0e6
_MIN_EVENTS_PER_MARK = 20
_MIN_EVENTS_TOTAL = 200


# ---------------------------------------------------------------------------
# Mark extraction — real tape and sim
# ---------------------------------------------------------------------------


def lobster_marks(message_path: Path) -> MarkStreams:
    """Per-mark arrival streams (event time in ms) from a LOBSTER message file.

    LOBSTER ``direction`` is the side of the *resting/passive* order, so an
    EXECUTION with direction -1 is a resting ask being lifted by a *buy*
    market order — the aggressor sign is ``-direction`` (same convention as
    ``tick_rule``). EXECUTION_HIDDEN prints are aggressive fills on hidden
    liquidity and count as market orders. Partial cancels (type 2) and full
    deletes (type 3) both join the ``cancel_*`` marks — both are liquidity
    withdrawal, the distinction matters for book reconstruction, not for the
    arrival process. HALT rows carry no order-flow information.
    """
    out: dict[str, list[float]] = {name: [] for name in MARKS}
    for ev in parse_messages(message_path):
        t_ms = ev.time_s * 1000.0
        if ev.event_type == EXECUTION or ev.event_type == EXECUTION_HIDDEN:
            out["MO_sell" if ev.direction > 0 else "MO_buy"].append(t_ms)
        elif ev.event_type == SUBMISSION:
            out["LO_bid_add" if ev.direction > 0 else "LO_ask_add"].append(t_ms)
        elif ev.event_type == CANCEL_PARTIAL or ev.event_type == DELETE:
            out["cancel_bid" if ev.direction > 0 else "cancel_ask"].append(t_ms)
    return {name: np.asarray(ts, dtype=float) for name, ts in out.items()}


def _mark_event(
    marks: list[tuple[float, int]],
    t_ms: float,
    kind: str,
    fills_before: int,
    bid_before: int,
    ask_before: int,
    trades: list[Any],
    sim: ZILobSimulator,
) -> None:
    """Translate one simulator step's observable deltas into a mark.

    The simulator does not log per-event sides; the mark is recovered from
    public counters — depth deltas for LO/cancel sides, the appended trade's
    aggressor for MOs. Dropped placements and empty-book cancels emit no mark,
    matching the tape where only applied events print.
    """
    if kind == "market":
        if sim.n_fills > fills_before:
            aggressor = trades[-1].aggressor
            marks.append((t_ms, _MARK_IDX["MO_buy" if aggressor == "buy" else "MO_sell"]))
        return
    d_bid = sim.bid_depth - bid_before
    d_ask = sim.ask_depth - ask_before
    if kind == "limit":
        if d_bid > 0:
            marks.append((t_ms, _MARK_IDX["LO_bid_add"]))
        elif d_ask > 0:
            marks.append((t_ms, _MARK_IDX["LO_ask_add"]))
    elif kind == "cancel":
        if d_bid < 0:
            marks.append((t_ms, _MARK_IDX["cancel_bid"]))
        elif d_ask < 0:
            marks.append((t_ms, _MARK_IDX["cancel_ask"]))


def sim_marks(
    config: ZILobConfig | None = None,
    flow: MOFlow | None = None,
    *,
    horizon: int = 20000,
    seed: int = 7,
) -> tuple[MarkStreams, float]:
    """Six-mark event stream (ms) from one ZI-LOB run under a flow arm."""
    cfg = config or ZILobConfig(seed=seed)
    sim = ZILobSimulator(cfg, flow=flow)
    marks: list[tuple[float, int]] = []
    for _ in range(horizon):
        fills, bid_d, ask_d = sim.n_fills, sim.bid_depth, sim.ask_depth
        kind = sim.step()
        _mark_event(marks, sim.t * 1000.0, kind, fills, bid_d, ask_d, sim.trades, sim)
    buckets: list[list[float]] = [[] for _ in range(N_MARKS)]
    for t_ms, mi in marks:
        buckets[mi].append(t_ms)
    return (
        {MARKS[j]: np.asarray(buckets[j], dtype=float) for j in range(N_MARKS)},
        sim.t * 1000.0,
    )


# ---------------------------------------------------------------------------
# Merged stream + exact loglikelihood / gradient
# ---------------------------------------------------------------------------


def _merge_streams(streams: MarkStreams) -> tuple[Array, Array]:
    """Merge per-mark strictly-increasing arrays into one event stream.

    Simultaneous events (real tape at ms resolution does tie) are ordered by
    the file order within the merge — each event excites only events after
    it in the merged sequence, never itself.
    """
    arrays = []
    marks_idx = []
    for j, name in enumerate(MARKS):
        arr = np.asarray(streams.get(name, np.empty(0)), dtype=float)
        arr = arr[np.isfinite(arr)]
        arr.sort()
        arrays.append(arr)
        marks_idx.append(np.full(arr.size, j, dtype=np.int64))
    times = np.concatenate(arrays) if arrays else np.empty(0)
    marks_i = np.concatenate(marks_idx) if marks_idx else np.empty(0, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    return times[order], marks_i[order]


@njit(cache=True)
def _forward_nb(
    times: np.ndarray,
    marks: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    horizon: float,
) -> np.ndarray:
    """Compensator-recursion loglik + d/d(theta) accumulators, flat packed out.

    ``R_j`` is the decayed sum over past mark-j events (excitation state);
    ``P_j = dR_j/dbeta_j`` propagates with the same recursion:
    ``P_j <- e^{-beta_j dt} (P_j - dt * R_j)`` — the event-add term is
    beta-independent, so it enters ``P`` only through later decays.

    Output layout: [ll] + dmu(M) + dalpha(M*M) + dbeta(M); ll = -inf on a
    nonpositive/nonfinite intensity so the caller can bail cheaply.
    """
    m = mu.shape[0]
    out = np.empty(1 + 2 * m + m * m, dtype=np.float64)
    r = np.zeros(m)
    p = np.zeros(m)
    dmu = np.zeros(m)
    dalpha = np.zeros((m, m))
    dbeta = np.zeros(m)
    ll = 0.0
    t_prev = 0.0
    for i in range(times.shape[0]):
        ti = times[i]
        mi = marks[i]
        dt = ti - t_prev
        if dt > 0.0:
            for j in range(m):
                dec = math.exp(-beta[j] * dt)
                p[j] = dec * (p[j] - dt * r[j])
                r[j] = r[j] * dec
        lam = mu[mi]
        for j in range(m):
            lam += alpha[mi, j] * r[j]
        if lam <= 0.0 or not math.isfinite(lam):
            out[0] = -np.inf
            out[1:] = 0.0
            return out
        inv = 1.0 / lam
        ll += math.log(lam)
        dmu[mi] += inv
        for j in range(m):
            dalpha[mi, j] += r[j] * inv
            dbeta[j] += alpha[mi, j] * p[j] * inv
        r[mi] += 1.0
        t_prev = ti
    for mi2 in range(m):
        ll -= mu[mi2] * horizon
        dmu[mi2] -= horizon
    out[0] = ll
    out[1 : 1 + m] = dmu
    out[1 + m : 1 + m + m * m] = dalpha.ravel()
    out[1 + m + m * m :] = dbeta
    return out


def _forward(
    times: Array,
    marks: Array,
    mu: Array,
    alpha: Array,
    beta: Array,
    horizon: float,
) -> tuple[float, Array, Array, Array]:
    """Thin typed wrapper over :func:`_forward_nb` (pure-Python if no numba)."""
    m = int(mu.shape[0])
    out = call_kernel(
        _forward_nb,
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(marks, dtype=np.int64),
        np.ascontiguousarray(mu, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(beta, dtype=np.float64),
        float(horizon),
    )
    return (
        float(out[0]),
        out[1 : 1 + m],
        out[1 + m : 1 + m + m * m].reshape(m, m),
        out[1 + m + m * m :],
    )


def _tail_sums(
    mark_times: list[Array], beta: list[float], horizon: float
) -> list[tuple[float, float, float]]:
    """Per excitor j: (count, sum_k e^{-beta_j tau_k}, sum_k tau_k e^{-beta_j tau_k})."""
    out: list[tuple[float, float, float]] = []
    for j in range(N_MARKS):
        tj = mark_times[j]
        tau = horizon - tj
        e = np.exp(-beta[j] * tau)
        out.append((float(tj.size), float(e.sum()), float((tau * e).sum())))
    return out


def _rho_and_grad(b: Array) -> tuple[float, Array | None]:
    """Perron-Frobenius rho(B) for B >= 0 plus its matrix gradient.

    For a simple dominant eigenvalue, d rho = v^T dB u / (v^T u) with u, v the
    right and left PF eigenvectors. Returns grad=None when eig is degenerate —
    the penalty then contributes value only, which is fine at the boundary.
    """
    vals, u_all = np.linalg.eig(b)
    k = int(np.argmax(np.real(vals)))
    rho = float(np.real(vals[k]))
    vals_t, v_all = np.linalg.eig(b.T)
    kv = int(np.argmax(np.real(vals_t)))
    v = np.real(v_all[:, kv])
    u = np.real(u_all[:, k])
    denom = float(np.dot(v, u))
    if not math.isfinite(denom) or abs(denom) < 1e-12:
        return rho, None
    grad = np.outer(v, u) / denom
    if not np.all(np.isfinite(grad)):
        return rho, None
    return rho, grad


def _pack(theta: Array, n: int) -> tuple[Array, Array, Array]:
    mu = theta[:n]
    alpha = theta[n : n + n * n].reshape(n, n)
    beta = theta[n + n * n :]
    return mu, alpha, beta


def _objective(
    z: Array,
    times: Array,
    marks: Array,
    mark_times: list[Array],
    horizon: float,
    rho_cap: float,
) -> tuple[float, Array]:
    """Negative joint loglik + stationarity barrier; gradient wrt z = log theta."""
    theta = np.exp(z)
    mu, alpha, beta = _pack(theta, N_MARKS)
    ll, dmu, dalpha, dbeta = _forward(times, marks, mu, alpha, beta, horizon)
    if not math.isfinite(ll):
        return 1.0e30, np.zeros_like(z)

    for j, (n_j, e_j, te_j) in enumerate(_tail_sums(mark_times, beta.tolist(), horizon)):
        inv_b = 1.0 / beta[j]
        inner = n_j - e_j  # sum_k (1 - e^{-beta_j tau_k})
        comp_j = inner * inv_b
        ll -= float(alpha[:, j].sum()) * comp_j
        # d/dalpha[m,j]: -(n_j - e_j)/beta_j ; d/dbeta_j: -alpha_mj * (te_j*beta_j - inner)/beta_j^2
        for mi in range(N_MARKS):
            dalpha[mi, j] -= comp_j
            dbeta[j] -= alpha[mi, j] * (te_j * inv_b - inner * inv_b * inv_b)
    grad_ll_theta = np.concatenate([dmu, dalpha.reshape(-1), dbeta])

    b_mat = alpha / beta[None, :]
    rho, grad_b = _rho_and_grad(b_mat)
    pen = 0.0
    grad_pen_theta = np.zeros_like(theta)
    if rho > rho_cap:
        pen = _RHO_PENALTY_W * (rho - rho_cap) ** 2
        if grad_b is not None:
            w = 2.0 * _RHO_PENALTY_W * (rho - rho_cap)
            # d rho/d alpha[m,j] = G[m,j] / beta_j ; d rho/d beta_j = -sum_m G[m,j] alpha[m,j] / beta_j^2
            grad_alpha = w * grad_b / beta[None, :]
            grad_beta = w * (-(grad_b * alpha).sum(axis=0) / (beta**2))
            grad_pen_theta = np.concatenate([np.zeros(N_MARKS), grad_alpha.reshape(-1), grad_beta])
    grad_z = theta * (-grad_ll_theta + grad_pen_theta)
    f = -ll + pen
    if not math.isfinite(f) or not np.all(np.isfinite(grad_z)):
        return 1.0e30, np.zeros_like(z)
    return f, grad_z


def hawkes_mv_loglik(
    streams: MarkStreams,
    mu: Array,
    alpha: Array,
    beta: Array,
    horizon_ms: float,
) -> float:
    """Exact joint loglik of the multivariate exp-kernel Hawkes on ``streams``."""
    times, marks_i = _merge_streams(streams)
    mark_times = [np.asarray(streams.get(name, np.empty(0)), dtype=float) for name in MARKS]
    mu_a = np.asarray(mu, dtype=float)
    alpha_a = np.asarray(alpha, dtype=float)
    beta_a = np.asarray(beta, dtype=float)
    ll, _dmu, _dalpha, _dbeta = _forward(times, marks_i, mu_a, alpha_a, beta_a, float(horizon_ms))
    if not math.isfinite(ll):
        return float("-inf")
    for j, (n_j, e_j, _te_j) in enumerate(_tail_sums(mark_times, beta_a.tolist(), horizon_ms)):
        ll -= float(alpha_a[:, j].sum()) * (n_j - e_j) / float(beta_a[j])
    return float(ll)


@dataclass(frozen=True)
class HawkesMVFit:
    """Fitted multivariate Hawkes: mu (M,), alpha (M,M), beta (M,) per excitor."""

    mu: Array
    alpha: Array
    beta: Array
    branching: Array  # B[m,j] = alpha[m,j] / beta[j]
    rho: float
    stationary: bool
    loglik: float
    converged: bool
    n_events: int
    mark_counts: dict[str, int]


def branching_matrix(alpha: Array, beta: Array) -> Array:
    """B[m,j] = alpha[m,j] / beta[j] — expected mark-m children per mark-j event."""
    out: Array = np.asarray(alpha, dtype=float) / np.asarray(beta, dtype=float)[None, :]
    return out


def spectral_radius(b: Array) -> float:
    """rho(B): dominant Perron eigenvalue magnitude; < 1 is stable (Hawkes 1971b)."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(b, dtype=float)))))


def hawkes_mv_fit(
    streams: MarkStreams,
    *,
    horizon_ms: float | None = None,
    rho_cap: float = _RHO_CAP,
) -> HawkesMVFit:
    """Joint MLE over (mu, alpha, beta) via L-BFGS-B on log-parameters.

    Analytic gradient via the compensator recursion; a squared-hinge penalty
    on rho(B) - rho_cap keeps the reported optimum inside the stable region
    (post-fit check: if the unconstrained optimum still exceeds the cap the
    alpha block is rescaled linearly — which scales rho linearly — and
    ``stationary``/``converged`` stay honest flags on the reported optimum).
    """
    cap = float(rho_cap)
    mark_times = [np.asarray(streams.get(name, np.empty(0)), dtype=float) for name in MARKS]
    counts = [int(t.size) for t in mark_times]
    total = sum(counts)
    times, marks_i = _merge_streams(streams)
    if total < _MIN_EVENTS_TOTAL or min(counts) < _MIN_EVENTS_PER_MARK:
        return HawkesMVFit(
            mu=np.zeros(N_MARKS),
            alpha=np.zeros((N_MARKS, N_MARKS)),
            beta=np.ones(N_MARKS),
            branching=np.zeros((N_MARKS, N_MARKS)),
            rho=0.0,
            stationary=False,
            loglik=float("nan"),
            converged=False,
            n_events=total,
            mark_counts={name: counts[j] for j, name in enumerate(MARKS)},
        )
    if horizon_ms is None:
        horizon_ms = float(times.max())
    # the horizon must cover the last event for the tail compensator
    horizon_ms = max(float(horizon_ms), float(times.max()))

    def obj(z: Array) -> tuple[float, Array]:
        return _objective(z, times, marks_i, mark_times, horizon_ms, cap)

    rates = np.asarray([max(c / max(horizon_ms, 1e-9), 1e-9) for c in counts])
    bounds = [(-25.0, 15.0)] * (2 * N_MARKS + N_MARKS * N_MARKS)
    best_x: Array | None = None
    best_f = math.inf
    best_ok = False
    # deterministic restarts over decay scale; alpha0 set so B row-sum ~ 0.5
    for beta0 in (2.0e-4, 2.0e-3, 2.0e-2):
        theta0 = np.concatenate(
            [
                0.7 * rates,
                np.full(N_MARKS * N_MARKS, 0.5 / N_MARKS * beta0),
                np.full(N_MARKS, beta0),
            ]
        )
        res = minimize(
            obj,
            np.log(theta0),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if np.isfinite(res.fun) and res.fun < best_f:
            best_f = float(res.fun)
            best_x = np.asarray(res.x, dtype=float)
            best_ok = bool(res.success)
    if best_x is None:
        raise ValueError("hawkes_mv_fit: no finite optimum over restarts")

    theta = np.exp(best_x)
    mu, alpha, beta = _pack(theta, N_MARKS)
    b = branching_matrix(alpha, beta)
    rho, _ = _rho_and_grad(b)
    rho_projected = False
    if rho >= cap:
        # linear rescale of alpha scales B (and rho) linearly — honest fallback
        alpha = alpha * (cap / rho)
        b = branching_matrix(alpha, beta)
        rho, _ = _rho_and_grad(b)
        rho_projected = True
    theta_f = np.concatenate([mu.reshape(-1), alpha.reshape(-1), beta.reshape(-1)])
    loglik = -_objective(np.log(theta_f), times, marks_i, mark_times, horizon_ms, cap)[0]
    return HawkesMVFit(
        mu=np.asarray(mu, dtype=float),
        alpha=np.asarray(alpha, dtype=float),
        beta=np.asarray(beta, dtype=float),
        branching=np.asarray(b, dtype=float),
        rho=float(rho),
        stationary=bool(rho < cap),
        loglik=float(loglik),
        converged=bool(best_ok) and not rho_projected,
        n_events=total,
        mark_counts={name: counts[j] for j, name in enumerate(MARKS)},
    )


# ---------------------------------------------------------------------------
# Ogata thinning sampler (test ground truth + negative controls)
# ---------------------------------------------------------------------------


def hawkes_mv_simulate(
    mu: Array,
    alpha: Array,
    beta: Array,
    horizon_ms: float,
    *,
    seed: int = 0,
) -> MarkStreams:
    """Multivariate Ogata (1981) thinning simulation of the exp-kernel Hawkes.

    Candidate arrivals are drawn at the current upper bound
    ``lam_bar = sum_m lam_m(t)``; acceptance samples the mark proportionally
    to ``lam_m``. Fail-closed on supercritical (rho(B) >= 1) parameters —
    a supercritical Hawkes explodes with probability one.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    if mu.shape != (N_MARKS,) or alpha.shape != (N_MARKS, N_MARKS) or beta.shape != (N_MARKS,):
        raise ValueError("mu (M,), alpha (M,M), beta (M,) shapes required")
    if np.any(mu <= 0.0) or np.any(alpha < 0.0) or np.any(beta <= 0.0):
        raise ValueError("mu>0, alpha>=0, beta>0 required")
    if not math.isfinite(horizon_ms) or horizon_ms <= 0.0:
        raise ValueError("horizon_ms must be positive and finite")
    if spectral_radius(branching_matrix(alpha, beta)) >= 1.0:
        raise ValueError("rho(B) >= 1 is supercritical; refusing to simulate")
    rng = np.random.default_rng(seed)
    r = np.zeros(N_MARKS)
    out: list[list[float]] = [[] for _ in range(N_MARKS)]
    t = 0.0
    while True:
        lam = mu + alpha @ r
        lam_bar = float(lam.sum())
        if lam_bar <= 0.0 or not math.isfinite(lam_bar):  # pragma: no cover - guarded
            raise RuntimeError("degenerate intensity upper bound")
        dt = float(rng.exponential(1.0 / lam_bar))
        t += dt
        if t >= horizon_ms:
            break
        r *= np.exp(-beta * dt)
        lam_new = mu + alpha @ r
        total_new = float(lam_new.sum())
        if float(rng.random()) * lam_bar <= total_new:
            probs = lam_new / total_new
            m = int(rng.choice(N_MARKS, p=probs))
            out[m].append(t)
            r[m] += 1.0
    return {MARKS[j]: np.asarray(out[j], dtype=float) for j in range(N_MARKS)}


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def stationary_rates(mu: Array, branching: Array) -> Array | None:
    """Mean event rate per mark under stationarity: lam_bar = (I - B)^{-1} mu."""
    m = np.asarray(mu, dtype=float)
    b = np.asarray(branching, dtype=float)
    if spectral_radius(b) >= 1.0:
        return None
    return np.linalg.solve(np.eye(b.shape[0]) - b, m)


def excitation_share(mu: Array, branching: Array) -> tuple[Array, Array] | None:
    """Per-mark endogenous share decomposition.

    ``share[m,j] = B[m,j] * lam_bar_j / lam_bar_m`` is the fraction of mark m's
    mean intensity attributable to mark-j ancestors; ``endo[m] = sum_j
    share[m,j] = 1 - mu_m / lam_bar_m`` is the reflexivity share of
    Filimonov-Sornette. None when the matrix is not subcritical.
    """
    lam_bar = stationary_rates(mu, branching)
    if lam_bar is None or np.any(lam_bar <= 0.0):
        return None
    b = np.asarray(branching, dtype=float)
    share = b * lam_bar[None, :] / lam_bar[:, None]
    return share, share.sum(axis=1)


def dominant_pairs(branching: Array, *, top: int = 6) -> list[dict[str, Any]]:
    """Ranked B entries as ``excitor -> excited`` pairs, largest first."""
    b = np.asarray(branching, dtype=float)
    flat = np.argsort(-b, axis=None)[:top]
    out = []
    for idx in flat:
        m, j = divmod(int(idx), N_MARKS)
        out.append(
            {
                "pair": f"{MARKS[j]}->{MARKS[m]}",
                "excitor": MARKS[j],
                "excited": MARKS[m],
                "self": bool(m == j),
                "b_mj": round(float(b[m, j]), 5),
            }
        )
    return out


def lag_profile(
    streams_a: MarkStreams,
    streams_b: MarkStreams | None = None,
    *,
    bin_ms: float = 50.0,
    lags: tuple[int, ...] = (0, 1, 2, 4, 8, 16),
) -> dict[str, dict[str, float | None]]:
    """Binned-count cross-correlation per ordered pair at each lag (in bins).

    ``profile["j->m"][lag] = corr(counts_j(t - lag*bin), counts_m(t))`` — the
    lead-lag structure a Hawkes model should reproduce. ``streams_b`` defaults
    to ``streams_a`` (self-profile); passing another dataset compares the same
    marks across sources.
    """
    ref = streams_b if streams_b is not None else streams_a
    t_max = 0.0
    for s in streams_a.values():
        if s.size:
            t_max = max(t_max, float(np.max(s)))
    if streams_b is not None:
        for s in streams_b.values():
            if s.size:
                t_max = max(t_max, float(np.max(s)))
    if t_max <= 0.0:
        return {}
    n_bins = int(t_max // bin_ms) + 2
    if n_bins < 32:
        return {}
    binned: dict[str, Array] = {}
    for name in MARKS:
        c = np.zeros(n_bins)
        sa = np.asarray(ref.get(name, np.empty(0)), dtype=float)
        if sa.size:
            idx = np.minimum((sa / bin_ms).astype(np.int64), n_bins - 1)
            np.add.at(c, idx, 1.0)
        binned[name] = c
    out: dict[str, dict[str, float | None]] = {}
    for excitor in MARKS:
        for excited in MARKS:
            x = binned[excitor]
            y = binned[excited]
            row: dict[str, float | None] = {}
            for lag in lags:
                xs = x[: n_bins - lag] if lag else x
                ys = y[lag:] if lag else y
                if xs.size < 8 or xs.std() == 0.0 or ys.std() == 0.0:
                    row[str(lag)] = None
                else:
                    row[str(lag)] = round(float(np.corrcoef(xs, ys)[0, 1]), 5)
            out[f"{excitor}->{excited}"] = row
    return out


def _fit_block(fit: HawkesMVFit) -> dict[str, Any]:
    """Receipt block for one fitted lane."""
    share_out = excitation_share(fit.mu, fit.branching)
    block: dict[str, Any] = {
        "ok": bool(fit.converged or (np.isfinite(fit.loglik) and fit.stationary)),
        "converged": fit.converged,
        "n_events": fit.n_events,
        "mark_counts": fit.mark_counts,
        "rho_branching": round(float(fit.rho), 5),
        "stationary": fit.stationary,
        "loglik": round(float(fit.loglik), 3),
        "mu_per_s": {MARKS[i]: round(float(fit.mu[i]) * 1000.0, 6) for i in range(N_MARKS)},
        "beta_inv_lifetime_ms": {
            MARKS[j]: round(1.0 / float(fit.beta[j]), 2) for j in range(N_MARKS)
        },
        "branching_matrix": {
            MARKS[m]: {MARKS[j]: round(float(fit.branching[m, j]), 5) for j in range(N_MARKS)}
            for m in range(N_MARKS)
        },
        "dominant_pairs": dominant_pairs(fit.branching),
    }
    if share_out is not None:
        share, endo = share_out
        block["excitation_share"] = {
            MARKS[m]: {MARKS[j]: round(float(share[m, j]), 5) for j in range(N_MARKS)}
            for m in range(N_MARKS)
        }
        block["endo_share"] = {MARKS[m]: round(float(endo[m]), 4) for m in range(N_MARKS)}
    return block


def write_lobster_tape(streams: MarkStreams, path: Path) -> int:
    """Write mark streams back out as a LOBSTER-format message CSV.

    Inverse of ``lobster_marks`` for the SYNTHETIC fixture path: MO_* map to
    EXECUTION rows (direction = resting side, i.e. -aggressor), LO_*_add to
    SUBMISSION, cancel_* to DELETE. Order ids are synthetic sequences;
    price/size are valid-but-token placeholders — the mark extractor reads
    only (time, type, direction).
    """
    rows: list[tuple[float, int, int, int, int, int]] = []
    oid = 0
    for name in MARKS:
        for t_ms in np.asarray(streams.get(name, np.empty(0)), dtype=float):
            oid += 1
            if name == "MO_buy":
                rows.append((t_ms / 1000.0, EXECUTION, oid, 1, 10000, -1))
            elif name == "MO_sell":
                rows.append((t_ms / 1000.0, EXECUTION, oid, 1, 10000, 1))
            elif name == "LO_bid_add":
                rows.append((t_ms / 1000.0, SUBMISSION, oid, 1, 10000, 1))
            elif name == "LO_ask_add":
                rows.append((t_ms / 1000.0, SUBMISSION, oid, 1, 10000, -1))
            elif name == "cancel_bid":
                rows.append((t_ms / 1000.0, DELETE, oid, 1, 10000, 1))
            else:
                rows.append((t_ms / 1000.0, DELETE, oid, 1, 10000, -1))
    rows.sort(key=lambda r: r[0])
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([f"{r[0]:.6f}", r[1], r[2], r[3], r[4], r[5]])
    return len(rows)


def hawkes_mv_bench(
    tape_dir: Path,
    ticker: str = "AMZN",
    *,
    seed: int = 7,
    sim_horizon: int = 20000,
    data_label: str = "SYNTHETIC",
    message_path: Path | None = None,
) -> dict[str, Any]:
    """Six-mark Hawkes on the real tape vs each sim arm. Sealed receipt.

    ``data_label`` is honest bookkeeping: SYNTHETIC when the tape argument is a
    fixture, REAL/MIXED when the caller points at an actual LOBSTER dir. The
    committed receipt in ``receipts/hawkes_mv.json`` is a SYNTHETIC run — the
    tape itself is licensed and not committed.
    """
    msg = message_path or (tape_dir / f"{ticker}_2012-06-21_34200000_57600000_message_10.csv")
    real_marks = lobster_marks(msg)  # FileNotFoundError propagates — fail closed
    real_fit = hawkes_mv_fit(real_marks)

    arms: dict[str, MarkStreams] = {}
    sim_hz: dict[str, float] = {}
    streams, hz = sim_marks(seed=seed, horizon=sim_horizon)
    arms["iid"] = streams
    sim_hz["iid"] = hz
    streams, hz = sim_marks(
        flow=MarkovRegimeFlow(
            states=(RegimeState("calm", 1.0, 0.5), RegimeState("bursty", 3.0, 0.62)),
            stay_probs=(0.995, 0.985),
            seed=seed + 1,
        ),
        seed=seed + 1,
        horizon=sim_horizon,
    )
    arms["regime"] = streams
    sim_hz["regime"] = hz
    streams, hz = sim_marks(
        flow=SplitFlow(
            p_start=0.10,
            size_tail=1.2,
            k_min=10,
            k_max=600,
            intensity_mult=3.0,
            seed=seed + 2,
        ),
        seed=seed + 2,
        horizon=sim_horizon,
    )
    arms["split"] = streams
    sim_hz["split"] = hz

    arm_fits = {name: hawkes_mv_fit(arms[name], horizon_ms=sim_hz[name]) for name in arms}

    probes: list[tuple[str, bool]] = []
    probes.append(("real_tape_fit_ok", bool(np.isfinite(real_fit.loglik))))
    probes.append(("real_tape_stationary", real_fit.stationary))
    probes.append(("real_tape_events_sufficient", real_fit.n_events >= _MIN_EVENTS_TOTAL))
    for name, fit in arm_fits.items():
        probes.append((f"{name}_fit_ok", bool(np.isfinite(fit.loglik))))
        probes.append((f"{name}_stationary", fit.stationary))
    n_passed = sum(1 for _, ok in probes if ok)

    delta_rho = {name: round(real_fit.rho - arm_fits[name].rho, 5) for name in arm_fits}

    # parity table: real B[m,j] vs each arm's, per ordered pair
    parity: dict[str, dict[str, float]] = {}
    for j in range(N_MARKS):
        for m in range(N_MARKS):
            key = f"{MARKS[j]}->{MARKS[m]}"
            parity[key] = {
                "real": round(float(real_fit.branching[m, j]), 5),
                **{name: round(float(arm_fits[name].branching[m, j]), 5) for name in arm_fits},
            }

    # lag profile for the dominant real pairs (lead-lag shape parity)
    top_pairs = [d["pair"] for d in dominant_pairs(real_fit.branching, top=4)]
    lag_tables: dict[str, Any] = {}
    real_lags = lag_profile(real_marks)
    arm_lags = {name: lag_profile(arms[name]) for name in arms}
    for pair in top_pairs:
        lag_tables[pair] = {
            "lags_bins_of_50ms": [0, 1, 2, 4, 8, 16],
            "real": real_lags.get(pair, {}),
            **{name: arm_lags[name].get(pair, {}) for name in arms},
        }

    divergences: list[str] = []
    for name, fit in arm_fits.items():
        if abs(fit.rho - real_fit.rho) > 0.3:
            divergences.append(f"{name}_rho_far_from_real")
    if real_fit.rho < 1e-3:
        divergences.append("real_tape_no_excitation")

    payload: dict[str, Any] = {
        "kind": "hawkes_mv",
        "schema": HAWKES_MV_SCHEMA,
        "ticker": ticker,
        "marks": list(MARKS),
        "kernel": "lambda_m(t) = mu_m + sum_j alpha_mj * exp(-beta_j (t - t_i^j)); B_mj = alpha_mj/beta_j",
        "estimator": (
            "joint MLE via O(n) compensator recursion + analytic gradient, "
            "L-BFGS-B on log-params, soft rho(B)<0.999 barrier"
        ),
        "real": _fit_block(real_fit),
        "sim_arms": {name: _fit_block(arm_fits[name]) for name in arm_fits},
        "delta_rho_vs_real": delta_rho,
        "parity_per_pair": parity,
        "lag_profile_top_pairs": lag_tables,
        "divergences": divergences,
        "probes": [{"name": n, "passed": bool(p)} for n, p in probes],
        "claim": {
            "results": {
                "rho_real": round(float(real_fit.rho), 5),
                "rho_iid": round(float(arm_fits["iid"].rho), 5),
                "rho_regime": round(float(arm_fits["regime"].rho), 5),
                "rho_split": round(float(arm_fits["split"].rho), 5),
                "top_pair": top_pairs[0] if top_pairs else None,
            },
            "ok": n_passed == len(probes),
            "n_probes": len(probes),
            "n_passed": n_passed,
        },
        "interpretation": (
            "B[m,j] is the expected number of mark-m children per mark-j "
            "event; rho(B)<1 is the Hawkes stability boundary. Off-diagonal "
            "cells are genuine cross-excitation (e.g. MO -> cancel, LO -> MO) "
            "that the univariate eta in hawkes_real cannot see. The sim arms "
            "isolate mechanisms: iid flow should sit near rho~0, regime flow "
            "excites only through intensity modulation, split flow should "
            "reproduce the MO->MO diagonal block but not the cross-mark "
            "structure the real tape shows."
        ),
        "git_revision": git_revision(),
        "data_label": data_label,
        "research_only": True,
        "live_pnl_claim": False,
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


__all__ = [
    "HAWKES_MV_SCHEMA",
    "MARKS",
    "HawkesMVFit",
    "MarkStreams",
    "branching_matrix",
    "dominant_pairs",
    "excitation_share",
    "hawkes_mv_bench",
    "hawkes_mv_fit",
    "hawkes_mv_loglik",
    "hawkes_mv_simulate",
    "lag_profile",
    "lobster_marks",
    "sim_marks",
    "spectral_radius",
    "stationary_rates",
    "write_lobster_tape",
]
