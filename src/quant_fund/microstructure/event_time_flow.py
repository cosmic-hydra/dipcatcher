"""Event-time order-flow memory, operational-time impact, subordinated observables.

Implements the clock-separation framework of

    Angstmann, C., Gebbie, T. (2026). "Event-Time Order-Flow Memory,
    Operational-Time Impact, and Subordinated Market Observables."
    arXiv:2609.13715 [q-fin.TR]. Citation verified against
    https://arxiv.org/abs/2609.13715 (fetched 2026-09-30): title, authors
    and abstract match this lane's spec verbatim.

The paper's point is that two canonical microstructure regularities live on
DIFFERENT clocks, and the empirically observed anomalies are projections:

1. **Event time.** The long memory of trade signs is a statement about the
   ordering and fragmentation of hidden orders: a metaorder of total length
   L is executed as L consecutive same-sign child trades, so the sign
   autocorrelation decays with EVENT COUNT, not wall-clock. We generate the
   canonical Lillo-Mike-Farmer / Tóth-Palit-Lillo-Farmer fragmentation
   ensemble: covering blocks of iid length L with a discrete power-law tail
   P(L >= l) = l^{-alpha}, alpha in (1, 2), and iid block signs. For this
   ensemble the covering sum telescopes exactly: C(k) = zeta(alpha, k+1) /
   zeta(alpha) ~ k^{-(alpha-1)} / ((alpha-1) zeta(alpha)) in event lag k
   (Lillo, Mike & Farmer 2005, Phys. Rev. E 71:066122; Tóth, Palit, Lillo &
   Farmer 2015, JEDC 51:218-239).

2. **Operational time.** The square-root metaorder impact law is a statement
   about front motion in a locally linear latent book — i.e. about the
   cumulative event-count of the executing flow. Here impact is represented
   by a transient propagator kernel G(l) = g0 * l^{-beta} on event lags
   (Bouchaud, Mézard & Potters 2002; Bouchaud, Farmer & Lillo 2009 handbook
   chapter): a metaorder that has executed n child events carries expected
   impact I(n) = sum_{k<=n} G(k) ~ n^{1-beta}, the square-root law at
   beta = 1/2. The kernel lives on event lags — it is the operational-time
   object.

3. **Calendar-time subordination.** The observable at wall-clock t is the
   event-time process composed with the stochastic counting process N(t) =
   #{events <= t}: S_cal(t) = S(N(t)), I_cal(T) = sum_k G(k) 1{k <= N_part(T)}.
   When the clock has finite-mean waiting times (Poisson, Markov-modulated
   regimes, exponential Hawkes), asymptotic exponents are preserved and only
   finite-horizon amplitudes/curvature shift. When the waiting-time tail is
   infinite-mean (fractional/tempered clock), the apparent calendar-time
   law is the event-time law RESCALED by the clock index: E[N(tau)] ~ tau^mu
   turns C(k) ~ k^{-gamma} into C_cal(tau) ~ tau^{-mu*gamma} and I(n) ~ n^{1-beta}
   into I_cal(T) ~ T^{mu*(1-beta)} — the "anomalous" fractional or tempered
   calendar-time laws of the paper arise from the event-to-calendar
   projection, not from a different operational-time mechanism.

   Two anchoring conventions produce two distinct observable distortions:
   EVENT-anchored pairs (metaorder impact curves, the event-conditioned sign
   ACF) realize the origin-anchored mixture E[C(N(tau))] and pick up the mu
   rescaling described above; UNIFORM-calendar probes under an infinite-mean
   clock land overwhelmingly inside a few giant waiting-time gaps, so the
   uniform-probe calendar ACF reads a near-plateau — a further amplification
   of the same projection. The operational-time correction — conditioning
   calendar-time pairs on their REALIZED event lag Delta N, or equivalently
   re-expressing calendar lags in mean event counts — recovers the
   event-time law; under infinite-mean clocks the conditioned estimator
   stays unbiased in the bulk but its sampling variance is dominated by a
   few giant gaps (and finite streams truncate covering blocks at
   ~n^{1/alpha}), so recovery degrades at lags beyond the well-populated
   bins — an honest, documented limit rather than a defect.

Composition (import, do not reimplement)
----------------------------------------
- ``microstructure.zi_lob_simulator`` — a real event-driven LOB whose trade
  tape is one valid EventStream: ``zi_lob_event_stream`` wraps
  ``ZILobSimulator`` (+ optional ``MarkovRegimeFlow`` for clustered MO
  intensity) instead of re-implementing a matching engine. (The wave-18
  ``agentic_lob`` lane exists only on ``devin/w18-*`` branches, not on main;
  this module composes zi_lob only, per the lane spec.)
- ``models.changepoint.binary_segmentation`` — the mean-shift segmenter used
  by ``activity_rate_boundaries`` to locate activity-regime boundaries in
  binned event counts. Not reimplemented.
- ``utils.series.finite_series`` — shared finiteness/length guard.

Honesty: every simulated path, kernel, autocorrelation, exponent fit and
bench number produced here is SYNTHETIC correctness evidence only — never
market evidence — and is labeled ``synthetic_*`` in report/bench keys. All
outputs are statistical diagnostics (kernel L2 recovery error, log-log
exponent estimates, counting-process moments, changepoint locations); no
PnL/Sharpe-family quantity exists anywhere in this module. There is no
broker connectivity and no live-trading claim. Degenerate inputs raise:
empty or non-increasing event times, single-event windows, non-finite or
non-positive rates, non-{+1,-1} sign streams, zero-variance interarrivals,
unattainable lags and saturated kernels all fail closed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import zeta as _sp_zeta
from scipy.stats import poisson as _poisson

from quant_fund.microstructure.zi_lob_simulator import (
    MarkovRegimeFlow,
    ZILobConfig,
    ZILobSimulator,
    santa_fe_config,
)

Array = NDArray[np.float64]
IntArray = NDArray[np.intp]
RNG = np.random.Generator | int

CLOCK_KINDS: tuple[str, ...] = ("poisson", "regime", "hawkes", "pareto")
EVENT_TIME_FLOW_REVISION = "SYNTHETIC_EVENT_TIME_FLOW_v1"


# ---------------------------------------------------------------------------
# Fail-closed validation helpers
# ---------------------------------------------------------------------------


def _pos_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return v


def _nonneg_finite(x: float, name: str) -> float:
    v = float(x)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{name} must be non-negative and finite, got {x!r}")
    return v


def _finite_1d(values: object, name: str, min_size: int = 1) -> Array:
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size < min_size:
        raise ValueError(f"{name} must contain at least {min_size} values")
    if not bool(np.all(np.isfinite(v))):
        raise ValueError(f"{name} must be finite")
    return v


def _check_n(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


def _resolve_rng(rng: RNG | None) -> np.random.Generator:
    if rng is None:
        raise ValueError("rng is required (seed int or Generator) — streams are seeded only")
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, bool) or not isinstance(rng, (int, np.integer)):
        raise ValueError(f"rng must be an int seed or Generator, got {rng!r}")
    return np.random.default_rng(int(rng))


def _check_times(times: object, name: str = "times") -> Array:
    t = _finite_1d(times, name, min_size=2)
    if bool(np.any(np.diff(t) <= 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return t


def _check_signs(signs: object, n: int | None = None) -> Array:
    s = _finite_1d(signs, "signs", min_size=2)
    if not bool(np.all((s == 1.0) | (s == -1.0))):
        raise ValueError("signs must take values in {-1, +1}")
    if n is not None and s.size != n:
        raise ValueError("signs must match the event count")
    return s


def _check_lags(lags: object, n: int) -> IntArray:
    lg = np.asarray(lags, dtype=np.intp).reshape(-1)
    if lg.size == 0:
        raise ValueError("lags must be non-empty")
    if bool(np.any(lg < 1)) or bool(np.any(lg >= n)):
        raise ValueError(f"lags must lie in [1, {n - 1}]")
    if np.unique(lg).size != lg.size:
        raise ValueError("lags must be distinct")
    return lg


def _check_taus(taus: object) -> Array:
    t = _finite_1d(taus, "taus", min_size=1)
    if bool(np.any(t <= 0.0)):
        raise ValueError("taus must be positive")
    if bool(np.any(np.diff(t) <= 0.0)):
        raise ValueError("taus must be strictly increasing")
    return t


def _check_kernel(kernel: object, min_lags: int = 2) -> Array:
    g = _finite_1d(kernel, "kernel", min_size=min_lags)
    if bool(np.any(g < 0.0)):
        raise ValueError("kernel values must be non-negative (a propagator kernel)")
    if float(g.sum()) <= 0.0:
        raise ValueError("kernel must carry positive mass")
    return g


def _check_zero_value(zero_value: float) -> float:
    v = float(zero_value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"zero_value must be non-negative and finite, got {zero_value!r}")
    return v


# ---------------------------------------------------------------------------
# Event clocks (the stochastic subordinator N(t))
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventClock:
    """Stochastic event clock specification (the subordinator of the paper).

    Kinds:

    - ``"poisson"``: homogeneous Poisson process at rate ``rate`` — iid
      exponential waiting times, the memoryless baseline.
    - ``"regime"``: two-state Markov-modulated intensity indexed on the
      EVENT clock (mirroring ``MarkovRegimeFlow``'s MO-clock design): in
      state i the next waiting time is Exp(``regime_rates[i]``) and the chain
      stays with probability ``stay_probs[i]`` after each event. Sticky
      states with unequal rates produce clustered calendar-time activity;
      equal rates are a degenerate regime model and fail closed.
    - ``"hawkes"``: self-exciting exponential Hawkes, intensity
      lam(t) = hawkes_mu + sum_i hawkes_alpha * hawkes_beta *
      exp(-hawkes_beta (t - t_i)) — ``hawkes_alpha`` is the branching ratio
      (must be < 1 for stability), stationary mean intensity
      hawkes_mu / (1 - hawkes_alpha). Simulated by exact Ogata thinning with
      an O(1) recursive excitation (exponential kernel). Finite-mean waits:
      asymptotic exponents survive subordination; finite-horizon distortion
      does not.
    - ``"pareto"``: iid waiting times with power-law tail
      P(W > w) = (w / tail_scale)^{-tail_exponent}, tail_exponent in (0, 1)
      — the infinite-mean fractional clock. N(tau) grows as tau^mu, so the
      calendar projection genuinely rescales the memory exponent (mu*gamma)
      and the impact exponent (mu*(1-beta)).

    ``times`` returns the first ``n_events`` event times; ``times_until``
    returns all event times in ``[0, horizon]``. Both are seeded only.
    """

    kind: str
    rate: float = 1.0
    regime_rates: tuple[float, float] = (0.5, 3.0)
    stay_probs: tuple[float, float] = (0.99, 0.99)
    hawkes_mu: float = 0.4
    hawkes_alpha: float = 0.7
    hawkes_beta: float = 1.0
    tail_exponent: float = 0.7
    tail_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in CLOCK_KINDS:
            raise ValueError(f"kind must be one of {CLOCK_KINDS}, got {self.kind!r}")
        _pos_finite(self.rate, "rate")
        for r in self.regime_rates:
            _pos_finite(r, "regime_rates")
        if len(self.regime_rates) != 2:
            raise ValueError("regime_rates must have exactly two entries")
        for p in self.stay_probs:
            v = float(p)
            if not math.isfinite(v) or v <= 0.0 or v > 1.0:
                raise ValueError(f"stay_probs must lie in (0, 1], got {p!r}")
        if len(self.stay_probs) != 2:
            raise ValueError("stay_probs must have exactly two entries")
        _pos_finite(self.hawkes_mu, "hawkes_mu")
        a = float(self.hawkes_alpha)
        if not math.isfinite(a) or a < 0.0 or a >= 1.0:
            raise ValueError(f"hawkes_alpha (branching ratio) must lie in [0, 1), got {a!r}")
        _pos_finite(self.hawkes_beta, "hawkes_beta")
        mu = float(self.tail_exponent)
        if not math.isfinite(mu) or mu <= 0.0 or mu >= 1.0:
            raise ValueError(f"tail_exponent must lie in (0, 1), got {mu!r}")
        _pos_finite(self.tail_scale, "tail_scale")
        if self.kind == "regime" and float(self.regime_rates[0]) == float(self.regime_rates[1]):
            raise ValueError("regime clock with equal rates is degenerate (constant rate)")

    def mean_intensity(self) -> float | None:
        """Stationary events/unit-time where it exists; ``None`` for pareto."""
        if self.kind == "poisson":
            return float(self.rate)
        if self.kind == "regime":
            p01, p10 = 1.0 - self.stay_probs[0], 1.0 - self.stay_probs[1]
            w0 = p10 / (p01 + p10)  # stationary event fraction in state 0
            # events per unit TIME is the harmonic rate: each event in state
            # i occupies Exp(rate_i) calendar time, so time weights w_i/rate_i
            return float(1.0 / (w0 / self.regime_rates[0] + (1.0 - w0) / self.regime_rates[1]))
        if self.kind == "hawkes":
            if self.hawkes_alpha == 0.0:
                return float(self.hawkes_mu)
            return float(self.hawkes_mu / (1.0 - self.hawkes_alpha))
        return None  # pareto: infinite-mean waits have no stationary rate

    def _regime_times(
        self, rng: np.random.Generator, n: int | None, horizon: float | None
    ) -> Array:
        rates = self.regime_rates
        stay = self.stay_probs
        out: list[float] = []
        t = 0.0
        state = 0
        target = n if n is not None else _check_n(int(1e9), "n", 1)  # horizon-bound
        while len(out) < target:
            e = float(rng.standard_exponential()) / rates[state]
            t += e
            if horizon is not None and t > horizon:
                break
            out.append(t)
            if float(rng.random()) > stay[state]:
                state = 1 - state
        return np.asarray(out, dtype=float)

    def _hawkes_times(
        self, rng: np.random.Generator, n: int | None, horizon: float | None
    ) -> Array:
        mu, alpha, beta = self.hawkes_mu, self.hawkes_alpha, self.hawkes_beta
        out: list[float] = []
        t = 0.0
        excitation = 0.0  # E = sum_i alpha*beta*exp(-beta*(t - t_i)); O(1) recursion
        lam_bar = mu
        target = n if n is not None else _check_n(int(1e9), "n", 1)
        while len(out) < target:
            dt = float(rng.exponential(1.0 / lam_bar))
            t += dt
            if horizon is not None and t > horizon:
                break
            excitation *= math.exp(-beta * dt)
            lam_t = mu + excitation
            if float(rng.random()) * lam_bar <= lam_t:
                out.append(t)
                excitation += alpha * beta
                lam_bar = lam_t + alpha * beta
            else:
                lam_bar = lam_t  # intensity is decaying: keep the tight bound
        return np.asarray(out, dtype=float)

    def _block_times(self, rng: np.random.Generator, horizon: float) -> Array:
        """Batch-draw waiting times until ``horizon`` is covered (iid clocks)."""
        chunks: list[Array] = []
        total = 0.0
        while total <= horizon or not chunks:
            size = 4096
            u = 1.0 - rng.random(size)  # (0, 1]
            if self.kind == "poisson":
                gaps = -np.log(u) / self.rate
            else:  # pareto
                gaps = self.tail_scale * u ** (-1.0 / self.tail_exponent)
            chunk = np.cumsum(gaps)
            chunks.append(chunk if not chunks else chunk + total)
            total = float(chunks[-1][-1])
        times = np.concatenate(chunks)
        return times[: int(np.searchsorted(times, horizon, side="right"))]

    def times(self, n_events: int, rng: RNG) -> Array:
        """First ``n_events`` event times under this clock (seeded)."""
        n = _check_n(n_events, "n_events", 2)
        g = _resolve_rng(rng)
        if self.kind == "poisson":
            gaps = g.standard_exponential(n) / self.rate
            return np.asarray(np.cumsum(gaps), dtype=float)
        if self.kind == "pareto":
            u = 1.0 - g.random(n)
            return np.asarray(np.cumsum(self.tail_scale * u ** (-1.0 / self.tail_exponent)))
        if self.kind == "regime":
            return self._regime_times(g, n, None)
        return self._hawkes_times(g, n, None)

    def times_until(self, horizon: float, rng: RNG) -> Array:
        """All event times in ``(0, horizon]`` (seeded)."""
        h = _pos_finite(horizon, "horizon")
        g = _resolve_rng(rng)
        if self.kind in ("poisson", "pareto"):
            return self._block_times(g, h)
        if self.kind == "regime":
            return self._regime_times(g, None, h)
        return self._hawkes_times(g, None, h)


# ---------------------------------------------------------------------------
# Event streams
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EventStream:
    """An event-indexed observable stream on a calendar-time event clock.

    ``times`` are the strictly increasing event times (the realisation of the
    subordinator); ``signs`` (optional) are the {-1, +1} trade signs on the
    event clock; ``meta_ids`` (optional) mark which hidden parent metaorder
    each event belongs to (the fragmentation structure of the paper).
    """

    times: Array
    signs: Array | None = None
    meta_ids: IntArray | None = None
    label: str = "stream"

    def __post_init__(self) -> None:
        t = _check_times(self.times)
        object.__setattr__(self, "times", t)
        if self.signs is not None:
            object.__setattr__(self, "signs", _check_signs(self.signs, t.size))
        if self.meta_ids is not None:
            m = np.asarray(self.meta_ids, dtype=np.intp).reshape(-1)
            if m.size != t.size:
                raise ValueError("meta_ids must match the event count")
            object.__setattr__(self, "meta_ids", m)

    @property
    def n_events(self) -> int:
        return int(self.times.size)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])


def zi_lob_event_stream(
    horizon: float,
    *,
    config: ZILobConfig | None = None,
    flow: MarkovRegimeFlow | None = None,
) -> EventStream:
    """Trade tape of the ZI-LOB simulator as an :class:`EventStream`.

    Composes ``ZILobSimulator`` (never re-implements a matching engine): the
    simulator's own event clock supplies calendar times, and each trade's
    aggressor maps to sign +1 (buy) / -1 (sell). Passing a
    ``MarkovRegimeFlow`` gives clustered MO intensity — the clustered
    activity-regime clock of the paper. Fail-closed when the horizon yields
    fewer than 2 trades.
    """
    h = _pos_finite(horizon, "horizon")
    cfg = config if config is not None else santa_fe_config(seed=0)
    sim = ZILobSimulator(cfg, flow)
    sim.run(h)
    if len(sim.trades) < 2:
        raise ValueError(f"horizon {h} produced {len(sim.trades)} trades (< 2)")
    times = np.asarray([tr.t for tr in sim.trades], dtype=float)
    signs = np.asarray([1.0 if tr.aggressor == "buy" else -1.0 for tr in sim.trades])
    return EventStream(times=times, signs=signs, label="zi_lob")


# ---------------------------------------------------------------------------
# Event-time sign memory: hidden-order fragmentation (LMF/TPLF ensemble)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LMFSigns:
    """Fragmented-metaorder sign sequence on the event clock."""

    signs: Array
    meta_ids: IntArray
    length_exponent: float
    n_metaorders: int


def simulate_lmf_signs(
    n_events: int,
    length_exponent: float = 1.5,
    p_buy: float = 0.5,
    rng: RNG = 0,
) -> LMFSigns:
    """Hidden-order fragmentation sign stream (Lillo-Mike-Farmer ensemble).

    The event index is tiled by covering blocks whose lengths are iid with
    discrete power-law tail P(L >= l) = l^{-alpha} (drawn exactly as
    ``floor(U^{-1/alpha})``), each block carrying one iid Rademacher sign
    (+1 with probability ``p_buy``). Event n's sign is the sign of the block
    covering it; ``meta_ids`` records the covering block index. The block
    covering event 0 is entered at a uniform internal offset (approximate
    stationarity — documented, not the full size-biased construction, which
    is improper for alpha < 2 anyway since E[L^2] diverges).

    For 1 < alpha < 2 the sign autocorrelation decays as
    C(k) ~ k^{-(alpha - 1)} in EVENT lag — the long-memory sign law of the
    paper lives on this clock, independent of any calendar projection.

    SYNTHETIC correctness ensemble only; never market evidence.
    """
    n = _check_n(n_events, "n_events", 16)
    alpha = float(length_exponent)
    if not math.isfinite(alpha) or alpha <= 1.0 or alpha >= 2.0:
        raise ValueError(f"length_exponent must lie in (1, 2), got {length_exponent!r}")
    pb = float(p_buy)
    if not math.isfinite(pb) or pb < 0.0 or pb > 1.0:
        raise ValueError(f"p_buy must be a probability, got {p_buy!r}")
    g = _resolve_rng(rng)
    mean_len = 1.0 / (1.0 - 2.0 ** (-alpha))  # E[floor(U^{-1/a})] asymptotic ~ zeta(a)
    chunk = max(64, int(1.5 * n / max(mean_len, 1.0)))
    lengths: list[int] = []
    while sum(lengths) < n + chunk:
        u = 1.0 - g.random(chunk)
        lengths.extend(np.floor(u ** (-1.0 / alpha)).astype(np.intp).tolist())
    lens = np.asarray(lengths, dtype=np.intp)
    block_signs = np.where(g.random(lens.size) < pb, 1.0, -1.0)
    offset = int(g.integers(0, int(lens[0])))  # uniform position in block 0
    signs = np.repeat(block_signs, lens)[offset : offset + n]
    meta_ids = np.repeat(np.arange(lens.size, dtype=np.intp), lens)[offset : offset + n]
    if signs.size < n:  # pragma: no cover - the coverage loop guards this
        raise RuntimeError("fragmentation coverage fell short")
    return LMFSigns(
        signs=np.asarray(signs, dtype=float),
        meta_ids=np.asarray(meta_ids, dtype=np.intp),
        length_exponent=alpha,
        n_metaorders=int(np.unique(meta_ids).size),
    )


def lmf_acf_theory(lags: object, length_exponent: float) -> Array:
    """Exact covering-model sign autocorrelation C(k) for the LMF ensemble.

    For contiguous covering blocks with iid length pmf p(l) = l^{-alpha} -
    (l+1)^{-alpha} and independent Rademacher block signs, stationarity gives

        C(k) = P(0 and k share a block) = sum_{l > k} p(l) (l - k) / E[L]
             = zeta(alpha, k + 1) / zeta(alpha)

    — the covering sum telescopes term-by-term onto the Hurwitz zeta
    (E[L] = zeta(alpha) for this pmf; C(0) = 1 falls out since
    zeta(alpha, 1) = zeta(alpha)). Asymptotically C(k) ~ k^{-(alpha-1)} /
    ((alpha-1) zeta(alpha)). Used as the closed-form reference for the
    simulator and as the event-time kernel inside the subordination
    mixture E[C(N(tau))].
    """
    alpha = float(length_exponent)
    if not math.isfinite(alpha) or alpha <= 1.0 or alpha >= 2.0:
        raise ValueError(f"length_exponent must lie in (1, 2), got {length_exponent!r}")
    ks = np.asarray(lags, dtype=np.intp).reshape(-1)
    if ks.size == 0:
        raise ValueError("lags must be non-empty")
    if bool(np.any(ks < 0)):
        raise ValueError("lags must be >= 0")
    return np.asarray(_sp_zeta(alpha, ks + 1.0) / float(_sp_zeta(alpha, 1.0)))


def sign_autocorrelation(signs: object, lags: object) -> Array:
    """Event-time sign autocorrelation C(k) = corr(eps_i, eps_{i+k}).

    Standard biased estimator: mean_t[(eps_t - m)(eps_{t+k} - m)] /
    mean_t[(eps_t - m)^2] at EVENT lags. Under LMF covering ensembles it
    reproduces :func:`lmf_acf_theory` up to the finite-stream truncation —
    no block longer than the stream exists, so estimated C(k) cuts off for
    k gtrsim n^{1/alpha}; keep queried lags well below that scale.
    Fail-closed on a constant sign stream (zero variance — memory
    undefined).
    """
    s = _check_signs(signs)
    lg = _check_lags(lags, s.size)
    m = float(s.mean())
    v = float(np.mean((s - m) ** 2))
    if v <= 0.0:
        raise ValueError("degenerate constant sign stream — autocorrelation undefined")
    d = s - m
    return np.asarray([float(np.mean(d[:-k] * d[k:]) / v) for k in lg])


# ---------------------------------------------------------------------------
# Operational-time impact: transient propagator on event lags
# ---------------------------------------------------------------------------


def propagator_kernel(n_lags: int, g0: float = 1.0, beta: float = 0.5) -> Array:
    """Transient propagator G(l) = g0 * l^{-beta} on event lags l = 1..n_lags.

    beta = 1/2 is the locally-linear-book regime of the paper: a metaorder's
    expected cumulative impact I(n) = sum_{k<=n} G(k) ~ n^{1-beta} is the
    square-root law in event count (operational time), NOT in calendar time.
    """
    n = _check_n(n_lags, "n_lags", 2)
    gg = _pos_finite(g0, "g0")
    b = _pos_finite(beta, "beta")
    lags = np.arange(1, n + 1, dtype=float)
    return np.asarray(gg * lags ** (-b), dtype=float)


@dataclass(frozen=True)
class PropagatorPath:
    """Event-time price path under the propagator model."""

    prices: Array  # p_n = sum_{l>=1} G(l) eps_{n-l} + eta_n (p_0 = eta_0)
    moves: Array  # dp_n = p_n - p_{n-1}, length n_events - 1
    kernel: Array


def simulate_propagator_path(
    signs: object,
    kernel: object,
    eta_sigma: float = 0.0,
    rng: RNG = 0,
) -> PropagatorPath:
    """Price path p_n = sum_{l=1..L} G(l) eps_{n-l} + eta_n on the event clock.

    Convention: trade i's contribution to the price at event index n is
    G(n - i), so p_0 = eta_0 and p_1 = G(1) eps_0 + eta_1 — each trade first
    impacts the very next event index (the paper's front moves as events
    accumulate, never ahead of them). eta_n is iid N(0, eta_sigma^2)
    microstructure noise from the seeded generator.
    """
    s = _check_signs(signs)
    g = _check_kernel(kernel)
    sig = _nonneg_finite(eta_sigma, "eta_sigma")
    gen = _resolve_rng(rng)
    conv = np.convolve(s, g)  # conv[n-1] = sum_l G(l) eps_{n-l}
    prices = np.concatenate(([0.0], conv[: s.size - 1]))
    if sig > 0.0:
        prices = prices + sig * gen.standard_normal(s.size)
    prices = np.asarray(prices, dtype=float)
    return PropagatorPath(prices=prices, moves=np.asarray(np.diff(prices)), kernel=g)


def metaorder_impact_curve(kernel: object, n_exec: int | None = None) -> Array:
    """Expected impact I(n) = sum_{k<=n} G(k) after n executed child events.

    The operational-time impact law of the paper: for G(l) ~ l^{-beta},
    I(n) ~ n^{1-beta} — square-root when beta = 1/2.
    """
    g = _check_kernel(kernel)
    m = g.size if n_exec is None else _check_n(n_exec, "n_exec", 1)
    if m > g.size:
        raise ValueError("n_exec exceeds the kernel support")
    return np.asarray(np.cumsum(g)[:m], dtype=float)


def estimate_event_kernel(prices: object, signs: object, n_lags: int) -> Array:
    """Least-squares estimate of G(1..L) from an event-time price path.

    Regression p_n = c + sum_{l=1..L} g_l eps_{n-l} + e_n over
    n = L..N-1 (an intercept absorbs any drift). For iid signs this is the
    exact lagged cross-covariance; under autocorrelated (fragmented) signs it
    is the consistent linear projection. Fail-closed when the design is
    underdetermined (rows < max(2L, 32)).
    """
    p = _finite_1d(prices, "prices", min_size=8)
    s = _check_signs(signs, p.size)
    L = _check_n(n_lags, "n_lags", 1)
    rows = p.size - L
    if rows < max(2 * L, 32):
        raise ValueError(
            f"need at least max(2*n_lags, 32)={max(2 * L, 32)} usable rows, got {rows}"
        )
    x = np.empty((rows, L + 1), dtype=np.float64)
    x[:, 0] = 1.0
    for lag in range(1, L + 1):
        x[:, lag] = s[L - lag : p.size - lag]
    # Contiguous float64 design + scipy's driver: numpy.linalg.lstsq under a
    # mismatched OPENBLAS_CORETYPE (e.g. HASWELL on AMD EPYC) has segfaulted
    # here and taken the xdist worker with it ("node down", no Python frame
    # on the faulting OpenBLAS thread). scipy.linalg.lstsq is the same solve
    # with a more defensive LAPACK binding.
    y = np.ascontiguousarray(p[L:], dtype=np.float64)
    from scipy.linalg import lstsq as _lstsq

    coef, _, _, _ = _lstsq(np.ascontiguousarray(x), y, lapack_driver="gelsy")
    return np.asarray(coef[1:], dtype=float)


def kernel_rel_l2(reference: object, estimate: object) -> float:
    """Relative L2 kernel-shape error ||g - g_hat||_2 / ||g||_2."""
    g = _check_kernel(reference, min_lags=1)
    h = _finite_1d(estimate, "estimate", min_size=1)
    if g.size != h.size:
        raise ValueError("reference and estimate must have the same length")
    denom = float(np.linalg.norm(g))
    if denom <= 0.0:
        raise ValueError("reference kernel has zero norm")
    return float(np.linalg.norm(g - h) / denom)


# ---------------------------------------------------------------------------
# Calendar-time subordination: compose observables with N(t)
# ---------------------------------------------------------------------------


def _count_path(times: Array, grid: Array) -> IntArray:
    """N(grid) for a simulated clock path that may hold 0 or 1 events.

    Public :func:`counting_process` fail-closes on degenerate streams; inside
    Monte-Carlo loops a heavy-tailed clock legitimately produces near-empty
    paths (N = 0 contributes ``zero_value``, N = 1 a single step).
    """
    if times.size == 0:
        return np.zeros(grid.size, dtype=np.intp)
    return np.searchsorted(times, grid, side="right").astype(np.intp)


def counting_process(times: object, grid: object) -> IntArray:
    """N(t) = #{events <= t} evaluated on an increasing time grid."""
    tt = _check_times(times)
    gr = _finite_1d(grid, "grid", min_size=1)
    if bool(np.any(np.diff(gr) < 0.0)):
        raise ValueError("grid must be non-decreasing")
    return np.searchsorted(tt, gr, side="right").astype(np.intp)


def subordinated_signs(signs: object, times: object, grid: object) -> Array:
    """S_cal(t) = sign of the most recent event at t (tick-rule projection).

    This is the paper's subordination applied to the sign observable itself:
    the event-indexed sign process read through the stochastic clock N(t).
    Fail-closed on grid points before the first event (N(t) = 0 has no sign).
    """
    s = _check_signs(signs)
    tt = _check_times(times)
    if s.size != tt.size:
        raise ValueError("signs must match the event count")
    gr = _finite_1d(grid, "grid", min_size=1)
    if bool(np.any(np.diff(gr) < 0.0)):
        raise ValueError("grid must be non-decreasing")
    if float(gr[0]) < float(tt[0]):
        raise ValueError("grid starts before the first event — N(t) = 0 has no sign")
    idx = np.clip(counting_process(tt, gr) - 1, 0, s.size - 1)
    return np.asarray(s[idx])


@dataclass(frozen=True)
class CalendarACF:
    """Calendar-time sign autocorrelation with its operational-time remap."""

    taus: Array  # calendar lags probed
    acf: Array  # corr(S_cal(t), S_cal(t + tau)) over uniform probes
    operational_lags: Array  # mean event count E[N(t+tau) - N(t)] per tau
    n_probes: int


def calendar_sign_acf(
    signs: object,
    times: object,
    taus: object,
    probe_spacing: float | None = None,
) -> CalendarACF:
    """Calendar-time autocorrelation of the subordinated sign process.

    Probes S_cal(t) = S(N(t)) on a uniform grid of spacing ``probe_spacing``
    (default: mean interarrival / 4) over the realised horizon, then
    correlates S_cal(t) with S_cal(t + tau) for each calendar lag tau.
    ``operational_lags`` reports the mean event count elapsed per tau on the
    SAME stream — the operational-time remap that, applied before fitting the
    exponent, recovers the event-time memory law the paper predicts. Under
    finite-mean clocks (poisson/regime/hawkes) the probed ACF tracks the
    origin-anchored mixture; under the infinite-mean pareto clock probes
    land inside giant gaps and the apparent ACF amplifies toward a plateau
    — use :func:`event_anchored_sign_acf` or :func:`operational_sign_acf`
    for the recovery side. Fail-closed when fewer than 16 probes fit the
    window.
    """
    s = _check_signs(signs)
    tt = _check_times(times)
    if s.size != tt.size:
        raise ValueError("signs must match the event count")
    tv = _check_taus(taus)
    t0, t1 = float(tt[0]), float(tt[-1])
    tau_max = float(tv[-1])
    if t1 - t0 <= 1.5 * tau_max:
        raise ValueError("horizon must exceed 1.5x the largest lag")
    if probe_spacing is None:
        spacing = (t1 - t0) / s.size / 4.0
    else:
        spacing = _pos_finite(probe_spacing, "probe_spacing")
    probes = np.arange(t0, t1 - tau_max, spacing)
    if probes.size < 16:
        raise ValueError("too few probes fit the window — widen the horizon or spacing")
    base = counting_process(tt, probes)
    s0 = s[np.clip(base - 1, 0, s.size - 1)]
    m0, v0 = float(s0.mean()), float(s0.var())
    if v0 <= 0.0:
        raise ValueError("subordinated sign stream is constant — ACF undefined")
    acf = np.empty(tv.size)
    op_lag = np.empty(tv.size)
    for j, tau in enumerate(tv):
        idx = np.clip(counting_process(tt, probes + tau) - 1, 0, s.size - 1)
        s1 = s[idx]
        m1, v1 = float(s1.mean()), float(s1.var())
        if v1 <= 0.0:
            raise ValueError("subordinated sign stream is constant — ACF undefined")
        acf[j] = float(np.mean((s0 - m0) * (s1 - m1)) / math.sqrt(v0 * v1))
        op_lag[j] = float(np.mean(counting_process(tt, probes + tau) - base))
    return CalendarACF(taus=tv, acf=acf, operational_lags=op_lag, n_probes=int(probes.size))


@dataclass(frozen=True)
class EventAnchoredACF:
    """Event-anchored calendar-lag sign autocorrelation."""

    taus: Array
    acf: Array  # mean eps_i * eps_{j}, j = N(t_i + tau)
    mean_event_lags: Array  # mean realized event lag j - i per tau
    n_pairs: int


def event_anchored_sign_acf(signs: object, times: object, taus: object) -> EventAnchoredACF:
    """ACF of sign pairs anchored at EVENT times with calendar separation tau.

    For each event i, the partner is j = N(t_i + tau) — the first event at
    or after t_i + tau — and the product eps_i eps_j is averaged over all
    pairs fully inside the stream. Because waiting times from an event are
    ordinary (renewal) waits, this estimator IS the origin-anchored mixture

        E[eps_i eps_{N(t_i+tau)}] = sum_k P(N(tau) = k) C_ev(k),  C_ev(0) = 1

    — exactly :func:`apparent_calendar_kernel` of the theoretical C_ev with
    ``zero_value = 1``. Under the fractional (pareto) clock this is the
    observable that carries the paper's mu * gamma rescaling; under
    finite-mean clocks it preserves gamma. Contrast
    :func:`calendar_sign_acf` (uniform probes): infinite-mean waits put most
    probes inside a few giant gaps, inflating the apparent ACF toward a
    plateau — a different, more extreme distortion of the same mechanism.
    """
    s = _check_signs(signs)
    tt = _check_times(times)
    if s.size != tt.size:
        raise ValueError("signs must match the event count")
    tv = _check_taus(taus)
    n = s.size
    idx = np.arange(n)
    acf = np.empty(tv.size)
    mean_lag = np.empty(tv.size)
    n_pairs = 0
    for j_i, tau in enumerate(tv):
        # partner = index of the LAST event <= t_i + tau (tick rule); k = 0
        # means no event in (t_i, t_i + tau] and contributes eps_i^2 = C(0).
        partner = np.searchsorted(tt, tt + tau, side="right") - 1
        valid = partner < n
        pv = partner[valid]
        if pv.size < 16:
            raise ValueError(f"fewer than 16 observable pairs at tau={tau} — shorten the lag range")
        k = pv - idx[valid]
        if bool(np.all(k == 0)):
            raise ValueError(f"no event inside (t_i, t_i + {tau}] for any anchor — clock frozen")
        prod = s[valid] * s[pv]  # k = 0 pairs contribute eps_i^2 = 1
        m = float(s[valid].mean())
        v = float(s[valid].var())
        if v <= 0.0:
            raise ValueError("constant sign stream — ACF undefined")
        acf[j_i] = float(prod.mean() - m * m) / v
        mean_lag[j_i] = float(k.mean())
        n_pairs = max(n_pairs, int(pv.size))
    return EventAnchoredACF(taus=tv, acf=acf, mean_event_lags=mean_lag, n_pairs=n_pairs)


@dataclass(frozen=True)
class OperationalACF:
    """Event-lag-conditioned ACF recovered from calendar-time probe pairs."""

    event_lags: IntArray  # realized event lag k = 0..k_max
    acf: Array  # E[eps(t) eps(t') | N(t') - N(t) = k]; nan where unobserved
    pair_counts: IntArray  # pairs contributing per bin
    n_probes: int


def operational_sign_acf(
    signs: object,
    times: object,
    taus: object,
    k_max: int,
    probe_spacing: float | None = None,
) -> OperationalACF:
    """Recover the event-time sign ACF from calendar-sampled pairs.

    The operational-time correction of the paper: probes t are uniform in
    calendar time, but each pair (t, t + tau) is binned by its REALIZED
    event lag k = N(t + tau) - N(t) — a quantity observable from the tape —
    instead of by wall-clock tau. Under finite-mean clocks the conditioned
    bin means reproduce C_ev(k) directly; under the infinite-mean pareto
    clock the estimator is still the correct operationalization but pair
    clusters inside giant gaps make single-path estimates noisy — pool over
    paths (``pair_counts`` is returned so callers can weight bins), and
    expect honest recovery over the well-populated bins rather than the
    full tail. Fail-closed if no probed pair realizes a nonzero event lag
    (clock frozen on the window).
    """
    s = _check_signs(signs)
    tt = _check_times(times)
    if s.size != tt.size:
        raise ValueError("signs must match the event count")
    tv = _check_taus(taus)
    km = _check_n(k_max, "k_max", 1)
    if km >= s.size:
        raise ValueError("k_max must be below the event count")
    t0, t1 = float(tt[0]), float(tt[-1])
    tau_max = float(tv[-1])
    if t1 - t0 <= 1.5 * tau_max:
        raise ValueError("horizon must exceed 1.5x the largest lag")
    if probe_spacing is None:
        spacing = (t1 - t0) / s.size / 4.0
    else:
        spacing = _pos_finite(probe_spacing, "probe_spacing")
    probes = np.arange(t0, t1 - tau_max, spacing)
    if probes.size < 16:
        raise ValueError("too few probes fit the window — widen the horizon or spacing")
    base = counting_process(tt, probes)
    s0 = s[np.clip(base - 1, 0, s.size - 1)]
    if float(s0.var()) <= 0.0:
        raise ValueError("subordinated sign stream is constant — ACF undefined")
    # Per-bin Pearson: signs are +-1 so each bin needs (n, sum s0, sum s1,
    # sum s0*s1); bin-conditional centering removes the global probe-mean
    # artifact that dominates under clustered clocks.
    cnt = np.zeros(km + 1, dtype=np.intp)
    sum0 = np.zeros(km + 1)
    sum1 = np.zeros(km + 1)
    sum01 = np.zeros(km + 1)
    for tau in tv:
        ahead = counting_process(tt, probes + tau)
        k = ahead - base
        s1 = s[np.clip(ahead - 1, 0, s.size - 1)]
        inside = k <= km  # pairs beyond k_max are dropped, never clipped in
        cnt += np.bincount(k[inside], minlength=km + 1)
        sum0 += np.bincount(k[inside], weights=s0[inside], minlength=km + 1)
        sum1 += np.bincount(k[inside], weights=s1[inside], minlength=km + 1)
        sum01 += np.bincount(k[inside], weights=s0[inside] * s1[inside], minlength=km + 1)
    if int(cnt[1:].sum()) == 0:
        raise ValueError("no probed pair realized a nonzero event lag — clock frozen")
    n = cnt.astype(float)
    denom = np.sqrt(np.maximum((n * n - sum0 * sum0) * (n * n - sum1 * sum1), 0.0))
    acf = np.full(km + 1, np.nan)
    seen = (cnt > 0) & (denom > 0.0)
    acf[seen] = (n[seen] * sum01[seen] - sum0[seen] * sum1[seen]) / denom[seen]
    return OperationalACF(
        event_lags=np.arange(km + 1, dtype=np.intp),
        acf=acf,
        pair_counts=cnt,
        n_probes=int(probes.size),
    )


def calendar_imbalance(signs: object, times: object, bucket_dt: float) -> tuple[Array, Array]:
    """Signed order-flow imbalance per calendar bucket: X_j = sum eps_i.

    The practitioner's calendar-time observable — unlike the tick-rule
    projection it also inherits the clock's own clustering (a busy bucket
    sums more correlated signs), which is exactly the mixed distortion the
    paper describes. Returns (bucket_edges, imbalance).
    """
    s = _check_signs(signs)
    tt = _check_times(times)
    if s.size != tt.size:
        raise ValueError("signs must match the event count")
    dt = _pos_finite(bucket_dt, "bucket_dt")
    edges = np.arange(float(tt[0]), float(tt[-1]) + dt, dt)
    if edges.size < 3:
        raise ValueError("bucket_dt too coarse for the stream horizon")
    counts = np.histogram(tt, bins=edges)[0].astype(float)
    signed = np.histogram(tt, bins=edges, weights=s)[0]
    _ = counts  # explicit: the imbalance uses sign-weighted counts
    return np.asarray(edges), np.asarray(signed, dtype=float)


# ---------------------------------------------------------------------------
# Clock moments + apparent kernels (theory-side and Monte-Carlo)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CountMoments:
    """Moments of the counting process N(tau) across seeded clock paths."""

    taus: Array
    n_mean: Array
    n_var: Array
    fano: Array  # var/mean; =1 Poisson, >1 clustered/overdispersed


def clock_count_moments(
    clock: EventClock,
    taus: object,
    n_paths: int = 64,
    rng: RNG = 0,
) -> CountMoments:
    """Monte-Carlo mean/variance of N(tau) under ``clock`` (seeded)."""
    if not isinstance(clock, EventClock):
        raise ValueError("clock must be an EventClock")
    tv = _check_taus(taus)
    r = _check_n(n_paths, "n_paths", 4)
    g = _resolve_rng(rng)
    counts = np.empty((r, tv.size))
    for i in range(r):
        ev = clock.times_until(float(tv[-1]), g)
        counts[i] = _count_path(ev, tv)
    mean = counts.mean(axis=0)
    var = counts.var(axis=0, ddof=1)
    if bool(np.any(mean <= 0.0)):
        raise ValueError("clock produced no events at some tau — enlarge the window")
    return CountMoments(
        taus=tv,
        n_mean=np.asarray(mean),
        n_var=np.asarray(var),
        fano=np.asarray(var / mean),
    )


def poisson_count_moments(taus: object, rate: float) -> CountMoments:
    """Closed-form N(tau) ~ Poisson(rate * tau): mean = var = rate * tau."""
    tv = _check_taus(taus)
    r = _pos_finite(rate, "rate")
    m = r * tv
    return CountMoments(taus=tv, n_mean=m, n_var=m.copy(), fano=np.ones_like(tv))


@dataclass(frozen=True)
class ApparentKernel:
    """Calendar-time apparent propagator G_cal(tau) = E[G(N(tau))]."""

    taus: Array
    g_apparent: Array
    n_mean: Array


def _g_of_n(kernel: Array, n: IntArray, zero_value: float = 0.0) -> Array:
    """Kernel evaluated at event count n: g(0) = ``zero_value``, g(n > L) = g_L."""
    g = kernel
    idx = np.clip(n - 1, -1, g.size - 1)
    out = np.where(idx < 0, zero_value, g[np.clip(idx, 0, g.size - 1)])
    return np.asarray(out, dtype=float)


def apparent_calendar_kernel(
    kernel: object,
    clock: EventClock,
    taus: object,
    n_paths: int = 64,
    rng: RNG = 0,
    zero_value: float = 0.0,
) -> ApparentKernel:
    """Monte-Carlo apparent kernel: G_cal(tau) = mean_i g(N_i(tau)).

    Event-anchored counting: each path injects at its own event origin and
    N(tau) counts events in (0, tau] — matching how metaorder impact and
    event-anchored pairing are actually measured. The kernel is defined on
    event lags 1..L; ``zero_value`` is the contribution when N(tau) = 0
    (0.0 for impact kernels — no event, no impact; 1.0 when ``kernel`` is an
    autocorrelation, since C(0) = 1). For finite-mean clocks the asymptotic
    decay mirrors the event-time law; for the fractional (pareto) clock it
    picks up the clock's rescaling — the observable distortion is a
    projection effect, not a different mechanism.
    """
    g = _check_kernel(kernel)
    if not isinstance(clock, EventClock):
        raise ValueError("clock must be an EventClock")
    zv = _check_zero_value(zero_value)
    tv = _check_taus(taus)
    r = _check_n(n_paths, "n_paths", 4)
    gen = _resolve_rng(rng)
    vals = np.empty((r, tv.size))
    counts = np.empty((r, tv.size), dtype=np.intp)
    for i in range(r):
        ev = clock.times_until(float(tv[-1]), gen)
        counts[i] = _count_path(ev, tv)
        vals[i] = _g_of_n(g, counts[i], zv)
    if bool(np.any(counts.mean(axis=0) <= 0.0)):
        raise ValueError("clock produced no events at some tau — enlarge the window")
    return ApparentKernel(
        taus=tv,
        g_apparent=np.asarray(vals.mean(axis=0)),
        n_mean=np.asarray(counts.mean(axis=0)),
    )


def poisson_apparent_kernel(
    kernel: object, taus: object, rate: float, zero_value: float = 0.0
) -> Array:
    """Closed-form apparent kernel under a Poisson clock.

    G_cal(tau) = sum_{k>=0} g(k) Pois(k; rate*tau), exact with the clamp
    convention g(n >= L) = g_L: ``zero_value`` * pmf(0) + sum_{k=1}^{L-1}
    g(k) pmf(k) + g_L * P(N >= L - 1). The MC estimator must converge to
    this — the canonical subordination sanity check. Pass
    ``zero_value = 1.0`` when ``kernel`` is an autocorrelation (C(0) = 1).
    """
    g = _check_kernel(kernel)
    tv = _check_taus(taus)
    zv = _check_zero_value(zero_value)
    r = _pos_finite(rate, "rate")
    lam = r * tv
    k = np.arange(g.size)  # k = 0..L-1 with g(0) = zero_value
    pmf = _poisson.pmf(k[None, :], lam[:, None])
    tail = _poisson.sf(g.size - 1, lam)  # P(N >= L), where g clamps to g_L
    gext = np.concatenate(([zv], g[:-1]))  # values at k = 0..L-1
    return np.asarray(pmf @ gext + g[-1] * tail, dtype=float)


@dataclass(frozen=True)
class ImpactProfile:
    """Calendar-time metaorder impact profile under subordination."""

    taus: Array
    impact_mean: Array
    impact_p10: Array
    impact_p90: Array
    n_mean: Array  # mean executed child events per tau


def calendar_impact_profile(
    kernel: object,
    clock: EventClock,
    participation: float,
    taus: object,
    n_paths: int = 64,
    rng: RNG = 0,
) -> ImpactProfile:
    """Expected metaorder impact vs CALENDAR time under the stochastic clock.

    A metaorder executing a fixed ``participation`` fraction of the flow
    completes n_child(tau) = floor(participation * N(tau)) child events by
    calendar time tau, carrying impact cumG(n_child). In operational time
    this is the square-root-like I(n) ~ n^{1-beta}; in calendar time a
    fractional clock rescales the apparent law — E[I] ~ tau^{mu*(1-beta)} —
    and even finite-mean clocks depress it (Jensen: E[sqrt N] < sqrt E[N]).
    """
    g = _check_kernel(kernel)
    if not isinstance(clock, EventClock):
        raise ValueError("clock must be an EventClock")
    f = float(participation)
    if not math.isfinite(f) or f <= 0.0 or f > 1.0:
        raise ValueError(f"participation must lie in (0, 1], got {participation!r}")
    tv = _check_taus(taus)
    r = _check_n(n_paths, "n_paths", 4)
    gen = _resolve_rng(rng)
    cum = np.concatenate(([0.0], np.cumsum(g)))  # cumG(n), n = 0..L
    vals = np.empty((r, tv.size))
    nchild = np.empty((r, tv.size))
    for i in range(r):
        ev = clock.times_until(float(tv[-1]), gen)
        nc = np.floor(f * _count_path(ev, tv)).astype(np.intp)
        nchild[i] = nc
        vals[i] = cum[np.clip(nc, 0, g.size)]
    if bool(np.any(nchild.mean(axis=0) <= 0.0)):
        raise ValueError("no child executions at some tau — raise participation or horizon")
    return ImpactProfile(
        taus=tv,
        impact_mean=np.asarray(vals.mean(axis=0)),
        impact_p10=np.asarray(np.quantile(vals, 0.1, axis=0)),
        impact_p90=np.asarray(np.quantile(vals, 0.9, axis=0)),
        n_mean=np.asarray(nchild.mean(axis=0)),
    )


# ---------------------------------------------------------------------------
# Diagnostics: exponents, stream stats, activity-regime boundaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoglogFit:
    """OLS on (log x, log y): the fitted power ``y ~ x^slope``."""

    slope: float
    intercept: float
    r2: float
    n: int


def fit_loglog_exponent(x: object, y: object) -> LoglogFit:
    """Fit log y = a + b log x; ``slope`` is the power-law exponent b.

    Decaying laws (sign ACF ~ k^{-gamma}, kernel ~ l^{-beta}) return NEGATIVE
    slopes — read gamma = -slope; growing laws (impact ~ n^{1-beta}) return
    positive slopes. Fail-closed on non-positive entries (log-domain), fewer
    than 3 points, or degenerate x (zero log-range — e.g. constant lags).
    """
    xx = _finite_1d(x, "x", min_size=3)
    yy = _finite_1d(y, "y", min_size=3)
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    if bool(np.any(xx <= 0.0)) or bool(np.any(yy <= 0.0)):
        raise ValueError("log-log fit requires strictly positive x and y")
    lx, ly = np.log(xx), np.log(yy)
    if float(lx.max() - lx.min()) <= 0.0:
        raise ValueError("degenerate x: zero log-range")
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0.0 else 0.0
    return LoglogFit(slope=float(slope), intercept=float(intercept), r2=float(r2), n=int(xx.size))


def interarrival_stats(times: object) -> dict[str, float]:
    """Waiting-time summary of an event stream (fail-closed on degenerate)."""
    tt = _check_times(times)
    gaps = np.diff(tt)
    if gaps.size < 2:
        raise ValueError("need at least 3 events (2 interarrivals)")
    sd = float(gaps.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("degenerate event stream: constant interarrival times")
    mean = float(gaps.mean())
    return {
        "n_events": float(tt.size),
        "n_gaps": float(gaps.size),
        "mean_gap": mean,
        "std_gap": sd,
        "cv_gap": float(sd / mean),
        "min_gap": float(gaps.min()),
        "max_gap": float(gaps.max()),
        "horizon": float(tt[-1] - tt[0]),
        "mean_rate": float(1.0 / mean),
    }


@dataclass(frozen=True)
class ActivityBoundaries:
    """Activity-regime boundaries from binned event counts."""

    bucket_edges: Array
    counts: Array
    changepoint_buckets: IntArray
    changepoint_times: Array


def activity_rate_boundaries(
    times: object,
    bucket_dt: float,
    min_size: int = 32,
    penalty: float | None = None,
) -> ActivityBoundaries:
    """Locate activity-regime boundaries by binary segmentation of counts.

    Composes ``models.changepoint.binary_segmentation`` (Scott-Knott mean
    shift) on the per-bucket event count series — the activity rate's regime
    structure. Fail-closed when the binned window cannot contain a split
    (< 2 * min_size buckets) or the count series is degenerate.
    """
    from quant_fund.models.changepoint import (  # noqa: PLC0415  # layer-order: analytics-layer, imported lazily
        binary_segmentation,
    )

    tt = _check_times(times)
    dt = _pos_finite(bucket_dt, "bucket_dt")
    ms = _check_n(min_size, "min_size", 2)
    edges = np.arange(float(tt[0]), float(tt[-1]) + dt, dt)
    counts = np.histogram(tt, bins=edges)[0].astype(float)
    if counts.size < 2 * ms:
        raise ValueError(
            f"binned window has {counts.size} buckets < 2*min_size={2 * ms} — no split can fit"
        )
    if float(counts.std()) <= 0.0:
        raise ValueError("degenerate bucket counts (constant rate) — no regimes to bound")
    cps = binary_segmentation(counts, min_size=ms, penalty=penalty)
    return ActivityBoundaries(
        bucket_edges=np.asarray(edges),
        counts=np.asarray(counts),
        changepoint_buckets=cps,
        changepoint_times=np.asarray(edges[cps], dtype=float),
    )


def hawkes_mean_intensity(mu: float, alpha: float) -> float:
    """Stationary Hawkes intensity mu / (1 - alpha) (branching ratio < 1)."""
    m = _pos_finite(mu, "mu")
    a = float(alpha)
    if not math.isfinite(a) or a < 0.0 or a >= 1.0:
        raise ValueError(f"alpha (branching ratio) must lie in [0, 1), got {alpha!r}")
    return float(m / (1.0 - a))


# ---------------------------------------------------------------------------
# Seeded end-to-end bench (flat dict[str, float], SYNTHETIC labels)
# ---------------------------------------------------------------------------


def bench_event_time_flow(seed: int = 0, n_events: int = 40_000) -> dict[str, float]:
    """Seeded SYNTHETIC end-to-end diagnostic bundle (~seconds).

    Pipeline: LMF sign stream (alpha = 1.5 -> gamma_evt = 0.5) + propagator
    kernel (beta = 0.5 -> square-root operational impact) + three clocks
    (poisson / hawkes / pareto mu = 0.7). Reports: event-time sign exponent,
    kernel L2 recovery error, operational impact exponent, calendar vs
    operational sign-exponent estimates under the fractional clock (the
    mu*gamma distortion and its correction), apparent-kernel MC-vs-closed-
    form agreement under Poisson, and the impact-exponent rescaling.
    All keys are prefixed ``synthetic_`` — correctness evidence only.
    """
    gen = np.random.default_rng(_check_n(seed, "seed", 0))
    n = _check_n(n_events, "n_events", 4096)
    out: dict[str, float] = {}

    alpha_lmf, beta_kernel, tail_mu = 1.5, 0.5, 0.7
    gamma_evt = alpha_lmf - 1.0

    # Event-time sign memory vs the exact covering-model curve (lags kept
    # below the finite-stream block truncation scale n^{1/alpha}).
    lmf = simulate_lmf_signs(n, alpha_lmf, 0.5, gen)
    lags_fit = np.unique(np.geomspace(4, 256, 10).astype(np.intp))
    acf = sign_autocorrelation(lmf.signs, lags_fit)
    theory = lmf_acf_theory(lags_fit, alpha_lmf)
    out["synthetic_sign_acf_l2_vs_theory"] = kernel_rel_l2(
        np.clip(theory, 1e-12, None), np.clip(acf, 1e-12, None)
    )
    fit = fit_loglog_exponent(lags_fit.astype(float), np.clip(acf, 1e-6, None))
    out["synthetic_sign_slope_event"] = fit.slope
    out["synthetic_sign_slope_event_true"] = -gamma_evt

    # Operational-time impact + kernel recovery.
    kernel = propagator_kernel(256, 1.0, beta_kernel)
    impact = metaorder_impact_curve(kernel)
    out["synthetic_impact_slope_operational"] = fit_loglog_exponent(
        np.arange(16, impact.size + 1, dtype=float), impact[15:]
    ).slope
    white = np.where(np.random.default_rng(seed + 1).random(n) < 0.5, 1.0, -1.0)
    path = simulate_propagator_path(white, kernel, 0.0, seed + 2)
    g_hat = estimate_event_kernel(path.prices, white, kernel.size)
    out["synthetic_kernel_l2_rel_error"] = kernel_rel_l2(kernel, g_hat)

    # Fractional-clock subordination: event-anchored apparent sign law.
    pareto = EventClock(kind="pareto", tail_exponent=tail_mu, tail_scale=0.01)
    cev = lmf_acf_theory(np.arange(1, 8193), alpha_lmf)
    taus = np.geomspace(0.5, 500.0, 14)
    mc = apparent_calendar_kernel(cev, pareto, taus, n_paths=96, rng=gen, zero_value=1.0)
    pos = mc.g_apparent > 0.0
    out["synthetic_sign_slope_calendar_pareto"] = fit_loglog_exponent(
        taus[pos], mc.g_apparent[pos]
    ).slope
    out["synthetic_sign_slope_calendar_theory"] = -tail_mu * gamma_evt

    # Event-anchored estimator agreement + operational recovery (sim side).
    accs = np.zeros(taus.size)
    reps = 16
    for _ in range(reps):
        tp = pareto.times(n, gen)
        ea = event_anchored_sign_acf(lmf.signs, tp, taus)
        accs += ea.acf
    accs /= reps
    pos_s = accs > 0.0
    out["synthetic_sign_slope_event_anchored_sim"] = fit_loglog_exponent(
        taus[pos_s], accs[pos_s]
    ).slope
    out["synthetic_event_anchored_max_abs_err"] = float(np.max(np.abs(accs - mc.g_apparent)))

    # Poisson apparent kernel: MC vs closed form (zero_value=1 for an ACF).
    pois = EventClock(kind="poisson", rate=2.0)
    taus_k = np.geomspace(0.25, 40.0, 10)
    mc_p = apparent_calendar_kernel(cev[:1024], pois, taus_k, n_paths=256, rng=gen, zero_value=1.0)
    exact = poisson_apparent_kernel(cev[:1024], taus_k, 2.0, zero_value=1.0)
    denom = np.maximum(np.abs(exact), 1e-12)
    out["synthetic_apparent_kernel_mc_max_rel_err"] = float(
        np.max(np.abs(mc_p.g_apparent - exact) / denom)
    )

    # Impact exponent in calendar time under the fractional clock.
    taus_i = np.geomspace(10.0, 3e4, 12)
    prof = calendar_impact_profile(kernel, pareto, 0.2, taus_i, n_paths=96, rng=gen)
    ok = (prof.impact_mean > 0.0) & (prof.n_mean < kernel.size)
    out["synthetic_impact_slope_calendar_pareto"] = fit_loglog_exponent(
        taus_i[ok], prof.impact_mean[ok]
    ).slope
    out["synthetic_impact_slope_calendar_theory"] = tail_mu * (1.0 - beta_kernel)

    # Hawkes overdispersion check.
    hawk = EventClock(kind="hawkes", hawkes_mu=0.4, hawkes_alpha=0.7, hawkes_beta=1.0)
    cm = clock_count_moments(hawk, np.array([50.0, 200.0, 800.0]), n_paths=64, rng=gen)
    out["synthetic_hawkes_fano_max"] = float(cm.fano.max())
    out["synthetic_seed"] = float(seed)
    return out


__all__ = [
    "CLOCK_KINDS",
    "EVENT_TIME_FLOW_REVISION",
    "ActivityBoundaries",
    "ApparentKernel",
    "CalendarACF",
    "CountMoments",
    "EventAnchoredACF",
    "EventClock",
    "EventStream",
    "ImpactProfile",
    "LMFSigns",
    "LoglogFit",
    "OperationalACF",
    "PropagatorPath",
    "activity_rate_boundaries",
    "apparent_calendar_kernel",
    "bench_event_time_flow",
    "calendar_imbalance",
    "calendar_impact_profile",
    "calendar_sign_acf",
    "clock_count_moments",
    "counting_process",
    "estimate_event_kernel",
    "event_anchored_sign_acf",
    "fit_loglog_exponent",
    "hawkes_mean_intensity",
    "interarrival_stats",
    "kernel_rel_l2",
    "lmf_acf_theory",
    "metaorder_impact_curve",
    "operational_sign_acf",
    "poisson_apparent_kernel",
    "poisson_count_moments",
    "propagator_kernel",
    "sign_autocorrelation",
    "simulate_lmf_signs",
    "simulate_propagator_path",
    "subordinated_signs",
    "zi_lob_event_stream",
]
