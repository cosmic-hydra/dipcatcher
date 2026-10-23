"""Benchmark batteries for SOTA canon wave 20 (see waves 11-19 for the pattern).

Wave 20 lands six OPTIONAL research families, each pinned to the paper
shipped in its lane module (see the module docstrings for full citations;
all six specs fetched and verified against arXiv before implementation):

- ``varswap_stopping``: closed-form optimal entry/exit for a perpetual,
  continuously settled variance swap (Maeda 2026, arXiv:2609.19102) --
  accrued-variance separation, a single forcing term whose sign fixes the
  exercise-region geometry, and confluent-hypergeometric smooth-pasting
  free boundaries under the physical measure. Pure numpy/scipy.
- ``hidden_markov_equilibrium``: equilibrium prices under hidden Markov
  fundamentals (Pages, Possamai & Rodriguez Polo 2026, arXiv:2609.21684)
  -- two-state Wonham belief filter, belief-Markovian pricing equation,
  belief-dependent stock volatility, European option PDE, and the
  leading short-maturity risk-neutral skewness expansion. Pure numpy/scipy.
- ``arl_mm``: adversarial-RL market making with Hawkes arrivals and trade
  price impact (arXiv:2609.22785) -- zero-sum maker-vs-environment game,
  LSTM policies, left-tail robustness protocol. TORCH-GATED.
- ``liquidity_tail_lob``: liquidity tail risk and price discovery in a
  sequential LOB (arXiv:2607.01198) -- nonlinear fixed point for the
  marginal-cost schedule inside a tail-controlled class under heavy-tailed
  uninformed flow, posterior consistency, tail asymptotics, crossover and
  spread-persistence diagnostics. Pure numpy/scipy.
- ``gaussian_normalized_coords``: Gaussian normalized coordinates and
  risk-neutral CDF deformations (Sun 2026, arXiv:2609.14212) -- the
  normalized-coordinate transform, CDF-deformation lift to implied-vol
  space, normalized-space arbitrage checks, and a minimal-L2 repair
  projection. Pure numpy/scipy.

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite -- ``_finite_blob`` gate, waves 15-19 precedent.

SHRUNK Monte-Carlo budgets (relative to each lane's own test suite):
module-default bench fixtures; lane modules report their own budgets. The
battery is designed to fit inside the ~60 s envelope with the torch family
the dominant term; torch-gated and not-yet-merged families return ``{}``
cleanly -- they are registered only once their branch lands (the
executed-flag invariant in ``test_research_recovers_synthetic_oracle``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

_SEED = 20261026  # wave-20 stamp seed

# --- varswap_stopping: perpetual variance-swap optimal stopping --------------
_VS_SEED = _SEED + 80

# --- hidden_markov_equilibrium: belief-Markovian pricing bench ---------------
_HME_SEED = _SEED + 81

# --- arl_mm: adversarial-RL market making bench (torch-gated) ----------------
_ARL_SEED = _SEED + 83

# --- liquidity_tail_lob: heavy-tailed sequential-LOB equilibrium -------------
_LTL_SEED = _SEED + 84

# --- gaussian_normalized_coords: normalized-coord / CDF-deformation ----------
_GNC_SEED = _SEED + 85


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: Mapping[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps and runtime telemetry."""
    return {
        k: float(v)
        for k, v in raw.items()
        if (
            isinstance(v, (int, float))
            and not isinstance(v, bool)
            and k not in {"runtime_seconds", "live_pnl_claim"}
        )
    }


def bench_varswap_stopping() -> dict[str, float]:
    """Perpetual variance-swap optimal stopping diagnostics (wave 20).

    Thin adapter over the lane module's own ``bench_varswap_stopping``:
    smooth-pasting residuals, forcing-sign -> exercise-region geometry,
    short-side entry interval, upper-tail trigger depth. Returns ``{}``
    while the lane module is absent or on fail-closed rejection.
    """
    try:
        from quant_fund.execution.varswap_stopping import (
            bench_varswap_stopping as _vs_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_vs_core_bench(seed=_VS_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_hidden_markov_equilibrium() -> dict[str, float]:
    """Hidden-Markov equilibrium pricing diagnostics (wave 20).

    Thin adapter over the lane module's own
    ``bench_hidden_markov_equilibrium``: pricing-ODE residuals, factor
    flatness, belief-dependent vol shape, option-PDE solve, and the
    short-maturity skewness expansion vs MC. Returns ``{}`` while the lane
    module is absent or on fail-closed rejection.
    """
    try:
        from quant_fund.models.hidden_markov_equilibrium import (
            bench_hidden_markov_equilibrium as _hme_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_hme_core_bench(seed=_HME_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_arl_mm() -> dict[str, float]:
    """Adversarial-RL market making with Hawkes flow + impact (wave 20).

    Thin adapter over the lane module's own ``arl_mm_bench``: adversary
    effectiveness, left-tail improvement of the ARL maker vs the
    non-adversarial baseline, no-directional-bias check, Hawkes cluster
    statistics. Returns ``{}`` while the lane module is absent, when the
    ``nn`` extra is missing, or on fail-closed rejection.
    """
    try:
        from quant_fund.execution.arl_mm import arl_mm_bench as _arl_core_bench
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_arl_core_bench(seed=_ARL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_liquidity_tail_lob() -> dict[str, float]:
    """Heavy-tailed sequential-LOB equilibrium diagnostics (wave 20).

    Thin adapter over the lane module's own ``liquidity_tail_bench``:
    marginal-cost fixed-point residual inside the tail-controlled class,
    crossover-depth shift vs Gaussian, informed-demand dominance size,
    spread persistence, posterior consistency. Returns ``{}`` while the
    lane module is absent or on fail-closed rejection.
    """
    try:
        from quant_fund.microstructure.liquidity_tail_lob import (
            liquidity_tail_bench as _ltl_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_ltl_core_bench(seed=_LTL_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_gaussian_normalized_coords() -> dict[str, float]:
    """Gaussian normalized coordinates / CDF-deformation bench (wave 20).

    Thin adapter over the lane module's own
    ``bench_gaussian_normalized_coords``: round-trip error, normalized-space
    arbitrage-violation detection, admissibility of deformations, repair
    projection quality. Returns ``{}`` while the lane module is absent or
    on fail-closed rejection.
    """
    try:
        from quant_fund.models.gaussian_normalized_coords import (
            bench_gaussian_normalized_coords as _gnc_core_bench,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_gnc_core_bench(seed=_GNC_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
