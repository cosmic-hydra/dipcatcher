"""Benchmark batteries for SOTA canon wave 28 (see waves 11-27 for the pattern).

Wave 28 lands OPTIONAL research families, each pinned to the work shipped
in its lane module (see the module docstrings for full citations):

- ``enkf``: sequential ensemble data assimilation — stochastic EnKF
  (Evensen 2003), deterministic EAKF (Anderson 2001), LETKF ensemble
  transform (Ott et al. 2004), inflation + Gaspari-Cohn taper, Lorenz-96
  twin-experiment RMSE-vs-climatology and spread/skill telemetry.
- ``causal_discovery``: structure learning — stable-PC skeleton with
  OLS-residual partial correlations + Fisher-z, v-structures, Meek
  rules, CPDAG encoding; GES forward-edge BIC search; pairwise LiNGAM.
- ``knockoffs``: Barber-Candès model-X knockoff filter — equi/SDP
  Gaussian knockoffs, signed-max OMP and ridge importance statistics,
  knockoff+ threshold, stability selection, honest FDR-breach telemetry.
- ``callaway_did``: Callaway-Sant'Anna group-time ATT — staggered-
  adoption ATT(g,t) with never/not-yet controls, aggregations (simple,
  calendar, event, group), cluster bootstrap SEs, pretrend Wald test.
- ``surrogate_nonlinear``: surrogate-data nonlinearity tests — AAFT/IAAFT
  surrogates, BDS correlation-integral test, Keenan and Tsay
  conditional-mean tests, size/power telemetry on AR/GARCH/tent/bilinear.
- ``sindy``: sparse identification of nonlinear dynamics — monomial
  libraries, STLSQ hard-threshold regression, Lorenz/Van-der-Pol
  recovery, trajectory fidelity, sparse-model selection.

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-27 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261231  # wave-28 stamp seed

# --- per-family seeds -------------------------------------------------------
_ENKF_SEED = _SEED + 151
_CD_SEED = _SEED + 152
_KNOCK_SEED = _SEED + 153
_CSDID_SEED = _SEED + 154
_SURR_SEED = _SEED + 155
_SINDY_SEED = _SEED + 156


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(
    raw: dict[str, float] | dict[str, object],
) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_enkf() -> dict[str, float]:
    """Ensemble Kalman data assimilation bench (wave 28).

    EnKF/EAKF/LETKF analysis RMSE vs climatology on a Lorenz-96 twin
    experiment, spread/skill ratio, GC-taper monotonicity, determinism.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.enkf import bench_enkf as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ENKF_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_causal_discovery() -> dict[str, float]:
    """Causal structure-discovery bench (wave 28).

    Stable-PC skeleton edge precision/recall, v-structure orientation,
    GES DAG recovery, LiNGAM pairwise direction accuracy on synthetic
    DAGs. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.causal_discovery import (
            bench_causal_discovery as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CD_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_knockoffs() -> dict[str, float]:
    """Knockoff FDR-filter bench (wave 28).

    Empirical FDR/power of OMP vs ridge statistics, SDP-vs-equi knockoff
    comparison, exchangeability errors, null rejection rate, honest
    OMP sign-flip breach flag. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.knockoffs import bench_knockoffs as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_KNOCK_SEED, n_rep=8, null_rep=5)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_callaway_did() -> dict[str, float]:
    """Callaway-Sant'Anna staggered-DiD bench (wave 28).

    Simple/event ATT bias on staggered panels, pretrend Wald p-values
    (clean vs planted pretrend), per-cell CI coverage, control-choice
    sensitivity, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.callaway_did import bench_callaway_did as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_CSDID_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_surrogate_nonlinear() -> dict[str, float]:
    """Surrogate-data nonlinearity bench (wave 28).

    BDS size/power, Keenan/Tsay power on GARCH vs bilinear targets,
    surrogate-test p on tent vs AR, AAFT spectral fidelity, determinism.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.surrogate_nonlinear import (
            bench_surrogate_nonlinear as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SURR_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_sindy() -> dict[str, float]:
    """SINDy sparse-identification bench (wave 28).

    Coefficient relative error on Lorenz/Van-der-Pol, trajectory
    fidelity, sparsity recovery, noisy-library Jaccard, determinism.
    Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.sindy import bench_sindy as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_SINDY_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
