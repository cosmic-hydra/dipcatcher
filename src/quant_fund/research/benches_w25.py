"""Benchmark batteries for SOTA canon wave 25 (see waves 11-24 for the pattern).

Wave 25 lands OPTIONAL research families, each pinned to the paper shipped
in its lane module (see the module docstrings for full citations; citation
ids verified against arXiv before implementation):

- ``tda_persistence``: topological data analysis for regime detection —
  Takens embedding, Vietoris-Rips persistence (0-dim union-find + 1-dim
  boundary reduction), persistence landscapes + bottleneck distance
  (Chazal-Michel TDA, Bubenik landscapes — see module docstring).
- ``fractional_ou``: fractional Ornstein-Uhlenbeck — exact circulant
  spectral simulation, Whittle Gaussian MLE on the fOU spectrum,
  theoretical autocovariance (see module docstring).
- ``fourier_hermite``: Fourier-Hermite density + pricing series —
  arbitrary-order Gram-Charlier-style expansion, density repair,
  analytic option moments (see module docstring).
- ``kernel_changepoint``: kernel-MMD change-point detection — unbiased
  MMD scan statistic, permutation-calibrated thresholds, top-down
  binary segmentation (Flynn-Yoo KCUSUM; see module docstring).
- ``kinetic_ising``: kinetic Ising co-movement — parallel-update
  dynamics, nMF/TAP/pseudo-likelihood coupling inversion, Boltzmann
  enumeration checks (Bury kinetic-Ising markets; see docstring).
- ``heterogeneous_abm``: Brock-Hommes heterogeneous-agent market —
  discrete-choice strategy switching, bounded market map, stylized-fact
  and bifurcation diagnostics (Brock-Hommes HAM; see docstring).

Honesty contract: all blobs are SYNTHETIC correctness/telemetry checks, never
market evidence. No Sharpe/Sortino/Calmar/PnL/NAV tokens appear in emitted
keys; every bench returns ``{}`` (fails soft) if any emitted value is
non-finite — the ``_finite_blob`` gate, waves 15-24 precedent.
"""

from __future__ import annotations

import math

_SEED = 20261229  # wave-25 stamp seed

# --- per-family seeds -------------------------------------------------------
_TDA_SEED = _SEED + 121
_FOU_SEED = _SEED + 122
_HERMITE_SEED = _SEED + 123
_KERNEL_CP_SEED = _SEED + 124
_ISING_SEED = _SEED + 125
_ABM_SEED = _SEED + 126


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


_FORBIDDEN_TOKENS = frozenset({"sharpe", "sortino", "calmar", "pnl", "nav"})


def _isinstance_floats(raw: dict[str, float] | dict[str, object]) -> dict[str, float]:
    """Float-coerce a lane blob, dropping str stamps, runtime telemetry,
    and any key carrying a forbidden headline metric token."""
    return {
        k: float(v)
        for k, v in raw.items()
        if isinstance(v, (int, float)) and _FORBIDDEN_TOKENS.isdisjoint(k.lower().split("_"))
    }


def bench_tda_persistence() -> dict[str, float]:
    """TDA persistence regime-detection bench (wave 25).

    Landscape/bottleneck separation between regimes, detection AUC on
    drifting synthetic series. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.tda_persistence import (
            bench_tda_persistence as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_TDA_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fractional_ou() -> dict[str, float]:
    """Fractional OU bench (wave 25).

    Exact spectral simulation, joint Whittle MLE recovery of (H, a, sigma),
    autocovariance inversion checks. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.fractional_ou import bench_fractional_ou as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_FOU_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_fourier_hermite() -> dict[str, float]:
    """Fourier-Hermite density/pricing bench (wave 25).

    Coefficient recovery, repaired-density mass, analytic option moments
    vs mixture truth. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.fourier_hermite import (
            bench_fourier_hermite as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_HERMITE_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_kernel_changepoint() -> dict[str, float]:
    """Kernel-MMD change-point bench (wave 25).

    Size under the null, detection rate + delay vs shift magnitude,
    MMD-vs-CUSUM edge on variance shifts. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.metrics.kernel_changepoint import (
            bench_kernel_changepoint as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(
            _isinstance_floats(_core(seed=_KERNEL_CP_SEED, trials=4, n_perm=24, reps=3))
        )
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_kinetic_ising() -> dict[str, float]:
    """Kinetic Ising co-movement bench (wave 25).

    nMF/TAP/PL coupling recovery vs synthetic truth, delayed-correlation
    recovery, small-system enumeration gap. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.kinetic_ising import bench_kinetic_ising as _core
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ISING_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}


def bench_heterogeneous_abm() -> dict[str, float]:
    """Brock-Hommes HAM bench (wave 25).

    Skeleton convergence, sorting sharpens with beta, stylized facts on
    synthetic returns, determinism. Soft ``{}`` while absent.
    """
    try:
        from quant_fund.models.heterogeneous_abm import (
            bench_heterogeneous_abm as _core,
        )
    except ImportError:
        return {}
    try:
        return _finite_blob(_isinstance_floats(_core(seed=_ABM_SEED)))
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
