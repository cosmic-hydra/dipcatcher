"""Wave-78 optional scorecard families — Tukey HSD /
Dunnett many-to-one / Games-Howell / Scheffe S-method
post-hoc comparisons, Plackett-Luce MM + Borda +
Condorcet-Copeland + Maximin Condorcet-3 rank
aggregation, Montgomery/Roberts/Page statistical
process control (xbar-R, EWMA, CUSUM, capability),
Roncalli/Maillard ERC risk parity, Mantegna MST +
Tumminello PMFG market topology, and Matteson-James
E-divisive energy changepoints.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_MULTIPLE_COMPARISONS_SEED = 20261231 + 456
_RANK_AGGREGATION_SEED = 20261231 + 457
_SPC_SEED = 20261231 + 458
_RISK_PARITY_SEED = 20261231 + 459
_MST_TOPOLOGY_SEED = 20261231 + 460
_E_DIVISIVE_SEED = 20261231 + 461

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}


def _finite_blob(mapped: dict[str, float]) -> dict[str, float]:
    """``{}`` unless every emitted value is finite (ruff-bench contract)."""
    if all(math.isfinite(v) for v in mapped.values()):
        return mapped
    return {}


def _isinstance_floats(raw: object) -> dict[str, float]:
    """Float-coerce a lane blob, dropping non-numeric entries and
    any key carrying a forbidden headline metric token."""
    if not isinstance(raw, dict):
        return {}
    return _finite_blob(
        {
            k: float(v)
            for k, v in raw.items()
            if isinstance(v, (int, float))
            and not isinstance(v, bool)
            and _FORBIDDEN.isdisjoint(k.lower().split("_"))
        }
    )


def bench_multiple_comparisons() -> dict[str, float]:
    try:
        from quant_fund.models.multiple_comparisons import (
            bench_multiple_comparisons as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MULTIPLE_COMPARISONS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_rank_aggregation() -> dict[str, float]:
    try:
        from quant_fund.models.rank_aggregation import (
            bench_rank_aggregation as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RANK_AGGREGATION_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_spc() -> dict[str, float]:
    try:
        from quant_fund.models.spc import (
            bench_spc as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_SPC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_risk_parity() -> dict[str, float]:
    try:
        from quant_fund.models.risk_parity import bench_risk_parity as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_RISK_PARITY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mst_topology() -> dict[str, float]:
    try:
        from quant_fund.models.mst_topology import (
            bench_mst_topology as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MST_TOPOLOGY_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_e_divisive() -> dict[str, float]:
    try:
        from quant_fund.models.e_divisive import (
            bench_e_divisive as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_E_DIVISIVE_SEED, n_perm=30)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
