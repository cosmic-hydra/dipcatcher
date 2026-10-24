"""Wave-83 optional scorecard families — Lin (1989)
concordance correlation + Bland-Altman (1986) limits of
agreement, Belsley-Kuh-Welsch (1980) OLS influence
diagnostics (leverage, studentized residuals, Cook's D,
DFBETAS/DFFITS/COVRATIO via QR rank-1 updates),
Lan-DeMets (1983) alpha-spending group-sequential
boundaries (O'Brien-Fleming/Pocock) with conditional
power, Torgerson (1958) classical MDS + SMACOF
(Borg-Groenen) metric scaling, Benzécri (1973)
correspondence analysis, and Mardia/Fisher-Lee/
Jammalamadaka-Sarma circular-circular correlation.
Emitted only when the corresponding module's `bench_*` self-check
completes on its SYNTHETIC fixture.
"""

from __future__ import annotations

import math

_LIN_CCC_SEED = 20261231 + 486
_INFLUENCE_SEED = 20261231 + 487
_GROUP_SEQUENTIAL_SEED = 20261231 + 488
_MDS_SEED = 20261231 + 489
_CORRESPONDENCE_ANALYSIS_SEED = 20261231 + 490
_CIRCULAR_CORRELATION_SEED = 20261231 + 491

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


def bench_lin_ccc() -> dict[str, float]:
    try:
        from quant_fund.models.lin_ccc_blandaltman import (
            bench_lin_ccc as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_LIN_CCC_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_influence() -> dict[str, float]:
    try:
        from quant_fund.models.influence_diagnostics import (
            bench_influence as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_INFLUENCE_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_group_sequential() -> dict[str, float]:
    try:
        from quant_fund.models.group_sequential import (
            bench_group_sequential as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_GROUP_SEQUENTIAL_SEED, n_sim=8_000)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_mds() -> dict[str, float]:
    try:
        from quant_fund.models.mds import bench_mds as _core
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_MDS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_correspondence_analysis() -> dict[str, float]:
    try:
        from quant_fund.models.correspondence_analysis import (
            bench_ca as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CORRESPONDENCE_ANALYSIS_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)


def bench_circular_correlation() -> dict[str, float]:
    try:
        from quant_fund.models.circular_correlation import (
            bench_circular_correlation as _core,
        )
    except ImportError:  # pragma: no cover - optional
        return {}
    try:
        out = _core(seed=_CIRCULAR_CORRELATION_SEED)
    except (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError):
        return {}
    return _isinstance_floats(out)
