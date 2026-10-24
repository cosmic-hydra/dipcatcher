"""Wave-118 adapters: POMDP-solver canon — QMDP, grid value
iteration, PBVI, Perseus, HSVI, and POMCP — each benched on the
SYNTHETIC Tiger POMDP. Adapters flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.grid_pomdp import bench_grid
from quant_fund.models.hsvi import bench_hsvi
from quant_fund.models.pbvi import bench_pbvi
from quant_fund.models.perseus import bench_perseus
from quant_fund.models.pomcp import bench_pomcp
from quant_fund.models.qmdp import bench_qmdp

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim == 0:
            f = float(arr)
            if not np.isfinite(f):
                raise ValueError(f"non-finite {k} in {name}")
            flat[k] = f
        else:
            for i, val in enumerate(arr.ravel()):
                f = float(val)
                if not np.isfinite(f):
                    raise ValueError(f"non-finite {k}[{i}] in {name}")
                flat[f"{k}_{i}"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_qmdp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("qmdp", bench_qmdp(seed=_SEED + 696)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"qmdp bench failed: {exc}") from exc


def bench_grid_pomdp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("grid_pomdp", bench_grid(seed=_SEED + 697)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"grid_pomdp bench failed: {exc}") from exc


def bench_pbvi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pbvi", bench_pbvi(seed=_SEED + 698)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pbvi bench failed: {exc}") from exc


def bench_perseus_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("perseus", bench_perseus(seed=_SEED + 699)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"perseus bench failed: {exc}") from exc


def bench_hsvi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hsvi", bench_hsvi(seed=_SEED + 700)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hsvi bench failed: {exc}") from exc


def bench_pomcp_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "pomcp",
                bench_pomcp(seed=_SEED + 701, n_episodes=4, n_sims=80, n_particles=80),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pomcp bench failed: {exc}") from exc
