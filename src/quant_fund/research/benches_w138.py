"""Wave-126 adapters: exec-summary memory + world-model canon — reformer_lsh,
memorizing_transformer, ntm_memory, dnc_memory, rssm_world, mpc_planning —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.dnc_memory import bench_dnc_memory
from quant_fund.models.memorizing_transformer import bench_memorizing_transformer
from quant_fund.models.mpc_planning import bench_mpc_planning
from quant_fund.models.ntm_memory import bench_ntm_memory
from quant_fund.models.reformer_lsh import bench_reformer_lsh
from quant_fund.models.rssm_world import bench_rssm_world

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
                flat[f"{k}[{i}]"] = f
    return flat


def _floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_reformer_lsh_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("reformer_lsh", bench_reformer_lsh(seed=_SEED + 816, iters=200))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reformer_lsh bench failed: {exc}") from exc


def bench_memorizing_transformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "memorizing_transformer",
                bench_memorizing_transformer(seed=_SEED + 817, iters=200),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"memorizing_transformer bench failed: {exc}") from exc


def bench_ntm_memory_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ntm_memory", bench_ntm_memory(seed=_SEED + 818, iters=160)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ntm_memory bench failed: {exc}") from exc


def bench_dnc_memory_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dnc_memory", bench_dnc_memory(seed=_SEED + 819, iters=160)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dnc_memory bench failed: {exc}") from exc


def bench_rssm_world_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rssm_world", bench_rssm_world(seed=_SEED + 820)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rssm_world bench failed: {exc}") from exc


def bench_mpc_planning_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("mpc_planning", bench_mpc_planning(seed=_SEED + 821)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mpc_planning bench failed: {exc}") from exc
