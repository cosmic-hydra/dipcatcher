"""Wave-116 adapters: game-theory canon — counterfactual regret
minimization (Kuhn poker), Lemke–Howson bimatrix equilibrium,
replicator dynamics, Wardrop user equilibrium, VCG mechanisms, and
Nash bargaining — each benched on SYNTHETIC games. Adapters flatten
to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cfr import bench_cfr
from quant_fund.models.lemke_howson import bench_lemke_howson
from quant_fund.models.nash_bargain import bench_nash_bargain
from quant_fund.models.replicator import bench_replicator
from quant_fund.models.vcg import bench_vcg
from quant_fund.models.wardrop import bench_wardrop

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


def bench_cfr_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cfr", bench_cfr(seed=_SEED + 684, n_iter=12_000)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cfr bench failed: {exc}") from exc


def bench_lemke_howson_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lemke_howson", bench_lemke_howson(seed=_SEED + 685)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lemke_howson bench failed: {exc}") from exc


def bench_replicator_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("replicator", bench_replicator(seed=_SEED + 686)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"replicator bench failed: {exc}") from exc


def bench_wardrop_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("wardrop", bench_wardrop(seed=_SEED + 687)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"wardrop bench failed: {exc}") from exc


def bench_vcg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vcg", bench_vcg(seed=_SEED + 688)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vcg bench failed: {exc}") from exc


def bench_nash_bargain_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nash_bargain", bench_nash_bargain(seed=_SEED + 689)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nash_bargain bench failed: {exc}") from exc
