"""Wave-126 adapters: exec-summary neural-operator canon — fno_1d,
deeponet, lowrank_op, pino_residual, gno_lite, cno_lite —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cno_lite import bench_cno_lite
from quant_fund.models.deeponet import bench_deeponet
from quant_fund.models.fno_1d import bench_fno_1d
from quant_fund.models.gno_lite import bench_gno_lite
from quant_fund.models.lowrank_op import bench_lowrank_op
from quant_fund.models.pino_residual import bench_pino_residual

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


def bench_fno_1d_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("fno_1d", bench_fno_1d(seed=_SEED + 954)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"fno_1d bench failed: {exc}") from exc


def bench_deeponet_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deeponet", bench_deeponet(seed=_SEED + 955)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deeponet bench failed: {exc}") from exc


def bench_lowrank_op_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("lowrank_op", bench_lowrank_op(seed=_SEED + 956)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lowrank_op bench failed: {exc}") from exc


def bench_pino_residual_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("pino_residual", bench_pino_residual(seed=_SEED + 957)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"pino_residual bench failed: {exc}") from exc


def bench_gno_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("gno_lite", bench_gno_lite(seed=_SEED + 958, iters=40)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gno_lite bench failed: {exc}") from exc


def bench_cno_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cno_lite", bench_cno_lite(seed=_SEED + 959)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cno_lite bench failed: {exc}") from exc
