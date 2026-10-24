"""Wave-126 adapters: exec-summary neural-process + amortized-UQ canon — neural_process,
attentive_np, deep_kernel_gp, convnp, meta_uq, llaplace_gp —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.attentive_np import bench_attentive_np
from quant_fund.models.convnp import bench_convnp
from quant_fund.models.deep_kernel_gp import bench_deep_kernel_gp
from quant_fund.models.llaplace_gp import bench_llaplace_gp
from quant_fund.models.meta_uq import bench_meta_uq
from quant_fund.models.neural_process import bench_neural_process

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


def bench_neural_process_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "neural_process",
                bench_neural_process(seed=_SEED + 786, n_train=160, iters=220),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"neural_process bench failed: {exc}") from exc


def bench_attentive_np_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "attentive_np", bench_attentive_np(seed=_SEED + 787, n_train=160, iters=250)
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"attentive_np bench failed: {exc}") from exc


def bench_deep_kernel_gp_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "deep_kernel_gp",
                bench_deep_kernel_gp(seed=_SEED + 788, n_train=120, iters=180),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_kernel_gp bench failed: {exc}") from exc


def bench_convnp_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("convnp", bench_convnp(seed=_SEED + 789, n_train=160, iters=220))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"convnp bench failed: {exc}") from exc


def bench_meta_uq_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("meta_uq", bench_meta_uq(seed=_SEED + 790)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"meta_uq bench failed: {exc}") from exc


def bench_llaplace_gp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("llaplace_gp", bench_llaplace_gp(seed=_SEED + 791)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"llaplace_gp bench failed: {exc}") from exc
