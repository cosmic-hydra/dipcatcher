"""Wave-126 adapters: exec-summary differentiable-optimization canon — optnet_qp,
cvxpy_layer, input_convex, deep_declarative, spd_net, diff_mpc —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cvxpy_layer import bench_cvxpy_layer
from quant_fund.models.deep_declarative import bench_deep_declarative
from quant_fund.models.diff_mpc import bench_diff_mpc
from quant_fund.models.input_convex import bench_input_convex
from quant_fund.models.optnet_qp import bench_optnet_qp
from quant_fund.models.spd_net import bench_spd_net

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


def bench_optnet_qp_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("optnet_qp", bench_optnet_qp(seed=_SEED + 792)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"optnet_qp bench failed: {exc}") from exc


def bench_cvxpy_layer_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("cvxpy_layer", bench_cvxpy_layer(seed=_SEED + 793)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cvxpy_layer bench failed: {exc}") from exc


def bench_input_convex_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "input_convex",
                bench_input_convex(seed=_SEED + 794, n_train=800, n_test=200, iters=400),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"input_convex bench failed: {exc}") from exc


def bench_deep_declarative_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("deep_declarative", bench_deep_declarative(seed=_SEED + 795)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"deep_declarative bench failed: {exc}") from exc


def bench_spd_net_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("spd_net", bench_spd_net(seed=_SEED + 796)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"spd_net bench failed: {exc}") from exc


def bench_diff_mpc_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("diff_mpc", bench_diff_mpc(seed=_SEED + 797, iters=80, t_final=16))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"diff_mpc bench failed: {exc}") from exc
