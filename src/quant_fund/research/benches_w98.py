"""Wave-98 adapters: tabular RL + probabilistic-graphical +
shallow-ML canon — value/policy iteration MDP solvers, TD(0)
+ SARSA + Q-learning, sum-product/loopy belief propagation,
extreme learning machines, nearest (shrunken) centroids, and
the cross-entropy method.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.belief_propagation import bench_belief_propagation
from quant_fund.models.cross_entropy_method import bench_cross_entropy_method
from quant_fund.models.extreme_learning import bench_extreme_learning
from quant_fund.models.mdp_solvers import bench_mdp_solvers
from quant_fund.models.nearest_centroid import bench_nearest_centroid
from quant_fund.models.td_learning import bench_td_learning

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


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_mdp_solvers_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("mdp_solvers", bench_mdp_solvers(seed=_SEED + 576)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mdp_solvers bench failed: {exc}") from exc


def bench_td_learning_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("td_learning", bench_td_learning(seed=_SEED + 577, episodes=1500))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"td_learning bench failed: {exc}") from exc


def bench_belief_propagation_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("belief_propagation", bench_belief_propagation(seed=_SEED + 578))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"belief_propagation bench failed: {exc}") from exc


def bench_extreme_learning_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("extreme_learning", bench_extreme_learning(seed=_SEED + 579))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"extreme_learning bench failed: {exc}") from exc


def bench_nearest_centroid_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("nearest_centroid", bench_nearest_centroid(seed=_SEED + 580))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nearest_centroid bench failed: {exc}") from exc


def bench_cross_entropy_method_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "cross_entropy_method",
                bench_cross_entropy_method(seed=_SEED + 581),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cross_entropy_method bench failed: {exc}") from exc
