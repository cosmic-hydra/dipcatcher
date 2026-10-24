"""Wave-201 adapters: stochastic-control/American canon — psor_american,
crr_tree, kushner_mca, hjb_penalty, dual_american, exercise_boundary —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.crr_tree import bench_crr_tree
from quant_fund.models.dual_american import bench_dual_american
from quant_fund.models.exercise_boundary import bench_exercise_boundary
from quant_fund.models.hjb_penalty import bench_hjb_penalty
from quant_fund.models.kushner_mca import bench_kushner_mca
from quant_fund.models.psor_american import bench_psor_american

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


def bench_dual_american_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "dual_american",
                bench_dual_american(seed=_SEED + 960, n_crr=400, n_tree=80, n_path=80),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dual_american bench failed: {exc}") from exc


def bench_psor_american_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("psor_american", bench_psor_american(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"psor_american bench failed: {exc}") from exc


def bench_hjb_penalty_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hjb_penalty", bench_hjb_penalty(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hjb_penalty bench failed: {exc}") from exc


def bench_crr_tree_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crr_tree", bench_crr_tree(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crr_tree bench failed: {exc}") from exc


def bench_kushner_mca_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("kushner_mca", bench_kushner_mca(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kushner_mca bench failed: {exc}") from exc


def bench_exercise_boundary_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("exercise_boundary", bench_exercise_boundary(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"exercise_boundary bench failed: {exc}") from exc
