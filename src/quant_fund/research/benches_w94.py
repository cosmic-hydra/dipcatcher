"""Wave-94 adapters: classic supervised/unsupervised ML —
Pegasos + kernel SVMs, Fisher LDA/QDA, coordinate-descent
elastic-net paths, KRR/RFF/Nyström kernel methods, collapsed
Gibbs LDA topics, online convex learners.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.coordinate_descent_enet import (
    bench_coordinate_descent_enet,
)
from quant_fund.models.discriminant_analysis import bench_discriminant
from quant_fund.models.kernel_methods import bench_kernel_methods
from quant_fund.models.lda_topics import bench_lda_topics
from quant_fund.models.online_convex import bench_online_convex
from quant_fund.models.svm_classifiers import bench_svm

_FORBIDDEN = {"sharpe", "sortino", "calmar", "pnl", "nav"}
_SEED = 20261231

_BENCH_EXC = (ValueError, RuntimeError, FloatingPointError, KeyError, TypeError)


def _finite_blob(name: str, out: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for k, v in out.items():
        if any(bad in k.lower() for bad in _FORBIDDEN):
            raise ValueError(f"forbidden metric key {k} in {name}")
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"non-finite {k} in {name}")
        flat[k] = float(arr.ravel()[0])
    if not flat:
        raise ValueError(f"{name} returned no metrics")
    return flat


def _isinstance_floats(out: dict[str, float]) -> dict[str, float]:
    return {k: float(v) for k, v in out.items()}


def bench_svm_classifiers() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("svm_classifiers", bench_svm(seed=_SEED + 552)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"svm_classifiers bench failed: {exc}") from exc


def bench_discriminant_analysis() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("discriminant_analysis", bench_discriminant(seed=_SEED + 553))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"discriminant_analysis bench failed: {exc}") from exc


def bench_coordinate_descent_enet_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("coordinate_descent_enet", bench_coordinate_descent_enet(seed=_SEED + 554))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"coordinate_descent_enet bench failed: {exc}") from exc


def bench_kernel_methods_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("kernel_methods", bench_kernel_methods(seed=_SEED + 555))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kernel_methods bench failed: {exc}") from exc


def bench_lda_topics_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("lda_topics", bench_lda_topics(seed=_SEED + 556, n_docs=40, it=80))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"lda_topics bench failed: {exc}") from exc


def bench_online_convex_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("online_convex", bench_online_convex(seed=_SEED + 557))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"online_convex bench failed: {exc}") from exc
