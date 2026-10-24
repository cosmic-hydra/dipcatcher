"""Wave-126 adapters: exec-summary DL SOTA-4 — itransformer,
tcn_forecaster, ft_transformer, nbeats_deep, mambats, crossformer —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.crossformer import bench_crossformer
from quant_fund.models.ft_transformer import bench_ft_transformer
from quant_fund.models.itransformer import bench_itransformer
from quant_fund.models.mambats import bench_mambats
from quant_fund.models.nbeats_deep import bench_nbeats_deep
from quant_fund.models.tcn_forecaster import bench_tcn_forecaster

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


def bench_itransformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "itransformer",
                bench_itransformer(seed=_SEED + 762, n=160, iters=280),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"itransformer bench failed: {exc}") from exc


def bench_tcn_forecaster_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "tcn_forecaster",
                bench_tcn_forecaster(seed=_SEED + 763, n=160, iters=140),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"tcn_forecaster bench failed: {exc}") from exc


def bench_ft_transformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "ft_transformer",
                bench_ft_transformer(seed=_SEED + 764, n=200, iters=360),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ft_transformer bench failed: {exc}") from exc


def bench_nbeats_deep_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "nbeats_deep",
                bench_nbeats_deep(seed=_SEED + 765, n=180, iters=180),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nbeats_deep bench failed: {exc}") from exc


def bench_mambats_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "mambats",
                bench_mambats(seed=_SEED + 766, n=160, iters=160),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mambats bench failed: {exc}") from exc


def bench_crossformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "crossformer",
                bench_crossformer(seed=_SEED + 767, n=160, iters=160),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crossformer bench failed: {exc}") from exc
