"""Wave-126 adapters: exec-summary amortized-inference canon — advi_bbvi,
iwae_bound, nf_vi, sparse_gp_sv, structured_vi, vrnn_seq —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.advi_bbvi import bench_advi_bbvi
from quant_fund.models.iwae_bound import bench_iwae_bound
from quant_fund.models.nf_vi import bench_nf_vi
from quant_fund.models.sparse_gp_sv import bench_sparse_gp_sv
from quant_fund.models.structured_vi import bench_structured_vi
from quant_fund.models.vrnn_seq import bench_vrnn_seq

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


def bench_advi_bbvi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("advi_bbvi", bench_advi_bbvi(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"advi_bbvi bench failed: {exc}") from exc


def bench_iwae_bound_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("iwae_bound", bench_iwae_bound(seed=_SEED + 961)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iwae_bound bench failed: {exc}") from exc


def bench_nf_vi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("nf_vi", bench_nf_vi(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nf_vi bench failed: {exc}") from exc


def bench_sparse_gp_sv_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("sparse_gp_sv", bench_sparse_gp_sv(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sparse_gp_sv bench failed: {exc}") from exc


def bench_structured_vi_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("structured_vi", bench_structured_vi(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"structured_vi bench failed: {exc}") from exc


def bench_vrnn_seq_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("vrnn_seq", bench_vrnn_seq(seed=_SEED + 965, steps=500)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"vrnn_seq bench failed: {exc}") from exc
