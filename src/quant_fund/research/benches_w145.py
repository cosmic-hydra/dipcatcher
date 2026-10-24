"""Wave-126 adapters: exec-summary retrieval-RAG canon — bm25_retriever,
dpr_retriever, colbert_late, hyde_retrieval, reranker_crossenc, rrf_fusion —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.bm25_retriever import bench_bm25_retriever
from quant_fund.models.colbert_late import bench_colbert_late
from quant_fund.models.dpr_retriever import bench_dpr_retriever
from quant_fund.models.hyde_retrieval import bench_hyde_retrieval
from quant_fund.models.reranker_crossenc import bench_reranker_crossenc
from quant_fund.models.rrf_fusion import bench_rrf_fusion

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


def bench_bm25_retriever_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("bm25_retriever", bench_bm25_retriever(seed=_SEED + 858)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"bm25_retriever bench failed: {exc}") from exc


def bench_dpr_retriever_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dpr_retriever", bench_dpr_retriever(seed=_SEED + 859)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dpr_retriever bench failed: {exc}") from exc


def bench_colbert_late_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "colbert_late",
                bench_colbert_late(seed=_SEED + 860, n_docs=120, n_queries=40, iters=150),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"colbert_late bench failed: {exc}") from exc


def bench_hyde_retrieval_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("hyde_retrieval", bench_hyde_retrieval(seed=_SEED + 861)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"hyde_retrieval bench failed: {exc}") from exc


def bench_reranker_crossenc_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("reranker_crossenc", bench_reranker_crossenc(seed=_SEED + 862)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"reranker_crossenc bench failed: {exc}") from exc


def bench_rrf_fusion_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("rrf_fusion", bench_rrf_fusion(seed=_SEED + 863)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"rrf_fusion bench failed: {exc}") from exc
