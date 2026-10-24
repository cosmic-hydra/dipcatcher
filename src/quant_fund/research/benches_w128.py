"""Wave-126 adapters: exec-summary DL SOTA-3 — kan_forecaster,
ts_mixer, informer_attn, cnn_alpha, mask_autoencoder, graph_temporal —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cnn_alpha import bench_cnn_alpha
from quant_fund.models.graph_temporal import bench_graph_temporal
from quant_fund.models.informer_attn import bench_informer_attn
from quant_fund.models.kan_forecaster import bench_kan_forecaster
from quant_fund.models.mask_autoencoder import bench_mask_autoencoder
from quant_fund.models.ts_mixer import bench_ts_mixer

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


def bench_kan_forecaster_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "kan_forecaster",
                bench_kan_forecaster(seed=_SEED + 756, n=1200, steps=280, mlp_steps=140, batch=64),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kan_forecaster bench failed: {exc}") from exc


def bench_ts_mixer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "ts_mixer",
                bench_ts_mixer(seed=_SEED + 757, t_len=800, steps=120, mlp_steps=60, batch=64),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ts_mixer bench failed: {exc}") from exc


def bench_informer_attn_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("informer_attn", bench_informer_attn(seed=_SEED + 758)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"informer_attn bench failed: {exc}") from exc


def bench_cnn_alpha_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "cnn_alpha",
                bench_cnn_alpha(seed=_SEED + 759, n=800, steps=60, batch=64),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cnn_alpha bench failed: {exc}") from exc


def bench_mask_autoencoder_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "mask_autoencoder",
                bench_mask_autoencoder(seed=_SEED + 760, steps=120, probe_steps=60, batch=64),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"mask_autoencoder bench failed: {exc}") from exc


def bench_graph_temporal_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "graph_temporal",
                bench_graph_temporal(
                    seed=_SEED + 761,
                    t_len=700,
                    joint_steps=100,
                    ind_steps=40,
                    batch=64,
                ),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"graph_temporal bench failed: {exc}") from exc
