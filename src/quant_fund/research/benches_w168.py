"""Wave-126 adapters: exec-summary RL-exotics canon — awac,
redq, td7_lite, crossq, dr3_reg, ob2i —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.awac import bench_awac
from quant_fund.models.crossq import bench_crossq
from quant_fund.models.dr3_reg import bench_dr3_reg
from quant_fund.models.ob2i import bench_ob2i
from quant_fund.models.redq import bench_redq
from quant_fund.models.td7_lite import bench_td7_lite

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


def bench_awac_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("awac", bench_awac(seed=_SEED + 960)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"awac bench failed: {exc}") from exc


def bench_redq_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("redq", bench_redq(seed=_SEED + 961, steps=600)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"redq bench failed: {exc}") from exc


def bench_td7_lite_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("td7_lite", bench_td7_lite(seed=_SEED + 962)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"td7_lite bench failed: {exc}") from exc


def bench_crossq_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("crossq", bench_crossq(seed=_SEED + 963)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"crossq bench failed: {exc}") from exc


def bench_dr3_reg_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("dr3_reg", bench_dr3_reg(seed=_SEED + 964)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"dr3_reg bench failed: {exc}") from exc


def bench_ob2i_family() -> dict[str, float]:
    try:
        return _floats(_finite_blob("ob2i", bench_ob2i(seed=_SEED + 965)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"ob2i bench failed: {exc}") from exc
