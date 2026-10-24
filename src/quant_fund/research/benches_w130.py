"""Wave-126 adapters: exec-summary offline-RL sequence-decision — decision_transformer,
cql_agent, iql_agent, trajectory_transformer, sac_agent, gail_imitation —
benched on SYNTHETIC corpora/generators (torch `nn` extra). Adapters
flatten to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.cql_agent import bench_cql_agent
from quant_fund.models.decision_transformer import bench_decision_transformer
from quant_fund.models.gail_imitation import bench_gail_imitation
from quant_fund.models.iql_agent import bench_iql_agent
from quant_fund.models.sac_agent import bench_sac_agent
from quant_fund.models.trajectory_transformer import bench_trajectory_transformer

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


def bench_decision_transformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "decision_transformer",
                bench_decision_transformer(seed=_SEED + 768, n_ep=60, iters=200),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"decision_transformer bench failed: {exc}") from exc


def bench_cql_agent_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("cql_agent", bench_cql_agent(seed=_SEED + 769, n_ep=60, iters=200))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"cql_agent bench failed: {exc}") from exc


def bench_iql_agent_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("iql_agent", bench_iql_agent(seed=_SEED + 770, n_ep=60, iters=200))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"iql_agent bench failed: {exc}") from exc


def bench_trajectory_transformer_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "trajectory_transformer",
                bench_trajectory_transformer(seed=_SEED + 771, n_ep=60, iters=150),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"trajectory_transformer bench failed: {exc}") from exc


def bench_sac_agent_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob("sac_agent", bench_sac_agent(seed=_SEED + 772, steps=800, batch=64))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"sac_agent bench failed: {exc}") from exc


def bench_gail_imitation_family() -> dict[str, float]:
    try:
        return _floats(
            _finite_blob(
                "gail_imitation",
                bench_gail_imitation(seed=_SEED + 773, n_ep=40, iters=200),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"gail_imitation bench failed: {exc}") from exc
