"""Wave-102 adapters: multi-armed bandit canon — UCB1/
epsilon-greedy/explore-then-commit stochastic bandits,
Bernoulli KL-UCB, LinUCB + linear Thompson contextual
bandits, EXP3 + Hedge adversarial bandits, successive-
elimination + LUCB best-arm identification, and sliding-
window + discounted UCB for non-stationary rewards.

All families run SYNTHETIC self-check benches only; adapters
flatten the returned dict to a finite float scorecard.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quant_fund.models.adversarial_bandits import bench_adversarial_bandits
from quant_fund.models.best_arm import bench_best_arm
from quant_fund.models.contextual_bandits import bench_contextual_bandits
from quant_fund.models.kl_bandits import bench_kl_bandits
from quant_fund.models.nonstationary_bandits import bench_nonstationary_bandits
from quant_fund.models.stochastic_bandits import bench_stochastic_bandits

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


def bench_stochastic_bandits_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("stochastic_bandits", bench_stochastic_bandits(seed=_SEED + 600))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"stochastic_bandits bench failed: {exc}") from exc


def bench_kl_bandits_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("kl_bandits", bench_kl_bandits(seed=_SEED + 601, steps=600, runs=4))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"kl_bandits bench failed: {exc}") from exc


def bench_contextual_bandits_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("contextual_bandits", bench_contextual_bandits(seed=_SEED + 602))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"contextual_bandits bench failed: {exc}") from exc


def bench_adversarial_bandits_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob("adversarial_bandits", bench_adversarial_bandits(seed=_SEED + 603))
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"adversarial_bandits bench failed: {exc}") from exc


def bench_best_arm_family() -> dict[str, float]:
    try:
        return _isinstance_floats(_finite_blob("best_arm", bench_best_arm(seed=_SEED + 604)))
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"best_arm bench failed: {exc}") from exc


def bench_nonstationary_bandits_family() -> dict[str, float]:
    try:
        return _isinstance_floats(
            _finite_blob(
                "nonstationary_bandits",
                bench_nonstationary_bandits(seed=_SEED + 605),
            )
        )
    except ImportError:
        raise
    except _BENCH_EXC as exc:
        raise ValueError(f"nonstationary_bandits bench failed: {exc}") from exc
