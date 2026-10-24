"""Numerical guards."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Relative step for the central second-difference Hessian. One scale, shared
# by the count, fractional, and discrete-choice MLEs.
HESSIAN_STEP_SCALE = 1e-5


class ScalarObjective(Protocol):
    """Negative log-likelihood of a parameter vector."""

    def __call__(self, theta: Array, /) -> float: ...


def require_finite(x: Array, name: str = "array") -> Array:
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or inf")
    return x


def stable_decimal(value: float, sig: int = 8) -> float:
    """Round to ``sig`` significant decimal digits via CPython's formatter.

    ``np.exp`` / ``np.sqrt`` follow libm, so the same seeded synthetic path
    differs by an ulp between glibc builds. Eight significant digits sit well
    above that noise and below the figures these receipts report. Non-finite
    values and exact zeros pass through.
    """
    number = float(value)
    if not math.isfinite(number) or number == 0.0:
        return number
    return float(format(number, f".{sig}g"))


def stable_decimal_array(values: Array, sig: int = 8) -> Array:
    """Elementwise :func:`stable_decimal`. Shape is preserved."""
    arr = np.asarray(values, dtype=np.float64)
    flat = arr.reshape(-1)
    out = np.empty(flat.shape[0], dtype=np.float64)
    spec = f".{sig}g"
    for index, item in enumerate(flat):
        number = float(item)
        if not math.isfinite(number) or number == 0.0:
            out[index] = number
        else:
            out[index] = float(format(number, spec))
    return out.reshape(arr.shape)


def clip_positive(x: Array | float, floor: float = 1e-12) -> Array | float:
    if np.isscalar(x):
        v = float(np.asarray(x).item())
        if not np.isfinite(v) or v < floor:
            return floor
        return v
    out = np.asarray(x, dtype=float)
    out = np.where(np.isfinite(out), out, floor)
    return np.asarray(np.maximum(out, floor), dtype=np.float64)


def positive_integral_count(value: object) -> int:
    """Return a positive integral count, or 0 when the evidence is not one.

    Bools, strings, non-integers, and non-finite values are rejected rather
    than coerced. Callers treat 0 as missing evidence and fail closed.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0 or not numeric.is_integer():
        return 0
    return int(numeric)


def midrank(x: Array) -> Array:
    """Average ranks (1-based), ties share the group mean — scipy ``rankdata``
    ``'average'`` semantics. Ordinal ranks (``argsort`` of ``argsort``) break
    ties by position, so downstream values change when input rows are permuted;
    midranks are permutation-invariant."""
    v = np.asarray(x, dtype=float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(v.shape[0], dtype=float)
    ranks[order] = np.arange(1, v.shape[0] + 1, dtype=float)
    _uniq, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def require_upper_tail_alpha(alpha: float) -> float:
    """Require a loss-quantile alpha in (0.5, 1)."""
    level = float(alpha)
    if not np.isfinite(level) or not (0.5 < level < 1.0):
        raise ValueError("alpha must be in (0.5, 1)")
    return level


def numeric_hessian(objective: ScalarObjective, theta: Array) -> Array:
    """Central second-difference Hessian of a scalar objective."""
    width = theta.size
    step = HESSIAN_STEP_SCALE * np.maximum(1.0, np.abs(theta))
    hessian = np.zeros((width, width))
    for i in range(width):
        for j in range(i, width):
            step_i = np.zeros(width)
            step_j = np.zeros(width)
            step_i[i] = step[i]
            step_j[j] = step[j]
            hessian[i, j] = hessian[j, i] = (
                float(objective(theta + step_i + step_j))
                - float(objective(theta + step_i - step_j))
                - float(objective(theta - step_i + step_j))
                + float(objective(theta - step_i - step_j))
            ) / (4.0 * step[i] * step[j])
    return hessian
