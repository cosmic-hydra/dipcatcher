"""Numba kernels for sequential recurrences that do not vectorize.

Each kernel repeats the previous Python floating-point steps (reset
accumulators, not a global cumsum; Goertzel in the published order).
``cache=True`` keeps the compiled code for later processes.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from numba import njit

from quant_fund.compute.numba_guard import call_kernel


def _bind_python_body(public: Any, kernel: Any) -> None:
    """Publish ``kernel``'s undecorated body as ``public.py_func``.

    ``call_kernel`` is the runtime path (compiled, or the Python body when
    Numba's overload dispatcher is broken). Parity tests call ``py_func``
    directly so that path stays reachable after the wrapper.
    """
    body = getattr(kernel, "py_func", None)
    public.py_func = kernel if body is None else body


@njit(cache=True)
def _reset_bounds_nb(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Bar-start indices for a reset accumulator (volume / dollar bars).

    Excess past the threshold is discarded, matching the Python loop.
    Starts at the final tick (``n - 1``) are dropped, also matching that loop.
    """
    n = weights.shape[0]
    starts = np.empty(n, dtype=np.int64)
    n_starts = 0
    acc = 0.0
    start = 0
    for t in range(n):
        acc += weights[t]
        if acc >= threshold:
            starts[n_starts] = start
            n_starts += 1
            start = t + 1
            acc = 0.0
    out_n = 0
    last = n - 1
    for i in range(n_starts):
        if starts[i] < last:
            starts[out_n] = starts[i]
            out_n += 1
    return starts[:out_n]


@njit(cache=True)
def _imbalance_bounds_nb(
    values: np.ndarray, alpha: float, e_ticks: float, e_init: float
) -> np.ndarray:
    """Tick- or dollar-imbalance bar starts (AFML 2.5.2)."""
    n = values.shape[0]
    starts = np.empty(n, dtype=np.int64)
    n_starts = 0
    theta = 0.0
    e_theta = e_init
    start = 0
    for t in range(n):
        theta += values[t]
        e_theta = (1.0 - alpha) * e_theta + alpha * abs(values[t])
        floor = e_theta if e_theta > 1e-12 else 1e-12
        if abs(theta) >= e_ticks * floor:
            starts[n_starts] = start
            n_starts += 1
            start = t + 1
            theta = 0.0
    out_n = 0
    last = n - 1
    for i in range(n_starts):
        if starts[i] < last:
            starts[out_n] = starts[i]
            out_n += 1
    return starts[:out_n]


@njit(cache=True)
def _run_bounds_nb(signs: np.ndarray, alpha: float, e_ticks: float) -> np.ndarray:
    """Tick-run bar starts. Zeros count as neither side, same as the slice sums."""
    n = signs.shape[0]
    starts = np.empty(n, dtype=np.int64)
    n_starts = 0
    start = 0
    e_share = 0.5
    run_buy = 0.0
    run_sell = 0.0
    for t in range(n):
        is_buy = 1.0 if signs[t] > 0.0 else 0.0
        e_share = (1.0 - alpha) * e_share + alpha * is_buy
        if signs[t] > 0.0:
            run_buy += 1.0
        elif signs[t] < 0.0:
            run_sell += 1.0
        run = run_buy if run_buy >= run_sell else run_sell
        share = e_share if e_share >= (1.0 - e_share) else (1.0 - e_share)
        thresh = e_ticks * share
        if thresh < 2.0:
            thresh = 2.0
        if run >= thresh:
            starts[n_starts] = start
            n_starts += 1
            start = t + 1
            run_buy = 0.0
            run_sell = 0.0
    out_n = 0
    last = n - 1
    for i in range(n_starts):
        if starts[i] < last:
            starts[out_n] = starts[i]
            out_n += 1
    return starts[:out_n]


@njit(cache=True)
def _goertzel_powers_nb(demeaned: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Normalized Goertzel power at each period. ``periods`` are float bars."""
    n = demeaned.shape[0]
    n_periods = periods.shape[0]
    out = np.empty(n_periods, dtype=np.float64)
    n2 = float(n * n)
    for pi in range(n_periods):
        coeff = 2.0 * math.cos(2.0 * math.pi / periods[pi])
        s_prev = 0.0
        s_prev2 = 0.0
        for i in range(n):
            s = demeaned[i] + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s
        power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        out[pi] = power / n2
    return out


def reset_bounds(weights: np.ndarray, threshold: float) -> np.ndarray:
    """Bar-start indices. Numba when it compiles; the same loop otherwise."""
    return cast(np.ndarray, call_kernel(_reset_bounds_nb, weights, threshold))


def imbalance_bounds(values: np.ndarray, alpha: float, e_ticks: float, e_init: float) -> np.ndarray:
    """Tick- or dollar-imbalance bar starts. Numba when it compiles."""
    return cast(np.ndarray, call_kernel(_imbalance_bounds_nb, values, alpha, e_ticks, e_init))


def run_bounds(signs: np.ndarray, alpha: float, e_ticks: float) -> np.ndarray:
    """Tick-run bar starts. Numba when it compiles; the same loop otherwise."""
    return cast(np.ndarray, call_kernel(_run_bounds_nb, signs, alpha, e_ticks))


def goertzel_powers(demeaned: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Normalized Goertzel power at each period. ``periods`` are float bars."""
    return cast(np.ndarray, call_kernel(_goertzel_powers_nb, demeaned, periods))


_bind_python_body(reset_bounds, _reset_bounds_nb)
_bind_python_body(imbalance_bounds, _imbalance_bounds_nb)
_bind_python_body(run_bounds, _run_bounds_nb)
_bind_python_body(goertzel_powers, _goertzel_powers_nb)
