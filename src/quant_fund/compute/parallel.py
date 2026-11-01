"""Order-preserving process maps for embarrassingly parallel research sweeps.

Each task receives ``derive_seed(base_seed, index)``. The seed depends only
on the base and the task index, not on which worker runs it and not on
completion order. Callers that draw random numbers must use that seed.
Callers whose existing math is deterministic ignore it so published numbers
stay on the original formula.

Linux uses ``fork`` so workers inherit the already-imported parent. macOS
and Windows use ``spawn``: ``fork`` after Objective-C or a BLAS pool is
initialized can deadlock, and Windows has no fork. A pool that cannot start,
or a task that cannot be pickled for spawn, falls back to the calling
process. Task exceptions still propagate. Below ``min_items`` the map stays
inline.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

_MIX = 0x9E3779B97F4A7C15
_MIX_1 = 0xBF58476D1CE4E5B9
_MIX_2 = 0x94D049BB133111EB


def derive_seed(base_seed: int, index: int) -> int:
    """Stable non-negative 31-bit seed for task ``index`` under ``base_seed``."""
    x = (int(base_seed) + _MIX) & 0xFFFFFFFFFFFFFFFF
    x ^= (int(index) + 1) * _MIX_1
    x &= 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 30)) * _MIX_1 & 0xFFFFFFFFFFFFFFFF
    x = (x ^ (x >> 27)) * _MIX_2 & 0xFFFFFFFFFFFFFFFF
    x ^= x >> 31
    return int(x & 0x7FFFFFFF)


def _invoke[T, R](payload: tuple[Callable[[T, int], R], int, T]) -> R:
    fn, seed, item = payload
    return fn(item, seed)


def _process_start_method() -> str:
    """``fork`` on Linux; ``spawn`` everywhere else.

    Spawn workers re-import the callable, so it must be a module-level
    function. ``_invoke`` is. Callables that cannot be pickled fall back
    to the parent process inside ``process_map``.
    """
    if sys.platform == "linux":
        return "fork"
    return "spawn"


def process_map[T, R](
    fn: Callable[[T, int], R],
    items: Sequence[T],
    *,
    base_seed: int,
    min_items: int = 4,
    max_workers: int | None = None,
) -> list[R]:
    """Map ``fn(item, seed)`` across processes. Result order matches ``items``."""
    sequenced = list(items)
    seeds = [derive_seed(base_seed, i) for i in range(len(sequenced))]

    def _serial() -> list[R]:
        return [fn(item, seed) for item, seed in zip(sequenced, seeds, strict=True)]

    if len(sequenced) < min_items:
        return _serial()
    workers = (
        max_workers if max_workers is not None else min(8, os.cpu_count() or 1, len(sequenced))
    )
    if workers < 2:
        return _serial()
    payloads = [(fn, seed, item) for seed, item in zip(seeds, sequenced, strict=True)]
    try:
        ctx = multiprocessing.get_context(_process_start_method())
        pool = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
    except (OSError, ValueError):
        # The start method is unavailable, or the pool cannot be created.
        return _serial()
    try:
        try:
            return list(pool.map(_invoke, payloads))
        except pickle.PicklingError:
            # Spawn cannot ship a closure or a test-local callable. The
            # serial map is the same function of (item, seed).
            return _serial()
    finally:
        pool.shutdown(wait=True)
