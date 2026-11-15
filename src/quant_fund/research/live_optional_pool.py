"""Process pool for the seeded live-optional research benches (``run_research(jobs=...)``).

The bench table (``_live_optional_benches``) stays in ``quant_fund.research.agent``
next to its imports; it is imported lazily here so this module does not form an
import cycle with ``agent`` and spawn workers can unpickle the top-level callables.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from quant_fund.utils.seeds import set_global_seed

if TYPE_CHECKING:
    from quant_fund.config.models import AppConfig


def _run_live_optional_bench(name: str, seed: int) -> Any:
    """Run one live-optional bench from freshly seeded process RNGs.

    Reseeding per bench makes a blob independent of whatever ran before it in
    the same process, which is what lets ``jobs`` reorder the benches.
    """
    from quant_fund.research.agent import _live_optional_benches

    set_global_seed(seed)
    return _live_optional_benches()[name]()


def _live_optional_worker_init(seed: int) -> None:
    """One BLAS / OpenMP / torch thread per pool worker: the pool is the parallelism.

    The env vars cover libraries a bench loads later; ``threadpool_limits``
    covers the ones already loaded (including torch, imported by the seeding).
    """
    from threadpoolctl import threadpool_limits

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    set_global_seed(seed)
    threadpool_limits(limits=1)


def _run_live_optional(names: Sequence[str], *, jobs: int, seed: int) -> dict[str, Any]:
    """Run the named live-optional benches over ``jobs`` processes; blobs in ``names`` order.

    ``jobs > 1`` uses a spawn-context pool (not fork: the parent may already
    hold BLAS / torch thread pools). A failing bench still raises.
    """
    if jobs <= 1:
        return {name: _run_live_optional_bench(name, seed) for name in names}
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    from itertools import repeat

    with ProcessPoolExecutor(
        max_workers=min(jobs, max(len(names), 1)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_live_optional_worker_init,
        initargs=(seed,),
    ) as pool:
        blobs = list(pool.map(_run_live_optional_bench, names, repeat(seed)))
    return dict(zip(names, blobs, strict=True))


def _live_optional_family_blobs(*, jobs: int = 1, seed: int = 0) -> dict[str, Any]:
    """Seeded live-optional benches. Default ``run_research`` still runs them.

    Every bench starts from ``seed``, so ``jobs`` changes the wall clock, not a
    blob (tests/unit/research/test_live_optional_pool.py), and the parent is
    reseeded afterwards so the steps that follow see one RNG state for any
    ``jobs``.
    """
    from quant_fund.research.agent import _live_optional_benches

    blobs = _run_live_optional(tuple(_live_optional_benches()), jobs=jobs, seed=seed)
    set_global_seed(seed)
    return blobs


def _merge_live_optional(
    families: dict[str, Any], include: bool, jobs: int, config: AppConfig
) -> None:
    """Add the seeded live-optional blobs to ``families`` when ``include`` is set."""
    if include:
        families.update(_live_optional_family_blobs(jobs=jobs, seed=config.train.random_seed))
