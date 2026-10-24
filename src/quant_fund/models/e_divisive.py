"""Energy-based changepoint detection (E-divisive, Matteson & (SYNTHETIC)
James 2014).

Between two candidate segments X, Y the energy divergence is

    D(X,Y) = 2 E|x - y| - E|x - x'| - E|y - y'|,

which is >= 0 with equality iff the distributions coincide
(Szekely & Rizzo 2004/2005). E-divisive recursively bisects
the series at the split maximizing D over all candidate
boundaries, then keeps the split iff its significance exceeds
a permutation p-value (rows of the distance computations
permuted, the max statistic re-computed) — a nonparametric
analog of binary segmentation with distribution-free
significance.

Honesty: energy distance is computed on the full feature
vector (works multivariate, but heavier); permutation p-values
with R replicates have ~1/R resolution. Depth cap on the
bisection recursion is fixed at 3 (documented limit). The
bench plants two mean shifts at known locations and requires
both detected boundaries within tolerance. Fail-closed on
n < 24 or zero-variance input.

References: Matteson & James (2014) JASA 109:654; Szekely &
Rizzo (2004) InterStat; Szekely & Rizzo (2005) J. Multiv.
Anal. 96:58; James & Matteson (2015) "ecp" R package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _ediv(x: FloatArray, a: int, b: int) -> float:
    """Energy divergence between x[:a] and x[a:b] segments."""
    left = x[:a]
    right = x[a:b]
    n_l = left.shape[0]
    n_r = right.shape[0]
    if n_l < 2 or n_r < 2:
        return 0.0

    # pairwise L2 distances
    def pdist(m1: FloatArray, m2: FloatArray) -> FloatArray:
        d = (m1 * m1).sum(axis=1)[:, None] + (m2 * m2).sum(axis=1)[None, :] - 2.0 * (m1 @ m2.T)
        return np.sqrt(np.clip(d, 0.0, None))

    xy = float(pdist(left, right).mean())
    xx = float(pdist(left, left).mean())
    yy = float(pdist(right, right).mean())
    return 2.0 * xy - xx - yy


def _best_split(x: FloatArray, lo: int, hi: int, min_seg: int) -> tuple[int, float]:
    """Argmax boundary in (lo, hi) with min segment size."""
    n = hi - lo
    best_t, best_d = -1, 0.0
    seg = x[lo:hi]
    for t in range(min_seg, n - min_seg):
        d = _ediv(seg, t, n)
        if d > best_d:
            best_d, best_t = d, t
    return (lo + best_t if best_t > 0 else -1), best_d


def _perm_p(
    x: FloatArray, lo: int, hi: int, min_seg: int, r: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Permutation p-value for the best split in [lo, hi)."""
    t_star, d_star = _best_split(x, lo, hi, min_seg)
    if t_star < 0:
        return -1.0, 1.0
    seg = x[lo:hi].copy()
    exceed = 0
    for _ in range(r):
        perm = rng.permutation(seg.shape[0])
        _, d_b = _best_split(seg[perm], 0, seg.shape[0], min_seg)
        if d_b >= d_star:
            exceed += 1
    return float(t_star), float((exceed + 1) / (r + 1))


def e_divisive(
    x: FloatArray,
    r: int = 199,
    alpha: float = 0.05,
    min_seg: int = 10,
    seed: int = 0,
    max_depth: int = 3,
) -> dict[str, float | FloatArray]:
    """Recursive energy-divergence bisection. Returns sorted
    changepoint indices."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim == 1:
        xx = xx[:, None]
    n = xx.shape[0]
    if n < 2 * min_seg + 4:
        raise ValueError("series too short")
    if float(np.ptp(xx, axis=0).max()) <= 1e-12:
        raise ValueError("constant input")
    rng = np.random.default_rng(seed)
    cps: list[int] = []
    stack = [(0, n, 0)]
    while stack:
        lo, hi, depth = stack.pop()
        if depth >= max_depth or hi - lo < 2 * min_seg + 2:
            continue
        t, p_val = _perm_p(xx, lo, hi, min_seg, r, rng)
        if t >= 0 and p_val < alpha:
            cps.append(int(t))
            stack.append((lo, int(t), depth + 1))
            stack.append((int(t), hi, depth + 1))
    cps.sort()
    return {
        "changepoints": np.asarray(cps, dtype=np.float64),
        "n_changepoints": float(len(cps)),
    }


def bench_e_divisive(seed: int = 20261231 + 461, *, n_perm: int = 99) -> dict[str, float]:
    """SYNTHETIC check — detects two planted mean shifts."""
    rng = np.random.default_rng(seed)
    y = np.concatenate(
        [
            rng.normal(0.0, 0.5, size=100),
            rng.normal(1.5, 0.5, size=100),
            rng.normal(-0.8, 0.5, size=100),
        ]
    )
    out = e_divisive(y, r=n_perm, seed=seed)
    cps = np.asarray(out["changepoints"], dtype=np.float64)
    if cps.size < 2:
        raise ValueError(f"edivisive missed splits: {cps.tolist()}")
    near = all(any(abs(cp - true_cp) <= 12 for cp in cps) for true_cp in (100, 200))
    if not near:
        raise ValueError(f"edivisive off: {cps.tolist()}")
    return {
        "synthetic_cps": float(cps.size),
        "synthetic_first_cp": float(cps[0]),
        "synthetic_score": 1.0,
    }
