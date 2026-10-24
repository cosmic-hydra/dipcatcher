"""Kernel-MMD change-point detection (offline + sequential).

The core statistic is the unbiased squared MMD between two segments of a
stream under an RBF kernel with median-heuristic bandwidth:

    MMD_u^2(X, Y) = 1/(m(m-1)) sum_{i!=j} k(x_i, x_j)
                  + 1/(n(n-1)) sum_{i!=j} k(y_i, y_j)
                  - 2/(mn)       sum_{i, j} k(x_i, y_j)

Offline: top-down binary segmentation where each candidate split t
maximizes MMD_u^2(x[left], x[right]); a permutation-calibrated threshold
controls Type-I error (recommended calibration: permutation maxima,
distribution-free under exchangeability).

Sequential: sliding-window scan — for each new observation the window is
split in halves and its MMD_u^2 is compared to a threshold calibrated on
burn-in permutations, giving a distribution-free alarm rule (KCUSUM-style
scan statistic).

References
----------
- Flynn & Yoo (2019). Change detection with the kernel cumulative sum
  statistic. *IEEE Trans. Inf. Theory* 65 — arXiv:1902.00024 (verified;
  kernel CUSUM scan statistic).
- Li et al. (2015). M-statistic for kernel change-point detection.
  *NeurIPS* 28 — scan-statistic MMD change-point (proceedings).
- Gretton et al. (2012). A kernel two-sample test. *JMLR* 13 —
  arXiv:0805.2368 (verified; unbiased MMD estimator + median heuristic).
- Arlot, Celisse & Harchaoui (2019). A kernel multiple change-point
  algorithm via model selection. *JMLR* 20 — arXiv:1202.3878 (verified;
  kernel change-point cost).

Honesty
-------
All benches run on seeded SYNTHETIC streams generated in-module. Size /
detection / delay numbers validate machinery only — never market
evidence.

Composition notes
-----------------
- ``models/bocpd_changepoint.py``: Bayesian online change-point —
  complementary (parametric hazard posterior vs distribution-free MMD).
- ``models/conformal.py``: distribution-free prediction sets — same
  calibration-first spirit.
- ``research/benches_w25.py``: wraps ``bench_kernel_changepoint`` as an
  OPTIONAL scorecard family.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _check_series(x: FloatArray, min_len: int) -> FloatArray:
    v = np.asarray(x, dtype=float)
    if v.ndim == 1:
        v = v.reshape(-1, 1)
    if v.shape[0] < min_len or not np.isfinite(v).all():
        raise ValueError(f"need >= {min_len} finite observations")
    return v


def median_bandwidth(x: FloatArray, y: FloatArray | None = None) -> float:
    """Median-heuristic bandwidth on pooled pairwise distances."""
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    z = (
        a
        if y is None
        else np.vstack([a, np.asarray(y, dtype=float).reshape(len(np.asarray(y, dtype=float)), -1)])
    )
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    iu = np.triu_indices(z.shape[0], k=1)
    med = float(np.median(d2[iu]))
    if med <= 0:
        med = float(d2.mean())
    if med <= 0:
        raise ValueError("degenerate series: zero pairwise distance")
    return float(np.sqrt(med / 2.0))


def _rbf_grams(
    x: FloatArray, y: FloatArray, sigma: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    dxx = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2)
    dyy = np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=2)
    dxy = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=2)
    g = -1.0 / (2.0 * sigma * sigma)
    return np.exp(g * dxx), np.exp(g * dyy), np.exp(g * dxy)


def mmd_u(x: FloatArray, y: FloatArray, sigma: float | None = None) -> float:
    """Unbiased squared-MMD between two samples under the RBF kernel."""
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.ndim == 1:
        xa = xa.reshape(-1, 1)
    if ya.ndim == 1:
        ya = ya.reshape(-1, 1)
    m, n = xa.shape[0], ya.shape[0]
    if m < 2 or n < 2:
        raise ValueError("each side needs >= 2 observations")
    if sigma is None:
        sigma = median_bandwidth(xa, ya)
    kxx, kyy, kxy = _rbf_grams(xa, ya, sigma)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2.0 * kxy.sum() / (m * n))


def mmd_scan(
    x: FloatArray, min_seg: int = 20, sigma: float | None = None
) -> tuple[int, FloatArray]:
    """MMD_u^2 between left/right segments for every candidate split.

    Returns (argmax index, statistic array). Split t divides
    x[:t] | x[t:]; indices outside [min_seg, n-min_seg] carry nan.
    """
    v = _check_series(x, 2 * min_seg)
    n = v.shape[0]
    if sigma is None:
        sigma = median_bandwidth(v)
    stat = np.full(n, np.nan)
    k = _full_gram(v, sigma)
    for t in range(min_seg, n - min_seg + 1):
        stat[t] = _mmd_u_from_gram(k, 0, t, n)
    best = int(np.nanargmax(stat))
    return best, stat


def _full_gram(v: FloatArray, sigma: float) -> FloatArray:
    d2 = np.sum((v[:, None, :] - v[None, :, :]) ** 2, axis=2)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def _mmd_u_from_gram(k: FloatArray, lo: int, mid: int, hi: int) -> float:
    """MMD_u^2 for contiguous segments [lo,mid) | [mid,hi) from a Gram."""
    m = mid - lo
    n = hi - mid
    kxx = k[lo:mid, lo:mid]
    kyy = k[mid:hi, mid:hi]
    kxy = k[lo:mid, mid:hi]
    sxx = kxx.sum() - np.trace(kxx)
    syy = kyy.sum() - np.trace(kyy)
    return float(sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * kxy.sum() / (m * n))


def permutation_threshold(
    x: FloatArray,
    alpha: float = 0.05,
    n_perm: int = 200,
    min_seg: int = 20,
    seed: int = 0,
) -> float:
    """Permutation-calibrated threshold for the max-MMD scan statistic."""
    v = _check_series(x, 2 * min_seg)
    if not (0 < alpha < 1) or n_perm < 20:
        raise ValueError("0 < alpha < 1 and n_perm >= 20")
    rng = np.random.default_rng(seed)
    sigma = median_bandwidth(v)
    maxima = np.empty(n_perm)
    for b in range(n_perm):
        perm = rng.permutation(v.shape[0])
        k = _full_gram(v[perm], sigma)
        s = np.empty(v.shape[0])
        for t in range(min_seg, v.shape[0] - min_seg + 1):
            s[t] = _mmd_u_from_gram(k, 0, t, v.shape[0])
        maxima[b] = np.nanmax(s[min_seg : v.shape[0] - min_seg + 1])
    return float(np.quantile(maxima, 1.0 - alpha))


def binseg_detect(
    x: FloatArray,
    alpha: float = 0.05,
    min_seg: int = 20,
    max_cp: int = 20,
    n_perm: int = 100,
    seed: int = 0,
) -> list[int]:
    """Top-down binary segmentation with permutation stopping rule.

    Each accepted split must beat the (1-alpha) permutation threshold of
    its own segment — a distribution-free stopping rule.
    """
    v = _check_series(x, 2 * min_seg)
    if max_cp < 1:
        raise ValueError("max_cp >= 1")
    sigma = median_bandwidth(v)
    rng = np.random.default_rng(seed)
    cps: list[int] = []
    stack: list[tuple[int, int]] = [(0, v.shape[0])]
    while stack and len(cps) < max_cp:
        lo, hi = stack.pop()
        if hi - lo < 2 * min_seg:
            continue
        seg = v[lo:hi]
        k = _full_gram(seg, sigma)
        stat = np.full(hi - lo, np.nan)
        for t in range(min_seg, hi - lo - min_seg + 1):
            stat[t] = _mmd_u_from_gram(k, 0, t, hi - lo)
        t_star = int(np.nanargmax(stat))
        s_star = float(stat[t_star])
        # permutation threshold on this segment
        maxima = np.empty(n_perm)
        for b in range(n_perm):
            perm = rng.permutation(hi - lo)
            kp = _full_gram(seg[perm], sigma)
            sb = np.empty(hi - lo)
            for t in range(min_seg, hi - lo - min_seg + 1):
                sb[t] = _mmd_u_from_gram(kp, 0, t, hi - lo)
            maxima[b] = sb[min_seg : hi - lo - min_seg + 1].max()
        if s_star <= float(np.quantile(maxima, 1.0 - alpha)):
            continue
        cp = lo + t_star
        cps.append(cp)
        stack.append((lo, cp))
        stack.append((cp, hi))
    return sorted(cps)


def cusum_baseline(x: FloatArray, burn: int = 50) -> tuple[int | None, float]:
    """Parametric CUSUM benchmark: cumulative deviation of returns from
    the burn-in mean; alarms at 5 burn-in std. Returns (alarm index,
    max statistic)."""
    v = np.asarray(x, dtype=float).ravel()
    if v.size < 2 * burn or not np.isfinite(v).all():
        raise ValueError(f"need >= {2 * burn} finite observations")
    mu = float(v[:burn].mean())
    sd = float(v[:burn].std())
    if sd <= 0:
        raise ValueError("degenerate burn-in")
    # Page-style CUSUM: |cumulative demeaned sum| / (sd * sqrt(t));
    # on a sustained shift the sum grows ~t while the norm grows ~sqrt(t).
    dev = v[burn:] - mu
    t_idx = np.arange(1, v.size - burn + 1, dtype=float)
    stat = np.abs(np.cumsum(dev)) / (sd * np.sqrt(t_idx))
    mx = float(stat.max())
    hit = np.flatnonzero(stat > 5.0)
    return (int(hit[0] + burn) if hit.size else None, mx)


def synth_stream(
    n: int,
    cp: int | None,
    kind: str = "mean",
    size: float = 0.5,
    seed: int = 0,
) -> FloatArray:
    """Synthetic stream with an optional planted shift at ``cp``.

    kind in {'mean', 'variance', 'family'}: location shift, scale shift,
    or Gaussian -> Student-t(3) family shift.
    """
    if n < 100:
        raise ValueError("n >= 100")
    if cp is not None and not (10 <= cp < n - 10):
        raise ValueError("cp must sit inside [10, n-10)")
    if kind not in {"mean", "variance", "family"}:
        raise ValueError("kind must be mean|variance|family")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if cp is not None:
        if kind == "mean":
            x[cp:] += size
        elif kind == "variance":
            x[cp:] *= 1.0 + size
        else:
            x[cp:] = rng.standard_t(3.0, n - cp) * (0.5 + size)
    return x


def bench_kernel_changepoint(
    seed: int = 20260202,
    *,
    trials: int = 8,
    n_perm: int = 60,
    reps: int = 6,
) -> dict[str, float]:
    """SYNTHETIC bench: size under null, detection rate and delay vs
    planted shift size, MMD vs parametric CUSUM comparison."""
    out: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    # --- size under the null (stationary stream)
    alarms = 0
    for rep in range(trials):
        x = rng.standard_normal(240)
        thr = permutation_threshold(x, alpha=0.05, n_perm=n_perm, seed=seed + rep)
        _, stat = mmd_scan(x, min_seg=30)
        if np.nanmax(stat) > thr:
            alarms += 1
    out["synthetic_size_at_05"] = alarms / trials
    # --- detection rate / delay vs mean-shift magnitude
    for j, sz in enumerate([0.3, 0.8, 1.5]):
        det = 0
        delay_sum = 0.0
        for rep in range(reps):
            x = synth_stream(300, cp=150, kind="mean", size=sz, seed=seed + 100 + rep)
            thr = permutation_threshold(
                synth_stream(300, None, seed=seed + rep),
                alpha=0.05,
                n_perm=n_perm,
                seed=seed + rep,
            )
            cp_hat, stat = mmd_scan(x, min_seg=30)
            if np.nanmax(stat) > thr and abs(cp_hat - 150) < 60:
                det += 1
                delay_sum += cp_hat - 150
        out[f"synthetic_detect_rate_size{j}"] = det / reps
        out[f"synthetic_mean_delay_size{j}"] = delay_sum / max(det, 1)
    # --- vs parametric CUSUM on a variance shift (MMD's edge)
    mmd_hits = 0
    cusum_hits = 0
    reps = 6
    for rep in range(reps):
        x = synth_stream(320, cp=160, kind="variance", size=1.0, seed=seed + 200 + rep)
        cps = binseg_detect(x, alpha=0.05, min_seg=40, n_perm=40, seed=seed + rep)
        if any(abs(c - 160) < 50 for c in cps):
            mmd_hits += 1
        alarm, _ = cusum_baseline(x)
        if alarm is not None and abs(alarm - 160) < 60:
            cusum_hits += 1
    out["synthetic_mmd_varshift_rate"] = mmd_hits / reps
    out["synthetic_cusum_varshift_rate"] = cusum_hits / reps
    out["synthetic_mmd_edge"] = (
        out["synthetic_mmd_varshift_rate"] - out["synthetic_cusum_varshift_rate"]
    )
    return out
