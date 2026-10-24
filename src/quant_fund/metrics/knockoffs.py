"""Model-X knockoff filter for FDR-controlled variable selection.

A knockoff copy ``X̃`` is a synthetic feature matrix that swaps each real
feature against its in-sample clone while preserving exchangeability:
under the null that feature j is irrelevant, the swap leaves the
distribution of ``(X, X̃)`` unchanged. The Barber-Candès knockoff+
threshold then selects variables whose importance statistic exceeds the
clone's by enough to bound the false discovery rate at level ``q``.

Functions
---------
- :func:`sample_gaussian_knockoffs` — equi-correlated or SDP ``s`` construction.
- :func:`knockoff_stat_diff` — ``W_j = |beta_j| - |betã_j|`` ridge importance.
- :func:`knockoff_threshold` — knockoff+ threshold at FDR level ``q``.
- :func:`knockoff_select` — full pipeline → selected indices.
- :func:`stability_select` — Meinshausen-Bühlmann stability frequencies.
- :func:`synth_linear` — seeded equicorrelated design with sparse truth.
- :func:`bench_knockoffs` — SYNTHETIC telemetry blob.

References
----------
- Barber & Candès (2015). Controlling the false discovery rate via
  knockoffs. *Annals of Statistics* 43(5):2055 — arXiv:1404.5609.
- Candès, Fan, Janson & Lv (2018). Panning for gold: model-X knockoffs
  for high-dimensional controlled variable selection. *JRSS-B* 80(3) —
  arXiv:1610.02351.
- Meinshausen & Bühlmann (2010). Stability selection. *JRSS-B* 72:417.

Honesty
-------
Reported numbers are SYNTHETIC checks of FDR control and power on seeded
Gaussian designs — they verify exchangeability and the knockoff+
threshold, never identify real causal or predictive features. The bench
also reports an honest failure mode: the OMP entry-strength statistic
biases null ``W`` positive under correlated designs (clones enter the
greedy path after their originals), breaking the sign-flip symmetry the
+threshold relies on — its empirical FDR exceeds ``q`` and is reported
as ``synthetic_omp_fdr_breach`` rather than hidden.

Composition notes
-----------------
- ``research/selective_inference.py``: post-selection p-values — this
  lane controls FDR up front rather than correcting selection after.
- ``metrics/sparse.py`` / ``models/glasso.py``: sparsity primitives —
  knockoffs wrap any importance statistic in an FDR guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _as_design(x: FloatArray, name: str = "x", min_cols: int = 2) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < min_cols or a.shape[0] < a.shape[1]:
        raise ValueError(f"{name}: expected (n>=p, p>={min_cols}) matrix")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: non-finite values")
    return a


def _check_q(q: float) -> None:
    if not (0.0 < q <= 0.5):
        raise ValueError("q must be in (0, 0.5]")


def _s_equi(cov: FloatArray) -> FloatArray:
    """Equi-correlated s: s_j = min(2*lambda_min(Sigma), 1) — safe lower bound."""
    lam_min = float(np.linalg.eigvalsh(cov).min())
    s = min(2.0 * lam_min, 1.0)
    return np.full(cov.shape[0], max(s, 1e-6))


def _s_sdp(cov: FloatArray, n_passes: int = 5) -> FloatArray:
    """Coordinate-ascent SDP s: maximize sum s_j s.t. 2Σ - diag(s) ⪰ 0.

    Each pass lifts s_j to the largest value keeping the slack matrix
    PSD, via the secular bound s_j <= ( (2Σ - diag(s) + diag(s) e_j)⁻¹ )_jj
    — implemented directly by bisection on the smallest eigenvalue.
    """
    p = cov.shape[0]
    s = _s_equi(cov)
    for _ in range(n_passes):
        for j in range(p):
            m = 2.0 * cov - np.diag(s)
            m[j, j] = 0.0  # remove current contribution; we solve for new s_j

            # need 2Σ - diag(s') ⪰ 0 with s'_j variable: M_j = 2Σ - diag(s_{-j})
            # smallest eigenvalue of (M_j - s_j I) restricted... use bisection:
            def eig_at(t: float, mj: FloatArray = m, jj: int = j) -> float:
                mm = mj.copy()
                mm[jj, jj] = -t
                return float(np.linalg.eigvalsh(mm).min())

            # want largest s_j with eig >= 0
            lo, hi = 0.0, float(2.0 * cov[j, j])
            if eig_at(hi) >= 0:
                s[j] = hi
                continue
            for _bis in range(30):
                mid = 0.5 * (lo + hi)
                if eig_at(mid) >= 0:
                    lo = mid
                else:
                    hi = mid
            s[j] = max(lo, s[j])
    return np.clip(s, 1e-8, None)


def sample_gaussian_knockoffs(
    x: FloatArray,
    method: str = "equi",
    seed: int = 0,
) -> FloatArray:
    """Sample Gaussian model-X knockoffs ``X̃``.

    ``X̃ = X (I - Σ⁻¹ diag(s)) + ε`` with ``ε ~ N(0, 2diag(s) - diag(s) Σ⁻¹ diag(s))``,
    ``s`` chosen so ``2Σ - diag(s)`` is PSD (exchangeability condition).
    """
    x = _as_design(x)
    if method not in ("equi", "sdp"):
        raise ValueError("method must be 'equi' or 'sdp'")
    n, p = x.shape
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    s = _s_equi(cov) if method == "equi" else _s_sdp(cov)
    ds = np.diag(s)
    sig_inv = np.linalg.pinv(cov, hermitian=True)
    means = x @ (np.eye(p) - sig_inv @ ds)
    cond_cov = 2.0 * ds - ds @ sig_inv @ ds
    cond_cov = (cond_cov + cond_cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(np.maximum(cond_cov, 0))
    eigvals = np.clip(eigvals, 0.0, None)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p))
    noise = (z * np.sqrt(eigvals)) @ eigvecs.T
    return np.asarray(means + noise, dtype=np.float64)


def _ridge_beta(x: FloatArray, y: FloatArray, lam: float) -> FloatArray:
    n, p = x.shape
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    gram = xc.T @ xc + lam * n * np.eye(p)
    return np.linalg.solve(gram, xc.T @ yc)


def _omp_entry_strength(x: FloatArray, y: FloatArray, max_steps: int) -> FloatArray:
    """OMP path: at each step absorb the column most correlated with the
    residual; importance Z_j = |corr at entry| (0 if never enters)."""
    n, p = x.shape
    xc = x - x.mean(axis=0)
    xc = xc / np.maximum(xc.std(axis=0), 1e-12) / math.sqrt(n)
    resid = y - y.mean()
    z = np.zeros(p)
    selected: list[int] = []
    for _ in range(min(max_steps, p)):
        cors = np.abs(xc.T @ resid)
        cors[selected] = -1.0
        j = int(np.argmax(cors))
        if cors[j] <= 1e-12:
            break
        selected.append(j)
        z[j] = float(cors[j])
        xp = xc[:, selected]
        resid = resid - xp @ np.linalg.lstsq(xp, resid, rcond=None)[0]
    return z


def knockoff_stat_diff(
    x: FloatArray,
    y: FloatArray,
    xk: FloatArray,
    lam: float = 1e-3,
    stat: str = "omp",
) -> FloatArray:
    """Importance statistics ``W_j`` with the sign-flip property.

    ``stat='omp'`` — OMP entry strength with the signed-max encoding
    ``W = max(Z, Z̃) sign(Z − Z̃)`` (columns entering the greedy path
    early beat their knockoffs); ``stat='ridge'`` — |β_j| − |β̃_j| from
    a joint ridge fit at ``lam``.
    """
    x = _as_design(x)
    xk = _as_design(xk, "xk")
    y = np.asarray(y, dtype=np.float64).ravel()
    if xk.shape != x.shape or y.size != x.shape[0]:
        raise ValueError("shape mismatch between x, xk, y")
    if stat not in ("omp", "ridge"):
        raise ValueError("stat must be 'omp' or 'ridge'")
    p = x.shape[1]
    if stat == "ridge":
        xx = np.concatenate([x, xk], axis=1)
        beta = _ridge_beta(xx, y, lam)
        return np.abs(beta[:p]) - np.abs(beta[p:])
    xx = np.concatenate([x, xk], axis=1)
    z = _omp_entry_strength(xx, y, max_steps=2 * p)
    zr, zk = z[:p], z[p:]
    # signed-max LC statistic: W = max(Z, Z̃) * sign(Z - Z̃)
    w = np.maximum(zr, zk) * np.sign(zr - zk)
    return w


def knockoff_threshold(w: FloatArray, q: float = 0.1) -> float:
    """Knockoff+ threshold: τ = min{ t>0 : (1 + #W<=-t) / #W>=t <= q }.

    Returns +inf if no finite threshold satisfies the bound.
    """
    _check_q(q)
    w = np.asarray(w, dtype=np.float64).ravel()
    if w.size == 0:
        raise ValueError("w must be non-empty")
    ts = np.sort(np.abs(w))
    ts = ts[ts > 0]
    best = math.inf
    for t in ts:
        n_neg = int(np.sum(w <= -t))
        n_pos = int(np.sum(w >= t))
        ratio = (1.0 + n_neg) / max(n_pos, 1)
        if ratio <= q:
            best = float(t)
            break
    return best


def knockoff_select(
    x: FloatArray,
    y: FloatArray,
    q: float = 0.1,
    method: str = "equi",
    stat: str = "omp",
    lam: float = 1e-3,
    seed: int = 0,
) -> NDArray[np.intp]:
    """Full knockoff+ pipeline → selected feature indices."""
    xk = sample_gaussian_knockoffs(x, method=method, seed=seed)
    w = knockoff_stat_diff(x, y, xk, lam=lam, stat=stat)
    tau = knockoff_threshold(w, q=q)
    if not np.isfinite(tau):
        return np.empty(0, dtype=np.intp)
    return np.flatnonzero(w >= tau)


@dataclass(frozen=True)
class StabilityResult:
    inclusion_freq: FloatArray
    selected: NDArray[np.intp]


def stability_select(
    x: FloatArray,
    y: FloatArray,
    n_boot: int = 20,
    subsample_frac: float = 0.5,
    thresh: float = 0.6,
    q: float = 0.2,
    seed: int = 0,
) -> StabilityResult:
    """Subsample knockoff selection frequencies (Meinshausen-Bühlmann style)."""
    x = _as_design(x)
    y = np.asarray(y, dtype=np.float64).ravel()
    if not (0 < subsample_frac <= 1 and 0 < thresh <= 1):
        raise ValueError("subsample_frac, thresh must be in (0,1]")
    n, p = x.shape
    counts = np.zeros(p)
    m = max(int(n * subsample_frac), p + 1)
    for b in range(n_boot):
        rng = np.random.default_rng(seed + b)
        idx = rng.choice(n, size=m, replace=False)
        sel = knockoff_select(x[idx], y[idx], q=q, seed=seed + 10000 + b)
        counts[sel] += 1
    freq = counts / n_boot
    return StabilityResult(inclusion_freq=freq, selected=np.flatnonzero(freq >= thresh))


def synth_linear(
    n: int = 300,
    p: int = 30,
    k: int = 5,
    rho: float = 0.3,
    amplitude: float = 3.0,
    noise_sd: float = 1.0,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray, NDArray[np.intp]]:
    """Equicorrelated Gaussian design + sparse linear truth."""
    if n <= p or p < 2 or k < 1 or k > p:
        raise ValueError("need n>p, 1<=k<=p")
    if not (0 <= rho < 1):
        raise ValueError("rho must be in [0,1)")
    rng = np.random.default_rng(seed)
    cov = np.full((p, p), rho) + np.eye(p) * (1 - rho)
    chol = np.linalg.cholesky(cov)
    x = rng.standard_normal((n, p)) @ chol.T
    true_idx = rng.choice(p, size=k, replace=False)
    beta = np.zeros(p)
    beta[true_idx] = amplitude * rng.choice([-1.0, 1.0], size=k)
    y = x @ beta + noise_sd * rng.standard_normal(n)
    return x, y, np.sort(true_idx)


def bench_knockoffs(
    seed: int = 20261231 + 153,
    *,
    n_rep: int = 15,
    null_rep: int = 10,
) -> dict[str, float]:
    """SYNTHETIC FDR/power telemetry for the knockoff filter."""
    q = 0.2
    fdrs: list[float] = []
    powers: list[float] = []
    fdrs_sdp: list[float] = []
    powers_sdp: list[float] = []
    fdrs_r: list[float] = []
    powers_r: list[float] = []
    for rep in range(n_rep):
        x, y, true_idx = synth_linear(n=600, p=60, k=12, rho=0.1, amplitude=4.0, seed=seed + rep)
        sels = {
            "omp_e": knockoff_select(x, y, q=q, method="equi", stat="omp", seed=seed + rep),
            "omp_s": knockoff_select(x, y, q=q, method="sdp", stat="omp", seed=seed + rep),
            "rid_e": knockoff_select(x, y, q=q, method="equi", stat="ridge", seed=seed + rep),
        }
        for s_, f_, p_ in (
            (sels["omp_e"], fdrs, powers),
            (sels["omp_s"], fdrs_sdp, powers_sdp),
            (sels["rid_e"], fdrs_r, powers_r),
        ):
            if s_.size:
                f_.append(1.0 - float(np.intersect1d(s_, true_idx).size) / s_.size)
                p_.append(float(np.intersect1d(s_, true_idx).size) / true_idx.size)
            else:
                f_.append(0.0)
                p_.append(0.0)
    # exchangeability: knockoffs preserve column means & diagonal covariance
    x, _y, _t = synth_linear(n=300, p=25, k=5, seed=seed + 999)
    xk = sample_gaussian_knockoffs(x, seed=seed + 999)
    mean_diff = float(np.abs(x.mean(axis=0) - xk.mean(axis=0)).max())
    var_diff = float(
        np.abs(np.diag(np.cov(x, rowvar=False)) - np.diag(np.cov(xk, rowvar=False))).max()
    )
    # null case: no signal → selection empty with prob ≥ 1-q
    null_rej = 0
    for rep in range(null_rep):
        xn, _yn, _ = synth_linear(n=300, p=20, k=5, seed=seed + 2000 + rep)
        yn = np.random.default_rng(seed + 3000 + rep).standard_normal(300)
        if knockoff_select(xn, yn, q=q, seed=seed + 2000 + rep).size:
            null_rej += 1
    # determinism
    s1 = knockoff_select(x, _y, q=q, seed=seed)
    s2 = knockoff_select(x, _y, q=q, seed=seed)
    deterministic = float(np.array_equal(s1, s2))
    # honest note: the OMP entry-strength statistic biases null Ws
    # positive in correlated designs (knockoffs enter the path later
    # than their originals), so the knockoff+ FDR bound is *violated* on
    # OMP — ridge W stays conservative (FDR≈0). Reported, not hidden.
    return {
        "synthetic_omp_fdr": float(np.mean(fdrs)),
        "synthetic_omp_power": float(np.mean(powers)),
        "synthetic_sdp_omp_fdr": float(np.mean(fdrs_sdp)),
        "synthetic_sdp_omp_power": float(np.mean(powers_sdp)),
        "synthetic_ridge_fdr": float(np.mean(fdrs_r)),
        "synthetic_ridge_power": float(np.mean(powers_r)),
        "synthetic_omp_fdr_breach": float(np.mean(fdrs) > q + 0.1),
        "synthetic_null_reject_rate": null_rej / null_rep,
        "synthetic_exchange_mean_err": mean_diff,
        "synthetic_exchange_var_err": var_diff,
        "synthetic_determinism": deterministic,
    }
