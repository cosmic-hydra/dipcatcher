"""energy_score — multivariate distributional calibration.

Univariate PIT/pinball lanes score each name in isolation; they cannot
see dependence misspecification. Gneiting & Raftery (2007) energy score
and the variogram score (Scheuerer & Hamill 2015) score the *joint*
predictive distribution — the correct lens for cross-asset calibration.

- ES(X, y) = E‖X−y‖ − ½·E‖X−X′‖, proper for the full law.
- VS_p(y, X) = Σᵢⱼ (|yᵢ−yⱼ|^p − E|Xᵢ−Xⱼ|^p)² — emphasizes dependence
  structure via pairwise differences.

Bench couples marginal forecast samples into joint draws through a
Gaussian copula at planted ρ vs misspecified ρ and independence —
measuring that both scores rank the correctly-specified copula first.
All SYNTHETIC; sealed `energy_score.v1`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

ENERGY_SCORE_SCHEMA = "energy_score.v1"


def energy_score(y: NDArray[np.float64], samples: NDArray[np.float64]) -> float:
    """ES(y, X) = E‖X−y‖ − ½E‖X−X′‖. samples: (m, d), y: (d,)."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or y.ndim != 1 or s.shape[1] != y.size:
        raise ValueError("samples must be (m, d) and y (d,)")
    term1 = float(np.linalg.norm(s - y, axis=1).mean())
    diff = s[:, None, :] - s[None, :, :]
    term2 = float(np.linalg.norm(diff, axis=2).mean())
    return term1 - 0.5 * term2


def variogram_score(
    y: NDArray[np.float64],
    samples: NDArray[np.float64],
    *,
    p: float = 0.5,
) -> float:
    """VS_p = Σᵢⱼ (|yᵢ−yⱼ|^p − E|Xᵢ−Xⱼ|^p)²."""
    y = np.asarray(y, dtype=np.float64)
    s = np.asarray(samples, dtype=np.float64)
    if s.ndim != 2 or y.ndim != 1 or s.shape[1] != y.size:
        raise ValueError("samples must be (m, d) and y (d,)")
    dy = np.abs(y[:, None] - y[None, :]) ** p
    ds = np.abs(s[:, :, None] - s[:, None, :]) ** p  # (m, d, d)
    return float(((dy[None] - ds.mean(axis=0)) ** 2).sum())


def gaussian_copula_samples(
    marginals: list[NDArray[np.float64]],
    rho: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Couple per-name marginal samples into joint draws at equicorrelation ρ.

    marginals[j]: (m,) samples for name j. Draws z ~ N(0, corr(ρ)) and
    assigns each marginal's order statistics by the latent rank, so the
    column is a permutation of that marginal. Returns (m, d).
    """
    d = len(marginals)
    m = marginals[0].size
    if any(x.size != m for x in marginals):
        raise ValueError("marginals must share length m")
    if not (-1.0 / (d - 1) < rho <= 1.0):
        raise ValueError(f"equicorrelation rho out of feasible range: {rho}")
    corr = np.full((d, d), rho)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = rng.standard_normal((m, d)) @ chol.T
    out = np.empty((m, d))
    for j, x in enumerate(marginals):
        xs = np.sort(np.asarray(x, dtype=np.float64))
        # Rank-couple so each marginal sample is used once. Inverse-CDF with
        # replacement duplicates some draws and drops others, so the sorted
        # column is not the marginal.
        order = np.argsort(z[:, j], kind="mergesort")
        out[order, j] = xs
    return out


def energy_score_bench(
    *,
    n_reps: int = 40,
    m: int = 256,
    d: int = 8,
    n_names_corr: float = 0.6,
    seed: int = 0,
) -> dict[str, Any]:
    """Correct-rho copula vs misspecified rho vs independence."""
    rng = np.random.default_rng(seed)
    sigma = np.full((d, d), n_names_corr)
    np.fill_diagonal(sigma, 1.0)
    chol_true = np.linalg.cholesky(sigma)
    arms: dict[str, dict[str, list[float]]] = {
        a: {"es": [], "vs": []} for a in ("correct", "misspecified", "independent")
    }
    for _ in range(n_reps):
        y = rng.standard_normal(d) @ chol_true.T
        marginals = [rng.standard_normal(m) for _ in range(d)]
        for arm, rho in (("correct", n_names_corr), ("misspecified", 0.1), ("independent", 0.0)):
            s = gaussian_copula_samples(marginals, rho, rng)
            arms[arm]["es"].append(energy_score(y, s))
            arms[arm]["vs"].append(variogram_score(y, s))

    def stats(xs: list[float]) -> dict[str, float]:
        a = np.asarray(xs)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": float(a.size)}

    out = {
        a: {"energy_score": stats(v["es"]), "variogram_score": stats(v["vs"])}
        for a, v in arms.items()
    }
    correct_es = out["correct"]["energy_score"]["mean"]
    payload: dict[str, Any] = {
        "schema": ENERGY_SCORE_SCHEMA,
        "kind": "energy_score",
        "n_reps": n_reps,
        "m": m,
        "d": d,
        "planted_rho": n_names_corr,
        "arms": out,
        "correct_copula_best_es": bool(
            correct_es < out["misspecified"]["energy_score"]["mean"]
            and correct_es < out["independent"]["energy_score"]["mean"]
        ),
        "correct_copula_best_vs": bool(
            out["correct"]["variogram_score"]["mean"]
            < out["misspecified"]["variogram_score"]["mean"]
            and out["correct"]["variogram_score"]["mean"]
            < out["independent"]["variogram_score"]["mean"]
        ),
        "interpretation": (
            "Joint draws through a Gaussian copula at planted rho vs "
            "misspecified/independence: both ES and VS should rank the "
            "correct copula first — VS especially, since it isolates "
            "dependence structure"
        ),
    }
    payload["git_revision"] = git_revision()
    payload["data_label"] = "SYNTHETIC"
    payload["research_only"] = True
    payload["payload_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload
