"""Parametric VaR/ES estimators beyond historical and Gaussian.

``metrics.risk`` already has historical and Gaussian VaR/ES;
``metrics.extremes.gpd_var_es`` covers the GPD tail.  This module adds the
Cornish–Fisher expansion and Student-t fits, plus a convenience dispatcher.

References:
- Cornish, Fisher (1937) / Fisher & Cornish (1960). Moments and cumulants —
  quantile expansion with skewness/kurtosis corrections.
- Zangari (1996). RiskMetrics modified VaR — the standard CF-VaR usage.
- Baillie, Bollerslev (1992). Prediction in dynamic models with
  time-dependent conditional variances — t VaR usage.
- McNeil, Frey (2000). Estimation of tail-related risk measures for
  heteroscedastic returns — GPD/t ES formulae.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import optimize as opt
from scipy import stats as sstats
from scipy.special import gammaln

from quant_fund.utils.numeric import require_upper_tail_alpha

Array = NDArray[np.float64]


def _as_losses(losses: Array, name: str = "losses") -> Array:
    v = np.asarray(losses, dtype=float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 30:
        raise ValueError(f"{name} must have >= 30 finite observations")
    return v


def _require_alpha(alpha: float) -> float:
    return require_upper_tail_alpha(alpha)


def cornish_fisher_var(losses: Array, alpha: float = 0.95) -> dict[str, float]:
    """Cornish–Fisher (1937) modified VaR for loss observations.

    ``z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36``;
    ``VaR = mu + sigma * z_cf`` where ``S`` is skewness and ``K`` excess
    kurtosis.  Returns the adjusted quantile, z_cf, and inputs.
    """
    v = _as_losses(losses)
    a = _require_alpha(alpha)
    mu = float(v.mean())
    sd = float(v.std(ddof=1))
    if sd <= 0.0:
        raise ValueError("losses must have positive variance")
    sk = float(sstats.skew(v))
    ku = float(sstats.kurtosis(v))  # excess kurtosis
    z = float(sstats.norm.ppf(a))
    z_cf = (
        z
        + (z**2 - 1.0) * sk / 6.0
        + (z**3 - 3.0 * z) * ku / 24.0
        - (2.0 * z**3 - 5.0 * z) * sk**2 / 36.0
    )
    var = mu + sd * z_cf
    return {
        "var": var,
        "z_cf": z_cf,
        "z": z,
        "skew": sk,
        "excess_kurtosis": ku,
        "mu": mu,
        "sigma": sd,
    }


def cornish_fisher_es(losses: Array, alpha: float = 0.95, n_quad: int = 512) -> dict[str, float]:
    """ES under the Cornish–Fisher quantile mapping.

    Integrates the CF quantile function from ``alpha`` to 1 (numeric
    quadrature on the probability axis — CF ES has no closed form).
    """
    v = _as_losses(losses)
    a = _require_alpha(alpha)
    cf = cornish_fisher_var(v, a)
    sk, ku = cf["skew"], cf["excess_kurtosis"]
    ps = np.linspace(a, 1.0 - 1e-6, n_quad)
    zs = sstats.norm.ppf(ps)
    z_cf = (
        zs
        + (zs**2 - 1.0) * sk / 6.0
        + (zs**3 - 3.0 * zs) * ku / 24.0
        - (2.0 * zs**3 - 5.0 * zs) * sk**2 / 36.0
    )
    es = float(cf["mu"] + cf["sigma"] * np.mean(z_cf))
    return {"es": es, "var": cf["var"], "alpha": a}


def _t_neg_log_likelihood(theta: Array, v: Array) -> float:
    """Vectorized negative log-likelihood of a location-scale Student-t (T8).

    This is the exact objective ``fit_student_t`` minimizes, written to avoid
    the per-call ``scipy.stats.t.logpdf`` dispatch overhead that dominated the
    fit.  It is mathematically identical to
    ``-sum(sstats.t.logpdf(v, df=nu, loc=mu, scale=s))`` (agrees to ~1e-13;
    the Nelder-Mead trajectory and ``res.x`` are unchanged), so this is a pure
    speedup with NO change to the fit's convergence tolerance or result.  These
    fits underpin proper scores (CRPS/pinball/QLIKE), so fidelity is preserved
    exactly.  Not a live-trading or P&L claim.
    """
    nu, mu, log_s = theta
    if nu <= 2.01 or log_s < -20.0 or log_s > 5.0:
        return 1e12
    z = (v - mu) / math.exp(log_s)
    base = gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0) - 0.5 * math.log(nu * math.pi)
    logpdf = base - ((nu + 1.0) / 2.0) * np.log1p(z**2 / nu) - log_s
    return -float(np.sum(logpdf))


def fit_student_t(losses: Array) -> dict[str, float]:
    """MLE fit of location-scale Student-t to loss observations.

    Caveat (T8 equivalence proof, Lead ruling 2026-10-08): on near-degenerate
    inputs ``nu`` is statistically UNIDENTIFIED (flat likelihood ridge as
    nu -> inf) — its exact value there is not a stable quantity across
    implementations, and no receipt, report or headline may quote a
    degenerate-cell ``nu`` as an estimate. The proper-score surface
    (mu/sigma/loglik/CRPS/quantiles) is the quotable output. See
    ``tests/unit/metrics/test_fit_student_t_equivalence.py`` — its
    non-identifiability test is the authority for why ``fit_nu`` carries an
    equivalence atol of 5e-5, tighter than the fit's own 1e-4 xatol.
    """
    v = _as_losses(losses)
    x0 = np.array([8.0, float(v.mean()), math.log(float(v.std(ddof=1)))])
    res = opt.minimize(
        lambda theta: _t_neg_log_likelihood(theta, v),
        x0,
        method="Nelder-Mead",
        options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4, "adaptive": False},
    )
    if not res.success and not np.all(np.isfinite(res.x)):
        raise ValueError("Student-t fit failed")
    nu, mu, log_s = res.x
    if not np.isfinite(nu) or nu <= 2.01:
        raise ValueError("Student-t fit produced invalid nu")
    return {"nu": float(nu), "mu": float(mu), "sigma": float(math.exp(log_s))}


def student_t_var_es(losses: Array, alpha: float = 0.95) -> dict[str, float]:
    """VaR/ES under the fitted Student-t (McNeil–Frey 2000 ES formula).

    ``ES = mu + sigma * (f(z)/(1-a)) * ((nu-2)+z^2)/(nu-1)`` with
    ``z = t_nu^{-1}(alpha)``.
    """
    v = _as_losses(losses)
    a = _require_alpha(alpha)
    fit = fit_student_t(v)
    nu, mu, s = fit["nu"], fit["mu"], fit["sigma"]
    z = float(sstats.t.ppf(a, df=nu))
    var = mu + s * z
    fz = float(sstats.t.pdf(z, df=nu))
    es = mu + s * (fz / (1.0 - a)) * ((nu - 2.0) + z**2) / (nu - 1.0)
    return {"var": var, "es": es, **fit, "alpha": a}


def parametric_var_es(
    losses: Array,
    alpha: float = 0.95,
    method: str = "student_t",
) -> dict[str, float | str]:
    """Dispatch to a parametric VaR/ES estimator.

    ``method`` in {"gaussian", "student_t", "cornish_fisher"}; the Gaussian
    branch defers to the closed form (mean + z*sd, ES = mu + sd*phi(z)/(1-a)).
    """
    v = _as_losses(losses)
    a = _require_alpha(alpha)
    m = method.lower()
    if m == "gaussian":
        mu = float(v.mean())
        sd = float(v.std(ddof=1))
        z = float(sstats.norm.ppf(a))
        es_z = float(sstats.norm.pdf(z)) / (1.0 - a)
        return {
            "var": mu + sd * z,
            "es": mu + sd * es_z,
            "method": "gaussian",
            "alpha": a,
        }
    if m == "student_t":
        out: dict[str, float | str] = dict(student_t_var_es(v, a))
        out["method"] = "student_t"
        return out
    if m == "cornish_fisher":
        cf = cornish_fisher_var(v, a)
        es = cornish_fisher_es(v, a)
        return {
            "var": cf["var"],
            "es": es["es"],
            "method": "cornish_fisher",
            "alpha": a,
        }
    raise ValueError(f"unknown method: {method!r}")
