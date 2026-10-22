"""Covariance estimators and PSD repair (SYNTHETIC)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.covariance import OAS, LedoitWolf

from quant_fund.compute.numba_guard import call_kernel
from quant_fund.models.quest import quest_covariance
from quant_fund.utils.logging import get_logger

Array = NDArray[np.float64]
log = get_logger(module="covariance")


def is_symmetric(sigma: Array, tol: float = 1e-10) -> bool:
    return bool(np.max(np.abs(sigma - sigma.T)) <= tol)


def min_eigenvalue(sigma: Array) -> float:
    return float(np.min(np.linalg.eigvalsh(0.5 * (sigma + sigma.T))))


def repair_psd(sigma: Array, tol: float = 1e-10) -> tuple[Array, dict[str, float]]:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    s = 0.5 * (s + s.T)
    eig_min = float(np.min(np.linalg.eigvalsh(s)))
    if eig_min >= -tol and is_symmetric(sigma):
        return s, {"repaired": 0.0, "eig_min_before": eig_min, "eig_min_after": eig_min}
    # Work directly in the symmetric eigensystem.  ``cov_nearest`` can return
    # NaNs for finite covariance-like inputs with negative diagonal entries;
    # clipping eigenvalues is deterministic and guarantees a finite PSD result.
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    clipped = np.maximum(eigenvalues, max(tol, 0.0))
    repaired = (eigenvectors * clipped) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    after = float(np.min(np.linalg.eigvalsh(repaired)))
    fro = float(np.linalg.norm(repaired - s, "fro"))
    log.warning("psd_repair", eig_min_before=eig_min, eig_min_after=after, frobenius=fro)
    return repaired, {
        "repaired": 1.0,
        "eig_min_before": eig_min,
        "eig_min_after": after,
        "frobenius": fro,
    }


def _clean_returns(returns: Array, *, min_rows: int = 2) -> Array:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError("returns must be a non-empty 2D array")
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < min_rows:
        raise ValueError(f"at least {min_rows} finite return rows are required")
    return x


def _validate_lambda(lam: float) -> None:
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and between 0 and 1")


def sample_cov(returns: Array) -> Array:
    """Unbiased listwise sample covariance (``ddof=1``).

    Catalog ``sample`` / named ``optimizer.covariance=sample`` use the
    stamped ``sample`` helper, which repairs this matrix. The default
    Ledoit–Wolf path stays Ledoit–Wolf when ``T<=N`` and does not call
    this helper.
    """
    x = _clean_returns(returns)
    return np.atleast_2d(np.asarray(np.cov(x, rowvar=False, ddof=1), dtype=float))


def sample(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Unbiased sample covariance. Returns trailing \(\Sigma\) and params.

    ``np.cov(..., ddof=1)`` on the listwise-complete trailing window. This
    is not sequential \(H_{t+1}\): interior holes are dropped, not treated
    as adjacent observations, and an incomplete asof row is omitted rather
    than fail-closed. This is not Ledoit–Wolf, OAS, EWMA, or DCC:
    ``sample`` does not call those fitters. When \(T>N\) the estimator
    stays sample rather than silently switching to Ledoit–Wolf. When
    \(T\le N\) it stays sample (singular matrices are eigenvalue-clipped).
    ``optimize_asof`` / ``/risk/portfolio`` use this matrix when
    ``optimizer.covariance=sample`` and apply the GARCH/RGARCH overlay
    (trailing sample has no \(D_{t+1}\)). Params stamp ``family=sample``,
    ``spec=unbiased_sample``, ``covariance_object=trailing``,
    ``sample=listwise_complete``, and ``ddof=1``. The default Ledoit–Wolf
    path stays Ledoit–Wolf when \(T\le N\) rather than calling this
    helper. This does not invent HF RV, implement matrix AG-DCC, or wire
    factor covariance.
    """
    x = _clean_returns(returns)
    sigma = np.atleast_2d(np.asarray(np.cov(x, rowvar=False, ddof=1), dtype=float))
    if not np.isfinite(sigma).all():
        raise ValueError("sample covariance produced non-finite values")
    repaired, _ = repair_psd(sigma)
    t, n = x.shape
    return np.asarray(repaired, dtype=float), {
        "family": OPTIMIZER_COVARIANCE_SAMPLE,
        "spec": SAMPLE_SPEC_UNBIASED,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "ddof": 1.0,
        "n_obs": float(t),
        "n_assets": float(n),
    }


try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover - numba is a required dependency

    def _njit(*_a: object, **_k: object) -> object:  # type: ignore[no-redef]
        def _deco(fn: object) -> object:
            return fn

        if len(_a) == 1 and callable(_a[0]) and not _k:
            return _a[0]
        return _deco


@_njit(cache=True)
def _ewma_cov_nb(x: np.ndarray, lam: float) -> np.ndarray:
    """RiskMetrics recursion. Each element reads only its own previous value."""
    t_n, n = x.shape
    cov = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        shock_i = x[0, i]
        for j in range(n):
            cov[i, j] = shock_i * x[0, j]
    one = 1.0 - lam
    for t in range(1, t_n):
        for i in range(n):
            shock_i = x[t, i]
            for j in range(n):
                cov[i, j] = lam * cov[i, j] + one * shock_i * x[t, j]
    for i in range(n):
        for j in range(i + 1, n):
            mid = 0.5 * (cov[i, j] + cov[j, i])
            cov[i, j] = mid
            cov[j, i] = mid
    return cov


def ewma_cov(returns: Array, lam: float = 0.94, *, min_rows: int = 2) -> Array:
    r"""One-step-ahead RiskMetrics covariance \(H_{t+1}\).

    Sequential EWMA treats consecutive rows as consecutive observations.
    The estimation sample is the trailing contiguous complete-case window
    ending at the last row; holes are not listwise-concatenated, and an
    incomplete terminal row fails closed so advertised \(H_{t+1}\) cannot
    drop asof \(r_t\). The last return enters
    \(H_{t+1}=\lambda H_t+(1-\lambda)r_t r_t'\). This is not in-sample last
    \(H_t\) that omits \(r_t\), and not Ledoit–Wolf/sample. Catalog
    ``ewma`` / named ``optimizer.covariance=ewma`` use this matrix.
    """
    _validate_lambda(lam)
    x = np.ascontiguousarray(
        dcc_trailing_complete_window(returns, min_rows=min_rows), dtype=np.float64
    )
    return np.asarray(call_kernel(_ewma_cov_nb, x, float(lam)), dtype=float)


def ledoit_wolf_cov(returns: Array) -> Array:
    """Ledoit–Wolf 2004 linear shrinkage.

    Catalog ``ledoit_wolf`` / default ``optimizer.covariance=ledoit_wolf``
    use this matrix. The helper returns the repaired trailing matrix;
    ``ledoit_wolf`` adds the family/spec stamp. When ``T<=N`` the
    estimator stays Ledoit–Wolf rather than switching to sample.
    """
    sigma, _params = ledoit_wolf(returns)
    return sigma


def oracle_approximating_shrinkage_cov(returns: Array) -> Array:
    """Estimate covariance with finite-sample Oracle Approximating Shrinkage.

    OAS is Chen–Wiesel–Eldar–Hero (2010) linear shrinkage, not Ledoit–Wolf
    2004 and not nonlinear spectral shrinkage. Catalog ``oas`` / named
    ``optimizer.covariance=oas`` use this matrix. The helper returns the
    repaired trailing matrix; ``oas`` adds the family/spec stamp.
    """
    sigma, _params = oas(returns)
    return sigma


def oas(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Chen–Wiesel–Eldar–Hero OAS. Returns trailing \(\Sigma\) and params.

    sklearn ``OAS`` shrinks the listwise-complete sample covariance toward
    \(\mu I\). This is not sequential \(H_{t+1}\): interior holes are
    dropped, not treated as adjacent observations, and an incomplete asof
    row is omitted rather than fail-closed. This is not Ledoit–Wolf,
    sample, EWMA, or DCC: ``oas`` does not call those fitters. When
    \(T\le N\) the estimator stays OAS (the high-dimensional formula), not
    a silent sample covariance. ``optimize_asof`` / ``/risk/portfolio`` use
    this matrix when ``optimizer.covariance=oas`` and apply the
    GARCH/RGARCH overlay (trailing shrinkage has no \(D_{t+1}\)). Params
    stamp ``family=oas``, ``spec=chen_wiesel_eldar_hero_2010``,
    ``covariance_object=trailing``, and ``sample=listwise_complete``. This
    does not invent HF RV or implement matrix AG-DCC.
    """
    x = _clean_returns(returns)
    model = OAS().fit(x)
    sigma = np.asarray(model.covariance_, dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("OAS covariance produced non-finite values")
    repaired, _ = repair_psd(sigma)
    shrinkage = float(getattr(model, "shrinkage_", float("nan")))
    t, n = x.shape
    return np.asarray(repaired, dtype=float), {
        "family": OPTIMIZER_COVARIANCE_OAS,
        "spec": OAS_SPEC_CHEN_2010,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "shrinkage": shrinkage,
        "n_obs": float(t),
        "n_assets": float(n),
    }


def ledoit_wolf(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Ledoit–Wolf 2004 linear shrinkage. Returns trailing \(\Sigma\) and params.

    sklearn ``LedoitWolf`` shrinks the listwise-complete sample covariance
    toward \(\mu I\). This is the 2004 linear formula, not nonlinear
    spectral shrinkage. Named ``ledoit_wolf_nonlinear`` is the 2020
    analytical formula and must not be run by this function. This is not
    sequential \(H_{t+1}\): interior holes are dropped, not treated as
    adjacent observations, and an incomplete asof row is omitted rather
    than fail-closed. This is not OAS, sample, EWMA, or DCC:
    ``ledoit_wolf`` does not call those fitters. When \(T\le N\) the
    estimator stays Ledoit–Wolf (the high-dimensional 2004 formula), not
    a silent sample covariance. ``optimize_asof`` / ``/risk/portfolio``
    use this matrix when ``optimizer.covariance=ledoit_wolf`` (the
    default) and apply the GARCH/RGARCH overlay (trailing shrinkage has
    no \(D_{t+1}\)). Params stamp ``family=ledoit_wolf``,
    ``spec=ledoit_wolf_2004_linear``, ``covariance_object=trailing``, and
    ``sample=listwise_complete``. This does not invent HF RV or wire
    factor covariance.
    """
    x = _clean_returns(returns)
    model = LedoitWolf().fit(x)
    sigma = np.asarray(model.covariance_, dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError("Ledoit-Wolf covariance produced non-finite values")
    repaired, _ = repair_psd(sigma)
    shrinkage = float(getattr(model, "shrinkage_", float("nan")))
    t, n = x.shape
    return np.asarray(repaired, dtype=float), {
        "family": OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
        "spec": OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "shrinkage": shrinkage,
        "n_obs": float(t),
        "n_assets": float(n),
    }


def _analytical_nonlinear_shrinkage(centered: Array, n_eff: int) -> Array:
    r"""Ledoit–Wolf (2020) analytical nonlinear eigenvalue map.

    Sample covariance uses \(1/n_{\mathrm{eff}}\) on the centered matrix,
    not unbiased ``ddof=1``. Bandwidth is \(n_{\mathrm{eff}}^{-1/3}\).
    When \(p>n_{\mathrm{eff}}\) the null eigenvalues use the singular-case
    formula rather than silently falling back to 2004 linear shrinkage.
    """
    if n_eff < NLSHRINK_MIN_EFF_OBS:
        raise ValueError(
            f"analytical nonlinear shrinkage requires effective sample size "
            f">= {NLSHRINK_MIN_EFF_OBS}"
        )
    values = np.asarray(centered, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("analytical nonlinear shrinkage requires a non-empty 2D array")
    _t, p = values.shape
    sample = (values.T @ values) / float(n_eff)
    sample = 0.5 * (sample + sample.T)
    if not np.isfinite(sample).all():
        raise ValueError("analytical nonlinear shrinkage sample covariance is non-finite")
    eigenvalues, eigenvectors = np.linalg.eigh(sample)
    start = max(0, p - n_eff)
    lam = np.asarray(eigenvalues[start:], dtype=float)
    rank = int(min(p, n_eff))
    if lam.size != rank or not np.isfinite(lam).all() or float(np.min(lam)) <= 0.0:
        raise ValueError("analytical nonlinear shrinkage sample covariance is singular")
    lam_sum = float(np.sum(lam))
    if not np.isfinite(lam_sum) or lam_sum <= 0.0 or float(np.min(lam / lam_sum)) < 1e-8:
        raise ValueError("analytical nonlinear shrinkage sample covariance is singular")
    lam_col = lam.reshape(rank, 1)
    lam_row = lam.reshape(1, rank)
    h = float(n_eff) ** (-1.0 / 3.0)
    scale = h * lam_row
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (lam_col - lam_row) / scale
        ftilde = (3.0 / 4.0 / np.sqrt(5.0)) * np.mean(
            np.maximum(1.0 - (x * x) / 5.0, 0.0) / scale, axis=1
        )
        log_term = np.log(np.abs((np.sqrt(5.0) - x) / (np.sqrt(5.0) + x)))
        hilbert = (-3.0 / 10.0 / np.pi) * x + (3.0 / 4.0 / np.sqrt(5.0) / np.pi) * (
            1.0 - (x * x) / 5.0
        ) * log_term
    edge = np.isclose(np.abs(x), np.sqrt(5.0), rtol=0.0, atol=1e-12)
    hilbert = np.where(edge, (-3.0 / 10.0 / np.pi) * x, hilbert)
    hftilde = np.mean(hilbert / scale, axis=1)
    concentration = p / float(n_eff)
    if p <= n_eff:
        dtilde = lam / (
            (np.pi * concentration * lam * ftilde) ** 2
            + (1.0 - concentration - np.pi * concentration * lam * hftilde) ** 2
        )
    else:
        hftilde0 = (
            (1.0 / np.pi)
            * (
                3.0 / 10.0 / (h * h)
                + 3.0
                / 4.0
                / np.sqrt(5.0)
                / h
                * (1.0 - 1.0 / 5.0 / (h * h))
                * np.log((1.0 + np.sqrt(5.0) * h) / (1.0 - np.sqrt(5.0) * h))
            )
            * float(np.mean(1.0 / lam))
        )
        dtilde0 = 1.0 / (np.pi * (p - n_eff) / float(n_eff) * hftilde0)
        dtilde1 = lam / (np.pi**2 * lam**2 * (ftilde**2 + hftilde**2))
        dtilde = np.concatenate([np.full(p - n_eff, dtilde0, dtype=float), dtilde1])
    shrunk = np.asarray(dtilde, dtype=float).reshape(-1)
    if shrunk.size != p or not np.isfinite(shrunk).all() or bool(np.any(shrunk <= 0.0)):
        raise ValueError("analytical nonlinear shrinkage produced non-finite eigenvalues")
    sigma = (eigenvectors * shrunk) @ eigenvectors.T
    return np.asarray(0.5 * (sigma + sigma.T), dtype=float)


def ledoit_wolf_nonlinear_cov(returns: Array) -> Array:
    """Analytical nonlinear Ledoit–Wolf 2020 shrinkage.

    Catalog ``ledoit_wolf_nonlinear`` / named
    ``optimizer.covariance=ledoit_wolf_nonlinear`` use this matrix. The
    helper returns the repaired trailing matrix; ``ledoit_wolf_nonlinear``
    adds the family/spec stamp. This is not 2004 linear shrinkage.
    """
    sigma, _params = ledoit_wolf_nonlinear(returns)
    return sigma


def ledoit_wolf_nonlinear(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Analytical nonlinear Ledoit–Wolf. Returns trailing \(\Sigma\) and params.

    Ledoit–Wolf (2020) closed-form spectral shrinkage on the
    listwise-complete trailing window. This is the analytical successor
    to QuEST, not numerical QuEST inversion (2017) and not 2004 linear
    shrinkage toward \(\mu I\). ``ledoit_wolf_nonlinear`` does not call
    ``ledoit_wolf``, OAS, sample, EWMA, or DCC. When \(T\le N\) the
    estimator stays on the singular-case analytical map rather than
    switching to sample or 2004 linear shrinkage. Interior holes are
    dropped; an incomplete asof row is omitted rather than fail-closed.
    Effective sample size after demeaning must be at least 12.
    ``optimize_asof`` / ``/risk/portfolio`` use this matrix when
    ``optimizer.covariance=ledoit_wolf_nonlinear`` and apply the
    GARCH/RGARCH overlay (trailing shrinkage has no \(D_{t+1}\)). Params
    stamp ``family=ledoit_wolf_nonlinear``,
    ``spec=ledoit_wolf_2020_analytical``, ``covariance_object=trailing``,
    and ``sample=listwise_complete``. Default ``ledoit_wolf`` stays 2004.
    Generic ``shrinkage`` / ``ledoit_wolf_2017`` stay unknown. This does
    not invent HF RV or wire factor covariance.
    """
    x = _clean_returns(returns, min_rows=NLSHRINK_MIN_OBS)
    centered = x - x.mean(axis=0)
    n_eff = int(x.shape[0] - 1)
    sigma = _analytical_nonlinear_shrinkage(centered, n_eff)
    if not np.isfinite(sigma).all():
        raise ValueError("analytical nonlinear shrinkage produced non-finite values")
    repaired, _ = repair_psd(sigma)
    t, n = x.shape
    return np.asarray(repaired, dtype=float), {
        "family": OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        "spec": OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "demean": "true",
        "n_obs": float(t),
        "n_eff": float(n_eff),
        "n_assets": float(n),
        "concentration": float(n) / float(n_eff),
        "bandwidth": float(n_eff) ** (-1.0 / 3.0),
    }


def ledoit_wolf_quest_cov(returns: Array) -> Array:
    """Numerical QuEST (Ledoit–Wolf 2015/2017) nonlinear shrinkage matrix.

    Catalog ``ledoit_wolf_quest`` / named
    ``optimizer.covariance=ledoit_wolf_quest`` use this matrix. The helper
    returns the repaired trailing matrix; ``ledoit_wolf_quest`` adds the
    family/spec stamp. This is numerical QuEST inversion (2017), not the
    analytical 2020 closed-form and not 2004 linear shrinkage.
    """
    sigma, _params = ledoit_wolf_quest(returns)
    return sigma


def ledoit_wolf_quest(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Numerical QuEST nonlinear Ledoit–Wolf. Returns trailing \(\Sigma\).

    Ledoit–Wolf (2015) spectrum estimation via numerical inversion of the
    QuEST function (Ledoit–Wolf 2017), applied to the listwise-complete
    trailing window. The population eigenvalue spectrum is estimated by
    nonlinear least-squares inversion of the discretized
    Marcenko–Pastur equation (warm-started from 2004 linear shrinkage,
    analytic Jacobian, bounded L-BFGS-B), then the oracle shrinkage map
    \(x / |1 - c\,\breve m(x)|^2\) is integrated over the implied sample
    law. This is the numerical QuEST estimator, not the LW-2020
    analytical map (``ledoit_wolf_nonlinear``) and not 2004 linear
    shrinkage (``ledoit_wolf``); ``ledoit_wolf_quest`` does not call
    those fitters, OAS, sample, EWMA, or DCC. When \(T\le N\) the
    singular-case null-space adjustment stays active rather than
    switching estimators. Interior holes are dropped; an incomplete asof
    row is omitted rather than fail-closed. Effective sample size after
    demeaning must be at least 12. ``optimize_asof`` /
    ``/risk/portfolio`` use this matrix when
    ``optimizer.covariance=ledoit_wolf_quest`` (or the ``quest`` /
    ``ledoit_wolf_2017`` aliases) and apply the GARCH/RGARCH overlay —
    trailing shrinkage has no \(D_{t+1}\). Params stamp
    ``family=ledoit_wolf_quest``, ``spec=ledoit_wolf_2017_quest``,
    ``covariance_object=trailing``, ``sample=listwise_complete``, and the
    inversion diagnostics. This does not invent HF RV or wire factor
    covariance.
    """
    x = _clean_returns(returns, min_rows=NLSHRINK_MIN_OBS)
    centered = x - x.mean(axis=0)
    n_eff = int(x.shape[0] - 1)
    if n_eff < NLSHRINK_MIN_EFF_OBS:
        raise ValueError(
            f"quest nonlinear shrinkage requires effective sample size >= {NLSHRINK_MIN_EFF_OBS}"
        )
    sigma, info = quest_covariance(centered, n_eff)
    repaired, _ = repair_psd(sigma)
    t, n = x.shape
    params: dict[str, float | str] = {
        "family": OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST,
        "spec": OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_QUEST,
        "covariance_object": OPTIMIZER_COVARIANCE_OBJECT_TRAILING,
        "sample": OAS_SAMPLE_LISTWISE,
        "demean": "true",
        "n_obs": float(t),
        "n_eff": float(n_eff),
        "n_assets": float(n),
        "concentration": float(n) / float(n_eff),
        "quest_numint": str(info["quest_numint"]),
        "quest_objective": float(info["quest_objective"]),
        "quest_iterations": float(info["quest_iterations"]),
    }
    return np.asarray(repaired, dtype=float), params


def factor_cov(betas: Array, factor_cov: Array, idio_var: Array) -> Array:
    b = np.asarray(betas, dtype=float)
    f = np.asarray(factor_cov, dtype=float)
    dvar = np.asarray(idio_var, dtype=float)
    if b.ndim != 2 or f.shape != (b.shape[1], b.shape[1]) or dvar.shape != (b.shape[0],):
        raise ValueError("factor covariance inputs have incompatible shapes")
    if not np.isfinite(b).all() or not np.isfinite(f).all() or not np.isfinite(dvar).all():
        raise ValueError("factor covariance inputs must be finite")
    if not np.allclose(f, f.T) or np.any(dvar < 0):
        raise ValueError("factor covariance must be symmetric and idio_var non-negative")
    if float(np.min(np.linalg.eigvalsh(f))) < -1e-10:
        raise ValueError("factor covariance must be positive semidefinite")
    d = np.diag(dvar)
    return np.asarray(b @ f @ b.T + d, dtype=np.float64)


DCC_STAGE1_MIN_OBS = 50
DCC_STAGE1_SCOPE = "dcc_stage1_univariate"
DCC_FAMILY_GAUSSIAN = "dcc_gaussian"
DCC_FAMILY_STUDENT_T = "dcc_student_t"
DCC_FAMILY_ADCC = "adcc"
DCC_FAMILY_AGDCC = "agdcc"
DCC_FAMILY_AGDCC_FULL = "agdcc_full"
DCC_FAMILY_CCC = "ccc"
DCC_SPEC_ENGLE_2002 = "engle_2002_gaussian_dcc"
DCC_SPEC_STUDENT_T = "engle_2002_student_t_dcc"
DCC_SPEC_ADCC = "cappiello_engle_sheppard_2006"
DCC_SPEC_AGDCC = "cappiello_engle_sheppard_2006_diagonal_agdcc"
DCC_SPEC_AGDCC_FULL = "cappiello_engle_sheppard_2006_full_agdcc"
DCC_SPEC_CCC = "bollerslev_1990_ccc"
DCC_PARAMETERIZATION_DIAGONAL = "diagonal"
DCC_PARAMETERIZATION_FULL = "full"
DCC_DIST_NORMAL = "normal"
DCC_DIST_STUDENT_T = "student_t"
DCC_COVARIANCE_OBJECT_ONE_STEP = "one_step_ahead"
DCC_NU_MIN = 2.05
DCC_NU_MAX = 50.0
IMPLEMENTED_COVARIANCE_SPECS = (
    "sample",
    "ewma",
    "ledoit_wolf",
    "oas",
    "ledoit_wolf_nonlinear",
    "ledoit_wolf_quest",
    "factor",
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    DCC_FAMILY_CCC,
)
UNSPECIFIED_COVARIANCE_SPECS: tuple[str, ...] = ()
OPTIMIZER_COVARIANCE_LEDOIT_WOLF = "ledoit_wolf"
OPTIMIZER_COVARIANCE_SAMPLE = "sample"
OPTIMIZER_COVARIANCE_EWMA = "ewma"
OPTIMIZER_COVARIANCE_OAS = "oas"
OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR = "ledoit_wolf_nonlinear"
OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST = "ledoit_wolf_quest"
OPTIMIZER_COVARIANCE_HOMOSKEDASTIC_PROXY = "homoskedastic_proxy"
OPTIMIZER_COVARIANCE_OBJECT_TRAILING = "trailing"
OPTIMIZER_COVARIANCE_OBJECT_DIAGONAL_PROXY = "diagonal_proxy"
OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF = "ledoit_wolf_2004_linear"
OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR = "ledoit_wolf_2020_analytical"
OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_QUEST = "ledoit_wolf_2017_quest"
SAMPLE_SPEC_UNBIASED = "unbiased_sample"
OPTIMIZER_COVARIANCE_SPEC_SAMPLE = SAMPLE_SPEC_UNBIASED
OPTIMIZER_COVARIANCE_SPEC_DIAGONAL_PROXY = "diagonal_2pct"
EWMA_SPEC_RISKMETRICS = "jpmorgan_riskmetrics_1996"
EWMA_MIN_OBS = 20
EWMA_DEFAULT_LAM = 0.94
OAS_SPEC_CHEN_2010 = "chen_wiesel_eldar_hero_2010"
OAS_SAMPLE_LISTWISE = "listwise_complete"
NLSHRINK_MIN_EFF_OBS = 12
NLSHRINK_MIN_OBS = NLSHRINK_MIN_EFF_OBS + 1
IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS = (
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF,
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
    OPTIMIZER_COVARIANCE_EWMA,
    OPTIMIZER_COVARIANCE_OAS,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
    OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST,
    OPTIMIZER_COVARIANCE_SAMPLE,
)
IMPLEMENTED_OPTIMIZER_DCC_FAMILIES = (
    DCC_FAMILY_GAUSSIAN,
    DCC_FAMILY_STUDENT_T,
    DCC_FAMILY_ADCC,
    DCC_FAMILY_CCC,
    DCC_FAMILY_AGDCC,
    DCC_FAMILY_AGDCC_FULL,
)
IMPLEMENTED_OPTIMIZER_ONE_STEP_SPECS = (
    *IMPLEMENTED_OPTIMIZER_DCC_FAMILIES,
    OPTIMIZER_COVARIANCE_EWMA,
)
IMPLEMENTED_OPTIMIZER_NAMED_SPECS = tuple(
    spec
    for spec in IMPLEMENTED_OPTIMIZER_COVARIANCE_SPECS
    if spec != OPTIMIZER_COVARIANCE_LEDOIT_WOLF
)
UNWIRED_OPTIMIZER_COVARIANCE_SPECS = ("factor",)
DCC_SAMPLE_TRAILING_COMPLETE = "trailing_complete_window"
_DCC_STUDENT_T_ALIASES = frozenset({DCC_FAMILY_STUDENT_T, "studentt", "student_t", "std", "t"})
_DCC_ADCC_ALIASES = frozenset(
    {
        DCC_FAMILY_ADCC,
        "dcc_asymmetric",
        "asymmetric_dcc",
        "a-dcc",
        "a_dcc",
        DCC_SPEC_ADCC,
        "ces_2006",
        "cappiello_engle_sheppard",
    }
)
_DCC_CCC_ALIASES = frozenset(
    {
        DCC_FAMILY_CCC,
        "constant_conditional_correlation",
        "constant-conditional-correlation",
        "constant_correlation",
        DCC_SPEC_CCC,
        "bollerslev_1990",
        "bollerslev_1990_ccc",
    }
)
_DCC_AGDCC_ALIASES = frozenset(
    {
        DCC_FAMILY_AGDCC,
        "ag_dcc",
        "ag-dcc",
        "diagonal_agdcc",
        "diagonal_ag_dcc",
        "diagonal-ag-dcc",
        DCC_SPEC_AGDCC,
        "cappiello_engle_sheppard_2006_agdcc",
        "ces_2006_agdcc",
        "ces_agdcc",
        "asymmetric_generalized_dcc",
        "asymmetric-generalized-dcc",
    }
)
_DCC_AGDCC_FULL_ALIASES = frozenset(
    {
        DCC_FAMILY_AGDCC_FULL,
        "agdcc_unrestricted",
        "full_agdcc",
        "full_ag_dcc",
        "unrestricted_agdcc",
        "unrestricted_ag_dcc",
        "matrix_agdcc_full",
        DCC_SPEC_AGDCC_FULL,
        "cappiello_engle_sheppard_2006_full_agdcc",
        "ces_2006_full_agdcc",
        "ces_agdcc_full",
    }
)
_OPTIMIZER_AGDCC_ALIASES = frozenset(alias.replace("-", "_") for alias in _DCC_AGDCC_ALIASES)
_OPTIMIZER_AGDCC_FULL_ALIASES = frozenset(
    alias.replace("-", "_") for alias in _DCC_AGDCC_FULL_ALIASES
)
_OPTIMIZER_CCC_ALIASES = frozenset(alias.replace("-", "_") for alias in _DCC_CCC_ALIASES)
_AMBIGUOUS_OPTIMIZER_GAUSSIAN_ALIASES = frozenset({"gaussian", "normal"})
_AMBIGUOUS_OPTIMIZER_STUDENT_T_ALIASES = frozenset({"t", "student_t", "studentt", "std"})
_OPTIMIZER_EWMA_ALIASES = frozenset(
    {
        OPTIMIZER_COVARIANCE_EWMA,
        "riskmetrics",
        "risk_metrics",
        EWMA_SPEC_RISKMETRICS,
        "riskmetrics_1996",
    }
)
_OPTIMIZER_OAS_ALIASES = frozenset(
    {
        OPTIMIZER_COVARIANCE_OAS,
        "oracle_approximating_shrinkage",
        "oracle_approximating",
        OAS_SPEC_CHEN_2010,
        "chen_2010",
    }
)
_OPTIMIZER_SAMPLE_ALIASES = frozenset(
    {
        OPTIMIZER_COVARIANCE_SAMPLE,
        "sample_cov",
        SAMPLE_SPEC_UNBIASED,
        "sample_covariance",
    }
)
_OPTIMIZER_LEDOIT_WOLF_NONLINEAR_ALIASES = frozenset(
    {
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR,
        "nlshrink",
        "nonlinear_ledoit_wolf",
        "ledoitwolf_nonlinear",
        "lw_nonlinear",
        "analytical_nonlinear_shrinkage",
        "ledoit_wolf_2020",
        OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_NONLINEAR,
    }
)
_QUEST_OPTIMIZER_ALIASES = frozenset(
    {
        OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST,
        "ledoit_wolf_2017",
        "ledoitwolf_2017",
        "lw_2017",
        "quest",
        "nonlinear_shrinkage_2017",
        OPTIMIZER_COVARIANCE_SPEC_LEDOIT_WOLF_QUEST,
        "numerical_quest",
        "quest_2017",
    }
)
_OPTIMIZER_ADCC_ALIASES = frozenset(
    {
        "dcc_asymmetric",
        "asymmetric_dcc",
        "a_dcc",
        DCC_SPEC_ADCC,
        "ces_2006",
        "cappiello_engle_sheppard",
    }
)


def dcc_trailing_complete_window(returns: Array, *, min_rows: int = DCC_STAGE1_MIN_OBS) -> Array:
    r"""Keep the trailing contiguous complete-case block ending at the last row.

    DCC QML, scalar ADCC, diagonal AG-DCC, unrestricted AG-DCC,
    Bollerslev CCC, and RiskMetrics EWMA treat consecutive rows as
    consecutive observations. Listwise
    deletion that concatenates across holes would silently treat
    non-adjacent days as adjacent, and dropping a missing terminal row
    would make \(H_{t+1}\) one-step from an earlier date than advertised.
    An incomplete last row fails closed. Rows before the last interior
    hole are unused, not concatenated. Ledoit–Wolf 2004 / nonlinear 2020
    / sample / OAS still listwise-delete because they are not sequential
    likelihoods.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("returns must be a non-empty 2D array")
    if not np.isfinite(x[-1]).all():
        raise ValueError("incomplete_terminal_row; refusing to drop asof z_t from H_{t+1}")
    complete = np.isfinite(x).all(axis=1)
    start = 0
    for index in range(x.shape[0] - 2, -1, -1):
        if not complete[index]:
            start = index + 1
            break
    window = np.array(x[start:], dtype=float, copy=True)
    if window.shape[0] < min_rows:
        raise ValueError(
            f"insufficient_contiguous_rows:{window.shape[0]}; "
            f"at least {min_rows} contiguous complete return rows are required"
        )
    return window


def require_implemented_dcc_spec(family: str) -> str:
    """Resolve a DCC family name.

    Gaussian DCC, Student-t DCC, Cappiello–Engle–Sheppard scalar ADCC,
    diagonal AG-DCC, unrestricted full-matrix AG-DCC, and Bollerslev (1990)
    CCC are implemented catalog estimators. Generic ``dcc`` stays unknown
    so it cannot be read as any of those specs. Unrestricted ``agdcc_full``
    is not diagonal ``agdcc``. Scalar ``adcc`` is not diagonal ``agdcc``.
    """
    if not isinstance(family, str) or isinstance(family, bool) or not family.strip():
        raise ValueError("DCC family must be a non-empty string")
    key = family.strip().lower()
    if key in {DCC_FAMILY_GAUSSIAN, "gaussian", "normal", "engle_2002"}:
        return DCC_FAMILY_GAUSSIAN
    if key in _DCC_STUDENT_T_ALIASES:
        return DCC_FAMILY_STUDENT_T
    if key in _DCC_AGDCC_FULL_ALIASES:
        return DCC_FAMILY_AGDCC_FULL
    if key in _DCC_AGDCC_ALIASES:
        return DCC_FAMILY_AGDCC
    if key in _DCC_ADCC_ALIASES:
        return DCC_FAMILY_ADCC
    if key in _DCC_CCC_ALIASES:
        return DCC_FAMILY_CCC
    raise ValueError(f"unknown_dcc_spec:{family.strip()}")


def require_implemented_optimizer_covariance(name: str) -> str:
    r"""Resolve the named ``optimize_asof`` / ``/risk/portfolio`` covariance path.

    Default remains Ledoit–Wolf. When \(T\le N\) that default stays
    Ledoit–Wolf rather than silently switching to sample. Gaussian DCC,
    Student-t DCC, scalar Cappiello–Engle–Sheppard ADCC, Bollerslev CCC,
    diagonal CES AG-DCC, unrestricted CES AG-DCC, RiskMetrics EWMA, Chen
    OAS, analytical nonlinear Ledoit–Wolf, and unbiased sample are
    explicit named paths, not silent replacements for each other or for
    Ledoit–Wolf 2004. Estimators that exist in the covariance catalog but
    are not wired into the optimizer (factor) fail closed rather than
    running Ledoit–Wolf or inventing factor returns. Ambiguous aliases
    such as ``gaussian`` / ``normal`` / ``t`` / ``student_t`` are rejected
    so they cannot be read as either a return law or Engle DCC. Generic
    ``dcc`` stays unknown so it cannot be read as any of the named DCC
    families, CCC, diagonal AG-DCC, or unrestricted AG-DCC. Generic
    ``shrinkage`` stays unknown so it cannot be read as OAS, 2004 linear
    Ledoit–Wolf, or 2020 analytical nonlinear shrinkage. Named
    ``ledoit_wolf_2017`` / ``quest`` resolve to the real numerical QuEST
    inversion estimator, which must not masquerade as the analytical 2020
    shrinker or vice versa. Named ``agdcc`` is diagonal CES
    AG-DCC; named ``agdcc_full`` is unrestricted CES AG-DCC and must not
    silently size as diagonal AG-DCC. Scalar CES ADCC is not diagonal
    AG-DCC.
    """
    named_paths = (
        f"{OPTIMIZER_COVARIANCE_LEDOIT_WOLF}, {DCC_FAMILY_GAUSSIAN}, "
        f"{DCC_FAMILY_STUDENT_T}, {DCC_FAMILY_ADCC}, {DCC_FAMILY_CCC}, "
        f"{DCC_FAMILY_AGDCC}, {DCC_FAMILY_AGDCC_FULL}, "
        f"{OPTIMIZER_COVARIANCE_EWMA}, "
        f"{OPTIMIZER_COVARIANCE_OAS}, "
        f"{OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR}, "
        f"{OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST}, or "
        f"{OPTIMIZER_COVARIANCE_SAMPLE}"
    )
    if not isinstance(name, str) or isinstance(name, bool) or not name.strip():
        raise ValueError("optimizer covariance must be a non-empty string")
    key = name.strip().lower().replace("-", "_")
    if key in {OPTIMIZER_COVARIANCE_LEDOIT_WOLF, "lw", "ledoitwolf"}:
        return OPTIMIZER_COVARIANCE_LEDOIT_WOLF
    if key in UNWIRED_OPTIMIZER_COVARIANCE_SPECS:
        raise ValueError(
            f"unwired_optimizer_covariance:{key}; named optimize_asof path is {named_paths}"
        )
    if key in _AMBIGUOUS_OPTIMIZER_GAUSSIAN_ALIASES:
        raise ValueError(
            f"unknown_optimizer_covariance:{name.strip()}; name {DCC_FAMILY_GAUSSIAN} explicitly"
        )
    if key in _AMBIGUOUS_OPTIMIZER_STUDENT_T_ALIASES:
        raise ValueError(
            f"unknown_optimizer_covariance:{name.strip()}; name {DCC_FAMILY_STUDENT_T} explicitly"
        )
    if key == "dcc":
        raise ValueError(
            "unknown_dcc_spec:dcc; name dcc_gaussian, dcc_student_t, adcc, "
            "ccc, agdcc, or agdcc_full explicitly"
        )
    if key in {DCC_FAMILY_GAUSSIAN, "engle_2002", DCC_SPEC_ENGLE_2002}:
        return DCC_FAMILY_GAUSSIAN
    if key in {DCC_FAMILY_STUDENT_T, DCC_SPEC_STUDENT_T}:
        return DCC_FAMILY_STUDENT_T
    if key in {DCC_FAMILY_ADCC, DCC_SPEC_ADCC} or key in _OPTIMIZER_ADCC_ALIASES:
        return DCC_FAMILY_ADCC
    if key in _OPTIMIZER_CCC_ALIASES:
        return DCC_FAMILY_CCC
    if key in _OPTIMIZER_AGDCC_FULL_ALIASES:
        return DCC_FAMILY_AGDCC_FULL
    if key in _OPTIMIZER_AGDCC_ALIASES:
        return DCC_FAMILY_AGDCC
    if key in _OPTIMIZER_EWMA_ALIASES:
        return OPTIMIZER_COVARIANCE_EWMA
    if key in _OPTIMIZER_OAS_ALIASES:
        return OPTIMIZER_COVARIANCE_OAS
    if key in _OPTIMIZER_LEDOIT_WOLF_NONLINEAR_ALIASES:
        return OPTIMIZER_COVARIANCE_LEDOIT_WOLF_NONLINEAR
    if key in _QUEST_OPTIMIZER_ALIASES:
        return OPTIMIZER_COVARIANCE_LEDOIT_WOLF_QUEST
    if key in _OPTIMIZER_SAMPLE_ALIASES:
        return OPTIMIZER_COVARIANCE_SAMPLE
    resolved = require_implemented_dcc_spec(name)
    if resolved in UNWIRED_OPTIMIZER_COVARIANCE_SPECS:
        raise ValueError(
            f"unwired_optimizer_covariance:{resolved}; named optimize_asof path is {named_paths}"
        )
    return resolved


def _dcc_prepare_window(returns: Array) -> tuple[Array, float]:
    raw = np.asarray(returns, dtype=float)
    x = dcc_trailing_complete_window(raw, min_rows=DCC_STAGE1_MIN_OBS)
    n_prefix_dropped = float(raw.shape[0] - x.shape[0]) if raw.ndim == 2 else 0.0
    return x, n_prefix_dropped


def _dcc_qbar(z: Array) -> Array:
    with np.errstate(divide="ignore", invalid="ignore"):
        qbar = np.asarray(np.corrcoef(z, rowvar=False), dtype=float)
    if not np.isfinite(qbar).all():
        raise ValueError("DCC correlation target is non-finite; input has unusable variance")
    np.fill_diagonal(qbar, 1.0)
    qbar, _ = repair_psd(qbar, tol=1e-12)
    return qbar


def _dcc_r_from_q(q: Array) -> Array:
    d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
    r = q / np.outer(d, d)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    return np.asarray(r, dtype=float)


def _dcc_step_q(q: Array, qbar: Array, z_prev: Array, a: float, b: float) -> Array:
    return np.asarray((1.0 - a - b) * qbar + a * np.outer(z_prev, z_prev) + b * q, dtype=float)


def _dcc_one_step_h(z: Array, qbar: Array, a: float, b: float, sigma_one_step: Array) -> Array:
    q = qbar.copy()
    t = int(z.shape[0])
    for i in range(1, t + 1):
        q = _dcc_step_q(q, qbar, z[i - 1], a, b)
    r = _dcc_r_from_q(q)
    d_next = np.diag(sigma_one_step)
    repaired, _ = repair_psd(np.asarray(d_next @ r @ d_next, dtype=float))
    return np.asarray(repaired, dtype=float)


def _adcc_negative_shocks(z: Array) -> Array:
    r"""Cappiello–Engle–Sheppard \(n_t = I[z_t < 0] \odot z_t\)."""
    residual = np.asarray(z, dtype=float)
    return np.where(residual < 0.0, residual, 0.0)


def _adcc_nbar(n_shock: Array) -> Array:
    shocks = np.asarray(n_shock, dtype=float)
    if shocks.ndim != 2 or shocks.shape[0] == 0 or shocks.shape[1] == 0:
        raise ValueError("ADCC negative-shock sample must be a non-empty 2D array")
    nbar = (shocks.T @ shocks) / float(shocks.shape[0])
    nbar = 0.5 * (nbar + nbar.T)
    nbar, _ = repair_psd(np.asarray(nbar, dtype=float), tol=1e-12)
    return np.asarray(nbar, dtype=float)


def _adcc_kappa(qbar: Array, nbar: Array) -> float:
    r"""Largest eigenvalue of \(\bar Q^{-1/2}\bar N\bar Q^{-1/2}\).

    Equals the largest generalized eigenvalue of \(\bar N v = \lambda \bar Q v\).
    Used for the CES scalar PD constraint \(a + b + \kappa g < 1\).
    """
    q = 0.5 * (np.asarray(qbar, dtype=float) + np.asarray(qbar, dtype=float).T)
    n = 0.5 * (np.asarray(nbar, dtype=float) + np.asarray(nbar, dtype=float).T)
    if q.shape != n.shape or q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("ADCC Qbar and Nbar must be matching square matrices")
    q, _ = repair_psd(q, tol=1e-12)
    try:
        evals = np.linalg.eigvalsh(np.linalg.solve(q, n))
    except np.linalg.LinAlgError as exc:
        raise ValueError("ADCC kappa is non-finite") from exc
    kappa = float(np.max(evals))
    if not np.isfinite(kappa):
        raise ValueError("ADCC kappa is non-finite")
    return max(kappa, 0.0)


def _adcc_step_q(
    q: Array,
    qbar: Array,
    nbar: Array,
    z_prev: Array,
    n_prev: Array,
    a: float,
    b: float,
    g: float,
) -> Array:
    r"""Scalar CES (2006) \(Q_t = (1-a-b)\bar Q - g\bar N + a z z' + b Q + g n n'\)."""
    return np.asarray(
        (1.0 - a - b) * qbar
        - g * nbar
        + a * np.outer(z_prev, z_prev)
        + b * q
        + g * np.outer(n_prev, n_prev),
        dtype=float,
    )


def _adcc_one_step_h(
    z: Array,
    qbar: Array,
    nbar: Array,
    a: float,
    b: float,
    g: float,
    sigma_one_step: Array,
) -> Array:
    n_shock = _adcc_negative_shocks(z)
    q = qbar.copy()
    t = int(z.shape[0])
    for i in range(1, t + 1):
        q = _adcc_step_q(q, qbar, nbar, z[i - 1], n_shock[i - 1], a, b, g)
    r = _dcc_r_from_q(q)
    d_next = np.diag(sigma_one_step)
    repaired, _ = repair_psd(np.asarray(d_next @ r @ d_next, dtype=float))
    return np.asarray(repaired, dtype=float)


def _agdcc_congruence_diag(diag: Array, matrix: Array) -> Array:
    r"""Diagonal congruence \(A M A\) for CES AG-DCC."""
    scale = np.asarray(diag, dtype=float).reshape(-1)
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or values.shape[0] != scale.size:
        raise ValueError("AG-DCC diagonal congruence requires a matching square matrix")
    return np.asarray(np.outer(scale, scale) * values, dtype=float)


def _agdcc_news(diag: Array, shock: Array) -> Array:
    r"""Diagonal news \(A z z^\top A = (A z)(A z)^\top\)."""
    scaled = np.asarray(diag, dtype=float).reshape(-1) * np.asarray(shock, dtype=float).reshape(-1)
    return np.asarray(np.outer(scaled, scaled), dtype=float)


def _agdcc_intercept(qbar: Array, nbar: Array, a: Array, b: Array, g: Array) -> Array:
    r"""CES intercept \(\bar Q - A\bar Q A - B\bar Q B - G\bar N G\)."""
    intercept = (
        np.asarray(qbar, dtype=float)
        - _agdcc_congruence_diag(a, qbar)
        - _agdcc_congruence_diag(b, qbar)
        - _agdcc_congruence_diag(g, nbar)
    )
    intercept = 0.5 * (intercept + intercept.T)
    return np.asarray(intercept, dtype=float)


def _agdcc_step_q(
    q: Array,
    intercept: Array,
    z_prev: Array,
    n_prev: Array,
    a: Array,
    b: Array,
    g: Array,
) -> Array:
    r"""Diagonal CES AG-DCC \(Q_t = Q^\ast + Azz'A + B Q B + G nn'G\)."""
    return np.asarray(
        intercept + _agdcc_news(a, z_prev) + _agdcc_congruence_diag(b, q) + _agdcc_news(g, n_prev),
        dtype=float,
    )


def _agdcc_one_step_h(
    z: Array,
    qbar: Array,
    intercept: Array,
    a: Array,
    b: Array,
    g: Array,
    sigma_one_step: Array,
) -> Array:
    n_shock = _adcc_negative_shocks(z)
    q = np.asarray(qbar, dtype=float).copy()
    t = int(z.shape[0])
    for i in range(1, t + 1):
        q = _agdcc_step_q(q, intercept, z[i - 1], n_shock[i - 1], a, b, g)
    r = _dcc_r_from_q(q)
    d_next = np.diag(sigma_one_step)
    repaired, _ = repair_psd(np.asarray(d_next @ r @ d_next, dtype=float))
    return np.asarray(repaired, dtype=float)


def _agdcc_broadcast_diag(value: float | None, n: int, default: float, name: str) -> Array:
    if value is None:
        return np.full(n, default, dtype=float)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return np.full(n, float(value), dtype=float)


def _agdcc_scale_start(
    a: Array, b: Array, g: Array, qbar: Array, nbar: Array
) -> tuple[Array, Array, Array]:
    start_a = np.asarray(a, dtype=float).copy()
    start_b = np.clip(np.asarray(b, dtype=float), 1e-6, 0.995)
    start_g = np.asarray(g, dtype=float).copy()
    for _ in range(8):
        intercept = _agdcc_intercept(qbar, nbar, start_a, start_b, start_g)
        if min_eigenvalue(intercept) >= 1e-8:
            return start_a, start_b, start_g
        start_a *= 0.5
        start_g *= 0.5
    return (
        np.full_like(start_a, 1e-4),
        np.clip(start_b, 1e-6, 0.95),
        np.zeros_like(start_g),
    )


def _agdcc_full_congruence(mat: Array, values: Array) -> Array:
    r"""Unrestricted congruence \(A M A^\top\) for CES AG-DCC."""
    left = np.asarray(mat, dtype=float)
    right = np.asarray(values, dtype=float)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape != right.shape
        or left.shape[0] != left.shape[1]
    ):
        raise ValueError("AG-DCC full congruence requires matching square matrices")
    return np.asarray(left @ right @ left.T, dtype=float)


def _agdcc_full_news(mat: Array, shock: Array) -> Array:
    r"""Unrestricted news \(A z z^\top A^\top = (A z)(A z)^\top\)."""
    left = np.asarray(mat, dtype=float)
    residual = np.asarray(shock, dtype=float).reshape(-1)
    if left.ndim != 2 or left.shape[0] != left.shape[1] or left.shape[0] != residual.size:
        raise ValueError("AG-DCC full news requires a square matrix matching the shock")
    scaled = left @ residual
    return np.asarray(np.outer(scaled, scaled), dtype=float)


def _agdcc_full_intercept(qbar: Array, nbar: Array, a: Array, b: Array, g: Array) -> Array:
    r"""CES intercept \(\bar Q - A\bar Q A^\top - B\bar Q B^\top - G\bar N G^\top\)."""
    intercept = (
        np.asarray(qbar, dtype=float)
        - _agdcc_full_congruence(a, qbar)
        - _agdcc_full_congruence(b, qbar)
        - _agdcc_full_congruence(g, nbar)
    )
    intercept = 0.5 * (intercept + intercept.T)
    return np.asarray(intercept, dtype=float)


def _agdcc_full_step_q(
    q: Array,
    intercept: Array,
    z_prev: Array,
    n_prev: Array,
    a: Array,
    b: Array,
    g: Array,
) -> Array:
    r"""Unrestricted CES AG-DCC \(Q_t = Q^\ast + Azz'A^\top + B Q B^\top + G nn'G^\top\)."""
    return np.asarray(
        intercept
        + _agdcc_full_news(a, z_prev)
        + _agdcc_full_congruence(b, q)
        + _agdcc_full_news(g, n_prev),
        dtype=float,
    )


def _agdcc_full_one_step_h(
    z: Array,
    qbar: Array,
    intercept: Array,
    a: Array,
    b: Array,
    g: Array,
    sigma_one_step: Array,
) -> Array:
    n_shock = _adcc_negative_shocks(z)
    q = np.asarray(qbar, dtype=float).copy()
    t = int(z.shape[0])
    for i in range(1, t + 1):
        q = _agdcc_full_step_q(q, intercept, z[i - 1], n_shock[i - 1], a, b, g)
    r = _dcc_r_from_q(q)
    d_next = np.diag(sigma_one_step)
    repaired, _ = repair_psd(np.asarray(d_next @ r @ d_next, dtype=float))
    return np.asarray(repaired, dtype=float)


def _agdcc_full_kronecker_radius(a: Array, b: Array, g: Array) -> float:
    r"""Spectral radius of \(A\otimes A + B\otimes B + G\otimes G\)."""
    stacked = np.kron(a, a) + np.kron(b, b) + np.kron(g, g)
    if not np.isfinite(stacked).all():
        return float("inf")
    radius = float(np.max(np.abs(np.linalg.eigvals(stacked))))
    if not np.isfinite(radius):
        return float("inf")
    return radius


def _agdcc_full_offdiag_maxabs(mat: Array) -> float:
    values = np.asarray(mat, dtype=float).copy()
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("AG-DCC full off-diagonal stamp requires a square matrix")
    np.fill_diagonal(values, 0.0)
    return float(np.max(np.abs(values))) if values.size else 0.0


def _agdcc_full_unpack(params: Array, n: int) -> tuple[Array, Array, Array]:
    values = np.asarray(params, dtype=float).reshape(-1)
    width = n * n
    if values.size != 3 * width:
        raise ValueError("AG-DCC full parameter vector length mismatch")
    return (
        values[:width].reshape(n, n),
        values[width : 2 * width].reshape(n, n),
        values[2 * width :].reshape(n, n),
    )


def _agdcc_full_bounds(n: int) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for diag_lo, diag_hi in ((0.0, 0.8), (1e-6, 0.995), (0.0, 0.8)):
        for i in range(n):
            for j in range(n):
                bounds.append((diag_lo, diag_hi) if i == j else (-0.2, 0.2))
    return bounds


def _gaussian_corr_nll(z: Array, corr: Array) -> float:
    residual = np.asarray(z, dtype=float).reshape(-1)
    r = np.asarray(corr, dtype=float)
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e12
    try:
        quadratic = float(residual @ np.linalg.solve(r, residual))
    except np.linalg.LinAlgError:
        return 1e12
    if not np.isfinite(quadratic):
        return 1e12
    return float(logdet + quadratic)


def student_t_corr_nll(z: Array, corr: Array, nu: float) -> float:
    r"""One-observation covariance Student-t nll on a correlation matrix.

    Uses the Engle/Bauwens covariance-t law: scale \(((ν-2)/ν) R\) so
    \(\mathrm{Var}(z)=R\) when \(ν>2\). The \(n\log\pi\) term is dropped
    because it does not depend on \(a,b,ν\). Fail-closed when \(ν\le 2\).
    """
    from scipy.special import gammaln

    residual = np.asarray(z, dtype=float).reshape(-1)
    r = np.asarray(corr, dtype=float)
    n = int(residual.size)
    if n < 1 or r.shape != (n, n):
        raise ValueError("student-t DCC residual and correlation shapes disagree")
    if not np.isfinite(nu) or nu <= 2.0:
        raise ValueError("student-t DCC nu must be finite and greater than 2")
    sign, logdet = np.linalg.slogdet(r)
    if sign <= 0 or not np.isfinite(logdet):
        return 1e12
    try:
        quadratic = float(residual @ np.linalg.solve(r, residual))
    except np.linalg.LinAlgError:
        return 1e12
    if not np.isfinite(quadratic) or quadratic < -1e-8:
        return 1e12
    quadratic = max(quadratic, 0.0)
    return float(
        -2.0 * gammaln(0.5 * (nu + n))
        + 2.0 * gammaln(0.5 * nu)
        + n * np.log(nu - 2.0)
        + logdet
        + (nu + n) * np.log1p(quadratic / (nu - 2.0))
    )


def _dcc_stage1_sigma_z_and_one_step(
    returns_1d: Array, *, dist: str = "normal"
) -> tuple[Array, Array, float]:
    r"""Fit univariate GARCH(1,1); return in-sample path plus one-step sigma.

    In-sample ``sigma`` / ``z`` feed stage-2 QML. The one-step decimal sigma is
    the Engle (2002) \(D_{t+1}\) diagonal, not the last in-sample \(\sigma_t\).
    ``dist`` is the univariate innovation law (``normal`` for Gaussian DCC,
    ``t`` for Student-t DCC). A failed, short, fallback, or invalid one-step
    univariate fit fails closed. RiskMetrics EWMA is not a silent substitute
    for stage-1 residuals.
    """
    from quant_fund.models.volatility import GARCHVol

    if dist not in {"normal", "t"}:
        raise ValueError("DCC stage-1 dist must be 'normal' or 't'")
    values = np.asarray(returns_1d, dtype=float).reshape(-1)
    model = GARCHVol(
        p=1,
        q=1,
        dist=dist,
        vol="garch",
        min_obs=DCC_STAGE1_MIN_OBS,
        mean="Constant",
        series_scope=DCC_STAGE1_SCOPE,
    )
    model.fit_returns(values)
    try:
        sigma, z = model.in_sample_sigma_and_z()
    except ValueError as exc:
        reason = model.fallback_reason or model.fit_status
        raise ValueError(f"DCC stage-1 GARCH failed: {reason}") from exc
    if sigma.size != values.size or z.size != values.size:
        raise ValueError("DCC stage-1 GARCH path length mismatch")
    forecast = model.forecast(horizon=1)
    variance = np.asarray(forecast["variance"], dtype=float).reshape(-1)
    fit_status = str(forecast.get("fit_status", model.fit_status))
    if (
        variance.size < 1
        or not np.isfinite(variance[0])
        or float(variance[0]) <= 0.0
        or fit_status != "fitted"
    ):
        reason = model.fallback_reason or fit_status
        raise ValueError(f"DCC stage-1 GARCH one-step forecast failed: {reason}")
    return sigma, z, float(np.sqrt(variance[0]))


def dcc_gaussian(
    returns: Array, a0: float | None = None, b0: float | None = None
) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage Gaussian DCC(1,1). Returns one-step-ahead H_{t+1} and params.

    Stage 1: univariate Gaussian GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: QML on a, b with correlation targeting.
    The returned matrix is Engle (2002) \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\):
    \(Q_{t+1}\) uses the last standardized residual \(z_t\), and \(D_{t+1}\) is
    the univariate one-step GARCH sigma, not in-sample last \(\sigma_t\).
    A failed univariate fit or one-step forecast fails closed; EWMA residuals
    are not substituted. The estimation sample is the trailing contiguous
    complete-case window ending at the last row: holes are not concatenated,
    and an incomplete terminal row fails closed so \(z_t\) cannot be silently
    dropped from \(H_{t+1}\). Params stamp ``family=dcc_gaussian``,
    ``covariance_object=one_step_ahead``, ``sample=trailing_complete_window``,
    and ``asymmetric=false``. Student-t DCC, Cappiello–Engle–Sheppard ADCC,
    and Bollerslev CCC are separate catalog estimators and must not be run
    by this function.
    ``optimize_asof`` uses this matrix only when
    ``optimizer.covariance=dcc_gaussian``; Student-t DCC, scalar ADCC, and
    CCC are separate named paths and must not silently size as Gaussian
    DCC. The default remains trailing Ledoit–Wolf plus the GARCH/RGARCH
    overlay. This does not invent HF RV.
    """
    from scipy.optimize import minimize

    if a0 is not None and (not np.isfinite(a0) or a0 < 0):
        raise ValueError("a0 must be finite and non-negative")
    if b0 is not None and (not np.isfinite(b0) or b0 < 0):
        raise ValueError("b0 must be finite and non-negative")
    raw = np.asarray(returns, dtype=float)
    x = dcc_trailing_complete_window(raw, min_rows=DCC_STAGE1_MIN_OBS)
    n_prefix_dropped = float(raw.shape[0] - x.shape[0]) if raw.ndim == 2 else 0.0
    t, n = x.shape
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j])
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    with np.errstate(divide="ignore", invalid="ignore"):
        qbar = np.asarray(np.corrcoef(z, rowvar=False), dtype=float)
    if not np.isfinite(qbar).all():
        raise ValueError("DCC correlation target is non-finite; input has unusable variance")
    np.fill_diagonal(qbar, 1.0)
    qbar, _ = repair_psd(qbar, tol=1e-12)

    def nll(params: Array) -> float:
        a, b = float(params[0]), float(params[1])
        if a < 0 or b < 0 or a + b >= 0.999:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = (1 - a - b) * qbar + a * np.outer(z[i - 1], z[i - 1]) + b * q
            d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
            r = q / np.outer(d, d)
            r = 0.5 * (r + r.T)
            np.fill_diagonal(r, 1.0)
            sign, logdet = np.linalg.slogdet(r)
            if sign <= 0:
                return 1e12
            try:
                quadratic = float(z[i] @ np.linalg.solve(r, z[i]))
            except np.linalg.LinAlgError:
                return 1e12
            ll += logdet + quadratic
        return float(ll / t)

    x0 = np.array([0.05 if a0 is None else a0, 0.9 if b0 is None else b0])
    x0 = np.clip(x0, 1e-6, 0.99)
    if x0.sum() >= 0.99:
        x0 *= 0.98 / x0.sum()
    res = minimize(
        nll,
        x0,
        bounds=[(1e-6, 0.5), (1e-6, 0.99)],
        constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},
        method="SLSQP",
    )
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b = float(candidate[0]), float(candidate[1])
    q = qbar.copy()
    # In-sample Q_1..Q_t, then one extra step Q_{t+1} from z_t.
    for i in range(1, t + 1):
        q = (1 - a - b) * qbar + a * np.outer(z[i - 1], z[i - 1]) + b * q
    d = np.sqrt(np.clip(np.diag(q), 1e-12, None))
    r = q / np.outer(d, d)
    r = 0.5 * (r + r.T)
    np.fill_diagonal(r, 1.0)
    d_next = np.diag(sigma_one_step)
    h = d_next @ r @ d_next
    h, _ = repair_psd(h)
    return h, {
        "a": a,
        "b": b,
        "success": float(res.success),
        "family": DCC_FAMILY_GAUSSIAN,
        "spec": DCC_SPEC_ENGLE_2002,
        "dist": "normal",
        "asymmetric": "false",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "normal",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def dcc_student_t(
    returns: Array,
    a0: float | None = None,
    b0: float | None = None,
    *,
    nu: float | None = None,
) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage Student-t DCC(1,1). Returns one-step-ahead H_{t+1} and params.

    Stage 1: univariate Student-t GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: QML on \(a,b,ν\) using the covariance Student-t correlation
    likelihood (scale \(((ν-2)/ν)R\) so \(\mathrm{Var}(z)=R\) when \(ν>2\)).
    This is not Gaussian DCC with a ``dist`` stamp: the stage-2 nll uses
    ``student_t_corr_nll``. The public matrix is the same Engle (2002)
    one-step object \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing
    contiguous complete-case window. \(ν\le 2\) fails closed because the
    covariance does not exist.     ``optimize_asof`` uses this matrix only when
    ``optimizer.covariance=dcc_student_t``; that named path must not silently
    size as Gaussian DCC or Ledoit–Wolf. Scalar ADCC and CCC are separate
    named paths. This does not invent HF RV.
    """
    from scipy.optimize import minimize

    if a0 is not None and (not np.isfinite(a0) or a0 < 0):
        raise ValueError("a0 must be finite and non-negative")
    if b0 is not None and (not np.isfinite(b0) or b0 < 0):
        raise ValueError("b0 must be finite and non-negative")
    if nu is not None and (not np.isfinite(nu) or nu <= 2.0):
        raise ValueError("nu must be finite and greater than 2")
    x, n_prefix_dropped = _dcc_prepare_window(returns)
    t, n = x.shape
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j], dist="t")
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    qbar = _dcc_qbar(z)

    def nll(params: Array) -> float:
        a, b, nu_hat = float(params[0]), float(params[1]), float(params[2])
        if a < 0 or b < 0 or a + b >= 0.999 or nu_hat <= 2.0:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = _dcc_step_q(q, qbar, z[i - 1], a, b)
            r = _dcc_r_from_q(q)
            term = student_t_corr_nll(z[i], r, nu_hat)
            if term >= 1e12:
                return 1e12
            ll += term
        return float(ll / t)

    nu0 = 8.0 if nu is None else float(nu)
    x0 = np.array(
        [0.05 if a0 is None else a0, 0.9 if b0 is None else b0, nu0],
        dtype=float,
    )
    x0[0] = float(np.clip(x0[0], 1e-6, 0.5))
    x0[1] = float(np.clip(x0[1], 1e-6, 0.99))
    if x0[0] + x0[1] >= 0.99:
        x0[:2] *= 0.98 / (x0[0] + x0[1])
    x0[2] = float(np.clip(x0[2], DCC_NU_MIN, DCC_NU_MAX))
    res = minimize(
        nll,
        x0,
        bounds=[(1e-6, 0.5), (1e-6, 0.99), (DCC_NU_MIN, DCC_NU_MAX)],
        constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1]},
        method="SLSQP",
    )
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b, nu_hat = float(candidate[0]), float(candidate[1]), float(candidate[2])
    if not np.isfinite(nu_hat) or nu_hat <= 2.0:
        raise ValueError("student-t DCC nu must be finite and greater than 2")
    h = _dcc_one_step_h(z, qbar, a, b, sigma_one_step)
    return h, {
        "a": a,
        "b": b,
        "nu": nu_hat,
        "success": float(res.success),
        "family": DCC_FAMILY_STUDENT_T,
        "spec": DCC_SPEC_STUDENT_T,
        "dist": DCC_DIST_STUDENT_T,
        "asymmetric": "false",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "t",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def adcc(
    returns: Array,
    a0: float | None = None,
    b0: float | None = None,
    g0: float | None = None,
) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage scalar Cappiello–Engle–Sheppard ADCC. Returns one-step \(H_{t+1}\).

    Stage 1: univariate Gaussian GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: QML on \(a,b,g\) with the CES (2006) scalar recursion
    \(Q_t=(1-a-b)\bar Q - g\bar N + a z_{t-1}z_{t-1}^\top + b Q_{t-1}
    + g n_{t-1}n_{t-1}^\top\), where \(n_t=I[z_t<0]\odot z_t\) and
    \(\bar N=\mathbb{E}[n_t n_t^\top]\). This is not Gaussian DCC with an
    ``asymmetric`` stamp: ``adcc`` does not call ``dcc_gaussian`` or
    ``dcc_student_t``. The public matrix is one-step-ahead
    \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous complete-case
    window. PD uses \(a,b,g\ge 0\) and \(a+b+\kappa g<1\) with
    \(\kappa=\lambda_{\max}(\bar Q^{-1/2}\bar N\bar Q^{-1/2})\).
    ``optimize_asof`` uses this matrix only when ``optimizer.covariance=adcc``;
    that named path must not silently size as Gaussian DCC, Student-t DCC,
    CCC, diagonal AG-DCC, or Ledoit–Wolf. Named CCC is a separate
    constant-correlation path. Named ``agdcc`` is a separate diagonal CES
    AG-DCC path and must not be run by this function. This is scalar CES
    ADCC, not matrix AG-DCC, and does not invent HF RV.
    """
    from scipy.optimize import minimize

    if a0 is not None and (not np.isfinite(a0) or a0 < 0):
        raise ValueError("a0 must be finite and non-negative")
    if b0 is not None and (not np.isfinite(b0) or b0 < 0):
        raise ValueError("b0 must be finite and non-negative")
    if g0 is not None and (not np.isfinite(g0) or g0 < 0):
        raise ValueError("g0 must be finite and non-negative")
    x, n_prefix_dropped = _dcc_prepare_window(returns)
    t, n = x.shape
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j])
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    qbar = _dcc_qbar(z)
    n_shock = _adcc_negative_shocks(z)
    nbar = _adcc_nbar(n_shock)
    kappa = _adcc_kappa(qbar, nbar)

    def nll(params: Array) -> float:
        a, b, g = float(params[0]), float(params[1]), float(params[2])
        if a < 0 or b < 0 or g < 0 or a + b + kappa * g >= 0.999:
            return 1e12
        intercept = (1.0 - a - b) * qbar - g * nbar
        if min_eigenvalue(intercept) < -1e-10:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = _adcc_step_q(q, qbar, nbar, z[i - 1], n_shock[i - 1], a, b, g)
            r = _dcc_r_from_q(q)
            term = _gaussian_corr_nll(z[i], r)
            if term >= 1e12:
                return 1e12
            ll += term
        return float(ll / t)

    x0 = np.array(
        [
            0.05 if a0 is None else a0,
            0.9 if b0 is None else b0,
            0.05 if g0 is None else g0,
        ],
        dtype=float,
    )
    x0[0] = float(np.clip(x0[0], 1e-6, 0.5))
    x0[1] = float(np.clip(x0[1], 1e-6, 0.99))
    if x0[0] + x0[1] >= 0.99:
        x0[:2] *= 0.98 / (x0[0] + x0[1])
    x0[2] = float(np.clip(x0[2], 0.0, 0.5))
    persist = x0[0] + x0[1] + kappa * x0[2]
    if persist >= 0.99:
        slack = 0.98 - x0[0] - x0[1]
        if kappa > 1e-12 and slack > 0.0:
            x0[2] = min(x0[2], slack / kappa)
        else:
            x0[2] = 0.0
    res = minimize(
        nll,
        x0,
        bounds=[(1e-6, 0.5), (1e-6, 0.99), (0.0, 0.5)],
        constraints={"type": "ineq", "fun": lambda p: 0.999 - p[0] - p[1] - kappa * p[2]},
        method="SLSQP",
    )
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b, g = float(candidate[0]), float(candidate[1]), float(candidate[2])
    if a < 0 or b < 0 or g < 0 or a + b + kappa * g >= 1.0 + 1e-8:
        a, b, g = float(x0[0]), float(x0[1]), float(x0[2])
    h = _adcc_one_step_h(z, qbar, nbar, a, b, g, sigma_one_step)
    return h, {
        "a": a,
        "b": b,
        "g": g,
        "kappa": kappa,
        "success": float(res.success),
        "family": DCC_FAMILY_ADCC,
        "spec": DCC_SPEC_ADCC,
        "dist": DCC_DIST_NORMAL,
        "asymmetric": "true",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "normal",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def agdcc(
    returns: Array,
    a0: float | None = None,
    b0: float | None = None,
    g0: float | None = None,
) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage diagonal Cappiello–Engle–Sheppard AG-DCC. Returns one-step \(H_{t+1}\).

    Stage 1: univariate Gaussian GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: QML on diagonal \(A=\mathrm{diag}(a)\), \(B=\mathrm{diag}(b)\),
    \(G=\mathrm{diag}(g)\) with the CES (2006) matrix recursion
    \(Q_t=(\bar Q-A\bar Q A-B\bar Q B-G\bar N G)+A z_{t-1}z_{t-1}^\top A
    + B Q_{t-1} B + G n_{t-1}n_{t-1}^\top G\), where
    \(n_t=I[z_t<0]\odot z_t\). This is not scalar ADCC with a
    ``parameterization`` stamp: ``agdcc`` does not call ``adcc``,
    ``dcc_gaussian``, or ``dcc_student_t``. Equal diagonals
    \(A=\sqrt{a}I\) recover the scalar CES recursion; heterogeneous
    diagonals do not. The public matrix is one-step-ahead
    \(H_{t+1}=D_{t+1}R_{t+1}D_{t+1}\) on the trailing contiguous
    complete-case window. PD uses \(a,b,g\ge 0\) and a positive-definite
    intercept. Params stamp ``family=agdcc``,
    ``spec=cappiello_engle_sheppard_2006_diagonal_agdcc``, and
    ``parameterization=diagonal``. Unrestricted full-matrix AG-DCC is a
    separate catalog estimator (``agdcc_full``); this function does not
    call it. ``optimize_asof`` / ``/risk/portfolio`` use this matrix
    when ``optimizer.covariance=agdcc`` and must not overlay GARCH/RGARCH
    (AG-DCC already supplies \(D_{t+1}\)) or silently size as scalar ADCC,
    Gaussian DCC, CCC, unrestricted AG-DCC, or Ledoit–Wolf. Named
    ``agdcc_full`` is a separate unrestricted CES AG-DCC path. This does
    not invent HF RV or wire factor covariance.
    """
    from scipy.optimize import minimize

    x, n_prefix_dropped = _dcc_prepare_window(returns)
    t, n = x.shape
    if n < 2:
        raise ValueError("AG-DCC requires at least two return series")
    start_a = _agdcc_broadcast_diag(a0, n, float(np.sqrt(0.05)), "a0")
    start_b = _agdcc_broadcast_diag(b0, n, float(np.sqrt(0.90)), "b0")
    start_g = _agdcc_broadcast_diag(g0, n, float(np.sqrt(0.05)), "g0")
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j])
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    qbar = _dcc_qbar(z)
    n_shock = _adcc_negative_shocks(z)
    nbar = _adcc_nbar(n_shock)
    start_a, start_b, start_g = _agdcc_scale_start(start_a, start_b, start_g, qbar, nbar)

    def _unpack(params: Array) -> tuple[Array, Array, Array]:
        values = np.asarray(params, dtype=float).reshape(-1)
        return values[:n], values[n : 2 * n], values[2 * n :]

    def nll(params: Array) -> float:
        a, b, g = _unpack(params)
        if np.any(a < 0) or np.any(b < 0) or np.any(g < 0):
            return 1e12
        intercept = _agdcc_intercept(qbar, nbar, a, b, g)
        if min_eigenvalue(intercept) < -1e-10:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = _agdcc_step_q(q, intercept, z[i - 1], n_shock[i - 1], a, b, g)
            r = _dcc_r_from_q(q)
            term = _gaussian_corr_nll(z[i], r)
            if term >= 1e12:
                return 1e12
            ll += term
        return float(ll / t)

    x0 = np.concatenate([start_a, start_b, start_g])
    bounds = [(0.0, 0.8)] * n + [(1e-6, 0.995)] * n + [(0.0, 0.8)] * n
    res = minimize(nll, x0, bounds=bounds, method="SLSQP")
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b, g = _unpack(candidate)
    intercept = _agdcc_intercept(qbar, nbar, a, b, g)
    if np.any(a < 0) or np.any(b < 0) or np.any(g < 0) or min_eigenvalue(intercept) < -1e-10:
        a, b, g = start_a, start_b, start_g
        intercept = _agdcc_intercept(qbar, nbar, a, b, g)
    h = _agdcc_one_step_h(z, qbar, intercept, a, b, g, sigma_one_step)
    return h, {
        "a_mean": float(np.mean(a)),
        "b_mean": float(np.mean(b)),
        "g_mean": float(np.mean(g)),
        "a_min": float(np.min(a)),
        "a_max": float(np.max(a)),
        "b_min": float(np.min(b)),
        "b_max": float(np.max(b)),
        "g_min": float(np.min(g)),
        "g_max": float(np.max(g)),
        "intercept_eig_min": float(min_eigenvalue(intercept)),
        "success": float(res.success),
        "family": DCC_FAMILY_AGDCC,
        "spec": DCC_SPEC_AGDCC,
        "parameterization": DCC_PARAMETERIZATION_DIAGONAL,
        "dist": DCC_DIST_NORMAL,
        "asymmetric": "true",
        "dynamic_correlation": "true",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "normal",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def agdcc_full(
    returns: Array,
    a0: float | None = None,
    b0: float | None = None,
    g0: float | None = None,
) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage unrestricted Cappiello–Engle–Sheppard AG-DCC. Returns one-step \(H_{t+1}\).

    Stage 1: univariate Gaussian GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: QML on unrestricted \(A,B,G\) with the CES (2006) recursion
    \(Q_t=(\bar Q-A\bar Q A^\top-B\bar Q B^\top-G\bar N G^\top)
    +A z_{t-1}z_{t-1}^\top A^\top + B Q_{t-1} B^\top
    + G n_{t-1}n_{t-1}^\top G^\top\), where \(n_t=I[z_t<0]\odot z_t\).
    Diagonal \(A,B,G\) recover Wave 135 diagonal AG-DCC; nonzero
    off-diagonals do not. This is not diagonal AG-DCC with a
    ``parameterization`` stamp: ``agdcc_full`` does not call ``agdcc``,
    ``adcc``, ``dcc_gaussian``, ``dcc_student_t``, or ``ccc``. PD uses a
    positive-definite intercept and spectral radius of
    \(A\otimes A+B\otimes B+G\otimes G\) strictly below one. Params stamp
    ``family=agdcc_full``,
    ``spec=cappiello_engle_sheppard_2006_full_agdcc``, and
    ``parameterization=full``. ``optimize_asof`` / ``/risk/portfolio`` use
    this matrix when ``optimizer.covariance=agdcc_full`` and must not
    overlay GARCH/RGARCH (AG-DCC already supplies \(D_{t+1}\)) or silently
    size as diagonal AG-DCC, scalar ADCC, Gaussian DCC, CCC, or
    Ledoit–Wolf. This does not invent HF RV or wire factor covariance.
    """
    from scipy.optimize import minimize

    x, n_prefix_dropped = _dcc_prepare_window(returns)
    t, n = x.shape
    if n < 2:
        raise ValueError("AG-DCC requires at least two return series")
    start_a = _agdcc_broadcast_diag(a0, n, float(np.sqrt(0.05)), "a0")
    start_b = _agdcc_broadcast_diag(b0, n, float(np.sqrt(0.90)), "b0")
    start_g = _agdcc_broadcast_diag(g0, n, float(np.sqrt(0.05)), "g0")
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j])
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    qbar = _dcc_qbar(z)
    n_shock = _adcc_negative_shocks(z)
    nbar = _adcc_nbar(n_shock)
    start_a, start_b, start_g = _agdcc_scale_start(start_a, start_b, start_g, qbar, nbar)
    start_a_mat = np.diag(start_a)
    start_b_mat = np.diag(start_b)
    start_g_mat = np.diag(start_g)
    x0 = np.concatenate([start_a_mat.reshape(-1), start_b_mat.reshape(-1), start_g_mat.reshape(-1)])

    def nll(params: Array) -> float:
        a, b, g = _agdcc_full_unpack(params, n)
        if np.any(np.diag(a) < 0) or np.any(np.diag(b) < 0) or np.any(np.diag(g) < 0):
            return 1e12
        intercept = _agdcc_full_intercept(qbar, nbar, a, b, g)
        if min_eigenvalue(intercept) < -1e-10:
            return 1e12
        if _agdcc_full_kronecker_radius(a, b, g) >= 0.999:
            return 1e12
        q = qbar.copy()
        ll = 0.0
        for i in range(1, t):
            q = _agdcc_full_step_q(q, intercept, z[i - 1], n_shock[i - 1], a, b, g)
            r = _dcc_r_from_q(q)
            term = _gaussian_corr_nll(z[i], r)
            if term >= 1e12:
                return 1e12
            ll += term
        return float(ll / t)

    res = minimize(nll, x0, bounds=_agdcc_full_bounds(n), method="SLSQP")
    candidate = np.asarray(res.x if res.success and np.isfinite(res.fun) else x0)
    a, b, g = _agdcc_full_unpack(candidate, n)
    intercept = _agdcc_full_intercept(qbar, nbar, a, b, g)
    radius = _agdcc_full_kronecker_radius(a, b, g)
    if (
        np.any(np.diag(a) < 0)
        or np.any(np.diag(b) < 0)
        or np.any(np.diag(g) < 0)
        or min_eigenvalue(intercept) < -1e-10
        or radius >= 1.0 + 1e-8
    ):
        a, b, g = start_a_mat, start_b_mat, start_g_mat
        intercept = _agdcc_full_intercept(qbar, nbar, a, b, g)
        radius = _agdcc_full_kronecker_radius(a, b, g)
    h = _agdcc_full_one_step_h(z, qbar, intercept, a, b, g, sigma_one_step)
    return h, {
        "a_mean": float(np.mean(np.diag(a))),
        "b_mean": float(np.mean(np.diag(b))),
        "g_mean": float(np.mean(np.diag(g))),
        "a_min": float(np.min(np.diag(a))),
        "a_max": float(np.max(np.diag(a))),
        "b_min": float(np.min(np.diag(b))),
        "b_max": float(np.max(np.diag(b))),
        "g_min": float(np.min(np.diag(g))),
        "g_max": float(np.max(np.diag(g))),
        "a_offdiag_maxabs": _agdcc_full_offdiag_maxabs(a),
        "b_offdiag_maxabs": _agdcc_full_offdiag_maxabs(b),
        "g_offdiag_maxabs": _agdcc_full_offdiag_maxabs(g),
        "intercept_eig_min": float(min_eigenvalue(intercept)),
        "kronecker_radius": float(radius),
        "success": float(res.success),
        "family": DCC_FAMILY_AGDCC_FULL,
        "spec": DCC_SPEC_AGDCC_FULL,
        "parameterization": DCC_PARAMETERIZATION_FULL,
        "dist": DCC_DIST_NORMAL,
        "asymmetric": "true",
        "dynamic_correlation": "true",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "normal",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def ccc(returns: Array) -> tuple[Array, dict[str, float | str]]:
    r"""Two-stage Bollerslev (1990) CCC. Returns one-step-ahead \(H_{t+1}\).

    Stage 1: univariate Gaussian GARCH(1,1) via ``arch`` / ``GARCHVol``.
    Stage 2: constant \(R=\mathrm{corr}(z)\) of standardized residuals.
    There is no \(Q\) recursion and no \(a,b\) QML: this is not Gaussian
    DCC with \(a=b=0\) fitted by Engle QML. ``ccc`` does not call
    ``dcc_gaussian``, ``dcc_student_t``, ``adcc``, ``agdcc``, or
    ``agdcc_full``. The public matrix is \(H_{t+1}=D_{t+1} R D_{t+1}\) on
    the trailing contiguous complete-case window. \(D_{t+1}\) is the
    univariate GARCH one-step
    sigma, not in-sample last \(\sigma_t\). Params stamp ``family=ccc``,
    ``spec=bollerslev_1990_ccc``, ``dynamic_correlation=false``, and
    ``covariance_object=one_step_ahead``. ``optimize_asof`` /
    ``/risk/portfolio`` use this matrix when ``optimizer.covariance=ccc``
    and must not overlay GARCH/RGARCH (CCC already supplies \(D_{t+1}\))
    or silently size as Gaussian DCC, Student-t DCC, scalar ADCC,
    diagonal AG-DCC, unrestricted AG-DCC, or Ledoit–Wolf. Named ``agdcc``
    is a separate diagonal CES AG-DCC path. Named ``agdcc_full`` is a
    separate unrestricted CES AG-DCC path. This does not invent HF RV or
    wire factor covariance.
    """
    x, n_prefix_dropped = _dcc_prepare_window(returns)
    t, n = x.shape
    if n < 2:
        raise ValueError("CCC requires at least two return series")
    z = np.zeros_like(x)
    sigma_one_step = np.zeros(n, dtype=float)
    for j in range(n):
        _sigma_j, z_j, sigma_next_j = _dcc_stage1_sigma_z_and_one_step(x[:, j])
        z[:, j] = z_j
        sigma_one_step[j] = sigma_next_j
    corr = _dcc_qbar(z)
    d_next = np.diag(sigma_one_step)
    h, _ = repair_psd(np.asarray(d_next @ corr @ d_next, dtype=float))
    return np.asarray(h, dtype=float), {
        "family": DCC_FAMILY_CCC,
        "spec": DCC_SPEC_CCC,
        "dist": DCC_DIST_NORMAL,
        "asymmetric": "false",
        "dynamic_correlation": "false",
        "stage1": "garch",
        "stage1_vol": "garch",
        "stage1_dist": "normal",
        "stage1_mean": "Constant",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def ewma(returns: Array, lam: float = EWMA_DEFAULT_LAM) -> tuple[Array, dict[str, float | str]]:
    r"""RiskMetrics EWMA covariance. Returns one-step-ahead \(H_{t+1}\) and params.

    \(H_{t+1}=\lambda H_t+(1-\lambda)r_t r_t'\) on the trailing contiguous
    complete-case window. An incomplete terminal row fails closed so asof
    \(r_t\) cannot be dropped from advertised \(H_{t+1}\). Holes are not
    concatenated. This is not Gaussian DCC, Student-t DCC, scalar ADCC, or
    Ledoit–Wolf: ``ewma`` does not call those fitters. ``optimize_asof`` /
    ``/risk/portfolio`` use this matrix when ``optimizer.covariance=ewma``
    and must not overlay GARCH/RGARCH (the EWMA recursion already supplies
    \(H_{t+1}\)) or silently substitute sample, Ledoit–Wolf, or DCC.
    Params stamp ``family=ewma``, ``spec=jpmorgan_riskmetrics_1996``,
    ``covariance_object=one_step_ahead``, and
    ``sample=trailing_complete_window``. This does not invent HF RV or
    implement matrix AG-DCC.
    """
    if not np.isfinite(lam) or not 0.0 <= float(lam) < 1.0:
        raise ValueError("ewma lambda must be finite and in [0, 1)")
    raw = np.asarray(returns, dtype=float)
    x = dcc_trailing_complete_window(raw, min_rows=EWMA_MIN_OBS)
    n_prefix_dropped = float(raw.shape[0] - x.shape[0]) if raw.ndim == 2 else 0.0
    sigma = ewma_cov(x, float(lam), min_rows=EWMA_MIN_OBS)
    if not np.isfinite(sigma).all():
        raise ValueError("EWMA covariance produced non-finite values")
    repaired, _ = repair_psd(sigma)
    t, n = x.shape
    return np.asarray(repaired, dtype=float), {
        "family": OPTIMIZER_COVARIANCE_EWMA,
        "spec": EWMA_SPEC_RISKMETRICS,
        "lambda": float(lam),
        "asymmetric": "false",
        "covariance_object": DCC_COVARIANCE_OBJECT_ONE_STEP,
        "horizon": 1.0,
        "sample": DCC_SAMPLE_TRAILING_COMPLETE,
        "n_obs": float(t),
        "n_prefix_dropped": n_prefix_dropped,
        "n_assets": float(n),
    }


def ewma_variance_1d(r: Array, lam: float = 0.94) -> Array:
    _validate_lambda(lam)
    r = np.asarray(r, dtype=float)
    if r.ndim != 1 or r.size == 0 or not np.isfinite(r).all():
        raise ValueError("r must be a non-empty finite 1D array")
    v = np.empty_like(r, dtype=float)
    v[0] = r[0] ** 2
    for t in range(1, r.size):
        v[t] = lam * v[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return v


def condition_number(sigma: Array) -> float:
    s = np.asarray(sigma, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] == 0:
        raise ValueError("sigma must be a non-empty square matrix")
    if not np.isfinite(s).all():
        raise ValueError("sigma must contain only finite values")
    eig = np.linalg.eigvalsh(0.5 * (s + s.T))
    if float(eig[0]) <= 1e-12:
        return float("inf")
    return float(eig.max() / eig.min())
