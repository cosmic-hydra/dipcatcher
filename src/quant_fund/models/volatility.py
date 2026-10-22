"""Volatility forecasts: rolling, EWMA, GARCH, HAR-RV, trees (SYNTHETIC)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from quant_fund.compute.cache import FitCache
from quant_fund.compute.numba_guard import call_kernel
from quant_fund.models.base import JoblibMixin, ModelMeta
from quant_fund.models.ranking import _finite
from quant_fund.models.vol_scope import (
    POOLED_DATE_LEVEL_SCOPE,
    SECURITY_LEVEL_RET_1_SCOPE,
    UNIVARIATE_RETURN_SERIES_SCOPE,
    assert_scope_compatible,
)

try:
    from numba import njit

    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is a required dependency
    _HAVE_NUMBA = False

    def njit(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        def _deco(fn: Any) -> Any:
            return fn

        if len(_args) == 1 and callable(_args[0]) and not _kwargs:
            return _args[0]
        return _deco


@njit(cache=True)
def _ewma_variance_nb(r: np.ndarray, lam: float) -> np.ndarray:
    """Same recurrence as the Python loop, including non-finite propagation."""
    n = r.shape[0]
    var = np.empty(n, dtype=np.float64)
    var[0] = r[0] * r[0]
    one_minus = 1.0 - lam
    for t in range(1, n):
        prev = r[t - 1] * r[t - 1]
        var[t] = lam * var[t - 1] + one_minus * prev
    return var


_GARCH_FIT_CACHE = FitCache(max_entries=64)


def _garch_fit_cache_key(model: Any, raw: object) -> str | None:
    try:
        values = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    finite = np.ascontiguousarray(values[np.isfinite(values)], dtype=np.float64)
    # The fitter identity is part of the key so a patched ``arch.arch_model``
    # cannot reuse a real fit, and a real fit cannot reuse a patched one.
    try:
        import arch

        fitter_id = id(arch.arch_model)
    except ImportError:
        fitter_id = 0
    return _GARCH_FIT_CACHE.hash_key(
        (
            int(model.p),
            int(model.q),
            str(model.dist),
            str(model.vol),
            int(model.min_obs),
            str(model.mean),
            float(model.power),
            str(getattr(model, "series_scope", "")),
            int(fitter_id),
        ),
        (finite,),
    )


def _garch_snapshot(model: Any) -> dict[str, Any]:
    returns = getattr(model, "_returns_percent", None)
    copied = None if returns is None else np.array(returns, dtype=float, copy=True)
    return {
        "n_obs": int(model.n_obs),
        "result": model.result,
        "converged": bool(model.converged),
        "fit_status": model.fit_status,
        "fallback_reason": model.fallback_reason,
        "last_sigma": float(model.last_sigma),
        "_returns_percent": copied,
    }


def _restore_garch_snapshot(model: Any, snap: dict[str, Any]) -> None:
    model.n_obs = int(snap["n_obs"])
    model.result = snap["result"]
    model.converged = bool(snap["converged"])
    model.fit_status = snap["fit_status"]
    model.fallback_reason = snap["fallback_reason"]
    model.last_sigma = float(snap["last_sigma"])
    cached = snap["_returns_percent"]
    model._returns_percent = (
        np.empty(0, dtype=float) if cached is None else np.array(cached, dtype=float, copy=True)
    )


class RollingVol(JoblibMixin):
    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> RollingVol:
        return self

    def predict_from_returns(self, log_returns: NDArray[np.float64]) -> NDArray[np.float64]:
        r = np.asarray(log_returns, dtype=float)
        window = int(self.window)
        # Sliding-window std matches the per-slice ``np.std(..., ddof=1)``
        # bit for bit on a 1-d series (NumPy reduces each window the same way).
        if r.ndim == 1 and window >= 2 and r.size >= window:
            out = np.full_like(r, np.nan)
            view = np.lib.stride_tricks.sliding_window_view(r, window)
            out[window - 1 :] = np.std(view, axis=1, ddof=1)
            return out
        out = np.full_like(r, np.nan)
        for i in range(window, r.size + 1):
            out[i - 1] = np.std(r[i - window : i], ddof=1)
        return out

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        # last column assumed to be trailing vol already
        return x[:, -1]

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="rolling", version="v1")


def ewma_variance(log_returns: NDArray[np.float64], lam: float = 0.94) -> NDArray[np.float64]:
    """RiskMetrics-style recursive variance. Fail-closed on invalid ``lam`` / empty series."""
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and between 0 and 1")
    r = np.ascontiguousarray(np.asarray(log_returns, dtype=np.float64).reshape(-1))
    if r.size == 0:
        raise ValueError("log_returns must be non-empty")
    if _HAVE_NUMBA:
        return np.asarray(call_kernel(_ewma_variance_nb, r, float(lam)), dtype=np.float64)
    var = np.empty_like(r)
    var[0] = r[0] ** 2
    for t in range(1, r.size):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r[t - 1] ** 2
    return var


class EWMAVol(JoblibMixin):
    def __init__(self, lam: float = 0.94) -> None:
        if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
            raise ValueError("lam must be finite and between 0 and 1")
        self.lam = lam

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> EWMAVol:
        return self

    def predict_from_returns(self, log_returns: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(ewma_variance(log_returns, self.lam))

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return x[:, -1]

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name="ewma", version="v1", extra={"lambda": self.lam})


_ALLOWED_GARCH_VOLS = {
    "garch": ("GARCH", 0),
    "egarch": ("EGARCH", 0),
    "gjr": ("GARCH", 1),
    "aparch": ("APARCH", 1),
    "figarch": ("FIGARCH", 0),
}
_ALLOWED_GARCH_DISTS = {"normal", "t", "skewt"}
_SIMULATED_MULTI_STEP_VOLS = frozenset({"egarch", "aparch"})
_GARCH_PERSISTENCE_VOLS = frozenset({"garch", "gjr", "aparch"})
_GARCH_SCALE = 100.0
_GARCH_VARIANCE_FLOOR = 1e-16
GARCH_DATE_LEVEL_SCOPE = POOLED_DATE_LEVEL_SCOPE
GARCH_SECURITY_LEVEL_SCOPE = SECURITY_LEVEL_RET_1_SCOPE


class GARCHVol(JoblibMixin):
    """Causal univariate GARCH-family volatility model.

    The model is fitted to *decimal returns*, not to a forward realized
    variance label.  Returns are converted to percent units only at the
    ``arch`` boundary and forecast variances are converted back to decimal
    squared units.  UNITS: ``forecast(...)['variance']`` is always a
    **variance** (sigma^2) in decimal-squared return units
    (``variance_units='decimal_squared'``) — never a volatility — and
    ``forecast(...)['sigma']`` is the explicit square-root convenience in
    decimal return units (``sigma_units='decimal'``); the two are never
    silently interconverted.      ``forecast`` is the explicit probabilistic API; ``predict``
    remains a compatibility adapter returning the current-origin sigma.
    ``log_density`` / ``pit`` score that origin law on decimal returns.

    ``vol`` selects symmetric GARCH, EGARCH, GJR-GARCH, APARCH, or FIGARCH and
    ``dist`` selects normal, Student-t, or skewed Student-t innovations.
    Failed, non-finite, non-converged, or too-short fits fail closed to a
    documented sample-sigma fallback and expose the reason through diagnostics
    attributes. Date-level equal-weight overlays keep
    ``series_scope=date_level_equal_weight_cross_section``. Per-security
    causal fits live in a separate namespace
    (``series_scope=security_level_ret_1``) and cannot replace the market
    overlay or name-level ``vol_20``. This is still not a live-performance
    claim.
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        vol: str = "garch",
        *,
        min_obs: int = 50,
        mean: str = "Constant",
        power: float = 2.0,
        series_scope: str = "univariate_return_series",
    ) -> None:
        if (
            isinstance(p, bool)
            or isinstance(q, bool)
            or not isinstance(p, int)
            or not isinstance(q, int)
        ):
            raise ValueError("p and q must be positive integers")
        if isinstance(dist, bool) or not isinstance(dist, str):
            raise ValueError("dist must be a string")
        if isinstance(vol, bool) or not isinstance(vol, str):
            raise ValueError("vol must be a string")
        dist_key = dist.lower()
        vol_key = vol.lower()
        if dist_key not in _ALLOWED_GARCH_DISTS:
            raise ValueError(f"dist must be one of {sorted(_ALLOWED_GARCH_DISTS)}")
        if vol_key not in _ALLOWED_GARCH_VOLS:
            raise ValueError(f"vol must be one of {sorted(_ALLOWED_GARCH_VOLS)}")
        if vol_key == "figarch":
            if p not in (0, 1) or q not in (0, 1):
                raise ValueError("FIGARCH p and q must be 0 or 1")
        elif p < 1 or q < 1:
            raise ValueError("p and q must be positive integers")
        if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
            raise ValueError("min_obs must be an integer >= 2")
        if mean not in {"Constant", "Zero"}:
            raise ValueError("mean must be 'Constant' or 'Zero'")
        if not np.isfinite(power) or power <= 0.0:
            raise ValueError("power must be finite and positive")
        if not isinstance(series_scope, str) or not series_scope.strip():
            raise ValueError("series_scope must be a non-empty string")
        self.p = int(p)
        self.q = int(q)
        self.dist = dist_key
        self.vol = vol_key
        self.min_obs = min_obs
        self.mean = mean
        self.power = float(power)
        self.series_scope = series_scope.strip()
        self.result: Any = None
        self.last_sigma: float = 0.01
        self.n_obs = 0
        self.fit_status = "unfitted"
        self.converged = False
        self.fallback_reason: str | None = None
        self.returns_scale = "decimal_returns_to_percent"
        self._returns_percent = np.empty(0, dtype=float)

    def _fallback(self, returns_percent: NDArray[np.float64], reason: str) -> GARCHVol:
        self.result = None
        self.converged = False
        self.fit_status = "fallback"
        self.fallback_reason = reason
        if returns_percent.size > 1:
            sigma = float(np.std(returns_percent, ddof=1) / _GARCH_SCALE)
            self.last_sigma = sigma if np.isfinite(sigma) and sigma > 0.0 else 0.01
        else:
            self.last_sigma = 0.01
        return self

    @staticmethod
    def _param_values(params: Any, prefix: str) -> list[float]:
        try:
            keys = list(params.index)
            return [float(params[key]) for key in keys if str(key).startswith(prefix)]
        except (AttributeError, KeyError, TypeError, ValueError):
            return []

    @staticmethod
    def _param_scalar(params: Any, name: str) -> float:
        try:
            return float(params[name])
        except (AttributeError, KeyError, TypeError, ValueError):
            return float("nan")

    def fit_returns(self, returns: NDArray[np.float64]) -> GARCHVol:
        """Fit directly on a causal decimal-return series."""
        values = np.asarray(returns, dtype=float).reshape(-1)
        return self.fit(np.empty((values.size, 0)), values, returns=values)

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> GARCHVol:
        """Fit on causal returns; pass ``returns=`` when ``y`` is a label.

        ``y`` is retained in the protocol for compatibility with other models.
        Training code with a forward target must pass the historical return
        series explicitly as ``returns=``; silently treating a variance target
        as returns is prohibited by the pipeline contract.
        """
        if "returns" not in kwargs:
            raise ValueError(
                "returns= must be supplied explicitly; y may be a forward label and "
                "cannot be used as a GARCH likelihood input"
            )
        cache_key = _garch_fit_cache_key(self, kwargs["returns"])
        if cache_key is not None:
            cached = _GARCH_FIT_CACHE.get(cache_key)
            if cached is not None:
                _restore_garch_snapshot(self, cached)
                return self
        self._fit_uncached(x, y, **kwargs)
        if cache_key is not None:
            _GARCH_FIT_CACHE.put(cache_key, _garch_snapshot(self))
        return self

    def _fit_uncached(
        self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any
    ) -> GARCHVol:
        from arch import arch_model

        raw = kwargs.pop("returns")
        try:
            values = np.asarray(raw, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return self._fallback(np.empty(0), "invalid_returns")
        finite = values[np.isfinite(values)]
        self.n_obs = int(finite.size)
        returns_percent = finite * _GARCH_SCALE
        self._returns_percent = returns_percent.copy()
        if finite.size < self.min_obs:
            return self._fallback(returns_percent, "insufficient_observations")
        if np.std(returns_percent) <= 0.0 or not np.isfinite(np.std(returns_percent)):
            return self._fallback(returns_percent, "zero_variance")
        arch_vol, o = _ALLOWED_GARCH_VOLS[self.vol]
        try:
            am = arch_model(
                returns_percent,
                vol=arch_vol,  # type: ignore[arg-type]
                p=self.p,
                o=o,
                q=self.q,
                dist=self.dist,  # type: ignore[arg-type]
                mean=self.mean,  # type: ignore[arg-type]
                power=self.power,
                rescale=False,
            )
            result = am.fit(disp="off", show_warning=False)
            flag = int(getattr(result, "convergence_flag", 0))
            cond_vol = np.asarray(result.conditional_volatility, dtype=float).reshape(-1)
            if flag != 0:
                return self._fallback(returns_percent, f"nonconvergence:{flag}")
            if cond_vol.size == 0 or not np.isfinite(cond_vol).all() or cond_vol[-1] <= 0.0:
                return self._fallback(returns_percent, "invalid_conditional_volatility")
            params = result.params
            if not np.isfinite(np.asarray(params, dtype=float)).all():
                return self._fallback(returns_percent, "nonfinite_parameters")
            spec_reason = self._spec_inadmissible_reason(params)
            if spec_reason is not None:
                return self._fallback(returns_percent, spec_reason)
        except Exception as exc:
            return self._fallback(returns_percent, f"fit_error:{type(exc).__name__}")
        self.result = result
        self.converged = True
        self.fit_status = "fitted"
        self.fallback_reason = None
        self.last_sigma = float(cond_vol[-1] / _GARCH_SCALE)
        return self

    def _spec_inadmissible_reason(self, params: Any) -> str | None:
        """Return a fail-closed fit reason when the fitted spec is not usable."""
        if self.vol in _GARCH_PERSISTENCE_VOLS:
            alpha = sum(self._param_values(params, "alpha["))
            beta = sum(self._param_values(params, "beta["))
            gamma = sum(self._param_values(params, "gamma["))
            persistence = alpha + beta + (0.5 * gamma if self.vol == "gjr" else 0.0)
            if persistence >= 1.0:
                return "nonstationary_persistence"
        if self.vol == "aparch":
            delta = self._param_scalar(params, "delta")
            if not np.isfinite(delta) or delta <= 0.0:
                return "invalid_aparch_delta"
        if self.vol == "figarch":
            frac_d = self._param_scalar(params, "d")
            if not np.isfinite(frac_d) or not 0.0 < frac_d < 1.0:
                return "invalid_fractional_d"
        return None

    def forecast(
        self,
        horizon: int = 1,
        *,
        method: str = "analytic",
        simulations: int = 1000,
        quantiles: tuple[float, ...] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Return horizon-indexed decimal variance, sigma and optional quantiles.

        UNITS CONTRACT (explicit, never silently rescaled):
        ``forecast(...)["variance"]`` is a **variance** (sigma^2) in
        decimal-squared return units (``variance_units="decimal_squared"``) —
        it is NOT a volatility.  ``forecast(...)["sigma"]`` is the documented
        square-root convenience in decimal return units
        (``sigma_units="decimal"``).  A variance input to any consumer of this
        dict stays a variance; converting to or from volatility requires the
        explicit ``quant_fund.metrics.vol_eval.rescale_units`` call.

        Each element is the conditional variance of the corresponding future
        return.  ``cumulative_variance`` is supplied for an h-bar realized
        variance target under the usual zero autocovariance approximation.
        """
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if method not in {"analytic", "simulation", "bootstrap"}:
            raise ValueError("method must be analytic, simulation, or bootstrap")
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if self.result is None:
            variance = np.full(horizon, max(self.last_sigma**2, _GARCH_VARIANCE_FLOOR))
            out: dict[str, Any] = {
                "variance": variance,
                "sigma": np.sqrt(variance),
                "cumulative_variance": np.cumsum(variance),
                "mean": self._mean_decimal(),
                "variance_units": "decimal_squared",
                "sigma_units": "decimal",
                # Fallbacks use an explicitly Gaussian predictive law.  Keep the
                # requested specification separate so consumers cannot mistake a
                # t/skew-t request for calibrated non-Gaussian tail forecasts.
                "distribution": "normal",
                "requested_distribution": self.dist,
                "horizon": horizon,
                "fit_status": self.fit_status,
            }
            if quantiles is not None:
                levels = np.asarray(quantiles, dtype=float)
                if levels.ndim != 1 or np.any((levels <= 0.0) | (levels >= 1.0)):
                    raise ValueError("quantiles must lie strictly between 0 and 1")
                from scipy.stats import norm

                out["quantiles"] = (
                    self.last_sigma * norm.ppf(levels)[None, :] + self._mean_decimal()
                )
            return out
        # arch cannot analytically recurse EGARCH/APARCH beyond one step.
        # Use a deterministic simulation rather than returning an exception
        # or silently substituting a repeated one-step value.
        forecast_method = method
        if self.vol in _SIMULATED_MULTI_STEP_VOLS and horizon > 1 and method == "analytic":
            forecast_method = "simulation"
        random_state = np.random.RandomState(0 if seed is None else seed)
        forecast = self.result.forecast(
            horizon=horizon,
            method=forecast_method,
            simulations=simulations,
            random_state=random_state,
            reindex=False,
        )
        variance_percent = np.asarray(forecast.variance, dtype=float)
        variance = np.asarray(variance_percent[-1], dtype=float).reshape(-1) / (_GARCH_SCALE**2)
        variance = np.maximum(variance[:horizon], _GARCH_VARIANCE_FLOOR)
        out = {
            "variance": variance,
            "sigma": np.sqrt(variance),
            "cumulative_variance": np.cumsum(variance),
            "mean": self._mean_decimal(),
            "variance_units": "decimal_squared",
            "sigma_units": "decimal",
            "distribution": self.dist,
            "requested_distribution": self.dist,
            "horizon": horizon,
            "fit_status": self.fit_status,
        }
        if quantiles is not None:
            levels = np.asarray(quantiles, dtype=float)
            if levels.ndim != 1 or np.any((levels <= 0.0) | (levels >= 1.0)):
                raise ValueError("quantiles must lie strictly between 0 and 1")
            distribution = self.result.model.distribution
            names = distribution.parameter_names()
            params = [float(self.result.params[name]) for name in names]
            standardized = np.asarray(distribution.ppf(levels, params), dtype=float)
            out["quantiles"] = (
                self._mean_decimal() + np.sqrt(variance)[:, None] * standardized[None, :]
            )
        return out

    def _mean_decimal(self) -> float:
        if self.result is None or self.mean == "Zero":
            return 0.0
        try:
            return float(self.result.params["mu"]) / _GARCH_SCALE
        except (KeyError, TypeError, ValueError):
            return 0.0

    def pit(
        self, returns: NDArray[np.float64], sigma: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Evaluate predictive PIT values for returns and supplied sigmas."""
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float).reshape(-1)
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        out = np.full(values.size, np.nan)
        if self.result is None:
            from scipy.stats import norm

            out[valid] = norm.cdf((values[valid] - self._mean_decimal()) / scales[valid])
            return np.asarray(np.clip(out, 0.0, 1.0), dtype=float)
        distribution = self.result.model.distribution
        names = distribution.parameter_names()
        params = [float(self.result.params[name]) for name in names]
        standardized = (values[valid] - self._mean_decimal()) / scales[valid]
        out[valid] = distribution.cdf(standardized, params)
        return np.asarray(np.clip(out, 0.0, 1.0), dtype=float)

    def log_density(
        self, returns: NDArray[np.float64], sigma: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Log predictive density of decimal returns under the origin law.

        Sigma must be in decimal return units.  Fitted Student-t / skew-t
        scores use the ``arch`` standardized innovation (unit variance), not
        a textbook location-scale *t*.  Fallback fits are scored as Gaussian.
        Empty-valid inputs return NaNs rather than a silent zero score.
        """
        values = np.asarray(returns, dtype=float).reshape(-1)
        scales = (
            np.full(values.size, self.last_sigma)
            if sigma is None
            else np.asarray(sigma, dtype=float).reshape(-1)
        )
        if values.size != scales.size:
            raise ValueError("returns and sigma must have the same length")
        valid = np.isfinite(values) & np.isfinite(scales) & (scales > 0.0)
        out = np.full(values.size, np.nan)
        if not np.any(valid):
            return out
        mu = self._mean_decimal()
        resids = values[valid] - mu
        sigma2 = np.square(scales[valid])
        if self.result is None:
            z = resids / scales[valid]
            out[valid] = -0.5 * np.log(2.0 * np.pi) - np.log(scales[valid]) - 0.5 * z * z
            return out
        distribution = self.result.model.distribution
        names = distribution.parameter_names()
        params = [float(self.result.params[name]) for name in names]
        ll = np.asarray(
            distribution.loglikelihood(params, resids, sigma2, individual=True),
            dtype=float,
        ).reshape(-1)
        if ll.size != int(valid.sum()):
            raise ValueError("GARCH log-density size mismatch")
        out[valid] = ll
        return out

    def diagnostics(self) -> dict[str, Any]:
        """Return a small JSON/joblib-safe audit record for the fitted model."""
        return {
            "n_obs": int(self.n_obs),
            "fit_status": self.fit_status,
            "converged": bool(self.converged),
            "fallback_reason": self.fallback_reason,
            "returns_scale": self.returns_scale,
            "variance_units": "decimal_squared",
            "p": int(self.p),
            "q": int(self.q),
            "dist": self.dist,
            "vol": self.vol,
            "series_scope": getattr(self, "series_scope", "univariate_return_series"),
        }

    def assert_consumer_scope(self, consumer_scope: str) -> None:
        """Reject a pooled artifact at a per-security volatility consumer.

        Delegates to the shared fail-closed contract in
        ``quant_fund.models.vol_scope``: the matrix is symmetric (pooled
        artifacts reject per-security consumers AND ``per_security`` artifacts
        reject pooled/date-level consumers) and missing/unknown/corrupted scope
        tokens are rejected rather than guessed.
        """
        artifact_scope = getattr(self, "series_scope", UNIVARIATE_RETURN_SERIES_SCOPE)
        assert_scope_compatible(artifact_scope, consumer_scope)

    def in_sample_sigma_and_z(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return decimal in-sample sigma and standardized residuals.

        Requires a successful ``arch`` fit. Sample-sigma fallbacks fail closed
        so covariance consumers cannot treat EWMA or sample residuals as GARCH
        stage-1 innovations.
        """
        if self.result is None or self.fit_status != "fitted":
            reason = self.fallback_reason or self.fit_status
            raise ValueError(f"in-sample GARCH path requires a successful fit; {reason}")
        sigma = np.asarray(self.result.conditional_volatility, dtype=float).reshape(-1)
        sigma = sigma / _GARCH_SCALE
        try:
            z = np.asarray(self.result.std_resid, dtype=float).reshape(-1)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("in-sample GARCH standardized residuals are unavailable") from exc
        if sigma.size == 0 or z.size != sigma.size:
            raise ValueError("in-sample GARCH path length mismatch")
        if not np.isfinite(sigma).all() or not np.isfinite(z).all() or np.any(sigma <= 0.0):
            raise ValueError("in-sample GARCH path is non-finite or non-positive")
        return sigma, z

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compatibility adapter: current-origin sigma for each requested row."""
        n = int(np.asarray(x).shape[0])
        return np.full(n, float(self.forecast(horizon=1)["sigma"][0]))

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="volatility",
            name=f"{self.vol}{self.p}-{self.q}_{self.dist}",
            version="v2",
            extra={
                "p": self.p,
                "q": self.q,
                "dist": self.dist,
                "vol": self.vol,
                "series_scope": getattr(self, "series_scope", "univariate_return_series"),
            },
        )


class HARVol(JoblibMixin):
    def __init__(self, use_log: bool = True) -> None:
        self.use_log = use_log
        self.model = LinearRegression()
        self.fitted = False

    @staticmethod
    def har_design(rv: NDArray[np.float64]) -> NDArray[np.float64]:
        """Build HAR regressors available immediately before each target time.

        Row ``t`` forecasts realized volatility at ``t`` using ``rv[t-1]`` and
        trailing averages ending at ``t-1``.  The prior implementation included
        ``rv[t]`` in every row, making in-sample forecasts contemporaneously
        leak their target.
        """
        values = np.asarray(rv, dtype=float).reshape(-1)
        n = values.size
        lagged = np.full(n, np.nan)
        if n > 1:
            lagged[1:] = values[:-1]

        def trailing_mean(window: int) -> NDArray[np.float64]:
            result = np.full(n, np.nan)
            if n <= window or window < 1:
                return result
            # Index 0 of ``lagged`` is the NaN pad. Windows that start at
            # index 1 are the same slices the Python loop averaged.
            view = np.lib.stride_tricks.sliding_window_view(lagged, window)
            result[window:] = np.mean(view[1:], axis=1)
            return result

        return np.column_stack([np.ones(n), lagged, trailing_mean(5), trailing_mean(22)])

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> HARVol:
        """Fit on the declared, origin-time volatility feature matrix.

        ``train_volatility`` supplies the five-column volatility feature contract.
        The previous implementation ignored those features and rebuilt regressors
        from the forward target ``y``.  Besides creating a feature-count mismatch at
        prediction time, that made the model use future observations as regressors.
        HAR's standalone lag builder remains available for callers that explicitly
        construct a causal HAR design, but the estimator itself is a normal
        supervised model: fit and predict consume the same ``x`` schema.
        """
        features = np.asarray(x, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        if features.ndim != 2:
            raise ValueError("x must be a two-dimensional feature matrix")
        if features.shape[0] != target.size:
            raise ValueError("x and y must have the same number of rows")
        mask = np.isfinite(features).all(axis=1) & np.isfinite(target)
        if mask.sum() < 10:
            return self
        response = np.log(np.clip(target, 1e-12, None)) if self.use_log else target
        self.model.fit(features[mask], response[mask])
        self.fitted = True
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        features = np.asarray(x, dtype=float)
        if features.ndim != 2:
            raise ValueError("x must be a two-dimensional feature matrix")
        if not self.fitted:
            return np.full(features.shape[0], np.nan)
        pred = self.model.predict(features)
        if self.use_log:
            pred = np.exp(pred)
        return np.asarray(pred, dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="volatility",
            name="har_rv",
            version="v2",
            extra={"input_contract": "supervised_volatility_features"},
        )


class TreeVol(JoblibMixin):
    def __init__(self, backend: str = "lightgbm", seed: int = 42) -> None:
        self.backend = backend
        if backend == "xgboost":
            from xgboost import XGBRegressor

            self.model = XGBRegressor(
                n_estimators=80, max_depth=3, n_jobs=1, random_state=seed, verbosity=0
            )
        else:
            from lightgbm import LGBMRegressor

            self.model = LGBMRegressor(
                n_estimators=80,
                num_leaves=15,
                n_jobs=1,
                random_state=seed,
                verbosity=-1,
            )

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64], **kwargs: Any) -> TreeVol:
        xx, yy, _ = _finite(x, y)
        self.model.fit(xx, yy)
        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.model.predict(x), dtype=float)

    def metadata(self) -> ModelMeta:
        return ModelMeta(family="volatility", name=f"{self.backend}_vol", version="v1")
