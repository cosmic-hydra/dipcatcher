"""Keyed per-security volatility forecasts — scope token ``per_security``.

This is the keyed, per-instrument volatility lane: one variance forecast per
symbol key with an explicit key axis, unlike the date-level equal-weight
cross-sectional pooled overlay (``date_level_equal_weight_cross_section``)
in ``quant_fund.models.volatility``.  Per-asset risk scaling, per-name vol
targeting and per-name ES/VaR consume THIS lane — never the pooled overlay.

UNITS CONTRACT (explicit, never silently rescaled)
--------------------------------------------------
``forecast(...)["variance"]`` is a **variance** (sigma^2) in decimal-squared
log-return units, tagged ``variance_units="decimal_squared"``.  It is NOT a
volatility.  ``forecast(...)["sigma"]`` is the single documented square-root
convenience in decimal units (``sigma_units="decimal"``); ``["quantiles"]`` are
predictive return quantiles in decimal units (``quantile_units="decimal"``).
A variance stays a variance and a volatility stays a volatility end to end;
the only sanctioned conversion is the explicit
``quant_fund.metrics.vol_eval.rescale_units``.  ``coerce_units`` preserves
values bit-identically within one unit and raises ``VolUnitsError`` across
units, so a variance in produces a variance out with no silent rescale.

PIT CONTRACT (trailing / as-of windows only)
--------------------------------------------
Every fit sees only returns with ``time`` strictly BEFORE the forecast origin
(``as_of``).  Forward labels (``KeyedReturnPanel.labels``) are EVALUATION-ONLY
metadata: no fitting or forecasting path reads them, so shifting labels
forward in time leaves every forecast bit-identical (proven by assertion in
``tests/unit/models/test_vol_per_security.py``, including a leaky-mutant
harness that proves the proof can fail).  A key that cannot be forecast fails
closed to honest NaN for THAT key only, without poisoning the other keys.

Supported per-key specifications: ``garch`` (``GARCHVol`` GARCH(1,1)),
``ewma`` (RiskMetrics recursion in ``models.volatility.ewma_variance``),
``har`` (``HARVol`` on the trailing squared-return variance proxy), and the
``garch_ext`` QMLE families ``aparch`` / ``figarch`` (one-step only).

Scoring is proper-scores-only (QLIKE on variance, pinball on quantiles) via
``quant_fund.metrics.vol_eval``.  Nothing here is a live-trading,
profitability or market-evidence claim; seeded SYNTHETIC panels in tests are
correctness fixtures only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from quant_fund.metrics.vol_eval import (
    VARIANCE_UNITS,
    VOLATILITY_UNITS,
)
from quant_fund.models.base import ModelMeta
from quant_fund.models.garch_ext import figarch_variance, fit_aparch, fit_figarch
from quant_fund.models.vol_scope import (
    PER_SECURITY_CONSUMER,
    PER_SECURITY_SCOPE,
    assert_scope_compatible,
)
from quant_fund.models.volatility import GARCHVol, HARVol, ewma_variance

Array = NDArray[np.float64]

KEYED_VOL_SPECS: Final[tuple[str, ...]] = ("ewma", "garch", "har", "aparch", "figarch")
_ONE_STEP_ONLY_SPECS: Final[frozenset[str]] = frozenset({"aparch", "figarch"})
_GARCH_DISTS: Final[tuple[str, ...]] = ("normal", "t", "skewt")
STATUS_OK: Final[str] = "ok"
_MIN_HAR_ROWS: Final[int] = 10


class PerKeyVolError(ValueError):
    """One key failed to fit or forecast; surfaced as per-key honest NaN."""


class _KeyFit(Protocol):
    """One fitted per-instrument volatility model."""

    def forecast(
        self, horizon: int, quantiles: tuple[float, ...] | None, seed: int | None
    ) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Keyed return panel (explicit key axis, integer event-time axis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyedReturnPanel:
    """Long-format keyed return panel with an explicit key axis.

    ``returns`` are decimal log returns.  ``labels`` (optional) are FORWARD
    realized-variance labels in ``VARIANCE_UNITS`` indexed at their forecast
    origin — EVALUATION ONLY; forecasting never reads them.  ``times`` is a
    strictly increasing integer event-time axis per key.  ``source_label``
    records provenance; test panels MUST say ``"SYNTHETIC"`` (synthetic
    results are correctness tests, never market evidence).
    """

    keys: NDArray[Any]
    times: NDArray[Any]
    returns: Array
    labels: Array | None = None
    source_label: str = "UNSPECIFIED"

    def __post_init__(self) -> None:
        _validate_panel(self.keys, self.times, self.returns, self.labels)

    def key_names(self) -> tuple[str, ...]:
        """Sorted unique keys — the explicit key axis."""
        return tuple(sorted({str(key) for key in np.asarray(self.keys)}))

    def key_returns(self, key: str) -> Array:
        """All returns of one key in strictly increasing time order."""
        rows = self._rows(key)
        return np.asarray(self.returns, dtype=float)[rows]

    def trailing(self, key: str, as_of: int) -> Array:
        """Returns of one key with ``time`` STRICTLY BEFORE ``as_of`` (as-of)."""
        rows = self._rows(key)
        times = np.asarray(self.times)[rows]
        keep = rows[times < int(as_of)]
        return np.asarray(self.returns, dtype=float)[keep]

    def label_at(self, key: str, time: int) -> float:
        """Forward label at (key, time); NaN when absent. EVALUATION ONLY."""
        return self._value_at(self.labels, key, time)

    def realized_return_at(self, key: str, time: int) -> float:
        """Realized return at (key, time); NaN when absent. EVALUATION ONLY."""
        return self._value_at(self.returns, key, time)

    def with_labels(self, labels: Array | None) -> KeyedReturnPanel:
        """Same panel with replaced forward labels (proof-test helper)."""
        return KeyedReturnPanel(
            keys=self.keys,
            times=self.times,
            returns=self.returns,
            labels=labels,
            source_label=self.source_label,
        )

    def with_returns(self, returns: Array) -> KeyedReturnPanel:
        """Same panel with replaced returns (proof-test helper)."""
        return KeyedReturnPanel(
            keys=self.keys,
            times=self.times,
            returns=returns,
            labels=self.labels,
            source_label=self.source_label,
        )

    def _rows(self, key: str) -> NDArray[np.int64]:
        return _sorted_rows(np.asarray(self.keys), np.asarray(self.times), key)

    def _value_at(self, values: Array | None, key: str, time: int) -> float:
        if values is None:
            return float("nan")
        rows = self._rows(key)
        times = np.asarray(self.times)[rows]
        hit = rows[times == int(time)]
        if hit.size == 0:
            return float("nan")
        return float(np.asarray(values, dtype=float)[hit[0]])


def _sorted_rows(keys: NDArray[Any], times: NDArray[Any], key: str) -> NDArray[np.int64]:
    """Row indices of one key, stably sorted by event time."""
    rows = np.flatnonzero(keys == key)
    order = np.argsort(np.asarray(times)[rows], kind="stable")
    return np.asarray(rows[order], dtype=np.int64)


def _validate_panel(
    keys: NDArray[Any], times: NDArray[Any], returns: Array, labels: Array | None
) -> None:
    """Fail closed on malformed keyed panels (shapes, keys, time axis)."""
    keys_array = np.asarray(keys)
    times_array = np.asarray(times)
    returns_array = np.asarray(returns, dtype=float)
    if keys_array.ndim != 1 or times_array.ndim != 1 or returns_array.ndim != 1:
        raise ValueError("panel arrays must be one-dimensional")
    if not (keys_array.size == times_array.size == returns_array.size):
        raise ValueError("panel keys, times and returns must have the same length")
    if keys_array.size == 0:
        raise ValueError("panel must be non-empty")
    if labels is not None and np.asarray(labels, dtype=float).shape != returns_array.shape:
        raise ValueError("panel labels must match returns")
    _validate_keys(keys_array)
    _validate_time_axis(keys_array, times_array)


def _validate_keys(keys: NDArray[Any]) -> None:
    """Keys must be non-empty strings (the explicit key axis)."""
    for key in keys:
        if not isinstance(key, str) or not key.strip():
            raise ValueError("panel keys must be non-empty strings")


def _validate_time_axis(keys: NDArray[Any], times: NDArray[Any]) -> None:
    """Times must be integers and strictly increasing within each key."""
    if not np.issubdtype(np.asarray(times).dtype, np.integer):
        raise ValueError("panel times must be integers")
    for key in sorted({str(item) for item in keys}):
        rows = _sorted_rows(keys, times, key)
        if rows.size > 1 and np.any(np.diff(np.asarray(times)[rows]) <= 0):
            raise ValueError(f"panel times for key {key!r} must be strictly increasing")


# ---------------------------------------------------------------------------
# Per-key fitters (dispatch table + tiny adapters)
# ---------------------------------------------------------------------------


def _finite_history(history: Array, min_obs: int) -> Array:
    """Finite trailing returns; insufficient history fails the key honestly."""
    values = np.asarray(history, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size < min_obs:
        raise PerKeyVolError(f"insufficient_observations:{finite.size}<{min_obs}")
    return finite


def _failure_reason(exc: Exception) -> str:
    """Stable per-key failure reason for the status channel."""
    return str(exc) if isinstance(exc, PerKeyVolError) else type(exc).__name__


def _ewma_path_variance(returns: Array, lam: float, horizon: int) -> Array:
    """RiskMetrics one-step variance; multi-step is flat (documented)."""
    trailing = np.asarray(ewma_variance(returns, lam), dtype=float)
    one_step = lam * trailing[-1] + (1.0 - lam) * returns[-1] ** 2
    return np.full(horizon, float(one_step))


def _standardized_residual_quantiles(
    returns: Array, sigma: Array, levels: tuple[float, ...]
) -> Array:
    """Empirical quantiles of PIT standardized residuals (distribution-free)."""
    valid = np.isfinite(returns) & np.isfinite(sigma) & (sigma > 0.0)
    residuals = returns[valid] / sigma[valid]
    if residuals.size == 0:
        raise PerKeyVolError("no_standardized_residuals")
    return np.asarray(np.quantile(residuals, levels), dtype=float)


class _EwmaKeyFit:
    """EWMA per key: variance recursion on trailing returns."""

    def __init__(self, returns: Array, lam: float) -> None:
        self._returns = returns
        self._lam = lam

    def forecast(
        self, horizon: int, quantiles: tuple[float, ...] | None, seed: int | None
    ) -> dict[str, Any]:
        variance = _ewma_path_variance(self._returns, self._lam, horizon)
        out: dict[str, Any] = {"variance": variance, "mean": 0.0, "quantiles": None}
        if quantiles is not None:
            sigma_in_sample = np.sqrt(ewma_variance(self._returns, self._lam))
            levels = _standardized_residual_quantiles(self._returns, sigma_in_sample, quantiles)
            out["quantiles"] = np.sqrt(variance)[:, None] * levels[None, :]
        return out


class _GarchKeyFit:
    """GARCH(1,1)-family per key via ``GARCHVol`` (analytic variance term structure)."""

    def __init__(self, model: GARCHVol) -> None:
        self._model = model

    def forecast(
        self, horizon: int, quantiles: tuple[float, ...] | None, seed: int | None
    ) -> dict[str, Any]:
        raw = self._model.forecast(horizon=horizon, quantiles=quantiles, seed=seed)
        return {
            "variance": np.asarray(raw["variance"], dtype=float).reshape(-1),
            "mean": float(raw["mean"]),
            "quantiles": None if quantiles is None else np.asarray(raw["quantiles"], dtype=float),
        }


class _HarKeyFit:
    """HAR-style per key on the trailing squared-return variance proxy."""

    def __init__(self, model: HARVol, returns: Array, rv: Array, sigma: Array) -> None:
        self._model = model
        self._returns = returns
        self._rv = rv
        self._sigma = sigma

    def forecast(
        self, horizon: int, quantiles: tuple[float, ...] | None, seed: int | None
    ) -> dict[str, Any]:
        row = _har_next_row(self._rv)
        one_step = max(float(self._model.predict(row[None, :])[0]), 0.0)
        variance = np.full(horizon, one_step)
        out: dict[str, Any] = {"variance": variance, "mean": 0.0, "quantiles": None}
        if quantiles is not None:
            levels = _standardized_residual_quantiles(self._returns, self._sigma, quantiles)
            out["quantiles"] = np.sqrt(variance)[:, None] * levels[None, :]
        return out


class _QmleKeyFit:
    """APARCH/FIGARCH QMLE per key (``garch_ext``): one-step variance only."""

    def __init__(self, family: str, result: dict[str, Array], returns: Array) -> None:
        self._family = family
        self._result = result
        self._returns = returns

    def forecast(
        self, horizon: int, quantiles: tuple[float, ...] | None, seed: int | None
    ) -> dict[str, Any]:
        one_step = _QMLE_ONE_STEP[self._family](self._result, self._returns)
        return {
            "variance": np.full(horizon, one_step),
            "mean": 0.0,
            "quantiles": None,
        }


def _har_next_row(rv: Array) -> Array:
    """Next causal HAR design row: lagged RV plus trailing 5/22 means."""
    return np.array([1.0, rv[-1], float(np.mean(rv[-5:])), float(np.mean(rv[-22:]))])


def _fit_ewma_key(returns: Array, config: PerSecurityVol) -> _KeyFit:
    return _EwmaKeyFit(returns, config.lam)


def _fit_garch_key(returns: Array, config: PerSecurityVol) -> _KeyFit:
    model = GARCHVol(p=config.p, q=config.q, dist=config.dist, min_obs=config.min_obs)
    model.fit_returns(returns)
    if model.fit_status != "fitted":
        raise PerKeyVolError(f"fallback:{model.fallback_reason or model.fit_status}")
    return _GarchKeyFit(model)


def _fit_har_key(returns: Array, config: PerSecurityVol) -> _KeyFit:
    rv = returns * returns
    design = HARVol.har_design(rv)
    valid = np.isfinite(design).all(axis=1) & np.isfinite(rv)
    if int(valid.sum()) < _MIN_HAR_ROWS:
        raise PerKeyVolError(f"insufficient_har_rows:{int(valid.sum())}")
    model = HARVol(use_log=True)
    model.fit(design[valid], rv[valid])
    if not model.fitted:
        raise PerKeyVolError("har_fit_failed")
    sigma = np.full(rv.size, np.nan)
    sigma[valid] = np.sqrt(model.predict(design[valid]))
    return _HarKeyFit(model, returns, rv, sigma)


def _fit_aparch_key(returns: Array, config: PerSecurityVol) -> _KeyFit:
    return _QmleKeyFit("aparch", fit_aparch(returns), returns)


def _fit_figarch_key(returns: Array, config: PerSecurityVol) -> _KeyFit:
    return _QmleKeyFit("figarch", fit_figarch(returns), returns)


_KEY_FIT_DISPATCH: Final[dict[str, Any]] = {
    "ewma": _fit_ewma_key,
    "garch": _fit_garch_key,
    "har": _fit_har_key,
    "aparch": _fit_aparch_key,
    "figarch": _fit_figarch_key,
}


def _aparch_one_step(result: dict[str, Array], returns: Array) -> float:
    """Ding-Granger-Engle one-step APARCH variance (decimal squared)."""
    omega, alpha = float(result["omega"][0]), float(result["alpha"][0])
    gamma, beta, delta = (
        float(result["gamma"][0]),
        float(result["beta"][0]),
        float(result["delta"][0]),
    )
    sigma_last = float(np.sqrt(result["sigma2"][-1]))
    shock = float(returns[-1])
    news = (abs(shock) - gamma * shock) ** delta
    powered = omega + alpha * news + beta * sigma_last**delta
    return float(powered ** (2.0 / delta))


def _figarch_one_step(result: dict[str, Array], returns: Array) -> float:
    """BBM one-step FIGARCH variance from the extended truncated recursion."""
    phi, d, beta = float(result["phi"][0]), float(result["d"][0]), float(result["beta"][0])
    extended = np.concatenate([returns, np.zeros(1)])
    path = figarch_variance(extended, phi, d, beta, sigma2_0=float(result["sigma2"][0]))
    return float(path[-1])


_QMLE_ONE_STEP: Final[dict[str, Any]] = {
    "aparch": _aparch_one_step,
    "figarch": _figarch_one_step,
}


# ---------------------------------------------------------------------------
# Forecast containers (per-key ``forecast(...)["variance"]``)
# ---------------------------------------------------------------------------


def _forecast_dict(key: str, status: str, raw: dict[str, Any], horizon: int) -> dict[str, Any]:
    """Assemble the per-key forecast dict; variance stays in its own units."""
    variance = np.asarray(raw["variance"], dtype=float).reshape(-1)
    out: dict[str, Any] = {
        "key": key,
        "scope": PER_SECURITY_SCOPE,
        "series_scope": PER_SECURITY_SCOPE,
        "variance": variance,
        "variance_units": VARIANCE_UNITS,
        "sigma": np.sqrt(variance),
        "sigma_units": VOLATILITY_UNITS,
        "cumulative_variance": np.cumsum(variance),
        "mean": float(raw.get("mean", 0.0)),
        "horizon": horizon,
        "fit_status": status,
    }
    quantiles = raw.get("quantiles")
    if quantiles is not None:
        out["quantiles"] = np.asarray(quantiles, dtype=float)
        out["quantile_units"] = VOLATILITY_UNITS
    return out


def _failed_forecast(
    key: str, status: str, horizon: int, levels: tuple[float, ...] | None
) -> dict[str, Any]:
    """Honest-NaN forecast dict for one failed key."""
    quantiles = None if levels is None else np.full((horizon, len(levels)), np.nan)
    raw = {"variance": np.full(horizon, np.nan), "mean": float("nan"), "quantiles": quantiles}
    return _forecast_dict(key, status, raw, horizon)


def _validate_levels(quantiles: Sequence[float] | None) -> tuple[float, ...] | None:
    """Validate quantile levels strictly inside (0, 1)."""
    if quantiles is None:
        return None
    levels = tuple(float(level) for level in quantiles)
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantiles must lie strictly between 0 and 1")
    return levels


def _validate_forecast_request(spec: str, horizon: int, quantiles: Sequence[float] | None) -> None:
    """Fail closed on invalid horizons and unsupported per-key requests."""
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if horizon > 1 and spec in _ONE_STEP_ONLY_SPECS:
        raise ValueError(f"spec {spec!r} supports horizon=1 only (documented one-step QMLE)")
    if quantiles is not None and spec in _ONE_STEP_ONLY_SPECS:
        raise ValueError(f"spec {spec!r} has no documented quantile law; refusing to invent one")


@dataclass(frozen=True)
class KeyedVarianceForecast:
    """Batch per-key variance forecasts with an explicit key axis.

    ``variance`` is (n_keys, horizon) in ``VARIANCE_UNITS`` (decimal squared);
    rows for failed keys are honest NaN with their reason in ``status``.
    """

    keys: tuple[str, ...]
    variance: Array
    status: tuple[str, ...]
    horizon: int
    means: Array
    quantiles: Array | None = None
    units: str = VARIANCE_UNITS
    scope: str = PER_SECURITY_SCOPE
    source_label: str = "UNSPECIFIED"

    def forecast(self, key: str) -> dict[str, Any]:
        """Per-key ``forecast(...)["variance"]`` view of this batch result."""
        try:
            index = self.keys.index(key)
        except ValueError:
            raise ValueError(f"unknown key {key!r}; known keys: {list(self.keys)}") from None
        raw = {
            "variance": self.variance[index],
            "mean": float(self.means[index]),
            "quantiles": None if self.quantiles is None else self.quantiles[index],
        }
        return _forecast_dict(key, self.status[index], raw, self.horizon)


def _validate_garch_order(p: int, q: int) -> None:
    if (
        isinstance(p, bool)
        or isinstance(q, bool)
        or not isinstance(p, int)
        or not isinstance(q, int)
    ):
        raise ValueError("p and q must be positive integers")
    if p < 1 or q < 1:
        raise ValueError("p and q must be positive integers")


def _validate_spec(spec: str) -> str:
    if not isinstance(spec, str) or spec not in KEYED_VOL_SPECS:
        raise ValueError(f"spec must be one of {list(KEYED_VOL_SPECS)}")
    return spec


class PerSecurityVol:
    """Keyed per-instrument volatility forecaster (``series_scope=per_security``).

    Fit one model per symbol key on that key's own trailing returns and read
    per-key ``forecast(key, ...)["variance"]`` — a **variance** (sigma^2) in
    decimal-squared units (see module docstring for the full units and PIT
    contracts).  Per-key failures degrade to honest NaN for that key only.

    Scope: the artifact is stamped ``series_scope="per_security"`` and the
    symmetric fail-closed matrix from ``models.vol_scope`` admits it ONLY at
    ``per_security`` consumers; pooled/date-level consumers reject it and this
    model rejects pooled consumers.  This is not a live-performance claim.
    """

    series_scope: str = PER_SECURITY_SCOPE

    def __init__(
        self,
        spec: str = "garch",
        *,
        lam: float = 0.94,
        min_obs: int = 50,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        seed: int = 0,
    ) -> None:
        self.spec = _validate_spec(spec)
        self.lam = _validate_lam(lam)
        self.min_obs = _validate_min_obs(min_obs)
        _validate_garch_order(p, q)
        self.p, self.q = int(p), int(q)
        self.dist = _validate_dist(dist)
        self.seed = int(seed)
        self._fits: dict[str, _KeyFit] = {}
        self._status: dict[str, str] = {}
        self._keys: tuple[str, ...] = ()

    def clone(self) -> PerSecurityVol:
        """Fresh unfitted copy with the same specification (walk-forward)."""
        return PerSecurityVol(
            self.spec,
            lam=self.lam,
            min_obs=self.min_obs,
            p=self.p,
            q=self.q,
            dist=self.dist,
            seed=self.seed,
        )

    def fit(self, panel: KeyedReturnPanel, *, as_of: int | None = None) -> PerSecurityVol:
        """Fit each key on its own history (strictly before ``as_of`` when given).

        Label-free: ``panel.labels`` are never read.  A failing key records a
        ``failed:<reason>`` status instead of raising, so one bad key cannot
        poison the other keys.
        """
        self._fits = {}
        self._status = {}
        self._keys = panel.key_names()
        for key in self._keys:
            history = panel.key_returns(key) if as_of is None else panel.trailing(key, int(as_of))
            self._fit_one(key, history)
        return self

    def _fit_one(self, key: str, history: Array) -> None:
        try:
            values = _finite_history(history, self.min_obs)
            self._fits[key] = _KEY_FIT_DISPATCH[self.spec](values, self)
            self._status[key] = STATUS_OK
        except (
            PerKeyVolError,
            ValueError,
            ArithmeticError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as exc:
            # Per-key isolation: one bad series must not poison the panel fit.
            self._status[key] = f"failed:{_failure_reason(exc)}"

    def forecast(
        self,
        key: str,
        horizon: int = 1,
        *,
        quantiles: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Per-key ``forecast(...)["variance"]`` in decimal-squared units.

        Returns honest NaN for a failed key instead of raising, so one bad key
        cannot take down the rest.  Unknown keys raise (caller error).
        """
        _validate_forecast_request(self.spec, horizon, quantiles)
        levels = _validate_levels(quantiles)
        if key not in self._status:
            raise ValueError(f"key {key!r} is not fitted; fitted keys: {list(self._keys)}")
        if self._status[key] != STATUS_OK:
            return _failed_forecast(key, self._status[key], horizon, levels)
        try:
            raw = self._fits[key].forecast(horizon, levels, self.seed if seed is None else seed)
        except (
            PerKeyVolError,
            ValueError,
            ArithmeticError,
            RuntimeError,
            np.linalg.LinAlgError,
        ) as exc:
            return _failed_forecast(key, f"failed:{_failure_reason(exc)}", horizon, levels)
        return _forecast_dict(key, self._status[key], raw, horizon)

    def forecast_keys(
        self,
        horizon: int = 1,
        *,
        keys: Sequence[str] | None = None,
        quantiles: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> KeyedVarianceForecast:
        """Batch per-key variance forecasts with honest NaN on failed keys."""
        _validate_forecast_request(self.spec, horizon, quantiles)
        levels = _validate_levels(quantiles)
        targets = self._keys if keys is None else tuple(str(key) for key in keys)
        variance = np.full((len(targets), horizon), np.nan)
        means = np.full(len(targets), np.nan)
        quantile_rows = (
            None if levels is None else np.full((len(targets), horizon, len(levels)), np.nan)
        )
        status: list[str] = []
        for index, key in enumerate(targets):
            forecast = self.forecast(key, horizon, quantiles=levels, seed=seed)
            variance[index] = forecast["variance"]
            means[index] = forecast["mean"]
            status.append(str(forecast["fit_status"]))
            if quantile_rows is not None:
                quantile_rows[index] = forecast.get("quantiles", quantile_rows[index])
        return KeyedVarianceForecast(
            keys=tuple(targets),
            variance=variance,
            status=tuple(status),
            horizon=horizon,
            means=means,
            quantiles=quantile_rows,
        )

    def assert_consumer_scope(self, consumer_scope: str) -> None:
        """Symmetric fail-closed scope check (``per_security`` consumers only)."""
        assert_scope_compatible(self.series_scope, consumer_scope)

    def diagnostics(self) -> dict[str, Any]:
        """JSON-safe audit record: scope, units and per-key fit status."""
        return {
            "series_scope": self.series_scope,
            "scope": PER_SECURITY_SCOPE,
            "key_axis": "symbol",
            "spec": self.spec,
            "variance_units": VARIANCE_UNITS,
            "sigma_units": VOLATILITY_UNITS,
            "n_keys": len(self._keys),
            "status": dict(self._status),
            "min_obs": self.min_obs,
            "label_usage": "none_forecasts_are_label_free",
            "pit_contract": "trailing_returns_strictly_before_as_of",
        }

    def metadata(self) -> ModelMeta:
        return ModelMeta(
            family="volatility",
            name=f"per_security_{self.spec}",
            version="v1",
            extra={"series_scope": self.series_scope, "variance_units": VARIANCE_UNITS},
        )


def _validate_lam(lam: float) -> float:
    if not np.isfinite(lam) or not 0.0 <= lam <= 1.0:
        raise ValueError("lam must be finite and between 0 and 1")
    return float(lam)


def _validate_min_obs(min_obs: int) -> int:
    if isinstance(min_obs, bool) or not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError("min_obs must be an integer >= 2")
    return int(min_obs)


def _validate_dist(dist: str) -> str:
    if not isinstance(dist, str) or dist.lower() not in _GARCH_DISTS:
        raise ValueError(f"dist must be one of {list(_GARCH_DISTS)}")
    return dist.lower()


# ---------------------------------------------------------------------------
# Label-free walk-forward with evaluation-only labels alongside
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyedWalkForwardResult:
    """Per-key, per-origin walk-forward forecasts plus EVALUATION-ONLY labels.

    ``one_step_variance`` / ``cumulative_variance`` are (n_keys, n_origins) in
    ``VARIANCE_UNITS``; ``cumulative_variance`` is the h-bar realized-variance
    target (sum of the horizon one-step variances, zero-autocovariance
    approximation — same convention as ``GARCHVol``).  ``realized_variance``
    (forward labels) and ``realized_return`` are EVALUATION ONLY and never
    feed any forecast.
    """

    keys: tuple[str, ...]
    origins: Array
    one_step_variance: Array
    cumulative_variance: Array
    realized_variance: Array
    realized_return: Array
    status: dict[str, tuple[str, ...]]
    horizon: int
    quantiles: Array | None = None
    units: str = VARIANCE_UNITS
    scope: str = PER_SECURITY_SCOPE
    source_label: str = "UNSPECIFIED"


def walk_forward_per_security(
    panel: KeyedReturnPanel,
    model: PerSecurityVol,
    *,
    origins: Sequence[int],
    horizon: int = 1,
    quantiles: Sequence[float] | None = None,
    seed: int | None = None,
) -> KeyedWalkForwardResult:
    """Causal per-key walk-forward: refit each key on strictly-prior returns.

    At every origin each key is refit on ``panel.trailing(key, origin)`` and
    forecast for the horizon realized variance starting at that origin.  The
    forward labels are attached to the result for scoring ONLY — no fitting or
    forecasting path reads them (see the no-forward-label proof tests).
    """
    _validate_forecast_request(model.spec, horizon, quantiles)
    levels = _validate_levels(quantiles)
    keys = panel.key_names()
    origin_values = _validate_origins(origins)
    shape = (len(keys), origin_values.size)
    one_step = np.full(shape, np.nan)
    cumulative = np.full(shape, np.nan)
    realized_variance = np.full(shape, np.nan)
    realized_return = np.full(shape, np.nan)
    quantile_rows = None if levels is None else np.full((*shape, len(levels)), np.nan)
    status: dict[str, list[str]] = {key: [] for key in keys}
    for column, origin in enumerate(origin_values):
        _capture_origin(
            panel,
            model,
            keys,
            int(origin),
            column,
            horizon,
            levels,
            seed,
            one_step,
            cumulative,
            realized_variance,
            realized_return,
            quantile_rows,
            status,
        )
    return KeyedWalkForwardResult(
        keys=keys,
        origins=origin_values,
        one_step_variance=one_step,
        cumulative_variance=cumulative,
        realized_variance=realized_variance,
        realized_return=realized_return,
        status={key: tuple(rows) for key, rows in status.items()},
        horizon=horizon,
        quantiles=quantile_rows,
        source_label=panel.source_label,
    )


def _capture_origin(
    panel: KeyedReturnPanel,
    model: PerSecurityVol,
    keys: tuple[str, ...],
    origin: int,
    column: int,
    horizon: int,
    levels: tuple[float, ...] | None,
    seed: int | None,
    one_step: Array,
    cumulative: Array,
    realized_variance: Array,
    realized_return: Array,
    quantile_rows: Array | None,
    status: dict[str, list[str]],
) -> None:
    """Fit and capture ONE walk-forward origin for every key."""
    fitted = model.clone().fit(panel, as_of=origin)
    batch = fitted.forecast_keys(horizon=horizon, keys=keys, quantiles=levels, seed=seed)
    one_step[:, column] = batch.variance[:, 0]
    cumulative[:, column] = np.cumsum(batch.variance, axis=1)[:, -1]
    if quantile_rows is not None and batch.quantiles is not None:
        quantile_rows[:, column, :] = batch.quantiles[:, 0, :]
    for row, key in enumerate(keys):
        realized_variance[row, column] = panel.label_at(key, origin)
        realized_return[row, column] = panel.realized_return_at(key, origin)
        status[key].append(batch.status[row])


def _validate_origins(origins: Sequence[int]) -> Array:
    """Origins must be a non-empty, strictly increasing integer sequence."""
    values = np.asarray(origins, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("origins must be non-empty")
    if np.any(np.diff(values) <= 0):
        raise ValueError("origins must be strictly increasing")
    return np.asarray(values, dtype=float)


__all__ = [
    "KEYED_VOL_SPECS",
    "KeyedReturnPanel",
    "KeyedVarianceForecast",
    "KeyedWalkForwardResult",
    "PER_SECURITY_CONSUMER",
    "PER_SECURITY_SCOPE",
    "PerKeyVolError",
    "PerSecurityVol",
    "VARIANCE_UNITS",
    "VOLATILITY_UNITS",
    "walk_forward_per_security",
]
