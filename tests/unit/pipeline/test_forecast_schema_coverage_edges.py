"""Validator arms on AssetForecast and MarketState that the happy path never takes."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from quant_fund.schemas.forecast import AssetForecast, MarketState

ASOF = datetime(2020, 1, 2, tzinfo=UTC)
BASE = {"security_id": "A", "symbol": "A", "asof": ASOF, "model_version": "v1"}


def test_numeric_maps_and_impact_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="numeric forecast values must be finite"):
        AssetForecast(**BASE, expected_returns={"1d": float("nan")})
    with pytest.raises(ValidationError, match="expected_impact_bps"):
        AssetForecast(**BASE, expected_impact_bps={"1d": float("inf")})
    ok = AssetForecast(**BASE, expected_impact_bps=None)
    assert ok.expected_impact_bps is None


def test_intervals_require_alpha_when_bounds_are_present() -> None:
    with pytest.raises(ValidationError, match="interval_alpha is required"):
        AssetForecast(**BASE, interval_lo={"1d": -1.0}, interval_hi={"1d": 1.0})


def _garch(**overrides: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "garch_market_sigma": 0.01,
        "garch_market_variance": 0.0001,
        "garch_cumulative_variance": 0.0002,
        "garch_horizon": 5,
        "garch_series_scope": "market",
        "garch_fit_status": "ok",
        "garch_n_obs": 40,
        "market_risk_overlay": "garch",
    }
    fields.update(overrides)
    return fields


def test_garch_overlay_fields_fail_closed_and_accept_a_complete_set() -> None:
    with pytest.raises(ValidationError, match="strictly positive"):
        MarketState(asof=ASOF, forecasts=[], **_garch(garch_market_sigma=0.0))
    with pytest.raises(ValidationError, match="positive integers"):
        MarketState(asof=ASOF, forecasts=[], **_garch(garch_horizon=0))
    with pytest.raises(ValueError, match="positive integers"):
        MarketState.validate_garch_positive_ints(True)
    with pytest.raises(ValidationError, match="non-empty strings"):
        MarketState(asof=ASOF, forecasts=[], **_garch(garch_fit_status="  "))
    with pytest.raises(ValidationError, match="non-empty string"):
        MarketState(asof=ASOF, forecasts=[], **_garch(market_risk_overlay=""))
    with pytest.raises(ValidationError, match="must be one of"):
        MarketState(asof=ASOF, forecasts=[], **_garch(market_risk_overlay="bootstrap"))
    with pytest.raises(ValidationError, match="supplied together"):
        MarketState(asof=ASOF, forecasts=[], market_risk_overlay="garch")

    state = MarketState(asof=ASOF, forecasts=[], **_garch(market_risk_overlay=" realized_garch "))
    assert state.market_risk_overlay == "realized_garch"
    assert state.garch_series_scope == "market"
    assert MarketState.validate_garch_nonempty_strings(None) is None
