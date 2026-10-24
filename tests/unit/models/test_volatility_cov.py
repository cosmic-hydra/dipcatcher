"""Coverage for volatility.py guards, fallback machinery, and edge branches.

Complements test_volatility_edges / test_volatility_fit_paths /
test_garch_configurable / test_garch_contract with the fail-closed branches
those suites do not reach: constructor validation (min_obs / mean / power /
series_scope), fit-time fallback reasons driven through mocked arch results,
the parameter-extraction helpers' malformed-input exits, PIT/log-density
edge branches, consumer-scope matching, in-sample path validation, and the
HAR design/fit guards.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_fund.models.volatility import (
    GARCH_DATE_LEVEL_SCOPE,
    GARCH_SECURITY_LEVEL_SCOPE,
    GARCHVol,
    HARVol,
)


def _returns(n: int = 150, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def _mock_fit_result(**attrs) -> MagicMock:
    """arch result stand-in; call sites only read a handful of attributes."""
    result = MagicMock()
    result.convergence_flag = 0
    result.conditional_volatility = np.full(150, 0.5)
    result.params = pd.Series({"omega": 0.05, "alpha[1]": 0.1, "beta[1]": 0.8})
    for key, value in attrs.items():
        setattr(result, key, value)
    return result


# --- GARCHVol constructor guards (fail-closed) ---


@pytest.mark.parametrize("bad", [True, 1.5, 1, 0, -3])
def test_garch_rejects_invalid_min_obs(bad) -> None:
    with pytest.raises(ValueError, match="min_obs"):
        GARCHVol(min_obs=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["HAR", "constant", 1])
def test_garch_rejects_invalid_mean(bad) -> None:
    with pytest.raises(ValueError, match="mean"):
        GARCHVol(mean=bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_garch_rejects_invalid_power(bad: float) -> None:
    with pytest.raises(ValueError, match="power"):
        GARCHVol(power=bad)


@pytest.mark.parametrize("bad", ["", "   ", 0, None])
def test_garch_rejects_blank_series_scope(bad) -> None:
    with pytest.raises(ValueError, match="series_scope"):
        GARCHVol(series_scope=bad)  # type: ignore[arg-type]


def test_garch_series_scope_is_stripped_and_recorded() -> None:
    model = GARCHVol(series_scope="  security_level_ret_1  ")
    assert model.series_scope == "security_level_ret_1"
    assert model.diagnostics()["series_scope"] == "security_level_ret_1"
    assert GARCH_DATE_LEVEL_SCOPE == "date_level_equal_weight_cross_section"
    assert GARCH_SECURITY_LEVEL_SCOPE == "security_level_ret_1"


# --- parameter extraction helpers ---


def test_param_values_and_scalar_fail_closed_on_malformed_params() -> None:
    # Plain dicts lack the pandas-style ``index`` arch returns.
    assert GARCHVol._param_values({"alpha[1]": 0.3}, "alpha[") == []
    # Missing or non-numeric scalars report NaN rather than raising.
    assert np.isnan(GARCHVol._param_scalar({"omega": 0.1}, "delta"))
    assert np.isnan(GARCHVol._param_scalar({"omega": "bad"}, "omega"))


def test_param_values_selects_prefixed_keys_in_index_order() -> None:
    params = pd.Series({"alpha[1]": 0.2, "alpha[2]": 0.3, "beta[1]": 0.4})
    assert GARCHVol._param_values(params, "alpha[") == [0.2, 0.3]
    assert GARCHVol._param_scalar(params, "beta[1]") == pytest.approx(0.4)


# --- fit(): input conversion and fit-time fallback reasons ---


def test_garch_non_numeric_returns_fall_back() -> None:
    model = GARCHVol().fit(
        np.zeros((2, 1)),
        np.zeros(2),
        returns=["not", "numeric"],  # type: ignore[list-item]
    )
    assert model.result is None
    assert model.fit_status == "fallback"
    assert model.fallback_reason == "invalid_returns"
    # Empty converted series takes the default-sigma arm of _fallback.
    assert model.last_sigma == pytest.approx(0.01)
    assert model.diagnostics()["fallback_reason"] == "invalid_returns"


def test_garch_nonconverged_fit_falls_back_with_flag() -> None:
    result = _mock_fit_result(convergence_flag=4)
    with patch("quant_fund.models.volatility.load_arch_model") as factory:
        factory.return_value.fit.return_value = result
        model = GARCHVol().fit_returns(_returns())
    assert model.result is None
    assert model.fit_status == "fallback"
    assert model.fallback_reason == "nonconvergence:4"
    assert model.last_sigma > 0.0


def test_garch_invalid_conditional_volatility_falls_back() -> None:
    result = _mock_fit_result(conditional_volatility=np.full(150, np.nan))
    with patch("quant_fund.models.volatility.load_arch_model") as factory:
        factory.return_value.fit.return_value = result
        model = GARCHVol().fit_returns(_returns())
    assert model.result is None
    assert model.fallback_reason == "invalid_conditional_volatility"


def test_garch_nonfinite_params_fall_back() -> None:
    result = _mock_fit_result(params=pd.Series({"omega": 0.05, "alpha[1]": np.nan}))
    with patch("quant_fund.models.volatility.load_arch_model") as factory:
        factory.return_value.fit.return_value = result
        model = GARCHVol().fit_returns(_returns())
    assert model.result is None
    assert model.fallback_reason == "nonfinite_parameters"


def test_garch_inadmissible_persistence_falls_back() -> None:
    # alpha + beta >= 1 is nonstationary; the fit must not ship to consumers.
    result = _mock_fit_result(params=pd.Series({"omega": 0.05, "alpha[1]": 0.7, "beta[1]": 0.4}))
    with patch("quant_fund.models.volatility.load_arch_model") as factory:
        factory.return_value.fit.return_value = result
        model = GARCHVol(vol="garch").fit_returns(_returns())
    assert model.result is None
    assert model.fallback_reason == "nonstationary_persistence"


# --- unfitted / degraded predictive paths ---


def test_unfitted_garch_predict_and_forecast_use_gaussian_fallback() -> None:
    model = GARCHVol(dist="t")
    assert model.fit_status == "unfitted"
    pred = model.predict(np.zeros((3, 1)))
    assert np.allclose(pred, model.last_sigma)
    forecast = model.forecast(horizon=2, quantiles=(0.1, 0.9))
    assert forecast["fit_status"] == "unfitted"
    # Fallbacks are explicitly Gaussian even when a t law was requested.
    assert forecast["distribution"] == "normal"
    assert forecast["requested_distribution"] == "t"
    quantiles = forecast["quantiles"]
    assert np.all(quantiles[:, 0] < quantiles[:, 1])
    # Gaussian fallback law is symmetric about the (zero) predictive mean.
    assert np.allclose(quantiles[:, 0], -quantiles[:, 1])


def test_garch_mean_decimal_returns_zero_when_mu_missing_or_bad() -> None:
    model = GARCHVol()
    model.result = SimpleNamespace(params={})
    assert model._mean_decimal() == 0.0
    model.result = SimpleNamespace(params={"mu": "not-a-number"})
    assert model._mean_decimal() == 0.0


def test_garch_pit_rejects_sigma_length_mismatch() -> None:
    model = GARCHVol()
    with pytest.raises(ValueError, match="same length"):
        model.pit(np.array([0.01, 0.02]), np.array([0.01]))


def test_garch_log_density_all_invalid_returns_nans() -> None:
    model = GARCHVol()
    out = model.log_density(np.array([np.nan, 0.01]), np.array([0.01, -1.0]))
    assert out.shape == (2,)
    assert np.isnan(out).all()


def test_garch_log_density_size_mismatch_fails_closed() -> None:
    distribution = MagicMock()
    distribution.parameter_names.return_value = []
    distribution.loglikelihood.return_value = np.array([0.0])
    model = GARCHVol()
    model.result = SimpleNamespace(model=SimpleNamespace(distribution=distribution), params={})
    with pytest.raises(ValueError, match="size mismatch"):
        model.log_density(np.array([0.01, -0.02]), np.array([0.01, 0.01]))


# --- scope guard ---


def test_garch_assert_consumer_scope_validates_and_matches() -> None:
    model = GARCHVol()
    with pytest.raises(ValueError, match="consumer_scope"):
        model.assert_consumer_scope("")
    with pytest.raises(ValueError, match="consumer_scope"):
        model.assert_consumer_scope("   ")
    with pytest.raises(ValueError, match="consumer_scope"):
        model.assert_consumer_scope(5)  # type: ignore[arg-type]
    # Matching artifact/consumer scopes are admitted without error.
    model.assert_consumer_scope("univariate_return_series")


# --- in-sample sigma / z path validation ---


def test_in_sample_sigma_and_z_rejects_missing_residuals() -> None:
    model = GARCHVol()
    model.fit_status = "fitted"
    model.result = SimpleNamespace(conditional_volatility=np.ones(4), std_resid=object())
    with pytest.raises(ValueError, match="unavailable"):
        model.in_sample_sigma_and_z()


def test_in_sample_sigma_and_z_rejects_length_mismatch() -> None:
    model = GARCHVol()
    model.fit_status = "fitted"
    model.result = SimpleNamespace(conditional_volatility=np.ones(4), std_resid=np.ones(3))
    with pytest.raises(ValueError, match="length mismatch"):
        model.in_sample_sigma_and_z()


@pytest.mark.parametrize(
    "sigma,z",
    [
        (np.array([0.5, np.nan, 0.5]), np.ones(3)),
        (np.array([0.5, 0.0, 0.5]), np.ones(3)),
        (np.ones(3), np.array([1.0, np.inf, 0.0])),
    ],
)
def test_in_sample_sigma_and_z_rejects_non_finite_or_non_positive(
    sigma: np.ndarray, z: np.ndarray
) -> None:
    model = GARCHVol()
    model.fit_status = "fitted"
    model.result = SimpleNamespace(conditional_volatility=sigma, std_resid=z)
    with pytest.raises(ValueError, match="non-finite or non-positive"):
        model.in_sample_sigma_and_z()


# --- HAR design and estimator guards ---


def test_har_design_handles_tiny_series() -> None:
    single = HARVol.har_design(np.array([0.5]))
    assert single.shape == (1, 4)
    assert single[0, 0] == pytest.approx(1.0)
    assert np.isnan(single[0, 1:]).all()
    empty = HARVol.har_design(np.array([]))
    assert empty.shape == (0, 4)


def test_har_fit_rejects_non_matrix_and_row_mismatch() -> None:
    model = HARVol()
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(np.ones(10), np.ones(10))
    with pytest.raises(ValueError, match="same number of rows"):
        model.fit(np.ones((10, 3)), np.ones(8))


def test_har_predict_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        HARVol().predict(np.ones(7))
