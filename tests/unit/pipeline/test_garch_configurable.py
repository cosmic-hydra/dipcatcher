"""GARCHVol configurability: vol/dist variants, fail-closed edges, metadata.

Extends the volatility suite with the configurable GARCH family: variance
specs (garch/egarch/gjr), innovation distributions (normal/t/skewt),
optimizer-failure fallback, and config plumbing through train_volatility.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from quant_fund.config.models import AppConfig, GarchDist, GarchVolSpec
from quant_fund.models.volatility import _ALLOWED_GARCH_DISTS, _ALLOWED_GARCH_VOLS, GARCHVol


def _returns(n: int = 150, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=n)


def _clustered_returns(n: int = 400, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = 0.012
    out = np.empty(n)
    for i in range(n):
        shock = rng.normal(0.0, 1.0)
        out[i] = sigma * shock
        sigma = float(np.sqrt(8e-7 + 0.07 * out[i] ** 2 + 0.91 * sigma * sigma))
    return out


# --- construction validation (fail-closed) ---


@pytest.mark.parametrize("bad_p,bad_q", [(0, 1), (1, 0), (-1, 1), (1, -2)])
def test_garch_rejects_nonpositive_p_q(bad_p: int, bad_q: int) -> None:
    with pytest.raises(ValueError, match="p and q"):
        GARCHVol(p=bad_p, q=bad_q)


def test_garch_rejects_bool_p() -> None:
    # bool is an int subclass; must not sneak through as p=True=1
    with pytest.raises(ValueError, match="p and q"):
        GARCHVol(p=True)  # type: ignore[arg-type]


def test_garch_rejects_unknown_dist_and_vol() -> None:
    with pytest.raises(ValueError, match="dist"):
        GARCHVol(dist="laplace")
    with pytest.raises(ValueError, match="vol"):
        GARCHVol(vol="harch")


def test_garch_rejects_non_string_dist_vol() -> None:
    with pytest.raises(ValueError, match="dist"):
        GARCHVol(dist=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vol"):
        GARCHVol(vol=None)  # type: ignore[arg-type]


def test_garch_accepts_and_normalizes_case() -> None:
    m = GARCHVol(dist="SkewT", vol="GJR")
    assert m.dist == "skewt"
    assert m.vol == "gjr"
    assert GARCHVol(vol="APARCH").vol == "aparch"
    assert GARCHVol(vol="FIGARCH").vol == "figarch"


# --- fitted paths per variant ---


@pytest.mark.parametrize("vol_spec", ["garch", "gjr", "egarch", "aparch", "figarch"])
def test_garch_vol_specs_fit_and_predict(vol_spec: str) -> None:
    if vol_spec == "egarch" and sys.platform.startswith("linux"):
        # arch 8.0.0 EGARCH optimizer hits a platform boundary on Linux BLAS
        # wheels — the model fails closed (fallback path) there. Flaky across
        # CI runs: observed failing on ubuntu py3.13 and py3.12 and passing on
        # py3.12 with identical code, so the scope is Linux-wide, not a version.
        # Passes on macOS and Windows. Mirrored disclosure in PROOF.md.
        pytest.xfail("arch 8.0.0 egarch boundary on Linux (fail-closed path, flaky)")
    r = _clustered_returns() if vol_spec in {"aparch", "figarch"} else _returns(150)
    m = GARCHVol(vol=vol_spec)
    assert m.fit_returns(r) is m
    assert m.result is not None
    pred = m.predict(np.zeros((3, 1)))
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred) & (pred > 0.0))


def test_figarch_allows_zero_order_and_rejects_higher_order() -> None:
    GARCHVol(vol="figarch", p=0, q=1)
    GARCHVol(vol="figarch", p=1, q=0)
    with pytest.raises(ValueError, match="FIGARCH p and q"):
        GARCHVol(vol="figarch", p=2, q=1)
    with pytest.raises(ValueError, match="FIGARCH p and q"):
        GARCHVol(vol="figarch", p=1, q=2)


@pytest.mark.parametrize("dist", ["normal", "t", "skewt"])
def test_garch_dists_fit_and_predict(dist: str) -> None:
    r = _returns(160)
    m = GARCHVol(dist=dist)
    m.fit_returns(r)
    assert m.result is not None
    pred = m.predict(np.zeros((2, 1)))
    assert np.all(np.isfinite(pred) & (pred > 0.0))


def test_garch_higher_order_p_q_fits() -> None:
    r = _returns(200)
    m = GARCHVol(p=2, q=2)
    m.fit_returns(r)
    assert m.result is not None


def test_garch_fit_optimizer_exception_falls_back() -> None:
    m = GARCHVol()
    r = _returns(150)
    with patch("quant_fund.models.volatility.load_arch_model") as fake_factory:
        fake = MagicMock()
        fake.fit.side_effect = RuntimeError("boom")
        fake_factory.return_value = fake
        m.fit_returns(r)
    assert m.result is None
    assert m.last_sigma == pytest.approx(float(np.std(r * 100.0, ddof=1) / 100.0))
    pred = m.predict(np.zeros((2, 1)))
    assert np.allclose(pred, m.last_sigma)


def test_garch_metadata_reflects_spec() -> None:
    meta = GARCHVol(p=2, q=1, dist="t", vol="egarch").metadata()
    assert meta.name == "egarch2-1_t"
    assert meta.version == "v2"
    assert meta.extra == {
        "p": 2,
        "q": 1,
        "dist": "t",
        "vol": "egarch",
        "series_scope": "univariate_return_series",
    }


# --- config integration ---


def test_trainconfig_garch_fields_roundtrip() -> None:
    cfg = AppConfig()
    assert cfg.train.garch_p == 1
    assert cfg.train.garch_q == 1
    assert cfg.train.garch_dist is GarchDist.NORMAL
    assert cfg.train.garch_vol is GarchVolSpec.GARCH
    cfg2 = AppConfig.model_validate(
        {"train": {"garch_p": 2, "garch_q": 2, "garch_dist": "skewt", "garch_vol": "egarch"}}
    )
    assert cfg2.train.garch_dist is GarchDist.SKEWT
    assert cfg2.train.garch_vol is GarchVolSpec.EGARCH
    cfg3 = AppConfig.model_validate({"train": {"garch_vol": "aparch"}})
    assert cfg3.train.garch_vol is GarchVolSpec.APARCH
    cfg4 = AppConfig.model_validate({"train": {"garch_vol": "figarch"}})
    assert cfg4.train.garch_vol is GarchVolSpec.FIGARCH


def test_trainconfig_rejects_unknown_garch_values() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AppConfig.model_validate({"train": {"garch_dist": "laplace"}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"train": {"garch_vol": "harch"}})
    with pytest.raises(ValidationError, match="FIGARCH"):
        AppConfig.model_validate({"train": {"garch_vol": "figarch", "garch_p": 2}})


def test_garch_allowed_sets_match_config_enums() -> None:
    assert set(_ALLOWED_GARCH_DISTS) == {e.value for e in GarchDist}
    assert set(_ALLOWED_GARCH_VOLS) == {e.value for e in GarchVolSpec}


class _Params:
    def __init__(self, data: dict[str, float]) -> None:
        self._data = data
        self.index = list(data)

    def __getitem__(self, key: str) -> float:
        return self._data[key]


def test_aparch_and_figarch_inadmissible_specs_fail_closed() -> None:
    aparch = GARCHVol(vol="aparch")
    assert (
        aparch._spec_inadmissible_reason(
            _Params(
                {
                    "omega": 0.1,
                    "alpha[1]": 0.2,
                    "gamma[1]": 0.1,
                    "beta[1]": 0.85,
                    "delta": 1.5,
                }
            )
        )
        == "nonstationary_persistence"
    )
    assert (
        aparch._spec_inadmissible_reason(
            _Params(
                {
                    "omega": 0.1,
                    "alpha[1]": 0.05,
                    "gamma[1]": 0.1,
                    "beta[1]": 0.8,
                    "delta": 0.0,
                }
            )
        )
        == "invalid_aparch_delta"
    )
    figarch = GARCHVol(vol="figarch")
    assert (
        figarch._spec_inadmissible_reason(
            _Params({"omega": 0.1, "phi": 0.2, "d": 0.0, "beta": 0.3})
        )
        == "invalid_fractional_d"
    )
    assert (
        figarch._spec_inadmissible_reason(
            _Params({"omega": 0.1, "phi": 0.2, "d": 0.4, "beta": 0.3})
        )
        is None
    )
