"""Coverage for pipeline/train.py fail-closed guards and config branches.

Complements test_pipeline_training and test_train_walk_forward_branches with
the helper-level invariants (PIT stamp comparisons, GARCH history/ohlc guards,
OOS forecast validation) and the auto/auto-dispatch edges that were not
exercised by the happy-path suite. Heavy model fitting is stubbed where the
module allows; nothing touches MLflow or the network.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import pytest

from quant_fund.config import load_config
from quant_fund.config.models import ValidationConfig
from quant_fund.models.base import save_joblib_artifact
from quant_fund.models.calibration import ProbabilityCalibrator
from quant_fund.models.deep_rl import PolicyGradientTrace
from quant_fund.models.quantile_bandit import QuantileBanditTrace
from quant_fund.models.ranking import RidgeRanker
from quant_fund.models.realized_garch import REALIZED_GARCH_MEASURE
from quant_fund.models.rl import BanditTrace, LinUCBRanker
from quant_fund.pipeline import train as train_module
from quant_fund.pipeline.train import (
    _aligned_label_end_times,
    _available_stamp_is_missing,
    _chronological_split,
    _garch_name_oos_predictions,
    _garch_name_origin_return_lookup,
    _garch_name_return_history,
    _garch_name_row_id,
    _garch_one_step_sigma,
    _garch_oos_predictions,
    _garch_origin_density_record,
    _garch_return_history,
    _realized_garch_history,
    _realized_garch_ohlc_columns,
    _realized_garch_oos_predictions,
    _require_garch_security_keys,
    _split_fold,
    _stamp_at_or_before,
    _stamp_strictly_before,
    _walk_forward_splits,
    train_alpha,
    train_calibration,
    train_calibration_auto,
    train_distribution,
    train_distribution_auto,
    train_family,
    train_ranking,
    train_ranking_auto,
    train_regime,
    train_reinforcement,
    train_reinforcement_auto,
    train_robinhood_plus,
    train_tail,
    train_volatility,
    train_volatility_auto,
)
from quant_fund.schemas.errors import PointInTimeError
from quant_fund.validation.walk_forward import Fold
from tests.support.panel_sources import materialize_panel_sources, save_bound_stub

D0 = datetime(2020, 1, 1, tzinfo=UTC)


def _dates(n: int, start: datetime = D0) -> list[datetime]:
    return [start + timedelta(days=i) for i in range(n)]


def _cfg(tmp_path: Path, **validation: int):
    cfg = load_config("configs/research.yaml")
    # dataset identity is fail-closed on materialized panel sources (da5c3e78c)
    cfg.data.root = materialize_panel_sources(tmp_path)
    cfg.validation = ValidationConfig(
        scheme="expanding",
        train_bars=16,
        val_bars=4,
        test_bars=4,
        seed=42,
        **validation,
    )
    return cfg


def _stub_panel(n_dates: int = 64, n_secs: int = 6) -> pl.DataFrame:
    rng = np.random.default_rng(20260924)
    rows = []
    for day in range(n_dates):
        stamp = D0 + timedelta(days=day)
        mkt_vol = 0.02 + abs(float(rng.normal(0, 0.003)))
        for sec in range(n_secs):
            rows.append(
                {
                    "event_time": stamp,
                    "security_id": f"S{sec:02d}",
                    "ret_1": float(rng.normal(0.0004, 0.01)),
                    "vol_20": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_ewma": 0.02 + abs(float(rng.normal(0, 0.002))),
                    "vol_parkinson": 0.02 + abs(float(rng.normal(0, 0.003))),
                    "vol_of_vol": 0.005 + abs(float(rng.normal(0, 0.001))),
                    "mom_20": float(rng.normal(0, 0.01)),
                    "reversal_1": float(rng.normal(0, 0.01)),
                    "amihud": 1e-6 + abs(float(rng.normal(0, 1e-7))),
                    "future_idio_return_1": float(rng.normal(0, 0.01)),
                    "future_log_return_5": float(rng.normal(0, 0.02)),
                    "future_return_5": float(rng.normal(0, 0.02)),
                    "future_realized_var_5": 0.0004 + abs(float(rng.normal(0, 0.0002))),
                    "future_tail_event_5": float(rng.random() < 0.15),
                    "mkt_ret_1": float(rng.normal(0, 0.01)),
                    "mkt_vol_20": mkt_vol,
                    "cs_dispersion": 0.01 + abs(float(rng.normal(0, 0.002))),
                    "breadth": 0.5 + float(rng.normal(0, 0.05)),
                    "high": 101.0 + float(rng.normal(0, 1.0)),
                    "low": 99.0 + float(rng.normal(0, 1.0)),
                }
            )
    return pl.DataFrame(rows)


@pytest.fixture()
def stub_train(monkeypatch: pytest.MonkeyPatch):
    def _apply(frame: pl.DataFrame) -> None:
        monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
        monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
        monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run_test_0001")

    return _apply


def _date_level_panel(n_dates: int = 64) -> pl.DataFrame:
    """One row per date: the GARCH branches score date-level origins."""
    rng = np.random.default_rng(20260924)
    return pl.DataFrame(
        {
            "event_time": _dates(n_dates),
            "security_id": ["MKT"] * n_dates,
            "ret_1": rng.normal(0.0004, 0.01, n_dates),
            "vol_20": 0.02 + np.abs(rng.normal(0, 0.003, n_dates)),
            "vol_ewma": 0.02 + np.abs(rng.normal(0, 0.002, n_dates)),
            "vol_parkinson": 0.02 + np.abs(rng.normal(0, 0.003, n_dates)),
            "vol_of_vol": 0.005 + np.abs(rng.normal(0, 0.001, n_dates)),
            "mom_20": rng.normal(0, 0.01, n_dates),
            "reversal_1": rng.normal(0, 0.01, n_dates),
            "amihud": 1e-6 + np.abs(rng.normal(0, 1e-7, n_dates)),
            "future_idio_return_1": rng.normal(0, 0.01, n_dates),
            "future_log_return_5": rng.normal(0, 0.02, n_dates),
            "future_realized_var_5": 0.0004 + np.abs(rng.normal(0, 0.0002, n_dates)),
            "high": 101.0 + rng.normal(0, 1.0, n_dates),
            "low": 99.0 + rng.normal(0, 1.0, n_dates),
        }
    )


def _prepend_empty_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prepend a degenerate (empty train, empty test) fold to real splits."""
    original = train_module._walk_forward_splits

    def wrapped(*args, **kwargs):  # noqa: ANN001, ANN202
        splits = original(*args, **kwargs)
        empty = np.zeros(len(args[0]), dtype=bool)
        return [(empty, empty)] + splits

    monkeypatch.setattr(train_module, "_walk_forward_splits", wrapped)


class _StubPolicy:
    """Module-level stand-in for an optional trainer (must be picklable)."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def fit(self, *_args, **_kwargs):  # noqa: ANN001, ANN202
        return self


class _DensityGarch:
    """GARCH look-alike that also reports a one-step Gaussian density."""

    fit_status = "fitted"
    converged = True
    fallback_reason = None
    n_obs = 0
    returns_scale = 100.0

    def __init__(self, **_kwargs) -> None:
        self.last = 0.01

    def fit(self, _x, _y, *, returns, **_kwargs):  # noqa: ANN001, ANN202
        values = np.asarray(returns, dtype=float)
        self.last = float(values[-1])
        self.n_obs = int(values.size)
        return self

    def forecast(self, *, horizon, **_kwargs):  # noqa: ANN001, ANN202
        return {
            "cumulative_variance": np.repeat(abs(self.last) + 1e-4, horizon),
            "sigma": np.repeat(0.1, horizon),
            "mean": 0.0,
            "distribution": "normal",
            "fit_status": "fitted",
        }

    def log_density(self, y, sig):  # noqa: ANN001, ANN202
        return np.full(np.asarray(y).size, -1.0)

    def pit(self, y, sig):  # noqa: ANN001, ANN202
        return np.full(np.asarray(y).size, 0.5)

    def save(self, path):  # noqa: ANN001
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"fake-garch")


# --- split/config helpers ---------------------------------------------------


def test_chronological_split_rejects_negative_embargo() -> None:
    dates = np.asarray(_dates(4), dtype=object)
    with pytest.raises(ValueError, match="embargo_bars"):
        _chronological_split(dates, embargo_bars=-1)


def test_walk_forward_splits_rejects_misaligned_label_end_times() -> None:
    dates = np.asarray(_dates(6), dtype=object)
    config = ValidationConfig(train_bars=2, val_bars=1, test_bars=1)
    with pytest.raises(ValueError, match="label_end_times must align"):
        _walk_forward_splits(dates, config, horizon_bars=1, label_end_times=dates[:2])


def test_aligned_label_end_times_returns_row_aligned_endpoints() -> None:
    dates = _dates(3)
    ends = [d + timedelta(days=5) for d in dates]
    frame = pl.DataFrame(
        {
            "event_time": dates,
            "security_id": ["A", "B", "C"],
            "future_return_5": [0.01, 0.02, 0.03],
            "ret_1": [0.001, 0.002, 0.003],
            "label_end_time_5": ends,
        }
    )
    aligned = _aligned_label_end_times(frame, "future_return_5", ["ret_1"])
    assert aligned is not None
    # The frame's tz-aware datetimes come back through numpy as naive values.
    assert np.asarray(aligned).tolist() == [end.replace(tzinfo=None) for end in ends]
    assert (
        _aligned_label_end_times(frame.drop("label_end_time_5"), "future_return_5", ["ret_1"])
        is None
    )


# --- PIT stamp helpers ------------------------------------------------------


def test_stamp_comparators_handle_mixed_and_plain_timestamps() -> None:
    naive = datetime(2020, 1, 2)
    aware_early = datetime(2019, 12, 1, tzinfo=UTC)
    naive_later = datetime(2020, 2, 1)
    aware_same = datetime(2020, 1, 2, tzinfo=UTC)

    assert _stamp_strictly_before(naive, aware_early) is False  # naive localized to asof tz
    assert _stamp_strictly_before(aware_early, naive) is True  # aware de-zoned for compare
    assert _stamp_strictly_before(naive, naive_later) is True
    assert _stamp_strictly_before(2, 3) is True  # non-datetime falls back to <
    assert _stamp_strictly_before(3, 2) is False

    assert _stamp_at_or_before(naive, aware_early) is False
    assert _stamp_at_or_before(aware_same, naive) is True  # equal after dropping tz
    assert _stamp_at_or_before(naive, naive) is True
    assert _stamp_at_or_before(3, 3) is True
    assert _stamp_at_or_before(4, 3) is False


def test_available_stamp_is_missing_flags_none_nat_nan_and_chaotic() -> None:
    assert _available_stamp_is_missing(None) is True
    assert _available_stamp_is_missing(datetime(2020, 1, 1)) is False
    assert _available_stamp_is_missing(float("nan")) is True
    assert _available_stamp_is_missing(np.datetime64("NaT", "us")) is True
    assert _available_stamp_is_missing(0.25) is False

    class Chaotic:
        def __eq__(self, other):  # noqa: ANN001
            raise ValueError("uncomparable")

    assert _available_stamp_is_missing(Chaotic()) is True


# --- GARCH security-key and history guards ----------------------------------


def test_require_garch_security_keys_tolerates_absent_id_and_empty_frame() -> None:
    _require_garch_security_keys(pl.DataFrame({"event_time": [1], "ret_1": [0.1]}))
    empty = pl.DataFrame({"security_id": [], "event_time": [], "ret_1": []})
    _require_garch_security_keys(empty)


def test_garch_name_return_history_requires_nonempty_string_id() -> None:
    frame = pl.DataFrame({"event_time": _dates(1), "security_id": ["A"], "ret_1": [0.01]})
    with pytest.raises(ValueError, match="non-empty security_id"):
        _garch_name_return_history(frame, "   ")
    with pytest.raises(ValueError, match="non-empty security_id"):
        _garch_name_return_history(frame, 7)  # type: ignore[arg-type]


def test_garch_return_history_requires_event_time_and_ret_1() -> None:
    with pytest.raises(ValueError, match="event_time and ret_1"):
        _garch_return_history(pl.DataFrame({"ret_1": [0.01]}))


def test_garch_return_history_no_finite_ret_1_fail_closed() -> None:
    frame = pl.DataFrame({"event_time": [1, 2], "ret_1": [float("nan"), float("inf")]})
    with pytest.raises(ValueError, match="no finite ret_1"):
        _garch_return_history(frame)


def test_garch_return_history_asof_without_prior_rows_returns_empty() -> None:
    frame = pl.DataFrame({"event_time": _dates(2)[1:], "ret_1": [0.02]})
    dates, values = _garch_return_history(frame, asof=D0)
    assert dates.size == 0
    assert values.size == 0


def test_realized_ohlc_columns_prefers_raw_high_low_when_unadjusted() -> None:
    frame = pl.DataFrame({"high": [1.1], "low": [0.9]})
    assert _realized_garch_ohlc_columns(frame) == ("high", "low")


def test_realized_garch_history_requires_event_time_and_ret_1() -> None:
    with pytest.raises(ValueError, match="event_time and ret_1"):
        _realized_garch_history(pl.DataFrame({"high": [1.1], "low": [0.9]}))


def test_realized_garch_history_empty_pairs_fail_closed() -> None:
    frame = pl.DataFrame(
        {"event_time": [1], "ret_1": [None], "high": [1.1], "low": [0.9]},
        schema={"event_time": pl.Int64, "ret_1": pl.Float64, "high": pl.Float64, "low": pl.Float64},
    )
    with pytest.raises(ValueError, match="no finite OHLC/return pairs"):
        _realized_garch_history(frame)


def test_realized_garch_history_nonpositive_parkinson_fail_closed() -> None:
    frame = pl.DataFrame({"event_time": [1], "ret_1": [0.01], "high": [1.0], "low": [1.0]})
    with pytest.raises(ValueError, match="no finite Parkinson"):
        _realized_garch_history(frame)


def test_realized_garch_history_asof_without_prior_rows_returns_empty() -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(2)[1:],
            "ret_1": [0.02],
            "high": [1.1],
            "low": [0.9],
        }
    )
    dates, values, measures = _realized_garch_history(frame, asof=D0)
    assert dates.size == 0
    assert values.size == 0
    assert measures.size == 0


# --- _garch_oos_predictions validation --------------------------------------


def test_garch_oos_predictions_rejects_malformed_inputs() -> None:
    dates = np.array([0, 1, 2])
    x = np.zeros((3, 1))
    y = np.ones(3)
    mask = dates >= 1
    with pytest.raises(ValueError, match="dates and test_mask"):
        _garch_oos_predictions(
            object,
            x,
            y,
            np.array([[0, 1]]),
            mask,
            label_horizon=1,
            return_dates=dates,
            return_values=np.ones(3),
        )
    with pytest.raises(ValueError, match="dates and test_mask"):
        _garch_oos_predictions(
            object,
            x,
            y,
            dates,
            np.array([True]),
            label_horizon=1,
            return_dates=dates,
            return_values=np.ones(3),
        )
    with pytest.raises(ValueError, match="return_dates and return_values"):
        _garch_oos_predictions(
            object,
            x,
            y,
            dates,
            mask,
            label_horizon=1,
            return_dates=dates[:2],
            return_values=np.ones(3),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        _garch_oos_predictions(
            object,
            x,
            y,
            dates,
            mask,
            label_horizon=1,
            return_dates=np.array([]),
            return_values=np.array([]),
        )
    with pytest.raises(ValueError, match="must be finite"):
        _garch_oos_predictions(
            object,
            x,
            y,
            dates,
            mask,
            label_horizon=1,
            return_dates=dates,
            return_values=np.array([0.1, float("nan"), 0.2]),
        )
    with pytest.raises(ValueError, match="label_horizon"):
        _garch_oos_predictions(
            object,
            x,
            y,
            dates,
            mask,
            label_horizon=0,
            return_dates=dates,
            return_values=np.ones(3),
        )


def test_garch_oos_predictions_empty_test_mask_returns_empty() -> None:
    predictions, statuses, density = _garch_oos_predictions(
        object,
        np.zeros((3, 1)),
        np.ones(3),
        np.array([0, 1, 2]),
        np.zeros(3, dtype=bool),
        label_horizon=1,
        return_dates=np.arange(3),
        return_values=np.ones(3),
    )
    assert predictions.size == 0
    assert statuses == []
    assert density == {}


def test_garch_oos_predictions_without_prior_history_fail_closed() -> None:
    dates = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="no strictly prior"):
        _garch_oos_predictions(
            _DensityGarch,
            np.zeros((3, 1)),
            np.ones(3),
            dates,
            dates >= 0,
            label_horizon=1,
            return_dates=dates,
            return_values=np.ones(3),
        )


def test_garch_oos_predictions_pit_frame_without_prior_fail_closed() -> None:
    # Restatement only becomes observable after the origin: PIT history is empty.
    frame = pl.DataFrame(
        {
            "event_time": [D0],
            "ret_1": [0.01],
            "available_time": [D0 + timedelta(days=2)],
        }
    )
    with pytest.raises(ValueError, match="no strictly prior"):
        _garch_oos_predictions(
            _DensityGarch,
            np.zeros((1, 1)),
            np.ones(1),
            np.array([D0 + timedelta(days=1)], dtype=object),
            np.array([True]),
            label_horizon=1,
            return_dates=np.array([D0], dtype=object),
            return_values=np.array([0.01]),
            return_frame=frame,
        )


def test_garch_oos_predictions_invalid_forecast_fail_closed() -> None:
    class BadForecast:
        fit_status = "fitted"

        def fit(self, _x, _y, *, returns):  # noqa: ANN001, ANN202
            return self

        def forecast(self, *, horizon):  # noqa: ANN001, ANN202
            return {"cumulative_variance": np.array([float("nan")])}

    dates = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="invalid out-of-sample forecast"):
        _garch_oos_predictions(
            BadForecast,
            np.zeros((3, 1)),
            np.ones(3),
            dates,
            dates >= 1,
            label_horizon=1,
            return_dates=dates,
            return_values=np.ones(3),
        )


# --- _realized_garch_oos_predictions ----------------------------------------


def _ohlc_frame(dates: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": dates,
            "security_id": ["A"] * len(dates),
            "ret_1": [0.01 + 0.001 * i for i in range(len(dates))],
            "high": [1.10 + 0.01 * i for i in range(len(dates))],
            "low": [1.00 + 0.01 * i for i in range(len(dates))],
        }
    )


def test_realized_garch_oos_walk_forward_scores_origin_pairs() -> None:
    frame = _ohlc_frame(_dates(6))
    dates = np.asarray(frame["event_time"].to_numpy())
    cutoff = dates.tolist()[3]
    predictions, statuses, density = _realized_garch_oos_predictions(
        _DensityGarch,
        np.zeros((6, 2)),
        np.ones(6),
        dates,
        dates >= cutoff,
        label_horizon=2,
        return_frame=frame,
    )
    assert predictions.shape == (3,)
    assert np.all(predictions > 0.0)
    assert statuses == ["fitted"] * 3
    assert set(density) == set(dates.tolist()[3:])
    record = density[cutoff]
    assert record["crps_method"] == "gaussian_closed"
    assert record["density_horizon"] == 1
    assert np.isfinite(record["log_score"])
    assert np.isfinite(record["crps"])
    assert record["pit"] == 0.5


def test_realized_garch_oos_rejects_malformed_inputs() -> None:
    dates = np.array([0, 1, 2])
    x = np.zeros((3, 1))
    y = np.ones(3)
    mask = dates >= 1
    frame = pl.DataFrame({"event_time": _dates(1), "ret_1": [0.01], "high": [1.1], "low": [0.9]})
    with pytest.raises(ValueError, match="dates and test_mask"):
        _realized_garch_oos_predictions(
            object, x, y, dates, np.array([True]), label_horizon=1, return_frame=frame
        )
    with pytest.raises(ValueError, match="requires a return frame"):
        _realized_garch_oos_predictions(
            object, x, y, dates, mask, label_horizon=1, return_frame=None
        )
    with pytest.raises(ValueError, match="label_horizon"):
        _realized_garch_oos_predictions(
            object, x, y, dates, mask, label_horizon=0, return_frame=frame
        )
    predictions, statuses, density = _realized_garch_oos_predictions(
        object, x, y, dates, np.zeros(3, dtype=bool), label_horizon=1, return_frame=frame
    )
    assert predictions.size == 0
    assert statuses == []
    assert density == {}


def test_realized_garch_oos_without_prior_pairs_fail_closed() -> None:
    # Every OHLC row is at/after the earliest origin: no strictly prior pairs.
    frame = _ohlc_frame(_dates(3))
    dates = np.asarray(frame["event_time"].to_numpy())
    with pytest.raises(ValueError, match="no strictly prior Parkinson"):
        _realized_garch_oos_predictions(
            _DensityGarch,
            np.zeros((3, 1)),
            np.ones(3),
            dates,
            dates == dates.tolist()[0],
            label_horizon=1,
            return_frame=frame,
        )


def test_realized_garch_oos_invalid_forecast_fail_closed() -> None:
    class BadForecast:
        fit_status = "fitted"

        def fit(self, _x, _y, *, returns, realized_measure):  # noqa: ANN001, ANN202
            return self

        def forecast(self, *, horizon):  # noqa: ANN001, ANN202
            return {"cumulative_variance": np.array([float("nan")])}

    frame = _ohlc_frame(_dates(3))
    dates = np.asarray(frame["event_time"].to_numpy())
    with pytest.raises(ValueError, match="invalid out-of-sample forecast"):
        _realized_garch_oos_predictions(
            BadForecast,
            np.zeros((3, 1)),
            np.ones(3),
            dates,
            dates >= dates.tolist()[1],
            label_horizon=1,
            return_frame=frame,
        )


def test_realized_garch_oos_missing_origin_ret_1_fail_closed() -> None:
    # Test origin has no row in the return frame: the density target is absent.
    frame = _ohlc_frame(_dates(3))
    history_dates = np.asarray(frame["event_time"].to_numpy())
    late_origin = history_dates.tolist()[-1] + timedelta(days=5)
    dates = np.concatenate([history_dates, np.array([late_origin], dtype=object)])
    with pytest.raises(ValueError, match="missing origin ret_1"):
        _realized_garch_oos_predictions(
            _DensityGarch,
            np.zeros((4, 1)),
            np.ones(4),
            dates,
            dates == late_origin,
            label_horizon=1,
            return_frame=frame,
        )


# --- per-security GARCH helpers ---------------------------------------------


def test_garch_name_origin_return_lookup_edges() -> None:
    with pytest.raises(ValueError, match="event_time, security_id, and ret_1"):
        _garch_name_origin_return_lookup(pl.DataFrame({"event_time": [1], "ret_1": [0.1]}))

    empty = pl.DataFrame({"event_time": [], "security_id": [], "ret_1": []})
    assert _garch_name_origin_return_lookup(empty) == {}

    non_string = pl.DataFrame({"event_time": [D0], "security_id": [7], "ret_1": [0.1]})
    with pytest.raises(ValueError, match="must be strings"):
        _garch_name_origin_return_lookup(non_string)

    blank = pl.DataFrame({"event_time": [D0], "security_id": ["  "], "ret_1": [0.1]})
    with pytest.raises(PointInTimeError, match="blank security_id"):
        _garch_name_origin_return_lookup(blank)

    partly_nan = pl.DataFrame(
        {
            "event_time": [D0, D0 + timedelta(days=1)],
            "security_id": ["A", "A"],
            "ret_1": [float("nan"), 0.2],
        }
    )
    # The event_time column returns through numpy as a naive datetime.
    assert _garch_name_origin_return_lookup(partly_nan) == {
        ("A", (D0 + timedelta(days=1)).replace(tzinfo=None)): 0.2
    }

    # "A" and " A " normalize to the same key: distinct raw keys but a colliding
    # lookup entry must still fail closed.
    colliding = pl.DataFrame(
        {
            "event_time": [D0, D0],
            "security_id": ["A", " A"],
            "ret_1": [0.1, 0.2],
        }
    )
    with pytest.raises(PointInTimeError, match="duplicate security_id/event_time"):
        _garch_name_origin_return_lookup(colliding)


def test_garch_name_row_id_rejects_non_string_and_blank() -> None:
    with pytest.raises(ValueError, match="must be strings"):
        _garch_name_row_id(7)
    with pytest.raises(ValueError, match="must be non-empty"):
        _garch_name_row_id("   ")
    assert _garch_name_row_id("  A  ") == "A"


def _name_frame(dates: list[datetime], sid: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": dates,
            "security_id": [sid] * len(dates),
            "ret_1": [0.01 + 0.001 * i for i in range(len(dates))],
        }
    )


def test_garch_name_oos_rejects_malformed_inputs() -> None:
    dates = np.array([0, 1, 2])
    ids = np.array(["A", "A", "A"])
    x = np.zeros((3, 1))
    y = np.ones(3)
    mask = dates >= 1
    frame = _name_frame(_dates(3))
    with pytest.raises(ValueError, match="dates and test_mask"):
        _garch_name_oos_predictions(
            object, x, y, dates, ids, np.array([True]), label_horizon=1, return_frame=frame
        )
    with pytest.raises(ValueError, match="ids and dates"):
        _garch_name_oos_predictions(
            object, x, y, dates, ids[:2], mask, label_horizon=1, return_frame=frame
        )
    with pytest.raises(ValueError, match="x, y, dates, and ids"):
        _garch_name_oos_predictions(
            object, x[:2], y, dates, ids, mask, label_horizon=1, return_frame=frame
        )
    with pytest.raises(ValueError, match="label_horizon"):
        _garch_name_oos_predictions(
            object, x, y, dates, ids, mask, label_horizon=0, return_frame=frame
        )
    with pytest.raises(ValueError, match="requires a return frame"):
        _garch_name_oos_predictions(
            object, x, y, dates, ids, mask, label_horizon=1, return_frame=None
        )
    predictions, statuses, density = _garch_name_oos_predictions(
        object, x, y, dates, ids, np.zeros(3, dtype=bool), label_horizon=1, return_frame=frame
    )
    assert predictions.size == 0
    assert statuses == []
    assert density == {}


def test_garch_name_oos_skips_density_when_model_lacks_pit() -> None:
    class NoDensity:
        fit_status = "fitted"

        def fit(self, _x, _y, *, returns):  # noqa: ANN001, ANN202
            return self

        def forecast(self, *, horizon):  # noqa: ANN001, ANN202
            return {"cumulative_variance": np.repeat(0.5, horizon)}

    dates = np.asarray(_dates(3), dtype=object)
    ids = np.array(["A", "A", "A"], dtype=object)
    predictions, statuses, density = _garch_name_oos_predictions(
        NoDensity,
        np.zeros((3, 1)),
        np.ones(3),
        dates,
        ids,
        dates >= _dates(3)[1],
        label_horizon=1,
        return_frame=_name_frame(_dates(3)),
    )
    assert predictions.shape == (2,)
    assert statuses == ["fitted"] * 2
    assert density == {}


def test_garch_name_oos_invalid_forecast_fail_closed() -> None:
    class BadForecast:
        fit_status = "fitted"

        def fit(self, _x, _y, *, returns):  # noqa: ANN001, ANN202
            return self

        def forecast(self, *, horizon):  # noqa: ANN001, ANN202
            return {"cumulative_variance": np.array([float("nan")])}

    dates = np.asarray(_dates(3), dtype=object)
    ids = np.array(["A", "A", "A"], dtype=object)
    with pytest.raises(ValueError, match="invalid out-of-sample forecast"):
        _garch_name_oos_predictions(
            BadForecast,
            np.zeros((3, 1)),
            np.ones(3),
            dates,
            ids,
            dates >= _dates(3)[1],
            label_horizon=1,
            return_frame=_name_frame(_dates(3)),
        )


# --- one-step density helpers ------------------------------------------------


def test_garch_one_step_sigma_accepts_sigma_or_variance() -> None:
    assert _garch_one_step_sigma({"sigma": [0.2]}) == pytest.approx(0.2)
    assert _garch_one_step_sigma({"variance": [0.04]}) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="requires forecast sigma or variance"):
        _garch_one_step_sigma({})
    with pytest.raises(ValueError, match="invalid one-step sigma"):
        _garch_one_step_sigma({"variance": [-1.0]})
    with pytest.raises(ValueError, match="invalid one-step sigma"):
        _garch_one_step_sigma({"sigma": [0.0]})


class _DensityModel:
    def __init__(
        self,
        log_score=(-1.5,),
        pit_value=(0.4,),
        quantiles=None,
        mean_decimal=None,
    ) -> None:
        self._log_score = np.asarray(log_score, dtype=float)
        self._pit_value = np.asarray(pit_value, dtype=float)
        self._quantiles = quantiles
        if mean_decimal is not None:
            self._mean_decimal = lambda: mean_decimal

    def log_density(self, y, sig):  # noqa: ANN001, ANN202
        return self._log_score

    def pit(self, y, sig):  # noqa: ANN001, ANN202
        return self._pit_value

    def forecast(self, *, horizon, quantiles=None):  # noqa: ANN001, ANN202
        return {"quantiles": np.asarray(self._quantiles, dtype=float)}


def test_garch_origin_density_record_gaussian_path_and_edges() -> None:
    model = _DensityModel()
    forecast = {"sigma": [0.1], "mean": 0.0, "distribution": "normal", "fit_status": "fitted"}
    record = _garch_origin_density_record(model, 0.01, forecast)
    assert record["log_score"] == -1.5
    assert record["pit"] == 0.4
    assert record["mu"] == 0.0
    assert record["sigma"] == pytest.approx(0.1)
    assert record["crps_method"] == "gaussian_closed"
    assert np.isfinite(record["crps"])
    assert record["density_horizon"] == 1

    with pytest.raises(ValueError, match="must be finite"):
        _garch_origin_density_record(model, float("nan"), forecast)


def test_garch_origin_density_record_rejects_malformed_scores() -> None:
    model = _DensityModel(log_score=(-1.5, -2.0))
    forecast = {"sigma": [0.1], "mean": 0.0, "distribution": "normal"}
    with pytest.raises(ValueError, match="malformed score"):
        _garch_origin_density_record(model, 0.01, forecast)


def test_garch_origin_density_record_mu_fallback_and_quantile_crps() -> None:
    # forecast lacks "mean" -> falls back to model._mean_decimal()
    model = _DensityModel(mean_decimal=0.0123)
    record = _garch_origin_density_record(model, 0.01, {"sigma": [0.1], "distribution": "normal"})
    assert record["mu"] == pytest.approx(0.0123)

    # non-normal distribution -> quantile Riemann CRPS
    taus = np.linspace(0.05, 0.95, 19)
    quantile_forecast = np.repeat(0.02, taus.size).reshape(1, -1)
    qt_model = _DensityModel(quantiles=quantile_forecast)
    record = _garch_origin_density_record(
        qt_model, 0.02, {"sigma": [0.1], "mean": 0.0, "distribution": "student_t"}
    )
    assert record["crps_method"] == "quantile_riemann"
    assert record["crps"] == pytest.approx(0.0, abs=1e-12)


# --- train_ranking / train_ranking_auto --------------------------------------


def test_train_ranking_walk_forward_persists_artifact(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    result = train_ranking(cfg, "ridge")
    assert Path(result["path"]).name == "ranker_ridge.joblib"
    assert Path(result["path"]).is_file()
    assert result["features"]
    assert result["n"] > 0
    for key in ("mean_ic", "mean_rank_ic", "icir", "ic_tstat", "ic_pvalue", "n_ic_dates"):
        assert key in result["metrics"]


def test_train_ranking_skips_degenerate_fold_masks(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    _prepend_empty_fold(monkeypatch)
    result = train_ranking(_cfg(tmp_path), "ridge")
    assert Path(result["path"]).is_file()


def test_train_ranking_dispatches_auto_and_rejects_no_viable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "train_ranking_auto", lambda cfg: {"model": "auto"})
    cfg = _cfg(tmp_path)
    assert train_ranking(cfg, "auto") == {"model": "auto"}

    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        return {"path": "unused", "metrics": {"mean_ic": float("nan")}}

    monkeypatch.setattr(train_module, "train_ranking", fake_train)
    with pytest.raises(ValueError, match="no finite candidate metric"):
        train_ranking_auto(cfg)


# --- train_calibration / train_calibration_auto ------------------------------


def test_train_calibration_dispatches_auto_and_rejects_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "train_calibration_auto", lambda cfg: {"model": "auto"})
    cfg = _cfg(tmp_path)
    assert train_calibration(cfg, "auto") == {"model": "auto"}
    with pytest.raises(ValueError, match="unknown calibration model"):
        train_calibration(cfg, "ridge")


def _calibration_frame(n: int, *, single_class: bool = False) -> pl.DataFrame:
    dates = _dates(n)
    labels = np.linspace(-0.9, 0.9, n) if not single_class else np.full(n, 0.5)
    return pl.DataFrame(
        {
            "event_time": dates,
            "cs_pct_mom_20": np.linspace(0.05, 0.95, n),
            "future_idio_return_1": labels,
        }
    )


def test_train_calibration_requires_rows_and_both_classes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _calibration_frame(8))
    with pytest.raises(ValueError, match=">=10 finite rows and both label classes"):
        train_calibration(cfg, "isotonic")

    monkeypatch.setattr(
        "quant_fund.pipeline.train.panel",
        lambda *a, **k: _calibration_frame(24, single_class=True),
    )
    with pytest.raises(ValueError, match=">=10 finite rows and both label classes"):
        train_calibration(cfg, "isotonic")


def test_train_calibration_skips_unfitted_fold_calibrators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_train
) -> None:
    # A fold whose train block is too small to fit leaves no OOS calibration.
    frame = _calibration_frame(20)
    stub_train(frame)
    train_mask = np.zeros(20, dtype=bool)
    train_mask[:4] = True
    test_mask = np.zeros(20, dtype=bool)
    test_mask[10:] = True
    monkeypatch.setattr(
        train_module, "_walk_forward_splits", lambda *a, **k: [(train_mask, test_mask)]
    )
    result = train_calibration(_cfg(tmp_path), "isotonic")
    assert np.isnan(result["metrics"]["oos_brier"])
    assert result["metrics"]["n_oos_rows"] == 0.0
    assert Path(result["path"]).is_file()


def test_train_calibration_unfitted_final_artifact_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_train
) -> None:
    class NeverFits:
        fitted = False

        def __init__(self, _method) -> None:
            pass

        def fit(self, _scores, _labels):  # noqa: ANN001, ANN202
            return self

        def predict(self, scores):  # noqa: ANN001, ANN202
            return np.zeros(len(scores))

    stub_train(_calibration_frame(20))
    monkeypatch.setattr(train_module, "ProbabilityCalibrator", NeverFits)
    with pytest.raises(ValueError, match="unfitted artifact"):
        train_calibration(_cfg(tmp_path), "isotonic")


def test_train_calibration_auto_rejects_all_nan_brier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        return {"path": "unused", "metrics": {"oos_brier": float("nan")}}

    monkeypatch.setattr(train_module, "train_calibration", fake_train)
    with pytest.raises(ValueError, match="no finite Brier score"):
        train_calibration_auto(_cfg(tmp_path))


# --- train_distribution ------------------------------------------------------


def test_train_distribution_skips_degenerate_folds(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    _prepend_empty_fold(monkeypatch)
    result = train_distribution(_cfg(tmp_path), "empirical")
    assert np.isfinite(result["metrics"]["mean_pinball"])


def test_train_distribution_no_evaluable_fold_fail_closed(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_distribution(_cfg(tmp_path), "empirical")


# --- train_volatility --------------------------------------------------------


def test_train_volatility_skips_degenerate_folds_and_requires_folds(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    _prepend_empty_fold(monkeypatch)
    result = train_volatility(_cfg(tmp_path), "ewma")
    assert np.isfinite(result["metrics"]["qlike"])

    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_volatility(_cfg(tmp_path), "ewma")


def test_train_volatility_garch_reports_one_step_density(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_date_level_panel())
    monkeypatch.setattr(train_module, "GARCHVol", _DensityGarch)
    result = train_volatility(_cfg(tmp_path), "garch")
    metrics = result["metrics"]
    assert metrics["crps_method_one_step"] == "gaussian_closed"
    assert int(metrics["n_density_origins"]) >= 1
    assert np.isfinite(metrics["log_score_one_step"])
    assert np.isfinite(metrics["crps_one_step"])
    assert metrics["density_target"] == "date_level_ret_1"
    assert metrics["fallback_rate"] == 0.0
    assert result["diagnostics"]["n_density_origins"] == int(metrics["n_density_origins"])
    assert Path(result["path"]).is_file()


def test_train_volatility_garch_duplicate_density_origin_fail_closed(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_date_level_panel())
    monkeypatch.setattr(train_module, "GARCHVol", _DensityGarch)

    def fake_oos(_make_model, _x, _y, _dates, test_mask, **_kwargs):  # noqa: ANN001, ANN202
        return np.full(int(test_mask.sum()), 0.5), ["fitted"], {"fixed_origin": {"log_score": -1.0}}

    monkeypatch.setattr(train_module, "_garch_oos_predictions", fake_oos)
    with pytest.raises(ValueError, match="density origin repeated across folds"):
        train_volatility(_cfg(tmp_path), "garch")


def test_train_volatility_garch_without_statuses_omits_fallback_rate(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_date_level_panel())
    monkeypatch.setattr(train_module, "GARCHVol", _DensityGarch)

    def fake_oos(_make_model, _x, _y, _dates, test_mask, **_kwargs):  # noqa: ANN001, ANN202
        return np.full(int(test_mask.sum()), 0.5), [], {}

    monkeypatch.setattr(train_module, "_garch_oos_predictions", fake_oos)
    result = train_volatility(_cfg(tmp_path), "garch")
    assert "fallback_rate" not in result["metrics"]
    assert result["metrics"]["n_density_origins"] == 0


def test_train_volatility_realized_garch_branch(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_date_level_panel())
    monkeypatch.setattr(train_module, "RealizedGARCHVol", _DensityGarch)
    result = train_volatility(_cfg(tmp_path), "realized_garch")
    metrics = result["metrics"]
    assert metrics["realized_measure"] == REALIZED_GARCH_MEASURE
    assert metrics["intraday_realized_variance"] is False
    assert int(metrics["n_density_origins"]) >= 1
    assert metrics["crps_method_one_step"] == "gaussian_closed"
    diagnostics = result["diagnostics"]
    assert diagnostics["realized_measure"] == REALIZED_GARCH_MEASURE
    assert diagnostics["intraday_realized_variance"] is False
    assert diagnostics["n_density_origins"] == int(metrics["n_density_origins"])
    assert Path(result["path"]).is_file()


def test_train_volatility_auto_rejects_all_nan_qlike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        return {"path": "unused", "metrics": {"qlike": float("nan"), "n_oos_rows": 20}}

    monkeypatch.setattr(train_module, "train_volatility", fake_train)
    with pytest.raises(ValueError, match="no finite candidate metric"):
        train_volatility_auto(_cfg(tmp_path))


# --- train_alpha / train_regime / train_tail ---------------------------------


def test_train_alpha_delegates_non_mean_to_train_ranking(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(train_module, "train_ranking", lambda config, name: {"delegated": name})
    assert train_alpha(_cfg(tmp_path), "ridge") == {"delegated": "ridge"}
    assert train_alpha(_cfg(tmp_path), "ensemble") == {"delegated": "ensemble"}


def test_train_regime_skips_degenerate_folds(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    _prepend_empty_fold(monkeypatch)
    result = train_regime(_cfg(tmp_path), "single")
    assert Path(result["path"]).is_file()


def test_train_tail_missing_future_return_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(4),
            "security_id": ["A", "B", "C", "D"],
            "ret_1": [0.01, 0.02, 0.03, 0.04],
        }
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: frame)
    with pytest.raises(ValueError, match="no future_return"):
        train_tail(load_config("configs/research.yaml"), "historical")


def test_train_tail_skips_degenerate_folds_and_requires_scores(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    _prepend_empty_fold(monkeypatch)
    result = train_tail(_cfg(tmp_path), "historical")
    assert np.isfinite(result["metrics"]["oos_var_breach_rate"])

    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_tail(_cfg(tmp_path), "historical")


# --- train_reinforcement ------------------------------------------------------


def _rl_frame(n_dates: int = 8) -> pl.DataFrame:
    rows = []
    label = "future_idio_return_1"
    for date in _dates(n_dates):
        for name, signal in [("A", 1.0), ("B", 0.2), ("C", -0.5), ("D", -1.0)]:
            rows.append(
                {
                    "event_time": date,
                    "security_id": name,
                    label: signal,
                    "ret_1": signal,
                    "mom_20": signal,
                    "vol_20": 0.2,
                    "reversal_1": signal,
                    "amihud": 1e-6,
                }
            )
    return pl.DataFrame(rows)


def test_train_reinforcement_dispatches_specialized_trainers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "train_reinforcement_auto", lambda cfg: {"model": "auto"})
    monkeypatch.setattr(
        train_module, "train_policy_gradient", lambda cfg: {"model": "policy_gradient"}
    )
    monkeypatch.setattr(
        train_module, "train_quantile_bandit", lambda cfg: {"model": "quantile_thompson"}
    )
    cfg = _cfg(tmp_path)
    assert train_reinforcement(cfg, "auto") == {"model": "auto"}
    assert train_reinforcement(cfg, "policy_gradient") == {"model": "policy_gradient"}
    assert train_reinforcement(cfg, "quantile_thompson") == {"model": "quantile_thompson"}
    with pytest.raises(ValueError, match="unknown reinforcement model"):
        train_reinforcement(cfg, "dqn")


def test_train_reinforcement_requires_nonempty_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _rl_frame())
    monkeypatch.setattr(
        train_module,
        "design_matrix",
        lambda *a, **k: (
            np.zeros((4, 0)),
            np.ones(4),
            np.asarray(_dates(4), dtype=object),
            [],
            np.array(["A", "B", "C", "D"], dtype=object),
        ),
    )
    with pytest.raises(ValueError, match="non-empty feature panel"):
        train_reinforcement(_cfg(tmp_path), "linucb")


def test_train_reinforcement_no_evaluable_dates_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _rl_frame())
    empty = np.asarray([], dtype=float)
    monkeypatch.setattr(
        train_module,
        "run_linucb_panel",
        lambda *a, **k: BanditTrace(
            dates=[],
            policy_reward=empty,
            oracle_reward=empty,
            uniform_reward=empty,
            cumulative_regret=empty,
        ),
    )
    with pytest.raises(ValueError, match="no evaluable decision dates"):
        train_reinforcement(_cfg(tmp_path), "linucb")


def test_train_reinforcement_refit_skips_dates_without_finite_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _rl_frame())
    monkeypatch.setattr("quant_fund.pipeline.train.configure_tracking", lambda: None)
    monkeypatch.setattr("quant_fund.pipeline.train.log_run", lambda **kwargs: "run-rl")
    dates = np.asarray([stamp for stamp in _dates(4) for _ in range(2)], dtype=object)
    y = np.linspace(0.1, 0.8, 8)
    y[:2] = np.nan  # first decision date carries no finite reward
    x = np.column_stack([np.linspace(-1.0, 1.0, 8), np.linspace(0.5, -0.5, 8)])
    monkeypatch.setattr(
        train_module,
        "design_matrix",
        lambda *a, **k: (x, y, dates, ["f0", "f1"], np.array(["A"] * 8, dtype=object)),
    )
    monkeypatch.setattr(
        train_module,
        "run_linucb_panel",
        lambda *a, **k: BanditTrace(
            dates=["d0"],
            policy_reward=np.array([0.5]),
            oracle_reward=np.array([0.6]),
            uniform_reward=np.array([0.2]),
            cumulative_regret=np.array([0.1]),
        ),
    )
    result = train_reinforcement(_cfg(tmp_path), "linucb")
    assert result["research_only"] is True
    assert result["metrics"]["n_dates"] == 1.0
    assert Path(result["path"]).is_file()


def test_train_reinforcement_auto_records_policy_gradient_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    advantages = {"linucb": 0.01, "thompson": 0.05, "quantile_thompson": 0.02}

    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        if model_name == "policy_gradient":
            raise ImportError("torch is not installed")
        path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
        save_bound_stub(
            config,
            {"policy": LinUCBRanker(1), "policy_name": model_name, "features": ["f0"]},
            path,
        )
        return {
            "path": str(path),
            "metrics": {"mean_advantage_vs_uniform": advantages[model_name]},
        }

    monkeypatch.setattr(train_module, "train_reinforcement", fake_train)
    result = train_reinforcement_auto(_cfg(tmp_path))
    assert result["selected_model"] == "thompson"
    assert result["candidates"]["policy_gradient"]["unavailable"] == "torch is not installed"


def test_train_reinforcement_auto_propagates_non_pg_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        if model_name == "thompson":
            raise ImportError("unexpected missing dependency")
        path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
        save_joblib_artifact({"policy": LinUCBRanker(1), "policy_name": model_name}, path)
        return {"path": str(path), "metrics": {"mean_advantage_vs_uniform": 0.01}}

    monkeypatch.setattr(train_module, "train_reinforcement", fake_train)
    with pytest.raises(ImportError, match="unexpected missing dependency"):
        train_reinforcement_auto(_cfg(tmp_path))


def test_train_reinforcement_auto_no_viable_and_malformed_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)

    def nan_train(config, model_name):  # noqa: ANN001, ANN202
        return {"path": "unused", "metrics": {"mean_advantage_vs_uniform": float("nan")}}

    monkeypatch.setattr(train_module, "train_reinforcement", nan_train)
    with pytest.raises(ValueError, match="no finite candidate metric"):
        train_reinforcement_auto(cfg)

    def malformed_train(config, model_name):  # noqa: ANN001, ANN202
        path = Path(config.data.root) / "metadata" / f"rl_{model_name}.joblib"
        save_joblib_artifact({"not_a_policy": True}, path)
        return {"path": str(path), "metrics": {"mean_advantage_vs_uniform": 0.01}}

    monkeypatch.setattr(train_module, "train_reinforcement", malformed_train)
    with pytest.raises(ValueError, match="malformed"):
        train_reinforcement_auto(cfg)


def test_train_quantile_bandit_no_evaluable_dates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _rl_frame())

    class EmptyBandit:
        def __init__(self, seed=None) -> None:
            pass

        def run_panel(self, x, y, dates, k):  # noqa: ANN001, ANN202
            return type(
                "Trace",
                (),
                {
                    "policy_reward": np.asarray([], dtype=float),
                    "uniform_reward": np.asarray([], dtype=float),
                    "oracle_reward": np.asarray([], dtype=float),
                    "cumulative_regret": np.asarray([], dtype=float),
                    "dates": [],
                },
            )()

    monkeypatch.setattr(train_module, "QuantileThompson", EmptyBandit)
    with pytest.raises(ValueError, match="no evaluable dates"):
        train_reinforcement(_cfg(tmp_path), "quantile_thompson")


def test_train_policy_gradient_no_evaluable_dates_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _rl_frame())

    class EmptyTrace:
        policy_reward = np.asarray([], dtype=float)
        uniform_reward = np.asarray([], dtype=float)
        cumulative_regret = np.asarray([], dtype=float)
        dates: list = []

    monkeypatch.setattr(train_module, "run_policy_gradient_panel", lambda *a, **k: EmptyTrace())
    with pytest.raises(ValueError, match="no evaluable dates"):
        train_reinforcement(_cfg(tmp_path), "policy_gradient")


# --- train_robinhood_plus ------------------------------------------------------


def test_train_robinhood_plus_persists_engine_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _stub_panel())
    monkeypatch.setattr(
        "quant_fund.models.robinhood_plus.bench.bench_robinhood_plus",
        lambda frame, config: {"mean_ic": 0.07, "n_ok": 3, "ic_n_dates": 2},
    )
    cfg = _cfg(tmp_path)
    result = train_robinhood_plus(cfg, "hierarchical_markov")
    assert result["metrics"]["mean_ic"] == 0.07
    assert result["metrics"]["n_ok"] == 3
    assert result["family"] == "robinhood_plus"
    assert result["research_only"] is True
    assert result["execution_claim"] == "research_only"
    assert Path(result["path"]).name == "robinhood_plus_hierarchical_markov.joblib"
    assert Path(result["path"]).is_file()

    with pytest.raises(ValueError, match="unknown robinhood_plus model"):
        train_robinhood_plus(cfg, "not_a_decoder")


# --- train_family --------------------------------------------------------------


def test_train_family_non_synthetic_source_skips_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_module, "train_calibration", lambda config, model: {"metrics": {}})
    cfg = _cfg(tmp_path)
    cfg.data.source = "parquet"
    result = train_family(cfg, "calibration")
    assert "data_source" not in result
    assert result["evidence_report"]


def test_train_family_records_artifact_identity_and_attaches_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "metadata" / "calibrator_isotonic.joblib"
    model = RidgeRanker().fit(np.ones((4, 2)), np.arange(4, dtype=float))
    model.save(artifact)

    monkeypatch.setattr(
        train_module,
        "train_calibration",
        lambda config, model_name: {
            "metrics": {"oos_brier": 0.1},
            "path": str(artifact),
            "run_id": "run-9",
        },
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _stub_panel())
    attached: dict[str, object] = {}

    def capture(run_id, identity):  # noqa: ANN001
        attached["run_id"] = run_id
        attached["identity"] = identity

    monkeypatch.setattr(train_module, "attach_artifact_identity", capture)

    cfg = _cfg(tmp_path)
    result = train_family(cfg, "calibration")
    assert result["manifest_valid"] is True
    assert isinstance(result["artifact_sha256"], str)
    assert len(result["artifact_sha256"]) == 64
    assert result["artifact_class"] == "RidgeRanker"
    assert result["dataset_content_sha256"]
    assert attached["run_id"] == "run-9"
    assert attached["identity"]["manifest_valid"] is True


def test_train_family_marks_manifest_invalid_and_survives_fingerprint_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "metadata" / "bare.joblib"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"payload": 1}, artifact)  # artifact with no manifest sidecar

    monkeypatch.setattr(
        train_module,
        "train_calibration",
        lambda config, model_name: {"metrics": {}, "path": str(artifact)},
    )
    monkeypatch.setattr("quant_fund.pipeline.train.panel", lambda *a, **k: _stub_panel())

    def bad_fingerprint(_frame):  # noqa: ANN001
        raise ValueError("cannot fingerprint")

    monkeypatch.setattr(train_module, "canonical_frame_fingerprint", bad_fingerprint)

    result = train_family(_cfg(tmp_path), "calibration")
    assert result["manifest_valid"] is False
    assert "manifest" in result["artifact_identity_error"]
    assert result["dataset_fingerprint_error"] == "cannot fingerprint"


# --- split helpers, second pass ---------------------------------------------


def test_chronological_split_validates_inputs_and_splits_dates() -> None:
    dates = np.asarray(_dates(4), dtype=object)
    with pytest.raises(ValueError, match="train_fraction"):
        _chronological_split(dates, train_fraction=1.0)
    with pytest.raises(ValueError, match="horizon_bars"):
        _chronological_split(dates, horizon_bars=-1)
    with pytest.raises(ValueError, match="at least 2 unique dates"):
        _chronological_split(np.asarray([D0], dtype=object))

    train_mask, test_mask = _chronological_split(dates, train_fraction=0.5)
    assert train_mask.sum() == 1  # horizon 1 purges the boundary train date
    assert test_mask.sum() == 2
    assert not np.any(train_mask & test_mask)


def test_walk_forward_splits_short_sample_fallback_purges_labels() -> None:
    dates = np.asarray(_dates(8), dtype=object)
    config = ValidationConfig(train_bars=16, val_bars=4, test_bars=4)

    # Too few dates for a real fold: fallback still yields a genuine test block.
    splits = _walk_forward_splits(dates, config, horizon_bars=1)
    assert len(splits) == 1
    train_mask, test_mask = splits[0]
    test_dates = set(dates[test_mask].tolist())
    assert test_dates == set(dates.tolist()[4:])
    assert test_mask.any()

    # With observed label end-times, purge uses them instead of session math.
    ends = np.asarray([d + timedelta(days=1) for d in _dates(8)], dtype=object)
    splits = _walk_forward_splits(dates, config, horizon_bars=1, label_end_times=ends)
    assert len(splits) == 1
    train_mask, test_mask = splits[0]
    # Labels ending inside the test block purge that train date; the embargo
    # (== horizon) drops the boundary train date too.
    train_dates = set(dates[train_mask].tolist())
    assert dates.tolist()[3] not in train_dates
    assert set(dates[test_mask].tolist()) == set(dates.tolist()[4:])

    # Duplicated rows for one date keep the latest observed label end-time.
    dup_values = np.concatenate([dates, dates[:1], dates[:1]])
    dup_ends = np.concatenate(
        [
            ends,
            np.asarray(
                [dates.tolist()[0] + timedelta(days=2), dates.tolist()[0] + timedelta(hours=12)]
            ),
        ]
    )
    splits = _walk_forward_splits(dup_values, config, horizon_bars=1, label_end_times=dup_ends)
    assert len(splits) == 1
    assert splits[0][1].sum() >= 4


def test_split_fold_masks_align_to_fold_dates() -> None:
    dates = np.asarray(_dates(4), dtype=object)
    fold = Fold(
        train_times=dates.tolist()[:2],
        val_times=dates.tolist()[2:3],
        test_times=dates.tolist()[3:],
    )
    tr, va, te = _split_fold(dates, np.zeros(4), np.zeros(4), fold)
    assert tr.tolist() == [True, True, False, False]
    assert va.tolist() == [False, False, True, False]
    assert te.tolist() == [False, False, False, True]


# --- PIT availability filters -------------------------------------------------


def test_garch_return_history_rejects_null_available_time() -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(2),
            "ret_1": [0.01, 0.02],
            "available_time": [D0 + timedelta(hours=1), None],
        }
    )
    asof = D0 + timedelta(days=1, hours=12)
    with pytest.raises(PointInTimeError, match="null available_time"):
        _garch_return_history(frame, asof=asof)


def test_garch_return_history_drops_unpublished_rows() -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(2),
            "ret_1": [0.01, 0.02],
            "available_time": [
                D0 + timedelta(hours=1),
                D0 + timedelta(days=2),
            ],
        }
    )
    asof = D0 + timedelta(days=1, hours=12)
    dates, values = _garch_return_history(frame, asof=asof)
    # Only the first row is observable at asof; the second restates later.
    assert values.tolist() == [0.01]
    assert dates.size == 1


def test_realized_garch_ohlc_columns_prefers_split_adjusted() -> None:
    frame = pl.DataFrame(
        {
            "high": [1.1],
            "low": [0.9],
            "high_split_adjusted": [1.05],
            "low_split_adjusted": [0.95],
        }
    )
    assert _realized_garch_ohlc_columns(frame) == (
        "high_split_adjusted",
        "low_split_adjusted",
    )
    with pytest.raises(PointInTimeError, match="daily OHLC high/low"):
        _realized_garch_ohlc_columns(pl.DataFrame({"ret_1": [0.01]}))


def test_realized_garch_history_rejects_null_available_time() -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(2),
            "ret_1": [0.01, 0.02],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "available_time": [D0 + timedelta(hours=1), None],
        }
    )
    asof = D0 + timedelta(days=1, hours=12)
    with pytest.raises(PointInTimeError, match="null available_time"):
        _realized_garch_history(frame, asof=asof)


def test_realized_garch_history_drops_unpublished_rows() -> None:
    frame = pl.DataFrame(
        {
            "event_time": _dates(2),
            "ret_1": [0.01, 0.02],
            "high": [1.1, 1.2],
            "low": [0.9, 1.0],
            "available_time": [
                D0 + timedelta(hours=1),
                D0 + timedelta(days=2),
            ],
        }
    )
    asof = D0 + timedelta(days=1, hours=12)
    dates, values, measures = _realized_garch_history(frame, asof=asof)
    assert dates.size == 1
    assert values.tolist() == [0.01]
    assert measures.size == 1


def test_require_garch_security_keys_rejects_exact_duplicates() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [D0, D0],
            "security_id": ["A", "A"],
            "ret_1": [0.01, 0.02],
        }
    )
    with pytest.raises(PointInTimeError, match="duplicate security_id/event_time"):
        _require_garch_security_keys(frame)


def test_garch_name_return_history_missing_id_column_and_unknown_name() -> None:
    with pytest.raises(PointInTimeError, match="missing security_id"):
        _garch_name_return_history(pl.DataFrame({"event_time": _dates(1), "ret_1": [0.01]}), "A")
    frame = pl.DataFrame({"event_time": _dates(1), "security_id": ["A"], "ret_1": [0.01]})
    dates, values = _garch_name_return_history(frame, "ZZZ")
    assert dates.size == 0
    assert values.size == 0


# --- density-capable OOS paths --------------------------------------------------


def test_garch_oos_predictions_density_records_origin_scores() -> None:
    history_dates = np.asarray(_dates(6), dtype=object)
    dates = np.asarray(_dates(6), dtype=object)
    predictions, statuses, density = _garch_oos_predictions(
        _DensityGarch,
        np.zeros((6, 2)),
        np.ones(6),
        dates,
        dates >= _dates(6)[3],
        label_horizon=2,
        return_dates=history_dates,
        return_values=np.linspace(0.01, 0.02, 6),
    )
    assert predictions.shape == (3,)
    assert statuses == ["fitted"] * 3
    assert set(density) == set(_dates(6)[3:])
    record = density[_dates(6)[3]]
    assert record["crps_method"] == "gaussian_closed"
    assert np.isfinite(record["log_score"])
    assert np.isfinite(record["crps"])


def test_garch_oos_predictions_missing_density_origin_fail_closed() -> None:
    # Density-capable model + origin absent from the evaluation return frame.
    dates = np.asarray(_dates(6), dtype=object)
    history_dates = np.asarray(_dates(5), dtype=object)
    with pytest.raises(ValueError, match="missing origin ret_1"):
        _garch_oos_predictions(
            _DensityGarch,
            np.zeros((6, 2)),
            np.ones(6),
            dates,
            dates == _dates(6)[5],
            label_horizon=1,
            return_dates=history_dates,
            return_values=np.ones(5),
        )


def test_garch_name_oos_duplicate_test_key_fail_closed() -> None:
    frame = _name_frame(_dates(3))
    dates = np.asarray(frame["event_time"].to_numpy())
    dup_dates = np.concatenate([dates, dates[2:3]])
    ids = np.array(["A", "A", "A", "A"], dtype=object)
    mask = dup_dates == dates.tolist()[2]
    with pytest.raises(ValueError, match="duplicate security_id/event_time"):
        _garch_name_oos_predictions(
            _DensityGarch,
            np.zeros((4, 1)),
            np.ones(4),
            dup_dates,
            ids,
            mask,
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_no_prior_for_sid_fail_closed() -> None:
    frame = _name_frame(_dates(3))
    dates = np.asarray(_dates(4), dtype=object)
    ids = np.array(["A", "A", "A", "B"], dtype=object)
    mask = ids == "B"
    with pytest.raises(ValueError, match="no strictly prior returns for 'B'"):
        _garch_name_oos_predictions(
            _DensityGarch,
            np.zeros((4, 1)),
            np.ones(4),
            dates,
            ids,
            mask,
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_missing_density_origin_fail_closed() -> None:
    # "B" has prior history but no row at the test origin: density target absent.
    frame = pl.DataFrame(
        {
            "event_time": [D0],
            "security_id": ["B"],
            "ret_1": [0.01],
        }
    )
    late = D0 + timedelta(days=5)
    dates = np.asarray([D0, late], dtype=object)
    ids = np.array(["B", "B"], dtype=object)
    mask = dates == late
    with pytest.raises(ValueError, match="missing origin ret_1"):
        _garch_name_oos_predictions(
            _DensityGarch,
            np.zeros((2, 1)),
            np.ones(2),
            dates,
            ids,
            mask,
            label_horizon=1,
            return_frame=frame,
        )


def test_garch_name_oos_density_record_scored() -> None:
    frame = _name_frame(_dates(4))
    dates = np.asarray(frame["event_time"].to_numpy())
    ids = np.array(["A"] * 4, dtype=object)
    mask = dates >= dates.tolist()[2]
    predictions, statuses, density = _garch_name_oos_predictions(
        _DensityGarch,
        np.zeros((4, 1)),
        np.ones(4),
        dates,
        ids,
        mask,
        label_horizon=1,
        return_frame=frame,
    )
    assert predictions.shape == (2,)
    assert statuses == ["fitted"] * 2
    assert len(density) == 2
    key = ("A", dates.tolist()[2])
    assert key in density
    assert density[key]["crps_method"] == "gaussian_closed"


# --- auto-selection happy paths --------------------------------------------------


def test_train_ranking_auto_selects_best_finite_ic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        path = Path(config.data.root) / "metadata" / f"ranker_{model_name}.joblib"
        model = RidgeRanker().fit(np.ones((4, 2)), np.arange(4, dtype=float))
        save_bound_stub(config, model, path)
        ic = {"ridge": 0.05, "elasticnet": 0.02, "neural": float("nan"), "ensemble": 0.04}[
            model_name
        ]
        return {"path": str(path), "metrics": {"mean_ic": ic, "mean_rank_ic": ic}}

    monkeypatch.setattr(train_module, "train_ranking", fake_train)
    result = train_ranking_auto(_cfg(tmp_path))
    assert result["selected_model"] == "ridge"
    assert result["model"] == "auto"
    assert result["selection_metric"] == "mean_ic"
    assert Path(result["path"]).name == "ranker_auto.joblib"
    assert set(result["candidates"]) == {"ridge", "elasticnet", "neural", "ensemble"}


def test_train_calibration_requires_columns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "quant_fund.pipeline.train.panel",
        lambda *a, **k: pl.DataFrame(
            {"event_time": _dates(3), "future_idio_return_1": [0.1, -0.1, 0.2]}
        ),
    )
    with pytest.raises(ValueError, match="calibration requires columns"):
        train_calibration(_cfg(tmp_path), "isotonic")


def test_train_calibration_fitted_fold_scores_oos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_train
) -> None:
    frame = _calibration_frame(24)
    stub_train(frame)
    train_mask = np.zeros(24, dtype=bool)
    train_mask[:14] = True
    test_mask = np.zeros(24, dtype=bool)
    test_mask[18:] = True
    monkeypatch.setattr(
        train_module, "_walk_forward_splits", lambda *a, **k: [(train_mask, test_mask)]
    )
    result = train_calibration(_cfg(tmp_path), "isotonic")
    assert np.isfinite(result["metrics"]["oos_brier"])
    assert result["metrics"]["n_oos_rows"] == 6.0
    assert Path(result["path"]).is_file()
    loaded = ProbabilityCalibrator.load(Path(result["path"]))
    assert loaded.fitted is True
    assert loaded.oos_start is not None
    assert loaded.oos_end is not None


def test_train_calibration_auto_selects_lowest_brier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        path = Path(config.data.root) / "metadata" / f"calibrator_{model_name}.joblib"
        calibrator = ProbabilityCalibrator(model_name)
        calibrator.fit(
            np.linspace(0.05, 0.95, 12),
            (np.linspace(0.05, 0.95, 12) > 0.5).astype(float),
        )
        save_bound_stub(config, calibrator, path)
        brier = {"isotonic": 0.11, "platt": 0.09}[model_name]
        return {"path": str(path), "metrics": {"oos_brier": brier}}

    monkeypatch.setattr(train_module, "train_calibration", fake_train)
    result = train_calibration_auto(_cfg(tmp_path))
    assert result["selected_model"] == "platt"
    assert result["selection_metric"] == "oos_brier"
    assert Path(result["path"]).name == "calibrator_auto.joblib"
    assert set(result["candidates"]) == {"isotonic", "platt"}


def test_train_distribution_auto_selects_lowest_pinball(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pinball = {
        "empirical": 0.30,
        "gaussian": 0.31,
        "linear_qr": float("nan"),
        "xgboost": 0.29,
        "lightgbm": 0.32,
    }

    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        path = Path(config.data.root) / "metadata" / f"dist_{model_name}.joblib"
        save_bound_stub(config, {"model": model_name}, path)
        return {
            "path": str(path),
            "metrics": {
                "mean_pinball": pinball[model_name],
                "n_oos_rows": 20.0,
            },
        }

    monkeypatch.setattr(train_module, "train_distribution", fake_train)
    result = train_distribution_auto(_cfg(tmp_path))
    assert result["selected_model"] == "xgboost"
    assert Path(result["path"]).name == "dist_auto.joblib"
    assert set(result["candidates"]) == set(pinball)


def test_train_volatility_auto_selects_lowest_qlike(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    qlike = {
        "rolling": 0.5,
        "ewma": 0.4,
        "har": float("nan"),
        "xgboost": 0.6,
        "lightgbm": 0.7,
    }

    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        path = Path(config.data.root) / "metadata" / f"vol_{model_name}.joblib"
        save_bound_stub(config, {"model": model_name}, path)
        return {
            "path": str(path),
            "metrics": {"qlike": qlike[model_name], "n_oos_rows": 20.0},
        }

    monkeypatch.setattr(train_module, "train_volatility", fake_train)
    result = train_volatility_auto(_cfg(tmp_path))
    assert result["selected_model"] == "ewma"
    assert Path(result["path"]).name == "vol_auto.joblib"
    assert set(result["candidates"]) == set(qlike)


# --- remaining function branches --------------------------------------------------


def test_train_ranking_no_evaluable_fold_fail_closed(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_ranking(_cfg(tmp_path), "ridge")


def test_train_ranking_dated_and_id_rankers(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    fnw = train_ranking(cfg, "fnw")  # DATED_FIT + DATED_PREDICT path
    assert Path(fnw["path"]).is_file()
    pp = train_ranking(cfg, "pp")  # ID_FIT + ID_PREDICT path
    assert Path(pp["path"]).is_file()


def test_train_alpha_mean_branch(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    result = train_alpha(_cfg(tmp_path), "mean")
    assert np.isfinite(result["metrics"]["mean"])
    assert Path(result["path"]).name == "alpha_mean.joblib"


class _PersistedFakeHMM:
    """Module-level stand-in so the regime artifact can be joblib-pickled."""

    labels = {"states": 3}

    def __init__(self, *_args: object) -> None:
        pass

    def fit(self, _x: object) -> _PersistedFakeHMM:
        return self

    def aic_bic(self, _x: object) -> dict[str, float]:
        return {"avg_ll": -1.25}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"hmm")


def test_train_regime_hmm_scores_heldout_likelihood(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    monkeypatch.setattr(train_module, "GaussianHMMRegime", _PersistedFakeHMM)
    result = train_regime(_cfg(tmp_path), "hmm")
    assert result["metrics"]["oos_avg_ll"] == -1.25
    assert result["labels"] == {"states": 3}
    assert Path(result["path"]).is_file()


def test_train_regime_hmm_no_evaluable_fold_fail_closed(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_stub_panel())
    monkeypatch.setattr(train_module, "_walk_forward_splits", lambda *a, **k: [])
    with pytest.raises(ValueError, match="no trainable/evaluable fold"):
        train_regime(_cfg(tmp_path), "hmm")


def test_train_tail_gaussian_and_drawdown_branches(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    gaussian = train_tail(cfg, "gaussian")
    assert np.isfinite(gaussian["metrics"]["oos_var_breach_rate"])
    assert np.isfinite(gaussian["metrics"]["var"])
    assert np.isfinite(gaussian["metrics"]["es"])
    assert Path(gaussian["path"]).name == "tail_gaussian.joblib"

    drawdown = train_tail(cfg, "drawdown")
    assert np.isfinite(drawdown["metrics"]["oos_brier"])
    assert Path(drawdown["path"]).name == "tail_drawdown.joblib"


def test_train_quantile_bandit_persists_policy(tmp_path: Path, stub_train) -> None:
    stub_train(_rl_frame(10))
    result = train_reinforcement(_cfg(tmp_path), "quantile_thompson")
    metrics = result["metrics"]
    assert metrics["n_dates"] >= 1
    assert np.isfinite(metrics["mean_policy_reward"])
    assert np.isfinite(metrics["mean_advantage_vs_uniform"])
    assert result["research_only"] is True
    assert result["live_pnl_claim"] is False
    assert Path(result["path"]).name == "rl_quantile_thompson.joblib"


def test_train_policy_gradient_persists_policy(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_rl_frame(10))
    trace = PolicyGradientTrace(
        dates=["d0", "d1"],
        policy_reward=np.array([0.4, 0.6]),
        uniform_reward=np.array([0.2, 0.2]),
        cumulative_regret=np.array([0.1, 0.2]),
    )
    monkeypatch.setattr(train_module, "run_policy_gradient_panel", lambda *a, **k: trace)

    monkeypatch.setattr(train_module, "PolicyGradientRanker", _StubPolicy)
    result = train_reinforcement(_cfg(tmp_path), "policy_gradient")
    metrics = result["metrics"]
    assert metrics["n_dates"] == 2.0
    assert metrics["mean_advantage_vs_uniform"] == pytest.approx(0.3)
    assert Path(result["path"]).name == "rl_policy_gradient.joblib"


def test_train_family_rejects_unknown_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown family"):
        train_family(_cfg(tmp_path), "risk")


def test_empty_trace_sentinels_match_trace_fields() -> None:
    # Keep the empty-trace stubs honest: field names mirror the real dataclasses.
    empty = np.asarray([], dtype=float)
    bandit = BanditTrace(
        dates=[],
        policy_reward=empty,
        oracle_reward=empty,
        uniform_reward=empty,
        cumulative_regret=empty,
    )
    assert bandit.policy_reward.size == 0
    quantile = QuantileBanditTrace(
        dates=[],
        policy_reward=empty,
        oracle_reward=empty,
        uniform_reward=empty,
        cumulative_regret=empty,
    )
    assert quantile.cumulative_regret.size == 0
    pg = PolicyGradientTrace(
        dates=[],
        policy_reward=empty,
        uniform_reward=empty,
        cumulative_regret=empty,
    )
    assert pg.dates == []


def test_train_distribution_auto_requires_min_oos_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_train(config, model_name):  # noqa: ANN001, ANN202
        return {
            "path": "unused",
            "metrics": {"mean_pinball": 0.3, "n_oos_rows": 3.0},
        }

    monkeypatch.setattr(train_module, "train_distribution", fake_train)
    with pytest.raises(ValueError, match="no finite candidate metric"):
        train_distribution_auto(_cfg(tmp_path))


def test_train_ranking_group_and_plain_fit_paths(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    lambdarank = train_ranking(cfg, "lambdarank")  # group-aware LightGBM ranker
    assert Path(lambdarank["path"]).is_file()
    elasticnet = train_ranking(cfg, "elasticnet")  # plain fit/predict ranker
    assert Path(elasticnet["path"]).is_file()


def test_train_volatility_rolling_and_har_branches(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    cfg = _cfg(tmp_path)
    rolling = train_volatility(cfg, "rolling")  # sigma forecast from vol_20 column
    assert np.isfinite(rolling["metrics"]["qlike"])
    har = train_volatility(cfg, "har")  # sigma forecast from the full design
    assert np.isfinite(har["metrics"]["qlike"])
    assert Path(har["path"]).name == "vol_har.joblib"


def test_train_volatility_realized_duplicate_density_origin_fail_closed(
    tmp_path: Path, stub_train, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub_train(_date_level_panel())
    monkeypatch.setattr(train_module, "RealizedGARCHVol", _DensityGarch)

    def fake_oos(_make_model, _x, _y, _dates, test_mask, **_kwargs):  # noqa: ANN001, ANN202
        return np.full(int(test_mask.sum()), 0.5), ["fitted"], {"fixed": {"log_score": -1.0}}

    monkeypatch.setattr(train_module, "_realized_garch_oos_predictions", fake_oos)
    with pytest.raises(ValueError, match="density origin repeated across folds"):
        train_volatility(_cfg(tmp_path), "realized_garch")


def test_train_regime_threshold_branch(tmp_path: Path, stub_train) -> None:
    stub_train(_stub_panel())
    result = train_regime(_cfg(tmp_path), "threshold")
    assert result["metrics"] == {}
    assert Path(result["path"]).name == "regime_threshold.joblib"
