"""Edge branches of the bar quality checks that the scorecard happy path skips."""

from __future__ import annotations

from datetime import UTC, date, datetime

import polars as pl
import pytest

from quant_fund.data.quality.checks import (
    check_mad_outliers,
    check_missing_bars,
    check_non_finite,
    check_timezone,
    check_volume,
    translate_lakehouse_checks,
)
from quant_fund.data.quality.models import QualityConfig

ASOF = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)


def test_lakehouse_examples_skip_non_rows_and_blank_limits() -> None:
    report = {
        "checks": [
            {"name": "gaps", "applicable": True, "count": 3},
            {
                "name": "duplicates",
                "applicable": False,
                "count": None,
                "examples": "not-a-list",
            },
            {
                "name": "stale_prices",
                "applicable": True,
                "count": 1,
                "detail": "stale",
                "max_run_length": 4,
                "examples": ["skip", {"event_time": None, "symbol": "A", "note": 1}],
            },
        ]
    }
    skipped = translate_lakehouse_checks(
        report, pl.DataFrame({"x": [1]}), None, None, QualityConfig(max_offenders=0)
    )
    assert [item.name for item in skipped] == ["duplicates", "stale_prices"]
    assert skipped[0].worst == []
    assert skipped[0].checked == 0

    kept = translate_lakehouse_checks(
        report, pl.DataFrame({"x": [1]}), "symbol", "event_time", QualityConfig(max_offenders=2)
    )
    assert kept[1].worst[0].timestamp is None
    assert kept[1].worst[0].extra["note"] == 1
    assert kept[1].worst[0].extra["max_run_length"] == 4


def test_non_finite_and_volume_refuse_unusable_frames() -> None:
    empty = pl.DataFrame({"close": pl.Series([], dtype=pl.Float64)})
    assert check_non_finite(empty, QualityConfig()).applicable is False
    text = pl.DataFrame({"label": ["a"]})
    assert check_non_finite(text, QualityConfig()).applicable is False

    bare = pl.DataFrame({"close": [float("nan")]})
    offenders = check_non_finite(bare, QualityConfig(symbol_column=None, clock_column=None))
    assert offenders.violations == 1
    assert offenders.worst[0].symbol is None
    assert offenders.worst[0].timestamp is None

    assert check_volume(pl.DataFrame({"close": [1.0]}), QualityConfig()).applicable is False
    strings = pl.DataFrame({"volume": ["lots"]})
    assert check_volume(strings, QualityConfig()).detail.startswith("volume dtype")
    bad = pl.DataFrame({"volume": [0.0, -1.0, None]})
    assert check_volume(bad, QualityConfig()).violations == 2


def test_timezone_date_naive_and_non_temporal_clocks() -> None:
    cfg = QualityConfig(clock_column="event_time")
    assert check_timezone(pl.DataFrame({"close": [1.0]}), None, cfg).applicable is False
    aware = pl.DataFrame({"event_time": [ASOF]})
    assert check_timezone(aware, "event_time", cfg).violations == 0
    naive = pl.DataFrame({"event_time": [datetime(2024, 1, 2, 14, 30)]})
    assert check_timezone(naive, "event_time", cfg).violations == 1
    dated = pl.DataFrame({"event_time": [date(2024, 1, 2)]})
    dated_result = check_timezone(dated, "event_time", cfg)
    assert "Date-typed" in dated_result.detail
    assert (
        check_timezone(pl.DataFrame({"event_time": ["tuesday"]}), "event_time", cfg).applicable
        is False
    )


def test_mad_outliers_cover_missing_inputs_and_a_spike() -> None:
    cfg = QualityConfig(clock_column="event_time", symbol_column="symbol", mad_z_threshold=3.5)
    assert (
        check_mad_outliers(
            pl.DataFrame({"event_time": [ASOF]}), "symbol", "event_time", cfg
        ).applicable
        is False
    )
    strings = pl.DataFrame({"event_time": [ASOF], "close": ["ten"]})
    assert check_mad_outliers(strings, None, "event_time", cfg).applicable is False
    nulls = pl.DataFrame(
        {"symbol": ["A", "A"], "event_time": [ASOF, ASOF], "close": [None, None]},
        schema={"symbol": pl.Utf8, "event_time": pl.Datetime(time_zone="UTC"), "close": pl.Float64},
    )
    quiet = check_mad_outliers(nulls, "symbol", "event_time", cfg)
    assert quiet.detail == "no finite returns to score"

    spiked = pl.DataFrame(
        {
            "symbol": ["A"] * 8 + ["B"] * 2,
            "event_time": [ASOF] * 10,
            "close": [100.0] * 7 + [1.0, 50.0, 50.0],
        }
    )
    found = check_mad_outliers(spiked, "symbol", "event_time", cfg)
    assert found.violations >= 1
    assert found.worst[0].value is not None


def test_missing_bars_numeric_string_and_nonpositive_intervals() -> None:
    cfg = QualityConfig(clock_column="t", symbol_column="symbol")
    assert check_missing_bars(pl.DataFrame({"t": [1]}), "symbol", None, cfg).applicable is False

    lists = pl.DataFrame({"symbol": ["A", "A"], "t": [[1], [2]]})
    unmeasurable = check_missing_bars(lists, "symbol", "t", cfg)
    assert unmeasurable.applicable is False

    one = pl.DataFrame({"symbol": ["A"], "t": [1]})
    assert "fewer than two" in check_missing_bars(one, "symbol", "t", cfg).detail

    zero_cfg = QualityConfig(clock_column="t", expected_interval_seconds=0.0)
    flat = pl.DataFrame({"t": [1, 2, 3]})
    assert "could not infer" in check_missing_bars(flat, None, "t", zero_cfg).detail

    gapped = pl.DataFrame({"symbol": ["A", "A", "A"], "t": [0, 1, 10]})
    configured = check_missing_bars(
        gapped,
        "symbol",
        "t",
        QualityConfig(
            clock_column="t", symbol_column="symbol", expected_interval_seconds=1.0, gap_factor=1.5
        ),
    )
    assert configured.violations == 1
    assert "configured" in configured.detail

    inferred = check_missing_bars(
        gapped, None, "t", QualityConfig(clock_column="t", gap_factor=1.5)
    )
    assert inferred.violations == 1
    assert "median spacing" in inferred.detail

    def _zero(_value: object) -> float:
        return 0.0

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("quant_fund.data.quality.checks._as_float", _zero)
    try:
        skipped = check_missing_bars(
            gapped, "symbol", "t", QualityConfig(clock_column="t", symbol_column="symbol")
        )
    finally:
        monkeypatch.undo()
    assert skipped.violations == 0


def test_as_float_rejects_bools_and_non_numerics() -> None:
    from quant_fund.data.quality.checks import _as_float

    assert _as_float(True) == 0.0
    assert _as_float("nope") == 0.0
    assert _as_float(3) == 3.0
