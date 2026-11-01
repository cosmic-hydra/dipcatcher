"""SYNTHETIC equivalence: optimized hot paths match the previous definitions.

Cross-section tolerances are exact (frame equality). Backtest NAV and fills
are exact. Security-master rows match the previous per-symbol scan.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.backtest.engine import (
    _fast_replay_is_complete,
    _fast_replay_panel_supported,
    _run_backtest_event_loop,
    run_backtest,
)
from quant_fund.config.models import AppConfig, CostConfig, RiskGateConfig
from quant_fund.data.adapters.hf_ohlcv_1m import (
    OhlcvQualityError,
    _security_master_by_scan,
    security_master_from_bars,
)
from quant_fund.features.cross_sectional import apply_cross_sectional, decision_eligible_expr
from quant_fund.features.engine import CROSS_SECTIONAL_COLUMNS, compute_base_features
from tests.perf import check_regression
from tests.perf.synthetic import daily_ohlcv, minute_bars

_SCALE_FLOOR = 1e-12


def _reference_robust_z(value: pl.Expr, eligible: pl.Expr, keys: list[str]) -> pl.Expr:
    med = value.median().over(keys)
    mad = (value - med).abs().median().over(keys)
    sd = value.std().over(keys)
    scale = pl.when(mad > _SCALE_FLOOR).then(1.4826 * mad).otherwise(sd)
    z = pl.when(scale > _SCALE_FLOOR).then((value - med) / scale).otherwise(0.0)
    return pl.when(eligible).then(z).otherwise(None)


def _reference_apply_cross_sectional(
    df: pl.DataFrame,
    columns: list[str],
    winsor_p: float,
    sector: str | None = None,
) -> pl.DataFrame:
    """Previous per-column window implementation, kept as the oracle."""
    out = df
    lo, hi = winsor_p, 1.0 - winsor_p
    eligible = decision_eligible_expr(df)
    for col in columns:
        source = pl.when(eligible).then(pl.col(col)).otherwise(None)
        q_lo = source.quantile(lo).over("event_time")
        q_hi = source.quantile(hi).over("event_time")
        clipped_base = source.clip(q_lo, q_hi)
        clipped = pl.when(eligible).then(clipped_base).otherwise(None)
        robust_z = _reference_robust_z(clipped_base, eligible, ["event_time"])
        rank = source.rank("average").over("event_time")
        n = source.count().over("event_time")
        pct = pl.when(eligible).then((rank - 0.5) / n).otherwise(None)
        out = out.with_columns(
            clipped.alias(f"winsor_{col}"),
            robust_z.alias(f"cs_z_{col}"),
            pct.alias(f"cs_pct_{col}"),
        )
        if sector is not None and sector in out.columns:
            out = out.with_columns(
                _reference_robust_z(clipped_base, eligible, ["event_time", sector]).alias(
                    f"cs_z_sector_{col}"
                )
            )
    return out


def _assert_frames_equal(got: pl.DataFrame, ref: pl.DataFrame) -> None:
    assert got.columns == ref.columns
    assert got.schema == ref.schema
    assert got.equals(ref)


def test_cross_section_matches_window_oracle_on_features() -> None:
    bars = daily_ohlcv(12, 80, seed=11)
    built = compute_base_features(bars, AppConfig())
    columns = [name for name in CROSS_SECTIONAL_COLUMNS if name in built.columns]
    shuffled = built.sample(fraction=1.0, shuffle=True, seed=3)
    got = apply_cross_sectional(shuffled, columns, 0.01)
    ref = _reference_apply_cross_sectional(shuffled, columns, 0.01)
    _assert_frames_equal(got, ref)


def test_cross_section_matches_with_sector_nulls_and_membership() -> None:
    t0 = datetime(2020, 1, 2, tzinfo=UTC)
    t1 = datetime(2020, 1, 3, tzinfo=UTC)
    late = datetime(2020, 1, 4, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "event_time": [t0, t0, t0, t0, t1, t1, t1],
            "available_time": [t0, t0, late, t0, t1, t1, t1],
            "security_id": ["a", "b", "c", "d", "a", "b", "c"],
            "x": [1.0, 1.0, 50.0, None, 2.0, 3.0, 4.0],
            "y": [5.0, 5.0, 5.0, 9.0, 1.0, 1.0, 8.0],
            "sector": ["X", "X", None, "Y", "X", None, "X"],
            "_in_universe": [True, True, True, False, True, True, True],
        }
    )
    got = apply_cross_sectional(frame, ["x", "y"], 0.0, sector="sector")
    ref = _reference_apply_cross_sectional(frame, ["x", "y"], 0.0, sector="sector")
    _assert_frames_equal(got, ref)
    assert "cs_z_sector_x" in got.columns
    # The late row and the out-of-universe row do not receive a score.
    late_row = got.filter(pl.col("security_id") == "c").row(0, named=True)
    assert late_row["cs_z_x"] is None
    outside = got.filter(pl.col("security_id") == "d").row(0, named=True)
    assert outside["winsor_x"] is None


def test_cross_section_empty_columns_and_empty_frame() -> None:
    frame = pl.DataFrame({"event_time": [datetime(2020, 1, 2, tzinfo=UTC)], "x": [1.0]})
    assert apply_cross_sectional(frame, [], 0.01).equals(frame)
    empty = frame.clear()
    got = apply_cross_sectional(empty, ["x"], 0.01)
    ref = _reference_apply_cross_sectional(empty, ["x"], 0.01)
    _assert_frames_equal(got, ref)


def test_cross_section_preserves_user_columns_in_temporary_namespace() -> None:
    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 2, tzinfo=UTC)] * 3,
            "x": [1.0, 3.0, 2.0],
            "_cs_src_x": ["a", "b", "c"],
            "_cs_user": [10, 20, 30],
            "__cs_sec": ["first", "second", "third"],
        }
    )
    got = apply_cross_sectional(frame, ["x"], 0.0)
    ref = _reference_apply_cross_sectional(frame, ["x"], 0.0)
    _assert_frames_equal(got, ref)


def test_security_master_matches_row_scan() -> None:
    now = datetime(2024, 5, 1, 12, 30, tzinfo=UTC)
    bars = minute_bars(6, 4, seed=19)
    got = security_master_from_bars(bars, clock=lambda: now)
    ref = _security_master_by_scan(bars, now)
    _assert_frames_equal(got, ref)
    assert got["valid_to"].null_count() == got.height
    assert set(got["security_id"].to_list()) == set(bars["security_id"].unique().to_list())


def test_security_master_null_event_time_fails_closed() -> None:
    now = datetime(2024, 5, 1, tzinfo=UTC)
    bars = minute_bars(2, 1).with_columns(
        pl.lit(None).cast(pl.Datetime("us", "UTC")).alias("event_time")
    )
    with pytest.raises(OhlcvQualityError, match="has no event_time"):
        security_master_from_bars(bars, clock=lambda: now)


def test_security_master_empty() -> None:
    from quant_fund.data.adapters.hf_ohlcv_1m import empty_bars

    master = security_master_from_bars(empty_bars(), clock=lambda: datetime(2024, 1, 1, tzinfo=UTC))
    assert master.is_empty()
    assert "security_id" in master.columns


def _book(
    n_symbols: int = 6, n_days: int = 40, *, seed: int = 5
) -> tuple[pl.DataFrame, pl.DataFrame]:
    bars = daily_ohlcv(n_symbols, n_days, seed=seed).with_columns(
        (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.lit(0.02).alias("vol_20"),
    )
    # Drop one name on a few dates so the carried-target and missing-mark paths run.
    drop = (pl.col("security_id") == "S0001") & (pl.col("event_time").dt.day() % 7 == 0)
    bars = bars.filter(~drop)
    weights = bars.select("event_time", "security_id").with_columns(
        pl.when(pl.col("security_id").str.ends_with("0"))
        .then(0.0)
        .otherwise(0.5 / n_symbols)
        .alias("target_weight")
    )
    dates = weights.select("event_time").unique().sort("event_time").gather_every(2)
    weights = weights.join(dates, on="event_time", how="semi")
    return bars, weights


def _backtest_config(**overrides: object) -> AppConfig:
    config = AppConfig(
        costs=CostConfig(
            commission_bps=1.0,
            half_spread_bps=0.5,
            impact_y=0.1,
            participation_limit=0.2,
        ),
        risk_gate=RiskGateConfig(
            max_order_notional=1e12,
            max_gross=2.0,
            max_net=1.5,
            max_name=0.3,
            max_participation=0.5,
            max_predicted_vol=1.0,
            stale_price_bars=5,
        ),
    )
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_backtest_fast_dispatch_matches_event_loop(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    bars, weights = _book()
    config = _backtest_config()
    config.data.root = root
    assert _fast_replay_is_complete(config, None) is True
    got = run_backtest(bars, weights, config, initial_nav=1_000_000.0)
    ref = _run_backtest_event_loop(bars, weights, config, initial_nav=1_000_000.0)
    assert got.equity.equals(ref.equity)
    assert got.fills.equals(ref.fills)
    assert got.metrics == ref.metrics
    assert got.metrics["garch_risk_overlay_dates"] == 0
    assert got.metrics["realized_garch_risk_overlay_dates"] == 0
    nav = got.equity["nav"].to_numpy()
    ref_nav = ref.equity["nav"].to_numpy()
    assert nav.shape == ref_nav.shape
    assert np.array_equal(nav, ref_nav)


def test_backtest_mixed_datetime_units_stay_on_reference_path(tmp_path: object) -> None:
    from pathlib import Path

    bars, weights = _book(n_symbols=2, n_days=8)
    bars = bars.with_columns(pl.col("event_time").cast(pl.Datetime("ns", "UTC")))
    assert bars.schema["event_time"] != weights.schema["event_time"]
    assert _fast_replay_panel_supported(bars, weights) is False
    config = _backtest_config()
    config.data.root = Path(str(tmp_path))
    got = run_backtest(bars, weights, config)
    ref = _run_backtest_event_loop(bars, weights, config)
    assert got.equity.equals(ref.equity) and got.fills.equals(ref.fills)


def test_backtest_duplicate_bars_fail_closed_on_both_paths(tmp_path: object) -> None:
    """Duplicate bar keys are ambiguous — both engines refuse rather than
    last-write-wins (order-dependent fills). Fast replay already refused;
    the event loop now matches the paper-loop guard.
    """
    from pathlib import Path

    bars, weights = _book(n_symbols=2, n_days=8)
    duplicate = bars.head(1).with_columns(pl.lit(None).cast(pl.Float64).alias("adv"))
    bars = pl.concat([bars, duplicate])
    assert _fast_replay_panel_supported(bars, weights) is False
    config = _backtest_config()
    config.data.root = Path(str(tmp_path))
    with pytest.raises(ValueError, match="duplicate bars"):
        _run_backtest_event_loop(bars, weights, config)
    with pytest.raises(ValueError, match="duplicate bars"):
        run_backtest(bars, weights, config, fast=True)


def test_perf_ci_presence_requires_exact_benchmark_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(check_regression, "_CI_SUFFIXES", ("test_required",))
    payload = {
        "benchmarks": [
            {
                "fullname": "tests/perf/test_benchmarks.py::test_calibration_matmul",
                "stats": {"median": 1.0},
            },
            {
                "fullname": "tests/perf/test_benchmarks.py::fake_test_required",
                "stats": {"median": 1.0},
            },
        ]
    }
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text(json.dumps(payload), encoding="utf-8")
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    assert check_regression.main([str(current), str(baseline), "--ci"]) == 1


def test_regression_ignores_matmul_that_outruns_the_bench_pack() -> None:
    """Xeon Platinum shape: gemm slightly faster, benches moved together.

    The 2026-10-30 perf job failed three benches at 1.51× because the 512
    gemm (3.77 ms vs a 4.00 ms baseline) did not slow down with them. The
    same tree passed the next day when the gemm slowed with the benches.
    Factors are current/baseline medians from that job.
    """
    factors = [
        1.4244,
        1.3447,
        1.3245,
        1.2546,
        1.3254,
        1.4186,
        1.3115,
        1.2133,
        1.4474,
    ]
    pack = sorted(factors)[len(factors) // 2]
    cal = 3.770 / 4.004
    assert pack / cal <= 1.50
    for factor in factors:
        assert check_regression._is_regression(factor, pack, cal, 0.50) is False


def test_regression_flags_uniform_and_isolated_slowdowns() -> None:
    assert check_regression._is_regression(2.0, 2.0, 1.0, 0.50) is True
    assert check_regression._is_regression(2.0, 1.0, 1.0, 0.50) is True
    assert check_regression._is_regression(1.2, 1.1, 1.0, 0.50) is False


def test_regression_cli_passes_when_gemm_outruns_pack(tmp_path: Path) -> None:
    def _payload(medians: dict[str, float]) -> dict[str, object]:
        rows = [
            {"fullname": f"tests/perf/test_benchmarks.py::{name}", "stats": {"median": median}}
            for name, median in medians.items()
        ]
        return {"benchmarks": rows}

    # Pack stays within 1.5× of the fast gemm; one bench's raw ratio is over.
    matched = {
        "test_calibration_matmul": 0.94,
        "test_backtest[8sym-126d]": 1.30,
        "test_security_master[8sym-20d]": 1.42,
        "test_dip_detect[40sym-1000d]": 1.25,
    }
    current = tmp_path / "current.json"
    baseline = tmp_path / "baseline.json"
    current.write_text(json.dumps(_payload(matched)), encoding="utf-8")
    baseline.write_text(
        json.dumps(_payload({name: 1.0 for name in matched})),
        encoding="utf-8",
    )
    assert check_regression.main([str(current), str(baseline), "--threshold", "0.50"]) == 0
    slowed = {name: (1.0 if "calibration" in name else 2.0) for name in matched}
    current.write_text(json.dumps(_payload(slowed)), encoding="utf-8")
    assert check_regression.main([str(current), str(baseline), "--threshold", "0.50"]) == 1


def test_backtest_skips_fast_path_when_overlay_artifact_exists(tmp_path: object) -> None:
    from pathlib import Path

    root = Path(str(tmp_path))
    (root / "metadata").mkdir()
    (root / "metadata" / "vol_garch.joblib").write_bytes(b"present")
    config = _backtest_config()
    config.data.root = root
    assert _fast_replay_is_complete(config, None) is False
    (root / "metadata" / "vol_realized_garch.joblib").write_bytes(b"present")
    (root / "metadata" / "vol_garch.joblib").unlink()
    assert _fast_replay_is_complete(config, None) is False


def test_close_auction_does_not_dispatch(tmp_path: object) -> None:
    from pathlib import Path

    config = _backtest_config()
    config.data.root = Path(str(tmp_path))
    config.execution.allow_close_auction = True
    assert _fast_replay_is_complete(config, None) is False
    bars, weights = _book(n_symbols=4, n_days=15, seed=9)
    got = run_backtest(bars, weights, config, initial_nav=1_000_000.0)
    ref = _run_backtest_event_loop(bars, weights, config, initial_nav=1_000_000.0)
    assert got.equity.equals(ref.equity)
    assert got.fills.equals(ref.fills)
