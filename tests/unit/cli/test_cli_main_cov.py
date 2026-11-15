"""cli/main.py internals: command bodies beyond the --help surface tests.

Complements ``test_cli_surface_cov.py`` (registration/help/fail-clean surface)
and the per-command echo suites: exercises the bodies those files leave
uncovered — ingest/build-features/train dispatch, forecast/kronos/optimize/
backtest wiring, ``--book``/``--bars-path`` handling in candle-book and
kyle-ofi, the research family echoes, session-book ``--daily-out``,
vendor-book-map ``--columns``/``--parquet``, verify-research / lab / report /
api / paper / monitor / sim-live, and the ``__main__`` entrypoint.

Heavy pipeline backends are monkeypatched at their *source* module (every
command late-imports inside its body) so tests stay hermetic and offline;
configs point ``data.root`` at ``tmp_path``.
"""

from __future__ import annotations

import importlib
import re
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import _collect_param_value, app

# The package re-exports a function named ``ingest``, shadowing the submodule
# attribute — resolve the module itself so monkeypatching works.
ingest_mod = importlib.import_module("quant_fund.data.ingest")

runner = CliRunner()

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI styling — typer renders error usage with rich when color is forced."""
    return _ANSI.sub("", text)


def _write_config(tmp_path: Path, body: str) -> Path:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(body)
    return cfg


def _data_config(tmp_path: Path, source: str = "synthetic", extra: str = "") -> Path:
    return _write_config(
        tmp_path, f"data:\n  root: {tmp_path.as_posix()}\n  source: {source}\n{extra}"
    )


def _stub_logger(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(info=lambda *a, **k: None)


def _book_panel_frame() -> pl.DataFrame:
    """Minimal frame satisfying the Northset book-panel contract."""
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    return pl.DataFrame(
        {
            "security_id": ["A"],
            "event_time": [ts],
            "available_time": [ts],
            "source": ["synthetic"],
            "best_bid": [99.0],
            "best_ask": [101.0],
            "mid": [100.0],
            "spread": [2.0],
            "spread_bps": [200.0],
            "microprice": [100.0],
            "microprice_minus_mid": [0.0],
            "microprice_minus_mid_bps": [0.0],
            "imbalance_top": [0.0],
            "imbalance_depth": [0.0],
            "bid_depth": [10.0],
            "ask_depth": [10.0],
            "top_bid_size": [5.0],
            "top_ask_size": [5.0],
        }
    )


def _alpaca_quotes_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "timestamp": [
                datetime(2026, 1, 2, 15, 0, tzinfo=UTC),
                datetime(2026, 1, 2, 15, 1, tzinfo=UTC),
            ],
            "bid_price": [100.0, 100.1],
            "ask_price": [100.2, 100.3],
            "bid_size": [500.0, 600.0],
            "ask_size": [400.0, 300.0],
        }
    )


def _synth_bars(volume: float = 1_000.0, with_volume: bool = True) -> pl.DataFrame:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    data: dict[str, list] = {
        "security_id": ["A"],
        "event_time": [ts],
        "close": [100.0],
    }
    if with_volume:
        data["volume"] = [volume]
    return pl.DataFrame(data)


def _patch_synthetic_provider(monkeypatch: pytest.MonkeyPatch, bars: pl.DataFrame) -> None:
    monkeypatch.setattr(
        "quant_fund.data.adapters.synthetic.SyntheticMarketProvider",
        lambda *a, **k: SimpleNamespace(get_bars=lambda: bars),
    )


# --- _collect_param_value ------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5", 5),
        ("-12", -12),
        ("1.5", 1.5),
        ("1e3", 1000.0),
        (" 7 ", 7),
        ("alpha", "alpha"),
        ("", ""),
    ],
)
def test_collect_param_value_coercion(raw: str, expected: object) -> None:
    assert _collect_param_value(raw) == expected


# --- ingest ---------------------------------------------------------------


def test_ingest_synthetic_echoes_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    monkeypatch.setattr(ingest_mod, "ingest", lambda c: {"bars": tmp_path / "bars.parquet"})
    monkeypatch.setattr("quant_fund.cli.data_cmds.get_logger", _stub_logger)
    result = runner.invoke(app, ["ingest", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC ingest complete" in result.output
    assert "bars:" in result.output


def test_ingest_non_synthetic_skips_banner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path, source="file")
    monkeypatch.setattr(ingest_mod, "ingest", lambda c: {"bars": tmp_path / "b.parquet"})
    monkeypatch.setattr("quant_fund.cli.data_cmds.get_logger", _stub_logger)
    result = runner.invoke(app, ["ingest", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC ingest complete" not in result.output


# --- build-features / build-labels ----------------------------------------


def test_build_features_and_labels_echo_gold_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feats = pl.DataFrame({"a": [1, 2, 3]})
    labs = pl.DataFrame({"b": [1]})
    monkeypatch.setattr("quant_fund.pipeline.dataset.build_gold", lambda c: (feats, labs))
    cfg = _data_config(tmp_path)
    result = runner.invoke(app, ["build-features", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "features rows=3 cols=1" in result.output
    result = runner.invoke(app, ["build-labels", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "labels rows=1" in result.output


# --- train ----------------------------------------------------------------


def test_train_without_family_lists_families() -> None:
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0, result.output
    assert "Specify a family" in result.output


@pytest.mark.parametrize(
    ("family", "expected_model"),
    [
        ("ranking", "ridge"),
        ("distribution", "gaussian"),
        ("calibration", "isotonic"),
        ("volatility", "ewma"),
        ("alpha", "ridge"),
        ("regime", "hmm"),
        ("tail", "historical"),
        ("reinforcement", "linucb"),
        ("covariance", None),
        ("liquidity", None),
    ],
)
def test_train_family_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, expected_model: str | None
) -> None:
    calls: list[tuple[str, object]] = []

    def fake(cfg: object, fam: str, mdl: object) -> str:
        calls.append((fam, mdl))
        return f"trained {fam}"

    monkeypatch.setattr("quant_fund.cli.train_cmds.train_family", fake)
    result = runner.invoke(app, ["train", family, "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert calls == [(family, expected_model)]
    assert f"trained {family}" in result.output


def test_train_model_option_forwarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        "quant_fund.cli.train_cmds.train_family",
        lambda cfg, fam, mdl: calls.append((fam, mdl)) or "ok",
    )
    result = runner.invoke(
        app,
        ["train", "ranking", "--model", "xgboost", "--config", str(_data_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert calls == [("ranking", "xgboost")]


# --- forecast / kronos-forecast / optimize ---------------------------------


def _forecast_row(
    symbol: str = "BTC", lo: float | None = None, hi: float | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=symbol,
        interval_lo={"5d": lo} if lo is not None else {},
        interval_hi={"5d": hi} if hi is not None else {},
        interval_alpha=0.1,
        interval_method="cqr",
        alpha={"5d": 0.001},
        rank_percentile={"5d": 0.5},
        volatility={"5d": 0.02},
    )


def test_forecast_echoes_intervals_and_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = SimpleNamespace(
        notes={"SYNTHETIC"},
        forecasts=[_forecast_row(lo=-0.01, hi=0.02), _forecast_row(symbol="ETH")],
    )
    monkeypatch.setattr("quant_fund.pipeline.forecast.forecast_asof", lambda cfg, asof: state)
    result = runner.invoke(app, ["forecast", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "interval[5d]=" in result.output
    assert "ETH" in result.output


def test_forecast_passes_date_and_skips_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}
    state = SimpleNamespace(notes=set(), forecasts=[])

    def fake(cfg: object, asof: object) -> SimpleNamespace:
        seen["asof"] = asof
        return state

    monkeypatch.setattr("quant_fund.pipeline.forecast.forecast_asof", fake)
    result = runner.invoke(
        app,
        ["forecast", "--date", "2026-01-05", "--config", str(_data_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" not in result.output
    assert seen["asof"] == datetime(2026, 1, 5, tzinfo=UTC)


def test_kronos_forecast_echoes_quantiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = SimpleNamespace(
        symbol="BTC",
        expected_returns={"5d": 0.001},
        quantiles={"5d": {0.05: -0.01, 0.5: 0.0, 0.95: 0.02}},
        probability_positive={"5d": 0.6},
        volatility={"5d": 0.02},
    )
    state = SimpleNamespace(notes={"SYNTHETIC"}, forecasts=[row])
    monkeypatch.setattr(
        "quant_fund.pipeline.kronos.forecast_kronos_frame",
        lambda cfg, asof=None: state,
    )
    result = runner.invoke(app, ["kronos-forecast", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "kronos.adapter.v1" in result.output
    assert "q05=" in result.output


def test_kronos_forecast_non_synthetic_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = SimpleNamespace(notes=set(), forecasts=[])
    monkeypatch.setattr(
        "quant_fund.pipeline.kronos.forecast_kronos_frame",
        lambda cfg, asof=None: state,
    )
    result = runner.invoke(app, ["kronos-forecast", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" not in result.output
    assert "kronos.adapter.v1" in result.output


def test_kronos_forecast_value_error_becomes_bad_parameter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake(cfg: object, asof: object = None) -> SimpleNamespace:
        raise ValueError("kronos disabled")

    monkeypatch.setattr("quant_fund.pipeline.kronos.forecast_kronos_frame", fake)
    result = runner.invoke(app, ["kronos-forecast", "--config", str(_data_config(tmp_path))])
    assert result.exit_code != 0
    assert "kronos disabled" in result.output


def test_optimize_echoes_weights_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class _Weights:
        def head(self, n: int) -> str:
            return "weights-head"

    def fake(cfg: object, asof: object) -> _Weights:
        seen["asof"] = asof
        return _Weights()

    monkeypatch.setattr("quant_fund.pipeline.forecast.optimize_asof", fake)
    result = runner.invoke(
        app,
        ["optimize", "--date", "2026-01-05", "--config", str(_data_config(tmp_path))],
    )
    assert result.exit_code == 0, result.output
    assert "weights-head" in result.output
    assert seen["asof"] == datetime(2026, 1, 5, tzinfo=UTC)


# --- backtest -------------------------------------------------------------


def test_backtest_rejects_unknown_engine(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["backtest", "--engine", "bogus", "--config", str(_data_config(tmp_path))]
    )
    assert result.exit_code != 0
    assert "--engine must be 'ref' or 'fast'" in _plain(result.output)


@pytest.mark.parametrize("engine", ["ref", "fast"])
def test_backtest_dispatches_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, engine: str
) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    feat = pl.DataFrame({"event_time": [ts, datetime(2026, 1, 2, tzinfo=UTC)]})
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr("quant_fund.features.engine.build_features", lambda b, c: feat)
    monkeypatch.setattr("quant_fund.pipeline.forecast.decision_dates", lambda c, ds: ds)
    monkeypatch.setattr(
        "quant_fund.pipeline.forecast.build_causal_weight_panel",
        lambda c, d: pl.DataFrame({"target_weight": [0.5]}),
    )
    calls: list[str] = []
    run = SimpleNamespace(source_note="SYNTHETIC", metrics={"sharpe": 1.0})
    monkeypatch.setattr(
        "quant_fund.backtest.engine.run_backtest",
        lambda f, w, c: calls.append("ref") or run,
    )
    monkeypatch.setattr(
        "quant_fund.backtest.fast_replay.run_backtest_fast",
        lambda f, w, c: calls.append("fast") or run,
    )
    result = runner.invoke(
        app, ["backtest", "--engine", engine, "--config", str(_data_config(tmp_path))]
    )
    assert result.exit_code == 0, result.output
    assert calls == [engine]
    assert "SYNTHETIC" in result.output
    assert "sharpe" in result.output


def test_backtest_restricts_grid_to_gold_panel_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The feature grid is wider than the causal gold panel (membership
    warmup + label tail). backtest must intersect before optimizing —
    optimize_asof fails closed on off-panel dates."""
    d1 = datetime(2026, 1, 1, tzinfo=UTC)
    d2 = datetime(2026, 1, 2, tzinfo=UTC)
    d3 = datetime(2026, 1, 3, tzinfo=UTC)
    feat = pl.DataFrame({"event_time": [d1, d2, d3], "close": [1.0, 1.0, 1.0]})
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr("quant_fund.features.engine.build_features", lambda b, c: feat)
    monkeypatch.setattr(
        "quant_fund.pipeline.forecast.decision_dates",
        lambda c, ds: [d for d in ds if d in (d2, d3)],
    )
    seen: dict[str, object] = {}

    def _weights(cfg: object, ds: list[datetime]) -> pl.DataFrame:
        seen["dates"] = list(ds)
        return pl.DataFrame({"target_weight": [0.5]})

    def _run(f: pl.DataFrame, w: pl.DataFrame, c: object) -> SimpleNamespace:
        seen["feat_dates"] = f["event_time"].to_list()
        return SimpleNamespace(source_note="SYNTHETIC", metrics={"sharpe": 0.0})

    monkeypatch.setattr("quant_fund.pipeline.forecast.build_causal_weight_panel", _weights)
    monkeypatch.setattr("quant_fund.backtest.engine.run_backtest", _run)
    result = runner.invoke(app, ["backtest", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert seen["dates"] == [d2, d3]
    assert seen["feat_dates"] == [d2, d3]


def test_backtest_rejects_when_no_dates_overlap_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr(
        "quant_fund.features.engine.build_features",
        lambda b, c: pl.DataFrame({"event_time": [datetime(2026, 1, 1, tzinfo=UTC)]}),
    )
    monkeypatch.setattr("quant_fund.pipeline.forecast.decision_dates", lambda c, ds: [])
    result = runner.invoke(app, ["backtest", "--config", str(_data_config(tmp_path))])
    assert result.exit_code != 0
    assert "causal gold panel" in _plain(result.output)


def test_backtest_non_synthetic_source_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    feat = pl.DataFrame({"event_time": [ts, datetime(2026, 1, 2, tzinfo=UTC)]})
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr("quant_fund.features.engine.build_features", lambda b, c: feat)
    monkeypatch.setattr("quant_fund.pipeline.forecast.decision_dates", lambda c, ds: ds)
    monkeypatch.setattr(
        "quant_fund.pipeline.forecast.build_causal_weight_panel",
        lambda c, d: pl.DataFrame({"target_weight": [0.5]}),
    )
    run = SimpleNamespace(source_note="VENDOR_BARS", metrics={"sharpe": 0.0})
    monkeypatch.setattr("quant_fund.backtest.engine.run_backtest", lambda f, w, c: run)
    result = runner.invoke(app, ["backtest", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" not in result.output


# --- candle-book ----------------------------------------------------------


def _patch_candle_bench(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    receipt: dict[str, object] = {
        "n_bars": 2,
        "n_scored": 2,
        "depth": 5,
        "ic_demo": 0.5,
        "ic_nan": float("nan"),
    }
    monkeypatch.setattr(
        "quant_fund.microstructure.bench_candle_order_book", lambda *a, **k: receipt
    )
    _patch_synthetic_provider(monkeypatch, _synth_bars())
    return receipt


def test_candle_book_requires_vendor_for_raw_quotes(tmp_path: Path) -> None:
    cfg = _data_config(tmp_path)
    quotes = tmp_path / "quotes.parquet"
    pl.DataFrame({"symbol": ["A"], "ts": ["x"]}).write_parquet(quotes)
    result = runner.invoke(app, ["candle-book", "--config", str(cfg), "--book", str(quotes)])
    assert result.exit_code != 0
    assert "pass --vendor" in _plain(result.output)


def test_candle_book_vendor_remap_then_bench(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_candle_bench(monkeypatch)
    quotes = tmp_path / "quotes.parquet"
    _alpaca_quotes_frame().write_parquet(quotes)
    result = runner.invoke(
        app,
        ["candle-book", "--config", str(cfg), "--book", str(quotes), "--vendor", "alpaca"],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "ic_demo=0.5000" in result.output
    # NaN metric rows are filtered out of the ic echo, not printed.
    assert "ic_nan" not in result.output


def test_candle_book_accepts_validated_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_candle_bench(monkeypatch)
    book = tmp_path / "panel.parquet"
    _book_panel_frame().write_parquet(book)
    result = runner.invoke(app, ["candle-book", "--config", str(cfg), "--book", str(book)])
    assert result.exit_code == 0, result.output


def test_candle_book_panel_missing_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    book = tmp_path / "panel.parquet"
    _book_panel_frame().drop("source").write_parquet(book)
    monkeypatch.setattr(
        "quant_fund.microstructure.book_panel.load_book_panel",
        lambda p: pl.read_parquet(p),
    )
    result = runner.invoke(app, ["candle-book", "--config", str(cfg), "--book", str(book)])
    assert result.exit_code != 0
    assert "missing source column" in _plain(result.output)


def test_candle_book_reads_bars_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_candle_bench(monkeypatch)
    bars = tmp_path / "bars.parquet"
    _synth_bars().write_parquet(bars)
    result = runner.invoke(app, ["candle-book", "--config", str(cfg), "--bars-path", str(bars)])
    assert result.exit_code == 0, result.output


# --- kyle-ofi -------------------------------------------------------------


def _patch_kyle_backend(monkeypatch: pytest.MonkeyPatch, receipt: dict[str, object]) -> None:
    monkeypatch.setattr(
        "quant_fund.northset.kyle_ofi.bench_kyle_ofi_fused", lambda *a, **k: receipt
    )
    monkeypatch.setattr(
        "quant_fund.northset.kyle_ofi.fuse_bars_l2_kyle_frame",
        lambda *a, **k: pl.DataFrame(
            {"flow": ["x"], "event_time": [datetime(2026, 1, 1, tzinfo=UTC)]}
        ),
    )
    monkeypatch.setattr(
        "quant_fund.northset.kyle_ofi.kyle_lambda_date_series_frame",
        lambda *a, **k: pl.DataFrame(
            {"flow": ["x"], "event_time": [datetime(2026, 1, 1, tzinfo=UTC)]}
        ),
    )


def _kyle_receipt(book_dgp: str = "synthetic_l2") -> dict[str, object]:
    return {
        "n_fused": 1,
        "n_scored": 1,
        "book_source": "synthetic",
        "book_dgp": book_dgp,
        "data_source": "synthetic",
    }


def test_kyle_ofi_valid_panel_skips_synth_book(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_kyle_backend(monkeypatch, _kyle_receipt())
    _patch_synthetic_provider(monkeypatch, _synth_bars())
    book = tmp_path / "panel.parquet"
    _book_panel_frame().write_parquet(book)
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg), "--book", str(book)])
    assert result.exit_code == 0, result.output
    assert "n_fused=1" in result.output


def test_kyle_ofi_requires_vendor_for_raw_quotes(tmp_path: Path) -> None:
    cfg = _data_config(tmp_path)
    quotes = tmp_path / "quotes.parquet"
    pl.DataFrame({"symbol": ["A"]}).write_parquet(quotes)
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg), "--book", str(quotes)])
    assert result.exit_code != 0
    assert "pass --vendor" in _plain(result.output)


def test_kyle_ofi_panel_missing_source_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    book = tmp_path / "panel.parquet"
    _book_panel_frame().drop("source").write_parquet(book)
    monkeypatch.setattr(
        "quant_fund.microstructure.book_panel.load_book_panel",
        lambda p: pl.read_parquet(p),
    )
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg), "--book", str(book)])
    assert result.exit_code != 0
    assert "missing source column" in _plain(result.output)


def test_kyle_ofi_vendor_remap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_kyle_backend(monkeypatch, _kyle_receipt())
    quotes = tmp_path / "quotes.parquet"
    _alpaca_quotes_frame().write_parquet(quotes)
    result = runner.invoke(
        app,
        ["kyle-ofi", "--config", str(cfg), "--book", str(quotes), "--vendor", "alpaca"],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output


def test_kyle_ofi_reads_bars_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_kyle_backend(monkeypatch, _kyle_receipt())
    bars = tmp_path / "bars.parquet"
    _synth_bars().write_parquet(bars)
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg), "--bars-path", str(bars)])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    ("bars", "fragment"),
    [
        (_synth_bars(with_volume=False), "requires bar volume"),
        (_synth_bars(volume=0.0), "positive finite bar volume"),
    ],
    ids=["no_volume_column", "zero_volume"],
)
def test_kyle_ofi_synth_book_requires_finite_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bars: pl.DataFrame, fragment: str
) -> None:
    cfg = _data_config(tmp_path)
    _patch_synthetic_provider(monkeypatch, bars)
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg)])
    assert result.exit_code != 0
    assert fragment in _plain(result.output)


def test_kyle_ofi_without_dump_skips_series_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_kyle_backend(monkeypatch, _kyle_receipt())
    _patch_synthetic_provider(monkeypatch, _synth_bars())
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "dumped_lambda_series" not in result.output


def test_kyle_ofi_dump_lambda_series(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_kyle_backend(monkeypatch, _kyle_receipt())
    _patch_synthetic_provider(monkeypatch, _synth_bars())
    dest = tmp_path / "lambda.parquet"
    result = runner.invoke(
        app, ["kyle-ofi", "--config", str(cfg), "--dump-lambda-series", str(dest)]
    )
    assert result.exit_code == 0, result.output
    assert dest.is_file()
    assert "dumped_lambda_series=" in result.output


def test_kyle_ofi_non_synthetic_book_dgp_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    receipt = _kyle_receipt(book_dgp="vendor_l2")
    receipt["book_source"] = "alpaca_feed"
    _patch_kyle_backend(monkeypatch, receipt)
    _patch_synthetic_provider(monkeypatch, _synth_bars())
    result = runner.invoke(app, ["kyle-ofi", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=synthetic" in result.output
    assert "alpaca_feed" in result.output


# --- research -------------------------------------------------------------


def _research_nb(synthetic: bool, families: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        synthetic=synthetic,
        data_source="synthetic" if synthetic else "file",
        product="Dipcatcher",
        firm="Artificial Hedge",
        version="1.0",
        disclaimer="research-only",
        hypotheses=[
            SimpleNamespace(id="h1", decision="reject", p_value=0.001),
        ],
        rankers=[
            {
                "name": "ridge",
                "mean_ic": 0.01,
                "mean_rank_ic": 0.02,
                "t_ic": 1.5,
                "decile_monotonicity": 0.3,
            },
            {
                "name": "_dm_pairs",
                "pairwise_dm_all": [
                    {
                        "a": "x",
                        "b": "y",
                        "statistic": 1.0,
                        "p_value": 0.04,
                        "preferred": "x",
                    }
                ],
            },
        ],
        families=families or {},
        artifacts={"json": "j.json", "markdown": "m.md"},
    )


def test_research_non_synthetic_empty_families(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "quant_fund.research.agent.run_research",
        lambda c, **_kw: _research_nb(synthetic=False),
    )
    result = runner.invoke(app, ["research", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=file" in result.output
    assert "DM x vs y" in result.output
    assert "ridge: IC=" in result.output
    assert "json=j.json" in result.output


def test_research_jobs_flag_reaches_run_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []

    def fake(c: object, **kw: object) -> SimpleNamespace:
        seen.append(kw.get("jobs"))
        return _research_nb(synthetic=True)

    monkeypatch.setattr("quant_fund.research.agent.run_research", fake)
    cfg = str(_data_config(tmp_path))
    assert runner.invoke(app, ["research", "--config", cfg]).exit_code == 0
    assert runner.invoke(app, ["research", "--config", cfg, "--jobs", "3"]).exit_code == 0
    rejected = runner.invoke(app, ["research", "--config", cfg, "--jobs", "0"])
    assert rejected.exit_code == 2
    assert seen == [1, 3]


def test_research_echoes_all_family_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    families: dict[str, object] = {
        "volatility": {"qlike_ewma": 1.0, "qlike_rolling": 1.1},
        "reinforcement": {
            "mean_policy_reward": 0.1,
            "mean_advantage_vs_uniform": 0.02,
            "mean_regret_vs_oracle": 0.03,
        },
        "conformal": {
            "aci": {"coverage": 0.9, "mean_width": 0.1},
            "gaussian_raw": {"coverage": 0.8},
            "wrappee": "scaled_gaussian",
            "cqr_raw": {"coverage": 0.81},
            "scaled": {"coverage": 0.9},
            "cqr": {"coverage": 0.9},
            "mondrian_aci": {
                "coverage": 0.9,
                "high_x_coverage": 0.8,
                "worst_x_coverage": 0.7,
            },
        },
        "evalues": {"coverage": 0.9, "e_final": 1.0, "ever_cross": False},
        "jackknife_plus": {"coverage": 0.9, "mean_width": 0.1, "coverage_floor": 0.8},
        "crc": {
            "wrappee": "w",
            "risk": 0.05,
            "crc_stat": 0.1,
            "lambda_hat": 0.2,
            "high_vol_mean_bound": 0.3,
            "low_vol_mean_bound": 0.1,
        },
        "weighted_conformal": {
            "coverage": 0.9,
            "mean_width": 0.1,
            "unweighted_coverage": 0.85,
        },
        "interval_risk": {"mean_cap": 0.1, "frac_binding": 0.2, "mean_width": 0.3},
        "quantile_bandit": {
            "mean_policy_reward": 0.1,
            "mean_advantage_vs_uniform": 0.01,
            "mean_regret_vs_oracle": 0.02,
        },
        "northset": {"ohlc_identity_rate": 1.0},
    }
    monkeypatch.setattr(
        "quant_fund.research.agent.run_research",
        lambda c, **_kw: _research_nb(synthetic=True, families=families),
    )
    result = runner.invoke(app, ["research", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    for token in (
        "SYNTHETIC",
        "volatility QLIKE",
        "linucb reward=",
        "conformal wrappee=",
        "evalues cov=",
        "jackknife_plus cov=",
        "crc wrappee=",
        "weighted_conformal cov=",
        "interval_risk mean_cap=",
        "quantile_bandit reward=",
        "northset ohlc_ok=",
    ):
        assert token in result.output


# --- session-book ----------------------------------------------------------


def test_session_book_daily_out_writes_aggregate(tmp_path: Path) -> None:
    cfg = _data_config(tmp_path)
    out = tmp_path / "session.parquet"
    daily = tmp_path / "daily.parquet"
    result = runner.invoke(
        app,
        [
            "session-book",
            "--config",
            str(cfg),
            "--out",
            str(out),
            "--daily-out",
            str(daily),
            "--n-session",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert daily.is_file()
    assert "session_book_daily rows=" in result.output
    assert pl.read_parquet(daily).height > 0


# --- vendor-book-map -------------------------------------------------------


def test_vendor_book_map_columns_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "vendor-book-map",
            "--vendor",
            "alpaca",
            "--columns",
            "symbol,timestamp,bid_price,ask_price,bid_size,ask_size",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "derivable_ok" in result.output


def test_vendor_book_map_rejects_unknown_vendor() -> None:
    result = runner.invoke(app, ["vendor-book-map", "--vendor", "nope"])
    assert result.exit_code != 0
    assert "vendor must be one of" in _plain(result.output)


def test_vendor_book_map_parquet_remap_writes_panel(tmp_path: Path) -> None:
    quotes = tmp_path / "quotes.parquet"
    _alpaca_quotes_frame().write_parquet(quotes)
    dest = tmp_path / "panel.parquet"
    result = runner.invoke(
        app,
        [
            "vendor-book-map",
            "--vendor",
            "alpaca",
            "--parquet",
            str(quotes),
            "--out",
            str(dest),
        ],
    )
    assert result.exit_code == 0, result.output
    assert dest.is_file()
    assert "remapped rows=" in result.output


# --- northset --------------------------------------------------------------


def _patch_northset_bench(
    monkeypatch: pytest.MonkeyPatch, bars: pl.DataFrame | None = None
) -> dict[str, pl.DataFrame]:
    captured: dict[str, pl.DataFrame] = {}
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: (
            bars
            if bars is not None
            else pl.DataFrame({"event_time": [datetime(2026, 1, 1, tzinfo=UTC)]})
        ),
    )

    def fake_bench(b: pl.DataFrame, c: object) -> dict[str, object]:
        captured["bars"] = b
        return {"join_coverage": 1.0, "book_source": "synthetic_l2"}

    monkeypatch.setattr("quant_fund.northset.benches.bench_northset", fake_bench)
    return captured


def test_northset_vendor_remap_non_synthetic_and_no_security_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path, source="file")
    quotes = tmp_path / "quotes.parquet"
    _alpaca_quotes_frame().write_parquet(quotes)
    _patch_northset_bench(monkeypatch)
    result = runner.invoke(
        app,
        [
            "northset",
            "--config",
            str(cfg),
            "--book",
            str(quotes),
            "--vendor",
            "alpaca",
            "--depth-shape-floor",
            "0.9",
            "--concentration-floor",
            "0.8",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "remapped vendor=alpaca" in result.output
    assert (tmp_path / "quotes_remapped_alpaca.parquet").is_file()
    assert "DATA_LABEL=file" in result.output
    assert "depth_shape_finite_floor=0.9" in result.output
    assert "concentration_top_finite_floor=0.8" in result.output


def test_northset_drops_benchmark_id_from_bars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    bars = pl.DataFrame(
        {
            "security_id": ["SEC_MKT", "A"],
            "event_time": [ts, ts],
            "close": [1.0, 2.0],
        }
    )
    captured = _patch_northset_bench(monkeypatch, bars=bars)
    result = runner.invoke(app, ["northset", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert captured["bars"]["security_id"].to_list() == ["A"]
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "SYNTHETIC" in result.output


def test_northset_book_panel_without_vendor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_northset_bench(monkeypatch)
    book = tmp_path / "panel.parquet"
    _book_panel_frame().write_parquet(book)
    result = runner.invoke(app, ["northset", "--config", str(cfg), "--book", str(book)])
    assert result.exit_code == 0, result.output
    assert "remapped vendor" not in result.output


# --- verify-research / lab / report / api -----------------------------------


@pytest.mark.parametrize(("valid", "code"), [(True, 0), (False, 1)])
def test_verify_research_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, valid: bool, code: int
) -> None:
    monkeypatch.setattr(
        "quant_fund.research.verify.verify_research_artifact",
        lambda p: {"valid": valid},
    )
    result = runner.invoke(app, ["verify-research", str(tmp_path / "r.json")])
    assert result.exit_code == code
    assert '"valid"' in result.output


def test_lab_alias_echoes_disclaimer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nb = SimpleNamespace(
        synthetic=True,
        disclaimer="research-only",
        artifacts={"json": "j.json"},
    )
    monkeypatch.setattr("quant_fund.research.agent.run_research", lambda c: nb)
    result = runner.invoke(app, ["lab", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "research-only" in result.output
    assert "j.json" in result.output


def test_lab_alias_non_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nb = SimpleNamespace(
        synthetic=False,
        disclaimer="research-only",
        artifacts={"json": "j.json"},
    )
    monkeypatch.setattr("quant_fund.research.agent.run_research", lambda c: nb)
    result = runner.invoke(app, ["lab", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "research-only" in result.output


def test_report_writes_latest_markdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dest_dir = tmp_path / "report_latest"
    calls: list[tuple[tuple, dict]] = []
    monkeypatch.setattr("quant_fund.reporting.report.latest_report_dir", lambda root: dest_dir)
    monkeypatch.setattr(
        "quant_fund.reporting.report.write_report",
        lambda *a, **k: calls.append((a, k)),
    )
    result = runner.invoke(app, ["report", "--latest", "--config", str(_data_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert str(dest_dir / "latest.md") in result.output
    assert calls and calls[0][1].get("synthetic") is True


def test_api_loopback_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    def fake(app_path: str, host: str, port: int, reload: bool) -> None:
        calls.update(app=app_path, host=host, port=port)

    monkeypatch.setattr("uvicorn.run", fake)
    result = runner.invoke(app, ["api"])
    assert result.exit_code == 0, result.output
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 8000


def test_api_non_loopback_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANT_API_KEY", "k")
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    result = runner.invoke(app, ["api", "--host", "0.0.0.0", "--port", "9000"])
    assert result.exit_code == 0, result.output


# --- paper ------------------------------------------------------------------


def _patch_paper_backend(
    monkeypatch: pytest.MonkeyPatch, source_note: str = "SYNTHETIC", latest: str | None = None
) -> SimpleNamespace:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    feat = pl.DataFrame(
        {
            "event_time": [
                ts,
                datetime(2026, 1, 2, tzinfo=UTC),
                datetime(2026, 1, 3, tzinfo=UTC),
            ]
        }
    )
    weights = pl.DataFrame({"event_time": [ts], "security_id": ["A"], "target_weight": [0.5]})
    monkeypatch.setattr(
        "quant_fund.pipeline.dataset.ensure_silver",
        lambda c: pl.DataFrame({"close": [1.0]}),
    )
    monkeypatch.setattr("quant_fund.features.engine.build_features", lambda b, c: feat)
    monkeypatch.setattr("quant_fund.pipeline.forecast.decision_dates", lambda c, ds: ds)
    monkeypatch.setattr(
        "quant_fund.pipeline.forecast.build_causal_weight_panel", lambda c, d: weights
    )
    monkeypatch.setattr(
        "quant_fund.paper.loop.build_scaled_challenger_weights",
        lambda w, s: [w] if s else [],
    )
    monkeypatch.setattr("quant_fund.paper.ledger.latest_run_id", lambda *a: latest)
    result = SimpleNamespace(
        source_note=source_note,
        run_id="r1",
        divergence=0.0,
        metrics={"n_steps": 1, "n_fills": 0},
        paths={},
    )
    monkeypatch.setattr("quant_fund.paper.loop.run_paper_loop", lambda *a, **k: result)
    return result


def test_paper_halt_run_echoes_simulated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_paper_backend(monkeypatch)
    result = runner.invoke(app, ["paper", "--config", str(cfg), "--halt", "--max-steps", "2"])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    assert "paper ledger is simulated" in result.output
    assert "run_id=r1" in result.output
    assert '"n_steps": 1' in result.output


def test_paper_resume_from_start_clear_halt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_paper_backend(monkeypatch, source_note="VENDOR_BARS")
    result = runner.invoke(
        app,
        [
            "paper",
            "--config",
            str(cfg),
            "--clear-halt",
            "--max-steps",
            "1",
            "--resume",
            "--run-id",
            "rid9",
            "--from-start",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=VENDOR_BARS" in result.output
    assert "paper ledger is simulated" not in result.output


def test_paper_resume_without_id_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_paper_backend(monkeypatch, latest=None)
    result = runner.invoke(app, ["paper", "--config", str(cfg), "--resume", "--max-steps", "1"])
    assert result.exit_code != 0
    assert "--resume needs --run-id" in _plain(result.output)


def test_paper_challenger_scales_and_shadow_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(
        tmp_path,
        extra="paper:\n  enable_shadow: false\n  challenger_scales: [0.5]\n",
    )
    _patch_paper_backend(monkeypatch)
    result = runner.invoke(app, ["paper", "--config", str(cfg), "--max-steps", "1"])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output


def test_paper_no_max_steps_uses_full_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_paper_backend(monkeypatch)
    result = runner.invoke(app, ["paper", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "run_id=r1" in result.output


# --- monitor ----------------------------------------------------------------


def _patch_monitor_backend(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    rid: str | None,
    broker: dict[str, object] | None,
    status: str = "ok",
) -> None:
    monkeypatch.setattr("quant_fund.paper.ledger.latest_run_id", lambda *a: rid)
    monkeypatch.setattr("quant_fund.paper.ledger.load_broker_state", lambda *a: broker)
    monkeypatch.setattr("quant_fund.paper.ledger.paper_root", lambda *a: tmp_path / "paper")
    monkeypatch.setattr(
        "quant_fund.monitoring.dashboard.ops_snapshot",
        lambda **k: {"overall_status": status},
    )
    monkeypatch.setattr("quant_fund.monitoring.dashboard.render_markdown", lambda s: "SNAP_MD")


def _broker_state() -> dict[str, object]:
    return {
        "champion": {
            "cash": 100.0,
            "last_marks": {"A": 10.0},
            "shares": {"A": 2.0},
            "open_orders": [],
            "kill_state": "ENABLED",
        },
        "mark_ages": {"A": 0},
    }


def test_monitor_requires_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_monitor_backend(monkeypatch, tmp_path, rid=None, broker=None)
    result = runner.invoke(app, ["monitor", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "no paper run found" in _plain(result.output)


def test_monitor_missing_broker_state_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_monitor_backend(monkeypatch, tmp_path, rid="rid1", broker=None)
    result = runner.invoke(app, ["monitor", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "no broker_state.json" in _plain(result.output)


@pytest.mark.parametrize(("status", "code"), [("ok", 0), ("warn", 1), ("breach", 2)])
def test_monitor_status_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, code: int
) -> None:
    cfg = _data_config(tmp_path)
    _patch_monitor_backend(monkeypatch, tmp_path, rid="rid1", broker=_broker_state(), status=status)
    eq_dir = tmp_path / "paper" / "rid1"
    eq_dir.mkdir(parents=True)
    pl.DataFrame({"asof": [datetime(2026, 1, 1, tzinfo=UTC)], "nav": [120.0]}).write_parquet(
        eq_dir / "equity.parquet"
    )
    result = runner.invoke(app, ["monitor", "--config", str(cfg)])
    assert result.exit_code == code
    assert "SNAP_MD" in result.output


def test_monitor_empty_equity_falls_back_to_broker_nav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    _patch_monitor_backend(monkeypatch, tmp_path, rid="rid1", broker=_broker_state())
    eq_dir = tmp_path / "paper" / "rid1"
    eq_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "asof": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "nav": pl.Series([], dtype=pl.Float64),
        }
    ).write_parquet(eq_dir / "equity.parquet")
    result = runner.invoke(app, ["monitor", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "SNAP_MD" in result.output


def test_monitor_nav_fallback_json_and_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_monitor_backend(monkeypatch, tmp_path, rid="rid2", broker=_broker_state())
    out = tmp_path / "snap.json"
    result = runner.invoke(
        app,
        [
            "monitor",
            "--config",
            str(cfg),
            "--run-id",
            "rid2",
            "--json",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    assert "wrote " in result.output


# --- sim-live ----------------------------------------------------------------


def _patch_sim_live(
    monkeypatch: pytest.MonkeyPatch, captured: dict[str, object] | None = None
) -> SimpleNamespace:
    result = SimpleNamespace(
        run_id="sim1",
        receipt={
            "data_label": "REAL_BARS",
            "champion_equity_stats": {"total_return": 0.01},
            "loop_metrics": {"n_fills": 3},
            "book_stats": {
                "champ": {
                    "status": "ok",
                    "total_return": 0.05,
                    "sharpe_simulated": 1.2,
                    "max_drawdown": -0.02,
                    "ann_vol": 0.10,
                    "n_fills": 4,
                },
                "chall": {"status": "no_bars"},
            },
        },
        receipt_path="receipt.json",
    )

    def fake(**kwargs: object) -> SimpleNamespace:
        if captured is not None:
            captured.update(kwargs)
        return result

    monkeypatch.setattr("quant_fund.paper.sim_live.run_sim_live", fake)
    return result


def test_sim_live_leaderboard_and_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_sim_live(monkeypatch)
    result = runner.invoke(app, ["sim-live", "--config", str(cfg), "--out", str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=REAL_BARS" in result.output
    assert "book leaderboard" in result.output
    assert "champ" in result.output
    assert "no_bars" in result.output
    assert "SIMULATED — no live-PnL claim" in result.output


def test_sim_live_git_sha_failure_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty PATH -> `git rev-parse` raises FileNotFoundError -> git_sha=None.
    monkeypatch.setenv("PATH", "/nonexistent")
    cfg = _data_config(tmp_path)
    _patch_sim_live(monkeypatch)
    result = runner.invoke(app, ["sim-live", "--config", str(cfg), "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, result.output


def test_sim_live_without_book_stats_skips_leaderboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _data_config(tmp_path)
    result_ns = SimpleNamespace(
        run_id="sim1",
        receipt={
            "data_label": "REAL_BARS",
            "champion_equity_stats": {"total_return": 0.01},
            "loop_metrics": {"n_fills": 0},
        },
        receipt_path="receipt.json",
    )
    monkeypatch.setattr("quant_fund.paper.sim_live.run_sim_live", lambda **k: result_ns)
    result = runner.invoke(app, ["sim-live", "--config", str(cfg), "--out", str(tmp_path / "o")])
    assert result.exit_code == 0, result.output
    assert "book leaderboard" not in result.output
    assert "SIMULATED — no live-PnL claim" in result.output


def test_sim_live_challenger_format_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    _patch_sim_live(monkeypatch)
    result = runner.invoke(app, ["sim-live", "--config", str(cfg), "--challenger", "bad"])
    assert result.exit_code != 0
    assert "name:spec:mode" in _plain(result.output)


def test_sim_live_challenger_kv_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _data_config(tmp_path)
    captured: dict[str, object] = {}
    _patch_sim_live(monkeypatch, captured=captured)
    challenger = (
        "ch1:empirical:symmetric:25:0.5:edge:risk"
        ":bvt=0.01:tg=0.02:pb=3:xp=2:tk=4:gross=1.2:nc=0.3:db=0.02"
        ":go=0.015:meta=0.1:rb=2:ac=0.5:cut=0.9:le=0.01:wa=0.9"
        ":bg=1.0:epow=2.0:rvol=0.03:rvlb=30:lead=BTCUSDT:bar_token"
    )
    result = runner.invoke(app, ["sim-live", "--config", str(cfg), "--challenger", challenger])
    assert result.exit_code == 0, result.output
    challengers = captured["challengers"]
    assert len(challengers) == 1
    ch = challengers[0]
    assert ch.name == "ch1"
    assert ch.spec == "empirical"
    assert ch.policy.mode == "symmetric"
    assert ch.policy.leader_sid == "BTCUSDT"
    assert ch.policy.persist_bars == 3
    assert ch.policy.top_k == 4
    assert ch.policy.w_alpha == 0.9


# --- __main__ entrypoint ------------------------------------------------------


def test_main_module_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["dipcatcher", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("quant_fund.cli.main", run_name="__main__")
    assert exc.value.code == 0
