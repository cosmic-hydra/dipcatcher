"""SYNTHETIC edges for fx1 flash, forecast, selftest, and audit fall-throughs.

Each case checks a fail-closed result or an identity the suite had not executed.
No network and no sleeps beyond the selftest server the production smoke already boots.
"""

from __future__ import annotations

import math
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from fx1.extensions.contracts import FeatureExtension
from fx1.flash.refresh import refresh_entry
from fx1.flash.retrieve import context_block, retrieve, score_entry
from fx1.flash.store import (
    FlashEntry,
    FlashSource,
    FlashStore,
    add,
    default_store,
    default_store_path,
    get,
    list_entries,
    remove,
    update,
)
from fx1.forecast.config import Fx1HarnessConfig
from fx1.forecast.dummy import MomentumForecastModel, ZeroForecastModel
from fx1.forecast.evaluate import (
    _decision_times,
    _mae_rmse,
    _signal_diagnostics,
    evaluate_forecasts,
    json_ready,
)
from fx1.forecast.runner import (
    _at_decision,
    _coerce_frame,
    _match_panel,
    _predict_batch,
    _predict_walk_forward,
    data_label,
    load_bars,
    resolve_pipeline,
    resolve_provider,
    run_inference,
)
from fx1.forecast.runner import (
    _decision_times as runner_decision_times,
)
from fx1.forecast.schema import SchemaError
from fx1.harness import HarnessResult
from quant_fund.schemas.errors import LeakageError

_STAMP = datetime(2020, 1, 1, 16, tzinfo=UTC)


def _config(tmp_path: Path, **overrides: object) -> Fx1HarnessConfig:
    payload: dict[str, object] = {
        "schema": "fx1.harness.config/v1",
        "data": {
            "provider": "entrypoint",
            "entrypoint": "x:Y",
            "symbols": ["AAA"],
        },
        "features": {"lookbacks": [1], "vol_window": 2, "horizon_bars": 1},
        "model": {"name": "dummy-zero"},
        "inference": {
            "mode": "walk_forward",
            "output_parquet": str(tmp_path / "forecasts.parquet"),
            "output_meta": str(tmp_path / "forecasts.meta.json"),
        },
        "signal": {"mapping": "sign", "threshold": 0.0, "cost_bps": 1.0},
        "walk_forward": {
            "scheme": "expanding",
            "train_bars": 1,
            "val_bars": 1,
            "test_bars": 1,
            "embargo_bars": 0,
        },
    }
    for key, value in overrides.items():
        section = payload[key]
        assert isinstance(section, dict)
        section.update(value)  # type: ignore[arg-type]
    return Fx1HarnessConfig.model_validate(payload)


def _bars(n: int = 4, *, late: bool = False) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for i in range(n):
        event = _STAMP + timedelta(days=i)
        available = event + timedelta(days=1 if late and i == 0 else 0)
        rows.append(
            {
                "security_id": "AAA",
                "event_time": event,
                "available_time": available,
                "open": 10.0 + i,
                "high": 11.0 + i,
                "low": 9.0 + i,
                "close": 10.0 + i,
                "volume": 100.0 + i,
                "source": "fixture",
            }
        )
    return pl.DataFrame(rows)


class _Provider:
    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame

    def get_bars(
        self,
        start: object = None,
        end: object = None,
        security_ids: object = None,
    ) -> pl.DataFrame:
        return self.frame


class _Pipe:
    def feature_columns(self) -> list[str]:
        return ["ret_1"]

    def build(self, bars: pl.DataFrame, decision_time: datetime | None = None) -> pl.DataFrame:
        del decision_time
        return bars.select("event_time", "security_id", "close").with_columns(
            pl.lit(0.1).alias("ret_1")
        )


class _Model:
    name = "synthetic"
    version: object = 7

    def load(self, checkpoint_path: object = None, config: object = None) -> None:
        del checkpoint_path, config

    def predict(self, features: pl.DataFrame) -> pl.DataFrame:
        n = features.height
        return pl.DataFrame(
            {
                "event_time": features["event_time"],
                "security_id": features["security_id"],
                "horizon_bars": pl.Series([1] * n, dtype=pl.Int64),
                "predicted_return": pl.Series([0.1] * n, dtype=pl.Float64),
            }
        )


def _forecast_row(
    stamp: datetime, horizon: int = 1, *, close: float | None = None
) -> dict[str, object]:
    row: dict[str, object] = {
        "event_time": stamp,
        "security_id": "AAA",
        "horizon_bars": horizon,
        "predicted_return": 0.1,
    }
    if close is not None:
        row["close"] = close
    return row


def test_flash_store_rejects_non_positive_staleness_and_bad_lines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stale_after_s must be positive"):
        FlashEntry(text="finding", stale_after_s=0)

    missing = FlashStore(tmp_path / "absent.jsonl")
    assert missing.all() == []
    assert missing.update("missing", {"text": "revision"}) is None

    store = FlashStore(tmp_path / "entries.jsonl")
    kept = store.add(text="inflation print", task="macro", tags=["cpi"])
    with store.path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write("1\n")
        handle.write('{"__tombstone__": true, "id": 5}\n')
        handle.write('{"note": "no id"}\n')
        handle.write('{"id": "bad", "text": "  ", "schema_version": "fx1.flash-entry/v1"}\n')
    assert [entry.id for entry in store.all()] == [kept.id]
    assert any("not an object" in error for error in store.read_errors())
    assert store.list(query="inflation")


def test_flash_store_without_directory_fds_still_roundtrips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fx1.flash.store as store_mod

    monkeypatch.setattr(store_mod, "_SUPPORTS_DIRECTORY_FD", False)
    store = FlashStore(tmp_path / "entries.jsonl")
    entry = store.add(text="directory fd fallback")
    assert store.get(entry.id) is not None
    assert store_mod._nofollow_stat(store.path, store_mod._NO_DIRECTORY_FD) is not None


def test_flash_store_refuses_a_file_that_is_not_regular_after_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_fstat = os.fstat

    def _directory(fd: int) -> os.stat_result:
        current = real_fstat(fd)
        mode = stat.S_IFDIR | 0o700
        return os.stat_result((mode, *tuple(current)[1:]))

    monkeypatch.setattr(os, "fstat", _directory)
    store = FlashStore(tmp_path / "entries.jsonl")
    with pytest.raises(ValueError, match="regular file"):
        store.add(text="secret")


def test_flash_module_helpers_delegate_to_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fx1.flash.store as store_mod

    monkeypatch.setattr(store_mod, "config_dir", lambda: tmp_path)
    assert default_store_path() == tmp_path / "flash" / "entries.jsonl"
    store = default_store()
    entry = add(store, text="helper finding", task="macro")
    assert get(store, entry.id) is not None
    assert list_entries(store, task="macro")
    revised = update(store, entry.id, {"task": "rates"})
    assert revised is not None and revised.task == "rates"
    assert remove(store, entry.id) is True


def test_retrieval_edges_fail_closed_and_render_honesty_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = FlashEntry(
        text="inflation print revised",
        task="macro desk",
        tags=["cpi"],
        sources=[FlashSource(url="https://example.test/cpi", title="")],
        conflicts=["vintage disagrees"],
        private=True,
        verified=False,
        evidence_at="2020-01-01T00:00:00",
        created_at="",
        refreshed_at="",
    )
    score, reasons = score_entry(entry, "   ")
    assert score == 0.0 and reasons == ["empty-query"]
    _score, flagged = score_entry(entry, "inflation", now=1_700_000_000.0)
    assert "has-conflicts" in flagged and "private" in flagged

    monkeypatch.setattr(math, "sqrt", lambda _value: 0.0)
    zero, _reasons = score_entry(entry, "inflation", now=1_700_000_000.0)
    assert zero == 0.0

    store = FlashStore(tmp_path / "entries.jsonl")
    store.add(text="inflation print", tags=["cpi"], task="macro")
    with pytest.raises(ValueError, match="k must be positive"):
        retrieve(store, "inflation", k=0)

    from fx1.flash.retrieve import Retrieval

    hit_entry = FlashEntry(
        text="inflation print",
        task="macro",
        tags=["cpi"],
        private=True,
        conflicts=["a"],
    )
    rendered = context_block(
        [Retrieval(entry=hit_entry, score=0.5, reasons=["term-match"])],
        max_chars=10,
    )
    assert "Flash context" in rendered
    assert "PRIVATE" not in rendered

    untagged = FlashEntry(text="plain finding", task="", tags=[])
    full = context_block(
        [
            Retrieval(entry=untagged, score=0.4, reasons=[]),
            Retrieval(entry=hit_entry, score=0.5, reasons=["term-match"]),
        ]
    )
    assert "PRIVATE" in full
    assert "conflicting evidence" in full
    assert "(untagged)" in full


def test_refresh_skips_blank_urls_and_reports_fetcher_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = FlashStore(tmp_path / "entries.jsonl")
    blank = store.add(text="unsourced", sources=[{"url": "", "title": "none"}])
    skipped = refresh_entry(store, blank, fetch_fn=lambda _url: None)
    assert skipped.note == "no sources recorded; nothing to refresh"

    sourced = store.add(text="sourced", sources=[{"url": "https://example.test/a"}])

    def _boom(_url: str) -> object:
        raise TimeoutError("synthetic")

    failed = refresh_entry(store, sourced, fetch_fn=_boom)
    assert "TimeoutError" in failed.unreachable[0]

    monkeypatch.setattr(
        "fx1.webresearch.fetch.fetch_page",
        lambda _url: SimpleNamespace(status=200, error=None),
    )
    reached = refresh_entry(store, sourced)
    assert reached.all_reachable is True


def test_doctor_audit_restores_keys_that_were_already_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.doctor_audit import doctor_audit

    monkeypatch.setenv("MOONSHOT_API_KEY", "already-set")
    monkeypatch.setenv("FX1_SIGNING_KEY", "already-set")
    report = doctor_audit()
    assert report["sentinel_absent"] is True
    assert report["flags_only"] is True


def test_mrm_audit_records_when_a_ghost_artifact_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fx1.mrm_audit as audit

    real = audit._compile

    def _wrapped(card: Path, art: dict[str, str | Path], root: Path) -> object:
        if str(art.get("development", "")).endswith("ghost.json"):
            return object()
        return real(card, art, root)

    monkeypatch.setattr(audit, "_compile", _wrapped)
    report = audit.mrm_audit()
    assert report["missing_artifact_raises"] == "no-raise"
    assert report["incomplete_vetoes_ship"] is True


def test_receipt_audit_records_when_malformed_json_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fx1.train.receipts as receipts
    from fx1.train.receipts_audit import train_receipt_audit

    real = receipts.verify_training_receipt

    def _wrapped(path: Path, **kwargs: object) -> bool:
        if Path(path).name == "bad.json":
            return False
        return real(path, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(receipts, "verify_training_receipt", _wrapped)
    report = train_receipt_audit()
    assert report["malformed_raises"] == "no-raise"
    assert report["forged_research_fails"] is True


def test_feature_build_without_an_injected_harness_uses_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Harness:
        def run(self, name: str, config: Path | None = None) -> HarnessResult:
            seen["name"] = name
            seen["config"] = config
            return HarnessResult(command=name, exit_code=0, stdout="SYNTHETIC", stderr="")

    monkeypatch.setattr("fx1.harness.Harness", _Harness)
    from fx1.capabilities import owner_references

    extension = FeatureExtension(
        kind="feature",
        owner="ret_1",
        references=list(owner_references("feature", "ret_1")[:1]),
        module="fx1.extensions.features.ret_1",
    )
    result = extension.build()
    assert result.ok is True
    assert seen == {"name": "build-features", "config": None}


def test_reference_models_apply_version_and_refuse_a_bad_horizon(tmp_path: Path) -> None:
    model = ZeroForecastModel()
    model.load(config={})
    model.load(config={"version": "from-config"})
    assert model.version == "from-config"

    path = tmp_path / "checkpoint.json"
    path.write_text('{"version": "artifact-9", "horizon_bars": 0}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="horizon_bars must be >= 1"):
        model.load(path, config={"checkpoint_format": "json"})
    path.write_text('{"version": "artifact-9"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="horizon_bars must be >= 1"):
        model.load(path, config={"horizon_bars": 0, "checkpoint_format": "json"})

    momentum = MomentumForecastModel()
    frame = pl.DataFrame(
        {
            "event_time": [_STAMP],
            "security_id": ["AAA"],
            "close": [10.0],
        }
    )
    with pytest.raises(ValueError, match="ret_1 or a mom_"):
        momentum.predict(frame)
    trailed = frame.with_columns(pl.lit(0.25).alias("mom_5"))
    predicted = momentum.predict(trailed)
    assert predicted["predicted_return"].to_list() == [0.25]


def test_provider_and_pipeline_entrypoints_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import quant_fund.data.adapters.parquet as parquet_mod

    monkeypatch.setattr(
        parquet_mod, "ParquetMarketProvider", lambda root: SimpleNamespace(root=root)
    )
    parquet_cfg = _config(
        tmp_path,
        data={"provider": "parquet", "root": str(tmp_path), "entrypoint": None, "symbols": []},
    )
    assert resolve_provider(parquet_cfg).root == tmp_path

    blank = _config(tmp_path)
    object.__setattr__(blank.data, "entrypoint", None)
    with pytest.raises(ValueError, match="entrypoint is required"):
        resolve_provider(blank)

    class _NoBars:
        pass

    class _Bars:
        def get_bars(self, **_kwargs: object) -> pl.DataFrame:
            return pl.DataFrame()

    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: _NoBars)
    with pytest.raises(TypeError, match="get_bars"):
        resolve_provider(_config(tmp_path))
    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: _NoBars())
    with pytest.raises(TypeError, match="get_bars"):
        resolve_provider(_config(tmp_path))
    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: _Bars)
    assert resolve_provider(_config(tmp_path)).get_bars().is_empty()
    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: _Bars())
    assert resolve_provider(_config(tmp_path)).get_bars().is_empty()

    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: object())
    with pytest.raises(TypeError, match="build"):
        resolve_pipeline(_config(tmp_path, features={"entrypoint": "x:Y"}))


def test_load_bars_resamples_and_refuses_an_empty_symbol_filter(tmp_path: Path) -> None:
    frame = _bars()
    resampled = _config(tmp_path, data={"resample": "1d", "symbols": []})
    loaded = load_bars(_Provider(frame), resampled)
    assert loaded.height >= 1

    empty = _config(tmp_path, data={"symbols": ["ZZZ"], "resample": None})
    with pytest.raises(ValueError, match="no bars remain"):
        load_bars(_Provider(frame), empty)

    mixed = pl.DataFrame({"source": ["live", "fixture"], "event_time": [_STAMP, _STAMP]})
    assert data_label(mixed, "entrypoint") == "mixed"
    vacant = pl.DataFrame({"source": pl.Series([], dtype=pl.Utf8)})
    assert data_label(vacant, "parquet") == "parquet"


def test_runner_decision_and_frame_edges(tmp_path: Path) -> None:
    pandas = pytest.importorskip("pandas")
    later = _STAMP + timedelta(days=3)
    bounded = _config(tmp_path, data={"end": _STAMP.isoformat(), "symbols": []})
    chosen = runner_decision_times(
        pl.DataFrame({"event_time": [_STAMP, later]}),
        bounded,
    )
    assert chosen == [_STAMP]

    strings = pl.DataFrame({"event_time": ["not-a-datetime"]})
    with pytest.raises(SchemaError, match="datetimes"):
        runner_decision_times(strings, _config(tmp_path, data={"symbols": []}))

    pandas_frame = pandas.DataFrame(
        {"event_time": [_STAMP], "security_id": ["AAA"], "close": [10.0]}
    )
    coerced = _coerce_frame(pandas_frame)
    assert isinstance(coerced, pl.DataFrame)

    features = _Pipe().build(_bars(1))
    prediction = _Model().predict(features)
    assert _at_decision(prediction, features, _STAMP + timedelta(days=9)).is_empty()

    future = prediction.with_columns(pl.lit(_STAMP + timedelta(days=9)).alias("event_time"))
    with pytest.raises(LeakageError, match="after the latest feature"):
        _match_panel(future, features)


def test_walk_forward_skips_an_empty_slice_and_refuses_future_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _config(tmp_path, data={"symbols": []})
    calls = {"n": 0}
    real = _at_decision

    def _skip_first(pred: pl.DataFrame, feats: pl.DataFrame, stamp: datetime) -> pl.DataFrame:
        calls["n"] += 1
        if calls["n"] == 1:
            return real(pred, feats, stamp).head(0)
        return real(pred, feats, stamp)

    monkeypatch.setattr("fx1.forecast.runner._at_decision", _skip_first)
    decisions = [_STAMP + timedelta(days=offset) for offset in range(3)]
    forecasts = _predict_walk_forward(_bars(3), _Pipe(), _Model(), decisions, cfg)
    assert forecasts.height >= 1
    assert calls["n"] >= 2

    class _Future(_Pipe):
        def build(self, bars: pl.DataFrame, decision_time: datetime | None = None) -> pl.DataFrame:
            stamp = decision_time or _STAMP
            base = super().build(bars, decision_time=stamp)
            extra = base.head(1).with_columns(pl.lit(stamp + timedelta(days=2)).alias("event_time"))
            return pl.concat([base.head(1), extra])

    with pytest.raises(LeakageError, match="extends past the decision"):
        _predict_walk_forward(_bars(2, late=True), _Future(), _Model(), [_STAMP], cfg)

    class _Empty(_Pipe):
        def build(self, bars: pl.DataFrame, decision_time: datetime | None = None) -> pl.DataFrame:
            del bars, decision_time
            return pl.DataFrame(
                schema={
                    "event_time": pl.Datetime(time_zone="UTC"),
                    "security_id": pl.Utf8,
                    "close": pl.Float64,
                    "ret_1": pl.Float64,
                }
            )

    with pytest.raises(ValueError, match="produced no forecasts"):
        _predict_walk_forward(_bars(2), _Empty(), _Model(), [_STAMP], cfg)


def test_batch_inference_refuses_a_window_with_no_features(tmp_path: Path) -> None:
    cfg = _config(tmp_path, data={"start": "2099-01-01T00:00:00+00:00", "symbols": []})
    with pytest.raises(ValueError, match="empty feature frame"):
        _predict_batch(_bars(3), _Pipe(), _Model(), cfg)


def test_inference_leaves_a_non_string_model_version_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fx1.forecast.runner.load_symbol", lambda _spec: _Pipe())
    result = run_inference(
        _config(tmp_path, features={"entrypoint": "x:Pipe"}, data={"symbols": []}),
        model=_Model(),
        provider=_Provider(_bars(4)),
    )
    assert result.metadata["model_version"] is None
    assert result.n_rows >= 1


def test_evaluation_edges_cover_empty_scores_and_a_lost_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    moment = datetime(2024, 5, 1, tzinfo=UTC)
    assert json_ready(moment) == moment.isoformat()
    with pytest.raises(SchemaError, match="datetimes"):
        _decision_times(pl.DataFrame({"event_time": ["nope"]}))
    mae, rmse, n = _mae_rmse(np.array([np.nan]), np.array([np.nan]))
    assert n == 0 and np.isnan(mae) and np.isnan(rmse)

    signal = pl.DataFrame(
        {
            "event_time": [_STAMP, _STAMP + timedelta(days=1)],
            "security_id": ["AAA", "AAA"],
            "signal": [1.0, -1.0],
        }
    )
    diagnostic = _signal_diagnostics(signal, cost_bps=1.0, stride=2)
    assert diagnostic["return_stride_bars"] == 2

    def _rows(days: int, horizon: int, *, close: float | None = None) -> list[dict[str, object]]:
        return [
            _forecast_row(_STAMP + timedelta(days=i), horizon, close=close) for i in range(days)
        ]

    bars = _bars(6)
    cfg = _config(
        tmp_path,
        data={"symbols": []},
        features={"horizon_bars": 1},
        walk_forward={"train_bars": 1, "val_bars": 1, "test_bars": 1, "embargo_bars": 0},
    )
    forecasts = pl.DataFrame(_rows(3, 1, close=10.0) + _rows(3, 2))
    report = evaluate_forecasts(forecasts, bars, cfg)
    assert "1" in report["forecast_metrics"]["by_horizon"]

    disjoint = bars.with_columns(pl.lit("ZZZ").alias("security_id"))
    with pytest.raises(SchemaError, match="no realized test-fold"):
        evaluate_forecasts(pl.DataFrame(_rows(3, 1)), disjoint, cfg)

    from fx1.forecast import signals as signals_mod

    real_map = signals_mod.map_signals

    def _drop_marker(frame: pl.DataFrame, mapping: str, threshold: float = 0.0) -> pl.DataFrame:
        mapped = real_map(frame, mapping, threshold=threshold)
        return mapped.with_columns(pl.lit("rewritten").alias("mapping_role"))

    monkeypatch.setattr(signals_mod, "map_signals", _drop_marker)
    monkeypatch.setattr("fx1.forecast.evaluate.map_signals", _drop_marker)
    with pytest.raises(SchemaError, match="placeholder marker"):
        evaluate_forecasts(forecasts, bars, cfg)


def test_wait_job_times_out_without_a_terminal_status() -> None:
    from fx1.selftest import _wait_job

    class _Queued:
        def job_status(self, _job_id: str) -> dict[str, str]:
            return {"status": "queued"}

    assert _wait_job(_Queued(), "job", timeout_s=0.0) == {"status": "timeout"}
    assert _wait_job(_Queued(), "job", timeout_s=0.06) == {"status": "timeout"}


def test_remote_selftest_counts_each_auth_failure_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.selftest import SelftestReport, _remote_checks
    from fx1.serve.client import HarnessAuthError
    from fx1.serve.contract import API_VERSION

    class _Outer:
        def health(self) -> dict[str, bool]:
            return {"ok": True}

        def server_version(self) -> dict[str, str]:
            return {"api_version": API_VERSION}

        def commands(self) -> list[str]:
            return ["doctor"]

        def score(self, _text: str) -> list[dict[str, int]]:
            return [{"total": -1}]

        def check_text(self, _text: str) -> SimpleNamespace:
            return SimpleNamespace(ok=False)

    mode = {"kind": "auth"}

    class _Bare:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def complete(self, *_args: object, **_kwargs: object) -> str:
            if mode["kind"] == "auth":
                raise HarnessAuthError("missing")
            if mode["kind"] == "boom":
                raise RuntimeError("synthetic")
            return "ok"

    monkeypatch.setattr("fx1.serve.client.HarnessClient", _Bare)
    for kind in ("auth", "boom", "ok"):
        mode["kind"] = kind
        report = SelftestReport(mode="remote")
        _remote_checks(report, _Outer(), "http://127.0.0.1:9", "probe-key")
        auth = next(
            check for check in report.checks if check.name == "auth_gate_rejects_missing_key"
        )
        assert auth.ok is (kind == "auth")


@pytest.mark.parametrize("mode", ["raise", "return"])
def test_local_selftest_records_defensive_client_arms(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The local smoke's except/else arms only run when the client misbehaves."""
    monkeypatch.setenv("FX1_API_KEY", "preset-key")
    from fx1.selftest import run_selftest
    from fx1.serve.client import HarnessClient

    real_complete = HarnessClient.complete
    real_run = HarnessClient.run
    state = {"runs": 0}

    def complete(self: HarnessClient, messages: list[dict[str, str]], **kwargs: object) -> object:
        backend = kwargs.pop("backend", "local_fx1")
        if backend == "bogus" or "X-API-Key" not in self._headers:
            if mode == "raise":
                raise RuntimeError("synthetic")
            return SimpleNamespace()
        return real_complete(self, messages, backend=str(backend), **kwargs)  # type: ignore[arg-type]

    def run(self: HarnessClient, *args: object, **kwargs: object) -> object:
        state["runs"] += 1
        if state["runs"] >= 2:
            if mode == "raise":
                raise RuntimeError("synthetic")
            return SimpleNamespace(exit_code=0)
        return real_run(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(HarnessClient, "complete", complete)
    monkeypatch.setattr(HarnessClient, "run", run)
    report = run_selftest()
    assert os.environ["FX1_API_KEY"] == "preset-key"
    by_name = {check.name: check for check in report.checks}
    assert by_name["auth_gate_rejects_missing_key"].ok is False
    assert by_name["error_envelope_maps_valueerror"].ok is False
    assert by_name["drained_rejects_work"].ok is False
    assert by_name["complete_byok"].ok is True
