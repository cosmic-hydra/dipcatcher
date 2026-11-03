"""LiveDesk event retry and partial-fill accounting regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.live.bridge import BrokerEvent
from quant_fund.live.desk import DeskOrder, LiveDesk
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config


def _config() -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open(tmp_path: Path) -> LiveDesk:
    return LiveDesk.open(
        tmp_path / "desk",
        hmac_key=HMAC_KEY,
        config=_config(),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )


def _working_order(desk: LiveDesk, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
    sent = desk.submit("AAPL", "BUY", 10, 50.0, TS_NS + 1, client_order_id="lifecycle")
    order = sent["order"]
    assert isinstance(order, dict)
    return str(order["order_id"])


def _deliver(desk: LiveDesk, monkeypatch: pytest.MonkeyPatch, event: BrokerEvent) -> None:
    monkeypatch.setattr(desk, "_events", lambda: [event])


def test_failed_fill_is_not_marked_seen_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _open(tmp_path) as desk:
        order_id = _working_order(desk, monkeypatch)
        event = BrokerEvent(kind="fill", order_id=order_id, event_id="partial", qty=4, price=50)
        _deliver(desk, monkeypatch, event)
        original_release = desk._release
        failed = False

        def fail_once(symbol: str, side: str, qty: float, px: float) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected release failure")
            original_release(symbol, side, qty, px)

        monkeypatch.setattr(desk, "_release", fail_once)
        with pytest.raises(OSError, match="injected release failure"):
            desk.poll(TS_NS + 2)

        assert event.event_id not in desk.seen
        assert desk.status()["positions"] == {}
        assert desk.status()["cash"] == 1_000_000.0
        assert desk.status()["reserved_cash"] == 500.0

        monkeypatch.setattr(desk, "_release", original_release)
        report = desk.poll(TS_NS + 3)

        assert report["applied"] == [{"event_id": "partial", "result": "fill", "synthetic": False}]
        assert report["positions"] == {"AAPL": 4.0}
        assert report["cash"] == pytest.approx(999_800.0)
        assert report["reserved_cash"] == pytest.approx(300.0)
        assert desk.orders[order_id].filled_qty == 4.0
        assert event.event_id in desk.seen


@pytest.mark.parametrize("terminal_kind", ["cancel", "reject"])
def test_partial_fill_terminal_retry_preserves_executed_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_kind: str
) -> None:
    with _open(tmp_path) as desk:
        order_id = _working_order(desk, monkeypatch)
        partial = BrokerEvent(
            kind="fill", order_id=order_id, event_id="executed", qty=4, price=45, fee=2
        )
        _deliver(desk, monkeypatch, partial)
        desk.poll(TS_NS + 2)
        assert desk.status()["positions"] == {"AAPL": 4.0}
        assert desk.status()["cash"] == pytest.approx(999_818.0)
        assert desk.status()["reserved_cash"] == pytest.approx(300.0)

        terminal = BrokerEvent(kind=terminal_kind, order_id=order_id, event_id=terminal_kind)
        _deliver(desk, monkeypatch, terminal)
        original_release = desk._release_remaining
        failed = False

        def fail_once(order: DeskOrder) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected terminal failure")
            original_release(order)

        monkeypatch.setattr(desk, "_release_remaining", fail_once)
        with pytest.raises(OSError, match="injected terminal failure"):
            desk.poll(TS_NS + 3)

        assert terminal.event_id not in desk.seen
        assert desk.status()["positions"] == {"AAPL": 4.0}
        assert desk.status()["cash"] == pytest.approx(999_818.0)
        assert desk.status()["reserved_cash"] == pytest.approx(300.0)

        monkeypatch.setattr(desk, "_release_remaining", original_release)
        report = desk.poll(TS_NS + 4)

        assert report["applied"] == [{"event_id": terminal_kind, "result": terminal_kind}]
        assert report["positions"] == {"AAPL": 4.0}
        assert report["cash"] == pytest.approx(999_818.0)
        assert report["reserved_cash"] == 0.0
        assert desk.orders[order_id].filled_qty == 4.0
        assert (
            desk.orders[order_id].status
            == {"cancel": "canceled", "reject": "rejected"}[terminal_kind]
        )
        assert report["trace_ok"] is True


@pytest.mark.parametrize(
    ("qty", "price", "fee"),
    [
        pytest.param(0.0, 50.0, 0.0, id="zero-qty"),
        pytest.param(-1.0, 50.0, 0.0, id="negative-qty"),
        pytest.param(float("nan"), 50.0, 0.0, id="nan-qty"),
        pytest.param(1e-9, 50.0, 0.0, id="below-lifecycle-epsilon"),
        pytest.param(2.0, 0.0, 0.0, id="zero-price"),
        pytest.param(2.0, float("nan"), 0.0, id="nan-price"),
        pytest.param(2.0, 50.0, -1.0, id="negative-fee"),
        pytest.param(2.0, 50.0, float("inf"), id="infinite-fee"),
    ],
)
def test_malformed_fill_cannot_change_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qty: float,
    price: float,
    fee: float,
) -> None:
    with _open(tmp_path) as desk:
        order_id = _working_order(desk, monkeypatch)
        event = BrokerEvent(
            kind="fill", order_id=order_id, event_id="malformed", qty=qty, price=price, fee=fee
        )
        _deliver(desk, monkeypatch, event)
        report = desk.poll(TS_NS + 2)

        assert report["applied"] == [{"event_id": "malformed", "result": "unapplied"}]
        assert report["positions"] == {}
        assert report["cash"] == 1_000_000.0
        assert report["reserved_cash"] == 500.0
        assert report["kill_latched"] is True
        assert desk.orders[order_id].filled_qty == 0.0
        assert desk.orders[order_id].lifecycle == "new"


def test_over_limit_fill_cannot_change_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _open(tmp_path) as desk:
        order_id = _working_order(desk, monkeypatch)
        event = BrokerEvent(kind="fill", order_id=order_id, event_id="overfill", qty=11, price=50)
        _deliver(desk, monkeypatch, event)
        report = desk.poll(TS_NS + 2)

        assert report["applied"] == [{"event_id": "overfill", "result": "unapplied"}]
        assert report["positions"] == {}
        assert report["cash"] == 1_000_000.0
        assert report["reserved_cash"] == 500.0
        assert report["kill_latched"] is True
        assert desk.orders[order_id].filled_qty == 0.0
        assert desk.orders[order_id].lifecycle == "new"
