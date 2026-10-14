"""Operator live desk: shadow, loopback rehearsal, and an armed drop folder."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.execution.order_recon import LiveEndpointRefused, refuse_live_endpoint
from quant_fund.live.desk import LiveDesk, LiveDeskError, session_id
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.pretrade.shadow import business_session_id
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

_FORBIDDEN = ("sharpe", "sortino", "calmar", "pnl", "nav")


def _config(**overrides: object) -> PretradeConfig:
    return PretradeConfig(
        schema_version=1,
        limits=limit_config(**overrides),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )


def _open(tmp_path: Path, **overrides: object) -> LiveDesk:
    root = tmp_path / "desk"
    cash = None if (root / "state.json").exists() else 1_000_000.0
    account = None if cash is None else 1_000_000.0
    return LiveDesk.open(
        root,
        hmac_key=HMAC_KEY,
        config=_config(**overrides),
        cash=cash,
        account_value=account,
        ts_ns=TS_NS,
    )


def test_session_index_matches_pretrade() -> None:
    assert session_id(date(2024, 1, 3)) == business_session_id(date(2024, 1, 3))


def test_paper_lane_still_refuses_live_endpoints() -> None:
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint("live")


def test_unfunded_desk_does_not_invent_buying_power(tmp_path: Path) -> None:
    desk = LiveDesk.open(
        tmp_path / "desk",
        hmac_key=HMAC_KEY,
        config=_config(),
        ts_ns=TS_NS,
    )
    try:
        status = desk.status()
        assert status["funded"] is False
        assert status["cash"] == 0.0
        assert status["transmission"] is False
        with pytest.raises(LiveDeskError, match="fund the desk"):
            desk.submit("aapl", "BUY", 1, 100.0, TS_NS)
        with pytest.raises(LiveDeskError, match="fund the desk"):
            desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
    finally:
        desk.close()


def test_shadow_transmits_nothing(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        report = desk.submit("aapl", "BUY", 10, 100.0, TS_NS, client_order_id="c1")
    assert report["accepted"] is True
    assert report["status"] == "shadowed"
    assert report["transmission"] is False
    assert report["live_pnl_claim"] is False
    assert report["would_promote_live"] is False
    assert report["evidence_conditions"] == "blocked"
    assert report["data_label"] == "SHADOW"
    assert report["positions"] == {}
    assert not (tmp_path / "desk" / "outbound").exists()
    for key in report:
        assert key.lower() not in _FORBIDDEN


def test_shadow_blocks_a_short_and_an_oversized_buy(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        short = desk.submit("aapl", "SELL", 1, 100.0, TS_NS)
    assert short["accepted"] is False
    assert "reg_sho" in short["order"]["reasons"]  # type: ignore[index]
    tight = LiveDesk.open(
        tmp_path / "tight",
        hmac_key=HMAC_KEY,
        config=_config(max_order_notional=100.0),
        cash=1_000_000.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )
    try:
        huge = tight.submit("msft", "BUY", 10, 100.0, TS_NS)
    finally:
        tight.close()
    assert huge["accepted"] is False
    assert "max_order_notional" in huge["order"]["reasons"]  # type: ignore[index]


def test_buying_power_and_kill_switch(tmp_path: Path) -> None:
    desk = LiveDesk.open(
        tmp_path / "desk",
        hmac_key=HMAC_KEY,
        config=_config(),
        cash=50.0,
        account_value=1_000_000.0,
        ts_ns=TS_NS,
    )
    try:
        denied = desk.submit("aapl", "BUY", 1, 100.0, TS_NS)
        assert denied["accepted"] is False
        assert "buying_power" in denied["order"]["reasons"]  # type: ignore[index]
        desk.kill("ops", "manual halt", TS_NS + 1)
        halted = desk.submit("aapl", "BUY", 1, 10.0, TS_NS + 2, client_order_id="after")
        assert halted["accepted"] is False
        assert halted["kill_latched"] is True
    finally:
        desk.close()


def test_loopback_fill_is_synthetic_and_resumes(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        desk.arm("loopback", "", TS_NS)
        report = desk.submit("aapl", "BUY", 10, 100.0, TS_NS + 1, client_order_id="fill-1")
        again = desk.submit("aapl", "BUY", 10, 100.0, TS_NS + 1, client_order_id="fill-1")
    assert report["data_label"] == "SYNTHETIC"
    assert report["status"] == "filled"
    assert report["transmission"] is True
    order = report["order"]
    assert isinstance(order, dict)
    assert order["synthetic"] is True
    assert order["filled_qty"] == 10
    assert again["idempotent"] is True
    assert report["cash"] == pytest.approx(999_000.0)
    assert report["positions"] == {"AAPL": 10}
    assert report["account_value"] == pytest.approx(1_000_000.0)
    assert report["trace_ok"] is True
    assert not (tmp_path / "desk" / "outbound").exists()

    with _open(tmp_path) as resumed:
        status = resumed.status()
    assert status["positions"] == {"AAPL": 10}
    assert status["cash"] == pytest.approx(999_000.0)
    assert status["kill_latched"] is False
    duplicate = None
    with _open(tmp_path) as resumed:
        duplicate = resumed.submit("aapl", "BUY", 10, 100.0, TS_NS + 2, client_order_id="fill-2")
    assert duplicate is not None
    assert duplicate["accepted"] is False
    assert "duplicate_order" in duplicate["order"]["reasons"]  # type: ignore[index]


def test_drop_stays_dark_until_armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIP_LIVE_ARM", raising=False)
    with _open(tmp_path) as desk:
        with pytest.raises(LiveDeskError, match="SEND-LIVE-ORDERS"):
            desk.arm("drop", "nope", TS_NS)
        assert desk.status()["venue"] == "shadow"
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        held = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 1, client_order_id="held")
    assert held["accepted"] is True
    assert held["status"] == "held"
    assert held["transmission"] is False
    assert list((tmp_path / "desk" / "outbound").glob("*.json")) == []

    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with _open(tmp_path) as desk:
        sent = desk.submit("aapl", "BUY", 4, 100.0, TS_NS + 2_000_000_000, client_order_id="sent")
        order = sent["order"]
        assert isinstance(order, dict)
        assert sent["transmission"] is True
        assert sent["status"] == "working"
        ticket = json.loads(
            (tmp_path / "desk" / "outbound" / f"{order['order_id']}.json").read_text()
        )
        assert ticket["live_pnl_claim"] is False
        assert ticket["evidence_conditions"] == "blocked"
        assert ticket["qty"] == 4
        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "fill.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": order["order_id"],
                    "event_id": "fill-1",
                    "qty": 4,
                    "price": 100,
                }
            ),
            encoding="utf-8",
        )
        applied = desk.poll(TS_NS + 2_000_000_001)
    assert applied["positions"] == {"AAPL": 4}
    assert applied["cash"] == pytest.approx(999_600.0)
    assert applied["data_label"] == "OPERATOR_BRIDGE"


def test_partial_fill_then_cancel_releases_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with _open(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 10, 50.0, TS_NS + 1, client_order_id="part")
        order = sent["order"]
        assert isinstance(order, dict)
        assert desk.status()["reserved_cash"] == pytest.approx(500.0)
        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "partial.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": order["order_id"],
                    "event_id": "p1",
                    "qty": 4,
                    "price": 50,
                }
            ),
            encoding="utf-8",
        )
        desk.poll(TS_NS + 2)
        assert desk.status()["positions"] == {"AAPL": 4}
        assert desk.status()["reserved_cash"] == pytest.approx(300.0)
        (inbound / "cancel.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "cancel",
                    "order_id": order["order_id"],
                    "event_id": "c1",
                }
            ),
            encoding="utf-8",
        )
        done = desk.poll(TS_NS + 3)
    assert done["positions"] == {"AAPL": 4}
    assert done["reserved_cash"] == 0
    assert done["cash"] == pytest.approx(999_800.0)


def test_orphan_fill_halts_and_forged_claim_refuses_to_open(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        desk.arm("loopback", "", TS_NS)
        inbound = tmp_path / "desk" / "inbound"
        inbound.mkdir()
        (inbound / "orphan.json").write_text(
            json.dumps(
                {
                    "schema": "live_broker_event.v1",
                    "kind": "fill",
                    "order_id": "missing",
                    "event_id": "o1",
                    "qty": 1,
                    "price": 10,
                }
            ),
            encoding="utf-8",
        )
        report = desk.poll(TS_NS + 1)
        assert report["kill_latched"] is True
        assert report["orphan_count"] == 1
    forged = {
        "schema": "live_arming.v1",
        "venue": "drop",
        "capital": "operator_bridge",
        "live_pnl_claim": True,
        "would_promote_live": False,
        "evidence_conditions": "blocked",
    }
    (tmp_path / "desk" / "arming.json").write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(Exception, match="live_pnl_claim"):
        _open(tmp_path)


def test_second_open_is_locked_until_close(tmp_path: Path) -> None:
    first = _open(tmp_path)
    try:
        with pytest.raises(LiveDeskError, match="locked"):
            _open(tmp_path)
    finally:
        first.close()
    with _open(tmp_path) as second:
        assert second.status()["venue"] == "shadow"


def test_rate_limit_survives_reopen(tmp_path: Path) -> None:
    with _open(tmp_path, max_orders_per_window=1) as desk:
        first = desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="one")
        assert first["accepted"] is True
    with _open(tmp_path, max_orders_per_window=1) as desk:
        second = desk.submit("msft", "BUY", 1, 100.0, TS_NS + 1, client_order_id="two")
    assert second["accepted"] is False
    assert "order_rate" in second["order"]["reasons"]  # type: ignore[index]


def test_drop_fill_survives_a_new_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with _open(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 3, 100.0, TS_NS + 1, client_order_id="cross")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
    inbound = tmp_path / "desk" / "inbound"
    inbound.mkdir()
    (inbound / "fill.json").write_text(
        json.dumps(
            {
                "schema": "live_broker_event.v1",
                "kind": "fill",
                "order_id": order_id,
                "event_id": "later",
                "qty": 3,
                "price": 100,
            }
        ),
        encoding="utf-8",
    )
    with _open(tmp_path) as desk:
        report = desk.poll(TS_NS + 2)
    assert report["positions"] == {"AAPL": 3}
    assert report["cash"] == pytest.approx(999_700.0)
    assert report["trace_ok"] is True


def test_cancel_of_a_sent_ticket_requires_the_arm_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DIP_LIVE_ARM", "1")
    with _open(tmp_path) as desk:
        desk.arm("drop", "SEND-LIVE-ORDERS", TS_NS)
        sent = desk.submit("aapl", "BUY", 2, 100.0, TS_NS + 1, client_order_id="work")
        order = sent["order"]
        assert isinstance(order, dict)
        order_id = str(order["order_id"])
    monkeypatch.delenv("DIP_LIVE_ARM", raising=False)
    with _open(tmp_path) as desk:
        refused = desk.cancel(order_id, TS_NS + 2)
        assert refused["accepted"] is False
        assert desk.status()["reserved_cash"] == pytest.approx(200.0)
        assert desk.status()["positions"] == {}


def test_tampered_journal_freezes_the_desk(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="j")
    journal = tmp_path / "desk" / "journal.jsonl"
    text = journal.read_text(encoding="utf-8")
    journal.write_text(text.replace("shadowed", "filled"), encoding="utf-8")
    with _open(tmp_path) as desk:
        status = desk.status()
        assert status["kill_latched"] is True
        assert status["transmission"] is False
        with pytest.raises(LiveDeskError, match="journal"):
            desk.submit("msft", "BUY", 1, 100.0, TS_NS + 2_000_000_000, client_order_id="next")


def test_cli_loopback_and_unarmed_drop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIP_LIVE_ARM", raising=False)
    root = tmp_path / "cli"
    runner = CliRunner()
    armed = runner.invoke(
        app,
        [
            "live",
            "arm",
            "--root",
            str(root),
            "--venue",
            "loopback",
            "--ts-ns",
            str(TS_NS),
            "--cash",
            "1000000",
            "--account",
            "1000000",
        ],
    )
    assert armed.exit_code == 0, armed.stdout
    submitted = runner.invoke(
        app,
        [
            "live",
            "submit",
            "--root",
            str(root),
            "--symbol",
            "aapl",
            "--side",
            "buy",
            "--qty",
            "2",
            "--mark",
            "100",
            "--ts-ns",
            str(TS_NS + 1),
            "--client-id",
            "cli-1",
        ],
    )
    assert submitted.exit_code == 0, submitted.stdout
    body = json.loads(submitted.stdout)
    assert body["data_label"] == "SYNTHETIC"
    assert body["status"] == "filled"
    assert body["live_pnl_claim"] is False
    refused = runner.invoke(
        app,
        [
            "live",
            "arm",
            "--root",
            str(root),
            "--venue",
            "drop",
            "--confirm",
            "no",
            "--ts-ns",
            str(TS_NS + 2),
        ],
    )
    assert refused.exit_code == 1
    assert "SEND-LIVE-ORDERS" in refused.stdout
