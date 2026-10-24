"""Armed order gateway: pre-trade deny, loopback fill, kill, resume, Alpaca host."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
import yaml
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.execution.order_recon import LiveEndpointRefused, refuse_live_endpoint
from quant_fund.live.arming import LIVE_CAPITAL_PHRASE
from quant_fund.live.brokers import (
    AlpacaBroker,
    BrokerAccount,
    BrokerAck,
    BrokerFill,
    FillPage,
    LiveOrder,
    LoopbackBroker,
    map_broker_status,
)
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import LiveBrokerRefused, LiveGateError
from quant_fund.live.ledger import LedgerFill, LiveLedger
from quant_fund.live.permissions import is_owner_only, restrict_to_owner
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
SATURDAY = int(
    datetime(2024, 1, 6, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)
RUNNER = CliRunner()


def _clock() -> int:
    return TS


def _config(tmp: Path, **overrides: object) -> LiveConfig:
    payload: dict[str, object] = {
        "schema_version": 1,
        "venue": "loopback",
        "starting_cash": 1_000_000.0,
        "calendar": "us_rth",
        "pretrade_config": str(PRETRADE),
        "state_dir": str(tmp),
        "cash_tolerance": 1e-6,
        "position_tolerance": 1e-6,
    }
    payload.update(overrides)
    return LiveConfig.model_validate(payload)


def _open(tmp: Path, **overrides: object) -> Any:
    return open_session(_config(tmp, **overrides), PRETRADE, state_dir=tmp, now_ns=_clock)


def _arm(session: Any, *, phrase: str = "") -> None:
    session.arm(actor="ada", confirm_live_capital=phrase)


def _buy(session: Any, **overrides: object) -> dict[str, Any]:
    payload: dict[str, object] = {
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 10.0,
        "limit_price": 100.0,
        "mark": 100.0,
        "actor": "ada",
        "at_ns": TS,
    }
    payload.update(overrides)
    return session.submit(**payload)


def test_paper_adapter_still_refuses_a_live_endpoint() -> None:
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint("alpaca-live")


def test_owner_only_mode_is_posix_0600(tmp_path: Path) -> None:
    path = tmp_path / "secret"
    path.write_bytes(b"x")
    os.chmod(path, 0o644)
    if os.name != "nt":
        assert not is_owner_only(path)
    restrict_to_owner(path)
    assert is_owner_only(path)
    if os.name != "nt":
        assert os.stat(path).st_mode & 0o777 == 0o600


def test_unarmed_submit_does_not_call_the_broker(tmp_path: Path) -> None:
    session = _open(tmp_path)
    with pytest.raises(LiveGateError, match="not armed"):
        _buy(session)
    assert session.broker.account().positions == {}
    assert session.orders == []


def test_loopback_fill_is_accounting_not_a_research_claim(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    result = _buy(session, client_order_id="dc-order-1")
    assert result["ok"] is True
    assert result["status"] == "filled"
    assert result["live_pnl_claim"] is False
    assert result["research_evidence"] is False
    assert result["filled_quantity"] == 10.0
    account = session.broker.account()
    assert account.positions["AAPL"] == 10.0
    assert account.cash == pytest.approx(1_000_000.0 - 1_000.0)
    replay = _buy(session, client_order_id="dc-order-1")
    assert replay["replayed"] is True
    assert session.broker.account().positions["AAPL"] == 10.0
    assert is_owner_only(tmp_path / "arm.json")


def test_pretrade_collar_blocks_before_the_broker(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    denied = _buy(session, limit_price=110.0)
    assert denied["ok"] is False
    assert denied["broker_called"] is False
    assert "price_collar" in denied["reasons"]
    assert session.broker.account().cash == pytest.approx(1_000_000.0)


def test_kill_switch_blocks_new_orders_and_cancel_still_works(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.defer_fill = True
    working = _buy(session, client_order_id="dc-working")
    assert working["status"] == "accepted"
    session.kill(actor="ada", reason="operator halt", at_ns=TS)
    blocked = _buy(session, quantity=11.0, client_order_id="dc-blocked")
    assert blocked["ok"] is False
    assert "kill_switch" in blocked["reasons"]
    canceled = session.cancel(client_order_id="dc-working", actor="ada", at_ns=TS + 1)
    assert canceled["status"] == "canceled"
    assert session.broker.account().positions == {}


def test_resume_does_not_duplicate_a_fill(tmp_path: Path) -> None:
    first = _open(tmp_path)
    _arm(first)
    _buy(first, client_order_id="dc-resume")
    second = _open(tmp_path)
    replay = _buy(second, client_order_id="dc-resume")
    assert replay["replayed"] is True
    assert second.broker.account().positions["AAPL"] == 10.0
    assert second.broker.account().cash == pytest.approx(1_000_000.0 - 1_000.0)


def test_cash_drift_alarms(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.cash += 50.0
    report = session.reconcile(at_ns=TS + 2)
    assert report["ok"] is False
    assert report["drift"]["alarm"] is True
    assert report["live_pnl_claim"] is False


def test_saturday_rth_is_closed_and_continuous_is_open(tmp_path: Path) -> None:
    closed = open_session(
        _config(tmp_path / "rth"),
        PRETRADE,
        state_dir=tmp_path / "rth",
        now_ns=lambda: SATURDAY,
    )
    _arm(closed)
    denied = closed.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        at_ns=SATURDAY,
    )
    assert "market_closed" in denied["reasons"]

    opened = open_session(
        _config(tmp_path / "all", calendar="continuous"),
        PRETRADE,
        state_dir=tmp_path / "all",
        now_ns=lambda: SATURDAY,
    )
    _arm(opened)
    filled = opened.submit(
        symbol="AAPL",
        side="buy",
        quantity=1.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        at_ns=SATURDAY,
    )
    assert filled["status"] == "filled"


def test_flatten_sells_the_loopback_position(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session, quantity=4.0)
    flat = session.flatten(actor="ada", at_ns=TS + 5)
    assert flat["ok"] is True
    assert session.broker.account().positions.get("AAPL", 0.0) == pytest.approx(0.0)


def test_alpaca_live_requires_the_phrase_and_refuses_a_backdated_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    session = _open(tmp_path, venue="alpaca-live")
    with pytest.raises(LiveGateError, match="confirm-live-capital"):
        session.arm(actor="ada", confirm_live_capital="please")
    session.arm(actor="ada", confirm_live_capital=LIVE_CAPITAL_PHRASE)
    with pytest.raises(LiveGateError, match="refused for alpaca-live"):
        _buy(session, quantity=1.0)
    with pytest.raises(LiveGateError, match="credentials"):
        session.submit(
            symbol="AAPL",
            side="buy",
            quantity=1.0,
            limit_price=100.0,
            mark=100.0,
            actor="ada",
        )


def test_alpaca_paper_posts_only_to_the_paper_host(tmp_path: Path) -> None:
    transport = _Script()
    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    session = open_session(
        _config(tmp_path, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(session)
    result = _buy(session, quantity=1.0, client_order_id="dc-paper-1")
    assert result["status"] == "filled"
    posts = [call for call in transport.calls if call[0] == "POST"]
    assert len(posts) == 1
    assert posts[0][1] == "https://paper-api.alpaca.markets/v2/orders"
    assert posts[0][2]["APCA-API-SECRET-KEY"] == "SECRETVALUE"
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    state = (tmp_path / "state.json").read_text(encoding="utf-8")
    assert "SECRETVALUE" not in audit
    assert "SECRETVALUE" not in state
    assert "KEYID" not in audit


def test_alpaca_live_host_is_not_the_paper_host() -> None:
    """The live host stays configured. Submission is refused before any socket."""
    transport = _Script()
    broker = AlpacaBroker("alpaca-live", ("KEYID", "SECRETVALUE"), transport=transport)
    assert broker.base == "https://api.alpaca.markets"
    with pytest.raises(LiveBrokerRefused, match="honesty contract"):
        broker.submit(
            LiveOrder(
                client_order_id="dc-live-1",
                symbol="AAPL",
                side="buy",
                quantity=1.0,
                limit_price=100.0,
                fill_time_ns=TS,
            )
        )
    assert transport.calls == []
    with pytest.raises(LiveBrokerRefused, match="honesty contract"):
        broker.recover("dc-live-1")
    assert transport.calls == []


def test_ledger_duplicate_orphan_and_out_of_order_are_idempotent() -> None:
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("a", expects_fill=True)
    first = LedgerFill("f1", "a", "AAPL", "buy", 1.0, 10.0, 20)
    assert ledger.record_fill(first) == "accepted"
    assert ledger.record_fill(first) == "duplicate_fill"
    assert ledger.computed_cash == pytest.approx(990.0)
    late = LedgerFill("f0", "a", "AAPL", "buy", 1.0, 10.0, 10)
    assert ledger.record_fill(late) == "out_of_order_fill"
    orphan = LedgerFill("fx", "missing", "AAPL", "buy", 1.0, 10.0, 30)
    assert ledger.record_fill(orphan) == "orphan_fill"
    report = ledger.reconcile()
    assert report["ok"] is False
    assert report["orphan_fills"] == 1
    assert report["out_of_order_fills"] == 1
    assert report["live_pnl_claim"] is False


def test_state_that_claims_live_profit_is_refused(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session)
    path = tmp_path / "state.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["live_pnl_claim"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LiveGateError, match="live profit"):
        _open(tmp_path)


def test_cli_arm_and_submit(tmp_path: Path) -> None:
    config_path = tmp_path / "live.yaml"
    state = tmp_path / "state"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "venue": "loopback",
                "starting_cash": 100000,
                "calendar": "us_rth",
                "pretrade_config": str(PRETRADE),
                "state_dir": str(state),
                "cash_tolerance": 1e-6,
                "position_tolerance": 1e-6,
            }
        ),
        encoding="utf-8",
    )
    common = ["--config", str(config_path), "--state-dir", str(state)]
    unarmed = RUNNER.invoke(
        app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
            "--actor",
            "ada",
            "--at",
            "2024-01-03T15:00:00-05:00",
        ],
    )
    assert unarmed.exit_code == 1
    armed = RUNNER.invoke(app, ["live", "arm", *common, "--actor", "ada"])
    assert armed.exit_code == 0
    submitted = RUNNER.invoke(
        app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
            "--actor",
            "ada",
            "--at",
            "2024-01-03T15:00:00-05:00",
        ],
    )
    assert submitted.exit_code == 0, submitted.stdout + submitted.stderr
    body = json.loads(submitted.stdout)
    assert body["status"] == "filled"
    assert body["live_pnl_claim"] is False
    status = RUNNER.invoke(app, ["live", "status", *common])
    assert status.exit_code == 0
    snapshot = json.loads(status.stdout)
    assert snapshot["armed"] is True
    assert snapshot["computed_positions"]["AAPL"] == 1.0


def _tight(tmp: Path, **limits: object) -> Path:
    raw = yaml.safe_load(PRETRADE.read_text(encoding="utf-8"))
    raw["limits"].update(limits)
    path = tmp / "pretrade.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def _gateway(tmp: Path, *, pretrade: Path, broker: Any = None) -> Any:
    return open_session(
        _config(tmp, pretrade_config=str(pretrade)),
        pretrade,
        state_dir=tmp,
        now_ns=_clock,
        broker=broker,
    )


def test_lost_broker_response_stays_inflight_until_it_can_be_canceled(tmp_path: Path) -> None:
    broker = _LosingBroker()
    first = open_session(
        _config(tmp_path, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(first)
    with pytest.raises(LiveGateError, match="stays inflight"):
        _buy(first, quantity=1.0, client_order_id="dc-lost")
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "inflight"
    second = open_session(
        _config(tmp_path, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    assert second.orders[0]["status"] == "inflight"
    broker.lose_recover = False
    canceled = second.cancel(client_order_id="dc-lost", actor="ada", at_ns=TS + 1)
    assert canceled["status"] == "canceled"
    assert canceled["replayed"] is False
    assert broker.cancels == 1


def test_pretrade_limits_survive_a_new_process(tmp_path: Path) -> None:
    pretrade = _tight(tmp_path, max_orders_per_window=1)
    first = _gateway(tmp_path / "rate", pretrade=pretrade)
    _arm(first)
    assert isinstance(first.broker, LoopbackBroker)
    first.broker.defer_fill = True
    accepted = _buy(first, quantity=1.0, client_order_id="dc-rate-1")
    assert accepted["status"] == "accepted"
    second = _gateway(tmp_path / "rate", pretrade=pretrade)
    denied = _buy(second, quantity=2.0, limit_price=100.0, client_order_id="dc-rate-2")
    assert denied["ok"] is False
    assert "order_rate" in denied["reasons"]

    dup_root = tmp_path / "dup"
    dup = _gateway(dup_root, pretrade=PRETRADE)
    _arm(dup)
    _buy(dup, client_order_id="dc-dup-1")
    resumed = _gateway(dup_root, pretrade=PRETRADE)
    again = _buy(resumed, client_order_id="dc-dup-2")
    assert again["ok"] is False
    assert "duplicate_order" in again["reasons"]


def test_working_orders_count_toward_position_limits(tmp_path: Path) -> None:
    pretrade = _tight(tmp_path, max_position_quantity=10)
    root = tmp_path / "work"
    first = _gateway(root, pretrade=pretrade)
    _arm(first)
    assert isinstance(first.broker, LoopbackBroker)
    first.broker.defer_fill = True
    working = _buy(first, quantity=10.0, client_order_id="dc-work")
    assert working["status"] == "accepted"
    assert first.broker.account().positions == {}
    blocked = _buy(first, quantity=1.0, limit_price=101.0, client_order_id="dc-work-2")
    assert blocked["ok"] is False
    assert "position_quantity" in blocked["reasons"]
    second = _gateway(root, pretrade=pretrade)
    again = _buy(second, quantity=1.0, limit_price=101.0, client_order_id="dc-work-3")
    assert again["ok"] is False
    assert "position_quantity" in again["reasons"]


def test_unmarked_broker_position_refuses_submission(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.positions["MSFT"] = 5.0
    with pytest.raises(LiveGateError, match="no mark"):
        _buy(session, quantity=1.0)


def test_recovery_uses_the_client_order_id_query(tmp_path: Path) -> None:
    del tmp_path
    seen: list[str] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        seen.append(url)
        return 404, b""

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    assert broker.recover("dc/lost") is None
    assert len(seen) == 1
    assert "/v2/orders:by_client_order_id?" in seen[0]
    assert "client_order_id=dc%2Flost" in seen[0]
    assert "/v2/orders:by_client_order_id/dc" not in seen[0]


def test_broker_json_longer_than_500_chars_parses() -> None:
    long_id = "ord-" + ("x" * 600)

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, timeout
        request = json.loads(body or b"{}")
        payload = {
            "id": long_id,
            "client_order_id": request["client_order_id"],
            "status": "accepted",
            "filled_qty": "0",
            "symbol": request["symbol"],
            "side": request["side"],
        }
        encoded = json.dumps(payload).encode()
        assert len(encoded) > 500
        return 200, encoded

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    ack = broker.submit(
        LiveOrder("dc-long", "AAPL", "buy", 1.0, 100.0, TS),
    )
    assert ack.status == "accepted"
    assert ack.broker_order_id == long_id


def test_descending_fill_page_is_recorded_chronologically() -> None:
    rows = [
        {
            "id": "f-late",
            "order_id": "ord-1",
            "side": "buy",
            "qty": "1",
            "price": "11",
            "symbol": "AAPL",
            "transaction_time": "2024-01-03T20:00:02+00:00",
        },
        {
            "id": "f-early",
            "order_id": "ord-1",
            "side": "buy",
            "qty": "1",
            "price": "10",
            "symbol": "AAPL",
            "transaction_time": "2024-01-03T20:00:01+00:00",
        },
    ]

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        assert "direction=desc" in url
        return 200, json.dumps(rows).encode()

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    page = broker.fills(broker_to_client={"ord-1": "dc-fills"})
    assert [fill.fill_id for fill in page.fills] == ["f-early", "f-late"]
    ledger = LiveLedger(baseline_cash=1_000.0)
    ledger.note_order("dc-fills", expects_fill=True)
    for fill in page.fills:
        assert (
            ledger.record_fill(
                LedgerFill(
                    fill.fill_id,
                    fill.client_order_id,
                    fill.symbol,
                    fill.side,
                    fill.quantity,
                    fill.price,
                    fill.fill_time_ns,
                )
            )
            == "accepted"
        )
    assert ledger.reconcile()["out_of_order_fills"] == 0


def test_reconcile_marks_a_working_order_filled(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    assert isinstance(session.broker, LoopbackBroker)
    session.broker.defer_fill = True
    working = _buy(session, client_order_id="dc-later")
    assert working["status"] == "accepted"
    assert working["filled_quantity"] == 0.0
    session.broker._fills.append(
        BrokerFill("loopback:dc-later", "dc-later", "AAPL", "buy", 10.0, 100.0, TS + 5)
    )
    session.broker.cash -= 1_000.0
    session.broker.positions["AAPL"] = 10.0
    report = session.reconcile(at_ns=TS + 6)
    assert report["ok"] is True
    row = session._find("dc-later")
    assert row is not None
    assert row["status"] == "filled"
    assert row["filled_quantity"] == 10.0
    replayed = session.cancel(client_order_id="dc-later", actor="ada", at_ns=TS + 7)
    assert replayed["replayed"] is True
    assert replayed["status"] == "filled"


def test_reopen_repairs_the_audit_log_mode(tmp_path: Path) -> None:
    session = _open(tmp_path)
    _arm(session)
    _buy(session, quantity=1.0)
    audit = tmp_path / "audit.jsonl"
    # Recreate the log under a permissive umask so the reopen path must
    # tighten it. No chmod literal — CodeQL flags those as too open.
    payload = audit.read_bytes()
    audit.unlink()
    previous = os.umask(0o022)
    try:
        audit.write_bytes(payload)
    finally:
        os.umask(previous)
    assert os.stat(audit).st_mode & 0o077 != 0
    _open(tmp_path)
    assert is_owner_only(audit)


def test_expired_and_unknown_broker_statuses_are_not_accepted(tmp_path: Path) -> None:
    assert map_broker_status("expired") == "expired"
    assert map_broker_status("done_for_day") == "done_for_day"
    assert map_broker_status("pending_cancel") == "pending_cancel"
    assert map_broker_status("suspended") == "suspended"
    assert map_broker_status("not-a-status") == "unmapped"
    assert map_broker_status("") == "unmapped"
    transport = _StatusScript("expired")
    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    session = open_session(
        _config(tmp_path, venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path,
        now_ns=_clock,
        broker=broker,
    )
    _arm(session)
    result = _buy(session, quantity=1.0, client_order_id="dc-expired")
    assert result["status"] == "expired"
    assert result["ok"] is False
    canceled = session.cancel(client_order_id="dc-expired", actor="ada", at_ns=TS + 1)
    assert canceled["replayed"] is True
    assert canceled["status"] == "expired"
    assert not any(call[0] == "DELETE" for call in transport.calls)

    unknown = _StatusScript("halted_for_review")
    other = open_session(
        _config(tmp_path / "unk", venue="alpaca-paper"),
        PRETRADE,
        state_dir=tmp_path / "unk",
        now_ns=_clock,
        broker=AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=unknown),
    )
    _arm(other)
    mapped = _buy(other, quantity=1.0, client_order_id="dc-unknown")
    assert mapped["status"] == "unmapped"
    assert mapped["status"] != "accepted"
    assert "unmapped broker status" in mapped["reason"]


def test_two_sessions_do_not_overwrite_each_others_orders(tmp_path: Path) -> None:
    first = _open(tmp_path)
    _arm(first)
    second = _open(tmp_path)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def buy(session: Any, client: str, quantity: float) -> None:
        try:
            barrier.wait(timeout=5)
            result = _buy(session, quantity=quantity, client_order_id=client)
            if result["ok"] is not True:
                errors.append(AssertionError(result))
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=buy, args=(first, "dc-race-1", 1.0)),
        threading.Thread(target=buy, args=(second, "dc-race-2", 2.0)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert errors == []
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert {row["client_order_id"] for row in stored["orders"]} == {"dc-race-1", "dc-race-2"}


class _LosingBroker:
    """Records the order, then drops the submit response and the lookup."""

    def __init__(self) -> None:
        self.lose_recover = True
        self.cancels = 0
        self._stored: BrokerAck | None = None
        self.marks: dict[str, float] = {}

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        return BrokerAccount(
            cash=1_000_000.0, equity=1_000_000.0, positions={}, marks=dict(self.marks)
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        self._stored = BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id="broker-lost",
            status="accepted",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )
        raise TimeoutError("response lost")

    def recover(self, client_order_id: str) -> BrokerAck | None:
        del client_order_id
        if self.lose_recover:
            raise TimeoutError("lookup failed")
        return self._stored

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del client_order_id
        self.cancels += 1
        ack = BrokerAck(
            client_order_id=self._stored.client_order_id if self._stored else "",
            broker_order_id=broker_order_id,
            status="canceled",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )
        self._stored = ack
        return ack

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=(), truncated=False)


class _StatusScript:
    def __init__(self, status: str) -> None:
        self.status = status
        self.calls: list[tuple[str, str]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, timeout
        self.calls.append((method, url))
        if method == "POST":
            request = json.loads(body or b"{}")
            payload = {
                "id": "ord-status",
                "client_order_id": request["client_order_id"],
                "status": self.status,
                "filled_qty": "0",
                "symbol": request["symbol"],
                "side": request["side"],
            }
            return 200, json.dumps(payload).encode()
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "1000000", "equity": "1000000"}).encode()
        if url.endswith("/v2/positions"):
            return 200, b"[]"
        return 404, b"{}"


class _Script:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.filled = False

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del timeout
        self.calls.append((method, url, headers, body))
        if method == "POST":
            self.filled = True
            request = json.loads(body or b"{}")
            payload = {
                "id": "ord-1",
                "client_order_id": request["client_order_id"],
                "status": "filled",
                "filled_qty": request["qty"],
                "filled_avg_price": request["limit_price"],
                "symbol": request["symbol"],
                "side": request["side"],
                "filled_at": "2024-01-03T20:00:00+00:00",
            }
            return 200, json.dumps(payload).encode()
        if url.endswith("/v2/account"):
            cash = "999900" if self.filled else "1000000"
            return 200, json.dumps({"cash": cash, "equity": "1000000"}).encode()
        if url.endswith("/v2/positions"):
            rows = [{"symbol": "AAPL", "qty": "1"}] if self.filled else []
            return 200, json.dumps(rows).encode()
        if "activities" in url:
            return 200, b"[]"
        return 404, b"{}"
