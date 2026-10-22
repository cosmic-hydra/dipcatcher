"""Order-desk arming/bridge/ledger edges and gateway fail-closed parsing.

No network, no live broker sockets. Exercises the validation branches that
keep transmission dark and refuse forged live-performance claims.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from quant_fund.live.arming import (
    ENV_ARM,
    Arming,
    ArmingError,
    held_reason,
    read_arming,
    transmits,
    write_arming,
)
from quant_fund.live.bridge import BridgeError, DropBridge, LoopbackBridge, parse_event
from quant_fund.live.config import load_gateway_pretrade, load_live_config, resolve_pretrade_path
from quant_fund.live.errors import LiveGateError
from quant_fund.live.gateway_arm import (
    LIVE_CAPITAL_PHRASE,
    arm_session,
    disarm_session,
    load_hmac_key,
    read_arm,
)
from quant_fund.live.ledger import LedgerFill, LiveLedger, ledger_fill_from_dict
from quant_fund.live.risk_memory import export_pretrade, restore_pretrade
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from quant_fund.pretrade.engine import PretradeEngine
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"


# -- arming --


def test_arming_unknown_venue_and_forged_claims(tmp_path: Path) -> None:
    with pytest.raises(ArmingError, match="unknown venue"):
        Arming("alpaca-live")  # type: ignore[arg-type]

    path = tmp_path / "arming.json"
    write_arming(path, Arming("shadow"))
    assert read_arming(path).venue == "shadow"
    assert read_arming(tmp_path / "missing.json").venue == "shadow"

    path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    with pytest.raises(ArmingError, match="JSON object"):
        read_arming(path)

    forged = {
        "schema": "live_arming.v1",
        "venue": "drop",
        "live_pnl_claim": False,
        "would_promote_live": True,
        "evidence_conditions": "blocked",
    }
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ArmingError, match="would_promote_live"):
        read_arming(path)

    forged["would_promote_live"] = False
    forged["evidence_conditions"] = "ready"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ArmingError, match="evidence conditions"):
        read_arming(path)

    forged["evidence_conditions"] = "blocked"
    forged["schema"] = "other"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ArmingError, match="schema"):
        read_arming(path)

    forged["schema"] = "live_arming.v1"
    forged["venue"] = 12
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(ArmingError, match="venue must be a string"):
        read_arming(path)


def test_transmits_and_held_reason_respect_env_and_halt() -> None:
    shadow = Arming("shadow")
    loopback = Arming("loopback")
    drop = Arming("drop")
    assert transmits(shadow, {ENV_ARM: "1"}, halted=False, trace_ok=True) is False
    assert held_reason(shadow, {}) == "shadow venue does not transmit"
    assert transmits(loopback, {}, halted=False, trace_ok=True) is True
    assert transmits(loopback, {}, halted=True, trace_ok=True) is False
    assert transmits(loopback, {}, halted=False, trace_ok=False) is False
    assert transmits(drop, {}, halted=False, trace_ok=True) is False
    assert held_reason(drop, {}) == f"{ENV_ARM} is not 1"
    assert transmits(drop, {ENV_ARM: "1"}, halted=False, trace_ok=True) is True
    assert held_reason(drop, {ENV_ARM: "1"}) == ""
    assert held_reason(loopback, {}) == ""
    assert Arming("drop").as_dict()["capital"] == "operator_bridge"
    assert Arming("loopback").as_dict()["capital"] == "none"


# -- bridge --


def test_parse_event_validation_matrix() -> None:
    base = {
        "schema": "live_broker_event.v1",
        "kind": "fill",
        "order_id": "o1",
        "event_id": "e1",
        "qty": 1.0,
        "price": 10.0,
    }
    ok = parse_event(base, synthetic=False, source="ok.json")
    assert ok.kind == "fill"
    assert ok.synthetic is False

    with pytest.raises(BridgeError, match="schema"):
        parse_event({**base, "schema": "other"}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="kind"):
        parse_event({**base, "kind": "partial"}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="order_id"):
        parse_event({**base, "order_id": ""}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="event_id"):
        parse_event({**base, "event_id": 1}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="fee cannot be negative"):
        parse_event({**base, "fee": -0.01}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="positive qty and price"):
        parse_event({**base, "qty": 0.0}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="reason must be a string"):
        parse_event({**base, "reason": 12}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="must be a number"):
        parse_event({**base, "qty": True}, synthetic=False, source="s")
    with pytest.raises(BridgeError, match="must be finite"):
        parse_event({**base, "price": float("nan")}, synthetic=False, source="s")


def test_loopback_and_drop_bridge_ticket_errors(tmp_path: Path) -> None:
    loop = LoopbackBridge()
    with pytest.raises(BridgeError, match="missing order_id"):
        loop.transmit({"ticket": "order", "qty": 1, "limit_price": 1})
    with pytest.raises(BridgeError, match="order or a cancel"):
        loop.transmit({"ticket": "amend", "order_id": "o1"})
    loop.transmit({"ticket": "cancel", "order_id": "o1"})
    loop.transmit({"ticket": "order", "order_id": "o2", "qty": 2, "limit_price": 5})
    events = loop.poll()
    assert [event.kind for event in events] == ["cancel", "ack", "fill"]
    assert events[-1].synthetic is True
    assert loop.poll() == []

    drop = DropBridge(tmp_path)
    with pytest.raises(BridgeError, match="missing order_id"):
        drop.transmit({"ticket": "order"})
    with pytest.raises(BridgeError, match="order or a cancel"):
        drop.transmit({"ticket": "noop", "order_id": "o1"})
    drop.transmit({"ticket": "order", "order_id": "o9", "qty": 1, "limit_price": 2})
    ticket = json.loads((tmp_path / "outbound" / "o9.json").read_text(encoding="utf-8"))
    assert ticket["live_pnl_claim"] is False
    assert ticket["evidence_conditions"] == "blocked"
    drop.transmit({"ticket": "cancel", "order_id": "o9"})
    assert (tmp_path / "outbound" / "cancel-o9.json").is_file()

    inbound = tmp_path / "inbound"
    inbound.mkdir()
    (inbound / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(BridgeError, match="invalid JSON"):
        drop.poll()
    (inbound / "bad.json").write_text("[]", encoding="utf-8")
    with pytest.raises(BridgeError, match="JSON object"):
        drop.poll()
    assert DropBridge(tmp_path / "empty").poll() == []


# -- ledger --


def test_ledger_validation_provisional_replace_and_declare_drift() -> None:
    with pytest.raises(ValueError, match="baseline cash"):
        LiveLedger(baseline_cash=float("nan"))
    with pytest.raises(ValueError, match="baseline position"):
        LiveLedger(baseline_cash=10.0, baseline_positions={"A": float("inf")})

    ledger = LiveLedger(baseline_cash=1_000.0)
    with pytest.raises(ValueError, match="not in the ledger"):
        ledger.set_expects_fill("missing", False)

    with pytest.raises(ValueError, match="buy or sell"):
        ledger_fill_from_dict(
            {
                "fill_id": "f",
                "client_order_id": "o",
                "symbol": "AAPL",
                "side": "short",
                "quantity": 1,
                "price": 1,
                "fill_time_ns": 0,
            }
        )
    with pytest.raises(ValueError, match="non-negative"):
        ledger_fill_from_dict(
            {
                "fill_id": "f",
                "client_order_id": "o",
                "symbol": "AAPL",
                "side": "buy",
                "quantity": 1,
                "price": 1,
                "fill_time_ns": -1,
            }
        )

    ledger.note_order("o1", expects_fill=True)
    provisional = LedgerFill("order:o1", "o1", "AAPL", "buy", 1.0, 10.0, 5)
    assert ledger.record_fill(provisional) == "accepted"
    real = LedgerFill("venue:1", "o1", "AAPL", "buy", 1.0, 10.0, 6)
    assert ledger.record_fill(real) == "replaced_provisional"
    assert ledger.computed_cash == pytest.approx(990.0)
    assert [fill.fill_id for fill in ledger.fills()] == ["venue:1"]

    ledger.declare(cash=990.0, positions={"AAPL": 1.0})
    clean = ledger.reconcile()
    assert clean["ok"] is True
    assert clean["live_pnl_claim"] is False

    ledger.declare(cash=500.0, positions={"AAPL": 1.0})
    drifted = ledger.reconcile()
    assert drifted["ok"] is False
    assert drifted["drift"]["alarm"] is True

    with pytest.raises(ValueError, match="declared cash"):
        ledger.declare(cash=float("nan"), positions={})
    with pytest.raises(ValueError, match="declared position"):
        ledger.declare(cash=1.0, positions={"AAPL": float("nan")})

    ledger.set_expects_fill("o1", False)
    assert ledger.reconcile()["missing_fills"] == []


# -- gateway arm / hmac / config --


def test_gateway_arm_read_and_hmac_fail_closed(tmp_path: Path) -> None:
    key_path = tmp_path / "hmac.key"
    key = load_hmac_key(key_path)
    assert len(key) == 32
    assert load_hmac_key(key_path) == key
    tiny = tmp_path / "tiny.key"
    tiny.write_bytes(b"short")
    with pytest.raises(LiveGateError, match="unusable"):
        load_hmac_key(tiny)

    arm_path = tmp_path / "arm.json"
    digest = "a" * 64
    with pytest.raises(LiveGateError, match="pretrade digest"):
        arm_session(arm_path, venue="loopback", actor="ada", pretrade_sha256="short")
    with pytest.raises(LiveGateError, match="visible characters"):
        arm_session(arm_path, venue="loopback", actor="", pretrade_sha256=digest)
    with pytest.raises(LiveGateError, match="only used for venue alpaca-live"):
        arm_session(
            arm_path,
            venue="loopback",
            actor="ada",
            pretrade_sha256=digest,
            confirm_live_capital="nope",
        )
    payload = arm_session(arm_path, venue="loopback", actor="ada", pretrade_sha256=digest)
    assert payload["live_pnl_claim"] is False
    assert read_arm(arm_path, venue="loopback", pretrade_sha256=digest) is not None

    with pytest.raises(LiveGateError, match="venue does not match"):
        read_arm(arm_path, venue="alpaca-paper", pretrade_sha256=digest)
    with pytest.raises(LiveGateError, match="stale"):
        read_arm(arm_path, venue="loopback", pretrade_sha256="b" * 64)

    bad = json.loads(arm_path.read_text(encoding="utf-8"))
    bad["live_pnl_claim"] = True
    arm_path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(LiveGateError, match="live profit"):
        read_arm(arm_path, venue="loopback", pretrade_sha256=digest)

    arm_path.write_text("[]", encoding="utf-8")
    with pytest.raises(LiveGateError, match="JSON object"):
        read_arm(arm_path, venue="loopback", pretrade_sha256=digest)

    assert disarm_session(arm_path, actor="ada")["armed"] is False
    assert read_arm(arm_path, venue="loopback", pretrade_sha256=digest) is None

    live_arm = tmp_path / "live-arm.json"
    with pytest.raises(LiveGateError, match="confirm-live-capital"):
        arm_session(
            live_arm,
            venue="alpaca-live",
            actor="ada",
            pretrade_sha256=digest,
            confirm_live_capital="please",
        )
    arm_session(
        live_arm,
        venue="alpaca-live",
        actor="ada",
        pretrade_sha256=digest,
        confirm_live_capital=LIVE_CAPITAL_PHRASE,
    )
    armed = read_arm(live_arm, venue="alpaca-live", pretrade_sha256=digest)
    assert armed is not None
    assert armed["live_capital_confirmed"] is True


def test_live_config_loader_and_continuous_calendar(tmp_path: Path) -> None:
    path = tmp_path / "live.yaml"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_live_config(path)

    payload = {
        "schema_version": 1,
        "venue": "loopback",
        "starting_cash": 1000.0,
        "calendar": "continuous",
        "pretrade_config": str(PRETRADE),
        "state_dir": str(tmp_path / "state"),
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = load_live_config(path)
    assert resolve_pretrade_path(config, path) == PRETRADE
    loaded, digest, signature = load_gateway_pretrade(config, path, hmac_key=HMAC_KEY)
    assert len(digest) == 64
    assert signature
    assert loaded.session.weekends_open is True


# -- risk memory restore fail-closed --


def test_risk_memory_export_restore_and_malformed_blobs() -> None:
    config = PretradeConfig(
        schema_version=1,
        limits=limit_config(),
        session=SessionConfig(timezone="America/New_York", open_minute=570, close_minute=960),
    )
    engine = PretradeEngine(
        config,
        hmac_key=HMAC_KEY,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=TS_NS,
    )
    engine.ensure_symbol("AAPL")
    blob = export_pretrade(engine)
    other = PretradeEngine(
        config,
        hmac_key=HMAC_KEY,
        initial_nav=1_000_000.0,
        initial_cash=1_000_000.0,
        ts_ns=TS_NS,
    )
    restore_pretrade(other, blob)
    assert other.book.session_start == engine.book.session_start

    with pytest.raises(LiveGateError, match="malformed"):
        restore_pretrade(other, [])
    with pytest.raises(LiveGateError, match="symbols"):
        restore_pretrade(other, {**blob, "symbols": "AAPL"})
    with pytest.raises(LiveGateError, match="floors"):
        restore_pretrade(other, {**blob, "session_start": "x"})
    with pytest.raises(LiveGateError, match="order timestamps"):
        restore_pretrade(other, {**blob, "order_ts": "nope"})
    with pytest.raises(LiveGateError, match="duplicates"):
        restore_pretrade(other, {**blob, "dupes": [[1]]})
    with pytest.raises(LiveGateError, match="duplicates"):
        restore_pretrade(other, {**blob, "dupes": "nope"})
    oversized = {**blob, "order_ts": list(range(10_000))}
    with pytest.raises(LiveGateError, match="does not fit"):
        restore_pretrade(other, oversized)
