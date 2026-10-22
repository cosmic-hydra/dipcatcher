"""Live desk lifecycle edges: disarm/reset/mark and halt-before-arm.

Fail-closed operator surface. No network, no live broker socket.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_fund.live.desk import LiveDesk, LiveDeskError
from quant_fund.pretrade.config import PretradeConfig, SessionConfig
from tests.unit.pretrade.support import HMAC_KEY, TS_NS, limit_config


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


def test_open_rejects_negative_clock(tmp_path: Path) -> None:
    with pytest.raises(LiveDeskError, match="ts_ns must be non-negative"):
        LiveDesk.open(
            tmp_path / "desk",
            hmac_key=HMAC_KEY,
            config=_config(),
            cash=1.0,
            account_value=1.0,
            ts_ns=-1,
        )


def test_unknown_venue_and_halt_blocks_arming_transmitting_venue(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        with pytest.raises(LiveDeskError, match="unknown venue"):
            desk.arm("alpaca-live", "", TS_NS)
        desk.kill("ops", "halt", TS_NS + 1)
        with pytest.raises(LiveDeskError, match="halted"):
            desk.arm("loopback", "", TS_NS + 2)
        reset = desk.reset("ops", "clear", TS_NS + 3)
        assert reset["status"] == "reset"
        assert reset["transmission_after_reset"] is False
        armed = desk.arm("loopback", "", TS_NS + 4)
        assert armed["status"] == "armed"
        disarmed = desk.disarm(TS_NS + 5)
        assert disarmed["status"] == "shadow"
        assert desk.status()["venue"] == "shadow"


def test_mark_and_non_monotonic_time_deny(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        marked = desk.mark("aapl", 101.0, TS_NS + 10)
        assert marked["status"] == "marked"
        first = desk.submit("aapl", "BUY", 1, 101.0, TS_NS + 11, client_order_id="t1")
        assert first["accepted"] is True
        denied = desk.submit("aapl", "BUY", 1, 101.0, TS_NS + 5, client_order_id="t2")
        assert denied["accepted"] is False
        assert "non_monotonic_time" in denied["order"]["reasons"]  # type: ignore[index]


def test_integrity_freeze_blocks_reset(tmp_path: Path) -> None:
    with _open(tmp_path) as desk:
        desk.submit("aapl", "BUY", 1, 100.0, TS_NS, client_order_id="j")
    journal = tmp_path / "desk" / "journal.jsonl"
    journal.write_text(
        journal.read_text(encoding="utf-8").replace("shadowed", "filled"), encoding="utf-8"
    )
    with _open(tmp_path) as desk:
        assert desk.status()["kill_latched"] is True
        with pytest.raises(LiveDeskError, match="journal"):
            desk.reset("ops", "nope", TS_NS + 5)
