from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from quant_fund.live.brokers import BrokerAccount, BrokerAck, BrokerFill, FillPage, LiveOrder
from quant_fund.live.config import LiveConfig
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.session import open_session

ROOT = Path(__file__).resolve().parents[3]
PRETRADE = ROOT / "configs" / "pretrade_risk.yaml"
TS = int(
    datetime(2024, 1, 3, 15, 0, tzinfo=ZoneInfo("America/New_York")).timestamp() * 1_000_000_000
)


def _open(tmp_path: Path, broker: _ScriptedBroker) -> Any:
    config = LiveConfig.model_validate(
        {
            "schema_version": 1,
            "venue": "alpaca-paper",
            "starting_cash": 1_000_000.0,
            "calendar": "us_rth",
            "pretrade_config": str(PRETRADE),
            "state_dir": str(tmp_path),
            "cash_tolerance": 1e-6,
            "position_tolerance": 1e-6,
        }
    )
    session = open_session(
        config,
        PRETRADE,
        state_dir=tmp_path,
        now_ns=lambda: TS,
        broker=broker,
    )
    session.arm(actor="ada")
    return session


def _buy(session: Any) -> dict[str, Any]:
    return session.submit(
        symbol="AAPL",
        side="buy",
        quantity=10.0,
        limit_price=100.0,
        mark=100.0,
        actor="ada",
        client_order_id="dc-partial",
        at_ns=TS,
    )


def _accounting(session: Any) -> tuple[float, dict[str, float]]:
    return session.ledger.computed_cash, session.ledger.computed_positions


def test_cancel_rejection_preserves_partial_order_and_reserved_exposure(tmp_path: Path) -> None:
    broker = _ScriptedBroker(reject_cancel=True)
    session = _open(tmp_path, broker)
    accepted = _buy(session)
    assert accepted["status"] == "partially_filled"
    assert accepted["filled_quantity"] == 4.0
    before = _accounting(session)
    exposure = session._working_exposure()
    assert exposure == ({"AAPL": 6.0}, 600.0)

    rejected = session.cancel(client_order_id="dc-partial", actor="ada", at_ns=TS + 1)

    assert rejected["ok"] is False
    assert rejected["status"] == "partially_filled"
    assert rejected["filled_quantity"] == 4.0
    assert rejected["reason"] == "fake broker rejected cancel"
    assert broker.cancel_calls == 1
    assert session._working_exposure() == exposure
    assert _accounting(session) == before
    stored = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert stored["orders"][0]["status"] == "partially_filled"
    assert stored["orders"][0]["reason"] == "fake broker rejected cancel"


def test_partial_then_terminal_ack_accounts_only_for_the_increment(tmp_path: Path) -> None:
    broker = _ScriptedBroker()
    session = _open(tmp_path, broker)
    partial = _buy(session)
    assert partial["status"] == "partially_filled"
    assert _accounting(session) == (999_600.0, {"AAPL": 4.0})

    terminal = session.cancel(client_order_id="dc-partial", actor="ada", at_ns=TS + 1)

    assert terminal["ok"] is True
    assert terminal["status"] == "filled"
    assert terminal["filled_quantity"] == 10.0
    assert _accounting(session) == (999_000.0, {"AAPL": 10.0})
    account = broker.account()
    assert account.cash == 999_000.0
    assert account.positions == {"AAPL": 10.0}

    report = session.reconcile(at_ns=TS + 2)
    assert report["ok"] is True
    assert report["duplicate_fills"] == 1
    assert _accounting(session) == (999_000.0, {"AAPL": 10.0})


@pytest.mark.parametrize(
    "case",
    ["client", "broker", "foreign_fill", "regressed", "overfill", "malformed", "mismatch"],
)
def test_invalid_ack_does_not_change_order_or_accounting(tmp_path: Path, case: str) -> None:
    broker = _ScriptedBroker()
    session = _open(tmp_path, broker)
    _buy(session)
    row = session._find("dc-partial")
    assert row is not None
    assert broker.ack is not None
    base = broker.ack
    fill = base.fills[0]
    if case == "client":
        invalid = replace(base, client_order_id="dc-other")
    elif case == "broker":
        invalid = replace(base, broker_order_id="broker-other")
    elif case == "foreign_fill":
        invalid = replace(base, fills=(replace(fill, client_order_id="dc-other"),))
    elif case == "regressed":
        invalid = replace(base, status="accepted", filled_quantity=0.0, fill_price=None, fills=())
    elif case == "overfill":
        invalid = replace(
            base,
            status="filled",
            filled_quantity=11.0,
            fills=(replace(fill, quantity=11.0),),
        )
    elif case == "malformed":
        invalid = replace(base, fills=(replace(fill, price=math.nan),))
    else:
        invalid = replace(base, fills=(replace(fill, quantity=5.0),))
    before_row = dict(row)
    before_accounting = _accounting(session)

    with pytest.raises(LiveGateError):
        session._apply_ack(row, invalid, TS + 1)

    assert row == before_row
    assert _accounting(session) == before_accounting


@pytest.mark.parametrize(
    ("field", "value"),
    [("quantity", 11.0), ("quantity", 0.0), ("price", math.nan), ("symbol", "MSFT")],
)
def test_invalid_execution_does_not_change_position_or_cash(
    tmp_path: Path, field: str, value: object
) -> None:
    broker = _ScriptedBroker()
    session = _open(tmp_path, broker)
    _buy(session)
    fill = BrokerFill("venue-fill", "dc-partial", "AAPL", "buy", 1.0, 100.0, TS + 1)
    invalid = replace(fill, **{field: value})
    before = _accounting(session)

    with pytest.raises(LiveGateError):
        session._record_fill(invalid, TS + 1)

    assert _accounting(session) == before


class _ScriptedBroker:
    def __init__(self, *, reject_cancel: bool = False) -> None:
        self.reject_cancel = reject_cancel
        self.cash = 1_000_000.0
        self.positions: dict[str, float] = {}
        self.marks: dict[str, float] = {}
        self.order: LiveOrder | None = None
        self.ack: BrokerAck | None = None
        self.filled = 0.0
        self.cancel_calls = 0

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[symbol] = price

    def account(self) -> BrokerAccount:
        equity = self.cash + sum(
            quantity * self.marks.get(symbol, 0.0) for symbol, quantity in self.positions.items()
        )
        return BrokerAccount(self.cash, equity, dict(self.positions), dict(self.marks))

    def submit(self, order: LiveOrder) -> BrokerAck:
        self.order = order
        self._set_filled(4.0)
        self.ack = self._ack("partially_filled", 4.0, TS)
        return self.ack

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        assert self.order is not None
        assert client_order_id == self.order.client_order_id
        assert broker_order_id == "broker-order"
        self.cancel_calls += 1
        if self.reject_cancel:
            raise BrokerRejected("fake broker rejected cancel")
        self._set_filled(self.order.quantity)
        self.ack = self._ack("filled", self.order.quantity, TS + 1)
        return self.ack

    def recover(self, client_order_id: str) -> BrokerAck | None:
        assert self.ack is None or client_order_id == self.ack.client_order_id
        return self.ack

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        assert broker_to_client == {"broker-order": "dc-partial"}
        return FillPage(fills=() if self.ack is None else self.ack.fills, truncated=False)

    def _set_filled(self, quantity: float) -> None:
        assert self.order is not None
        increment = quantity - self.filled
        self.cash -= increment * self.order.limit_price
        self.positions[self.order.symbol] = self.positions.get(self.order.symbol, 0.0) + increment
        self.filled = quantity

    def _ack(self, status: str, quantity: float, at_ns: int) -> BrokerAck:
        assert self.order is not None
        fill = BrokerFill(
            fill_id="order:broker-order:cum",
            client_order_id=self.order.client_order_id,
            symbol=self.order.symbol,
            side=self.order.side,
            quantity=quantity,
            price=self.order.limit_price,
            fill_time_ns=at_ns,
        )
        return BrokerAck(
            client_order_id=self.order.client_order_id,
            broker_order_id="broker-order",
            status=status,
            filled_quantity=quantity,
            fill_price=self.order.limit_price,
            fills=(fill,),
            reason="",
        )
