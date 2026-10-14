"""Broker order/fill reconciliation test matrix (PAPER/SIMULATED only).

SYNTHETIC fills only (correctness tests, never market evidence). Covers the
required acceptance matrix: duplicate fills, missing fills, out-of-order fills,
drift alarm, and the live-endpoint-refusal test (configuring a live endpoint
raises). Asserts ``live_pnl_claim=False`` and no live-connectivity claim.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.execution.order_recon import (
    Fill,
    LiveEndpointRefused,
    Order,
    OrderFillLedger,
    OrderLedgerError,
    refuse_live_endpoint,
)
from quant_fund.schemas.errors import KillSwitchActive

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)
T2 = datetime(2024, 1, 4, 21, 0, tzinfo=UTC)


def _order(oid: str = "o1", sec: str = "SEC_A", side: str = "BUY", qty: float = 10.0) -> Order:
    return Order(order_id=oid, security_id=sec, side=side, quantity=qty)  # type: ignore[arg-type]


def _fill(
    fid: str = "f1",
    oid: str = "o1",
    sec: str = "SEC_A",
    qty: float = 10.0,
    price: float = 10.0,
    t: datetime = T0,
    seq: int | None = 1,
) -> Fill:
    return Fill(
        fill_id=fid, order_id=oid, security_id=sec, quantity=qty, price=price, fill_time=t, seq=seq
    )


def _ledger(**kw) -> OrderFillLedger:
    return OrderFillLedger(endpoint_kind="paper", **kw)


# -- live-endpoint refusal (hard no-live guarantee) --


@pytest.mark.parametrize("kind", ["live", "prod", "production", "real_money", "ibkr", "broker"])
def test_configuring_live_endpoint_raises(kind: str) -> None:
    with pytest.raises(LiveEndpointRefused):
        OrderFillLedger(endpoint_kind=kind)


def test_unknown_endpoint_kind_refused() -> None:
    with pytest.raises(LiveEndpointRefused):
        OrderFillLedger(endpoint_kind="somewhere")


def test_refuse_live_endpoint_helper() -> None:
    assert refuse_live_endpoint("paper") == "paper"
    assert refuse_live_endpoint("simulated") == "simulated"
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint("LIVE")
    with pytest.raises(OrderLedgerError):
        refuse_live_endpoint("")


# -- duplicate fills: detected + idempotent (never double-counted) --


def test_duplicate_fill_detected_and_not_double_counted() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    assert led.record_fill(_fill(qty=10.0, price=10.0)) == "accepted"
    assert led.record_fill(_fill(qty=10.0, price=10.0)) == "duplicate_fill"
    report = led.reconcile()
    assert report["counters"]["duplicate_fills"] == 1
    assert report["counters"]["fills"] == 1  # duplicate not re-applied
    assert report["computed_positions"]["SEC_A"] == 10.0  # not 20.0
    assert report["ok"] is False  # duplicate is a finding
    assert report["live_pnl_claim"] is False


# -- missing fills --


def test_missing_fill_detected() -> None:
    led = _ledger()
    led.submit(_order(oid="o1"))
    led.submit(_order(oid="o2"))
    led.record_fill(_fill(fid="f1", oid="o1"))
    report = led.reconcile()
    assert report["missing_fills"] == ["o2"]
    assert report["ok"] is False


def test_missing_expected_fill_id_detected() -> None:
    led = _ledger()
    led.submit(_order(oid="o1"))
    led.record_fill(_fill(fid="f1", oid="o1"))
    report = led.reconcile(expected_fill_ids=["f1", "f2"])
    assert "f2" in report["missing_fills"]


def test_order_without_expected_fill_not_flagged() -> None:
    led = _ledger()
    led.submit(Order(order_id="o1", security_id="A", side="BUY", quantity=1.0, expects_fill=False))
    report = led.reconcile()
    assert report["missing_fills"] == []


# -- out-of-order fills --


def test_out_of_order_fill_detected() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(fid="f1", oid="o1", qty=5.0, t=T1, seq=2))
    verdict = led.record_fill(_fill(fid="f2", oid="o1", qty=5.0, t=T0, seq=1))
    assert verdict == "out_of_order_fill"
    report = led.reconcile()
    assert report["counters"]["out_of_order_fills"] == 1
    assert report["ok"] is False


def test_out_of_order_by_seq_detected() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(fid="f1", oid="o1", qty=5.0, t=T0, seq=5))
    assert led.record_fill(_fill(fid="f2", oid="o1", qty=5.0, t=T0, seq=3)) == "out_of_order_fill"


# -- orphan fill (unknown order) --


def test_orphan_fill_detected() -> None:
    led = _ledger()
    assert led.record_fill(_fill(oid="ghost")) == "orphan_fill"
    report = led.reconcile()
    assert report["counters"]["orphan_fills"] == 1
    assert report["ok"] is False


# -- drift alarms --


def test_position_drift_alarm() -> None:
    led = _ledger()
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(qty=10.0, price=10.0))
    led.declare_state(cash=-100.0, positions={"SEC_A": 999.0})  # wrong position
    report = led.reconcile()
    assert report["drift"]["declared"] is True
    assert report["drift"]["alarm"] is True
    assert report["drift"]["position_drift"]["SEC_A"] > 0
    assert report["ok"] is False


def test_cash_drift_alarm() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(qty=10.0, price=10.0))  # cash -> 900
    led.declare_state(cash=500.0, positions={"SEC_A": 10.0})
    report = led.reconcile()
    assert report["drift"]["alarm"] is True
    assert report["drift"]["cash_drift"] > 0


def test_no_drift_when_declared_matches() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    led.record_fill(_fill(qty=10.0, price=10.0))
    led.declare_state(cash=900.0, positions={"SEC_A": 10.0})
    report = led.reconcile()
    assert report["drift"]["alarm"] is False


# -- kill switch --


def test_kill_switch_blocks_new_orders() -> None:
    led = _ledger()
    led.trip_kill("test_trip")
    with pytest.raises(KillSwitchActive):
        led.submit(_order())
    assert led.reconcile()["kill_state"] == "HALT_NEW_ORDERS"
    led.clear_kill()
    assert led.submit(_order()) == "accepted"


# -- idempotent order replay --


def test_duplicate_order_submission_is_idempotent() -> None:
    led = _ledger()
    assert led.submit(_order(oid="o1")) == "accepted"
    assert led.submit(_order(oid="o1")) == "duplicate_order"
    assert led.reconcile()["counters"]["duplicate_orders"] == 1


def test_reconcile_flags_all_honesty() -> None:
    report = _ledger().reconcile()
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False


def test_overfill_is_not_applied() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    assert led.record_fill(_fill(fid="f1", qty=6.0, price=10.0)) == "accepted"
    assert led.record_fill(_fill(fid="f2", qty=5.0, price=10.0, t=T1, seq=2)) == "overfill"
    report = led.reconcile()
    assert report["counters"]["overfills"] == 1
    assert report["computed_positions"]["SEC_A"] == 6.0
    assert report["computed_cash"] == 940.0
    assert report["ok"] is False
    assert report["orders"][0]["status"] == "partial"


def test_security_mismatch_is_not_applied() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    mismatch = Fill(
        fill_id="f1",
        order_id="o1",
        security_id="OTHER",
        quantity=10.0,
        price=10.0,
        fill_time=T0,
        seq=1,
    )
    assert led.record_fill(mismatch) == "security_mismatch"
    report = led.reconcile()
    assert report["computed_positions"] == {}
    assert report["computed_cash"] == 1000.0
    assert report["counters"]["security_mismatches"] == 1
    assert report["ok"] is False


def test_cancel_then_late_fill_is_not_applied() -> None:
    led = _ledger(starting_cash=1000.0)
    led.submit(_order(qty=10.0))
    led.trip_kill("halt")
    assert led.cancel("o1") == "cancelled"
    assert led.record_fill(_fill(qty=10.0, price=10.0)) == "fill_after_cancel"
    report = led.reconcile()
    assert report["computed_positions"] == {}
    assert report["missing_fills"] == []
    assert report["orders"][0]["status"] == "cancelled"
    assert report["counters"]["fills_after_cancel"] == 1
    assert report["ok"] is False


def test_cancel_rejects_unknown_duplicate_and_filled() -> None:
    led = _ledger()
    assert led.cancel("missing") == "unknown_order"
    led.submit(_order(qty=10.0))
    assert led.cancel("o1") == "cancelled"
    assert led.cancel("o1") == "duplicate_cancel"
    led.submit(_order(oid="o2", qty=4.0))
    led.record_fill(_fill(fid="f2", oid="o2", qty=4.0))
    assert led.cancel("o2") == "already_filled"


def test_nonpositive_fill_is_rejected() -> None:
    led = _ledger()
    led.submit(_order())
    with pytest.raises(OrderLedgerError):
        led.record_fill(_fill(qty=0.0))
    with pytest.raises(OrderLedgerError):
        led.record_fill(_fill(fid="f-neg", qty=-1.0))
    with pytest.raises(OrderLedgerError):
        led.record_fill(_fill(fid="f-price", price=0.0))
