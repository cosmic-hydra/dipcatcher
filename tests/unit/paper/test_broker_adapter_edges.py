"""Paper broker adapter fail-closed edges: state schema, cancel replay, live refuse.

SYNTHETIC fills only. Never opens a network socket or live endpoint.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from quant_fund.execution.order_recon import Fill, LiveEndpointRefused, Order, OrderLedgerError
from quant_fund.paper.broker_adapter import PaperBrokerAdapter

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)


def _order(oid: str = "o1", qty: float = 10.0) -> Order:
    return Order(order_id=oid, security_id="SEC_A", side="BUY", quantity=qty)  # type: ignore[arg-type]


def _fill(fid: str, oid: str, qty: float = 10.0, price: float = 10.0) -> Fill:
    return Fill(
        fill_id=fid,
        order_id=oid,
        security_id="SEC_A",
        quantity=qty,
        price=price,
        fill_time=T0,
    )


@pytest.mark.parametrize("kind", ["live", "production", "ibkr", "real_money"])
def test_resume_and_construct_refuse_live_kinds(tmp_path: Path, kind: str) -> None:
    with pytest.raises(LiveEndpointRefused):
        PaperBrokerAdapter(state_path=tmp_path / "s.json", endpoint_kind=kind)


def test_read_state_rejects_missing_non_object_and_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    broker = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    with pytest.raises(OrderLedgerError, match="no broker state"):
        broker.seam_report()

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(OrderLedgerError, match="not an object"):
        broker.seam_report()

    path.write_text(json.dumps({"schema": "other"}), encoding="utf-8")
    with pytest.raises(OrderLedgerError, match="paper_broker_state.v1"):
        broker.seam_report()


def test_cancel_checkpoint_and_resume_preserve_cash_identity(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=1_000.0)
    assert run.submit(_order("o1", qty=5.0)) == "accepted"
    assert run.cancel("o1") == "cancelled"
    assert run.submit(_order("o2", qty=4.0)) == "accepted"
    assert run.apply_fill(_fill("f2", "o2", qty=4.0, price=10.0)) == "accepted"
    run.mark("SEC_A", 10.0)
    assert run.cash == pytest.approx(960.0)
    assert run.position("SEC_A") == pytest.approx(4.0)
    assert run.starting_cash == pytest.approx(1_000.0)
    report = run.reconcile()
    assert report["live_pnl_claim"] is False
    run.checkpoint()

    resumed, seam = PaperBrokerAdapter.resume(path)
    assert seam["ok"] is True
    assert seam["nav_seam_parity"] is True
    assert resumed.position("SEC_A") == pytest.approx(4.0)
    assert resumed.cash == pytest.approx(960.0)
    assert resumed.marks["SEC_A"] == pytest.approx(10.0)
    assert resumed.cancel("missing") != "cancelled"


def test_endpoint_kind_mismatch_marks_seam_not_ok(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, endpoint_kind="paper", starting_cash=500.0)
    run.submit(_order("o1", qty=1.0))
    run.checkpoint()
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["endpoint_kind"] = "simulated"
    path.write_text(json.dumps(payload), encoding="utf-8")
    other = PaperBrokerAdapter(state_path=path, endpoint_kind="paper", starting_cash=500.0)
    seam = other.seam_report()
    assert seam["ok"] is False
    assert "endpoint_kind_mismatch" in seam["errors"]
    assert seam["live_pnl_claim"] is False
