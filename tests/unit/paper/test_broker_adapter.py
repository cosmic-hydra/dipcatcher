"""Crash-resume recovery tests for the paper-only broker adapter.

SYNTHETIC fills/prices only (correctness tests, never market evidence). Proves
kill/resume mid-run NAV seam parity and NO duplicate orders, plus the hard
live-endpoint refusal. NAV is a simulated accounting identity, never live P&L.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_fund.execution.order_recon import Fill, LiveEndpointRefused, Order
from quant_fund.paper.broker_adapter import PaperBrokerAdapter

T0 = datetime(2024, 1, 2, 21, 0, tzinfo=UTC)
T1 = datetime(2024, 1, 3, 21, 0, tzinfo=UTC)


def _order(oid: str, sec: str = "SEC_A", side: str = "BUY", qty: float = 10.0) -> Order:
    return Order(order_id=oid, security_id=sec, side=side, quantity=qty)  # type: ignore[arg-type]


def _fill(fid: str, oid: str, qty: float, price: float, t: datetime = T0) -> Fill:
    return Fill(
        fill_id=fid, order_id=oid, security_id="SEC_A", quantity=qty, price=price, fill_time=t
    )


def test_live_endpoint_refused_at_construction(tmp_path) -> None:
    with pytest.raises(LiveEndpointRefused):
        PaperBrokerAdapter(state_path=tmp_path / "s.json", endpoint_kind="live")


def test_kill_resume_nav_seam_parity_and_no_duplicate_orders(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run1.submit(_order("o1", qty=10.0))
    run1.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0))
    run1.mark("SEC_A", 11.0)
    nav_before = run1.nav
    # "kill" mid-run: trip the switch, checkpoint, then drop the process.
    run1.trip_kill("mid_run_crash")
    run1.checkpoint()

    run2, seam = PaperBrokerAdapter.resume(path)
    assert seam["nav_seam_parity"] is True
    assert seam["ok"] is True
    assert seam["duplicate_orders"] == 0
    assert abs(nav_before - seam["persisted_nav"]) <= 1e-6
    assert abs(run2.nav - nav_before) <= 1e-6
    # the resumed run remains halted after the prior kill trip
    assert run2._ledger.kill_state == "HALT_NEW_ORDERS"
    report = run2.seam_report()
    assert report["live_pnl_claim"] is False


def test_no_duplicate_orders_across_resume(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    assert run1.submit(_order("o1")) == "accepted"
    assert run1.submit(_order("o1")) == "duplicate_order"
    run1.checkpoint()

    run2, seam = PaperBrokerAdapter.resume(path)
    # re-submitting the same order after resume stays idempotent (no dupes)
    assert run2.submit(_order("o1")) == "duplicate_order"
    assert seam["duplicate_orders"] == 0
    # replaying the same fill is idempotent too
    assert run2.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0)) == "accepted"
    assert run2.apply_fill(_fill("f1", "o1", qty=10.0, price=10.0)) == "duplicate_fill"


def test_resume_continues_and_totals_are_consistent(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run1.submit(_order("o1", qty=5.0))
    run1.apply_fill(_fill("f1", "o1", qty=5.0, price=10.0))
    run1.mark("SEC_A", 10.0)
    run1.checkpoint()

    run2, _ = PaperBrokerAdapter.resume(path)
    run2.submit(_order("o2", qty=5.0))
    run2.apply_fill(_fill("f2", "o2", qty=5.0, price=10.0, t=T1))
    run2.mark("SEC_A", 12.0)
    rep = run2.checkpoint()
    assert rep["ok"] is True
    assert rep["nav_seam_parity"] is True
    # 1000 cash - 100 spent on 10 shares @10; mark 12 -> nav 1000 - 100 + 120
    assert abs(run2.nav - 1020.0) <= 1e-6


def test_cancel_survives_resume_and_blocks_the_late_fill(tmp_path) -> None:
    path = tmp_path / "state.json"
    run1 = PaperBrokerAdapter(state_path=path, starting_cash=1000.0)
    run1.submit(_order("o1", qty=10.0))
    run1.mark("SEC_A", 10.0)
    run1.cancel("o1")
    run1.checkpoint()

    run2, seam = PaperBrokerAdapter.resume(path)
    assert seam["ok"] is True
    assert seam["nav_seam_parity"] is True
    assert run2.cancel("o1") == "duplicate_cancel"
    assert run2.apply_fill(_fill("f-late", "o1", qty=10.0, price=10.0)) == "fill_after_cancel"
    assert abs(run2.nav - 1000.0) <= 1e-6
    assert run2.position("SEC_A") == 0.0


def test_seam_reports_research_only(tmp_path) -> None:
    path = tmp_path / "state.json"
    run = PaperBrokerAdapter(state_path=path, starting_cash=0.0)
    run.checkpoint()
    report = run.seam_report()
    assert report["research_only"] is True
    assert report["live_pnl_claim"] is False
