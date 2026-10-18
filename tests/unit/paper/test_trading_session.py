"""Paper trading session: simulated book, live orders refused.

SYNTHETIC prices and fills are correctness fixtures, never market evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fx1.honesty import validate_fx1_output
from quant_fund.execution.order_recon import LiveEndpointRefused
from quant_fund.paper.trading_session import (
    PaperTradingSession,
    SessionCommandError,
    SessionLimits,
    load_commands,
    module_network_imports,
    run_session,
)
from quant_fund.validation.live_readiness import format_readiness, live_readiness_report

T0 = "2026-10-13T14:30:00+00:00"
T1 = "2026-10-13T14:31:00+00:00"


def _limits() -> SessionLimits:
    return SessionLimits(
        max_order_quantity=100.0,
        max_order_notional=2_000.0,
        max_position_notional=2_000.0,
        max_gross_notional=2_000.0,
    )


def test_readiness_cannot_become_eligible() -> None:
    report = live_readiness_report()
    assert report["eligible"] is False
    assert report["live_orders_enabled"] is False
    assert report["would_promote_live"] is False
    assert report["live_pnl_claim"] is False
    assert report["executable_mode"] == "paper"
    assert [row["status"] for row in report["conditions"]] == [
        "BLOCKED",
        "BLOCKED",
        "NOT_YET_COLLECTED",
        "BLOCKED",
        "ABSENT",
    ]
    text = format_readiness(report)
    assert "Live orders: disabled" in text
    validate_fx1_output(text)


def test_session_module_imports_no_network_client() -> None:
    source = Path("src/quant_fund/paper/trading_session.py").read_text(encoding="utf-8")
    assert module_network_imports(source) == []


def test_live_endpoint_refused(tmp_path: Path) -> None:
    with pytest.raises(LiveEndpointRefused):
        PaperTradingSession(state_path=tmp_path / "book.json", endpoint_kind="ibkr")


def test_round_trip_buy_sell_cancel_and_halt(tmp_path: Path) -> None:
    state = tmp_path / "book.json"
    commands = [
        {"op": "mark", "security_id": "AAPL", "price": 100.0},
        {
            "op": "submit",
            "order_id": "o1",
            "security_id": "AAPL",
            "side": "BUY",
            "quantity": 10,
        },
        {
            "op": "fill",
            "fill_id": "f1",
            "order_id": "o1",
            "security_id": "AAPL",
            "quantity": 4,
            "price": 100.0,
            "fill_time": T0,
        },
        {
            "op": "fill",
            "fill_id": "f2",
            "order_id": "o1",
            "security_id": "AAPL",
            "quantity": 6,
            "price": 101.0,
            "fill_time": T1,
        },
        {
            "op": "submit",
            "order_id": "o2",
            "security_id": "AAPL",
            "side": "SELL",
            "quantity": 10,
        },
        {"op": "cancel", "order_id": "o2"},
        {"op": "kill", "reason": "operator"},
        {
            "op": "submit",
            "order_id": "o3",
            "security_id": "AAPL",
            "side": "SELL",
            "quantity": 1,
        },
    ]
    result = run_session(
        state_path=state, commands=commands, starting_cash=10_000.0, limits=_limits()
    )
    assert result["consistent"] is False
    assert result["live_orders_enabled"] is False
    assert result["would_promote_live"] is False
    assert result["data_label"] == "SIMULATED"
    assert result["nav_seam_parity"] is True
    positions = result["reconciliation"]["computed_positions"]
    assert positions["AAPL"] == 10.0
    # 10000 - 4*100 - 6*101 = 8994. The halted submit is a refusal, not a fill.
    assert result["reconciliation"]["computed_cash"] == pytest.approx(8994.0)
    assert result["steps"][-1]["verdict"] == "rejected"
    assert "kill switch" in result["steps"][-1]["reason"]

    resumed = run_session(
        state_path=state,
        commands=[{"op": "clear_kill"}],
        resume=True,
        limits=_limits(),
    )
    assert resumed["consistent"] is True
    assert resumed["reconciliation"]["kill_state"] == "ENABLED"
    assert resumed["reconciliation"]["computed_positions"]["AAPL"] == 10.0


def test_short_overfill_and_cash_refuse(tmp_path: Path) -> None:
    state = tmp_path / "book.json"
    commands = [
        {"op": "mark", "security_id": "AAPL", "price": 50.0},
        {"op": "submit", "order_id": "short", "security_id": "AAPL", "side": "SELL", "quantity": 1},
        {"op": "submit", "order_id": "big", "security_id": "AAPL", "side": "BUY", "quantity": 1000},
        {"op": "submit", "order_id": "o1", "security_id": "AAPL", "side": "BUY", "quantity": 10},
        {
            "op": "fill",
            "fill_id": "too-big",
            "order_id": "o1",
            "security_id": "AAPL",
            "quantity": 11,
            "price": 50.0,
            "fill_time": T0,
        },
        {
            "op": "fill",
            "fill_id": "f1",
            "order_id": "o1",
            "security_id": "AAPL",
            "quantity": 10,
            "price": 50.0,
            "fill_time": T0,
        },
        {
            "op": "submit",
            "order_id": "o-unmarked",
            "security_id": "MSFT",
            "side": "BUY",
            "quantity": 1,
        },
    ]
    result = run_session(
        state_path=state, commands=commands, starting_cash=1_000.0, limits=_limits()
    )
    verdicts = [step["verdict"] for step in result["steps"]]
    assert verdicts[1] == "rejected"
    assert result["steps"][1]["reason"] == "short_not_authorized"
    assert result["steps"][2]["reason"] == "max_order_quantity"
    assert verdicts[4] == "overfill"
    assert verdicts[5] == "accepted"
    assert result["steps"][6]["reason"] == "missing_mark"
    assert result["consistent"] is False
    assert result["reconciliation"]["computed_positions"]["AAPL"] == 10.0
    assert result["reconciliation"]["computed_cash"] == pytest.approx(500.0)


def test_second_sell_cannot_exceed_the_position(tmp_path: Path) -> None:
    state = tmp_path / "book.json"
    commands = [
        {"op": "mark", "security_id": "AAPL", "price": 10.0},
        {"op": "submit", "order_id": "b1", "security_id": "AAPL", "side": "BUY", "quantity": 5},
        {
            "op": "fill",
            "fill_id": "bf",
            "order_id": "b1",
            "security_id": "AAPL",
            "quantity": 5,
            "price": 10.0,
            "fill_time": T0,
        },
        {"op": "submit", "order_id": "s1", "security_id": "AAPL", "side": "SELL", "quantity": 5},
        {"op": "submit", "order_id": "s2", "security_id": "AAPL", "side": "SELL", "quantity": 1},
    ]
    result = run_session(
        state_path=state, commands=commands, starting_cash=1_000.0, limits=_limits()
    )
    assert result["steps"][3]["verdict"] == "accepted"
    assert result["steps"][4]["reason"] == "short_not_authorized"
    assert result["consistent"] is False


def test_existing_state_requires_resume(tmp_path: Path) -> None:
    state = tmp_path / "book.json"
    run_session(state_path=state, commands=[], starting_cash=100.0, limits=_limits())
    with pytest.raises(SessionCommandError, match="resume"):
        run_session(state_path=state, commands=[], starting_cash=100.0, limits=_limits())


def test_load_commands_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "orders.jsonl"
    path.write_text('{"op": "checkpoint"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(SessionCommandError, match="line 2"):
        load_commands(path)
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["op"] == "checkpoint"
