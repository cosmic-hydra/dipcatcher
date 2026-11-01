"""Wave 8: risk_gate rejects increment counter, never fill; kill switch separate."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.execution.simulated_broker import RejectReason, SimulatedBroker
from quant_fund.monitoring.kill_switch import HALT_NEW_ORDERS
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

_PAPER_CONFIG: AppConfig | None = None


def _cfg(tmp_path: Path, *, max_order_notional: float = 1e12) -> AppConfig:
    global _PAPER_CONFIG
    if _PAPER_CONFIG is None:
        _PAPER_CONFIG = load_config("configs/paper.yaml")
    cfg = _PAPER_CONFIG.model_copy(deep=True)
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 3.0
    cfg.risk_gate.max_order_notional = max_order_notional
    cfg.risk_gate.max_participation = 1.0
    return cfg


def _order(qty: float, oid: str, t: datetime) -> Order:
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=OrderSide.BUY,
        quantity=qty,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
    )


@given(
    n_rejects=st.integers(min_value=1, max_value=6),
    qty=st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
@settings(
    max_examples=30, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_risk_gate_rejects_never_fill_and_count(tmp_path: Path, n_rejects: int, qty: float) -> None:
    cfg = _cfg(tmp_path, max_order_notional=10.0)  # force risk_gate on large buys
    broker = SimulatedBroker(config=cfg, initial_cash=1_000_000.0)
    price = 100.0
    broker.mark({"A": price})
    t = datetime(2024, 1, 2, tzinfo=UTC)
    cash0 = broker.cash
    for i in range(n_rejects):
        rec = broker.submit(
            _order(qty, f"rg{i}", t),
            price=price,
            nav=broker.nav({"A": price}),
            adv_dollars=1e12,
            sigma=0.02,
        )
        assert rec.order.status is OrderStatus.REJECTED
        assert rec.fill is None
        assert rec.reject_reason is not None
        assert rec.reject_reason.startswith(RejectReason.RISK_GATE.value)
    assert broker.risk_gate_reject_count == n_rejects
    assert broker.halt_count == 0
    assert broker.reject_count == n_rejects
    assert len(broker.fills) == 0
    assert broker.shares.get("A", 0.0) == 0.0
    assert broker.cash == pytest.approx(cash0)
    acct = broker.reject_accounting()
    assert acct["risk_gate_rejects"] == n_rejects
    assert acct["kill_switch_halts"] == 0
    assert acct["reject_total"] == n_rejects


def test_kill_switch_does_not_increment_risk_gate(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.kill_switch.state = HALT_NEW_ORDERS
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    t = datetime(2024, 1, 2, tzinfo=UTC)
    rec = broker.submit(_order(10.0, "k1", t), price=100.0, nav=100_000.0, adv_dollars=1e12)
    assert rec.order.status is OrderStatus.REJECTED
    assert rec.reject_reason == RejectReason.KILL_SWITCH.value
    assert rec.fill is None
    assert broker.halt_count == 1
    assert broker.risk_gate_reject_count == 0
    assert len(broker.fills) == 0
    acct = broker.reject_accounting()
    assert acct["kill_switch_halts"] == 1
    assert acct["risk_gate_rejects"] == 0


def test_mixed_kill_and_risk_gate_accounting(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, max_order_notional=50.0)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.mark({"A": 100.0})
    t = datetime(2024, 1, 2, tzinfo=UTC)
    # risk_gate reject
    r1 = broker.submit(_order(100.0, "rg1", t), price=100.0, nav=100_000.0, adv_dollars=1e12)
    assert r1.reject_reason and r1.reject_reason.startswith(RejectReason.RISK_GATE.value)
    # then trip kill
    broker.kill.state = HALT_NEW_ORDERS
    r2 = broker.submit(_order(1.0, "k1", t), price=100.0, nav=100_000.0, adv_dollars=1e12)
    assert r2.reject_reason == RejectReason.KILL_SWITCH.value
    assert broker.risk_gate_reject_count == 1
    assert broker.halt_count == 1
    assert broker.reject_count == 2
    assert len(broker.fills) == 0
    acct = broker.reject_accounting()
    assert acct["risk_gate_rejects"] == 1
    assert acct["kill_switch_halts"] == 1
    assert acct["reject_total"] == 2
