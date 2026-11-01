"""Property tests: simulated broker cash + positions = NAV (modulo known fees)."""

from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.execution.simulated_broker import SimulatedBroker
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

_PAPER_CONFIG: AppConfig | None = None


def _cfg(tmp_path):
    global _PAPER_CONFIG
    if _PAPER_CONFIG is None:
        _PAPER_CONFIG = load_config("configs/paper.yaml")
    cfg = _PAPER_CONFIG.model_copy(deep=True)
    cfg.data.root = tmp_path
    cfg.risk_gate.max_name = 1.0
    cfg.risk_gate.max_net = 1.0
    cfg.risk_gate.max_gross = 3.0
    cfg.risk_gate.max_order_notional = 1e12
    cfg.risk_gate.max_participation = 1.0
    cfg.costs.frictionless = False
    cfg.costs.participation_limit = 1.0
    return cfg


def _order(side, qty, oid, t):
    return Order(
        order_id=oid,
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=t,
        decision_time=t,
        order_time=t,
        status=OrderStatus.NEW,
    )


@given(
    qtys=st.lists(
        st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=8,
    ),
    sides=st.lists(st.sampled_from([OrderSide.BUY, OrderSide.SELL]), min_size=1, max_size=8),
)
@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_cash_plus_mv_equals_nav(tmp_path, qtys, sides):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    price = 100.0
    broker.mark({"A": price})
    t = datetime(2024, 1, 2, tzinfo=UTC)
    n = min(len(qtys), len(sides))
    for i in range(n):
        # Keep sells from blowing past long inventory into huge shorts under risk gate
        side = sides[i]
        qty = float(qtys[i])
        if side is OrderSide.SELL and broker.shares.get("A", 0.0) < qty:
            # flip to buy if we don't have inventory — still exercises cash path
            side = OrderSide.BUY
        rec = broker.submit(
            _order(side, qty, f"o{i}", t),
            price=price,
            nav=broker.nav({"A": price}),
            adv_dollars=1e12,
            sigma=0.02,
        )
        _ = rec
    ident = broker.cash_nav_identity({"A": price})
    assert ident["residual"] == pytest.approx(0.0, abs=1e-6)
    assert ident["nav"] == pytest.approx(ident["cash"] + ident["position_mv"], abs=1e-6)


def test_roundtrip_buy_sell_fees_reduce_nav(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    t = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 100.0})
    broker.submit(
        _order(OrderSide.BUY, 10.0, "b1", t), price=100.0, nav=100_000.0, adv_dollars=1e12
    )
    nav_mid = broker.nav({"A": 100.0})
    broker.submit(_order(OrderSide.SELL, 10.0, "s1", t), price=100.0, nav=nav_mid, adv_dollars=1e12)
    # Flat book; NAV < initial because fees burned cash
    assert abs(broker.shares.get("A", 0.0)) < 1e-9
    assert broker.cash < 100_000.0
    ident = broker.cash_nav_identity({"A": 100.0})
    assert ident["residual"] == pytest.approx(0.0, abs=1e-9)


def test_exposures_rejects_negative_nav(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=1.0)
    broker.cash = -1.0
    with pytest.raises(ValueError, match="NAV must be finite and non-negative"):
        broker.exposures({"A": 100.0})


def test_nav_rejects_missing_mark_for_held_position(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=100_000.0)
    broker.shares["A"] = 10.0
    with pytest.raises(ValueError, match="missing market mark"):
        broker.nav({})


def test_from_state_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    broker = SimulatedBroker(config=cfg, initial_cash=50_000.0)
    t = datetime(2024, 1, 2, tzinfo=UTC)
    broker.mark({"A": 50.0})
    broker.submit(_order(OrderSide.BUY, 20.0, "b1", t), price=50.0, nav=50_000.0, adv_dollars=1e12)
    state = broker.to_dict()
    restored = SimulatedBroker.from_state(cfg, state)
    assert restored.cash == pytest.approx(broker.cash)
    assert restored.shares["A"] == pytest.approx(broker.shares["A"])
