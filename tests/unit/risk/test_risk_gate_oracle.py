"""Risk-gate completeness oracle: reject iff some bound is actually violated.

The gate is the last safety boundary before a fill — this fuzzes every input
dimension against an independent reimplementation of the accept/reject
decision, so a future edit that silently drops a check (or adds a spurious
one) fails here rather than in a ledger.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quant_fund.config.loader import load_config
from quant_fund.config.models import AppConfig
from quant_fund.portfolio.risk_gate import check_order, exceeds_limit
from quant_fund.risk.overlay import BookRiskOverlay
from quant_fund.schemas.errors import RiskGateRejected
from quant_fund.schemas.orders import Order, OrderSide, OrderStatus

_T = datetime(2024, 1, 2, tzinfo=UTC)


# paper.yaml is a large inherited model. Hypothesis calls _cfg once per
# example (hundreds of times); re-parsing YAML dominated the test. A deep
# copy keeps each example's mutations off the shared original.
_PAPER_CONFIG: AppConfig | None = None


def _cfg(tmp_path: Path) -> AppConfig:
    global _PAPER_CONFIG
    if _PAPER_CONFIG is None:
        _PAPER_CONFIG = load_config("configs/paper.yaml")
    cfg = _PAPER_CONFIG.model_copy(deep=True)
    cfg.data.root = tmp_path
    return cfg


def _order(qty: float, side: OrderSide) -> Order:
    return Order(
        order_id="o1",
        security_id="A",
        symbol="A",
        side=side,
        quantity=qty,
        signal_time=_T,
        decision_time=_T,
        order_time=_T,
        status=OrderStatus.NEW,
    )


def _expected_reject(
    *,
    qty: float,
    side: OrderSide,
    nav: float,
    price: float,
    current_weight: float,
    gross_after: float,
    net_after: float,
    participation: float,
    predicted_vol: float,
    market_vol: float | None,
    price_age: int | None,
    model_age: float | None,
    g: object,
) -> bool:
    """Independent restatement of the gate contract (same slack policy)."""
    values = [nav, price, current_weight, gross_after, net_after, participation, predicted_vol]
    if market_vol is not None:
        values.append(market_vol)
    if any(not (v == v and abs(v) != float("inf")) for v in values):
        return True
    if nav <= 0 or price <= 0:
        return True
    if gross_after < 0 or participation < 0 or predicted_vol < 0:
        return True
    if market_vol is not None and market_vol < 0:
        return True
    if not (qty == qty and abs(qty) != float("inf")) or qty <= 0:
        return True
    if price_age is not None and (price_age < 0 or price_age > g.stale_price_bars):
        return True
    if model_age is not None and (model_age < 0 or model_age > g.stale_model_hours):
        return True
    signed = -qty if side is OrderSide.SELL else qty
    notional = abs(qty) * price
    name_w = abs(current_weight + (signed * price) / max(nav, 1e-12))
    gate_vol = float(predicted_vol) if market_vol is None else float(market_vol)
    return (
        exceeds_limit(notional, g.max_order_notional)
        or exceeds_limit(name_w, g.max_name)
        or exceeds_limit(gross_after, g.max_gross)
        or exceeds_limit(abs(net_after), g.max_net)
        or exceeds_limit(participation, g.max_participation)
        or exceeds_limit(gate_vol, g.max_predicted_vol)
    )


floats = st.one_of(
    st.floats(min_value=-1e12, max_value=1e12),
    st.sampled_from([float("nan"), float("inf"), -float("inf")]),
)
pos_floats = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)


@given(
    # Order itself validates quantity finite + strictly positive (pydantic
    # schema), so the fuzzed domain is the constructible one.
    qty=st.floats(min_value=1e-300, max_value=1e12),
    side=st.sampled_from([OrderSide.BUY, OrderSide.SELL]),
    nav=floats,
    price=floats,
    current_weight=floats,
    gross_after=floats,
    net_after=floats,
    participation=floats,
    predicted_vol=floats,
    market_vol=st.one_of(st.none(), floats),
    price_age=st.one_of(st.none(), st.integers(min_value=-5, max_value=50)),
    model_age=st.one_of(st.none(), st.floats(min_value=-10, max_value=200)),
)
@settings(
    max_examples=400, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_gate_rejects_iff_violation(
    tmp_path: Path,
    qty: float,
    side: OrderSide,
    nav: float,
    price: float,
    current_weight: float,
    gross_after: float,
    net_after: float,
    participation: float,
    predicted_vol: float,
    market_vol: float | None,
    price_age: int | None,
    model_age: float | None,
) -> None:
    cfg = _cfg(tmp_path)
    g = cfg.risk_gate
    expect = _expected_reject(
        qty=qty,
        side=side,
        nav=nav,
        price=price,
        current_weight=current_weight,
        gross_after=gross_after,
        net_after=net_after,
        participation=participation,
        predicted_vol=predicted_vol,
        market_vol=market_vol,
        price_age=price_age,
        model_age=model_age,
        g=g,
    )
    try:
        check_order(
            _order(qty, side),
            nav=nav,
            price=price,
            current_weight=current_weight,
            gross_after=gross_after,
            net_after=net_after,
            participation=participation,
            predicted_vol=predicted_vol,
            config=cfg,
            price_age_bars=price_age,
            model_age_hours=model_age,
            market_predicted_vol=market_vol,
        )
        accepted = True
    except RiskGateRejected:
        accepted = False
    assert accepted != expect, (
        f"gate {'accepted' if accepted else 'rejected'} but oracle says "
        f"{'reject' if expect else 'accept'}: qty={qty} nav={nav} price={price} "
        f"vol={predicted_vol} mvol={market_vol} age={price_age}/{model_age}"
    )


@given(
    qty=pos_floats,
    nav=pos_floats,
    price=pos_floats,
)
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_gate_accepts_under_wide_limits(
    tmp_path: Path, qty: float, nav: float, price: float
) -> None:
    """With every limit set to infinity-scale, any finite positive order passes."""
    if qty <= 0 or nav <= 0 or price <= 0:
        return
    cfg = _cfg(tmp_path)
    g = cfg.risk_gate
    for field in (
        "max_order_notional",
        "max_gross",
        "max_net",
        "max_name",
        "max_participation",
        "max_predicted_vol",
    ):
        setattr(g, field, 1e30)
    check_order(
        _order(qty, OrderSide.BUY),
        nav=nav,
        price=price,
        current_weight=0.0,
        gross_after=1.0,
        net_after=1.0,
        participation=0.5,
        predicted_vol=0.1,
        config=cfg,
    )


@given(
    nav_path=st.lists(
        st.floats(min_value=1.0, max_value=1e6, allow_nan=False), min_size=0, max_size=300
    )
)
@settings(max_examples=120, deadline=None)
def test_overlay_scale_bounded_and_deterministic(nav_path: list[float]) -> None:
    """scale ∈ [0,1] on every bar and the full replay is deterministic."""
    a, b = BookRiskOverlay(), BookRiskOverlay()
    for nav in nav_path:
        sa = a.preview_scale()
        sb = b.preview_scale()
        assert sa == sb
        assert 0.0 <= sa <= 1.0
        a.observe(nav)
        b.observe(nav)


@given(
    nav_path=st.lists(
        st.one_of(
            st.floats(min_value=-1e9, max_value=1e9),
            st.sampled_from([float("nan"), float("inf"), -float("inf")]),
        ),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=120, deadline=None)
def test_overlay_never_crashes_on_degenerate_navs(nav_path: list[float]) -> None:
    """NaN/inf/negative NAV observations are dropped, never poison the scaler."""
    overlay = BookRiskOverlay()
    for nav in nav_path:
        scale = overlay.preview_scale()
        assert 0.0 <= scale <= 1.0
        overlay.observe(nav)
    assert all(v > 0 and v == v for v in overlay.navs)


def test_overlay_causality_prefix_invariance() -> None:
    """The scale decided at time t cannot change when future NAVs arrive."""
    import numpy as np

    rng = np.random.default_rng(7)
    navs = list(np.cumprod(1.0 + rng.normal(0.0, 0.01, 250)) * 1e6)
    full, prefix = BookRiskOverlay(), BookRiskOverlay()
    prefix_scales: list[float] = []
    for nav in navs:
        prefix_scales.append(prefix.preview_scale())
        prefix.observe(nav)
    cutoff = 120
    for i, nav in enumerate(navs):
        s = full.preview_scale()
        if i < cutoff:
            assert s == pytest.approx(prefix_scales[i], rel=0, abs=0)
        full.observe(nav)
