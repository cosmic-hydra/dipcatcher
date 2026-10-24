"""Property tests: live endpoint tokens always refuse; paper/simulated always pass.

No network. Keeps the fail-closed honesty contract stable under string noise.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quant_fund.execution.order_recon import (
    ALLOWED_ENDPOINT_KINDS,
    LiveEndpointRefused,
    OrderLedgerError,
    refuse_live_endpoint,
)
from quant_fund.live.brokers import AlpacaBroker, LiveOrder
from quant_fund.live.errors import LiveBrokerRefused

_LIVE = ("live", "prod", "production", "real", "broker", "ibkr", "interactive", "real_money")


@given(
    token=st.sampled_from(_LIVE),
    prefix=st.text(alphabet=" _-", max_size=4),
    suffix=st.text(alphabet=" _-", max_size=4),
)
@settings(max_examples=40, deadline=None)
def test_refuse_live_endpoint_rejects_live_tokens(token: str, prefix: str, suffix: str) -> None:
    kind = f"{prefix}{token}{suffix}"
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint(kind)
    with pytest.raises(LiveEndpointRefused):
        refuse_live_endpoint(kind.upper())


@given(kind=st.sampled_from(sorted(ALLOWED_ENDPOINT_KINDS)))
@settings(max_examples=10, deadline=None)
def test_refuse_live_endpoint_allows_paper_simulated(kind: str) -> None:
    assert refuse_live_endpoint(kind) == kind
    assert refuse_live_endpoint(kind.upper()) == kind


@given(kind=st.one_of(st.just(""), st.just("   "), st.none()))
@settings(max_examples=5, deadline=None)
def test_refuse_live_endpoint_rejects_empty(kind: object) -> None:
    with pytest.raises((OrderLedgerError, TypeError, LiveEndpointRefused)):
        refuse_live_endpoint(kind)  # type: ignore[arg-type]


@given(qty=st.floats(min_value=0.01, max_value=1e3, allow_nan=False, allow_infinity=False))
@settings(max_examples=20, deadline=None)
def test_alpaca_live_submit_never_calls_transport(qty: float) -> None:
    calls: list[object] = []

    def transport(*args: object, **kwargs: object) -> tuple[int, bytes]:
        calls.append((args, kwargs))
        return 200, b"{}"

    broker = AlpacaBroker("alpaca-live", ("KEY", "SECRET"), transport=transport)
    with pytest.raises(LiveBrokerRefused):
        broker.submit(
            LiveOrder(
                client_order_id="prop",
                symbol="AAPL",
                side="buy",
                quantity=float(qty),
                limit_price=100.0,
                fill_time_ns=1,
            )
        )
    assert calls == []
