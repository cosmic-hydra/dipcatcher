from __future__ import annotations

import json
from typing import Any

import pytest

from quant_fund.live.brokers import AlpacaBroker, BrokerAck, BrokerFill
from quant_fund.live.errors import BrokerRejected


def _recover(payload: dict[str, Any]) -> BrokerAck:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, body, timeout
        assert method == "GET"
        assert url == (
            "https://paper-api.alpaca.markets/v2/orders:by_client_order_id?client_order_id=client-1"
        )
        return 200, json.dumps(payload).encode("utf-8")

    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)
    ack = broker.recover("client-1")
    assert ack is not None
    return ack


def _snapshot(**changes: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": "order-1",
        "client_order_id": "client-1",
        "symbol": "SYNTH",
        "side": "buy",
        "qty": "10",
        "status": "canceled",
        "filled_qty": "4",
        "filled_avg_price": "12.5",
        "filled_at": "2024-01-03T20:00:00+00:00",
    }
    payload.update(changes)
    return payload


@pytest.mark.parametrize("status", ["canceled", "rejected", "expired", "replaced"])
def test_terminal_partial_fill_snapshot_emits_cumulative_fill(status: str) -> None:
    ack = _recover(_snapshot(status=status))

    assert ack.status == status
    assert ack.filled_quantity == 4.0
    assert ack.fill_price == 12.5
    assert ack.fills == (
        BrokerFill(
            fill_id="order:order-1:cum",
            client_order_id="client-1",
            symbol="SYNTH",
            side="buy",
            quantity=4.0,
            price=12.5,
            fill_time_ns=1_704_312_000_000_000_000,
        ),
    )


@pytest.mark.parametrize("status", ["canceled", "rejected"])
def test_valid_zero_fill_terminal_snapshot_has_no_fill(status: str) -> None:
    ack = _recover(_snapshot(status=status, filled_qty="0", filled_avg_price=None))

    assert ack.status == status
    assert ack.filled_quantity == 0.0
    assert ack.fill_price is None
    assert ack.fills == ()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"filled_qty": "-1"}, "filled quantity was negative"),
        ({"filled_qty": "11"}, "filled quantity exceeded order quantity"),
        ({"filled_avg_price": None}, "filled quantity had no positive average price"),
        ({"filled_avg_price": ""}, "filled quantity had no positive average price"),
        ({"filled_avg_price": "0"}, "filled quantity had no positive average price"),
        ({"filled_avg_price": "-1"}, "filled quantity had no positive average price"),
        ({"filled_avg_price": "invalid"}, "average price was not a number"),
        ({"filled_qty": "nan"}, "filled quantity was not finite"),
        ({"filled_avg_price": "inf"}, "average price was not finite"),
        ({"qty": "-inf"}, "order quantity was not finite"),
    ],
)
def test_malformed_execution_snapshot_is_rejected(changes: dict[str, Any], message: str) -> None:
    with pytest.raises(BrokerRejected, match=message):
        _recover(_snapshot(**changes))
