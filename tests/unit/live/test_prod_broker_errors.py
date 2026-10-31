from __future__ import annotations

import json

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    LiveOrder,
    _parse_broker_response,
    alpaca_credentials,
)
from quant_fund.live.errors import BrokerRejected, LiveGateError


def _broker(payload: object, *, status: int = 200) -> AlpacaBroker:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return status, json.dumps(payload).encode("utf-8")

    return AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)


def test_broker_error_redacts_credentials_and_bounds_the_message() -> None:
    key = "unit-test-key-marker"
    secret = "unit-test-secret-marker"
    payload = json.dumps({"message": f"denied {key} and {secret} " + "x" * 400}).encode()

    with pytest.raises(BrokerRejected) as caught:
        _parse_broker_response(422, payload, (key, secret), allow_missing=False)

    message = str(caught.value)
    assert message.startswith("broker HTTP 422: denied [redacted] and [redacted]")
    assert key not in message
    assert secret not in message
    assert len(message) <= len("broker HTTP 422: ") + 300


def test_non_json_broker_error_does_not_echo_the_response_body() -> None:
    with pytest.raises(BrokerRejected, match="broker HTTP 502 returned non-JSON") as caught:
        _parse_broker_response(
            502,
            b"unit-test-sensitive-response",
            ("test-key", "test-secret"),
            allow_missing=False,
        )
    assert "unit-test-sensitive-response" not in str(caught.value)


def test_empty_broker_error_and_optional_missing_order_are_distinct() -> None:
    with pytest.raises(BrokerRejected, match="^broker HTTP 503$"):
        _parse_broker_response(503, b"", ("test-key", "test-secret"), allow_missing=False)
    assert _parse_broker_response(
        404, b"not-json", ("test-key", "test-secret"), allow_missing=True
    ) == (None, True)


def test_incomplete_and_malformed_credentials_fail_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALPACA_API_KEY_ID", "test-key")
    with pytest.raises(LiveGateError, match="credentials are incomplete"):
        alpaca_credentials()

    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "bad\nvalue")
    with pytest.raises(LiveGateError, match="credentials are malformed"):
        alpaca_credentials()


def test_submit_rejects_a_non_object_order_response() -> None:
    broker = _broker([])
    order = LiveOrder("client-1", "SYNTH", "buy", 1.0, 10.0, 1)
    with pytest.raises(BrokerRejected, match="order response was not an object"):
        broker.submit(order)


def test_account_rejects_a_non_finite_cash_value() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        payload: object = {"cash": "nan", "equity": "100"} if url.endswith("/account") else []
        return 200, json.dumps(payload).encode()

    broker = AlpacaBroker("alpaca-paper", ("test-key", "test-secret"), transport=transport)
    with pytest.raises(BrokerRejected, match="broker cash was not finite"):
        broker.account()


def test_fill_with_naive_timestamp_is_rejected() -> None:
    row = {
        "id": "fill-1",
        "order_id": "order-1",
        "side": "buy",
        "qty": "1",
        "price": "10",
        "symbol": "SYNTH",
        "transaction_time": "2024-01-03T20:00:00",
    }
    broker = _broker([row])
    with pytest.raises(BrokerRejected, match="fill time had no timezone"):
        broker.fills(broker_to_client={"order-1": "client-1"})
