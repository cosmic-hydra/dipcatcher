"""Broker refusal and fail-closed edges — no live sockets, no network.

Covers credential parsing, loopback cash/position refusals, Alpaca live-host
refusal on every exchange path, and transport/response parsing that must not
escape the configured host. SYNTHETIC fixtures only.
"""

from __future__ import annotations

import json
import math
from unittest.mock import MagicMock, patch

import pytest

from quant_fund.live.brokers import (
    AlpacaBroker,
    LiveOrder,
    LoopbackBroker,
    _broker_error,
    _empty_response,
    _money,
    _ns,
    _parse_broker_response,
    _parse_json_response,
    _position_mark,
    _RefuseRedirect,
    alpaca_credentials,
    map_broker_status,
    urllib_transport,
)
from quant_fund.live.errors import BrokerRejected, LiveBrokerRefused, LiveGateError

TS = 1_704_312_000_000_000_000


def _order(**overrides: object) -> LiveOrder:
    payload: dict[str, object] = {
        "client_order_id": "dc-1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": 1.0,
        "limit_price": 100.0,
        "fill_time_ns": TS,
    }
    payload.update(overrides)
    return LiveOrder(**payload)  # type: ignore[arg-type]


# -- credentials: incomplete / malformed fail closed --


def test_alpaca_credentials_absent_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
    assert alpaca_credentials() is None


def test_alpaca_credentials_incomplete_or_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPACA_API_KEY_ID", "only-key")
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(LiveGateError, match="incomplete"):
        alpaca_credentials()

    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "x" * 201)
    with pytest.raises(LiveGateError, match="incomplete"):
        alpaca_credentials()

    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "sec\nret")
    with pytest.raises(LiveGateError, match="malformed"):
        alpaca_credentials()

    monkeypatch.setenv("ALPACA_API_KEY_ID", "key\rid")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "secret")
    with pytest.raises(LiveGateError, match="malformed"):
        alpaca_credentials()


def test_alpaca_credentials_accept_apca_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    monkeypatch.setenv("APCA_API_KEY_ID", "kid")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "sec")
    assert alpaca_credentials() == ("kid", "sec")


# -- loopback: reject paths that never invent buying power --


def test_loopback_rejects_bad_side_and_insufficient_cash_or_position() -> None:
    broker = LoopbackBroker(cash=50.0, positions={"AAPL": 2.0})
    with pytest.raises(BrokerRejected, match="side must be buy or sell"):
        broker.submit(_order(side="short"))

    cash = broker.submit(_order(quantity=1.0, limit_price=100.0, client_order_id="cash"))
    assert cash.status == "rejected"
    assert "insufficient cash" in cash.reason
    assert broker.cash == pytest.approx(50.0)

    short = broker.submit(
        _order(side="sell", quantity=5.0, limit_price=10.0, client_order_id="short")
    )
    assert short.status == "rejected"
    assert "insufficient position" in short.reason
    assert broker.positions["AAPL"] == pytest.approx(2.0)


def test_loopback_cancel_unknown_and_idempotent_replay() -> None:
    broker = LoopbackBroker(cash=1_000.0)
    broker.defer_fill = True
    first = broker.submit(_order(client_order_id="work"))
    assert first.status == "accepted"
    again = broker.submit(_order(client_order_id="work"))
    assert again is first
    with pytest.raises(BrokerRejected, match="not at the loopback venue"):
        broker.cancel("missing", "loopback:missing")
    canceled = broker.cancel("work", first.broker_order_id)
    assert canceled.status == "canceled"
    assert broker.cancel("work", first.broker_order_id).status == "canceled"


# -- alpaca-live: honesty contract refuses every exchange --


@pytest.mark.parametrize(
    "method,args",
    [
        ("submit", (_order(client_order_id="live-1"),)),
        ("cancel", ("dc-1", "ord-1")),
        ("recover", ("dc-1",)),
        ("account", ()),
        ("fills", ()),
    ],
)
def test_alpaca_live_refuses_every_exchange_path(method: str, args: tuple[object, ...]) -> None:
    transport = MagicMock(return_value=(200, b"{}"))
    broker = AlpacaBroker("alpaca-live", ("KEY", "SECRET"), transport=transport)
    with pytest.raises(LiveBrokerRefused, match="honesty contract"):
        if method == "fills":
            broker.fills(broker_to_client={})
        else:
            getattr(broker, method)(*args)
    assert transport.call_count == 0


def test_alpaca_broker_rejects_unknown_venue_and_missing_credentials() -> None:
    with pytest.raises(LiveGateError, match="alpaca-paper or alpaca-live"):
        AlpacaBroker("loopback", ("K", "S"))  # type: ignore[arg-type]
    broker = AlpacaBroker("alpaca-paper", None, transport=MagicMock())
    with pytest.raises(LiveGateError, match="credentials are not set"):
        broker.submit(_order())


def test_alpaca_paper_malformed_and_http_error_responses_fail_closed() -> None:
    calls: list[tuple[int, bytes]] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, url, headers, body, timeout
        return calls.pop(0)

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)

    calls.append((200, b"[]"))
    with pytest.raises(BrokerRejected, match="not an object"):
        broker.submit(_order(client_order_id="bad-obj"))

    calls.append((500, b"not-json"))
    with pytest.raises(BrokerRejected, match="non-JSON"):
        broker.submit(_order(client_order_id="bad-json"))

    calls.append((403, json.dumps({"message": "SECRETVALUE leaked"}).encode()))
    with pytest.raises(BrokerRejected, match=r"broker HTTP 403: \[redacted\]"):
        broker.submit(_order(client_order_id="forbidden"))

    # DELETE returns empty body; recover then 404s so the local cancel ack is used.
    calls.append((204, b""))
    calls.append((404, b""))
    ack = broker.cancel("dc-empty", "ord-empty")
    assert ack.status == "canceled"
    assert ack.broker_order_id == "ord-empty"


def test_alpaca_paper_account_and_fill_parsing_edges() -> None:
    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "1000", "equity": "1100"}).encode()
        if url.endswith("/v2/positions"):
            return 200, json.dumps(
                [
                    {
                        "symbol": "AAPL",
                        "qty": "2",
                        "current_price": "50",
                    },
                    {
                        "symbol": "MSFT",
                        "qty": "2",
                        "market_value": "200",
                    },
                    {"symbol": "", "qty": "1", "current_price": "1"},
                ]
            ).encode()
        if "activities" in url:
            return 200, json.dumps(
                [
                    {
                        "id": "f1",
                        "order_id": "ord-1",
                        "side": "buy",
                        "qty": "1",
                        "price": "10",
                        "symbol": "AAPL",
                        "transaction_time": "2024-01-03T20:00:00+00:00",
                    },
                    {
                        "id": "f2",
                        "order_id": "ord-2",
                        "side": "hold",
                        "qty": "1",
                        "price": "10",
                        "symbol": "AAPL",
                        "transaction_time": "2024-01-03T20:00:00+00:00",
                    },
                    {"id": "", "order_id": "ord-3", "side": "buy", "qty": "1", "price": "1"},
                    "nope",
                ]
            ).encode()
        return 404, b"{}"

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    account = broker.account()
    assert account.cash == pytest.approx(1000.0)
    assert account.positions["AAPL"] == pytest.approx(2.0)
    assert account.marks["AAPL"] == pytest.approx(50.0)
    assert account.marks["MSFT"] == pytest.approx(100.0)
    assert "" not in account.positions

    page = broker.fills(broker_to_client={"ord-1": "dc-1"})
    assert [fill.fill_id for fill in page.fills] == ["f1"]
    assert page.fills[0].client_order_id == "dc-1"

    def bad_positions(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del method, headers, body, timeout
        if url.endswith("/v2/account"):
            return 200, json.dumps({"cash": "1", "equity": "1"}).encode()
        return 200, json.dumps([{"symbol": "AAPL", "qty": "1"}, "not-a-row"]).encode()

    broken = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=bad_positions)
    with pytest.raises(BrokerRejected, match="position row was malformed"):
        broken.account()


def test_position_mark_and_money_ns_helpers_fail_closed() -> None:
    assert (
        _position_mark({"current_price": "", "lastday_price": "0", "avg_entry_price": None}) is None
    )
    assert _position_mark({"market_value": "10", "qty": "0"}) is None
    assert _position_mark({"market_value": "10", "qty": ""}) is None
    assert _position_mark({"market_value": "-20", "qty": "2"}) == pytest.approx(10.0)

    with pytest.raises(BrokerRejected, match="was not a number"):
        _money("x", "cash")
    with pytest.raises(BrokerRejected, match="was not finite"):
        _money(math.inf, "cash")
    with pytest.raises(BrokerRejected, match="no timezone"):
        _ns("2024-01-03T20:00:00")
    assert _ns("2024-01-03T20:00:00Z") > 0

    with pytest.raises(BrokerRejected, match="broker HTTP 500"):
        _empty_response(500)
    assert _empty_response(204) == ({}, False)

    with pytest.raises(BrokerRejected, match="non-JSON"):
        _parse_json_response(200, "{", ("k", "s"))
    err = _broker_error(400, {"error": "bad KEY"}, ("KEY", "SECRET"))
    assert "[redacted]" in err
    assert "KEY" not in err.replace("[redacted]", "")


def test_parse_broker_response_404_optional_and_redirect_handler() -> None:
    assert _parse_broker_response(404, b"", ("k", "s"), allow_missing=True) == (None, True)
    with pytest.raises(BrokerRejected):
        _parse_broker_response(404, b"{}", ("k", "s"), allow_missing=False)
    handler = _RefuseRedirect()
    assert (
        handler.redirect_request(
            MagicMock(), MagicMock(), 302, "moved", MagicMock(), "https://evil.example/"
        )
        is None
    )


def test_urllib_transport_uses_refuse_redirect_and_maps_http_error() -> None:
    """Transport is exercised with a local fake opener — never a real host."""

    class _Resp:
        status = 200

        def read(self) -> bytes:
            return b'{"ok":true}'

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    class _HTTPError(Exception):
        code = 418

        def read(self) -> bytes:
            return b"teapot"

    fake_opener = MagicMock()
    fake_opener.open.return_value = _Resp()
    with patch("quant_fund.live.brokers.urllib.request.build_opener", return_value=fake_opener):
        status, payload = urllib_transport(
            "GET", "https://paper-api.alpaca.markets/v2/account", {}, None, 1.0
        )
    assert status == 200
    assert payload == b'{"ok":true}'

    import urllib.error

    err = urllib.error.HTTPError(
        "https://paper-api.alpaca.markets/v2/account",
        418,
        "teapot",
        hdrs=None,  # type: ignore[arg-type]
        fp=MagicMock(read=lambda: b"teapot"),
    )
    fake_opener.open.side_effect = err
    with patch(
        "quant_fund.live.brokers.urllib.request.build_opener", return_value=fake_opener
    ) as built:
        status, payload = urllib_transport(
            "GET", "https://paper-api.alpaca.markets/v2/account", {}, None, 1.0
        )
        assert built.call_args.args[0] is _RefuseRedirect
    assert status == 418
    assert payload == b"teapot"


def test_submit_recovers_duplicate_client_order_id() -> None:
    state = {"n": 0}

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, timeout
        state["n"] += 1
        if method == "POST":
            return 422, json.dumps({"message": "client_order_id already exists"}).encode()
        if "by_client_order_id" in url:
            return 200, json.dumps(
                {
                    "id": "ord-dup",
                    "client_order_id": "dc-dup",
                    "status": "accepted",
                    "filled_qty": "0",
                    "symbol": "AAPL",
                    "side": "buy",
                }
            ).encode()
        return 404, b"{}"

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    ack = broker.submit(_order(client_order_id="dc-dup"))
    assert ack.broker_order_id == "ord-dup"
    assert ack.status == "accepted"


def test_map_broker_status_is_conservative() -> None:
    assert map_broker_status(" NEW ") == "accepted"
    assert map_broker_status("cancelled") == "canceled"
    assert map_broker_status("mystery") == "unmapped"


def test_cancel_without_broker_id_looks_up_then_deletes() -> None:
    calls: list[str] = []

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes]:
        del headers, body, timeout
        calls.append(f"{method} {url}")
        if "by_client_order_id" in url:
            return 200, json.dumps(
                {
                    "id": "ord-found",
                    "client_order_id": "dc-found",
                    "status": "accepted",
                    "filled_qty": "0",
                    "symbol": "AAPL",
                    "side": "buy",
                }
            ).encode()
        if method == "DELETE":
            return 200, json.dumps(
                {
                    "id": "ord-found",
                    "client_order_id": "dc-found",
                    "status": "canceled",
                    "filled_qty": "0",
                    "symbol": "AAPL",
                    "side": "buy",
                }
            ).encode()
        return 404, b"{}"

    broker = AlpacaBroker("alpaca-paper", ("KEYID", "SECRETVALUE"), transport=transport)
    ack = broker.cancel("dc-found", "")
    assert ack.status == "canceled"
    assert any("DELETE" in call and "ord-found" in call for call in calls)
