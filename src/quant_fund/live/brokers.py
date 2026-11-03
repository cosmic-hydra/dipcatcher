"""Broker gateways for the armed session.

``loopback`` fills inside this process. Alpaca paper and Alpaca live use the
fixed public hosts only. Credentials stay in the request headers and never in
URLs, audit lines, or exception text.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, cast
from urllib.parse import quote, urlencode

from quant_fund.live.config import Venue
from quant_fund.live.errors import BrokerRejected, LiveBrokerRefused, LiveGateError

_HOSTS: dict[str, str] = {
    "alpaca-paper": "https://paper-api.alpaca.markets",
    "alpaca-live": "https://api.alpaca.markets",
}
_PAGE = 100
# Alpaca's documented lifecycle. Unknown values stay ``unmapped`` so an expired
# or suspended order is not stored as a working accept.
_STATUS_MAP: dict[str, str] = {
    "new": "accepted",
    "accepted": "accepted",
    "pending_new": "accepted",
    "accepted_for_bidding": "accepted",
    "pending_review": "accepted",
    "calculated": "accepted",
    "stopped": "accepted",
    "partially_filled": "partially_filled",
    "filled": "filled",
    "done_for_day": "done_for_day",
    "canceled": "canceled",
    "cancelled": "canceled",
    "expired": "expired",
    "replaced": "replaced",
    "pending_cancel": "pending_cancel",
    "pending_replace": "pending_replace",
    "rejected": "rejected",
    "suspended": "suspended",
    "unmapped": "unmapped",
}


def map_broker_status(raw: str) -> str:
    """Map one broker status. An unknown value is ``unmapped``, never ``accepted``."""
    return _STATUS_MAP.get(raw.strip().lower(), "unmapped")


Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes]]


@dataclass(frozen=True)
class BrokerFill:
    fill_id: str
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    fill_time_ns: int


@dataclass(frozen=True)
class BrokerAck:
    client_order_id: str
    broker_order_id: str
    status: str
    filled_quantity: float
    fill_price: float | None
    fills: tuple[BrokerFill, ...]
    reason: str


@dataclass(frozen=True)
class BrokerAccount:
    cash: float
    equity: float
    positions: dict[str, float]
    marks: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FillPage:
    fills: tuple[BrokerFill, ...]
    truncated: bool


@dataclass(frozen=True)
class LiveOrder:
    client_order_id: str
    symbol: str
    side: str
    quantity: float
    limit_price: float
    fill_time_ns: int


class Broker(Protocol):
    def set_mark(self, symbol: str, price: float) -> None: ...

    def submit(self, order: LiveOrder) -> BrokerAck: ...

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck: ...

    def recover(self, client_order_id: str) -> BrokerAck | None: ...

    def account(self) -> BrokerAccount: ...

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage: ...


def alpaca_credentials() -> tuple[str, str] | None:
    """Read one consistent pair; identical stripped duplicates in both namespaces are allowed."""
    namespaces = (
        ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"),
        ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
    )
    pairs = [
        (os.environ.get(key_name, "").strip(), os.environ.get(secret_name, "").strip())
        for key_name, secret_name in namespaces
    ]
    if not any(key or secret for key, secret in pairs):
        return None

    complete_pairs: list[tuple[str, str]] = []
    for key, secret in pairs:
        if not key and not secret:
            continue
        if not key or not secret or len(key) > 200 or len(secret) > 200:
            raise LiveGateError("broker credentials are incomplete")
        if any(char in key or char in secret for char in "\r\n\x00"):
            raise LiveGateError("broker credentials are malformed")
        complete_pairs.append((key, secret))
    if len(set(complete_pairs)) > 1:
        raise LiveGateError("broker credentials conflict across namespaces")
    return complete_pairs[0]


def urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes]:
    """HTTP/1.1 request that refuses redirects so the credential header cannot leave the host."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(_RefuseRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            status = int(response.status)
            payload = response.read()
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        payload = exc.read()
    return status, payload


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


def _redact(text: str, secrets: tuple[str, ...]) -> str:
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned[:500]


def _position_mark(row: dict[str, Any]) -> float | None:
    """Use the broker valuation when the operator has not stored a mark."""
    for key in ("current_price", "lastday_price", "avg_entry_price"):
        raw = row.get(key)
        if raw in (None, ""):
            continue
        price = _money(raw, "position mark")
        if price > 0.0:
            return price
    market_value = row.get("market_value")
    qty = row.get("qty")
    if market_value in (None, "") or qty in (None, ""):
        return None
    quantity = _money(qty, "position")
    if quantity == 0.0:
        return None
    price = abs(_money(market_value, "market value") / quantity)
    if price > 0.0:
        return price
    return None


def _money(value: object, label: str) -> float:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise BrokerRejected(f"broker {label} was not a number") from exc
    if not math.isfinite(number):
        raise BrokerRejected(f"broker {label} was not finite")
    return number


def _ns(text: str) -> int:
    stamp = text.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(stamp)
    if parsed.tzinfo is None:
        raise BrokerRejected("broker fill time had no timezone")
    return int(parsed.timestamp() * 1_000_000_000)


class LoopbackBroker:
    """In-process venue. State is rebuilt from the session ledger after a restart."""

    def __init__(
        self,
        *,
        cash: float,
        positions: dict[str, float] | None = None,
    ) -> None:
        self.cash = float(cash)
        self.positions = {str(k): float(v) for k, v in (positions or {}).items()}
        self.marks: dict[str, float] = {}
        self.defer_fill = False
        self._orders: dict[str, BrokerAck] = {}
        self._fills: list[BrokerFill] = []

    def set_mark(self, symbol: str, price: float) -> None:
        self.marks[str(symbol)] = float(price)

    def account(self) -> BrokerAccount:
        equity = self.cash
        for symbol, qty in self.positions.items():
            equity += qty * self.marks.get(symbol, 0.0)
        return BrokerAccount(
            cash=self.cash,
            equity=equity,
            positions=dict(self.positions),
            marks=dict(self.marks),
        )

    def submit(self, order: LiveOrder) -> BrokerAck:
        prior = self._orders.get(order.client_order_id)
        if prior is not None:
            return prior
        if order.side not in {"buy", "sell"}:
            raise BrokerRejected("side must be buy or sell")
        signed = order.quantity if order.side == "buy" else -order.quantity
        held = self.positions.get(order.symbol, 0.0)
        if order.side == "sell" and order.quantity > held + 1e-9:
            ack = self._ack(order, "rejected", 0.0, None, (), "insufficient position")
            self._orders[order.client_order_id] = ack
            return ack
        cost = signed * order.limit_price
        if order.side == "buy" and cost > self.cash + 1e-9:
            ack = self._ack(order, "rejected", 0.0, None, (), "insufficient cash")
            self._orders[order.client_order_id] = ack
            return ack
        if self.defer_fill:
            ack = self._ack(order, "accepted", 0.0, None, (), "")
            self._orders[order.client_order_id] = ack
            return ack
        self.cash -= cost
        self.positions[order.symbol] = held + signed
        fill = BrokerFill(
            fill_id=f"loopback:{order.client_order_id}",
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=order.limit_price,
            fill_time_ns=order.fill_time_ns,
        )
        self._fills.append(fill)
        ack = self._ack(order, "filled", order.quantity, order.limit_price, (fill,), "")
        self._orders[order.client_order_id] = ack
        return ack

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        del broker_order_id
        prior = self._orders.get(client_order_id)
        if prior is None:
            raise BrokerRejected("order is not at the loopback venue")
        if prior.status != "accepted":
            return prior
        canceled = BrokerAck(
            client_order_id=client_order_id,
            broker_order_id=prior.broker_order_id,
            status="canceled",
            filled_quantity=0.0,
            fill_price=None,
            fills=(),
            reason="",
        )
        self._orders[client_order_id] = canceled
        return canceled

    def recover(self, client_order_id: str) -> BrokerAck | None:
        return self._orders.get(client_order_id)

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        del broker_to_client
        return FillPage(fills=tuple(self._fills), truncated=False)

    def _ack(
        self,
        order: LiveOrder,
        status: str,
        filled: float,
        price: float | None,
        fills: tuple[BrokerFill, ...],
        reason: str,
    ) -> BrokerAck:
        return BrokerAck(
            client_order_id=order.client_order_id,
            broker_order_id=f"loopback:{order.client_order_id}",
            status=status,
            filled_quantity=filled,
            fill_price=price,
            fills=fills,
            reason=reason,
        )


class AlpacaBroker:
    """Alpaca trading API v2. Paper and live are different hosts, chosen by the venue."""

    def __init__(
        self,
        venue: Venue,
        credentials: tuple[str, str] | None,
        *,
        transport: Transport | None = None,
        timeout: float = 10.0,
    ) -> None:
        if venue not in _HOSTS:
            raise LiveGateError("Alpaca venue must be alpaca-paper or alpaca-live")
        self.venue = venue
        self.base = _HOSTS[venue]
        self._credentials = credentials
        self._transport = transport or urllib_transport
        self._timeout = float(timeout)
        self._marks: dict[str, float] = {}

    @property
    def credentials_present(self) -> bool:
        return self._credentials is not None

    def set_mark(self, symbol: str, price: float) -> None:
        self._marks[str(symbol)] = float(price)

    def submit(self, order: LiveOrder) -> BrokerAck:
        body = {
            "symbol": order.symbol,
            "qty": _decimal(order.quantity),
            "side": order.side,
            "type": "limit",
            "time_in_force": "day",
            "limit_price": _decimal(order.limit_price),
            "client_order_id": order.client_order_id,
        }
        try:
            payload = self._request("POST", "/v2/orders", body)
        except BrokerRejected as exc:
            if "client_order_id" not in str(exc):
                raise
            recovered = self.recover(order.client_order_id)
            if recovered is None:
                raise
            return recovered
        if not isinstance(payload, dict):
            raise BrokerRejected("broker order response was not an object")
        return self._ack(payload)

    def cancel(self, client_order_id: str, broker_order_id: str) -> BrokerAck:
        target = broker_order_id or ""
        if not target:
            found = self.recover(client_order_id)
            if found is None:
                raise BrokerRejected("order is not at the broker")
            target = found.broker_order_id
        payload = self._request("DELETE", f"/v2/orders/{quote(target, safe='')}", None)
        if payload == {}:
            recovered = self.recover(client_order_id)
            if recovered is not None:
                return recovered
            return BrokerAck(
                client_order_id=client_order_id,
                broker_order_id=target,
                status="canceled",
                filled_quantity=0.0,
                fill_price=None,
                fills=(),
                reason="",
            )
        if not isinstance(payload, dict):
            raise BrokerRejected("broker cancel response was not an object")
        return self._ack(payload)

    def recover(self, client_order_id: str) -> BrokerAck | None:
        # Alpaca documents a query parameter, not a path segment. A path
        # segment returns no order, so a lost POST cannot be found or canceled.
        query = urlencode({"client_order_id": client_order_id})
        path = f"/v2/orders:by_client_order_id?{query}"
        payload, missing = self._request_optional(path)
        if missing:
            return None
        if not isinstance(payload, dict):
            raise BrokerRejected("broker order lookup was not an object")
        return self._ack(payload)

    def account(self) -> BrokerAccount:
        payload = self._request("GET", "/v2/account", None)
        positions = self._request("GET", "/v2/positions", None)
        if not isinstance(payload, dict) or not isinstance(positions, list):
            raise BrokerRejected("broker account response was malformed")
        book: dict[str, float] = {}
        marks: dict[str, float] = {}
        for row in positions:
            if not isinstance(row, dict):
                raise BrokerRejected("broker position row was malformed")
            symbol = str(row.get("symbol") or "")
            book[symbol] = _money(row.get("qty"), "position")
            mark = _position_mark(row)
            if symbol and mark is not None:
                marks[symbol] = mark
        book.pop("", None)
        marks.pop("", None)
        return BrokerAccount(
            cash=_money(payload.get("cash"), "cash"),
            equity=_money(payload.get("equity"), "equity"),
            positions=book,
            marks=marks,
        )

    def fills(self, *, broker_to_client: dict[str, str]) -> FillPage:
        payload = self._request(
            "GET",
            f"/v2/account/activities/FILL?page_size={_PAGE}&direction=desc",
            None,
        )
        if not isinstance(payload, list):
            raise BrokerRejected("broker fill list was not a list")
        fills: list[BrokerFill] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            broker_order_id = str(row.get("order_id") or "")
            side = str(row.get("side") or "")
            if side not in {"buy", "sell"}:
                continue
            client = broker_to_client.get(broker_order_id, "")
            if not client and not broker_order_id:
                continue
            when = row.get("transaction_time")
            fill_id = str(row.get("id") or "")
            if not isinstance(when, str) or not fill_id:
                continue
            fills.append(
                BrokerFill(
                    fill_id=fill_id,
                    client_order_id=client,
                    symbol=str(row.get("symbol") or ""),
                    side=side,
                    quantity=_money(row.get("qty"), "fill quantity"),
                    price=_money(row.get("price"), "fill price"),
                    fill_time_ns=_ns(when),
                )
            )
        fills.sort(key=lambda fill: (fill.fill_time_ns, fill.fill_id))
        return FillPage(fills=tuple(fills), truncated=len(payload) >= _PAGE)

    def _ack(self, payload: dict[str, Any]) -> BrokerAck:
        raw_status = str(payload.get("status") or "")
        status = map_broker_status(raw_status)
        filled = _money(payload.get("filled_qty") or 0, "filled quantity")
        raw_price = payload.get("filled_avg_price")
        fill_price = None if raw_price in (None, "") else _money(raw_price, "average price")
        client = str(payload.get("client_order_id") or "")
        broker_id = str(payload.get("id") or "")
        symbol = str(payload.get("symbol") or "")
        side = str(payload.get("side") or "")
        fills: tuple[BrokerFill, ...] = ()
        if status in {"filled", "partially_filled"} and filled > 0.0 and fill_price is not None:
            when = (
                payload.get("filled_at") or payload.get("updated_at") or payload.get("submitted_at")
            )
            stamp = _ns(when) if isinstance(when, str) else 0
            fills = (
                BrokerFill(
                    fill_id=f"order:{broker_id}:cum",
                    client_order_id=client,
                    symbol=symbol,
                    side=side,
                    quantity=filled,
                    price=fill_price,
                    fill_time_ns=stamp,
                ),
            )
        reason = ""
        if status == "unmapped":
            shown = raw_status or "empty"
            reason = f"unmapped broker status {shown}"
        return BrokerAck(
            client_order_id=client,
            broker_order_id=broker_id,
            status=status,
            filled_quantity=filled,
            fill_price=fill_price,
            fills=fills,
            reason=reason,
        )

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        payload, missing = self._exchange(method, path, body, allow_missing=False)
        if missing:
            raise BrokerRejected("broker order was not found")
        return payload

    def _request_optional(self, path: str) -> tuple[Any, bool]:
        return self._exchange("GET", path, None, allow_missing=True)

    def _exchange(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        allow_missing: bool,
    ) -> tuple[Any, bool]:
        if self.venue == "alpaca-live":
            raise LiveBrokerRefused(
                "alpaca-live is refused: the honesty contract forbids live broker "
                "connectivity (AGENTS.md; docs/INSTITUTIONAL_READINESS.md). "
                "The venue and host api.alpaca.markets stay configured so this "
                "refusal is explicit. Paper and loopback are unchanged."
            )
        if self._credentials is None:
            raise LiveGateError("broker credentials are not set")
        key, secret = self._credentials
        url = self.base + path
        if not url.startswith(self.base + "/"):
            raise LiveGateError("refusing a broker URL outside the venue host")
        raw_body = _json_body(body)
        headers = _alpaca_headers(key, secret, has_body=raw_body is not None)
        status, payload = self._transport(method, url, headers, raw_body, self._timeout)
        return _parse_broker_response(status, payload, (key, secret), allow_missing=allow_missing)


def _decimal(value: float) -> str:
    text = format(float(value), ".10f").rstrip("0").rstrip(".")
    return text or "0"


def _json_body(body: dict[str, Any] | None) -> bytes | None:
    if body is None:
        return None
    return json.dumps(body, separators=(",", ":")).encode("utf-8")


def _alpaca_headers(key: str, secret: str, *, has_body: bool) -> dict[str, str]:
    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json",
        "User-Agent": "dipcatcher-live/1",
    }
    if has_body:
        headers["Content-Type"] = "application/json"
    return headers


def _parse_broker_response(
    status: int,
    payload: bytes,
    secrets: tuple[str, str],
    *,
    allow_missing: bool,
) -> tuple[Any, bool]:
    if status == 404 and allow_missing:
        return None, True
    text = payload.decode("utf-8", errors="replace")
    if status == 204 or not text.strip():
        return _empty_response(status)
    return _parse_json_response(status, text, secrets)


def _empty_response(status: int) -> tuple[Any, bool]:
    if 200 <= status < 300:
        return {}, False
    raise BrokerRejected(f"broker HTTP {status}")


def _parse_json_response(status: int, text: str, secrets: tuple[str, str]) -> tuple[Any, bool]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BrokerRejected(f"broker HTTP {status} returned non-JSON") from exc
    if status >= 400:
        raise BrokerRejected(_broker_error(status, parsed, secrets))
    return parsed, False


def _broker_error(status: int, parsed: Any, secrets: tuple[str, str]) -> str:
    message = ""
    if isinstance(parsed, dict):
        message = str(parsed.get("message") or parsed.get("error") or "")
    message = _redact(message, secrets)[:300]
    detail = f": {message}" if message else ""
    return f"broker HTTP {status}{detail}"
