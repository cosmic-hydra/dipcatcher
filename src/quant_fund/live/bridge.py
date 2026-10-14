"""Bridges the desk can hand a ticket to. Neither bridge opens a network socket.

Loopback acks and fills inside the process. Those fills are SYNTHETIC.
Drop writes one JSON ticket per order into ``outbound/`` and reads operator
events from ``inbound/``. The operator's own process is what talks to a broker.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class BridgeError(ValueError):
    """An inbound event is not a usable broker report."""


@dataclass(frozen=True)
class BrokerEvent:
    kind: str
    order_id: str
    event_id: str
    qty: float = 0.0
    price: float = 0.0
    fee: float = 0.0
    reason: str = ""
    synthetic: bool = False


_EVENT_KINDS = frozenset({"ack", "reject", "fill", "cancel"})


def parse_event(raw: Mapping[str, object], *, synthetic: bool, source: str) -> BrokerEvent:
    """Validate one inbound object. ``source`` is a filename or ``loopback``."""
    if raw.get("schema") != "live_broker_event.v1":
        raise BridgeError(f"{source}: schema must be live_broker_event.v1")
    kind = raw.get("kind")
    order_id = raw.get("order_id")
    event_id = raw.get("event_id")
    if kind not in _EVENT_KINDS or not isinstance(kind, str):
        raise BridgeError(f"{source}: kind must be ack, reject, fill, or cancel")
    if not isinstance(order_id, str) or not order_id:
        raise BridgeError(f"{source}: order_id is required")
    if not isinstance(event_id, str) or not event_id:
        raise BridgeError(f"{source}: event_id is required")
    qty = _num(raw.get("qty", 0.0), f"{source}: qty")
    price = _num(raw.get("price", 0.0), f"{source}: price")
    fee = _num(raw.get("fee", 0.0), f"{source}: fee")
    if fee < 0.0:
        raise BridgeError(f"{source}: fee cannot be negative")
    if kind == "fill" and (qty <= 0.0 or price <= 0.0):
        raise BridgeError(f"{source}: fill needs a positive qty and price")
    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        raise BridgeError(f"{source}: reason must be a string")
    return BrokerEvent(
        kind=kind,
        order_id=order_id,
        event_id=event_id,
        qty=qty,
        price=price,
        fee=fee,
        reason=reason,
        synthetic=synthetic,
    )


def _num(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BridgeError(f"{label} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise BridgeError(f"{label} must be finite")
    return number


class LoopbackBridge:
    """In-process venue. A transmitted order fills in full at its limit price."""

    def __init__(self) -> None:
        self._queued: list[BrokerEvent] = []

    def transmit(self, ticket: Mapping[str, object]) -> None:
        kind = ticket.get("ticket")
        order_id = ticket.get("order_id")
        if not isinstance(order_id, str) or not order_id:
            raise BridgeError("loopback ticket missing order_id")
        if kind == "cancel":
            self._queued.append(
                BrokerEvent(
                    kind="cancel",
                    order_id=order_id,
                    event_id=f"c-{order_id}",
                    synthetic=True,
                )
            )
            return
        if kind != "order":
            raise BridgeError("loopback ticket must be an order or a cancel")
        qty = _num(ticket.get("qty"), "loopback qty")
        price = _num(ticket.get("limit_price"), "loopback limit_price")
        self._queued.append(
            BrokerEvent(kind="ack", order_id=order_id, event_id=f"a-{order_id}", synthetic=True)
        )
        self._queued.append(
            BrokerEvent(
                kind="fill",
                order_id=order_id,
                event_id=f"f-{order_id}",
                qty=qty,
                price=price,
                synthetic=True,
            )
        )

    def poll(self) -> list[BrokerEvent]:
        queued = self._queued
        self._queued = []
        return queued


class DropBridge:
    """File drop. This process never opens a socket."""

    def __init__(self, root: Path) -> None:
        self.outbound = root / "outbound"
        self.inbound = root / "inbound"

    def transmit(self, ticket: Mapping[str, object]) -> None:
        order_id = ticket.get("order_id")
        ticket_kind = ticket.get("ticket")
        if not isinstance(order_id, str) or not order_id:
            raise BridgeError("drop ticket missing order_id")
        if ticket_kind == "cancel":
            name = f"cancel-{order_id}.json"
        elif ticket_kind == "order":
            name = f"{order_id}.json"
        else:
            raise BridgeError("drop ticket must be an order or a cancel")
        payload = dict(ticket)
        payload["live_pnl_claim"] = False
        payload["would_promote_live"] = False
        payload["evidence_conditions"] = "blocked"
        _atomic_json(self.outbound / name, payload)

    def poll(self) -> list[BrokerEvent]:
        if not self.inbound.exists():
            return []
        events: list[BrokerEvent] = []
        for path in sorted(self.inbound.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise BridgeError(f"{path.name}: invalid JSON") from exc
            if not isinstance(raw, dict):
                raise BridgeError(f"{path.name}: event must be a JSON object")
            events.append(parse_event(raw, synthetic=False, source=path.name))
        return events


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)
