"""Crash-resumable PAPER/SIMULATED broker adapter — no live orders, ever.

This adapter wraps :class:`~quant_fund.execution.order_recon.OrderFillLedger`
with a durable checkpoint so a run can be killed mid-flight and resumed with:

* **NAV seam parity** — the net asset value reconstructed after resume matches
  the value persisted at the crash seam (within tolerance).
* **No duplicate orders** — order intake is idempotent by ``order_id``; a
  replayed submission is counted as a duplicate and never double-counted.

It is a simulation only. Configuring a live / production / real-money endpoint
RAISES :class:`LiveEndpointRefused` (the hard no-live guarantee). Every report
carries ``live_pnl_claim=False`` and ``research_only=True``. No live
connectivity, live fills, or live profitability are claimed or possible here.

SYNTHETIC prices/fills in tests are correctness fixtures, never market
evidence. NAV is a simulated mark-to-model accounting identity, not live P&L.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from quant_fund.execution.order_recon import (
    Fill,
    LiveEndpointRefused,
    Order,
    OrderFillLedger,
    OrderLedgerError,
    refuse_live_endpoint,
)


def _fill_to_dict(fill: Fill) -> dict[str, Any]:
    return {
        "fill_id": fill.fill_id,
        "order_id": fill.order_id,
        "security_id": fill.security_id,
        "quantity": fill.quantity,
        "price": fill.price,
        "fill_time": fill.fill_time.isoformat(),
        "seq": fill.seq,
    }


def _fill_from_dict(raw: dict[str, Any]) -> Fill:
    return Fill(
        fill_id=str(raw["fill_id"]),
        order_id=str(raw["order_id"]),
        security_id=str(raw["security_id"]),
        quantity=float(raw["quantity"]),
        price=float(raw["price"]),
        fill_time=datetime.fromisoformat(str(raw["fill_time"])),
        seq=None if raw.get("seq") is None else int(raw["seq"]),
    )


def _order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "security_id": order.security_id,
        "side": order.side,
        "quantity": order.quantity,
        "submitted_time": order.submitted_time.isoformat() if order.submitted_time else None,
        "expects_fill": order.expects_fill,
    }


def _order_from_dict(raw: dict[str, Any]) -> Order:
    submitted = raw.get("submitted_time")
    return Order(
        order_id=str(raw["order_id"]),
        security_id=str(raw["security_id"]),
        side=raw["side"],
        quantity=float(raw["quantity"]),
        submitted_time=datetime.fromisoformat(str(submitted)) if submitted else None,
        expects_fill=bool(raw.get("expects_fill", True)),
    )


@dataclass
class SeamReport:
    """Result of a crash-resume reconstruction at the seam."""

    ok: bool
    nav_seam_parity: bool
    persisted_nav: float
    reconstructed_nav: float
    nav_delta: float
    duplicate_orders: int
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "nav_seam_parity": self.nav_seam_parity,
            "persisted_nav": self.persisted_nav,
            "reconstructed_nav": self.reconstructed_nav,
            "nav_delta": self.nav_delta,
            "duplicate_orders": self.duplicate_orders,
            "errors": list(self.errors),
            "research_only": True,
            "live_pnl_claim": False,
        }


class PaperBrokerAdapter:
    """Durable, idempotent, paper-only broker with crash-resume + NAV seam."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        endpoint_kind: str = "paper",
        starting_cash: float = 0.0,
        nav_tolerance: float = 1e-6,
    ) -> None:
        self.endpoint_kind = refuse_live_endpoint(endpoint_kind)
        self.state_path = Path(state_path)
        self._nav_tolerance = float(nav_tolerance)
        self._starting_cash = float(starting_cash)
        self._ledger = OrderFillLedger(
            endpoint_kind=self.endpoint_kind, starting_cash=starting_cash
        )
        self._orders: dict[str, Order] = {}
        self._order_sequence: list[str] = []
        self._fills: dict[str, Fill] = {}
        self._cancelled: list[str] = []
        self._marks: dict[str, float] = {}
        self._persisted_nav: float | None = None

    # -- NAV identity (simulated mark-to-model, never live P&L) --

    @property
    def starting_cash(self) -> float:
        return self._starting_cash

    @property
    def cash(self) -> float:
        return float(self._ledger.reconcile()["computed_cash"])

    @property
    def marks(self) -> dict[str, float]:
        return dict(self._marks)

    def position(self, security_id: str) -> float:
        positions = self._ledger.reconcile()["computed_positions"]
        return float(positions.get(str(security_id), 0.0))

    def reconcile(self) -> dict[str, Any]:
        return self._ledger.reconcile()

    @property
    def nav(self) -> float:
        state = self._ledger.reconcile()
        cash = float(state["computed_cash"])
        positions = state["computed_positions"]
        return cash + sum(float(positions[sec]) * self._marks.get(sec, 0.0) for sec in positions)

    def mark(self, security_id: str, price: float) -> None:
        self._marks[str(security_id)] = float(price)

    # -- intake (idempotent) --

    def submit(self, order: Order) -> str:
        verdict = self._ledger.submit(order)
        if verdict == "accepted":
            self._orders[order.order_id] = order
            self._order_sequence.append(order.order_id)
        return verdict

    def apply_fill(self, fill: Fill) -> str:
        verdict = self._ledger.record_fill(fill)
        if verdict in ("accepted", "out_of_order_fill"):
            self._fills[fill.fill_id] = fill
        return verdict

    def cancel(self, order_id: str) -> str:
        """Cancel a working order. Risk-reducing, so a halt does not block it."""
        verdict = self._ledger.cancel(order_id)
        if verdict == "cancelled":
            self._cancelled.append(order_id)
        return verdict

    def trip_kill(self, reason: str) -> str:
        return self._ledger.trip_kill(reason)

    def clear_kill(self) -> str:
        return self._ledger.clear_kill()

    # -- durability --

    def _state_dict(self) -> dict[str, Any]:
        return {
            "schema": "paper_broker_state.v1",
            "endpoint_kind": self.endpoint_kind,
            "starting_cash": self._starting_cash,
            "nav_tolerance": self._nav_tolerance,
            "orders": [_order_to_dict(self._orders[oid]) for oid in self._order_sequence],
            "order_sequence": list(self._order_sequence),
            "fills": [_fill_to_dict(f) for f in self._fills.values()],
            "cancelled_order_ids": list(self._cancelled),
            "marks": dict(self._marks),
            "kill_state": self._ledger.kill_state,
            "nav": self.nav,
            "research_only": True,
            "live_pnl_claim": False,
        }

    def checkpoint(self) -> dict[str, Any]:
        """Persist full state atomically; returns the seam report."""
        state = self._state_dict()
        self._persisted_nav = float(state["nav"])
        payload = json.dumps(state, sort_keys=True, default=str) + "\n"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_name(f".{self.state_path.name}.tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.state_path)
        return self.seam_report()

    def seam_report(self) -> dict[str, Any]:
        """Reconstruct from the durable state and check NAV seam parity."""
        persisted = self._read_state()
        errors: list[str] = []
        if persisted.get("endpoint_kind") != self.endpoint_kind:
            errors.append("endpoint_kind_mismatch")
        replayed = self._replay(persisted)
        duplicate_orders = replayed["duplicate_orders"]
        persisted_nav = float(persisted.get("nav", 0.0))
        reconstructed_nav = replayed["nav"]
        nav_delta = abs(reconstructed_nav - persisted_nav)
        parity = nav_delta <= self._nav_tolerance
        if not parity:
            errors.append("nav_seam_mismatch")
        if duplicate_orders:
            errors.append("duplicate_orders")
        report = SeamReport(
            ok=not errors,
            nav_seam_parity=parity,
            persisted_nav=persisted_nav,
            reconstructed_nav=reconstructed_nav,
            nav_delta=nav_delta,
            duplicate_orders=duplicate_orders,
            errors=errors,
        )
        return report.as_dict()

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            raise OrderLedgerError(f"no broker state at {self.state_path}")
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise OrderLedgerError(f"broker state at {self.state_path} is not an object")
        if data.get("schema") != "paper_broker_state.v1":
            raise OrderLedgerError("broker state schema must be paper_broker_state.v1")
        return cast(dict[str, Any], data)

    def _replay(self, persisted: dict[str, Any]) -> dict[str, Any]:
        ledger = OrderFillLedger(
            endpoint_kind=str(persisted.get("endpoint_kind", "paper")),
            starting_cash=float(persisted.get("starting_cash", 0.0)),
        )
        duplicate_orders = 0
        # Replay historical orders/fills BEFORE applying the persisted kill
        # state: reconstruction is not a new order, so it is not kill-guarded.
        for raw in persisted.get("orders", []):
            if ledger.submit(_order_from_dict(raw)) == "duplicate_order":
                duplicate_orders += 1
        for raw in persisted.get("fills", []):
            ledger.record_fill(_fill_from_dict(raw))
        for order_id in _cancelled_ids(persisted):
            verdict = ledger.cancel(order_id)
            if verdict != "cancelled":
                raise OrderLedgerError(f"cannot replay cancel of {order_id}: {verdict}")
        if persisted.get("kill_state") == "HALT_NEW_ORDERS":
            ledger.trip_kill("resumed_halt")
        marks = {str(k): float(v) for k, v in dict(persisted.get("marks", {})).items()}
        state = ledger.reconcile()
        cash = float(state["computed_cash"])
        nav = cash + sum(
            float(state["computed_positions"][sec]) * marks.get(sec, 0.0)
            for sec in state["computed_positions"]
        )
        return {"nav": nav, "duplicate_orders": duplicate_orders, "ledger": state}

    @classmethod
    def resume(
        cls,
        state_path: str | Path,
        *,
        endpoint_kind: str = "paper",
    ) -> tuple[PaperBrokerAdapter, dict[str, Any]]:
        """Rebuild an adapter from a checkpoint and report the seam.

        Returns ``(adapter, seam_report)``. The adapter is reconstructed by
        replaying the durable orders/fills so NAV seam parity and the
        no-duplicate-order guarantee are re-checked on every resume.
        """
        refuse_live_endpoint(endpoint_kind)
        persisted_path = Path(state_path)
        probe = cls(state_path=persisted_path, endpoint_kind=endpoint_kind)
        persisted = probe._read_state()
        adapter = cls(
            state_path=persisted_path,
            endpoint_kind=str(persisted.get("endpoint_kind", endpoint_kind)),
            starting_cash=float(persisted.get("starting_cash", 0.0)),
            nav_tolerance=float(persisted.get("nav_tolerance", 1e-6)),
        )
        for raw in persisted.get("orders", []):
            order = _order_from_dict(raw)
            adapter.submit(order)
        for raw in persisted.get("fills", []):
            adapter.apply_fill(_fill_from_dict(raw))
        for order_id in _cancelled_ids(persisted):
            verdict = adapter.cancel(order_id)
            if verdict != "cancelled":
                raise OrderLedgerError(f"cannot replay cancel of {order_id}: {verdict}")
        for sec, price in dict(persisted.get("marks", {})).items():
            adapter.mark(str(sec), float(price))
        if persisted.get("kill_state") == "HALT_NEW_ORDERS":
            adapter.trip_kill("resumed_halt")
        adapter._persisted_nav = float(persisted.get("nav", 0.0))
        return adapter, adapter.seam_report()


def _cancelled_ids(persisted: dict[str, Any]) -> list[str]:
    raw = persisted.get("cancelled_order_ids", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise OrderLedgerError("cancelled_order_ids must be a list of non-empty strings")
    return list(raw)


__all__ = [
    "LiveEndpointRefused",
    "PaperBrokerAdapter",
    "SeamReport",
    "refuse_live_endpoint",
]
