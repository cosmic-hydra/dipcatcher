"""Authenticated order/fill reconciliation for the PAPER/SIMULATED broker only.

This module implements the machine-checkable half of readiness **condition 2**:
an order-id <-> fill-id ledger with idempotent replay and detection of
duplicate, missing, and out-of-order fills, plus position/cash drift alarms and
a kill switch. It is the reconciliation surface a live broker adapter would
have to satisfy before condition 2 could be claimed.

**Hard guarantee of no live orders.** Configuring a live / production / real
endpoint RAISES :class:`LiveEndpointRefused` at construction — enforced in
code, not prose. Only ``paper`` and ``simulated`` endpoint kinds exist here.
Every report carries ``live_pnl_claim=False`` and ``research_only=True``; this
lane makes **no** live-connectivity or live-profitability claim and touches no
live capital.

SYNTHETIC fills in tests are correctness fixtures only, never market evidence.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from quant_fund.schemas.errors import ConfigError, KillSwitchActive

#: Float slack for filled-quantity comparisons. Not a price tolerance.
_QTY_EPS = 1e-9

#: The only endpoint kinds this paper/simulated adapter accepts.
ALLOWED_ENDPOINT_KINDS: frozenset[str] = frozenset({"paper", "simulated"})

#: Endpoint-kind tokens that MUST be refused (hard no-live guarantee).
_LIVE_TOKENS: frozenset[str] = frozenset(
    {"live", "prod", "production", "real", "broker", "ibkr", "interactive", "real_money"}
)

KillState = Literal["ENABLED", "HALT_NEW_ORDERS"]
Side = Literal["BUY", "SELL"]


class LiveEndpointRefused(ConfigError):
    """Raised when a live / production / real-money endpoint is configured."""


class OrderLedgerError(ConfigError):
    """Malformed order/fill input to the reconciliation ledger."""


def refuse_live_endpoint(endpoint_kind: str) -> str:
    """Return the normalized endpoint kind or RAISE if it is live.

    This is the hard-coded no-live guarantee: any live, production, real-money,
    or broker endpoint kind is refused here so no code path in this lane can
    ever send a live order.
    """
    if not isinstance(endpoint_kind, str) or not endpoint_kind.strip():
        raise OrderLedgerError("endpoint_kind must be a non-empty string")
    normalized = endpoint_kind.strip().lower()
    tokens = set(normalized.replace("-", "_").replace(" ", "_").split("_")) | {normalized}
    if tokens & _LIVE_TOKENS:
        raise LiveEndpointRefused(
            f"live endpoint kind {endpoint_kind!r} is refused; this adapter is "
            "PAPER/SIMULATED only and sends no live orders"
        )
    if normalized not in ALLOWED_ENDPOINT_KINDS:
        raise LiveEndpointRefused(
            f"unknown endpoint kind {endpoint_kind!r}; only paper/simulated are allowed"
        )
    return normalized


@dataclass(frozen=True)
class Order:
    order_id: str
    security_id: str
    side: Side
    quantity: float
    submitted_time: datetime | None = None
    expects_fill: bool = True


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    security_id: str
    quantity: float
    price: float
    fill_time: datetime
    seq: int | None = None


@dataclass
class _OrderState:
    order: Order
    filled_qty: float = 0.0
    fill_ids: list[str] = field(default_factory=list)
    last_fill_time: datetime | None = None
    last_seq: int | None = None
    cancelled: bool = False


def _signed_qty(side: Side, quantity: float) -> float:
    return quantity if side == "BUY" else -quantity


class OrderFillLedger:
    """Order-id <-> fill-id ledger with reconciliation and a kill switch.

    ``submit``/``record_fill`` are idempotent: replaying an already-seen
    ``order_id``/``fill_id`` is detected (counted as a duplicate) and does not
    double-count state. All ordering/consistency verdicts surface on
    :meth:`reconcile`.
    """

    def __init__(
        self,
        *,
        endpoint_kind: str = "paper",
        starting_cash: float = 0.0,
        cash_tolerance: float = 1e-6,
        position_tolerance: float = 1e-6,
    ) -> None:
        self.endpoint_kind = refuse_live_endpoint(endpoint_kind)
        self._cash_tolerance = float(cash_tolerance)
        self._position_tolerance = float(position_tolerance)
        self._orders: dict[str, _OrderState] = {}
        self._fills: dict[str, Fill] = {}
        self._computed_cash = float(starting_cash)
        self._computed_positions: dict[str, float] = {}
        self._declared_cash: float | None = None
        self._declared_positions: dict[str, float] | None = None
        self._kill_state: KillState = "ENABLED"
        self._kill_reason: str | None = None
        self._counters: dict[str, int] = {
            "orders": 0,
            "duplicate_orders": 0,
            "cancels": 0,
            "fills": 0,
            "duplicate_fills": 0,
            "orphan_fills": 0,
            "out_of_order_fills": 0,
            "overfills": 0,
            "security_mismatches": 0,
            "fills_after_cancel": 0,
        }

    # -- kill switch --

    def trip_kill(self, reason: str) -> KillState:
        self._kill_state = "HALT_NEW_ORDERS"
        self._kill_reason = reason or "manual"
        return self._kill_state

    def clear_kill(self) -> KillState:
        self._kill_state = "ENABLED"
        self._kill_reason = None
        return self._kill_state

    @property
    def kill_state(self) -> KillState:
        return self._kill_state

    # -- order intake --

    def submit(self, order: Order) -> str:
        """Record an order idempotently; refuse new orders after a kill trip."""
        if self._kill_state != "ENABLED":
            raise KillSwitchActive(f"kill switch {self._kill_state}: {self._kill_reason}")
        if order.order_id in self._orders:
            self._counters["duplicate_orders"] += 1
            return "duplicate_order"
        self._orders[order.order_id] = _OrderState(order=order)
        self._counters["orders"] += 1
        return "accepted"

    def cancel(self, order_id: str) -> str:
        """Cancel a working order. Allowed while the kill switch is tripped.

        Returns ``cancelled``, ``duplicate_cancel``, ``already_filled``, or
        ``unknown_order``. A cancel drops the residual from the missing-fill
        set. Fills that arrive afterward are recorded and not applied.
        """
        if not isinstance(order_id, str) or not order_id.strip():
            raise OrderLedgerError("order_id must be a non-empty string")
        state = self._orders.get(order_id)
        if state is None:
            return "unknown_order"
        if state.cancelled:
            return "duplicate_cancel"
        if state.filled_qty + _QTY_EPS >= state.order.quantity:
            return "already_filled"
        state.cancelled = True
        self._counters["cancels"] += 1
        return "cancelled"

    # -- fill intake --

    def record_fill(self, fill: Fill) -> str:
        """Record a fill idempotently, classifying every anomaly.

        Returns ``accepted``, ``duplicate_fill``, ``orphan_fill``,
        ``out_of_order_fill``, ``overfill``, ``security_mismatch``, or
        ``fill_after_cancel``. Duplicates are never double-counted. Overfills,
        security mismatches, and fills after cancel are stored for replay and
        are not applied to positions or cash.
        """
        _validate_fill(fill)
        if fill.fill_id in self._fills:
            self._counters["duplicate_fills"] += 1
            return "duplicate_fill"
        self._fills[fill.fill_id] = fill
        state = self._orders.get(fill.order_id)
        if state is None:
            self._counters["orphan_fills"] += 1
            return "orphan_fill"
        blocked = self._unapplied_fill(state, fill)
        if blocked is not None:
            counter, verdict = blocked
            self._counters[counter] += 1
            return verdict
        verdict = "accepted"
        if self._is_out_of_order(state, fill):
            self._counters["out_of_order_fills"] += 1
            verdict = "out_of_order_fill"
        self._apply_fill(state, fill)
        self._counters["fills"] += 1
        return verdict

    def _unapplied_fill(self, state: _OrderState, fill: Fill) -> tuple[str, str] | None:
        """Anomaly that must not move positions or cash, if any."""
        if fill.security_id != state.order.security_id:
            return ("security_mismatches", "security_mismatch")
        if state.cancelled:
            return ("fills_after_cancel", "fill_after_cancel")
        remaining = state.order.quantity - state.filled_qty
        if abs(fill.quantity) > remaining + _QTY_EPS:
            return ("overfills", "overfill")
        return None

    def _is_out_of_order(self, state: _OrderState, fill: Fill) -> bool:
        if state.last_fill_time is not None and fill.fill_time < state.last_fill_time:
            return True
        return state.last_seq is not None and fill.seq is not None and fill.seq < state.last_seq

    def _apply_fill(self, state: _OrderState, fill: Fill) -> None:
        state.filled_qty += abs(fill.quantity)
        state.fill_ids.append(fill.fill_id)
        state.last_fill_time = max(fill.fill_time, state.last_fill_time or fill.fill_time)
        if fill.seq is not None:
            state.last_seq = max(fill.seq, state.last_seq or fill.seq)
        signed = _signed_qty(state.order.side, abs(fill.quantity))
        self._computed_positions[fill.security_id] = (
            self._computed_positions.get(fill.security_id, 0.0) + signed
        )
        self._computed_cash -= signed * fill.price

    # -- declaration (broker-reported) state, for drift alarms --

    def declare_state(self, *, cash: float, positions: dict[str, float]) -> None:
        self._declared_cash = float(cash)
        self._declared_positions = {str(k): float(v) for k, v in positions.items()}

    # -- reconciliation --

    def _missing_fills(self) -> list[str]:
        return [
            oid
            for oid, st in self._orders.items()
            if st.order.expects_fill and not st.cancelled and not st.fill_ids
        ]

    def _order_status(self, state: _OrderState) -> str:
        if state.cancelled:
            return "cancelled"
        if state.filled_qty <= _QTY_EPS:
            return "open"
        if state.filled_qty + _QTY_EPS < state.order.quantity:
            return "partial"
        return "filled"

    def _order_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "order_id": oid,
                "security_id": state.order.security_id,
                "side": state.order.side,
                "quantity": state.order.quantity,
                "filled_qty": state.filled_qty,
                "status": self._order_status(state),
            }
            for oid, state in self._orders.items()
        ]

    def _drift_report(self) -> dict[str, Any]:
        if self._declared_cash is None or self._declared_positions is None:
            return {"declared": False, "cash_drift": None, "position_drift": {}, "alarm": False}
        cash_drift = abs(self._computed_cash - self._declared_cash)
        all_secs = set(self._computed_positions) | set(self._declared_positions)
        pos_drift = {
            sec: abs(
                self._computed_positions.get(sec, 0.0) - self._declared_positions.get(sec, 0.0)
            )
            for sec in sorted(all_secs)
        }
        alarm = cash_drift > self._cash_tolerance or any(
            d > self._position_tolerance for d in pos_drift.values()
        )
        return {
            "declared": True,
            "cash_drift": cash_drift,
            "position_drift": pos_drift,
            "alarm": bool(alarm),
        }

    def reconcile(self, *, expected_fill_ids: Iterable[str] = ()) -> dict[str, Any]:
        """Return a fail-loud reconciliation report over the whole ledger.

        Surfaces duplicate/missing/out-of-order fills, orphan fills, and a
        position/cash drift alarm. ``expected_fill_ids`` additionally flags any
        externally-expected fill id that never arrived as a missing fill.
        """
        expected = set(expected_fill_ids)
        absent_expected = sorted(expected - set(self._fills))
        missing = sorted(set(self._missing_fills()) | set(absent_expected))
        drift = self._drift_report()
        clean = not (
            self._counters["duplicate_fills"]
            or self._counters["orphan_fills"]
            or self._counters["out_of_order_fills"]
            or self._counters["overfills"]
            or self._counters["security_mismatches"]
            or self._counters["fills_after_cancel"]
            or missing
            or drift["alarm"]
        )
        return {
            "ok": clean,
            "endpoint_kind": self.endpoint_kind,
            "kill_state": self._kill_state,
            "counters": dict(self._counters),
            "missing_fills": missing,
            "orders": self._order_rows(),
            "drift": drift,
            "computed_cash": self._computed_cash,
            "computed_positions": dict(self._computed_positions),
            "n_orders": len(self._orders),
            "n_fills": len(self._fills),
            "research_only": True,
            "live_pnl_claim": False,
        }


def _validate_fill(fill: Fill) -> None:
    """Reject a fill that cannot be an execution print."""
    if isinstance(fill.quantity, bool) or not isinstance(fill.quantity, (int, float)):
        raise OrderLedgerError("fill quantity must be a finite number")
    if isinstance(fill.price, bool) or not isinstance(fill.price, (int, float)):
        raise OrderLedgerError("fill price must be a finite number")
    if not math.isfinite(float(fill.quantity)) or float(fill.quantity) <= 0.0:
        raise OrderLedgerError("fill quantity must be finite and strictly positive")
    if not math.isfinite(float(fill.price)) or float(fill.price) <= 0.0:
        raise OrderLedgerError("fill price must be finite and strictly positive")
    if not fill.fill_id or not fill.order_id or not fill.security_id:
        raise OrderLedgerError("fill_id, order_id, and security_id must be non-empty")


__all__ = [
    "ALLOWED_ENDPOINT_KINDS",
    "Fill",
    "KillState",
    "LiveEndpointRefused",
    "Order",
    "OrderFillLedger",
    "OrderLedgerError",
    "refuse_live_endpoint",
]
