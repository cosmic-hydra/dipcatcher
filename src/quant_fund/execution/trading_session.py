"""Operator paper-trading session. Live orders are refused.

This is the session a person can run: mark a simulated price, submit an
order, record a caller-supplied fill, cancel, and halt. It does not open a
socket, read broker credentials, or promote anything to live capital.

Live trading stays blocked until the five institutional conditions exist
(licensed point-in-time data, an authorized broker, a non-synthetic forward
record, venue measurements, and a signed live authorization). Those are
external evidence, not flags. :func:`live_readiness_report` is therefore
unconditionally ineligible, and constructing a session with a live endpoint
kind raises :class:`LiveEndpointRefused`.

Fills are simulated prints supplied by the caller. SYNTHETIC and SIMULATED
results here are accounting checks, never market evidence. The report carries
``live_pnl_claim=false`` and ``would_promote_live=false``.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from quant_fund.execution.order_recon import (
    Fill,
    LiveEndpointRefused,
    Order,
    Side,
    refuse_live_endpoint,
)
from quant_fund.paper.broker_adapter import PaperBrokerAdapter
from quant_fund.schemas.errors import KillSwitchActive

_QTY_EPS = 1e-9

_HARD_FAILURES = frozenset(
    {
        "rejected",
        "overfill",
        "orphan_fill",
        "security_mismatch",
        "fill_after_cancel",
        "out_of_order_fill",
        "unknown_order",
        "already_filled",
        "duplicate_cancel",
        "duplicate_fill",
        "error",
        "seam_mismatch",
    }
)

_ANOMALY_COUNTERS = (
    "duplicate_fills",
    "orphan_fills",
    "out_of_order_fills",
    "overfills",
    "security_mismatches",
    "fills_after_cancel",
)

_FIELDS: dict[str, frozenset[str]] = {
    "mark": frozenset({"security_id", "price"}),
    "submit": frozenset({"order_id", "security_id", "side", "quantity"}),
    "fill": frozenset({"fill_id", "order_id", "security_id", "quantity", "price", "fill_time"}),
    "cancel": frozenset({"order_id"}),
    "kill": frozenset({"reason"}),
    "clear_kill": frozenset(),
    "checkpoint": frozenset(),
}

_WORKING = frozenset({"open", "partial"})

# Import names that must never appear in this module. A broker client would
# be a live-trading path; the honesty contract forbids that.
_BANNED_IMPORT_ROOTS = frozenset(
    {
        "socket",
        "ssl",
        "urllib",
        "http",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
        "websocket",
    }
)


class SessionCommandError(ValueError):
    """A paper-session command is malformed."""


@dataclass(frozen=True)
class SessionLimits:
    """Hard caps for one simulated book. All quantities are strictly positive."""

    max_order_quantity: float = 1_000_000.0
    max_order_notional: float = 1_000_000.0
    max_position_notional: float = 5_000_000.0
    max_gross_notional: float = 5_000_000.0

    def __post_init__(self) -> None:
        for name in (
            "max_order_quantity",
            "max_order_notional",
            "max_position_notional",
            "max_gross_notional",
        ):
            _finite_positive(getattr(self, name), name)


def live_readiness_report() -> dict[str, Any]:
    """The five live-trading conditions. Always ineligible.

    There is no argument that flips this report. Code cannot purchase a
    vendor feed, authorize a broker, or sign a live-capital approval.
    """
    return {
        "schema": "live_readiness.v1",
        "eligible": False,
        "live_orders_enabled": False,
        "would_promote_live": False,
        "live_pnl_claim": False,
        "research_only": True,
        "executable_mode": "paper",
        "conditions": [
            {
                "id": 1,
                "name": "licensed_point_in_time_data",
                "status": "BLOCKED",
                "reason": "No entitled vendor feed is configured. Synthetic data is not a substitute.",
            },
            {
                "id": 2,
                "name": "authenticated_broker_reconciliation",
                "status": "BLOCKED",
                "reason": (
                    "No broker credentials. The paper ledger is the reference shape; "
                    "a live endpoint is refused."
                ),
            },
            {
                "id": 3,
                "name": "non_synthetic_forward_record",
                "status": "NOT_YET_COLLECTED",
                "reason": (
                    "forward_2026H2 is not yet collected. The 2025 holdout is spent. "
                    "The freeze is 2026-10-07; collection has not begun."
                ),
            },
            {
                "id": 4,
                "name": "venue_cost_liquidity_borrow_financing",
                "status": "BLOCKED",
                "reason": "No dated venue measurements. Modeled costs are not venue evidence.",
            },
            {
                "id": 5,
                "name": "signed_promotion_and_live_authorization",
                "status": "ABSENT",
                "reason": "No written live authorization. Promotion stays fail-closed.",
            },
        ],
    }


def format_readiness(report: Mapping[str, Any] | None = None) -> str:
    """Plain-text readiness for the chat shell. States the block, not a claim."""
    body = live_readiness_report() if report is None else report
    lines = [
        "Live orders: disabled",
        "Eligible for live capital: no",
        "Executable mode: paper session only (simulated fills, no broker connection)",
        "",
        "Conditions:",
    ]
    for condition in body["conditions"]:
        lines.append(
            f"{condition['id']}. {condition['name']} — {condition['status']}: {condition['reason']}"
        )
    lines.extend(
        [
            "",
            "would_promote_live=false",
            "live_pnl_claim=false",
            "research_only=true",
            "",
            "Paper session (shell):",
            "  dipcatcher trade session --state <path> --commands <jsonl>",
            "Ops: mark, submit, fill, cancel, kill, clear_kill, checkpoint",
            "Prices are caller-supplied. Short sales are refused.",
        ]
    )
    return "\n".join(lines)


def module_network_imports(source: str) -> list[str]:
    """Import roots in ``source`` that would open a broker or HTTP client."""
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            root = name.split(".", 1)[0]
            if root in _BANNED_IMPORT_ROOTS:
                found.append(name)
    return found


class PaperTradingSession:
    """Durable simulated book with pre-trade caps and a kill switch."""

    def __init__(
        self,
        *,
        state_path: str | Path,
        starting_cash: float = 1_000_000.0,
        limits: SessionLimits | None = None,
        endpoint_kind: str = "paper",
        adapter: PaperBrokerAdapter | None = None,
    ) -> None:
        kind = refuse_live_endpoint(endpoint_kind)
        self.limits = limits if limits is not None else SessionLimits()
        if adapter is None:
            self._broker = PaperBrokerAdapter(
                state_path=state_path,
                endpoint_kind=kind,
                starting_cash=_finite_positive(starting_cash, "starting_cash"),
            )
        else:
            if adapter.endpoint_kind != kind:
                raise LiveEndpointRefused(
                    f"session endpoint {kind!r} does not match the resumed book"
                )
            self._broker = adapter

    @classmethod
    def open(
        cls,
        state_path: str | Path,
        *,
        resume: bool,
        starting_cash: float,
        limits: SessionLimits | None = None,
        endpoint_kind: str = "paper",
    ) -> PaperTradingSession:
        """Open a new book, or resume one whose seam still matches."""
        kind = refuse_live_endpoint(endpoint_kind)
        path = Path(state_path)
        if resume:
            if not path.is_file():
                raise SessionCommandError(f"no paper session at {path}")
            adapter, seam = PaperBrokerAdapter.resume(path, endpoint_kind=kind)
            if not seam["ok"]:
                errors = ", ".join(seam["errors"])
                raise SessionCommandError(f"paper session seam failed: {errors}")
            return cls(
                state_path=path,
                starting_cash=adapter.starting_cash,
                limits=limits,
                endpoint_kind=adapter.endpoint_kind,
                adapter=adapter,
            )
        if path.exists():
            raise SessionCommandError(
                f"paper session already exists at {path}; resume to continue it"
            )
        return cls(
            state_path=path,
            starting_cash=starting_cash,
            limits=limits,
            endpoint_kind=kind,
        )

    def apply_command(self, command: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one command. Malformed input raises; a risk refusal is a verdict."""
        if not isinstance(command, Mapping):
            raise SessionCommandError("command must be an object")
        op = command.get("op")
        if not isinstance(op, str) or op not in _FIELDS:
            raise SessionCommandError(f"unknown op {op!r}")
        _require_fields(op, command)
        handler = getattr(self, f"_op_{op}")
        return cast(dict[str, Any], handler(command))

    def checkpoint(self) -> dict[str, Any]:
        return self._broker.checkpoint()

    def reconcile_view(self) -> dict[str, Any]:
        report = self._broker.reconcile()
        report["marks"] = self._broker.marks
        report["live_orders_enabled"] = False
        report["would_promote_live"] = False
        return report

    def _op_mark(self, command: Mapping[str, Any]) -> dict[str, Any]:
        security_id = _token(command["security_id"], "security_id")
        price = _finite_positive(command["price"], "price")
        self._broker.mark(security_id, price)
        return self._commit(_step("mark", "marked", security_id=security_id))

    def _op_submit(self, command: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _token(command["order_id"], "order_id")
        security_id = _token(command["security_id"], "security_id")
        side = _side(command["side"])
        quantity = _finite_positive(command["quantity"], "quantity")
        refusal = self._pretrade(security_id, side, quantity)
        if refusal is not None:
            return _step("submit", "rejected", reason=refusal, order_id=order_id)
        order = Order(
            order_id=order_id,
            security_id=security_id,
            side=side,
            quantity=quantity,
        )
        try:
            verdict = self._broker.submit(order)
        except KillSwitchActive as exc:
            return _step("submit", "rejected", reason=str(exc), order_id=order_id)
        step = _step(
            "submit",
            verdict,
            order_id=order_id,
            security_id=security_id,
            side=side,
            quantity=quantity,
        )
        if verdict == "accepted":
            return self._commit(step)
        return step

    def _op_fill(self, command: Mapping[str, Any]) -> dict[str, Any]:
        fill = Fill(
            fill_id=_token(command["fill_id"], "fill_id"),
            order_id=_token(command["order_id"], "order_id"),
            security_id=_token(command["security_id"], "security_id"),
            quantity=_finite_positive(command["quantity"], "quantity"),
            price=_finite_positive(command["price"], "price"),
            fill_time=_parse_time(command["fill_time"]),
        )
        refusal = self._fill_refusal(fill)
        if refusal is not None:
            return _step("fill", "rejected", reason=refusal, order_id=fill.order_id)
        verdict = self._broker.apply_fill(fill)
        step = _step("fill", verdict, order_id=fill.order_id, fill_id=fill.fill_id)
        if verdict in {"accepted", "out_of_order_fill"}:
            self._broker.mark(fill.security_id, fill.price)
            return self._commit(step)
        return step

    def _op_cancel(self, command: Mapping[str, Any]) -> dict[str, Any]:
        order_id = _token(command["order_id"], "order_id")
        verdict = self._broker.cancel(order_id)
        step = _step("cancel", verdict, order_id=order_id)
        if verdict == "cancelled":
            return self._commit(step)
        return step

    def _op_kill(self, command: Mapping[str, Any]) -> dict[str, Any]:
        reason = _token(command["reason"], "reason")
        state = self._broker.trip_kill(reason)
        return self._commit(_step("kill", "killed", kill_state=state))

    def _op_clear_kill(self, command: Mapping[str, Any]) -> dict[str, Any]:
        del command
        state = self._broker.clear_kill()
        return self._commit(_step("clear_kill", "cleared", kill_state=state))

    def _op_checkpoint(self, command: Mapping[str, Any]) -> dict[str, Any]:
        del command
        return self._commit(_step("checkpoint", "checkpointed"))

    def _commit(self, step: dict[str, Any]) -> dict[str, Any]:
        seam = self._broker.checkpoint()
        step["nav_seam_parity"] = bool(seam["nav_seam_parity"])
        if not seam["ok"]:
            step["errors"] = list(seam["errors"])
            step["verdict"] = "seam_mismatch"
        return step

    def _pretrade(self, security_id: str, side: Side, quantity: float) -> str | None:
        price = _marked_price(self._broker.marks, security_id)
        if price is None:
            return "missing_mark"
        capped = _order_cap(quantity, price, self.limits)
        if capped is not None:
            return capped
        return _exposure_refusal(
            self._broker.reconcile(),
            self._broker.marks,
            self.limits,
            security_id,
            side,
            quantity,
            price,
        )

    def _fill_refusal(self, fill: Fill) -> str | None:
        """Refuse a buy fill the simulated cash cannot pay. Anomalies reach the ledger."""
        book = self._broker.reconcile()
        row = next((item for item in book["orders"] if item["order_id"] == fill.order_id), None)
        if row is None or row["security_id"] != fill.security_id or row["status"] == "cancelled":
            return None
        remaining = float(row["quantity"]) - float(row["filled_qty"])
        if fill.quantity > remaining + _QTY_EPS:
            return None
        if (
            row["side"] == "BUY"
            and fill.quantity * fill.price > float(book["computed_cash"]) + _QTY_EPS
        ):
            return "insufficient_cash"
        return None


def load_commands(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL command script. Blank lines are ignored."""
    commands: list[dict[str, Any]] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SessionCommandError(f"line {lineno}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise SessionCommandError(f"line {lineno}: command must be an object")
        commands.append(payload)
    return commands


def run_session(
    *,
    state_path: str | Path,
    commands: Sequence[Mapping[str, Any]],
    resume: bool = False,
    starting_cash: float = 1_000_000.0,
    limits: SessionLimits | None = None,
) -> dict[str, Any]:
    """Run a command list against a paper book and return the session report."""
    session = PaperTradingSession.open(
        state_path,
        resume=resume,
        starting_cash=starting_cash,
        limits=limits,
    )
    steps: list[dict[str, Any]] = []
    for command in commands:
        try:
            steps.append(session.apply_command(command))
        except SessionCommandError as exc:
            op = command.get("op") if isinstance(command, Mapping) else ""
            steps.append(_step(str(op), "error", reason=str(exc)))
            break
    seam = session.checkpoint()
    report = session.reconcile_view()
    consistent = _consistent(steps, report) and bool(seam["ok"])
    return {
        "schema": "paper_trading_session.v1",
        "consistent": consistent,
        "nav_seam_parity": bool(seam["nav_seam_parity"]),
        "steps": steps,
        "reconciliation": report,
        "readiness": live_readiness_report(),
        "executable_mode": "paper",
        "live_orders_enabled": False,
        "would_promote_live": False,
        "live_pnl_claim": False,
        "research_only": True,
        "data_label": "SIMULATED",
    }


def _consistent(steps: list[dict[str, Any]], report: Mapping[str, Any]) -> bool:
    if any(step["verdict"] in _HARD_FAILURES for step in steps):
        return False
    if report["drift"]["alarm"]:
        return False
    counters = report["counters"]
    return not any(counters[name] for name in _ANOMALY_COUNTERS)


def _step(op: str, verdict: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "op": op,
        "verdict": verdict,
        "research_only": True,
        "live_pnl_claim": False,
        "live_orders_enabled": False,
    }
    body.update(extra)
    return body


def _require_fields(op: str, command: Mapping[str, Any]) -> None:
    present = set(command) - {"op"}
    allowed = _FIELDS[op]
    unknown = sorted(present - allowed)
    missing = sorted(allowed - present)
    if unknown:
        raise SessionCommandError(f"unknown fields for {op}: {', '.join(unknown)}")
    if missing:
        raise SessionCommandError(f"missing fields for {op}: {', '.join(missing)}")


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SessionCommandError(f"{label} must be a non-empty string")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionCommandError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise SessionCommandError(f"{label} must be finite and strictly positive")
    return number


def _side(value: Any) -> Side:
    if not isinstance(value, str):
        raise SessionCommandError("side must be BUY or SELL")
    normalized = value.strip().upper()
    if normalized not in {"BUY", "SELL"}:
        raise SessionCommandError("side must be BUY or SELL")
    return cast(Side, normalized)


def _parse_time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise SessionCommandError("fill_time must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise SessionCommandError("fill_time must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise SessionCommandError("fill_time must include a timezone")
    return parsed


def _marked_price(marks: Mapping[str, float], security_id: str) -> float | None:
    price = marks.get(security_id)
    if price is None or not math.isfinite(price) or price <= 0.0:
        return None
    return float(price)


def _order_cap(quantity: float, price: float, limits: SessionLimits) -> str | None:
    if quantity > limits.max_order_quantity:
        return "max_order_quantity"
    if quantity * price > limits.max_order_notional:
        return "max_order_notional"
    return None


def _exposure_refusal(
    book: Mapping[str, Any],
    marks: Mapping[str, float],
    limits: SessionLimits,
    security_id: str,
    side: Side,
    quantity: float,
    price: float,
) -> str | None:
    try:
        working = _working_deltas(book["orders"], marks)
    except SessionCommandError as exc:
        return str(exc)
    position = float(book["computed_positions"].get(security_id, 0.0))
    signed = quantity if side == "BUY" else -quantity
    projected = position + working.get(security_id, 0.0) + signed
    refusal = _book_cap(side, projected, price, quantity, book, marks, limits)
    if refusal is not None:
        return refusal
    return _gross_cap(book, working, marks, limits, security_id, projected)


def _book_cap(
    side: Side,
    projected: float,
    price: float,
    quantity: float,
    book: Mapping[str, Any],
    marks: Mapping[str, float],
    limits: SessionLimits,
) -> str | None:
    if side == "SELL" and projected < -_QTY_EPS:
        return "short_not_authorized"
    if abs(projected) * price > limits.max_position_notional:
        return "max_position_notional"
    if side != "BUY":
        return None
    reserved = _reserved_buy_notional(book["orders"], marks)
    if reserved + quantity * price > float(book["computed_cash"]) + _QTY_EPS:
        return "insufficient_cash"
    return None


def _gross_cap(
    book: Mapping[str, Any],
    working: Mapping[str, float],
    marks: Mapping[str, float],
    limits: SessionLimits,
    security_id: str,
    projected: float,
) -> str | None:
    try:
        gross = _projected_gross(book["computed_positions"], working, marks, security_id, projected)
    except SessionCommandError as exc:
        return str(exc)
    if gross > limits.max_gross_notional:
        return "max_gross_notional"
    return None


def _working_deltas(orders: list[dict[str, Any]], marks: Mapping[str, float]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for row in orders:
        if row["status"] not in _WORKING:
            continue
        price = marks.get(row["security_id"])
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            raise SessionCommandError(f"unmarked working order in {row['security_id']}")
        residual = float(row["quantity"]) - float(row["filled_qty"])
        signed = residual if row["side"] == "BUY" else -residual
        security_id = str(row["security_id"])
        delta[security_id] = delta.get(security_id, 0.0) + signed
    return delta


def _reserved_buy_notional(orders: list[dict[str, Any]], marks: Mapping[str, float]) -> float:
    total = 0.0
    for row in orders:
        if row["side"] != "BUY" or row["status"] not in _WORKING:
            continue
        price = float(marks[row["security_id"]])
        residual = float(row["quantity"]) - float(row["filled_qty"])
        total += residual * price
    return total


def _projected_gross(
    positions: Mapping[str, float],
    working: Mapping[str, float],
    marks: Mapping[str, float],
    security_id: str,
    projected: float,
) -> float:
    securities = set(positions) | set(working) | {security_id}
    gross = 0.0
    for security in securities:
        quantity = (
            projected
            if security == security_id
            else (float(positions.get(security, 0.0)) + float(working.get(security, 0.0)))
        )
        if abs(quantity) <= _QTY_EPS:
            continue
        price = marks.get(security)
        if price is None or not math.isfinite(float(price)) or float(price) <= 0.0:
            raise SessionCommandError(f"unmarked position in {security}")
        gross += abs(quantity) * float(price)
    return gross


__all__ = [
    "PaperTradingSession",
    "SessionCommandError",
    "SessionLimits",
    "format_readiness",
    "live_readiness_report",
    "load_commands",
    "module_network_imports",
    "run_session",
]
