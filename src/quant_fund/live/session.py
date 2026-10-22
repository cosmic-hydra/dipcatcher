"""Armed order session.

Orders reach a broker only after the arm file matches this config and the
pre-trade engine allows them. Reports are broker accounting. ``live_pnl_claim``
is always false; nothing here is a research receipt.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import time
import urllib.error
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from quant_fund.live.brokers import (
    AlpacaBroker,
    Broker,
    BrokerAccount,
    BrokerAck,
    BrokerFill,
    LiveOrder,
    LoopbackBroker,
    alpaca_credentials,
    map_broker_status,
)
from quant_fund.live.config import LiveConfig, load_gateway_pretrade
from quant_fund.live.errors import BrokerRejected, LiveGateError
from quant_fund.live.gateway_arm import (
    arm_session,
    disarm_session,
    load_hmac_key,
    read_arm,
    zero_audit_head,
)
from quant_fund.live.ledger import LedgerFill, LiveLedger, ledger_fill_from_dict
from quant_fund.live.risk_memory import export_pretrade, restore_pretrade
from quant_fund.pretrade.codes import KIND_CANCEL, KIND_ORDER
from quant_fund.pretrade.config import PretradeConfig
from quant_fund.pretrade.engine import OrderView, PretradeEngine
from quant_fund.pretrade.shadow import business_session_id

_SYMBOL = re.compile(r"^[A-Z][A-Z0-9.]{0,9}$")
_CLIENT = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_ORDER_STATUSES = frozenset(
    {
        "inflight",
        "accepted",
        "filled",
        "partially_filled",
        "canceled",
        "rejected",
        "not_acknowledged",
        "expired",
        "replaced",
        "done_for_day",
        "pending_cancel",
        "pending_replace",
        "suspended",
        "unmapped",
    }
)
_TERMINAL = frozenset(
    {
        "filled",
        "canceled",
        "rejected",
        "not_acknowledged",
        "expired",
        "replaced",
        "done_for_day",
        "suspended",
    }
)
_WORKING = frozenset(
    {
        "inflight",
        "accepted",
        "partially_filled",
        "pending_cancel",
        "pending_replace",
        "unmapped",
    }
)
_STATE_SCHEMA = "live_session.v1"


def _actor(actor: str) -> str:
    cleaned = actor.strip()
    if not cleaned or len(cleaned) > 80 or any(ord(ch) < 32 for ch in cleaned):
        raise LiveGateError("actor must be 1..80 visible characters")
    return cleaned


def _reason(reason: str) -> str:
    cleaned = reason.strip()
    if not cleaned or len(cleaned) > 200:
        raise LiveGateError("reason must be 1..200 characters")
    return cleaned


def _symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not _SYMBOL.fullmatch(cleaned):
        raise LiveGateError("symbol must be 1..10 letters, digits, or dots")
    return cleaned


def _side(side: str) -> str:
    cleaned = side.strip().lower()
    if cleaned not in {"buy", "sell"}:
        raise LiveGateError("side must be buy or sell")
    return cleaned


def _positive(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise LiveGateError(f"{label} must be finite and positive")
    return number


def _new_client_id() -> str:
    return "dc" + secrets.token_hex(8)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LiveGateError(f"{path.name} is not a JSON object")
    return raw


class LiveSession:
    """One state directory, one venue, one pre-trade snapshot."""

    def __init__(
        self,
        config: LiveConfig,
        *,
        state_dir: Path,
        pretrade: PretradeConfig,
        pretrade_sha256: str,
        broker: Broker,
        now_ns: Callable[[], int],
    ) -> None:
        self.config = config
        self.venue = config.venue
        self.state_dir = state_dir
        self.pretrade = pretrade
        self.pretrade_sha256 = pretrade_sha256
        self.broker = broker
        self._now = now_ns
        self.orders: list[dict[str, Any]] = []
        self.marks: dict[str, float] = {}
        self.kill_state: dict[str, Any] = {"latched": False, "actor": "", "reason": ""}
        self.ledger = LiveLedger(
            baseline_cash=float(config.starting_cash),
            cash_tolerance=config.cash_tolerance,
            position_tolerance=config.position_tolerance,
        )
        self._baseline_from_broker = config.venue == "loopback"
        self.arm_path = state_dir / "arm.json"
        self.state_path = state_dir / "state.json"
        self.audit_path = state_dir / "audit.jsonl"
        self._arm: dict[str, Any] | None = None
        self.audit_head = zero_audit_head()
        self._sids: dict[str, int] = {}
        self._engine_noted: set[str] = set()
        self._pretrade_blob: object = None
        self._model_settled = float(config.starting_cash)
        self._disk_token: str | None = None
        self.engine = PretradeEngine(
            pretrade,
            hmac_key=_hmac_for_engine(state_dir),
            initial_nav=float(config.starting_cash),
            initial_cash=float(config.starting_cash),
            ts_ns=0,
        )
        # State and the audit head are read only after the interprocess lock
        # is held. A second CLI process cannot overwrite this snapshot.
        with self._lock():
            pass

    def status(self) -> dict[str, Any]:
        """Local snapshot. This does not place an order."""
        with self._lock():
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        return {
            "ok": True,
            "venue": self.venue,
            "calendar": self.config.calendar,
            "armed": self._arm is not None,
            "actor": None if self._arm is None else self._arm.get("actor"),
            "kill_latched": bool(self.kill_state["latched"]),
            "credentials_present": _credentials_present(self.broker),
            "orders": len(self.orders),
            "computed_cash": self.ledger.computed_cash,
            "computed_positions": self.ledger.computed_positions,
            "marks": dict(self.marks),
            "live_pnl_claim": False,
            "research_evidence": False,
        }

    def arm(self, *, actor: str, confirm_live_capital: str = "") -> dict[str, Any]:
        with self._lock():
            payload = arm_session(
                self.arm_path,
                venue=self.venue,
                actor=actor,
                pretrade_sha256=self.pretrade_sha256,
                confirm_live_capital=confirm_live_capital,
            )
            self._arm = payload
            self._audit("arm", {"actor": payload["actor"], "venue": self.venue})
            return {
                "ok": True,
                "armed": True,
                "venue": self.venue,
                "actor": payload["actor"],
                "live_capital_confirmed": payload["live_capital_confirmed"],
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def disarm(self, *, actor: str) -> dict[str, Any]:
        with self._lock():
            result = disarm_session(self.arm_path, actor=actor)
            self._arm = None
            self._audit("disarm", {"actor": result["actor"]})
            return result

    def kill(self, *, actor: str, reason: str, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            who = _actor(actor)
            why = _reason(reason)
            ts = self._clock(at_ns)
            self.engine.trip(who, why, ts)
            self.kill_state = {"latched": True, "actor": who, "reason": why}
            self._persist()
            self._audit("kill", {"actor": who, "reason": why})
            return {
                "ok": True,
                "kill_latched": True,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def reset_kill(self, *, actor: str, reason: str, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            who = _actor(actor)
            why = _reason(reason)
            ts = self._clock(at_ns)
            self.engine.reset(who, why, ts)
            self.kill_state = {"latched": False, "actor": who, "reason": why}
            self._persist()
            self._audit("reset", {"actor": who, "reason": why})
            return {
                "ok": True,
                "kill_latched": False,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def mark(self, *, symbol: str, price: float, actor: str) -> dict[str, Any]:
        with self._lock():
            self._require_trading()
            _actor(actor)
            cleaned = _symbol(symbol)
            number = _positive(price, "mark")
            self.marks[cleaned] = number
            self.broker.set_mark(cleaned, number)
            self._persist()
            return {
                "ok": True,
                "symbol": cleaned,
                "mark": number,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def submit(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        mark: float,
        actor: str,
        client_order_id: str | None = None,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            self._recover_inflight()
            return self._submit_locked(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=limit_price,
                mark=mark,
                actor=actor,
                client_order_id=client_order_id,
                at_ns=at_ns,
            )

    def cancel(
        self,
        *,
        client_order_id: str,
        actor: str,
        at_ns: int | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            self._recover_inflight()
            self._require_trading()
            _actor(actor)
            ts = self._clock(at_ns)
            row = self._find(client_order_id)
            if row is None:
                raise LiveGateError("order is not in this session")
            if row["status"] in _TERMINAL:
                return _public_order(row, venue=self.venue, replayed=True)
            self._sync_account(ts)
            decision = self.engine.decide(
                self._view(row, float(row["limit_price"]), ts, KIND_CANCEL),
                apply=False,
            )
            if not decision.allowed:
                denied = _denied(
                    reasons=decision.reasons,
                    symbol=str(row["symbol"]),
                    side=str(row["side"]),
                    quantity=float(row["quantity"]),
                    limit_price=float(row["limit_price"]),
                    client_order_id=str(row["client_order_id"]),
                    venue=self.venue,
                )
                self._audit(
                    "deny",
                    {"client_order_id": row["client_order_id"], "reasons": list(decision.reasons)},
                )
                return denied
            try:
                ack = self.broker.cancel(str(row["client_order_id"]), str(row["broker_order_id"]))
            except BrokerRejected as exc:
                row["reason"] = str(exc)
                self._persist()
                return _public_order(row, venue=self.venue, replayed=False)
            self._apply_ack(row, ack, ts)
            self._persist()
            self._audit(
                "cancel", {"client_order_id": row["client_order_id"], "status": row["status"]}
            )
            return _public_order(row, venue=self.venue, replayed=False)

    def reconcile(self, *, at_ns: int | None = None) -> dict[str, Any]:
        with self._lock():
            self._recover_inflight()
            self._require_trading()
            ts = self._clock(at_ns)
            self._capture_baseline()
            mapping = {
                str(row["broker_order_id"]): str(row["client_order_id"])
                for row in self.orders
                if row["broker_order_id"]
            }
            page = self.broker.fills(broker_to_client=mapping)
            seen: set[str] = set()
            duplicate_in_batch = False
            ordered = sorted(page.fills, key=lambda fill: (fill.fill_time_ns, fill.fill_id))
            for fill in ordered:
                if fill.fill_id in seen:
                    duplicate_in_batch = True
                seen.add(fill.fill_id)
                self._record_fill(fill, ts)
            account = self.broker.account()
            self.ledger.declare(cash=account.cash, positions=account.positions)
            report = self.ledger.reconcile()
            report["fills_truncated"] = page.truncated
            report["duplicate_in_batch"] = duplicate_in_batch
            if page.truncated or duplicate_in_batch:
                report["ok"] = False
            self._project_fills_onto_orders()
            report["venue"] = self.venue
            self._sync_account(ts)
            self._persist()
            self._audit("reconcile", {"ok": bool(report["ok"])})
            return report

    def flatten(self, *, actor: str, at_ns: int | None = None) -> dict[str, Any]:
        """Post a limit at the stored mark for every open position. No market orders."""
        with self._lock():
            self._recover_inflight()
            self._require_trading()
            _actor(actor)
            account = self.broker.account()
            results: list[dict[str, Any]] = []
            for symbol, qty in sorted(account.positions.items()):
                if abs(qty) <= 1e-9:
                    continue
                mark = self.marks.get(symbol)
                if mark is None:
                    results.append(
                        {
                            "ok": False,
                            "status": "skipped",
                            "symbol": symbol,
                            "reason": "no_mark",
                            "live_pnl_claim": False,
                            "research_evidence": False,
                        }
                    )
                    continue
                side = "sell" if qty > 0.0 else "buy"
                results.append(
                    self._submit_locked(
                        symbol=symbol,
                        side=side,
                        quantity=abs(qty),
                        limit_price=mark,
                        mark=mark,
                        actor=actor,
                        client_order_id=None,
                        at_ns=at_ns,
                    )
                )
            clean = all(bool(row.get("ok")) for row in results)
            return {
                "ok": clean,
                "orders": results,
                "live_pnl_claim": False,
                "research_evidence": False,
            }

    def _submit_locked(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        mark: float,
        actor: str,
        client_order_id: str | None,
        at_ns: int | None,
    ) -> dict[str, Any]:
        who = _actor(actor)
        ts = self._clock(at_ns)
        self._require_trading()
        cleaned = _symbol(symbol)
        order_side = _side(side)
        qty = _positive(quantity, "quantity")
        limit = _positive(limit_price, "limit price")
        reference = _positive(mark, "mark")
        client = client_order_id or _new_client_id()
        if not _CLIENT.fullmatch(client):
            raise LiveGateError("client_order_id must be 1..48 letters, digits, _ or -")
        existing = self._find(client)
        if existing is not None:
            return _public_order(existing, venue=self.venue, replayed=True)
        self._capture_baseline()
        self.marks[cleaned] = reference
        self.broker.set_mark(cleaned, reference)
        self._sync_account(ts, reset_session=not self.orders)
        placeholder = {
            "client_order_id": client,
            "broker_order_id": "",
            "symbol": cleaned,
            "side": order_side,
            "quantity": qty,
            "limit_price": limit,
            "status": "inflight",
            "filled_quantity": 0.0,
            "fill_price": None,
            "expects_fill": False,
            "reason": "",
        }
        decision = self.engine.decide(self._view(placeholder, limit, ts, KIND_ORDER), apply=False)
        if not decision.allowed:
            denied = _denied(
                reasons=decision.reasons,
                symbol=cleaned,
                side=order_side,
                quantity=qty,
                limit_price=limit,
                client_order_id=client,
                venue=self.venue,
            )
            self._audit(
                "deny", {"actor": who, "client_order_id": client, "reasons": list(decision.reasons)}
            )
            self._persist()
            return denied
        self.orders.append(placeholder)
        self.ledger.note_order(client, expects_fill=False)
        self._persist()
        live_order = LiveOrder(
            client_order_id=client,
            symbol=cleaned,
            side=order_side,
            quantity=qty,
            limit_price=limit,
            fill_time_ns=ts,
        )
        ack = self._send(placeholder, live_order)
        self._apply_ack(placeholder, ack, ts)
        self._persist()
        self._audit(
            "submit", {"actor": who, "client_order_id": client, "status": placeholder["status"]}
        )
        return _public_order(placeholder, venue=self.venue, replayed=False)

    def _send(self, row: dict[str, Any], order: LiveOrder) -> BrokerAck:
        try:
            return self.broker.submit(order)
        except BrokerRejected as exc:
            return BrokerAck(
                client_order_id=order.client_order_id,
                broker_order_id="",
                status="rejected",
                filled_quantity=0.0,
                fill_price=None,
                fills=(),
                reason=str(exc),
            )
        except LiveGateError:
            row["status"] = "not_acknowledged"
            row["reason"] = "broker credentials are not set"
            self._persist()
            raise
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            ValueError,
            TypeError,
            KeyError,
        ):
            recovered = self._safe_recover(order.client_order_id)
            if recovered is not None:
                return recovered
            # A lost response is not proof the broker rejected the order.
            # Leave it inflight so a later open can find it and cancel it.
            row["status"] = "inflight"
            row["reason"] = "broker response lost; order remains recoverable"
            self._persist()
            raise LiveGateError(
                "broker response lost; the order stays inflight until recovery succeeds"
            ) from None

    def _safe_recover(self, client_order_id: str) -> BrokerAck | None:
        try:
            return self.broker.recover(client_order_id)
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            BrokerRejected,
            ValueError,
            TypeError,
            KeyError,
        ):
            return None

    def _apply_ack(self, row: dict[str, Any], ack: BrokerAck, ts: int) -> None:
        row["broker_order_id"] = ack.broker_order_id
        status = map_broker_status(ack.status)
        row["status"] = status
        row["filled_quantity"] = float(ack.filled_quantity)
        row["fill_price"] = ack.fill_price
        row["reason"] = ack.reason
        if status == "unmapped" and not row["reason"]:
            row["reason"] = f"unmapped broker status {ack.status}"
        expects = ack.status in {"filled", "partially_filled"}
        row["expects_fill"] = expects
        self.ledger.set_expects_fill(str(row["client_order_id"]), expects)
        for fill in ack.fills:
            self._record_fill(fill, ts)

    def _record_fill(self, fill: BrokerFill, ts: int) -> str:
        if fill.side not in {"buy", "sell"}:
            raise LiveGateError("broker fill side is not buy or sell")
        ledger_fill = LedgerFill(
            fill_id=fill.fill_id,
            client_order_id=fill.client_order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            fill_time_ns=fill.fill_time_ns if fill.fill_time_ns > 0 else ts,
        )
        verdict = self.ledger.record_fill(ledger_fill)
        if verdict in {"duplicate_fill", "orphan_fill"}:
            return verdict
        if fill.client_order_id in self._engine_noted or verdict == "replaced_provisional":
            self._sync_account(ts)
        else:
            self._engine_from_fill(fill, ts)
            self._engine_noted.add(fill.client_order_id)
        return verdict

    def _engine_from_fill(self, fill: BrokerFill, ts: int) -> None:
        account = self.broker.account()
        pos_after = float(account.positions.get(fill.symbol, 0.0))
        signed = fill.quantity if fill.side == "buy" else -fill.quantity
        mark = self.marks.get(fill.symbol, fill.price)
        sid = self._install(fill.symbol, pos_after, mark, ts, seed_lots=False)
        session_id = self._session_id(ts)
        self.engine.set_settled_cash(self._model_settled)
        self.engine.note_fill(
            symbol_id=sid,
            side=1 if fill.side == "buy" else -1,
            qty=fill.quantity,
            px=fill.price,
            fee=0.0,
            ts_ns=ts,
            session_id=session_id,
            pos_before=pos_after - signed,
        )
        self._model_settled = float(self.engine.book.settled)
        self.engine.set_symbol(sid, pos=pos_after, ref_px=mark, ref_ts_ns=ts, bid=mark)

    def _sync_account(self, ts: int, *, reset_session: bool = False) -> None:
        account = self.broker.account()
        if not math.isfinite(account.cash) or account.cash < 0.0:
            raise LiveGateError("broker cash is not usable")
        if not math.isfinite(account.equity) or account.equity <= 0.0:
            raise LiveGateError("broker equity is not usable")
        self._absorb_broker_marks(account)
        reserved_qty, reserved_cash = self._working_exposure()
        self._require_marks(account.positions, reserved_qty)
        self.engine.update_account(
            nav=account.equity,
            mark_ts_ns=ts,
            session_start_nav=account.equity if reset_session else None,
            peak_nav=account.equity if reset_session else None,
        )
        available = self._model_settled - reserved_cash
        self.engine.set_settled_cash(available if available > 0.0 else 0.0)
        symbols = set(account.positions) | set(reserved_qty) | set(self.marks)
        for symbol in sorted(symbols):
            mark = self.marks.get(symbol)
            if mark is None or mark <= 0.0:
                continue
            qty = float(account.positions.get(symbol, 0.0)) + float(reserved_qty.get(symbol, 0.0))
            self._install(symbol, qty, mark, ts)

    def _absorb_broker_marks(self, account: BrokerAccount) -> None:
        for symbol, price in account.marks.items():
            if price > 0.0 and symbol not in self.marks:
                self.marks[symbol] = float(price)
                self.broker.set_mark(symbol, float(price))

    def _require_marks(self, positions: dict[str, float], reserved: dict[str, float]) -> None:
        symbols = set(positions) | set(reserved)
        for symbol in sorted(symbols):
            held = float(positions.get(symbol, 0.0))
            pending = float(reserved.get(symbol, 0.0))
            if abs(held) <= 1e-9 and abs(pending) <= 1e-9:
                continue
            mark = self.marks.get(symbol)
            if mark is None or mark <= 0.0:
                raise LiveGateError(
                    f"broker position {symbol} has no mark; submission is refused until it is marked"
                )

    def _working_exposure(self) -> tuple[dict[str, float], float]:
        """Unfilled working quantity and buy notional still reserved at the broker."""
        qty: dict[str, float] = {}
        cash = 0.0
        for row in self.orders:
            if row["status"] not in _WORKING:
                continue
            remaining = float(row["quantity"]) - float(row["filled_quantity"])
            if remaining <= 1e-9:
                continue
            symbol = str(row["symbol"])
            signed = remaining if row["side"] == "buy" else -remaining
            qty[symbol] = qty.get(symbol, 0.0) + signed
            if row["side"] == "buy":
                cash += remaining * float(row["limit_price"])
        return qty, cash

    def _project_fills_onto_orders(self) -> None:
        """Copy reconciled fills onto the order row so cancel sees a filled order."""
        grouped: dict[str, list[LedgerFill]] = {}
        for fill in self.ledger.fills():
            grouped.setdefault(fill.client_order_id, []).append(fill)
        for row in self.orders:
            fills = grouped.get(str(row["client_order_id"]))
            if not fills:
                continue
            qty = sum(fill.quantity for fill in fills)
            cost = sum(fill.quantity * fill.price for fill in fills)
            row["filled_quantity"] = qty
            if qty > 0.0:
                row["fill_price"] = cost / qty
            if qty + 1e-9 >= float(row["quantity"]):
                row["status"] = "filled"
                row["expects_fill"] = True
            elif qty > 1e-9 and row["status"] not in _TERMINAL:
                row["status"] = "partially_filled"
                row["expects_fill"] = True
            client = str(row["client_order_id"])
            if client in self.ledger._expects:
                self.ledger.set_expects_fill(client, bool(row["expects_fill"]))

    def _install(
        self, symbol: str, qty: float, mark: float, ts: int, *, seed_lots: bool = True
    ) -> int:
        sid = self.engine.ensure_symbol(symbol)
        if sid < 0:
            raise LiveGateError("symbol capacity is full")
        self.engine.set_symbol(sid, pos=qty, ref_px=mark, ref_ts_ns=ts, bid=mark)
        if seed_lots and qty > 0.0 and self.engine.book.syms[sid].nlot == 0:
            self.engine.restore_lot(sid, qty=qty, settles_ns=ts, unsettled_funded=False)
        self._sids[symbol] = sid
        return sid

    def _view(self, row: dict[str, Any], price: float, ts: int, kind: int) -> OrderView:
        symbol = str(row["symbol"])
        if symbol not in self._sids:
            mark = self.marks.get(symbol, price)
            self._install(symbol, 0.0, mark, ts)
        return OrderView(
            symbol_id=self._sids[symbol],
            side=1 if row["side"] == "buy" else -1,
            qty=float(row["quantity"]),
            px=float(price),
            ts_ns=ts,
            session_id=self._session_id(ts),
            is_limit=1,
            kind=kind,
        )

    def _session_id(self, ts: int) -> int:
        zone = ZoneInfo(self.pretrade.session.timezone)
        local = datetime.fromtimestamp(ts / 1_000_000_000, zone)
        return business_session_id(local.date())

    def _capture_baseline(self) -> None:
        if self.venue == "loopback" or self._baseline_from_broker:
            return
        if self.orders or self.ledger.fills():
            return
        account = self.broker.account()
        self.ledger = LiveLedger(
            baseline_cash=account.cash,
            baseline_positions=dict(account.positions),
            cash_tolerance=self.config.cash_tolerance,
            position_tolerance=self.config.position_tolerance,
        )
        self._baseline_from_broker = True

    def _recover_inflight(self) -> None:
        ts = self._now()
        changed = False
        for row in self.orders:
            if row["status"] != "inflight":
                continue
            ack = self._safe_recover(str(row["client_order_id"]))
            if ack is None:
                continue
            changed = True
            self._apply_ack(row, ack, ts)
        if changed:
            self._persist()

    def _require_trading(self) -> None:
        if self._arm is None:
            raise LiveGateError("session is not armed; run live arm")
        if not _credentials_present(self.broker) and self.venue != "loopback":
            raise LiveGateError("broker credentials are not set")

    def _clock(self, at_ns: int | None) -> int:
        if at_ns is None:
            return int(self._now())
        if self.venue == "alpaca-live":
            raise LiveGateError(
                "--at is refused for alpaca-live; the session clock is the wall clock"
            )
        if at_ns < 0:
            raise LiveGateError("timestamp must be non-negative")
        return int(at_ns)

    def _find(self, client_order_id: str) -> dict[str, Any] | None:
        for row in self.orders:
            if row["client_order_id"] == client_order_id:
                return row
        return None

    def _opening_balances(self) -> tuple[float, float]:
        if self.venue != "loopback":
            cash = float(self.ledger.computed_cash)
            nav = cash if cash > 0.0 else float(self.config.starting_cash)
            return nav, cash
        account = self.broker.account()
        nav = account.equity if account.equity > 0.0 else account.cash
        if nav <= 0.0:
            nav = float(self.config.starting_cash)
        return nav, account.cash

    def _load_state(self, stored: dict[str, Any] | None) -> None:
        if stored is None:
            self._load_fresh()
            return
        self._load_stored(stored)

    def _load_fresh(self) -> None:
        self.orders = []
        self.marks = {}
        self.kill_state = {"latched": False, "actor": "", "reason": ""}
        self._pretrade_blob = None
        self._baseline_from_broker = self.venue == "loopback"
        self.ledger = self._new_ledger(float(self.config.starting_cash), {})
        if self.venue == "loopback" and not isinstance(self.broker, LoopbackBroker):
            raise LiveGateError("loopback venue requires the loopback broker")
        if isinstance(self.broker, LoopbackBroker):
            self.broker.cash = float(self.config.starting_cash)

    def _load_stored(self, stored: dict[str, Any]) -> None:
        _require_state_identity(stored, self.venue)
        orders, fills, marks, kill = _state_collections(stored)
        self.orders = [_order_row(item) for item in orders]
        self.marks = {
            str(symbol): _positive(float(price), "stored mark") for symbol, price in marks.items()
        }
        self.kill_state = _kill_from_state(kill)
        self._pretrade_blob = stored.get("pretrade")
        baseline_positions = stored.get("baseline_positions") or {}
        if not isinstance(baseline_positions, dict):
            raise LiveGateError("baseline positions are malformed")
        self._baseline_from_broker = bool(
            stored.get("baseline_from_broker", self.venue == "loopback")
        )
        self.ledger = self._new_ledger(
            float(stored.get("baseline_cash", self.config.starting_cash)),
            {str(key): float(value) for key, value in baseline_positions.items()},
        )
        self._replay(fills)
        if isinstance(self.broker, LoopbackBroker):
            self.broker.cash = self.ledger.computed_cash
            self.broker.positions = self.ledger.computed_positions

    def _new_ledger(self, cash: float, positions: dict[str, float]) -> LiveLedger:
        return LiveLedger(
            baseline_cash=cash,
            baseline_positions=positions,
            cash_tolerance=self.config.cash_tolerance,
            position_tolerance=self.config.position_tolerance,
        )

    def _replay(self, fills: list[Any]) -> None:
        for row in self.orders:
            self.ledger.note_order(
                str(row["client_order_id"]), expects_fill=bool(row["expects_fill"])
            )
        for item in fills:
            if not isinstance(item, dict):
                raise LiveGateError("stored fill is malformed")
            self.ledger.record_fill(ledger_fill_from_dict(item))

    def _persist(self) -> None:
        payload: dict[str, Any] = {
            "schema": _STATE_SCHEMA,
            "venue": self.venue,
            "baseline_cash": self.ledger.baseline_cash,
            "baseline_positions": self.ledger.baseline_positions,
            "baseline_from_broker": self._baseline_from_broker,
            "marks": dict(self.marks),
            "orders": list(self.orders),
            "fills": [fill.as_dict() for fill in self.ledger.fills()],
            "kill": dict(self.kill_state),
            "pretrade": export_pretrade(self.engine),
            "live_pnl_claim": False,
            "research_evidence": False,
        }
        _atomic_json(self.state_path, payload)

    def _audit(self, event: str, fields: dict[str, Any]) -> None:
        body: dict[str, Any] = {
            "event": event,
            "prev_hash": self.audit_head,
            "live_pnl_claim": False,
            "research_evidence": False,
        }
        body.update(fields)
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        line = dict(body)
        line["event_hash"] = digest
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_line = (json.dumps(line, sort_keys=True) + "\n").encode("utf-8")
        fd = os.open(str(self.audit_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, encoded_line)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(self.audit_path, 0o600)
        self.audit_head = digest

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.state_dir, 0o700)
        path = self.state_dir / "lock"
        handle = path.open("a+b")
        try:
            if path.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            os.chmod(path, 0o600)
            _acquire(handle)
            try:
                token = self._file_token()
                if token != self._disk_token:
                    self._reload_locked()
                yield
                self._disk_token = self._file_token()
            finally:
                _release(handle)
        finally:
            handle.close()

    def _file_token(self) -> str:
        digest = hashlib.sha256()
        for path in (self.arm_path, self.state_path, self.audit_path):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes() if path.exists() else b"-")
        return digest.hexdigest()

    def _reload_locked(self) -> None:
        """Read the desk from disk. The caller already holds the process lock."""
        if self.audit_path.exists():
            os.chmod(self.audit_path, 0o600)
        self._arm = read_arm(self.arm_path, venue=self.venue, pretrade_sha256=self.pretrade_sha256)
        stored = _read_object(self.state_path)
        self.audit_head = _verify_audit(self.audit_path)
        self._load_state(stored)
        self._build_engine()
        for symbol, price in self.marks.items():
            self.broker.set_mark(symbol, price)
        if self.venue == "loopback":
            self._sync_account(self._now())
        if self.kill_state["latched"]:
            self.engine.trip(
                str(self.kill_state["actor"]), str(self.kill_state["reason"]), self._now()
            )
        self._recover_inflight()

    def _build_engine(self) -> None:
        """New engine, then the persisted rings and a fill replay for settlement."""
        cash = float(self.ledger.baseline_cash)
        if cash <= 0.0:
            cash = float(self.config.starting_cash)
        self._model_settled = cash
        self.engine = PretradeEngine(
            self.pretrade,
            hmac_key=_hmac_for_engine(self.state_dir),
            initial_nav=cash,
            initial_cash=cash,
            ts_ns=self._now(),
        )
        self._model_settled = float(self.engine.book.settled)
        self._sids = {}
        self._engine_noted = set()
        if self._pretrade_blob is not None:
            restore_pretrade(self.engine, self._pretrade_blob)
        self._replay_fills_into_engine(self._now())

    def _replay_fills_into_engine(self, ts: int) -> None:
        positions: dict[str, float] = {}
        for fill in self.ledger.fills():
            signed = fill.quantity if fill.side == "buy" else -fill.quantity
            before = positions.get(fill.symbol, 0.0)
            after = before + signed
            mark = self.marks.get(fill.symbol, fill.price)
            sid = self._install(fill.symbol, after, mark, ts, seed_lots=False)
            self.engine.set_settled_cash(self._model_settled)
            self.engine.note_fill(
                symbol_id=sid,
                side=1 if fill.side == "buy" else -1,
                qty=fill.quantity,
                px=fill.price,
                fee=0.0,
                ts_ns=fill.fill_time_ns if fill.fill_time_ns > 0 else ts,
                session_id=self._session_id(fill.fill_time_ns if fill.fill_time_ns > 0 else ts),
                pos_before=before,
            )
            self._model_settled = float(self.engine.book.settled)
            self._engine_noted.add(fill.client_order_id)
            positions[fill.symbol] = after


def open_session(
    config: LiveConfig,
    config_path: str | Path,
    *,
    state_dir: str | Path | None = None,
    now_ns: Callable[[], int] | None = None,
    broker: Broker | None = None,
    transport: Any = None,
) -> LiveSession:
    """Open or create the session directory for ``config``."""
    root = Path(state_dir) if state_dir is not None else Path(config.state_dir)
    root.mkdir(parents=True, exist_ok=True)
    key = load_hmac_key(root / "hmac.key")
    pretrade, digest, _signature = load_gateway_pretrade(config, config_path, hmac_key=key)
    clock = now_ns or _wall_ns
    if broker is None:
        broker = _default_broker(config, transport=transport)
    return LiveSession(
        config,
        state_dir=root,
        pretrade=pretrade,
        pretrade_sha256=digest,
        broker=broker,
        now_ns=clock,
    )


def _default_broker(config: LiveConfig, *, transport: Any) -> Broker:
    if config.venue == "loopback":
        return LoopbackBroker(cash=float(config.starting_cash))
    return AlpacaBroker(config.venue, alpaca_credentials(), transport=transport)


def _hmac_for_engine(state_dir: Path) -> bytes:
    return (state_dir / "hmac.key").read_bytes()


def _wall_ns() -> int:
    return time.time_ns()


def _credentials_present(broker: Broker) -> bool:
    if isinstance(broker, AlpacaBroker):
        return broker.credentials_present
    return True


def _verify_audit(path: Path) -> str:
    head = zero_audit_head()
    if not path.exists():
        return head
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parsed = json.loads(raw_line)
        if not isinstance(parsed, dict):
            raise LiveGateError("audit log line is not an object")
        row = dict(parsed)
        event_hash = row.pop("event_hash", None)
        if row.get("prev_hash") != head:
            raise LiveGateError("audit log chain does not match")
        if row.get("live_pnl_claim") is not False or row.get("research_evidence") is not False:
            raise LiveGateError("audit log claims live profit or research evidence")
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != event_hash:
            raise LiveGateError("audit log line was modified")
        head = digest
    return head


def _require_state_identity(stored: dict[str, Any], venue: str) -> None:
    if stored.get("schema") != _STATE_SCHEMA:
        raise LiveGateError("session state schema is not supported")
    if stored.get("venue") != venue:
        raise LiveGateError("session state venue does not match the live config")
    profit = stored.get("live_pnl_claim") is not False
    evidence = stored.get("research_evidence") is not False
    if profit or evidence:
        raise LiveGateError("session state must not claim research evidence or live profit")


def _state_collections(
    stored: dict[str, Any],
) -> tuple[list[Any], list[Any], dict[str, Any], dict[str, Any]]:
    orders = stored.get("orders")
    fills = stored.get("fills")
    marks = stored.get("marks")
    kill = stored.get("kill")
    if not isinstance(orders, list) or not isinstance(fills, list) or not isinstance(marks, dict):
        raise LiveGateError("session state is missing orders, fills, or marks")
    if not isinstance(kill, dict):
        raise LiveGateError("session state is missing the kill switch")
    return orders, fills, marks, kill


def _kill_from_state(kill: dict[str, Any]) -> dict[str, Any]:
    latched = bool(kill.get("latched"))
    actor = str(kill.get("actor") or "")
    reason = str(kill.get("reason") or "")
    if latched and (not actor.strip() or not reason.strip()):
        raise LiveGateError("a latched kill switch needs an actor and a reason")
    return {"latched": latched, "actor": actor, "reason": reason}


def _order_row(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise LiveGateError("stored order is malformed")
    status = str(raw.get("status") or "")
    if status not in _ORDER_STATUSES:
        raise LiveGateError("stored order status is not supported")
    fill_price = raw.get("fill_price")
    return {
        "client_order_id": str(raw["client_order_id"]),
        "broker_order_id": str(raw.get("broker_order_id") or ""),
        "symbol": _symbol(str(raw["symbol"])),
        "side": _side(str(raw["side"])),
        "quantity": _positive(float(raw["quantity"]), "stored quantity"),
        "limit_price": _positive(float(raw["limit_price"]), "stored limit"),
        "status": status,
        "filled_quantity": float(raw.get("filled_quantity") or 0.0),
        "fill_price": None if fill_price is None else float(fill_price),
        "expects_fill": bool(raw.get("expects_fill")),
        "reason": str(raw.get("reason") or ""),
    }


def _public_order(row: dict[str, Any], *, venue: str, replayed: bool) -> dict[str, Any]:
    status = str(row["status"])
    return {
        "ok": status in {"filled", "accepted", "partially_filled", "canceled"},
        "replayed": replayed,
        "status": status,
        "client_order_id": row["client_order_id"],
        "broker_order_id": row["broker_order_id"],
        "symbol": row["symbol"],
        "side": row["side"],
        "quantity": row["quantity"],
        "limit_price": row["limit_price"],
        "filled_quantity": row["filled_quantity"],
        "fill_price": row["fill_price"],
        "reason": row["reason"],
        "reasons": [],
        "venue": venue,
        "broker_called": True,
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def _denied(
    *,
    reasons: tuple[str, ...],
    symbol: str,
    side: str,
    quantity: float,
    limit_price: float,
    client_order_id: str,
    venue: str,
) -> dict[str, Any]:
    return {
        "ok": False,
        "replayed": False,
        "status": "denied",
        "client_order_id": client_order_id,
        "broker_order_id": "",
        "symbol": symbol,
        "side": side,
        "quantity": quantity,
        "limit_price": limit_price,
        "filled_quantity": 0.0,
        "fill_price": None,
        "reason": ",".join(reasons),
        "reasons": list(reasons),
        "venue": venue,
        "broker_called": False,
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def _windows_lock(handle: Any, operation: str) -> None:
    """Byte-range lock via the Windows C runtime. ``operation`` is LK_LOCK or LK_UNLCK."""
    import importlib
    from operator import attrgetter

    runtime = importlib.import_module("msvcrt")
    handle.seek(0)
    attrgetter("locking")(runtime)(handle.fileno(), attrgetter(operation)(runtime), 1)


def _acquire(handle: Any) -> None:
    if os.name == "nt":
        _windows_lock(handle, "LK_LOCK")
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle: Any) -> None:
    if os.name == "nt":
        _windows_lock(handle, "LK_UNLCK")
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_timestamp(text: str) -> int:
    """Parse an aware ISO-8601 timestamp into nanoseconds."""
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        raise LiveGateError("--at must include a timezone offset")
    return int(parsed.timestamp() * 1_000_000_000)
