"""Pre-trade engine: snapshot checks, latched kill switch, signed decisions.

Shadow checks are the engine's job. ``check`` returns a bitset and does not
route orders. ``quant_fund.live`` is the desk that consumes ``decide`` and,
when armed, hands a ticket to a bridge.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from quant_fund.pretrade.codes import (
    INTERNAL,
    KILL,
    KIND_CANCEL,
    MANUAL,
    UNKNOWN_SYMBOL,
    decision_allowed,
    reason_names,
)
from quant_fund.pretrade.config import PretradeConfig, sign_config
from quant_fund.pretrade.kernel import account_message, hot_check
from quant_fund.pretrade.state import HotBook, SymbolSlot

_EPS = 1e-9
_CASH_EPS = 1e-6


@dataclass(slots=True)
class OrderView:
    symbol_id: int
    side: int
    qty: float
    px: float
    ts_ns: int
    session_id: int
    is_limit: int
    kind: int


@dataclass(slots=True)
class Decision:
    allowed: bool
    reason_bits: int
    reasons: tuple[str, ...]
    config_sha256: str
    config_hmac_sha256: str
    schema_version: int
    kill_latched: bool
    ts_ns: int
    symbol_id: int
    kind: int

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason_bits": self.reason_bits,
            "reasons": list(self.reasons),
            "config_sha256": self.config_sha256,
            "config_hmac_sha256": self.config_hmac_sha256,
            "schema_version": self.schema_version,
            "kill_latched": self.kill_latched,
            "ts_ns": self.ts_ns,
            "symbol_id": self.symbol_id,
            "kind": self.kind,
            "mode": "shadow",
        }


class PretradeEngine:
    """Deterministic pre-trade gate over a precomputed limits snapshot."""

    def __init__(
        self,
        config: PretradeConfig,
        *,
        hmac_key: bytes,
        initial_nav: float,
        initial_cash: float,
        ts_ns: int,
    ) -> None:
        if not math.isfinite(initial_nav) or not math.isfinite(initial_cash):
            raise ValueError("initial nav and cash must be finite")
        if ts_ns < 0:
            raise ValueError("ts_ns must be non-negative")
        self.config = config
        self.config_sha256, self.config_hmac_sha256 = sign_config(config, hmac_key)
        self.schema_version = int(config.schema_version)
        self.book: HotBook = HotBook(config)
        self._tz = ZoneInfo(config.session.timezone)
        self._open_minute = int(config.session.open_minute)
        self._close_minute = int(config.session.close_minute)
        self._max_ref_age = int(config.limits.max_reference_age_ns)
        self._max_mark_age = int(config.limits.max_mark_age_ns)
        self._concentration = float(config.limits.max_name_concentration)
        self._pdt_equity = float(config.limits.pdt_equity_threshold)
        self._daily_loss = float(config.limits.max_daily_loss_fraction)
        self._trailing_dd = float(config.limits.max_trailing_drawdown_fraction)
        self._ids: dict[str, int] = {}
        self._n_reg = 0
        self.audit: list[dict[str, object]] = []
        self._audit_head = "0" * 64
        self.last_bits = 0
        self.last_allowed = False
        self._session_date: date | None = None
        nav = float(initial_nav)
        cash = float(initial_cash)
        self.book.session_start = nav
        self.book.peak = nav
        self.book.settled = cash
        self.update_account(
            nav=nav,
            mark_ts_ns=ts_ns,
            session_start_nav=nav,
            peak_nav=nav,
        )
        self.refresh_session(ts_ns)

    def ensure_symbol(self, symbol: str) -> int:
        sid = self._ids.get(symbol)
        if sid is not None:
            return sid
        if self._n_reg >= len(self.book.syms):
            return -1
        sid = self._n_reg
        self._n_reg += 1
        self._ids[symbol] = sid
        return sid

    def set_symbol(
        self,
        symbol_id: int,
        *,
        pos: float,
        ref_px: float,
        ref_ts_ns: int,
        halted: bool = False,
        sho_restricted: bool = False,
        locate_ok: bool = True,
        bid: float = 0.0,
    ) -> None:
        sym = self._symbol(symbol_id)
        if not math.isfinite(pos):
            raise ValueError("position must be finite")
        sym.pos = float(pos)
        self._set_ref(sym, float(ref_px), int(ref_ts_ns))
        sym.halted = 1 if halted else 0
        sym.sho = 1 if sho_restricted else 0
        sym.locate = 1 if locate_ok else 0
        sym.bid = float(bid) if bid == bid else 0.0
        self.recompute_exposure()

    def update_account(
        self,
        *,
        nav: float,
        mark_ts_ns: int,
        session_start_nav: float | None = None,
        peak_nav: float | None = None,
    ) -> None:
        self.book.nav = float(nav)
        if session_start_nav is not None:
            self.book.session_start = float(session_start_nav)
        if peak_nav is not None:
            self.book.peak = float(peak_nav)
        elif nav == nav and nav > self.book.peak:
            self.book.peak = float(nav)
        self._recompute_floors()
        stamp = int(mark_ts_ns)
        if nav == nav and nav > 0.0 and stamp >= 0:
            self.book.mark_deadline = stamp + self._max_mark_age
            self.book.mark_ts = stamp
            self.book.name_limit = self._concentration * float(nav)
            self.book.pdt_tight = 1 if float(nav) < self._pdt_equity else 0
        else:
            self.book.mark_deadline = 0
            self.book.name_limit = 0.0
            self.book.pdt_tight = 1

    def recompute_exposure(self) -> None:
        gross = 0.0
        net = 0.0
        i = 0
        while i < self._n_reg:
            sym = self.book.syms[i]
            if sym.ref_ok:
                notion = sym.pos * sym.ref
            else:
                notion = 0.0
                if sym.pos > _EPS or sym.pos < -_EPS:
                    self.book.mark_deadline = 0
            gross += notion if notion >= 0.0 else -notion
            net += notion
            i += 1
        self.book.gross = gross
        self.book.net = net

    def refresh_session(self, ts_ns: int) -> None:
        dt = datetime.fromtimestamp(ts_ns / 1_000_000_000, self._tz)
        local = dt.date()
        start = datetime(local.year, local.month, local.day, tzinfo=self._tz)
        end = start + timedelta(days=1)
        self.book.day_lo = int(start.timestamp() * 1_000_000_000)
        self.book.day_hi = int(end.timestamp() * 1_000_000_000)
        rolled = self._session_date is not None and local > self._session_date
        self._session_date = local
        if rolled and self.book.nav == self.book.nav and self.book.nav > 0.0:
            self.book.session_start = self.book.nav
            self._recompute_floors()
        if dt.weekday() >= 5:
            self.book.session_open = 0
            self.book.open_ns = self.book.day_lo
            self.book.close_ns = self.book.day_lo
            return
        open_h, open_m = divmod(self._open_minute, 60)
        close_h, close_m = divmod(self._close_minute, 60)
        open_dt = start.replace(hour=open_h, minute=open_m)
        close_dt = (start + timedelta(days=close_h // 24)).replace(
            hour=close_h % 24, minute=close_m
        )
        self.book.open_ns = int(open_dt.timestamp() * 1_000_000_000)
        self.book.close_ns = int(close_dt.timestamp() * 1_000_000_000)
        self.book.session_open = 1

    def restore_lot(
        self,
        symbol_id: int,
        *,
        qty: float,
        settles_ns: int,
        unsettled_funded: bool,
    ) -> None:
        """Cold-path lot restore. A full table fails closed."""
        sym = self._symbol(symbol_id)
        if qty != qty or qty <= 0.0:
            raise ValueError("lot quantity must be finite and positive")
        if sym.nlot >= self.book.max_lots:
            raise RuntimeError("lot table is full")
        sym.lq[sym.nlot] = float(qty)
        sym.ls[sym.nlot] = int(settles_ns)
        sym.lu[sym.nlot] = 1 if unsettled_funded else 0
        sym.nlot += 1

    def trip(self, actor: str, reason: str, ts_ns: int) -> None:
        actor_s, reason_s = _require_actor(actor, reason, "manual trip")
        self.book.killed = 1
        self.book.kill_reason |= MANUAL
        self._audit("trip", actor_s, reason_s, MANUAL, ts_ns)

    def reset(self, actor: str, reason: str, ts_ns: int) -> None:
        """Clear the latch. Rings, lots, and floors are left as they are."""
        actor_s, reason_s = _require_actor(actor, reason, "manual reset")
        prev = int(self.book.kill_reason)
        self.book.killed = 0
        self.book.kill_reason = 0
        self._flush_pending_trip(ts_ns)
        self._audit("reset", actor_s, reason_s, prev, ts_ns)

    def check(self, order: OrderView, *, apply: bool = True) -> int:
        """Hot path. Returns the reason bitset. Builds no decision record."""
        try:
            if order.symbol_id < 0 or order.symbol_id >= self._n_reg:
                bits = UNKNOWN_SYMBOL | account_message(self.book, order.ts_ns)
            else:
                if order.ts_ns < self.book.day_lo or order.ts_ns >= self.book.day_hi:
                    self.refresh_session(order.ts_ns)
                sym = self.book.syms[order.symbol_id]
                bits = hot_check(
                    self.book,
                    sym,
                    side=order.side,
                    qty=order.qty,
                    px=order.px,
                    ts=order.ts_ns,
                    session_id=order.session_id,
                    is_limit=order.is_limit,
                    kind=order.kind,
                    symbol_id=order.symbol_id,
                )
        except Exception as exc:
            self._latch_internal(f"internal:{type(exc).__name__}", order.ts_ns)
            self.last_bits = INTERNAL | KILL
            self.last_allowed = False
            return self.last_bits
        self._flush_pending_trip(order.ts_ns)
        allowed = decision_allowed(bits, order.kind)
        if apply and allowed and order.kind != KIND_CANCEL:
            try:
                self._apply(order)
            except Exception as exc:
                self._latch_internal(f"internal:{type(exc).__name__}", order.ts_ns)
                bits |= INTERNAL | KILL
                allowed = False
        self.last_bits = bits
        self.last_allowed = allowed
        return bits

    def decide(self, order: OrderView, *, apply: bool = True) -> Decision:
        bits = self.check(order, apply=apply)
        return self._decision(order, bits)

    def note_fill(
        self,
        *,
        symbol_id: int,
        side: int,
        qty: float,
        px: float,
        fee: float,
        ts_ns: int,
        session_id: int,
        pos_before: float,
    ) -> None:
        """Update lots, settled cash, and PDT memory from an observed fill.

        Position quantity is owned by the caller (the shadow adapter resyncs
        it from the simulated broker). This method does not submit anything.
        """
        try:
            if (
                not math.isfinite(qty)
                or not math.isfinite(px)
                or not math.isfinite(fee)
                or not math.isfinite(pos_before)
                or qty <= 0.0
                or px <= 0.0
                or fee < 0.0
                or ts_ns < 0
                or (side != 1 and side != -1)
            ):
                raise ValueError("fill fields must be finite")
            sym = self._symbol(symbol_id)
            signed = qty if side > 0 else -qty
            after = pos_before + signed
            self._note_pdt(sym, pos_before, after, session_id)
            self._mature(ts_ns)
            if side > 0:
                need = qty * px + fee
                unsettled = need > self.book.settled + _CASH_EPS
                if unsettled:
                    self.book.settled = 0.0
                    settles = ts_ns + self.book.settlement_ns
                else:
                    self.book.settled -= need
                    settles = ts_ns
                self._add_lot(sym, qty, settles, 1 if unsettled else 0)
            else:
                closed = qty if pos_before <= 0.0 else (qty if qty < pos_before else pos_before)
                if pos_before > 0.0:
                    self._consume_lots(sym, closed)
                proceeds = qty * px - fee
                if not self._enqueue_unsettled(proceeds, ts_ns + self.book.settlement_ns):
                    raise RuntimeError("unsettled cash queue is full")
        except Exception as exc:
            self._latch_internal(f"internal:{type(exc).__name__}", ts_ns)

    def _decision(self, order: OrderView, bits: int) -> Decision:
        return Decision(
            allowed=decision_allowed(bits, order.kind),
            reason_bits=bits,
            reasons=reason_names(bits),
            config_sha256=self.config_sha256,
            config_hmac_sha256=self.config_hmac_sha256,
            schema_version=self.schema_version,
            kill_latched=bool(self.book.killed),
            ts_ns=order.ts_ns,
            symbol_id=order.symbol_id,
            kind=order.kind,
        )

    def _symbol(self, symbol_id: int) -> SymbolSlot:
        if symbol_id < 0 or symbol_id >= self._n_reg:
            raise ValueError("symbol is not registered")
        return self.book.syms[symbol_id]

    def _set_ref(self, sym: SymbolSlot, ref: float, ref_ts: int) -> None:
        if ref != ref or ref <= 0.0:
            sym.ref_ok = 0
            sym.ref = 0.0
            sym.hi = 0.0
            sym.lo = 0.0
            sym.deadline = 0
            sym.ref_ts = 0
            return
        sym.ref = ref
        sym.ref_ok = 1
        sym.ref_ts = int(ref_ts)
        frac = self.book.collar_frac
        sym.hi = ref * (1.0 + frac)
        sym.lo = ref * (1.0 - frac)
        sym.deadline = int(ref_ts) + self._max_ref_age

    def _recompute_floors(self) -> None:
        start = self.book.session_start
        peak = self.book.peak
        if start == start and start > 0.0:
            self.book.daily_floor = start * (1.0 - self._daily_loss)
        else:
            self.book.daily_floor = float("inf")
        if peak == peak and peak > 0.0:
            self.book.dd_floor = peak * (1.0 - self._trailing_dd)
        else:
            self.book.dd_floor = float("inf")

    def _apply(self, order: OrderView) -> None:
        sym = self._symbol(order.symbol_id)
        before = sym.pos
        signed = order.qty if order.side > 0 else -order.qty
        after = before + signed
        ref = sym.ref if sym.ref_ok else 0.0
        old_n = before * ref
        new_n = after * ref
        abs_old = old_n if old_n >= 0.0 else -old_n
        abs_new = new_n if new_n >= 0.0 else -new_n
        self.book.gross = self.book.gross - abs_old + abs_new
        self.book.net = self.book.net - old_n + new_n
        sym.pos = after
        self._note_pdt(sym, before, after, order.session_id)
        self._mature(order.ts_ns)
        if order.side > 0:
            need = order.qty * order.px
            unsettled = need > self.book.settled + _CASH_EPS
            if unsettled:
                self.book.settled = 0.0
                settles = order.ts_ns + self.book.settlement_ns
            else:
                self.book.settled -= need
                settles = order.ts_ns
            self._add_lot(sym, order.qty, settles, 1 if unsettled else 0)
        else:
            closed = 0.0
            if before > 0.0:
                closed = order.qty if order.qty < before else before
                self._consume_lots(sym, closed)
            if not self._enqueue_unsettled(
                order.qty * order.px, order.ts_ns + self.book.settlement_ns
            ):
                raise RuntimeError("unsettled cash queue is full")

    def _note_pdt(self, sym: SymbolSlot, before: float, after: float, session_id: int) -> None:
        closes = (before > _EPS or before < -_EPS) and (
            (after <= _EPS and after >= -_EPS) or before * after < 0.0
        )
        opens = (after > _EPS or after < -_EPS) and (
            (before <= _EPS and before >= -_EPS) or before * after < 0.0
        )
        if closes and sym.opened_session == session_id and session_id > 0:
            self._push_day_trade(session_id)
        if after <= _EPS and after >= -_EPS:
            sym.opened_session = 0
        elif opens:
            sym.opened_session = session_id

    def _push_day_trade(self, session_id: int) -> None:
        if self.book.dt_n >= self.book.dt_cap:
            arr = self.book.dt_sess
            i = 1
            while i < self.book.dt_n:
                arr[i - 1] = arr[i]
                i += 1
            self.book.dt_n -= 1
        self.book.dt_sess[self.book.dt_n] = session_id
        self.book.dt_n += 1

    def _add_lot(self, sym: SymbolSlot, qty: float, settles: int, unsettled: int) -> None:
        if sym.nlot >= self.book.max_lots:
            raise RuntimeError("lot table is full")
        sym.lq[sym.nlot] = qty
        sym.ls[sym.nlot] = settles
        sym.lu[sym.nlot] = unsettled
        sym.nlot += 1

    def _consume_lots(self, sym: SymbolSlot, closed: float) -> None:
        left = closed
        nlot = sym.nlot
        k = 0
        while k < nlot and left > _EPS:
            take = sym.lq[k] if sym.lq[k] < left else left
            sym.lq[k] -= take
            left -= take
            k += 1
        write = 0
        read = 0
        while read < nlot:
            if sym.lq[read] > _EPS:
                sym.lq[write] = sym.lq[read]
                sym.ls[write] = sym.ls[read]
                sym.lu[write] = sym.lu[read]
                write += 1
            read += 1
        sym.nlot = write

    def _mature(self, ts: int) -> None:
        live = self.book.uc_live
        if live <= 0:
            return
        tail = self.book.uc_tail
        mask = self.book.uc_mask
        settled = self.book.settled
        while live > 0 and self.book.uc_ts[tail] <= ts:
            settled += self.book.uc_amt[tail]
            tail = (tail + 1) & mask
            live -= 1
        self.book.uc_tail = tail
        self.book.uc_live = live
        self.book.settled = settled

    def _enqueue_unsettled(self, amount: float, available_ns: int) -> bool:
        if not math.isfinite(amount):
            return False
        if amount <= 0.0:
            self.book.settled += amount
            return self.book.settled == self.book.settled
        if self.book.uc_live >= self.book.uc_cap:
            return False
        head = self.book.uc_head
        self.book.uc_amt[head] = amount
        self.book.uc_ts[head] = available_ns
        self.book.uc_head = (head + 1) & self.book.uc_mask
        self.book.uc_live += 1
        return True

    def _flush_pending_trip(self, ts_ns: int) -> None:
        trip_bits = int(self.book.pending_trip)
        if not trip_bits:
            return
        self.book.pending_trip = 0
        names = reason_names(trip_bits)
        self._audit(
            "trip", "engine", "+".join(names) if names else "circuit_breaker", trip_bits, ts_ns
        )

    def _latch_internal(self, detail: str, ts_ns: int) -> None:
        self.book.killed = 1
        self.book.kill_reason |= INTERNAL
        self._flush_pending_trip(ts_ns)
        self._audit("trip", "engine", detail, INTERNAL, ts_ns)

    def _audit(self, action: str, actor: str, reason: str, bits: int, ts_ns: int) -> None:
        seq = len(self.audit) + 1
        body: dict[str, object] = {
            "action": action,
            "actor": actor,
            "config_sha256": self.config_sha256,
            "prev_hash": self._audit_head,
            "reason": reason,
            "reason_bits": int(bits),
            "seq": seq,
            "ts_ns": int(ts_ns),
        }
        blob = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
        event_hash = hashlib.sha256(blob).hexdigest()
        body["event_hash"] = event_hash
        self.audit.append(body)
        self._audit_head = event_hash


def _require_actor(actor: str, reason: str, what: str) -> tuple[str, str]:
    actor_s = actor.strip()
    reason_s = reason.strip()
    if not actor_s or not reason_s:
        raise ValueError(f"{what} requires a non-empty actor and reason")
    return actor_s, reason_s
