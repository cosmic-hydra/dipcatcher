"""Delta-neutral funding-carry backtester: short perp + long spot per unit.

This is the honest version of the funding-carry trade — not a directional bet
on funding rates but the hedged book desks actually run:

- A *pair unit* for symbol s is ``long 1 coin spot + short 1 coin perp``.
  Directional PnL cancels: spot leg gains what the perp leg loses. Residual
  PnL is the basis move ``(spot − perp)`` plus funding receipts.
- Funding is cash: at each settlement, ``cash += units · perp_mark · rate``
  (short perp receives positive funding). Only positive-funding harvesting is
  modeled — negative funding would need spot borrow, which this book does not
  assume.
- The spot leg is fully paid from cash (no free leverage): order deltas are
  clamped so ``cash`` never drops below a margin buffer.
- The perp leg still carries maintenance margin: the wick-paranoid check marks
  the short at the bar's HIGH and the spot unwind at the bar's LOW, then
  force-unwinds pairs largest-first — the adverse ordering, since intra-bar
  sequencing is unknowable from OHLCV.

Same conventions as ``run_perp_backtest``: decisions on bar t fill at
t+1+fill_delay open, costs via ``total_cost`` on both legs, kill switch and
risk gate apply, and every result is ``research_only``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _make_order,
    _target_weight_map,
    _valid_price,
)
from quant_fund.backtest.perp_engine import (
    OverlayScaler,
    _bar_enrichment,
    _funding_by_time,
    infer_periods_per_year,
)
from quant_fund.config.models import AppConfig
from quant_fund.execution.costs import total_cost
from quant_fund.metrics.analytics import book_diagnostics
from quant_fund.metrics.returns import cagr, max_drawdown, sharpe_ratio
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.portfolio.risk_gate import check_order
from quant_fund.schemas.errors import KillSwitchActive, RiskGateRejected

_CARRY_COLS = ("event_time", "security_id", "target_weight")


@dataclass
class CarryBook:
    """Cash wallet + paired positions.

    ``units[s]`` > 0 means: long ``units`` coins spot, short ``units`` coins
    perp at vwap ``perp_entry[s]``. Spot has no entry — its PnL flows through
    cash at trade time.
    """

    cash: float
    units: dict[str, float] = field(default_factory=dict)
    perp_entry: dict[str, float] = field(default_factory=dict)

    def equity(self, spot_marks: dict[str, float], perp_marks: dict[str, float]) -> float:
        upnl = sum(
            -u * (perp_marks[s] - self.perp_entry.get(s, perp_marks[s]))
            for s, u in self.units.items()
            if abs(u) > 1e-12 and s in perp_marks
        )
        spot_val = sum(
            u * spot_marks[s] for s, u in self.units.items() if abs(u) > 1e-12 and s in spot_marks
        )
        return self.cash + spot_val + upnl

    def perp_gross(self, perp_marks: dict[str, float]) -> float:
        return sum(
            abs(u) * perp_marks[s]
            for s, u in self.units.items()
            if abs(u) > 1e-12 and s in perp_marks
        )


def _pair_bars(
    perp_bars: pl.DataFrame, spot_bars: pl.DataFrame
) -> dict[datetime, dict[str, dict[str, float]]]:
    """Inner-join perp and spot bars on (event_time, security_id).

    A pair can only be marked or traded when both venues printed a bar in the
    same window — no cross-venue stale-price hedging.
    """
    p = _bar_enrichment(perp_bars).rename(
        {"open": "po", "high": "ph", "low": "pl", "close": "pc", "volume": "pv"}
    )
    s = _bar_enrichment(spot_bars).rename(
        {"open": "so", "high": "sh", "low": "sl", "close": "sc", "volume": "sv"}
    )
    keep_p = ["event_time", "security_id", "po", "ph", "pl", "pc", "adv", "vol_20"]
    keep_s = [
        "event_time",
        "security_id",
        "so",
        "sh",
        "sl",
        "sc",
        pl.col("adv").alias("sadv"),
        pl.col("vol_20").alias("svol"),
    ]
    joined = p.select(keep_p).join(s.select(keep_s), on=["event_time", "security_id"], how="inner")
    out: dict[datetime, dict[str, dict[str, float]]] = {}
    for row in joined.iter_rows(named=True):
        slot = out.setdefault(row["event_time"], {})
        sid = str(row["security_id"])
        if sid in slot:
            raise ValueError(f"duplicate paired bar for {sid} at {row['event_time']}")
        slot[sid] = row
    return out


@dataclass
class _CarryBarMarks:
    """Opens, closes, ages, and liquidity stats for one paired bar."""

    exec_spot: dict[str, float]
    exec_perp: dict[str, float]
    close_spot: dict[str, float]
    close_perp: dict[str, float]
    ages: dict[str, int]
    advs_p: dict[str, float]
    vols_p: dict[str, float]
    advs_s: dict[str, float]
    vols_s: dict[str, float]


def _carry_bar_marks(
    slot: dict[str, dict[str, float]],
    last_spot: dict[str, float],
    last_perp: dict[str, float],
    mark_ages: dict[str, int],
) -> _CarryBarMarks:
    """Build execution and close marks from one joined perp/spot slot."""
    exec_spot: dict[str, float] = {}
    exec_perp: dict[str, float] = {}
    close_spot = dict(last_spot)
    close_perp = dict(last_perp)
    next_ages = dict(mark_ages)
    advs_p: dict[str, float] = {}
    vols_p: dict[str, float] = {}
    advs_s: dict[str, float] = {}
    vols_s: dict[str, float] = {}
    marked: set[str] = set()
    for sid, row in slot.items():
        po, pc = _valid_price(row["po"]), _valid_price(row["pc"])
        so, sc = _valid_price(row["so"]), _valid_price(row["sc"])
        if po is not None:
            exec_perp[sid] = po
        if so is not None:
            exec_spot[sid] = so
        if pc is not None and sc is not None:
            close_perp[sid] = pc
            close_spot[sid] = sc
            next_ages[sid] = 0
            marked.add(sid)
        advs_p[sid] = _valid_price(row.get("adv")) or 1.0
        vols_p[sid] = _valid_price(row.get("vol_20")) or 0.02
        advs_s[sid] = _valid_price(row.get("sadv")) or 1.0
        vols_s[sid] = _valid_price(row.get("svol")) or 0.02
    for sid in set(close_perp) & set(close_spot):
        if sid not in marked:
            next_ages[sid] = next_ages.get(sid, 0) + 1
    return _CarryBarMarks(
        exec_spot=exec_spot,
        exec_perp=exec_perp,
        close_spot=close_spot,
        close_perp=close_perp,
        ages=next_ages,
        advs_p=advs_p,
        vols_p=vols_p,
        advs_s=advs_s,
        vols_s=vols_s,
    )


def _apply_carry_funding(
    book: CarryBook,
    events: list[tuple[str, float]],
    last_perp: dict[str, float],
    attr_for: Callable[[str], dict[str, float]],
) -> tuple[float, float, int]:
    """Apply one bar's funding cashflows. Returns received, paid, and count."""
    received = 0.0
    paid = 0.0
    applied = 0
    for sid, rate in events:
        u = book.units.get(sid, 0.0)
        if abs(u) < 1e-12:
            continue
        mark = last_perp.get(sid)
        if mark is None:
            continue
        flow = u * mark * rate  # short perp: positive rate → receive
        book.cash += flow
        attr_for(sid)["funding"] += flow
        if flow >= 0:
            received += flow
        else:
            paid += -flow
        applied += 1
    return received, paid, applied


def _carry_liquidation_marks(
    slot: dict[str, dict[str, float]],
    last_spot: dict[str, float],
    last_perp: dict[str, float],
    units: dict[str, float],
    *,
    on_wick: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    """Adverse perp mark for a short, with the hedge still at the close.

    The venue can tag the SHORT at the bar's adverse excursion (perp high).
    The hedge leg is ours to unwind, so it marks at the bar's close — using
    the same bar's spot LOW while the perp sits at its HIGH assumes two
    opposite extremes filled simultaneously on two venues, fabricating a
    cross-venue basis that never traded.
    """
    liq_perp = dict(last_perp)
    liq_spot = dict(last_spot)
    if on_wick:
        for sid, row in slot.items():
            if abs(units.get(sid, 0.0)) < 1e-12:
                continue
            liq_perp[sid] = _valid_price(row["ph"]) or last_perp.get(sid, 0.0)
    return liq_perp, liq_spot


def _carry_attr_row(attr: dict[str, dict[str, float]], sid: str) -> dict[str, float]:
    return attr.setdefault(
        sid, {"funding": 0.0, "realized": 0.0, "fees": 0.0, "liq_realized": 0.0, "liq_fee": 0.0}
    )


def _carry_pair_costs(
    delta: float,
    po: float,
    so: float,
    sid: str,
    advs_p: dict[str, float],
    vols_p: dict[str, float],
    advs_s: dict[str, float],
    vols_s: dict[str, float],
    config: AppConfig,
) -> tuple[dict[str, float | str], dict[str, float | str]]:
    costs_p = total_cost(delta, po, advs_p.get(sid, 1.0), vols_p.get(sid, 0.02), config.costs)
    costs_s = total_cost(delta, so, advs_s.get(sid, 1.0), vols_s.get(sid, 0.02), config.costs)
    return costs_p, costs_s


def _carry_leverage_clamp(
    delta: float,
    current: float,
    po: float,
    equity: float,
    book: CarryBook,
    mark_p: dict[str, float],
    perp_cfg: Any,
) -> tuple[float, str]:
    """Clamp projected perp gross to max leverage. Status ``ok`` or ``margin_reject``."""
    # leverage cap on the perp leg (projected gross notional vs
    # equity): clamp the resulting position to the headroom left
    # after other pairs — capping |delta| on top of existing gross
    # would block deleveraging and trap an over-cap book.
    gross_now = book.perp_gross(mark_p)
    cap = perp_cfg.max_leverage * equity
    others_gross = gross_now - abs(current) * po
    projected = current + delta
    room_qty = max(0.0, cap - others_gross) / po
    if abs(projected) <= room_qty:
        return delta, "ok"
    capped_delta = np.sign(projected) * room_qty - current
    if capped_delta == 0.0 or np.sign(capped_delta) != np.sign(delta):
        return delta, "margin_reject"
    return float(capped_delta), "ok"


def _carry_budget_clamp(
    delta: float,
    po: float,
    so: float,
    equity: float,
    book: CarryBook,
    cash_buffer_frac: float,
    sid: str,
    advs_p: dict[str, float],
    vols_p: dict[str, float],
    advs_s: dict[str, float],
    vols_s: dict[str, float],
    config: AppConfig,
    costs_p: dict[str, float | str],
    costs_s: dict[str, float | str],
) -> tuple[float, dict[str, float | str], dict[str, float | str], str]:
    """Binary-search a cash-affordable size when costs + notional exceed buffer."""
    # Both legs' costs are cash debits too. The notional-only
    # clamp above is merely an upper bound; solve the monotone
    # spot-notional + pair-cost budget after the other caps.
    budget = max(0.0, book.cash - cash_buffer_frac * equity)
    required = delta * so + float(costs_p["total"]) + float(costs_s["total"])
    if required <= budget:
        return delta, costs_p, costs_s, "ok"
    lo, hi = 0.0, float(delta)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        cp, cs = _carry_pair_costs(mid, po, so, sid, advs_p, vols_p, advs_s, vols_s, config)
        if mid * so + float(cp["total"]) + float(cs["total"]) <= budget:
            lo = mid
        else:
            hi = mid
    delta = lo
    if delta * so < 1.0 or delta * po < 1.0:
        return delta, costs_p, costs_s, "skip"
    costs_p, costs_s = _carry_pair_costs(delta, po, so, sid, advs_p, vols_p, advs_s, vols_s, config)
    return delta, costs_p, costs_s, "ok"


def _carry_clamp_delta(
    *,
    delta: float,
    current: float,
    po: float,
    so: float,
    equity: float,
    book: CarryBook,
    mark_p: dict[str, float],
    cash_buffer_frac: float,
    perp_cfg: Any,
    advs_p: dict[str, float],
    vols_p: dict[str, float],
    advs_s: dict[str, float],
    vols_s: dict[str, float],
    sid: str,
    config: AppConfig,
) -> tuple[float, dict[str, float | str], dict[str, float | str], str]:
    """Apply cash/leverage/participation clamps.

    Status is ``ok``, ``skip``, or ``margin_reject`` (leverage headroom blocked).
    """
    if abs(delta) * so < 1.0:
        return delta, {}, {}, "skip"
    if delta > 0:
        # spot leg is paid from cash — clamp to liquidity minus buffer
        affordable = max(0.0, (book.cash - cash_buffer_frac * equity) / so)
        delta = min(delta, affordable)
        if delta * so < 1.0:
            return delta, {}, {}, "skip"
    delta, lev_status = _carry_leverage_clamp(delta, current, po, equity, book, mark_p, perp_cfg)
    if lev_status != "ok":
        return delta, {}, {}, lev_status
    if abs(delta) * po < 1.0:
        return delta, {}, {}, "skip"
    costs_p, costs_s = _carry_pair_costs(delta, po, so, sid, advs_p, vols_p, advs_s, vols_s, config)
    max_qty_p = config.costs.participation_limit * (advs_p.get(sid, 1.0) / po)
    max_qty_s = config.costs.participation_limit * (advs_s.get(sid, 1.0) / so)
    max_qty = min(max_qty_p, max_qty_s)
    if abs(delta) > max_qty > 0:
        delta = float(np.sign(delta) * max_qty)
        costs_p, costs_s = _carry_pair_costs(
            delta, po, so, sid, advs_p, vols_p, advs_s, vols_s, config
        )
    if delta <= 0:
        return delta, costs_p, costs_s, "ok"
    return _carry_budget_clamp(
        delta,
        po,
        so,
        equity,
        book,
        cash_buffer_frac,
        sid,
        advs_p,
        vols_p,
        advs_s,
        vols_s,
        config,
        costs_p,
        costs_s,
    )


def _carry_apply_fill(
    *,
    book: CarryBook,
    sid: str,
    delta: float,
    po: float,
    so: float,
    costs_p: dict[str, float | str],
    costs_s: dict[str, float | str],
    spot_vwap: dict[str, float],
    attr_row: dict[str, float],
    cost_sum: dict[str, float],
    fill_rows: list[dict[str, Any]],
    dt: datetime,
    signal_t: datetime,
    nav_safe: float,
) -> float:
    """Mutate book/attribution for one fill; return turnover contribution."""
    fee = float(costs_p["total"]) + float(costs_s["total"])
    current = book.units.get(sid, 0.0)
    new_units = current + delta
    # cash: spot leg pays/receives full notional; perp leg only settles
    # realized PnL on quantity reductions (same convention as perp engine)
    book.cash -= delta * so
    old_entry = book.perp_entry.get(sid, po)
    old_spot = spot_vwap.get(sid, so)
    if delta < 0:
        book.cash += (-delta) * (old_entry - po)  # buy back short: realize
        attr_row["realized"] += (-delta) * (so - old_spot) + (-delta) * (old_entry - po)
    if abs(current) < 1e-12:
        book.perp_entry[sid] = po
        spot_vwap[sid] = so
    elif abs(new_units) > abs(current):
        book.perp_entry[sid] = (current * old_entry + delta * po) / new_units
        spot_vwap[sid] = (current * old_spot + delta * so) / new_units
    elif abs(new_units) < 1e-12:
        book.perp_entry.pop(sid, None)
        spot_vwap.pop(sid, None)
    if abs(new_units) < 1e-12:
        book.units.pop(sid, None)
    else:
        book.units[sid] = new_units
    book.cash -= fee
    attr_row["fees"] += fee
    for k in ("commission", "spread", "impact"):
        cost_sum[k] += float(costs_p[k]) + float(costs_s[k])
    fill_rows.append(
        {
            "fill_time": dt,
            "signal_time": signal_t,
            "security_id": sid,
            "quantity": delta,
            "spot_price": so,
            "perp_price": po,
            "fee": float(costs_p["commission"]) + float(costs_s["commission"]),
            "spread_cost": float(costs_p["spread"]) + float(costs_s["spread"]),
            "impact_cost": float(costs_p["impact"]) + float(costs_s["impact"]),
        }
    )
    return abs(delta) * so / nav_safe


def _carry_execute_due(
    *,
    i: int,
    dt: datetime,
    book: CarryBook,
    pending: dict[str, tuple[float, datetime, int]],
    pending_exec_at: dict[str, int],
    exec_perp: dict[str, float],
    exec_spot: dict[str, float],
    mark_p: dict[str, float],
    mark_s: dict[str, float],
    advs_p: dict[str, float],
    vols_p: dict[str, float],
    advs_s: dict[str, float],
    vols_s: dict[str, float],
    cash_buffer_frac: float,
    perp_cfg: Any,
    config: AppConfig,
    kill: KillSwitch,
    spot_vwap: dict[str, float],
    attr: dict[str, dict[str, float]],
    cost_sum: dict[str, float],
    fill_rows: list[dict[str, Any]],
    order_seq: int,
) -> tuple[float, int, int, int, int]:
    """Fill pending pair orders due at bar *i*.

    Returns ``(traded_turn, margin_rejects, risk_rejects, halts, order_seq)``.
    """
    traded_turn = 0.0
    margin_reject_count = 0
    reject_count = 0
    halt_count = 0
    due = [sid for sid, idx in pending_exec_at.items() if idx <= i]
    for sid in sorted(due):
        target_w, signal_t, _ = pending.pop(sid)
        del pending_exec_at[sid]
        po = exec_perp.get(sid)
        so = exec_spot.get(sid)
        if po is None or so is None:
            continue  # cannot hedge without both prints — order lapses
        equity = book.equity(mark_s, mark_p)
        desired_units = max(0.0, target_w * equity / so)
        current = book.units.get(sid, 0.0)
        delta = desired_units - current
        delta, costs_p, costs_s, status = _carry_clamp_delta(
            delta=delta,
            current=current,
            po=po,
            so=so,
            equity=equity,
            book=book,
            mark_p=mark_p,
            cash_buffer_frac=cash_buffer_frac,
            perp_cfg=perp_cfg,
            advs_p=advs_p,
            vols_p=vols_p,
            advs_s=advs_s,
            vols_s=vols_s,
            sid=sid,
            config=config,
        )
        if status == "margin_reject":
            margin_reject_count += 1
            continue
        if status != "ok":
            continue
        try:
            kill.assert_new_orders_allowed()
        except KillSwitchActive:
            halt_count += 1
            continue
        nav_safe = max(equity, 1e-12)
        current_w = current * so / nav_safe
        order_seq += 1
        order = _make_order(
            sid=sid, delta=delta, signal_time=signal_t, order_time=dt, order_seq=order_seq
        )
        try:
            check_order(
                order,
                nav=equity,
                price=so,
                current_weight=current_w,
                gross_after=0.0,  # pair book: net market exposure ≈ 0 by construction
                net_after=0.0,
                participation=abs(delta) * po / max(advs_p.get(sid, 1.0), 1e-12),
                predicted_vol=vols_p.get(sid, 0.02),
                config=config,
            )
        except RiskGateRejected:
            reject_count += 1
            continue
        traded_turn += _carry_apply_fill(
            book=book,
            sid=sid,
            delta=delta,
            po=po,
            so=so,
            costs_p=costs_p,
            costs_s=costs_s,
            spot_vwap=spot_vwap,
            attr_row=_carry_attr_row(attr, sid),
            cost_sum=cost_sum,
            fill_rows=fill_rows,
            dt=dt,
            signal_t=signal_t,
            nav_safe=nav_safe,
        )
    return traded_turn, margin_reject_count, reject_count, halt_count, order_seq


def _carry_liquidate(
    *,
    book: CarryBook,
    slot: dict[str, dict[str, float]],
    last_perp: dict[str, float],
    last_spot: dict[str, float],
    perp_cfg: Any,
    advs_s: dict[str, float],
    vols_s: dict[str, float],
    config: AppConfig,
    spot_vwap: dict[str, float],
    attr: dict[str, dict[str, float]],
    cost_sum: dict[str, float],
    liq_rows: list[dict[str, Any]],
    dt: datetime,
) -> tuple[int, float]:
    """Wick-paranoid MMR liquidation loop. Returns (count, fee_cost)."""
    liq_perp, liq_spot = _carry_liquidation_marks(
        slot,
        last_spot,
        last_perp,
        book.units,
        on_wick=perp_cfg.liquidation_on_wick,
    )

    def _mmr_deficit(lp: dict[str, float] = liq_perp, ls: dict[str, float] = liq_spot) -> float:
        return float(perp_cfg.maint_margin_ratio * book.perp_gross(lp) - book.equity(ls, lp))

    liquidation_count = 0
    liquidation_cost = 0.0
    while _mmr_deficit() > 0 and book.perp_gross(liq_perp) > 0:
        victim = max(
            (s for s in book.units if abs(book.units[s]) > 1e-12),
            key=lambda s: abs(book.units[s]) * liq_perp.get(s, 0.0),
            default=None,
        )
        if victim is None:
            break
        u = book.units.pop(victim)
        entry = book.perp_entry.pop(victim, liq_perp.get(victim, 0.0))
        s_vwap = spot_vwap.pop(victim, liq_spot.get(victim, 0.0))
        liq_s = liq_spot.get(victim, 0.0)
        liq_p = liq_perp.get(victim, entry)
        # venue force-covers the short at the adverse liq price; the
        # orphaned hedge is unwound at market (bar close) with normal
        # costs, plus the liquidation fee on the perp leg
        book.cash += u * liq_s  # dump spot
        book.cash += u * (entry - liq_p)  # cover short
        liq_spot_costs = total_cost(
            u, liq_s, advs_s.get(victim, 1.0), vols_s.get(victim, 0.02), config.costs
        )
        book.cash -= float(liq_spot_costs["total"])
        liq_fee = u * liq_perp.get(victim, 0.0) * perp_cfg.liquidation_fee_bps / 1e4
        book.cash -= liq_fee
        liq_realized = u * (liq_s - s_vwap) + u * (entry - liq_p) - float(liq_spot_costs["total"])
        row_attr = _carry_attr_row(attr, victim)
        row_attr["liq_realized"] += liq_realized
        row_attr["liq_fee"] += liq_fee
        for k in ("commission", "spread", "impact"):
            cost_sum[k] += float(liq_spot_costs[k])
        liquidation_cost += liq_fee
        liquidation_count += 1
        liq_rows.append(
            {
                "event_time": dt,
                "security_id": victim,
                "quantity": u,
                "liq_perp_price": liq_perp.get(victim),
                "liq_spot_price": liq_spot.get(victim),
                "fee": liq_fee,
                "realized_pnl": liq_realized,
                "perp_entry": entry,
                "spot_vwap": s_vwap,
            }
        )
    return liquidation_count, liquidation_cost


def _carry_build_metrics(
    *,
    eq: pl.DataFrame,
    ppy: float,
    initial_nav: float,
    cost_sum: dict[str, float],
    funding_received: float,
    funding_paid: float,
    funding_events_applied: int,
    funding_events_dropped: int,
    liquidation_count: int,
    liquidation_cost: float,
    margin_reject_count: int,
    ruined: bool,
    reject_count: int,
    halt_count: int,
    pnl_attribution: dict[str, Any],
    config: AppConfig,
) -> dict[str, float | str | int | bool | dict[str, Any]]:
    if eq.height >= 2:
        rets = eq["nav"].pct_change().drop_nulls().to_numpy()
        sr = sharpe_ratio(rets, periods_per_year=ppy)
        navs_list = [float(v) for v in eq["nav"].to_list()]
        turns = [float(v) for v in eq["turnover"].to_list()]
        diag = book_diagnostics(
            rets,
            gross_exposure=eq["gross"].to_numpy(),
            net_exposure=eq["net"].to_numpy(),
            turnover=np.asarray(turns, dtype=float),
            label="CARRY_BACKTEST_SIM",
            data_source="file",
        )
        metrics: dict[str, float | str | int | bool | dict[str, Any]] = {
            "total_return": navs_list[-1] / float(initial_nav) - 1.0,
            "cagr": cagr(rets, periods_per_year=ppy),
            "sharpe": sr["sharpe"],
            "n": sr["n"],
            "periods_per_year": ppy,
            "max_drawdown": max_drawdown(rets),
            "mean_turnover": float(np.mean(turns)),
            "commission": cost_sum["commission"],
            "spread": cost_sum["spread"],
            "impact": cost_sum["impact"],
            "funding_received_total": funding_received,
            "funding_paid_total": funding_paid,
            "funding_net": funding_received - funding_paid,
            "funding_events_applied": funding_events_applied,
            "funding_events_dropped": funding_events_dropped,
            "liquidation_count": liquidation_count,
            "liquidation_cost": liquidation_cost,
            "margin_rejects": margin_reject_count,
            "ruined": ruined,
            "flag_high_sharpe": sr["flag_high_sharpe"],
            "risk_gate_rejects": reject_count,
            "kill_switch_halts": halt_count,
            "analytics": diag,
            "book_type": "delta_neutral_carry",
            "pnl_attribution": pnl_attribution,
            "research_only": True,
            "live_pnl_claim": False,
        }
    else:
        metrics = {
            "total_return": 0.0,
            "sharpe": float("nan"),
            "n": 0,
            "periods_per_year": ppy,
            "ruined": ruined,
            "liquidation_count": liquidation_count,
            "funding_events_applied": funding_events_applied,
            "book_type": "delta_neutral_carry",
            "pnl_attribution": pnl_attribution,
            "research_only": True,
            "live_pnl_claim": False,
        }
    if config.costs.frictionless:
        metrics["label"] = "FRICTIONLESS RESEARCH ONLY"
    metrics["data_source"] = "file"
    return metrics


def _carry_weight_map(weights: pl.DataFrame) -> dict[datetime, dict[str, float]]:
    if weights.height:
        _ = _target_weight_map(weights)
    if (weights["target_weight"] < 0).any() if weights.height else False:
        raise ValueError("carry book is long-spot/short-perp only: target_weight must be >= 0")
    wmap_by_time: dict[datetime, dict[str, float]] = {}
    for wrow in weights.select(_CARRY_COLS).iter_rows(named=True):
        wmap_by_time.setdefault(wrow["event_time"], {})[str(wrow["security_id"])] = float(
            wrow["target_weight"]
        )
    return wmap_by_time


def _carry_raise_if_stale(
    book: CarryBook, mark_ages: dict[str, int], stale_price_bars: int
) -> None:
    stale_held = {
        sid: mark_ages.get(sid)
        for sid, u in book.units.items()
        if abs(u) > 1e-12 and mark_ages.get(sid, 10**9) > stale_price_bars
    }
    if stale_held:
        details = ", ".join(f"{s}={a}" for s, a in stale_held.items())
        raise StaleValuationError("held carry pair valuation stale beyond limit: " + details)


def _carry_queue_targets(
    *,
    i: int,
    dt: datetime,
    times: list[datetime],
    fills_delay_bars: int,
    wmap_by_time: dict[datetime, dict[str, float]],
    scaler: OverlayScaler | None,
    pending: dict[str, tuple[float, datetime, int]],
    pending_exec_at: dict[str, int],
) -> None:
    if i + 1 + fills_delay_bars >= len(times):
        return
    targets = wmap_by_time.get(dt)
    if not targets:
        return
    if scaler is not None:
        targets = scaler.scale(dt, dict(targets))
    exec_idx = i + 1 + fills_delay_bars
    for sid, w in targets.items():
        pending[sid] = (w, dt, exec_idx)
        pending_exec_at[sid] = exec_idx


def _carry_pnl_attribution(
    *,
    book: CarryBook,
    last_spot: dict[str, float],
    last_perp: dict[str, float],
    spot_vwap: dict[str, float],
    attr: dict[str, dict[str, float]],
    liq_rows: list[dict[str, Any]],
    navs: list[dict[str, Any]],
    initial_nav: float,
) -> dict[str, Any]:
    # Per-symbol attribution: unrealized for still-open pairs marks at last
    # close vs vwap entries; realized/funding/fees/liq were tracked at events.
    for sid, u in book.units.items():
        if abs(u) < 1e-12:
            continue
        s_now = last_spot.get(sid)
        p_now = last_perp.get(sid)
        if s_now is None or p_now is None:
            continue
        _carry_attr_row(attr, sid)["unrealized"] = u * (s_now - spot_vwap.get(sid, s_now)) + u * (
            book.perp_entry.get(sid, p_now) - p_now
        )
    attr_totals = {
        k: sum(row.get(k, 0.0) for row in attr.values())
        for k in ("funding", "realized", "fees", "liq_realized", "liq_fee", "unrealized")
    }
    final_equity = float(navs[-1]["nav"]) if navs else float(initial_nav)
    conservation_error = (
        final_equity
        - float(initial_nav)
        - (
            attr_totals["funding"]
            + attr_totals["realized"]
            + attr_totals["liq_realized"]
            - attr_totals["fees"]
            - attr_totals["liq_fee"]
            + attr_totals["unrealized"]
        )
    )
    return {
        "by_symbol": {sid: row for sid, row in sorted(attr.items())},
        "totals": attr_totals,
        "conservation_error": conservation_error,
        "liquidation_events": liq_rows,
    }


@dataclass
class _CarryBarOutcome:
    """Close marks and counter deltas produced by one bar step."""

    last_spot: dict[str, float]
    last_perp: dict[str, float]
    mark_ages: dict[str, int]
    order_seq: int
    margin_rejects: int
    risk_rejects: int
    halts: int
    funding_received: float
    funding_paid: float
    funding_applied: int
    liquidation_count: int
    liquidation_cost: float
    ruined: bool


def _carry_run_bar(
    *,
    i: int,
    dt: datetime,
    times: list[datetime],
    slot: dict[str, dict[str, float]],
    book: CarryBook,
    last_spot: dict[str, float],
    last_perp: dict[str, float],
    mark_ages: dict[str, int],
    pending: dict[str, tuple[float, datetime, int]],
    pending_exec_at: dict[str, int],
    fund_events: list[tuple[str, float]],
    cash_buffer_frac: float,
    perp_cfg: Any,
    config: AppConfig,
    kill: KillSwitch,
    spot_vwap: dict[str, float],
    attr: dict[str, dict[str, float]],
    cost_sum: dict[str, float],
    fill_rows: list[dict[str, Any]],
    liq_rows: list[dict[str, Any]],
    navs: list[dict[str, Any]],
    wmap_by_time: dict[datetime, dict[str, float]],
    scaler: OverlayScaler | None,
    order_seq: int,
) -> _CarryBarOutcome:
    """One bar: marks → fills → funding → liquidation → NAV/queue."""
    marks = _carry_bar_marks(slot, last_spot, last_perp, mark_ages)
    # Marks knowable at execution time: before this bar's close prints.
    # Non-executing names valued at this bar's close would leak future
    # prices into pair sizing and the leverage cap.
    prev_spot, prev_perp = last_spot, last_perp
    last_spot, last_perp, mark_ages = marks.close_spot, marks.close_perp, marks.ages
    _carry_raise_if_stale(book, mark_ages, config.risk_gate.stale_price_bars)

    # --- execute pending pair orders at this bar's opens ---
    mark_p = {**prev_perp, **marks.exec_perp}
    mark_s = {**prev_spot, **marks.exec_spot}
    traded_turn, mrej, rrej, halts, order_seq = _carry_execute_due(
        i=i,
        dt=dt,
        book=book,
        pending=pending,
        pending_exec_at=pending_exec_at,
        exec_perp=marks.exec_perp,
        exec_spot=marks.exec_spot,
        mark_p=mark_p,
        mark_s=mark_s,
        advs_p=marks.advs_p,
        vols_p=marks.vols_p,
        advs_s=marks.advs_s,
        vols_s=marks.vols_s,
        cash_buffer_frac=cash_buffer_frac,
        perp_cfg=perp_cfg,
        config=config,
        kill=kill,
        spot_vwap=spot_vwap,
        attr=attr,
        cost_sum=cost_sum,
        fill_rows=fill_rows,
        order_seq=order_seq,
    )

    # --- funding on the perp leg ---
    # Funding accrues at the bar's close mark, matching the perp
    # engine's convention (exec opens in mark_p are not close marks).
    received, paid, applied = _apply_carry_funding(
        book, fund_events, last_perp, lambda sid: _carry_attr_row(attr, sid)
    )

    # --- wick-paranoid liquidation on the perp leg ---
    liq_n, liq_c = _carry_liquidate(
        book=book,
        slot=slot,
        last_perp=last_perp,
        last_spot=last_spot,
        perp_cfg=perp_cfg,
        advs_s=marks.advs_s,
        vols_s=marks.vols_s,
        config=config,
        spot_vwap=spot_vwap,
        attr=attr,
        cost_sum=cost_sum,
        liq_rows=liq_rows,
        dt=dt,
    )

    # --- mark close, queue targets ---
    nav_close = book.equity(last_spot, last_perp)
    gross_p = book.perp_gross(last_perp) / max(abs(nav_close), 1e-12)
    navs.append(
        {
            "event_time": dt,
            "nav": nav_close,
            "gross": gross_p,
            "net": gross_p,  # pair book: gross==net exposure metric is the perp leg
            "turnover": traded_turn,
        }
    )
    if scaler is not None:
        scaler.observe(dt, nav_close)
    ruined = nav_close <= 0
    if not ruined:
        _carry_queue_targets(
            i=i,
            dt=dt,
            times=times,
            fills_delay_bars=perp_cfg.fill_delay_bars,
            wmap_by_time=wmap_by_time,
            scaler=scaler,
            pending=pending,
            pending_exec_at=pending_exec_at,
        )
    return _CarryBarOutcome(
        last_spot=last_spot,
        last_perp=last_perp,
        mark_ages=mark_ages,
        order_seq=order_seq,
        margin_rejects=mrej,
        risk_rejects=rrej,
        halts=halts,
        funding_received=received,
        funding_paid=paid,
        funding_applied=applied,
        liquidation_count=liq_n,
        liquidation_cost=liq_c,
        ruined=ruined,
    )


def run_carry_backtest(
    perp_bars: pl.DataFrame,
    spot_bars: pl.DataFrame,
    funding: pl.DataFrame | None,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    scaler: OverlayScaler | None = None,
    cash_buffer_frac: float = 0.02,
) -> BacktestResult:
    """`weights` = target fraction of equity per pair (target_weight ≥ 0 only).

    Negative weights are rejected — negative-funding harvest would need spot
    borrow this book does not model.
    Funding follows ``run_perp_backtest``: exact application bar-label matches,
    applied after open fills and before liquidation using the perp close
    mark (or a permitted carried mark). Unmapped off-grid events are skipped and
    counted in ``funding_events_dropped`` when reported. By default the funding
    ``event_time`` supplies the label; explicit ``application_time`` mapping
    does not imply automatic interval bucketing.
    """
    wmap_by_time = _carry_weight_map(weights)
    by_time = _pair_bars(perp_bars, spot_bars)
    times = sorted(by_time)
    perp_cfg = config.perp
    ppy = (
        float(perp_cfg.periods_per_year_override)
        if perp_cfg.periods_per_year_override is not None
        else infer_periods_per_year(times, perp_cfg.bar_seconds_hint)
    )
    fund_map = (
        _funding_by_time(
            funding, multiplier=perp_cfg.funding_spike_multiplier, bar_times=set(times)
        )
        if perp_cfg.funding_enabled
        else {}
    )

    book = CarryBook(cash=float(initial_nav))
    pending: dict[str, tuple[float, datetime, int]] = {}
    pending_exec_at: dict[str, int] = {}
    navs: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    liq_rows: list[dict[str, Any]] = []
    cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0}
    funding_received = 0.0
    funding_paid = 0.0
    funding_events_applied = 0
    # Funding events whose timestamp never equals a bar time are silently
    # skipped by ``fund_map.get(dt)`` — count them so feed misalignment is
    # visible instead of fabricating zero-funding receipts.
    bar_times = set(times)
    funding_events_dropped = sum(
        len(events) for ts, events in fund_map.items() if ts not in bar_times
    )
    liquidation_count = 0
    liquidation_cost = 0.0
    margin_reject_count = 0
    reject_count = 0
    halt_count = 0
    order_seq = 0
    last_spot: dict[str, float] = {}
    last_perp: dict[str, float] = {}
    mark_ages: dict[str, int] = {}
    kill = KillSwitch(config.kill_switch)
    ruined = False
    # Per-symbol P&L attribution: spot vwap cost lets every fill/liquidation
    # be decomposed into realized pair P&L vs funding vs fees, so a receipt
    # can show exactly where equity went instead of a bare NAV path.
    spot_vwap: dict[str, float] = {}
    attr: dict[str, dict[str, float]] = {}

    for i, dt in enumerate(times):
        bar = _carry_run_bar(
            i=i,
            dt=dt,
            times=times,
            slot=by_time[dt],
            book=book,
            last_spot=last_spot,
            last_perp=last_perp,
            mark_ages=mark_ages,
            pending=pending,
            pending_exec_at=pending_exec_at,
            fund_events=fund_map.get(dt, []),
            cash_buffer_frac=cash_buffer_frac,
            perp_cfg=perp_cfg,
            config=config,
            kill=kill,
            spot_vwap=spot_vwap,
            attr=attr,
            cost_sum=cost_sum,
            fill_rows=fill_rows,
            liq_rows=liq_rows,
            navs=navs,
            wmap_by_time=wmap_by_time,
            scaler=scaler,
            order_seq=order_seq,
        )
        last_spot, last_perp, mark_ages = bar.last_spot, bar.last_perp, bar.mark_ages
        order_seq = bar.order_seq
        margin_reject_count += bar.margin_rejects
        reject_count += bar.risk_rejects
        halt_count += bar.halts
        funding_received += bar.funding_received
        funding_paid += bar.funding_paid
        funding_events_applied += bar.funding_applied
        liquidation_count += bar.liquidation_count
        liquidation_cost += bar.liquidation_cost
        if bar.ruined:
            ruined = True
            break

    eq = pl.DataFrame(navs) if navs else pl.DataFrame({"event_time": [], "nav": []})
    pnl_attribution = _carry_pnl_attribution(
        book=book,
        last_spot=last_spot,
        last_perp=last_perp,
        spot_vwap=spot_vwap,
        attr=attr,
        liq_rows=liq_rows,
        navs=navs,
        initial_nav=initial_nav,
    )
    metrics = _carry_build_metrics(
        eq=eq,
        ppy=ppy,
        initial_nav=initial_nav,
        cost_sum=cost_sum,
        funding_received=funding_received,
        funding_paid=funding_paid,
        funding_events_applied=funding_events_applied,
        funding_events_dropped=funding_events_dropped,
        liquidation_count=liquidation_count,
        liquidation_cost=liquidation_cost,
        margin_reject_count=margin_reject_count,
        ruined=ruined,
        reject_count=reject_count,
        halt_count=halt_count,
        pnl_attribution=pnl_attribution,
        config=config,
    )
    return BacktestResult(
        equity=eq,
        fills=pl.DataFrame(fill_rows) if fill_rows else pl.DataFrame(),
        metrics=metrics,
        frictionless=config.costs.frictionless,
        source_note="delta-neutral carry: short perp + long spot, funding harvest",
    )
