"""Vectorized fast path for ``run_backtest``.

Same economics as the reference event loop — decision at close ``t`` fills at
open ``t+1`` (or same-bar close when ``use_next_open`` is false), per-order
risk gate, kill switch, stale-valuation fail-closed, sequential cash and
contributor accounting in sorted-``security_id`` order — but the per-date
panel scan runs on pre-indexed numpy matrices instead of per-row polars
iteration.

Scope (fail-closed): this path exists for the matched-workload class used by
the incumbent benchmarks — target-weight panels, market fills, the full
flat ``CostConfig`` surface, and the full ``RiskGateConfig`` surface. It
refuses configs it does not replicate (``allow_close_auction=True``,
OHLC spread calibration via ``costs.spread_estimator != "flat"``).

Float-order contract: NAV and exposure sums follow the reference engine's
summation orders — shares-dict insertion order for book NAV, sorted
contributors for projected exposures — so outputs are expected to be
bit-identical to ``run_backtest``; the conformance suite asserts that.

Accelerator: when ``numba`` is importable the inner replay dispatches to a
compiled kernel (``_replay_driver`` below) that reproduces the same
semantics — including CPython 3.12 ``sum()`` prefix-commit behaviour over
mixed float/np.float64 terms — at native speed. Without numba the
interpreted loop below runs; both paths are covered by the conformance
suite.
"""

from __future__ import annotations

import inspect
import math
from typing import Any

import numpy as np
import polars as pl

from quant_fund.backtest.engine import (
    BacktestResult,
    StaleValuationError,
    _fast_replay_is_complete,
    _fast_replay_panel_supported,
    _target_weight_map,
    _validate_bar_panel,
    run_backtest,
)
from quant_fund.compute.numba_guard import compiler_defect
from quant_fund.config.models import AppConfig, CostConfig, FillConvention, RiskGateConfig
from quant_fund.execution.spread_calibration import is_calibrated_spread_estimator
from quant_fund.monitoring.kill_switch import KillSwitch
from quant_fund.portfolio.risk_gate import LIMIT_ABS_SLACK, LIMIT_REL_SLACK, exceeds_limit, funded
from quant_fund.schemas.errors import KillSwitchActive

try:
    from quant_fund.pipeline.forecast import (
        market_risk_overlay_asof as _mro_asof,
    )
except ImportError:  # pragma: no cover - older lineage lacks the module
    _mro_asof = None  # type: ignore[assignment]

try:
    from numba import njit

    HAVE_NUMBA = True
except (ImportError, OSError):  # pragma: no cover - exercised only on numba-less envs
    # Narrowed from `except Exception` (quality ratchet): a missing numba raises
    # ImportError; a broken llvmlite native load raises OSError; exotic errors
    # propagate instead of silently degrading every kernel.
    HAVE_NUMBA = False

    def njit(*_a: Any, **_k: Any) -> Any:  # type: ignore[no-redef]
        def _deco(fn: Any) -> Any:
            return fn

        if len(_a) == 1 and callable(_a[0]) and not _k:
            return _a[0]
        return _deco


def _order_costs(
    d: float, px: float, adv_d: float, vol_d: float, cfg: Any
) -> tuple[float, float, float, float]:
    """Identical expression order/types as ``total_cost`` (commission, spread,
    impact, total). ``d``/``px`` keep their operand types — a capped delta is
    np.float64 and that propagates exactly as it does in the reference."""
    # total_cost validates inputs before the frictionless early
    # return — a non-finite delta must fail closed, not trade.
    if not math.isfinite(d) or not math.isfinite(px) or px <= 0:
        raise ValueError("quantity must be finite and price must be finite and positive")
    if cfg.frictionless:
        return 0.0, 0.0, 0.0, 0.0
    nt = abs(d) * px
    comm = abs(nt) * cfg.commission_bps / 1e4
    spr = abs(nt) * cfg.half_spread_bps / 1e4
    part = nt / max(adv_d, 1e-12)
    imp = float(nt * cfg.impact_y * vol_d * math.sqrt(part))
    bpt = abs(nt) * cfg.bps_per_turnover / 1e4
    return comm, spr, imp, float(comm + spr + imp + bpt)


def _validate_panel_fast(weights: pl.DataFrame) -> None:
    """Vectorized ``_validate_target_weight_panel`` for the hot path.

    Same observable behaviour — raises the identical errors — but checks
    duplicates/finiteness in one vectorized pass instead of ``iter_rows``.
    Any anomaly falls back to the reference ``_target_weight_map`` so the
    exact error (message + exception type) is preserved.
    """
    required = {"event_time", "security_id", "target_weight"}
    if required.difference(weights.columns):
        _target_weight_map(weights)  # raises the missing-columns error
        return
    if weights.height == 0:
        return
    dup = weights.select(pl.struct(["event_time", "security_id"]).is_duplicated().any()).item()
    if dup:
        raise ValueError("duplicate target weights for event_time/security_id")
    w = weights["target_weight"]
    try:
        vals = w.cast(pl.Float64, strict=False).to_numpy()
        ok = bool(np.isfinite(vals).all())
    except (pl.exceptions.PolarsError, TypeError, ValueError):
        # Narrowed from `except Exception` (quality ratchet): cast/to_numpy faults
        # are polars/type errors; exotic errors propagate. Any enumerated anomaly
        # still falls back to the reference validator, preserving the exact error.
        ok = False
    if not ok:
        # Reproduce the reference's first-offender error verbatim.
        _target_weight_map(weights)


_BAR_COLS = (
    "security_id",
    "event_time",
    "open",
    "close",
    "close_total_return",
    "volume",
    "adv",
    "vol_20",
)


def _bars_to_matrices(
    bars: pl.DataFrame,
) -> tuple[
    list[Any],
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    bool,
]:
    """Build per-(date, asset) matrices mirroring ``day_rows_map`` semantics.

    Returns (dates, sids, open_px, close_px, ctr, adv, vol, synthetic).
    A missing row and an invalid price are indistinguishable downstream of
    ``_valid_price``, so absent cells are simply NaN.
    """
    px = bars.select(
        "security_id",
        "event_time",
        "open",
        "close",
        "close_total_return",
        "volume",
        pl.col("adv")
        if "adv" in bars.columns
        else (pl.col("close") * pl.col("volume")).alias("adv"),
        pl.col("vol_20") if "vol_20" in bars.columns else pl.lit(0.02).alias("vol_20"),
        "source",
    )
    sids = sorted(str(s) for s in px["security_id"].unique().to_list())
    sid_idx = {s: i for i, s in enumerate(sids)}
    et_raw = px["event_time"].cast(pl.Int64).to_numpy()
    dates_ns = np.unique(et_raw)
    t_idx = np.searchsorted(dates_ns, et_raw)
    a_idx = np.array([sid_idx[str(s)] for s in px["security_id"].to_list()], dtype=np.int64)

    n_dates = int(dates_ns.size)
    n_assets = len(sids)

    def _to_f64(name: str) -> np.ndarray:
        # strict=False mirrors _valid_price: unparseable/non-finite inputs
        # become null -> NaN downstream instead of raising at ingest.
        return px[name].cast(pl.Float64, strict=False).to_numpy()

    def _mark_col(name: str) -> np.ndarray:
        # Marks use "last *valid* write wins": an invalid duplicate row does
        # not clobber an earlier valid one (dict assignment is skipped when
        # _valid_price returns None).
        arr = np.full((n_dates, n_assets), np.nan)
        out = _to_f64(name)
        ok = np.isfinite(out) & (out > 0)
        arr[t_idx[ok], a_idx[ok]] = out[ok]
        return arr

    def _raw_col(name: str) -> np.ndarray:
        # adv/vol overwrite unconditionally per row — the last duplicate row
        # wins even when invalid (the default is applied downstream).
        arr = np.full((n_dates, n_assets), np.nan)
        arr[t_idx, a_idx] = _to_f64(name)
        return arr

    open_px = _mark_col("open")
    close_px = _mark_col("close")
    ctr = _mark_col("close_total_return")
    adv = _raw_col("adv")
    vol = _raw_col("vol_20")
    synthetic = (
        "synthetic" in set(px["source"].drop_nulls().to_list()) if "source" in px.columns else False
    )
    # Sorted unique event_times. Polars unique+sort runs in native code; a
    # null event_time falls back to the interpreter path so ``sorted`` still
    # raises TypeError exactly like ``sorted(day_rows_map)``.
    et_col = px["event_time"]
    if et_col.null_count() == 0:
        dates = et_col.unique(maintain_order=False).sort().to_list()
    else:
        dates = sorted(set(et_col.to_list()))
    if not (len(dates) == n_dates):
        raise ValueError("len(dates) == n_dates")
    return dates, dates_ns, sids, open_px, close_px, ctr, adv, vol, synthetic


# ---------------------------------------------------------------------------
# Optional numba-compiled replay kernel
#
# Bit-identical contract: the kernel reproduces the interpreted loop below
# statement-for-statement, including CPython 3.12 ``sum()`` behaviour —
# exact-``float`` prefixes accumulate with Neumaier compensation and the
# first ``np.float64`` term commits the compensated partial and degrades the
# rest to a naive left fold (verified empirically against CPython 3.12.13).
# ``np.float64`` propagation is tracked via per-asset/per-scalar "np64"
# flags: values are IEEE-754 float64 either way, so flags only choose the
# summation path, never the arithmetic. Without numba the interpreted loop
# runs (identical outputs, slower).
# ---------------------------------------------------------------------------


@njit(cache=True)
def _neumaier(v: np.ndarray) -> float:
    """CPython 3.12 float fast-path sum: Neumaier compensated accumulation."""
    hi = 0.0
    lo = 0.0
    for j in range(v.shape[0]):
        x = v[j]
        t = hi + x
        if abs(hi) >= abs(x):
            lo += (hi - t) + x
        else:
            lo += (x - t) + hi
        hi = t
    # CPython commits the compensation only when it is nonzero and finite —
    # ``if (c && Py_IS_FINITE(c)) f_result += c`` — so a saturated/NaN
    # compensation is dropped rather than contaminating the result.
    if lo != 0.0 and np.isfinite(lo):
        hi += lo
    return hi


@njit(cache=True)
def _csum(v: np.ndarray, f: np.ndarray) -> float:
    """``sum()`` over terms with per-term ``np.float64`` flags.

    Exact-float prefix -> Neumaier; first np.float64 term commits the
    compensated partial (hi + lo, finite-guarded like CPython) and the
    remaining terms naive-fold.
    """
    n = v.shape[0]
    k0 = n
    for j in range(n):
        if f[j]:
            k0 = j
            break
    if k0 == n:
        return float(_neumaier(v))
    hi = 0.0
    lo = 0.0
    for j in range(k0):
        x = v[j]
        t = hi + x
        if abs(hi) >= abs(x):
            lo += (hi - t) + x
        else:
            lo += (x - t) + hi
        hi = t
    acc = hi
    if lo != 0.0 and np.isfinite(lo):
        acc += lo
    for j in range(k0, n):
        acc += v[j]
    return acc


@njit(cache=True)
def _order_costs_nb(
    d: float,
    px: float,
    adv_d: float,
    vol_d: float,
    frictionless: bool,
    commission_bps: float,
    half_spread_bps: float,
    impact_y: float,
    bps_per_turnover: float,
) -> tuple[float, float, float, float]:
    # Caller validates ``d``/``px`` first — identical ordering to the
    # reference ``total_cost`` (validation precedes the frictionless return).
    if frictionless:
        return 0.0, 0.0, 0.0, 0.0
    nt = abs(d) * px
    comm = abs(nt) * commission_bps / 1e4
    spr = abs(nt) * half_spread_bps / 1e4
    part = nt / max(adv_d, 1e-12)
    imp = float(nt * impact_y * vol_d * np.sqrt(part))
    bpt = abs(nt) * bps_per_turnover / 1e4
    return comm, spr, imp, float(comm + spr + imp + bpt)


@njit(cache=True)
def _exceeds_nb(value: float, limit: float) -> bool:
    """Numba twin of ``exceeds_limit``. Keep the slack constants identical."""
    if not np.isfinite(value) or not np.isfinite(limit):
        return True
    if not (value > limit):
        return False
    slack = LIMIT_ABS_SLACK
    rel = LIMIT_REL_SLACK * abs(limit)
    if rel > slack:
        slack = rel
    return (value - limit) > slack


@njit(cache=True)
def _funded_nb(cash: float, needed: float) -> bool:
    """Numba twin of ``funded``."""
    if not np.isfinite(cash) or not np.isfinite(needed):
        return False
    if cash >= needed:
        return True
    return not _exceeds_nb(needed, cash)


@njit(cache=True)
def _replay_kernel(
    exec_px: np.ndarray,  # (T,A) f64 — exec source (open next_open / close)
    close_px: np.ndarray,  # (T,A) f64 — decision close + mark fallback
    ctr: np.ndarray,  # (T,A) f64 — preferred mark
    adv: np.ndarray,  # (T,A) f64
    vol: np.ndarray,  # (T,A) f64
    w_mat: np.ndarray,  # (T,A) f64 — carried target grid already applied
    market_vols: np.ndarray,  # (T,)  f64 — value only valid where has_mv
    has_mv: np.ndarray,  # (T,)  bool
    use_next_open: bool,
    ref_carries: bool,
    commission_bps: float,
    half_spread_bps: float,
    impact_y: float,
    bps_per_turnover: float,
    borrow_bps_per_year: float,
    participation_limit: float,
    frictionless: bool,
    stale_price_bars: int,
    max_order_notional: float,
    max_gross: float,
    max_net: float,
    max_name: float,
    max_participation: float,
    max_predicted_vol: float,
    kill_blocks: bool,
    initial_nav: float,
    navs_out: np.ndarray,  # (T,4) f64
    navs_t: np.ndarray,  # (T,) i64
    f_qty: np.ndarray,  # (M,) f64
    f_px: np.ndarray,  # (M,) f64
    f_fee: np.ndarray,  # (M,) f64
    f_spr: np.ndarray,  # (M,) f64
    f_imp: np.ndarray,  # (M,) f64
    f_bpt: np.ndarray,  # (M,) f64 turnover bps charged in ``total``
    f_asset: np.ndarray,  # (M,) i64
    f_et: np.ndarray,  # (M,) i64
    f_st: np.ndarray,  # (M,) i64
    f_dec: np.ndarray,  # (M,) f64
    f_decok: np.ndarray,  # (M,) bool
    stale_assets: np.ndarray,  # (A,) i64
    stale_ages: np.ndarray,  # (A,) i64
    stale_ever: np.ndarray,  # (A,) bool
    term_v: np.ndarray,  # (A,) f64 work buffer
    term_f: np.ndarray,  # (A,) bool work buffer
) -> tuple[int, int, int, int, int, int, float, float, float, int, int]:
    """Returns (status, n_navs, n_fills, rejects, cash_rejects, halts,
    commission_sum, spread_sum, impact_sum, order_seq, nstale).

    status: 0 ok (incl. nav<=0 early break), 1 stale-valuation (stale_* out
    params filled in the reference's two-pass order), 2 non-finite
    delta/price.
    """
    n_dates = exec_px.shape[0]
    n_assets = exec_px.shape[1]

    shares = np.zeros(n_assets, dtype=np.float64)
    share_np64 = np.zeros(n_assets, dtype=np.bool_)
    pos_seq = np.empty(n_assets, dtype=np.int64)
    npos = 0
    in_book = np.zeros(n_assets, dtype=np.bool_)
    last_mark = np.zeros(n_assets, dtype=np.float64)
    ever_marked = np.zeros(n_assets, dtype=np.bool_)
    mark_age = np.zeros(n_assets, dtype=np.int64)
    npv = np.zeros(n_assets, dtype=np.float64)
    marked_today = np.zeros(n_assets, dtype=np.bool_)

    cash = float(initial_nav)
    cash_np64 = False
    cost_comm = 0.0
    cost_spr = 0.0
    cost_imp = 0.0
    reject_count = 0
    cash_reject_count = 0
    halt_count = 0
    order_seq = 0
    n_navs = 0
    n_fills = 0

    last_i = n_dates - 1 if use_next_open else n_dates
    for i in range(last_i):
        exec_t = i + 1 if use_next_open else i

        # Sizing NAV must not see this bar's close. Snapshot the marks
        # knowable before the fill, then update last_mark for the close.
        pre_mark = last_mark.copy()
        pre_ever = ever_marked.copy()

        # --- mark update ------------------------------------------------
        for a in range(n_assets):
            c_ok = np.isfinite(ctr[exec_t, a]) and ctr[exec_t, a] > 0
            cl_ok = np.isfinite(close_px[exec_t, a]) and close_px[exec_t, a] > 0
            if c_ok:
                last_mark[a] = ctr[exec_t, a]
                mark_age[a] = 0
                ever_marked[a] = True
                marked_today[a] = True
            elif cl_ok:
                last_mark[a] = close_px[exec_t, a]
                mark_age[a] = 0
                ever_marked[a] = True
                marked_today[a] = True
            else:
                mark_age[a] += 1
                marked_today[a] = False

        # --- stale-valuation fail-closed on held positions --------------
        # Two passes, matching the reference's dict comprehension order:
        # never-marked held names first (detail "unknown"), then over-limit.
        nstale = 0
        for j in range(npos):
            a = pos_seq[j]
            if abs(shares[a]) > 1e-12 and not ever_marked[a]:
                stale_assets[nstale] = a
                stale_ages[nstale] = mark_age[a]
                stale_ever[nstale] = False
                nstale += 1
        for j in range(npos):
            a = pos_seq[j]
            if abs(shares[a]) > 1e-12 and ever_marked[a] and mark_age[a] > stale_price_bars:
                stale_assets[nstale] = a
                stale_ages[nstale] = mark_age[a]
                stale_ever[nstale] = True
                nstale += 1
        if nstale:
            return (
                1,
                n_navs,
                n_fills,
                reject_count,
                cash_reject_count,
                halt_count,
                cost_comm,
                cost_spr,
                cost_imp,
                order_seq,
                nstale,
            )

        # --- nav at execution marks --------------------------------------
        # Names without an execution print stay on the pre-bar mark.
        # last_mark already holds this bar's close; using it here leaks
        # that close into order sizing (AUDIT_P61 finding 7).
        for a in range(n_assets):
            e_ok = np.isfinite(exec_px[exec_t, a]) and exec_px[exec_t, a] > 0
            if e_ok:
                npv[a] = exec_px[exec_t, a]
            elif pre_ever[a]:
                npv[a] = pre_mark[a]
            else:
                npv[a] = 0.0
        any_share_np64 = False
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = shares[a] * npv[a]
            term_f[j] = share_np64[a]
            if share_np64[a]:
                any_share_np64 = True
        pos_val = _csum(term_v[:npos], term_f[:npos])
        nav = cash + pos_val
        if nav <= 0:
            break
        nav_safe = max(nav, 1e-12)
        nav_np64 = cash_np64 or any_share_np64

        traded_turn = 0.0
        has = has_mv[i]
        mv = market_vols[i]
        mv_bad = has and (not np.isfinite(mv) or mv < 0.0)

        for a in range(n_assets):
            if not (np.isfinite(exec_px[exec_t, a]) and exec_px[exec_t, a] > 0):
                continue
            price = exec_px[exec_t, a]
            tw = w_mat[i, a] if (not ref_carries or marked_today[a]) else 0.0
            desired = tw * nav / price
            current = shares[a]
            delta = desired - current
            if abs(delta) * price < 1.0:
                continue
            adv_a = adv[i, a]
            if (
                not np.isfinite(adv_a)
                or adv_a <= 0
                or not np.isfinite(participation_limit)
                or not 0 < participation_limit <= 1
            ):
                reject_count += 1
                continue
            adv_eff = adv_a
            vol_a = vol[i, a]
            vol_eff = vol_a if np.isfinite(vol_a) and vol_a > 0 else 0.02
            delta_np64 = nav_np64 or share_np64[a]

            # ``total_cost`` validates before the frictionless early return —
            # a non-finite delta fails closed rather than trading.
            if not np.isfinite(delta) or not np.isfinite(price) or price <= 0:
                return (
                    2,
                    n_navs,
                    n_fills,
                    reject_count,
                    cash_reject_count,
                    halt_count,
                    cost_comm,
                    cost_spr,
                    cost_imp,
                    order_seq,
                    0,
                )
            comm, spr, imp, total = _order_costs_nb(
                delta,
                price,
                adv_eff,
                vol_eff,
                frictionless,
                commission_bps,
                half_spread_bps,
                impact_y,
                bps_per_turnover,
            )
            max_qty = participation_limit * (adv_eff / price)
            if abs(delta) > max_qty:
                delta = np.sign(delta) * max_qty
                delta_np64 = True
                comm, spr, imp, total = _order_costs_nb(
                    delta,
                    price,
                    adv_eff,
                    vol_eff,
                    frictionless,
                    commission_bps,
                    half_spread_bps,
                    impact_y,
                    bps_per_turnover,
                )

            if kill_blocks:
                halt_count += 1
                continue

            # --- risk gate ----------------------------------------------
            current_w = current * npv[a] / nav_safe
            # Contributor set mirrors ``_projected_exposures``: an id is a
            # contributor iff its *projected shares* and *price* are both
            # nonzero — filtering on operands (not the product) keeps an
            # underflowed-to-zero np.float64 term, which still flips sum()
            # to the naive path exactly as the reference does.
            k = 0
            for b in range(n_assets):
                pb = (current + delta) if b == a else shares[b]
                if pb != 0.0 and npv[b] != 0.0:
                    term_v[k] = pb * npv[b]
                    term_f[k] = (share_np64[a] or delta_np64) if b == a else share_np64[b]
                    k += 1
            abs_v = np.empty(k, dtype=np.float64)
            for j in range(k):
                abs_v[j] = abs(term_v[j])
            gross_after = _csum(abs_v, term_f[:k]) / nav_safe
            net_after = _csum(term_v[:k], term_f[:k]) / nav_safe
            participation = abs(delta) * price / max(adv_eff, 1e-12)
            order_seq += 1
            gate_vol = mv if has else vol_eff
            rejected = (
                mv_bad
                or not np.isfinite(nav)
                or not np.isfinite(price)
                or not np.isfinite(current_w)
                or not np.isfinite(gross_after)
                or not np.isfinite(net_after)
                or not np.isfinite(participation)
                or not np.isfinite(vol_eff)
                or nav <= 0.0
                or price <= 0.0
                or gross_after < 0.0
                or participation < 0.0
                or vol_eff < 0.0
                or not np.isfinite(delta)
                or delta == 0.0
                or _exceeds_nb(abs(delta) * price, max_order_notional)
                or _exceeds_nb(abs(current_w + (delta * price) / nav_safe), max_name)
                or _exceeds_nb(gross_after, max_gross)
                or _exceeds_nb(abs(net_after), max_net)
                or _exceeds_nb(participation, max_participation)
                or _exceeds_nb(gate_vol, max_predicted_vol)
            )
            if rejected:
                reject_count += 1
                continue

            notional = delta * price
            if delta > 0 and not _funded_nb(cash, notional + total):
                cash_reject_count += 1
                continue
            cash -= notional + total
            if delta_np64:
                cash_np64 = True
            if not in_book[a]:
                in_book[a] = True
                pos_seq[npos] = a
                npos += 1
            shares[a] = current + delta
            if delta_np64:
                share_np64[a] = True
            traded_turn += abs(notional) / nav_safe
            cost_comm += comm
            cost_spr += spr
            cost_imp += imp
            if frictionless:
                bpt = 0.0
            else:
                nt = abs(delta) * price
                bpt = abs(nt) * bps_per_turnover / 1e4
            f_qty[n_fills] = delta
            f_px[n_fills] = price
            f_fee[n_fills] = comm
            f_spr[n_fills] = spr
            f_imp[n_fills] = imp
            f_bpt[n_fills] = bpt
            f_asset[n_fills] = a
            f_et[n_fills] = exec_t
            f_st[n_fills] = i
            d_ok = np.isfinite(close_px[i, a]) and close_px[i, a] > 0
            f_decok[n_fills] = d_ok
            f_dec[n_fills] = close_px[i, a] if d_ok else 0.0
            n_fills += 1

        # --- mark to close -----------------------------------------------
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = shares[a] * last_mark[a]
            term_f[j] = share_np64[a]
        pos_close = _csum(term_v[:npos], term_f[:npos])
        nav_close = cash + pos_close

        any_short_np64 = False
        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = (-shares[a] if shares[a] < 0.0 else 0.0) * last_mark[a]
            term_f[j] = share_np64[a] and shares[a] <= 0.0
            if term_f[j]:
                any_short_np64 = True
        short_notional = _csum(term_v[:npos], term_f[:npos])

        for j in range(npos):
            a = pos_seq[j]
            term_v[j] = abs(shares[a] * last_mark[a])
            term_f[j] = share_np64[a]
        gross_v = _csum(term_v[:npos], term_f[:npos])
        net_v = pos_close

        borrow = short_notional * (borrow_bps_per_year / 1e4) / 252.0
        if not frictionless:
            cash -= borrow
            if any_short_np64:
                cash_np64 = True
            nav_close -= borrow
        navs_out[n_navs, 0] = nav_close
        navs_out[n_navs, 1] = gross_v / max(nav_close, 1e-12)
        navs_out[n_navs, 2] = net_v / max(nav_close, 1e-12)
        navs_out[n_navs, 3] = traded_turn
        navs_t[n_navs] = exec_t
        n_navs += 1

    return (
        0,
        n_navs,
        n_fills,
        reject_count,
        cash_reject_count,
        halt_count,
        cost_comm,
        cost_spr,
        cost_imp,
        order_seq,
        0,
    )


def _replay_driver(
    *,
    exec_px: np.ndarray,
    close_px: np.ndarray,
    ctr: np.ndarray,
    adv: np.ndarray,
    vol: np.ndarray,
    w_mat: np.ndarray,
    market_vols: list[float | None] | None,
    use_next_open: bool,
    ref_carries: bool,
    costs_cfg: CostConfig,
    gate: RiskGateConfig,
    kill_blocks: bool,
    initial_nav: float,
    dates: list[Any],
    sids: list[str],
) -> tuple[Any, ...] | None:
    """Allocate buffers, run the compiled kernel, materialise result rows.

    Returns ``(navs, fill_rows, cost_sum, reject_count, cash_reject_count,
    halt_count, order_seq)`` where ``navs``/``fill_rows`` are row-dict lists
    matching the interpreted path — or ``None`` when numba is unavailable.
    Raises ``StaleValuationError`` / ``ValueError`` with the reference's
    exact messages on fail-closed paths.
    """
    if not HAVE_NUMBA:
        return None

    n_dates = exec_px.shape[0]
    n_assets = exec_px.shape[1]
    mv_arr = np.full(n_dates, np.nan)
    has_mv = np.zeros(n_dates, dtype=np.bool_)
    if market_vols is not None:
        for j, v in enumerate(market_vols):
            if v is not None:
                has_mv[j] = True
                mv_arr[j] = float(v)

    max_fills = max(n_dates * n_assets, 1)
    navs_out = np.empty((n_dates, 4))
    navs_t = np.empty(n_dates, dtype=np.int64)
    f_qty = np.empty(max_fills)
    f_px = np.empty(max_fills)
    f_fee = np.empty(max_fills)
    f_spr = np.empty(max_fills)
    f_imp = np.empty(max_fills)
    f_bpt = np.empty(max_fills)
    f_asset = np.empty(max_fills, dtype=np.int64)
    f_et = np.empty(max_fills, dtype=np.int64)
    f_st = np.empty(max_fills, dtype=np.int64)
    f_dec = np.empty(max_fills)
    f_decok = np.empty(max_fills, dtype=np.bool_)
    stale_assets = np.empty(n_assets, dtype=np.int64)
    stale_ages = np.empty(n_assets, dtype=np.int64)
    stale_ever = np.empty(n_assets, dtype=np.bool_)
    term_v = np.empty(n_assets)
    term_f = np.empty(n_assets, dtype=np.bool_)

    (
        status,
        n_navs,
        n_fills,
        reject_count,
        cash_reject_count,
        halt_count,
        cost_comm,
        cost_spr,
        cost_imp,
        order_seq,
        nstale,
    ) = _replay_kernel(
        np.ascontiguousarray(exec_px),
        np.ascontiguousarray(close_px),
        np.ascontiguousarray(ctr),
        np.ascontiguousarray(adv),
        np.ascontiguousarray(vol),
        np.ascontiguousarray(w_mat),
        mv_arr,
        has_mv,
        use_next_open,
        ref_carries,
        float(costs_cfg.commission_bps),
        float(costs_cfg.half_spread_bps),
        float(costs_cfg.impact_y),
        float(costs_cfg.bps_per_turnover),
        float(costs_cfg.borrow_bps_per_year),
        float(costs_cfg.participation_limit),
        bool(costs_cfg.frictionless),
        int(gate.stale_price_bars),
        float(gate.max_order_notional),
        float(gate.max_gross),
        float(gate.max_net),
        float(gate.max_name),
        float(gate.max_participation),
        float(gate.max_predicted_vol),
        bool(kill_blocks),
        float(initial_nav),
        navs_out,
        navs_t,
        f_qty,
        f_px,
        f_fee,
        f_spr,
        f_imp,
        f_bpt,
        f_asset,
        f_et,
        f_st,
        f_dec,
        f_decok,
        stale_assets,
        stale_ages,
        stale_ever,
        term_v,
        term_f,
    )

    if status == 1:
        details = ", ".join(
            f"{sids[stale_assets[k]]}={int(stale_ages[k]) if stale_ever[k] else 'unknown'}"
            for k in range(nstale)
        )
        raise StaleValuationError(
            "held position valuation is stale beyond the configured limit: " + details
        )
    if status == 2:
        raise ValueError("quantity must be finite and price must be finite and positive")

    # Row-dict construction straight from the output buffers — the same
    # Python scalars the interpreted path appends (datetimes, floats, strs,
    # None for null decision prices), so ``_build_result``'s dict transpose
    # re-infers identical columns/dtypes — including the empty-weights edge
    # where the reference produces a 0x0 fills frame. Building dicts directly
    # (no intermediate polars frame) is ~10x faster than frame -> to_dicts.
    navs: list[dict[str, Any]] = [
        {
            "event_time": dates[int(navs_t[j])],
            "nav": navs_out[j, 0],
            "gross": navs_out[j, 1],
            "net": navs_out[j, 2],
            "turnover": navs_out[j, 3],
        }
        for j in range(n_navs)
    ]
    fill_rows: list[dict[str, Any]] = [
        {
            "fill_time": dates[int(f_et[j])],
            "signal_time": dates[int(f_st[j])],
            "security_id": sids[f_asset[j]],
            "quantity": f_qty[j],
            "price": f_px[j],
            "fee": f_fee[j],
            "spread_cost": f_spr[j],
            "impact_cost": f_imp[j],
            "turnover_cost": f_bpt[j],
            "decision_price": f_dec[j] if f_decok[j] else None,
        }
        for j in range(n_fills)
    ]
    # Sequential left fold, matching the reference's ``+=`` per fill —
    # np.sum's pairwise reduction can differ at the last ulp.
    turnover_sum = 0.0
    for _v in f_bpt[:n_fills]:
        turnover_sum += float(_v)
    cost_sum = {
        "commission": cost_comm,
        "spread": cost_spr,
        "impact": cost_imp,
        "turnover": turnover_sum,
    }
    return (
        navs,
        fill_rows,
        cost_sum,
        reject_count,
        cash_reject_count,
        halt_count,
        order_seq,
    )


def _require_no_risk_overlay(risk_overlay: Any) -> None:
    """Fast replay never runs with a market-risk overlay (fail closed)."""
    if risk_overlay is not None:
        raise ValueError("fast replay does not support risk_overlay")


def run_backtest_fast(
    bars: pl.DataFrame,
    weights: pl.DataFrame,
    config: AppConfig,
    *,
    initial_nav: float = 1_000_000.0,
    risk_overlay: Any = None,
) -> BacktestResult:
    """Fast replay of ``run_backtest`` on the supported config class.

    Fails closed — ``ValueError`` — on any workload outside the class the
    vectorized path reproduces bit-identically; it never approximates.
    """
    if config.execution.allow_close_auction:
        raise ValueError("fast replay does not support allow_close_auction")
    if is_calibrated_spread_estimator(config.costs.spread_estimator):
        raise ValueError(
            "fast replay does not support OHLC spread calibration "
            f"(spread_estimator={config.costs.spread_estimator!r}); "
            "use run_backtest(..., fast=False) for the calibrated cost path"
        )
    _require_no_risk_overlay(risk_overlay)
    _validate_panel_fast(weights)
    # Shared with the event loop: duplicate bar keys raise the same
    # ValueError("duplicate bars …") rather than a fast-only refuse string.
    _validate_bar_panel(bars)
    # Panel-shape and completeness refusals mirror the dispatcher's, so a
    # direct call is guarded exactly like run_backtest(fast=True). Without
    # them the matrices would silently collapse duplicate bar keys, align
    # mismatched datetime units to empty weight cells, or IndexError on an
    # empty panel instead of producing the reference's answers/errors.
    if not _fast_replay_panel_supported(bars, weights):
        raise ValueError(
            "fast replay requires a non-empty bars panel with unique "
            "(event_time, security_id) keys and a Datetime event_time "
            "matching the weights panel"
        )
    if not _fast_replay_is_complete(config, risk_overlay):
        raise ValueError(
            "fast replay does not run while a market-risk-overlay artifact "
            "exists: garch_risk_overlay_dates counters are stamped by the "
            "event loop"
        )

    # Lineage detection: the newer engine signature carries ``risk_overlay``
    # and treats weight panels as a sparse rebalance grid — the last target
    # holds until the next decision date, and names without a fresh mark are
    # flattened. The older lineage zero-fills absent dates. Detect once and
    # replicate whichever semantics the deployed ``run_backtest`` implements.
    ref_carries = "risk_overlay" in inspect.signature(run_backtest).parameters

    dates, dates_ns, sids, open_px, close_px, ctr, adv, vol, synthetic = _bars_to_matrices(bars)
    n_dates = len(dates)
    n_assets = len(sids)
    sid_idx = {s: i for i, s in enumerate(sids)}

    # Target-weight matrix on bar dates only — the reference loop only ever
    # looks up weights on dates that have bars. Fast path: one numpy scatter
    # (searchsorted + fancy assign) replaces the polars pivot — duplicate
    # (event_time, security_id) keys were already rejected by validation, so
    # every weight row maps to at most one cell unambiguously. Non-string or
    # null-keyed panels fall back to the pivot path, whose semantics were
    # already conformance-verified.
    w_mat = np.zeros((n_dates, n_assets))
    w_present = np.zeros(n_dates, dtype=bool)
    wet = weights["event_time"]
    wsid = weights["security_id"]
    if (
        weights.height
        and wsid.dtype == pl.String
        and wet.null_count() == 0
        and wsid.null_count() == 0
    ):
        w_ns = wet.cast(pl.Int64).to_numpy()
        ti = np.searchsorted(dates_ns, w_ns)
        in_range = (ti < n_dates) & (dates_ns[np.minimum(ti, n_dates - 1)] == w_ns)
        # A date with ANY weight row replaces the carried target — even rows
        # whose sid has no bars (they flatten known names by absence).
        w_present[ti[in_range]] = True
        sids_arr = np.asarray(sids)
        w_sid_arr = wsid.to_numpy()
        ai = np.searchsorted(sids_arr, w_sid_arr)
        known = (ai < n_assets) & (sids_arr[np.minimum(ai, n_assets - 1)] == w_sid_arr)
        wv = weights["target_weight"].cast(pl.Float64, strict=False).to_numpy()
        wv = np.where(np.isfinite(wv), wv, 0.0)
        good = in_range & known
        w_mat[ti[good], ai[good]] = wv[good]
    else:
        piv = weights.pivot(
            on="security_id",
            index="event_time",
            values="target_weight",
            aggregate_function="first",
        )
        if piv.height:
            piv_ns = piv["event_time"].cast(pl.Int64).to_numpy()
            piv_ti = np.searchsorted(dates_ns, piv_ns)
            in_range = (piv_ti < n_dates) & (dates_ns[np.minimum(piv_ti, n_dates - 1)] == piv_ns)
            for col in piv.columns:
                if col == "event_time":
                    continue
                ai_c = sid_idx.get(col)
                if ai_c is None:
                    # Weight on a symbol with no bars — validated upstream but
                    # can never trade; identical to the reference loop's skip.
                    continue
                v = piv[col].cast(pl.Float64, strict=False).to_numpy()
                w_mat[piv_ti[in_range], ai_c] = np.where(np.isfinite(v), v, 0.0)[in_range]
            w_present[piv_ti[in_range]] = True

    market_vols: list[float | None] | None = None
    if ref_carries:
        # ``if dt in weights_by_date: last_target_w = ...`` — presence is at
        # the whole-row level: a date with ANY row replaces the carried target
        # (absent sids target 0); a date with none inherits the prior row.
        for i in range(n_dates):
            if not w_present[i]:
                w_mat[i] = w_mat[i - 1] if i else 0.0
        # Market overlay: artifact existence is config/fs-level — probe the
        # last date once; (None, None) means it can never engage.
        if _mro_asof is not None:
            probe_v, _ = _mro_asof(config, bars, dates[-1])
            if probe_v is not None:
                market_vols = [_mro_asof(config, bars, d)[0] for d in dates]

    use_next_open = (
        config.execution.fill is FillConvention.NEXT_OPEN
        and not config.execution.allow_close_auction
    )

    # Per-(date, asset) rows converted to Python lists once — indexing a list
    # is faster than a per-day ``.tolist()`` on a numpy row. Values are
    # identical (``tolist`` preserves float64 bit patterns exactly).
    src_mat = open_px if use_next_open else close_px

    costs_cfg = config.costs
    gate = config.risk_gate
    kill = KillSwitch(config.kill_switch)

    # KillSwitch state is fixed at construction — evaluate once; each blocked
    # order is still counted identically in the loop below.
    try:
        kill.assert_new_orders_allowed()
        kill_blocks = False
    except KillSwitchActive:
        kill_blocks = True

    kernel_out = None
    if HAVE_NUMBA:
        from numba.core.errors import NumbaError  # HAVE_NUMBA guarantees the import

        try:
            kernel_out = _replay_driver(
                exec_px=src_mat,
                close_px=close_px,
                ctr=ctr,
                adv=adv,
                vol=vol,
                w_mat=w_mat,
                market_vols=market_vols,
                use_next_open=use_next_open,
                ref_carries=ref_carries,
                costs_cfg=costs_cfg,
                gate=gate,
                kill_blocks=kill_blocks,
                initial_nav=float(initial_nav),
                dates=dates,
                sids=sids,
            )
        except (StaleValuationError, ValueError):
            # Fail-closed paths reproduced by the kernel — same exceptions
            # the reference raises; propagate, do not fall back.
            raise
        except (NumbaError, TypeError, RuntimeError):
            # Narrowed from `except Exception` (quality ratchet). Only numba
            # machinery failures (compile/type errors on an exotic environment)
            # land here — degrade to the interpreted loop, which is bit-identical,
            # rather than failing the run. Exotic errors propagate.
            kernel_out = None
        except AttributeError as exc:
            # Numba 0.68 types np.empty's overload as a plain function
            # (ol_np_empty.impl has no get_call_template). Same degrade.
            # Any other AttributeError still propagates.
            if not compiler_defect(exc):
                raise
            kernel_out = None
    if kernel_out is not None:
        (
            navs,
            fill_rows,
            cost_sum,
            reject_count,
            cash_reject_count,
            halt_count,
            order_seq,
        ) = kernel_out
    else:
        # Interpreted fallback (numba absent). Per-(date, asset) rows are
        # converted to Python lists once — indexing a list is faster than a
        # per-day ``.tolist()`` on a numpy row.
        exec_valid_m = np.isfinite(src_mat) & (src_mat > 0)
        src_l = src_mat.tolist()
        adv_l = adv.tolist()
        vol_l = vol.tolist()
        dec_px_l = close_px.tolist()
        dec_ok_l = (np.isfinite(close_px) & (close_px > 0)).tolist()
        # Whole-matrix mark validity — the per-day mark update only carries
        # last_mark/mark_age state; today's valid marks are row lookups.
        ctr_ok_m = np.isfinite(ctr) & (ctr > 0)
        close_ok_m = np.isfinite(close_px) & (close_px > 0)
        new_mark_m = np.where(ctr_ok_m, ctr, np.where(close_ok_m, close_px, 0.0))

        cash = float(initial_nav)
        # Python list (not ndarray): element types must propagate exactly like
        # the reference's shares dict — float until a participation cap stores
        # np.float64.
        shares: list[float] = [0.0] * n_assets
        pos_seq: list[int] = []  # shares-dict insertion order for NAV sums
        in_book = np.zeros(n_assets, dtype=bool)
        last_mark = np.zeros(n_assets)  # carried close marks (0 = never marked)
        ever_marked = np.zeros(n_assets, dtype=bool)
        mark_age = np.zeros(n_assets, dtype=np.int64)

        navs = []
        fill_rows = []
        cost_sum = {"commission": 0.0, "spread": 0.0, "impact": 0.0, "turnover": 0.0}
        reject_count = 0
        cash_reject_count = 0
        halt_count = 0
        order_seq = 0

        loop_range = range(n_dates - 1) if use_next_open else range(n_dates)
        for i in loop_range:
            exec_t = i + 1 if use_next_open else i
            exec_src = src_mat[exec_t]
            exec_valid = exec_valid_m[exec_t]

            # --- mark update (close_total_return preferred, close fallback) ----
            marked_today = ctr_ok_m[exec_t] | close_ok_m[exec_t]
            # Pre-bar marks for sizing. ``np.where`` / ``|`` allocate, so
            # these names keep the arrays from before today's close update.
            pre_mark = last_mark
            pre_ever = ever_marked
            mark_age = np.where(marked_today, 0, mark_age + 1)
            last_mark = np.where(marked_today, new_mark_m[exec_t], last_mark)
            ever_marked = ever_marked | marked_today

            # --- stale-valuation fail-closed on held positions -----------------
            # Detail order follows the reference's two dict passes over
            # book.shares insertion order: never-marked names first (detail
            # "unknown"), then over-limit names.
            stale = [a for a in pos_seq if abs(shares[a]) > 1e-12 and not ever_marked[a]] + [
                a
                for a in pos_seq
                if abs(shares[a]) > 1e-12 and ever_marked[a] and mark_age[a] > gate.stale_price_bars
            ]
            if stale:
                details = ", ".join(
                    f"{sids[a]}={int(mark_age[a]) if ever_marked[a] else 'unknown'}" for a in stale
                )
                raise StaleValuationError(
                    "held position valuation is stale beyond the configured limit: " + details
                )

            # --- decision marks (signal bar close at decision date) ------------
            dec_l_i = dec_px_l[i]
            dec_ok_i = dec_ok_l[i]

            # --- nav at execution marks (exec price overrides carried close) ---
            # book.nav uses builtin sum() — CPython 3.12+ applies Neumaier
            # compensated summation, which differs from a naive += loop at the
            # last ulp. Keep builtin sum over the same insertion-ordered terms
            # for bit-identical floats. NB the compensated path only applies to
            # exact Python floats — np.float64 terms must be coerced per-product.
            nav_price = np.where(exec_valid, exec_src, np.where(pre_ever, pre_mark, 0.0))
            # Prices coerce to Python float; shares keep their dict type so a
            # capped fill's np.float64 propagates into products exactly as the
            # reference's does (np.float64 term → sum() degrades to naive).
            npv = nav_price.tolist()
            pos_val = sum(shares[a] * npv[a] for a in pos_seq)
            nav = cash + pos_val
            if nav <= 0:
                break

            esrc = src_l[exec_t]
            adv_row = adv_l[i]
            vol_row = vol_l[i]
            dec_l = dec_l_i
            dec_ok_d = dec_ok_i
            # ``ids = exec | shares | target``; unpriced sids are skipped inside.
            # Per-day contributor products shares*price; position a's term is
            # patched per order. A zero product is dropped — bitwise-identical to
            # the reference's (proj!=0 and price!=0) contributor filter since a
            # zero term never changes a compensated sum.
            terms_base = [shares[b] * npv[b] for b in range(n_assets)]
            nav_safe = max(nav, 1e-12)
            traded_turn = 0.0
            # Date-level market overlay vol (newer lineage): replaces name-level
            # vol for the max_predicted_vol gate; absent -> name vol governs.
            mv = market_vols[i] if market_vols is not None else None
            mv_bad = mv is not None and (not math.isfinite(mv) or mv < 0.0)
            # Carrying lineage flattens names without a fresh mark — a missing
            # mark is an exit, not a ghost hold.
            tw_arr = np.where(marked_today, w_mat[i], 0.0) if ref_carries else w_mat[i]
            tw_l = tw_arr.tolist()
            # Candidate mask: assets with an executable mark whose dust check
            # passes. Vectorized IEEE-754 float64 math — the scalar loop below
            # recomputes the identical delta in Python for the order path;
            # flatnonzero preserves ascending sid order (reference iterates
            # sorted ids). ``shares`` keeps mixed types; sh_arr is values-only.
            px_ok = np.where(exec_valid, exec_src, 1.0)
            desired_arr = tw_arr * nav / px_ok
            sh_arr = np.asarray(shares)
            cand = exec_valid & (np.abs(desired_arr - sh_arr) * px_ok >= 1.0)
            for a in np.flatnonzero(cand).tolist():
                price = esrc[a]
                tw = tw_l[a]
                desired = tw * nav / price
                current = shares[a]
                delta = desired - current
                adv_a = adv_row[a]
                if (
                    not math.isfinite(adv_a)
                    or adv_a <= 0
                    or not math.isfinite(costs_cfg.participation_limit)
                    or not 0 < costs_cfg.participation_limit <= 1
                ):
                    reject_count += 1
                    continue
                adv_eff = adv_a
                vol_a = vol_row[a]
                vol_eff = vol_a if math.isfinite(vol_a) and vol_a > 0 else 0.02

                comm, spr, imp, total_trade_cost = _order_costs(
                    delta, price, adv_eff, vol_eff, costs_cfg
                )
                max_qty = costs_cfg.participation_limit * (adv_eff / price)
                if abs(delta) > max_qty:
                    # np.sign() yields np.float64; the reference does NOT coerce it
                    # back — the np.float64 delta contaminates shares/cash and
                    # degrades later sum() calls to the naive path. Replicate.
                    delta = np.sign(delta) * max_qty
                    comm, spr, imp, total_trade_cost = _order_costs(
                        delta, price, adv_eff, vol_eff, costs_cfg
                    )

                if kill_blocks:
                    halt_count += 1
                    continue

                # --- risk gate (same checks, same order outcomes) --------------
                current_w = current * npv[a] / nav_safe
                # _projected_exposures sums |v| and v over sorted contributors
                # with builtin sum(); price operands are float, share operands
                # keep dict types — identical propagation to the reference.
                terms = terms_base[:]
                terms[a] = (current + delta) * npv[a]
                terms = [t for t in terms if t != 0.0]
                gross_after = sum(map(abs, terms)) / nav_safe
                net_after = sum(terms) / nav_safe
                participation = abs(delta) * price / max(adv_eff, 1e-12)
                order_seq += 1
                # check_order's signed_qty = +/-|delta| by side — i.e. delta itself.
                # gate_vol: a present market overlay vol replaces name vol for the
                # max_predicted_vol comparison (resolve_gate_predicted_vol).
                gate_vol = mv if mv is not None else vol_eff
                rejected = (
                    mv_bad
                    or any(
                        not math.isfinite(v)
                        for v in (
                            nav,
                            price,
                            current_w,
                            gross_after,
                            net_after,
                            participation,
                            vol_eff,
                        )
                    )
                    or nav <= 0.0
                    or price <= 0.0
                    or gross_after < 0.0
                    or participation < 0.0
                    or vol_eff < 0.0
                    or not math.isfinite(delta)
                    or delta == 0.0
                    or exceeds_limit(abs(delta) * price, gate.max_order_notional)
                    or exceeds_limit(abs(current_w + (delta * price) / nav_safe), gate.max_name)
                    or exceeds_limit(gross_after, gate.max_gross)
                    or exceeds_limit(abs(net_after), gate.max_net)
                    or exceeds_limit(participation, gate.max_participation)
                    or exceeds_limit(gate_vol, gate.max_predicted_vol)
                )
                if rejected:
                    reject_count += 1
                    continue

                notional = delta * price
                if delta > 0 and not funded(cash, notional + total_trade_cost):
                    cash_reject_count += 1
                    continue
                cash -= notional + total_trade_cost
                if not in_book[a]:
                    in_book[a] = True
                    pos_seq.append(a)
                shares[a] = current + delta
                terms_base[a] = shares[a] * npv[a]
                traded_turn += abs(notional) / nav_safe
                if costs_cfg.frictionless:
                    bpt = 0.0
                else:
                    nt = abs(delta) * price
                    bpt = abs(nt) * costs_cfg.bps_per_turnover / 1e4
                cost_sum["commission"] += float(comm)
                cost_sum["spread"] += float(spr)
                cost_sum["impact"] += float(imp)
                cost_sum["turnover"] += bpt
                fill_rows.append(
                    {
                        "fill_time": dates[exec_t],
                        "signal_time": dates[i],
                        "security_id": sids[a],
                        "quantity": delta,
                        "price": price,
                        "fee": comm,
                        "spread_cost": spr,
                        "impact_cost": imp,
                        "turnover_cost": bpt,
                        "decision_price": dec_l[a] if dec_ok_d[a] else None,
                    }
                )

            # --- mark to close ---------------------------------------------------
            # Builtin sum() over shares-order terms; marks coerce to float, shares
            # keep dict types — same propagation as the reference.
            lm = last_mark.tolist()
            pos_close = sum(shares[a] * lm[a] for a in pos_seq)
            nav_close = cash + pos_close
            short_notional = sum(abs(min(shares[a], 0.0)) * lm[a] for a in pos_seq)
            gross_v = sum(abs(shares[a] * lm[a]) for a in pos_seq)
            # net exposure is the identical summation as pos_close — reuse.
            net_v = pos_close
            borrow = short_notional * (costs_cfg.borrow_bps_per_year / 1e4) / 252.0
            if not costs_cfg.frictionless:
                cash -= borrow
                nav_close -= borrow
            navs.append(
                {
                    "event_time": dates[exec_t],
                    "nav": nav_close,
                    "gross": gross_v / max(nav_close, 1e-12),
                    "net": net_v / max(nav_close, 1e-12),
                    "turnover": traded_turn,
                }
            )

    # ---- identical metrics tail as the reference engine --------------------
    from quant_fund.backtest.engine import _build_result

    return _build_result(
        navs=navs,
        fill_rows=fill_rows,
        cost_sum=cost_sum,
        reject_count=reject_count,
        cash_reject_count=cash_reject_count,
        halt_count=halt_count,
        synthetic=synthetic,
        config=config,
        initial_nav=initial_nav,
    )


__all__ = ["run_backtest_fast"]
