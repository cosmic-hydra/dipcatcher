"""mid_dark — does hidden midpoint liquidity close the wave-23 residual?

``closure_stack.v1`` proved the two winning knobs interfere: the chase
must persist AT the touch to lift the continuation channel, but resting
visible depth at the touch presses it and destroys instant impact. The
``wave23_map.v1`` synthesis named the missing primitive: a placement
class that rests at the inside WITHOUT contributing visible depth — the
iceberg/dark channel ``hidden_depth.v1`` measured at ~21% of fills on
the tape.

This module adds that class to the sim (``mid_dark_frac`` — midpoint
pegs that fill marketable flow at mid and never appear in the visible
book) and scans (mid_dark_frac × refill_cooldown × unhit_imp) on the
calibrated wave-23 base. Each cell is scored on the five channel
targets (instant 0.887, k200 4.6447, lo +2.81, fill +2.52, cxl −0.68).
The capstone claim is ``joint_closure_exists``: some cell lands every
channel inside tolerance at once — the dark channel either lets the
stack compose where visible-only couplings failed, or it doesn't.
"""

from __future__ import annotations

from typing import Any

from quant_fund.microstructure.continuation_attr_bench import (
    _attr_totals,
    _AttrSim,
)
from quant_fund.microstructure.place_law_bench import _calibrated, _split
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes
from quant_fund.utils.reproducibility import git_revision

MID_DARK_SCHEMA = "mid_dark.v1"

# Tape reference values (sealed in continuation_attr.v1 / closure_fit.v1).
_LO_CH_TAPE = 2.81
_FILL_CH_TAPE = 2.52
_CXL_CH_TAPE = -0.68
_K200_TAPE = 4.6447
_INSTANT_TAPE = 0.887
_K200_LAG = 200

# (refill_cooldown, mid_dark_frac, mid_dark_ttl, unhit_imp, imp_window).
# Row 1 is the calibrated base; 2–3 dark alone; 4 chase alone; 5–8 the
# dark×chase product; 9–10 add the instant channel on top.
_GRID: tuple[tuple[int, float, int, float, int], ...] = (
    (0, 0.0, 0, 0.0, 0),
    (0, 0.4, 0, 0.0, 0),
    (0, 0.4, 200, 0.0, 0),
    (0, 0.0, 0, 0.5, 200),
    (0, 0.2, 0, 0.5, 200),
    (0, 0.4, 0, 0.5, 200),
    (0, 0.4, 200, 0.5, 200),
    (0, 0.6, 200, 0.5, 200),
    (400, 0.4, 200, 0.5, 200),
    (400, 0.6, 200, 0.5, 200),
)


def _dark_cell(
    cooldown: int,
    dark_frac: float,
    dark_ttl: int,
    imp_frac: float,
    imp_window: int,
    *,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    """One cell: channel attribution + instant/k200 kernel per fill."""
    cfg = _calibrated(
        seed,
        {
            "refill_cooldown": cooldown,
            "mid_dark_frac": dark_frac,
            "mid_dark_ttl": dark_ttl,
            "unhit_imp_frac": imp_frac,
            "unhit_imp_window": imp_window,
        },
    )
    sim = _AttrSim(cfg, _split(3.0, seed + 1))

    # mid[e] = state after event e (1-indexed, matching n_events/mut_log).
    mid: list[float | None] = [None]
    fills: list[tuple[int, float]] = []
    dark_fills = 0
    seen = 0
    for _ in range(horizon):
        sim.step()
        bb, ba = sim.best_bid_level, sim.best_ask_level
        mid.append(0.5 * (bb + ba) if (bb is not None and ba is not None) else None)
        while seen < len(sim.trades):
            tr = sim.trades[seen]
            fills.append((sim.n_events, 1.0 if tr.aggressor == "buy" else -1.0))
            if tr.maker_tag == "mid_dark":
                dark_fills += 1
            seen += 1
    n_ev = len(mid) - 1

    mut_by_ev: dict[int, tuple[str, str]] = {}
    for ev_idx, ch, side in sim.mut_log:
        if ev_idx not in mut_by_ev:
            mut_by_ev[ev_idx] = (ch, side)

    per_event: list[tuple[int, float, int, str, str, float]] = []
    i_sum = 0.0
    i_n = 0
    k_sum = 0.0
    k_n = 0
    for j, sign in fills:
        m0 = mid[j - 1] if j >= 1 else None
        if m0 is None:
            continue
        m1 = mid[j] if j < len(mid) else None
        if m1 is not None:
            i_sum += sign * (m1 - m0)
            i_n += 1
        j200 = j + _K200_LAG
        mj = mid[j200] if j200 < len(mid) else None
        if mj is not None:
            k_sum += sign * (mj - m0)
            k_n += 1
        hit_side = "sell" if sign > 0 else "buy"
        for m_ev in range(j + 1, min(j + 1 + _K200_LAG, n_ev + 1)):
            d1, d0 = mid[m_ev], mid[m_ev - 1]
            if d1 is None or d0 is None or d1 == d0:
                continue
            mut = mut_by_ev.get(m_ev)
            if mut is None:
                continue
            ch, side = mut
            rel = "hit" if side == hit_side else "unhit"
            per_event.append((j, sign, m_ev, ch, rel, d1 - d0))

    attr = _attr_totals(per_event)
    per_ch = attr["k200_per_channel_ticks"]
    return {
        "refill_cooldown": cooldown,
        "mid_dark_frac": dark_frac,
        "mid_dark_ttl": dark_ttl,
        "unhit_imp_frac": imp_frac,
        "unhit_imp_window": imp_window,
        "n_fills": len(fills),
        "n_dark_fills": dark_fills,
        "dark_fill_share": None if not fills else round(dark_fills / len(fills), 4),
        "instant_signed_ticks": None if i_n == 0 else round(i_sum / i_n, 4),
        "k200_ticks": None if k_n == 0 else round(k_sum / k_n, 4),
        "k200_per_channel_ticks": {ch: round(per_ch[ch], 4) for ch in per_ch},
        "lo_channel_ticks": round(per_ch["lo"], 4),
        "fill_channel_ticks": round(per_ch["fill"], 4),
        "cxl_channel_ticks": round(per_ch["cxl"], 4),
        "attr_windows": attr["windows"],
    }


def _in_tol(v: float | None, target: float, tol: float) -> bool:
    return v is not None and abs(v - target) <= tol


def _tol_flags(c: dict[str, Any]) -> None:
    c["instant_in_tol"] = _in_tol(c["instant_signed_ticks"], _INSTANT_TAPE, 0.2)
    c["k200_in_tol"] = _in_tol(c["k200_ticks"], _K200_TAPE, 1.0)
    c["lo_in_tol"] = _in_tol(c["lo_channel_ticks"], _LO_CH_TAPE, 1.0)
    c["fill_in_tol"] = _in_tol(c["fill_channel_ticks"], _FILL_CH_TAPE, 1.0)
    c["cxl_in_tol"] = _in_tol(c["cxl_channel_ticks"], _CXL_CH_TAPE, 0.5)
    c["all_in_tol"] = all(
        c[k] for k in ("instant_in_tol", "k200_in_tol", "lo_in_tol", "fill_in_tol", "cxl_in_tol")
    )


def mid_dark_bench(*, horizon: int = 20000, seed: int = 7) -> dict[str, Any]:
    """Scan (cooldown × dark × chase); verdict = joint closure of all targets."""
    cells = [
        _dark_cell(cd, df, dt, f, iw, horizon=horizon, seed=seed + i)
        for i, (cd, df, dt, f, iw) in enumerate(_GRID)
    ]
    for c in cells:
        _tol_flags(c)

    closers = [c for c in cells if c["all_in_tol"]]
    dark_on = [c for c in cells if c["mid_dark_frac"] > 0.0]
    # Paired comparison: cells sharing (cooldown, imp) with dark on vs off.
    off_by_key = {
        (c["refill_cooldown"], c["unhit_imp_frac"]): c for c in cells if c["mid_dark_frac"] == 0.0
    }
    inst_diluted = 0
    paired = 0
    lo_preserved = 0
    for c in dark_on:
        off = off_by_key.get((c["refill_cooldown"], c["unhit_imp_frac"]))
        if off is None or c["instant_signed_ticks"] is None or off["instant_signed_ticks"] is None:
            continue
        paired += 1
        if c["instant_signed_ticks"] < off["instant_signed_ticks"]:
            inst_diluted += 1
        if c["lo_in_tol"] or (c["lo_channel_ticks"] or 0.0) > (off["lo_channel_ticks"] or 0.0):
            lo_preserved += 1
    claims = {
        "grid_evaluated": len(cells) == len(_GRID),
        # The dark channel is live: at least one cell booked dark fills.
        "dark_channel_active": any((c["n_dark_fills"] or 0) > 0 for c in cells),
        # Falsification/verification of the dilution worry: dark fills
        # contribute sign·Δmid = 0 at fill time, pulling the per-fill
        # instant mean down — measured per paired cell.
        "dark_dilutes_instant": bool(paired) and inst_diluted == paired,
        # Dark preserves the chase's lo lift on at least one paired cell.
        "dark_preserves_lo": lo_preserved > 0,
        # The capstone: some cell lands all five channels in tolerance.
        "joint_closure_exists": bool(closers),
    }
    best = min(
        cells,
        key=lambda c: sum(
            1
            for k in (
                "instant_in_tol",
                "k200_in_tol",
                "lo_in_tol",
                "fill_in_tol",
                "cxl_in_tol",
            )
            if not c[k]
        ),
    )
    payload: dict[str, Any] = {
        "schema": MID_DARK_SCHEMA,
        "kind": "sim_bench",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "git_revision": git_revision(),
        "seed": seed,
        "horizon_events": horizon,
        "target_instant_ticks": _INSTANT_TAPE,
        "target_k200_ticks": _K200_TAPE,
        "tolerances": {
            "instant": 0.2,
            "k200": 1.0,
            "lo": 1.0,
            "fill": 1.0,
            "cxl": 0.5,
        },
        "grid": cells,
        "claims": claims,
        "best_cell": {
            "refill_cooldown": best["refill_cooldown"],
            "mid_dark_frac": best["mid_dark_frac"],
            "mid_dark_ttl": best["mid_dark_ttl"],
            "unhit_imp_frac": best["unhit_imp_frac"],
            "channels_in_tol": sum(
                1
                for k in (
                    "instant_in_tol",
                    "k200_in_tol",
                    "lo_in_tol",
                    "fill_in_tol",
                    "cxl_in_tol",
                )
                if best[k]
            ),
        },
    }
    payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(payload))
    return payload


def main() -> None:
    """Run the bench and write a sealed receipt under ``receipts/``."""
    import json
    from pathlib import Path

    out = mid_dark_bench()
    path = Path("receipts") / "mid_dark_amzn.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
