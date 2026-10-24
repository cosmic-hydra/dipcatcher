"""Compact seed layout for the generated capability declaration ledger.

``scripts/generated_capability_declarations/`` holds one expanded shard per
registered extension owner; each ``_register(seed_id)`` line names a discovery
record. This module keeps the same layout in compact form: one arithmetic
progression per owner. ``owner_references`` is the owner lookup the per-module
wrappers in :mod:`fx1.extensions` call, and ``resolve_seed_id`` reverses an id
back to the owner that registered it.

The records are discovery identifiers only. They are not independently
verified capabilities, never market evidence, and live P&L claims must never
be derived from them.
"""

from __future__ import annotations

from typing import Any, Literal

SeedKind = Literal["feature", "skill", "plugin"]

# (owner, first seed, stride, count), generated from the declaration ledger.
_FEATURES: tuple[tuple[str, int, int, int], ...] = (
    ("adv", 107, 135, 7407),
    ("adv_ratio_20_60", 122, 135, 7407),
    ("amihud", 101, 135, 7407),
    ("amihud_60", 104, 135, 7407),
    ("beta_60", 92, 135, 7407),
    ("cs_z_planted_signal", 134, 135, 7407),
    ("cs_z_ret_1", 128, 135, 7407),
    ("dollar_volume", 110, 135, 7407),
    ("downside_vol_20", 86, 135, 7407),
    ("high_52w_prox", 47, 135, 7408),
    ("idio_mom_20", 98, 135, 7407),
    ("idio_vol_60", 95, 135, 7407),
    ("kurt_20", 65, 135, 7407),
    ("log_price", 125, 135, 7407),
    ("log_ret_1", 11, 135, 7408),
    ("max_ret_20", 56, 135, 7407),
    ("min_ret_20", 59, 135, 7407),
    ("mom_126", 38, 135, 7408),
    ("mom_12_1", 44, 135, 7408),
    ("mom_20", 29, 135, 7408),
    ("mom_252", 41, 135, 7408),
    ("mom_5", 26, 135, 7408),
    ("mom_60", 35, 135, 7408),
    ("mom_skip_5_20", 32, 135, 7408),
    ("planted_signal", 131, 135, 7407),
    ("rel_volume", 113, 135, 7407),
    ("ret_1", 2, 135, 7408),
    ("ret_20", 8, 135, 7408),
    ("ret_5", 5, 135, 7408),
    ("ret_intraday_20", 23, 135, 7408),
    ("ret_open_close", 17, 135, 7408),
    ("ret_overnight", 14, 135, 7408),
    ("ret_overnight_20", 20, 135, 7408),
    ("reversal_1", 50, 135, 7408),
    ("skew_20", 62, 135, 7407),
    ("turnover_proxy", 116, 135, 7407),
    ("vol_20", 68, 135, 7407),
    ("vol_60", 71, 135, 7407),
    ("vol_ewma", 74, 135, 7407),
    ("vol_garman_klass", 80, 135, 7407),
    ("vol_of_vol", 83, 135, 7407),
    ("vol_parkinson", 77, 135, 7407),
    ("vol_ratio_20_60", 89, 135, 7407),
    ("volume_vol", 119, 135, 7407),
    ("z_vs_ma20", 53, 135, 7408),
)

_SKILLS: tuple[tuple[str, int, int, int], ...] = (
    ("backtest", 30, 69, 14493),
    ("book-panel", 51, 69, 14493),
    ("build-features", 18, 69, 14493),
    ("build-labels", 21, 69, 14493),
    ("candle-book", 39, 69, 14493),
    ("collect", 15, 69, 14493),
    ("doctor", 0, 69, 14493),
    ("forecast", 33, 69, 14493),
    ("ingest", 12, 69, 14493),
    ("kronos-forecast", 36, 69, 14493),
    ("kyle-ofi", 42, 69, 14493),
    ("monitor", 9, 69, 14493),
    ("northset", 27, 69, 14493),
    ("optimize", 63, 69, 14492),
    ("paper", 66, 69, 14492),
    ("report", 54, 69, 14492),
    ("research", 24, 69, 14493),
    ("session-book", 45, 69, 14493),
    ("tearsheet", 57, 69, 14492),
    ("train", 60, 69, 14492),
    ("validate", 6, 69, 14493),
    ("vendor-book-map", 48, 69, 14493),
    ("verify-research", 3, 69, 14493),
)

_PLUGINS: tuple[tuple[str, int, int, int], ...] = (
    ("binance_crypto", 1, 54, 18519),
    ("caixin", 4, 54, 18519),
    ("cls", 7, 54, 18519),
    ("dongcai", 10, 54, 18519),
    ("finance_fetch", 13, 54, 18519),
    ("finance_research", 16, 54, 18519),
    ("finenter", 19, 54, 18519),
    ("gildata", 22, 54, 18519),
    ("ifind", 25, 54, 18519),
    ("igo_open_data", 28, 54, 18519),
    ("imf", 31, 54, 18518),
    ("sec_edgar", 34, 54, 18518),
    ("sp_data", 37, 54, 18518),
    ("tianyancha", 40, 54, 18518),
    ("wind", 43, 54, 18518),
    ("world_bank", 46, 54, 18518),
    ("xhcj", 49, 54, 18518),
    ("yahoo_finance", 52, 54, 18518),
)

_TABLES: dict[SeedKind, tuple[tuple[str, int, int, int], ...]] = {
    "feature": _FEATURES,
    "skill": _SKILLS,
    "plugin": _PLUGINS,
}

# The owner field each extension kind reports back to its contract dataclass.
_OWNER_FIELD: dict[SeedKind, str] = {
    "feature": "feature",
    "skill": "command",
    "plugin": "source",
}


def owner_references(kind: SeedKind, owner: str) -> tuple[int, ...]:
    """Return every seed id registered for one extension owner, in order."""
    try:
        table = _TABLES[kind]
    except KeyError:
        raise ValueError(f"unknown extension kind {kind!r}") from None
    for entry_owner, first, stride, count in table:
        if entry_owner == owner:
            return tuple(first + stride * index for index in range(count))
    raise KeyError(f"unknown {kind} extension owner {owner!r}")


def resolve_seed_id(seed_id: int) -> dict[str, Any]:
    """Resolve one seed id to the extension owner that registered it."""
    if not 0 <= seed_id <= 1_000_000:
        raise KeyError(f"seed id {seed_id!r} is outside the registered layout")
    for kind, table in _TABLES.items():
        for owner, first, stride, count in table:
            offset = seed_id - first
            if offset >= 0 and offset % stride == 0 and offset // stride < count:
                return {
                    "kind": kind,
                    "seed_id": seed_id,
                    "owner": owner,
                    _OWNER_FIELD[kind]: owner,
                }
    # Unreachable: every integer in 0..1_000_000 sits on a registered owner stride.
    raise KeyError(  # pragma: no cover
        f"seed id {seed_id!r} is not registered by any extension owner"
    )
