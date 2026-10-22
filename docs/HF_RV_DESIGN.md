# HF-RV — high-frequency realized variance design (refs Wave 140 gap)

**Status**: design drop, not implementation. This document is a contract for a
follow-up implementation wave.

**Refs**: persistent "Next gap" across `day_grind_progress.md` Waves 130–140
("high-frequency RV remains unavailable"). Wave 115–118 landed a daily
Parkinson-based RGARCH (`intraday_realized_variance=false`); this drops the
5-min intraday extension on the same fail-closed / PIT-correct / source-honest
contract.

## 1. Scope

Compute intraday realized variance (RV) on a 5-min grid from one or more HF
sources, fail-closed on null/missing/patchy data, PIT-correct, and expose a
single function that plugs into the existing overlay / forecast / risk
pipeline without changing the public surface of `forecast_asof`,
`optimize_asof`, or `/risk/portfolio`.

## 2. Data ingest

### Required columns per 5-min bar

| Column | Type | Notes |
|---|---|---|
| `event_time` | `pd.Timestamp` (tz-aware) | bar close time, UTC |
| `available_time` | `pd.Timestamp` (tz-aware) | when the bar became observable |
| `open`, `high`, `low`, `close` | `float` | per-bar OHLC |
| `volume` | `float` (optional) | used by jump test as tiebreaker |

### Sources (via `fx1 sources ...`)

- **Primary**: CLS (FX 1-min) — already on the PIT-correct `fx1 sources`
  harness, credentials in `KIMI_API_KEY` / `AGENT_GW_TOKEN`.
- **Secondary**: Binance public 1-min klines (for crypto overlay), IMF
  Riezer (overnight RV cross-check).
- **Authority order**: CLS > Binance > IMF, recorded in the receipt. If the
  primary source is missing at `asof`, the receipt records
  `hf_rv_source_missing` and the function returns `HFRVResult(honest=False)`
  — fail-closed.

### Schema contract

- One `pandas.DataFrame` per `(security_id, asof)`.
- 5-min grid aligned to UTC clock (e.g. 00:00, 00:05, …, 23:55), allowing
  for source-specific clock drift ≤ 60 s (recorded as
  `clock_drift_seconds`).
- Holes (missing bars): fail-closed on more than 1 contiguous missing bar
  or > 5% of the trailing window. Single-bar holes are linearly interpolated
  with the interpolation flagged in `holes_count`.
- Bars whose `available_time > asof` are dropped (PIT gate, parallel to
  Wave 119).

## 3. Aggregation

### 5-min realized variance

```
RV = sum over 5-min bars i in [asof - window, asof) of (log(p_i / p_{i-1}))^2
```

with `window` defaulting to 1 trading day (288 bars at 5-min).

### Bipower variation (jump-robust)

```
BPV = (pi/2) * sum |r_i| * |r_{i-1}|
```

where `r_i = log(p_i / p_{i-1})`. BPV is consistent for the diffusive part
of the quadratic variation; the difference `RV - BPV` is the jump
contribution.

### Jump test (Lee–Mykland)

Use the rolling Lee–Mykland statistic over a 5-min bar:

```
J_i = |r_i| / sqrt(BPV_window)
```

Reject the `|J_i| > sqrt(2 log n)` null at significance 1%. Jump count is
returned as `n_jumps` and the bars flagged are excluded from the RV
estimate; the function returns both `rv_5min` (with jumps) and `rv_5min_jump_clean`
(without), with a receipt `n_jumps_stripped`.

### Tick subsampling

For sources with multi-tick bars (CLS FX is mid-frequency), apply
5-tick sub-sampling with a deterministic seed derived from `asof` to avoid
microstructure-noise inflation. This is opt-in via the
`apply_tick_subsample=True` parameter (default).

## 4. PIT gate (parallel to Wave 119)

- Drop bars with `available_time > asof` (would leak future information).
- Null `available_time` → fail-closed (`HFRVResult(honest=False, reason="null_availability")`).
- Compute `HFRVResult.available_time = max(bars.available_time)`.

This matches the optimizer covariance PIT contract from Wave 119 and the
GARCH overlay PIT contract from Wave 111 — same shape, same fail-closed
behavior.

## 5. Plug-in points

### `optimizer.covariance` (rgarch branch)

Currently: `rgarch_market_forecast_asof` scales trailing name-covariance by
causal Parkinson-based RGARCH. After HF-RV lands, the same overlay function
checks for `vol_hf_rv.joblib` artifact first; if present, uses HF-RV's
`rv_5min_jump_clean` as the per-name overlay scale; falls back to
`realized_garch_market_forecast_asof` if not. `MarketState.market_risk_overlay`
stamps `hf_rv` vs `realized_garch` vs `garch` so the receipt is unambiguous.

### `forecast_asof` / `check_order`

Same overlay chain. `max_predicted_vol` checks the HF-RV one-step market
sigma, fail-closed on missing intraday.

### Diagnostics

- `data_quality.hf_rv_5min` — boolean per `asof` for whether HF-RV was
  computable.
- `hf_rv_diagnostic_string` — nonempty when stamped; the per-source
  attribution + hole count + jump count.

## 6. Test surface

Landed coverage lives in `tests/unit/test_hf_rv.py` (contract cases below)
and `tests/test_hf_rv_scaffolding.py` (public surface + fail-closed pins).

### Synthetic intraday

- Brownian motion + sparse jumps: expect `rv_5min ≈ σ^2 * window` and
  `n_jumps` matching the injected count.
- Jump detection accuracy: at SNR=10, ≥ 95% true-positive rate; at SNR=1,
  ≤ 5% false-positive rate (Lee–Mykland operating characteristics).
- BPV consistency: `RV - BPV ≥ 0` on synthesized jumpy series.

### PIT contamination

- Feed a bar with `available_time > asof`, expect drop + receipt stamp.
  Feed null `available_time`, expect `HFRVResult(honest=False)`.

### Fail-closed for missing intraday

- Empty bars, single-bar windows, 100% holes: expect
  `HFRVResult(honest=False)` and no `market_risk_overlay=hf_rv` stamp.

### Source authority

- CLS + Binance both available → CLS wins, receipt records
  `primary=cls, secondary=binance`.
- Source missing → `hf_rv_source_missing` stamp + `honest=False`.

## 7. Open questions

1. **HF source authority** — is CLS authoritative for all FX, or do we
   accept vendor-specific feeds for specific name ranges? Currently the
   answer is "CLS for any name that has a CLS mapping" but the mapping
   table is not in the repo. Either we add `data/raw/sources/cls_map.json`
   or we accept the source-agnostic receipt stamp.
2. **PIT clock alignment** — sources use different ingest clocks (CLS
   `event_time` is bar close UTC; Binance uses exchange time). We need
   either a per-source clock-correction table or an explicit
   `clock_drift_seconds` allowance.
3. **Patchy intraday** — what if the trailing 1-day window has 30%
   holes? Current spec fails-closed, but a real deployment might want
   partial success. Decision: keep fail-closed (matches project
   contract); revisit if user feedback says otherwise.
4. **Cross-asset HF-RV** — current scope is per-name. For correlation /
   co-RV we'd need a 2-bar frame with aligned `event_time`. Deferred.
5. **Receipt schema** — `HFRVResult` is a Python dataclass, but receipts
   are JSON. Do we serialize via `dataclasses.asdict` or define a
   stable JSON schema under the quality pin directory? Prefer a pinned
   schema file once the wire format is frozen (not landed yet).

## 8. Honest reporting (per AGENTS.md)

- The resulting HF-RV-fed overlay will be evaluated using **proper scores**:
  pinball, CRPS, QLIKE, Brier, ECE, Kupiec, HMM likelihood.
- **No** Sharpe / Sortino / Calmar / P&L / NAV headlines.
- **No** live-trading claims.
- **No** synthetic data labeled as live.
- Receipts are immutable; the HF-RV result hash is part of the receipt
  ledger.

## 9. Implementation plan (one wave)

- This design is for one implementation wave, not staged. Estimated: 4–6
  PRs, each ≤ 500 lines, each cleared by `make lint && make typecheck &&
  make test`. The full impl wave is a follow-up to this design drop.
- The follow-up should land the **scaffolding** in this same PR
  (`compute_hf_rv` raises `NotImplementedError` with the planned
  signature) so the contract is pinned.

## 10. Scaffolding signature (pinned by tests)

```python
from dataclasses import dataclass
import pandas as pd

@dataclass
class HFRVResult:
    rv_5min: float                  # realized variance, 5-min grid
    bpv: float                      # bipower variation
    jump_stat: float                # max |Lee-Mykland| over the window
    n_jumps: int                    # count of bars flagged as jumps
    rv_5min_jump_clean: float       # RV excluding flagged bars
    n_obs: int                      # # of 5-min bars used
    asof: pd.Timestamp
    available_time: pd.Timestamp    # latest observable restatement
    source: str                     # 'cls' / 'binance' / 'imf'
    source_secondary: str | None    # cross-check source if present
    clock_drift_seconds: float
    holes_count: int                # single-bar holes interpolated
    honest: bool                    # fail-closed flag (False = not used)

def compute_hf_rv(
    bars: pd.DataFrame,
    asof: pd.Timestamp,
    available_time: pd.Timestamp,
    *,
    window_bars: int = 288,
    apply_tick_subsample: bool = True,
    subsample_stride: int = 5,
    rng_seed: int | None = None,
) -> HFRVResult:
    """Scaffold — raises NotImplementedError. See docs/HF_RV_DESIGN.md."""
    raise NotImplementedError("hf_rv_committed: see docs/HF_RV_DESIGN.md")
```
