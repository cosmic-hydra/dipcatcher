# Pre-trade risk engine

Standalone shadow checks for orders a simulated broker, paper loop, or the
operator live desk would send. The engine does not submit, cancel, or amend
anything. It is not imported by `quant_fund.execution.simulated_broker` or
`quant_fund.paper.loop`. The desk in `quant_fund.live` calls `decide` and
routes only after that decision. See [LIVE_DESK.md](LIVE_DESK.md).

This is an operational control. It is not a research score and it does not
report P&L.

## What it checks

Every `PretradeEngine.check` call evaluates, in a fixed order:

| Check | Fail-closed behavior |
| --- | --- |
| Kill switch | Latched. New orders and replaces stay denied until manual reset. Cancels are still allowed unless the latch reason is an internal fault. |
| Stale reference or account mark | Deny. The latch is not set; a fresh mark can clear it. |
| Non-finite price, quantity, side, or account value | Deny. |
| Session clock | Regular hours in the configured timezone, weekdays only. The close is exclusive. Weekends are closed. Exchange holidays are not modeled. |
| Halt | New orders and replaces denied. Cancels allowed. |
| Max order quantity and notional | Inclusive ceiling. Notional uses the larger of the order price and the reference mark. |
| Fat-finger collar | Limit price versus the reference, in basis points. Market orders are not collared. |
| Position quantity and notional | Projected position marked at the reference. |
| Gross and net exposure | Projected from the precomputed book. |
| Concentration | Projected name notional versus `max_name_concentration * nav`, precomputed when the account is updated. |
| Duplicate order | Same symbol, side, quantity tick, and price tick inside `duplicate_window_ns`. |
| Order rate and message rate | Orders and replaces count against both. Cancels count as messages only. |
| Reg SHO | A cash account cannot open or increase a short. A margin short needs `locate_ok`. If the symbol is flagged `sho_restricted` (Rule 201 circuit), a short must be a limit at or above the national best bid. |
| Pattern day trader | Same numeric rule as the event-sim constraint book: when equity is below `pdt_equity_threshold` (default 25,000), the order that would be the next day trade after `pdt_max_day_trades` (default 3) round trips in `pdt_window_sessions` (default 5) weekday sessions is denied. The adapter's session index counts weekdays and does not drop exchange holidays. |
| Good-faith violation | Selling a lot that was funded with unsettled cash before `settles_ns`. Cash accounts default to `restrict_to_settled_cash`, so a buy larger than settled cash is denied before a GFV lot can be created. |
| Buying power | Settled-cash ceiling when `restrict_to_settled_cash` is set. |
| Daily loss and trailing drawdown | Breach latches the kill switch. Recovering the account value does not clear it. |

Hard size and exposure checks are inclusive ceilings (`value > limit` denies) and commute: any evaluation order ORs to the same bitset. Stateful checks (rates, duplicates, the latch) are deterministic because the engine always applies them in the order above.

## Kill switch

`trip(actor, reason, ts_ns)` and a circuit-breaker breach set a latch. `reset` requires a non-empty actor and a non-empty reason and appends to the audit trail. Empty credentials raise and leave the latch set.

The audit trail is a hash chain. Each event stores `prev_hash` and `event_hash = SHA-256` of the canonical JSON of the event without `event_hash`. An internal exception also latches the switch and blocks cancels, because the book is no longer trustworthy. Reset clears the latch and does not rebuild rings or lot tables; after an internal fault, construct a new engine.

## Configuration

`configs/pretrade_risk.yaml` is schema version 1. Extra keys, unknown timezones, and unknown versions are rejected. The signed payload is canonical JSON (`sort_keys`, tight separators) of the validated model.

```python
from quant_fund.pretrade import load_pretrade_config

config, digest, signature = load_pretrade_config(
    "configs/pretrade_risk.yaml",
    hmac_key=hmac_key,  # at least 16 bytes; not stored in the file
)
```

`digest` is SHA-256 of the payload. `signature` is HMAC-SHA256 of the same bytes. Every `Decision` and every shadow log row carries both. The HMAC key is discarded after signing except inside a shadow adapter that still has to build per-broker engines.

## Shadow mode

```python
from quant_fund.paper.loop import run_paper_loop
from quant_fund.pretrade import ShadowRiskAdapter, load_pretrade_config

config, _, _ = load_pretrade_config("configs/pretrade_risk.yaml", hmac_key=key)
adapter = ShadowRiskAdapter(config, hmac_key=key, log_path="shadow.jsonl")

# Brokers created inside the paper loop:
with adapter.observe_simulated_broker():
    run_paper_loop(...)

# Or one broker the caller already holds:
adapter.attach(broker)
```

The wrapper evaluates the order, appends an allow or deny record, then calls the original `submit`, `cancel_order`, or `amend_order`. Observer exceptions are recorded as `internal_error` and are not raised into the broker. `behavior_changed` on each record is false. The adapter does not call a live gateway.

Regulatory lots and pattern-day-trade memory are updated from observed fills after the broker returns. Position and marks are read back from the broker on the next submission, so a broker reject cannot leave the exposure book ahead of the fill.

## Latency

The hot path is `PretradeEngine.check` on a precomputed `HotBook`: fixed rings, precomputed collar bands, concentration dollars, and loss floors. The allow path does not allocate containers, strings, or decision records. CPython still creates transient integers for arithmetic. Numba and Rust are not used; the measured CPython path is what the gate runs.

```bash
uv run python -m quant_fund.pretrade.bench --gate
```

The command prints a JSON distribution (`min`, `p50`, `p90`, `p95`, `p99`, `p99.9`, `max`, `mean`) for several trials. GC is disabled during each sample loop. The gate passes when the median trial, ordered by p99, has p50 under 5 microseconds and p99 under 20 microseconds. A sample that is not an allow fails the run, so the gate cannot pass by timing an early deny. CI runs this command in the `pretrade-risk` job, without coverage tracing. A coverage run still executes the benchmark and checks that `gate_pass` matches the measured percentiles; it does not treat an instrumented timing as a gate failure.

The measured scenario is a limit buy inside every ceiling, with rate and duplicate windows of 50 microseconds and a 1 microsecond step, so those rings hold about 50 live entries and expire one per call. PDT is evaluated and is not binding at the scenario's equity. Session bounds use `datetime.timestamp`, which is floating-point and not nanosecond-exact; the hot path only compares the precomputed integer bounds.

## Limits that are intentional

- Holidays are not a calendar. Saturday and Sunday are closed. The weekday session index used for PDT does not skip holidays.
- A full lot table, duplicate ring, or unsettled-cash queue denies with `state_full` instead of allocating.
- Shadow mode will not stop the simulated broker. A deny in the log can sit next to a broker fill.
- The engine does not model locate files, SIP halt feeds, or a broker's buying-power API. Those inputs are fields on the snapshot the caller refreshes.
