# Operator live desk

`dipcatcher live` is the desk an operator runs. It is not a broker session
and it does not satisfy institutional condition 2. Reports keep
`live_pnl_claim` false, `would_promote_live` false, and
`evidence_conditions` blocked. Research headlines still do not carry
Sharpe, Sortino, Calmar, P&L, or NAV.

The paper broker is unchanged: a live endpoint still raises
`LiveEndpointRefused`.

## Venues

| Venue | What leaves the process | Fills |
|---|---|---|
| `shadow` (default) | Nothing | None. The order is journaled as `shadowed`. |
| `loopback` | Nothing. Matching stays in-process. | Immediate and full, at the limit. Labeled `SYNTHETIC`. |
| `drop` | One JSON ticket under `outbound/` | Only what the operator's bridge writes to `inbound/`. |

Drop transmission needs both:

1. `dipcatcher live arm --venue drop --confirm SEND-LIVE-ORDERS`
2. `DIP_LIVE_ARM=1` in the environment of the process that submits

Either one alone holds the order. The chat command `/live` prints this card
and does not transmit. `/harness run live` is status only, and it asks
before it runs.

A new desk has no buying power. The first `arm` or `submit` has to pass
`--cash` and `--account`. Those values are the risk-book limits, not a
broker balance and not a performance figure. Later commands reuse the
persisted numbers and refuse a second grant.

## Layout

The default directory is `~/.dipcatcher/live` (`DIP_LIVE_ROOT` overrides it).

```text
hmac.key          local pre-trade HMAC, mode 0600, created on first use
arming.json       venue. A file that sets live_pnl_claim true is refused
state.json        positions, reservations, lifecycle trace
journal.jsonl     hash-chained audit
outbound/         order and cancel tickets
inbound/          bridge events
.lock             one process at a time; a dead pid is cleared
```

Cash and positions are the risk book used for buying power and limits.
They are not a performance report. Sale proceeds stay unsettled until the
configured settlement time, which is the cash-account rule.

## Tickets and events

Order ticket `outbound/<order_id>.json`:

```json
{
  "schema": "live_order_ticket.v1",
  "ticket": "order",
  "order_id": "<id>",
  "symbol": "AAPL",
  "side": "BUY",
  "qty": 1,
  "limit_price": 100,
  "live_pnl_claim": false,
  "would_promote_live": false,
  "evidence_conditions": "blocked"
}
```

The bridge polls that directory, sends the order through the operator's
broker, and writes one event per file in `inbound/`:

```json
{
  "schema": "live_broker_event.v1",
  "kind": "ack",
  "order_id": "<id>",
  "event_id": "ack-1"
}
```

`kind` is `ack`, `reject`, `fill`, or `cancel`. A fill needs a positive
`qty` and `price`. `fee` defaults to 0. `dipcatcher live poll` applies new
`event_id`s once. Duplicate ids are ignored. An orphan fill or a fill that
does not match the lifecycle latches the kill switch.

Cancel tickets are `outbound/cancel-<order_id>.json`. A ticket that was
already sent is not cancelled locally while `DIP_LIVE_ARM` is off, so the
desk cannot pretend a working order is gone.

## Risk and the halt

Every new order and cancel passes `PretradeEngine` (`configs/pretrade_risk.yaml`)
before a ticket exists. The session clock is the equity session in that
file. Limits, the kill switch, and the duplicate window are persisted, so
a new process does not reset them.

```sh
dipcatcher live kill --reason "feed gap"
dipcatcher live reset --actor ops --reason "feed restored"
```

Reset re-enables drop transmission when the venue is still `drop` and
`DIP_LIVE_ARM=1`. An internal fault is not resettable; start a new desk
directory. A journal that does not match `state.json` freezes the desk.

`dipcatcher live flatten` submits a reduce-only limit at the last mark for
each position. Each of those orders still has to pass pre-trade, and a
drop flatten transmits only when the arming switch is on.

## What this is not

- Not a brokerage login. No API key for a venue is read.
- Not condition 1 (licensed point-in-time data), condition 3 (the forward
  record), condition 4 (venue measurements), or condition 5 (a signed
  promotion). See [INSTITUTIONAL_READINESS.md](INSTITUTIONAL_READINESS.md).
- Not evidence. A loopback fill is a SYNTHETIC correctness result. A drop
  fill is the operator's bridge report. Neither is a research score and
  neither promotes a strategy.
