# Evidence procurement & authorization packet

**Audience: a non-engineer operator, procurement lead, or compliance reviewer.**
This is a checklist of exactly what to buy or authorize, from which *class* of
provider, and the pass/fail tests each item must clear before it counts as
evidence for the five minimum-evidence conditions in
[INSTITUTIONAL_READINESS.md](INSTITUTIONAL_READINESS.md). Every row has an
owner and an objective pass criterion. Nothing here is a live-trading claim;
it is the evidence needed *before* any live claim could ever be considered.

> **Honesty rule that governs this whole document.** A module existing, an
> adapter compiling, or a SYNTHETIC fixture passing is **never** evidence. Only
> externally produced, externally timestamped, independently reviewable
> artifacts count. Conditions 1 and 2 below stay **BLOCKED** until the external
> purchase/authorization actually happens — no code change can satisfy them.

---

## 0. The 2025 holdout is SPENT — read this first

The current vendor pool **discloses survivorship bias** and **already has
results for the 2025 holdout** (see
[REPO_IMPROVEMENT_PLAN.md](REPO_IMPROVEMENT_PLAN.md)). That holdout is **SPENT**:
it has been inspected and cannot be reused as new forward or holdout evidence.
Any number already seen for 2025 is retrospective and carries selection and
survivorship contamination.

**Named forward window: `forward_2026H2`** — the calendar span label for H2 2026,
whose nominal span begins 2026-07-01. **2026-07-01 is not the eligibility boundary.**
The operative rule is the receipt's `splits.forward`: the first accepted decision must
occur on a market date **later than the recorded freeze (2026-10-07)**. Warmup is all
bars on/before 2026-09-18 (already inspected, warmup only).

This window is **NOT YET COLLECTED**. Its external timestamp requirement is now
**met**: the pre-registration in [REALITY_PREREGISTRATION.md](REALITY_PREREGISTRATION.md)
and its hash seal (`receipts/forward_record_preregistration_v1.json` + `.seal.json`)
were RFC 3161-anchored on **2026-10-08**
(`quality/timestamps/receipts__forward_record_preregistration_v1.json.seal.json.tsr`,
`chain_verified` + `fresh`). The freeze instant is therefore asserted by an external
authority, not by the repository.

What remains is **collection itself**: no forward close has been recorded. Editing the
pre-registration after outcomes arrive is detectable (the seal breaks) and voids the
record.

| Item | Status |
|---|---|
| 2025 holdout | **SPENT** — retrospective only, survivorship-biased |
| Forward window `forward_2026H2` | **NOT YET COLLECTED** — freeze externally anchored 2026-10-08 ✅; first eligible session = first market date after 2026-10-07 |

---

## 1. Condition 1 — licensed point-in-time market data

### 1a. What to buy / authorize

| # | Buy / authorize | Provider class (not a specific vendor) | Owner | Pass criterion |
|---|---|---|---|---|
| 1.1 | A **licensed** historical + daily point-in-time equity feed with **release and ingestion timestamps** and **revision (vintage) history** | A regulated market-data redistributor or the exchange's own licensed data product (e.g. a SIP/consolidated-tape licensee or a point-in-time fundamentals vendor). Not a free/undocumented API. | Procurement + Compliance | Contract explicitly grants: (a) PIT vintages, (b) redistribution/derived-use rights for internal research, (c) written universe-completeness attestation. |
| 1.2 | **Universe-completeness attestation** (the full listed universe including delisted names over the window) | Same provider, in writing | Compliance | Attestation letter on file; the feed is the *point-in-time* member list, not a present-day survivor list. |
| 1.3 | **Adjustment / corporate-action lineage** (split, dividend, spin-off factors with their own release/ingest timestamps) | Same provider | Procurement | Every adjusted price traces to a dated corporate-action record. |
| 1.4 | An **entitlement** (API key / license acknowledgment) wired to the vendor adapter's fail-closed `UNAVAILABLE` path | Engineering + Procurement | Engineering | With no entitlement the adapter reports `UNAVAILABLE` and **never** substitutes synthetic data (`synthetic_substitution=false`). |

### 1b. Required feed fields (per record)

Every record the feed delivers **must** carry these. A missing field makes the
feed **non-qualifying**; a broken causal chain or a decision-time leak makes it
**rejected**. The machine gate is `quant_fund.data.qualifying` (test matrix in
`tests/unit/data/test_qualifying_feed.py`).

| Field | Meaning | Acceptance |
|---|---|---|
| `event_time` | When the economic event occurred | Present, tz-aware |
| `available_time` | When the record became publicly available (release) | Present; `event_time <= available_time` |
| `ingested_time` | When we ingested it | Present; `available_time <= ingested_time`; not in the future |
| `source_id` | Which source produced the row | Present; a single consistent source per batch |
| `revision_id` | Version of this record | Present; no late/out-of-order revisions |
| universe completeness | The feed covers the full point-in-time universe | Attested in writing |
| adjustment lineage | Corporate-action factors with lineage | Present and traceable |

### 1c. Feed acceptance tests (must PASS to count as condition-1 evidence)

Run the coded gate and keep the output as the acceptance record. Owner:
Engineering; reviewer: independent (not the implementer).

| # | Test | Expected | Owner |
|---|---|---|---|
| 1.c.1 | Every required field present | Else **non-qualifying** | Engineering |
| 1.c.2 | `event_time <= available_time <= ingested_time` | Else **rejected** | Engineering |
| 1.c.3 | `available_time <= decision_time` at every consumer | A `decision_time` leak is **rejected** | Engineering |
| 1.c.4 | No late/out-of-order revision of an already-seen record | A late revision is **rejected** | Engineering |
| 1.c.5 | Universe-completeness + adjustment-lineage attested | Else **non-qualifying** | Compliance |
| 1.c.6 | Hashed PIT receipt reproduces and detects any edit | Receipt verifies | Engineering |
| 1.c.7 | No entitlement configured | `UNAVAILABLE`, `synthetic_substitution=false` | Engineering |

The feed is **QUALIFYING** (usable as condition-1 evidence) only when the
verdict is `QUALIFYING` — every field present and consistent, nothing rejected.

---

## 2. Condition 2 — broker adapter with authenticated order & fill reconciliation

### 2a. What to authorize

| # | Buy / authorize | Provider class | Owner | Pass criterion |
|---|---|---|---|---|
| 2.1 | A **brokerage account + API credentials** with authenticated order submission and fill/execution reporting | A regulated broker/dealer with a documented order+fill API and drop-copy/executions feed | Compliance + Trading | Credentials enable authenticated order and *execution* (fill) queries, not just order status. |
| 2.2 | A reconciliation entitlement: historical execution drop-copy or a fills export to reconcile against | Same broker | Trading | Fills can be independently re-pulled and matched. |

> Until 2.1 is authorized and connected **outside** this repo, condition 2 is
> **BLOCKED**. This lane ships the **paper/simulated** reconciliation surface
> only and sends **no live orders** (configuring a live endpoint raises
> `LiveEndpointRefused`). It makes **no** live-connectivity or live-P&L claim.

### 2b. Broker order/fill-reconciliation acceptance tests (condition 2)

The coded reference is `quant_fund.execution.order_recon` +
`quant_fund.paper.broker_adapter` (test matrix in
`tests/unit/execution/test_order_recon.py` and
`tests/unit/paper/test_broker_adapter.py`). A live adapter must clear the same
matrix against real broker drop-copy. Owner: Engineering; reviewer: independent.

| # | Test | Expected | Owner |
|---|---|---|---|
| 2.b.1 | Order-id <-> fill-id ledger reconciles | Every fill maps to exactly one order | Engineering |
| 2.b.2 | Duplicate fill (same `fill_id`) | Detected, **idempotent** (never double-counted) | Engineering |
| 2.b.3 | Missing fill (order expecting a fill, none arrived) | Detected and reported | Engineering |
| 2.b.4 | Out-of-order fill (regressing `fill_time`/`seq`) | Detected | Engineering |
| 2.b.5 | Orphan fill (unknown `order_id`) | Detected | Engineering |
| 2.b.6 | Position/cash drift vs broker statement | Drift **alarm** beyond tolerance | Engineering |
| 2.b.7 | Kill switch | Trips and blocks new orders | Trading |
| 2.b.8 | **Live-endpoint refusal** | Configuring a live endpoint **raises** | Engineering |
| 2.b.9 | Crash/resume mid-run | NAV seam parity, **no duplicate orders** | Engineering |
| 2.b.10 | Overfill (fill quantity above the residual) | Detected and **not applied** to position or cash | Engineering |
| 2.b.11 | Fill security differs from the order | Detected and **not applied** | Engineering |
| 2.b.12 | Cancel, then a late fill | Cancel is durable across resume; the late fill is **not applied**. A halt does not block cancel | Engineering |
| 2.b.13 | Paper session pre-trade | Short sales, missing marks, cash, and size caps refuse before the book moves. No network client | Engineering |

---

## 3. Condition 4 — venue measurement checklist (target venue)

Measure, from the **target venue** (not modeled assumptions), on the actual
instruments and size ranges to be traded. Owner: Trading + Quant; reviewer:
independent. Each item is a measurement with a date and a source, not a
parameter file.

| # | Measure | What to record | Owner |
|---|---|---|---|
| 4.1 | **Cost** | Effective spread, commissions, fees, taxes per instrument/size | Trading |
| 4.2 | **Liquidity** | ADV, book depth, participation-cap behavior at intended size | Trading |
| 4.3 | **Borrow** | Borrow availability, borrow fee (short cost), recalls/locate behavior | Trading |
| 4.4 | **Financing** | Margin/financing rate on leverage | Trading |
| 4.5 | **Failure modes** | Reject reasons, halts, partial fills, auction/closure behavior, feed gaps, order-rate limits | Trading + Eng |

Pass criterion for condition 4: all five rows carry a dated, sourced
measurement from the target venue, reviewed independently. Modeled values
(`configs/backtest.yaml` etc.) are **planning inputs**, not venue evidence.

---

## 4. Condition 5 — signed promotion receipt + explicit live authorization

| # | Requirement | Owner | Pass criterion |
|---|---|---|---|
| 5.1 | A **signed promotion receipt** whose immutable verifier passes | Quant + Independent reviewer | `dipcatcher verify-research` accepts it; the hash sidecar verifies |
| 5.2 | **Explicit, written live authorization** naming who authorized live capital | Compliance + Principal | Signed authorization on file, distinct from the promotion receipt |

Condition 5 is composed only after conditions 1–4 exist and a genuine
non-synthetic forward/shadow record (condition 3) has been collected and
admitted by the frozen referee. See
[FORWARD_SHADOW_POWER.md](FORWARD_SHADOW_POWER.md) for the admission rule.

---

## 5. Condition 3 — non-synthetic forward/shadow record

This is the one condition that is **collected over time** rather than bought.
It cannot be shortcut: the `forward_2026H2` window (Section 0) must be frozen
and externally timestamped **before** it runs, then collected for at least the
predeclared number of eligible paired sessions (see
[FORWARD_SHADOW_POWER.md](FORWARD_SHADOW_POWER.md) — 1,400 eligible paired
sessions at the +5 bps/day planning effect). The 2025 window is **SPENT** and
cannot serve as this record.

---

## 6. Non-engineer action summary

1. **Do not** treat any 2025 number as evidence — it is spent and
   survivorship-biased.
2. **Procure** the PIT feed + universe-completeness attestation + adjustment
   lineage (Section 1a). This unblocks condition 1 only after the acceptance
   tests (1c) pass on the real feed.
3. **Authorize** a broker account + API (Section 2a) for real order/fill
   reconciliation. Condition 2 stays blocked until then.
4. **Measure** the target venue (Section 3) with dated, sourced numbers.
5. **Freeze and externally timestamp** the `forward_2026H2` pre-registration
   (Section 0) before 2026-07-01, then collect the forward record honestly.
6. **Only then** compose a signed promotion receipt + explicit live
   authorization (Section 4).

Every step is evidence-gated. No code artifact in this repository, by itself,
satisfies any of conditions 1–5.
