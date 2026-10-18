# Institutional readiness matrix

Dipcatcher is Artificial Hedge's research lab. This matrix is deliberately
evidence-gated: a capability is not marked ready because a module exists or a
synthetic fixture passes.

| Control area | Current state | Evidence / gate |
|---|---|---|
| Point-in-time data contract | Implemented for file adapters; revision-aware on the evaluation side | PIT timestamp validation, duplicate-key checks, OHLCV checks, manifest hash verification, and decision-time filtering for late cross-sectional/market aggregates. Wave 15 adds `validation/vintage_eval.py` (VINTAGE-TS): validity-interval reconstruction, delayed-label filtering, and hindsight-contamination audits over SYNTHETIC revision regimes — the *evaluation* layer now detects vintage cheating even though true as-of vintages on the public tape remain a procurement gap. |
| Configuration safety | Implemented | Unsupported data sources, non-finite simulator parameters, and non-stationary synthetic persistence fail validation |
| Data lineage | Implemented | `dipcatcher doctor` requires source identity, four canonical artifacts, SHA-256s, nonnegative row counts, and data-root containment |
| Research reproducibility | Implemented | Immutable receipt binds Git revision, dirty-worktree hash, config, dataset content, runtime, package versions, and benchmark catalog version |
| SOTA scientific benches | Implemented as research diagnostics on the lab panel | Canonical verifier requires the versioned 23-family catalog; fixture-only toys are `dgp=fixture` and excluded from panel Kupiec H-rows; panel-or-skip (empty blob) fails scorecard honesty rather than silently mixing DGPs |
| Validation integrity | Implemented | Walk-forward, purge/embargo, CPCV audit, HAC/DM, multiple-testing and conformal family gates. Waves 12–16 add the anytime-valid layer: e-BH/e-LORD/e-SAFFRON online FDR, minimax-optimal conformal e-detectors, rank confidence sequences (anytime-valid ranker ordering), replicable conformal (auditable calibration thresholds), and `validation/agent_referee.py` — a frozen post-submission-only betting referee so agent-proposed factors are judged at every stopping time by a procedure the proposer cannot touch. |
| Numerical risk controls | Implemented | Covariance inputs are finite and square; PSD repair is deterministic eigenvalue clipping with finite-output guarantees |
| Execution research | Implemented as simulation | Next-open fills, costs, participation, Almgren–Chriss/TWAP comparisons, and tamper-evident backtest artifacts |
| Paper / shadow operation | Implemented as simulation | Crash-resumable simulated broker; resume appends prior orders/equity/shadow-equity/positions/cash rows and fill receipts; deterministic target/exposure ordering; kill switch, champion/shadow dry-run, no live capital |
| Promotion | Fail-closed | Synthetic evidence, missing metrics, non-finite metrics, and invalid receipts cannot promote |
| CI reproducibility | Implemented | Frozen `uv.lock`, lock consistency check, full tests, type/lint checks, canonical receipt verification |
| CI token scope | Implemented | Workflow `GITHUB_TOKEN` is restricted to repository contents read access; no job in CI requires write access |
| Package distribution | Implemented | CI builds wheel and source distribution, installs the wheel without the checkout installed, and runs the installed `dipcatcher --help` entry point |
| API security boundary | Implemented for local/service operation | API-key authentication, loopback fail-closed default, constant-time key comparison, secure response headers, artifact-root containment, 64 KiB streamed request-body cap, and strict request schemas |
| Dependency and code security | Implemented as CI gates | Locked dependency `pip-audit` report, medium/high Bandit gate, and retained machine-readable audit artifact |
| Container operations | CI-gated | Non-root runtime user, loopback default, healthcheck, immutable dependency sync, host-local optional MLflow binding; CI builds the image and waits for its healthcheck |
| Vendor market data | **Blocked — procurement/authorization pending** | A fail-closed licensed-vendor adapter and a condition-1 `QUALIFYING` gate exist (`quant_fund.data.qualifying`), enforcing `event_time <= available_time <= ingested_time`, `available_time <= decision_time`, revision handling, and hashed PIT receipts, with an `UNAVAILABLE` path and **no synthetic substitution**. This is **not evidence**: no entitlement and no qualifying externally timestamped complete-universe feed are configured. The local retrospective snapshot does not qualify. See [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). |
| Live broker connectivity | **Not implemented — authorization pending** | No broker credentials and no live P&L claims. The paper/simulated reconciliation surface (`quant_fund.execution.order_recon`, `quant_fund.paper.broker_adapter`) is the condition-2 reference shape; configuring a live endpoint **raises** (`LiveEndpointRefused`). `dipcatcher trade readiness` and in-chat `/trade` report that block. `dipcatcher trade session` (`quant_fund.paper.trading_session`) is a simulated book with no broker client. `dipcatcher live` (`quant_fund.live`) journals shadow orders, rehearses on an in-process loopback, or writes drop-folder tickets after an explicit arm. That desk is not authenticated broker reconciliation and does not satisfy this condition. See [LIVE_DESK.md](LIVE_DESK.md) and [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md). |
| Live readiness | **Blocked by missing external evidence** | Requires authorized vendor data, a real broker adapter with authenticated reconciliation, venue measurements, a non-synthetic forward record, and independently verified evidence. The 2025 holdout is **SPENT**; the named forward window `forward_2026H2` is **NOT YET COLLECTED** (freeze externally anchored 2026-10-08; first eligible session = first market date after the 2026-10-07 freeze, not 2026-07-01). |


## Dual honesty catalogs (research vs analytics export)

Two different surfaces — do not conflate:

1. **Research scorecard / family blobs** — `FORBIDDEN_RESEARCH_METRIC_KEYS` /
   `family_blob_forbidden_metrics_absent` reject sharpe/sortino/calmar/pnl/nav
   *key tokens* in lab headlines.
2. **Paper/backtest `analytics_export`** — may nest equity `nav_*` and stress
   `*_pnl` diagnostics under `ANALYTICS_SCHEMA_KEYS`; honesty gate is
   `live_pnl_claim=false` via `validate_analytics_export` (fail-closed on true).

Never treat a valid analytics export as a research family scorecard blob.
`/ready` is fail-closed when `data_manifest` or the immutable research receipt is
missing/invalid (the configs allowlist resolves `_CONFIGS_DIR` before containment
checks). A green readiness response therefore proves both data provenance and
research-artifact integrity for the configured research runtime.

## Minimum evidence before any live claim

All of the following must be present and independently reviewable:

1. A licensed, point-in-time data source with release and ingestion timestamps.
2. A broker adapter with authenticated order and fill reconciliation.
3. A non-synthetic holdout and deployment-shaped forward/shadow record.
4. Cost, liquidity, borrow, financing, and failure-mode measurements from the
   target venue.
5. A signed promotion receipt whose immutable verifier passes and whose live
   authorization is explicit.

Until those conditions exist, Dipcatcher outputs are research, backtest, or
simulated paper/shadow evidence only.

### Where each condition is anchored in this repo

Each condition maps to a concrete artifact/command that enforces or will hold
it. The acceptance tests per condition are
[EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md); none of these code
artifacts *satisfies* a condition by itself.

| # | Condition | Repo artifact / command | Status (2026-10-08) |
|---|---|---|---|
| 1 | Licensed PIT data with release + ingestion timestamps | `src/quant_fund/data/qualifying.py` (`quant_fund.data.qualifying`) — condition-1 `QUALIFYING` gate: `event_time <= available_time <= ingested_time`, `available_time <= decision_time`, revision handling, hashed PIT receipts, `UNAVAILABLE` path with `synthetic_substitution=false` | **BLOCKED — procurement.** No entitlement and no qualifying feed; acceptance tests EVIDENCE_PROCUREMENT §1c must pass on the real feed |
| 2 | Broker adapter with authenticated order + fill reconciliation | `src/quant_fund/execution/order_recon.py` + `src/quant_fund/paper/broker_adapter.py` (condition-2 reference shape; a live endpoint raises `LiveEndpointRefused`). Operator paper book: `quant_fund.paper.trading_session` (`dipcatcher trade session`). `quant_fund.live` is the operator desk and is not this adapter. Acceptance tests EVIDENCE_PROCUREMENT §2b | **BLOCKED — authorization.** No broker credentials exist |
| 3 | Non-synthetic holdout + deployment-shaped forward/shadow record | Hash-sealed pre-registration ([REALITY_PREREGISTRATION.md](REALITY_PREREGISTRATION.md), `forward_2026H2`); `uv run dipcatcher forward-shadow freeze …`; `dipcatcher reality sweep` / `reality survivorship` fail closed (exit 4) instead of substituting synthetic data; admission referee `src/quant_fund/validation/agent_referee.py`; power rule [FORWARD_SHADOW_POWER.md](FORWARD_SHADOW_POWER.md) | **NOT YET COLLECTED.** 2025 holdout is SPENT; `forward_2026H2` must be externally timestamped before it runs |
| 4 | Venue cost, liquidity, borrow, financing, failure-mode measurements | Measurement checklist EVIDENCE_PROCUREMENT §3 (dated, sourced, independently reviewed). Repo-side consumers are planning inputs only — `configs/backtest.yaml` modeled costs, `uv run dipcatcher execution-sensitivity` latency/impact grid, [STRESS.md](STRESS.md) failure-mode catalog | **BLOCKED — no venue measurements exist.** Modeled values are never venue evidence |
| 5 | Signed promotion receipt with passing immutable verifier + explicit live authorization | Signed pins + checkpoint (`make sign-pins`, `make checkpoint`, Ed25519 `GATE_SIGNING_KEY`); verifiers `uv run dipcatcher verify-research` / `verify-receipt`, `make receipts-reverify`, `make evidence-audit`, `make checkpoint-chain`, `make verify-rotations` (see [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)); promotion fail-closed (`would_promote_live=false`, `live_pnl_claim=false`) | Verifier machinery exists and is mutation-tested; **explicit written live authorization does not exist** and is composed only after conditions 1–4 |

A working fail-closed adapter, a compiling module, or a passing SYNTHETIC
fixture is **not** evidence and never marks a condition ready. Conditions 1 and
2 remain **BLOCKED on procurement/authorization** — they require an external
purchase/entitlement and an independently reviewable artifact, not code. The
actionable checklist for what to buy/authorize and the pass/fail acceptance
tests for each condition is [EVIDENCE_PROCUREMENT.md](EVIDENCE_PROCUREMENT.md).

**Holdout status:** the 2025 holdout is **SPENT** (the vendor pool already has
results for it and discloses survivorship bias). The named forward window is
**`forward_2026H2`** (the calendar span label for H2 2026), currently **NOT YET
COLLECTED**. Its pre-registration is hash-sealed and was **externally RFC 3161
anchored on 2026-10-08** (`chain_verified`, `fresh`), so the freeze instant is
asserted by a third party rather than by this repository.

**`window.start = 2026-07-01` is the span label, not the eligibility boundary.** The
operative rule is the receipt's `splits.forward`: the first accepted decision must fall
on a market date **later than the recorded freeze (2026-10-07)**. Warmup is all bars on
or before 2026-09-18. See [REALITY_PREREGISTRATION.md](REALITY_PREREGISTRATION.md).

## Waves 12–16 addendum (2026-09-30)

What changed relative to the five conditions — and what did not:

- **Condition 1 (licensed PIT data)**: unchanged. Procurement, not code.
  The evaluation side is now vintage-aware (`vintage_eval`), so when a
  real as-of source arrives the harness can already audit revision
  hindsight instead of silently absorbing it.
- **Condition 3 (forward/shadow record)**: strengthened in *kind*, not in
  substance — the frozen referee (`agent_referee`) and rank confidence
  sequences give any future forward record anytime-valid admission
  semantics (no optional-stopping inflation), and `capability_value`
  makes self-evolution claims falsifiable via Cap-swap accounting. No
  non-synthetic forward record exists yet.
- **Condition 5 (signed promotion receipt / verifier)**: the immutable
  verifier itself is now mutation-tested. The precise history matters: the
  often-quoted `research/verify.py` figure of 99.52% (209/210 killed) was
  measured against a 50-mutant seed-7 sample, **not** the full 210. The
  genuinely-full 210-mutant run first surfaced 15 survivors; 14 were closed
  by dedicated kill tests (2026-10-01) and the remaining one (M0032) is a
  provably equivalent mutant pinned by a documentation test
  (`tests/unit/research/test_research_verify.py`). `catalog/registry.py`
  100% (24/24). "Independently reviewable" now includes measured
  fault-detection evidence for the reviewer code.
- **Scorecard governance**: `docs/BENCHMARK_FAMILY_LIFECYCLE.md` adopts
  OPTIONAL→REQUIRED/RETIRED lifecycle rules, runtime budgets, and review
  cadence for the benchmark-family scorecard.
- None of the above is market evidence. Every new module ships seeded
  SYNTHETIC correctness tests; the honesty contract is unchanged.
