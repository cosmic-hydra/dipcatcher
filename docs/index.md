# dipcatcher

**dipcatcher** is Artificial Hedge's state-of-the-art harness for orchestrating its **fx1** and **fx1-lite** models from a chat-first, Codex-style CLI: plain text is a conversation with the active model, and `/superpower` plans and runs the research families that fit your goal. Type `dipcatcher` in a terminal to open the chat; see [the dipcatcher CLI](DIP_CONCIERGE.md) for the design and its current status.

Underneath the chat, `quant_fund` is the research harness: the data engine, the evaluation bench, and the receipt verifier. **fx-1** (`src/fx1`) is a sub-project in this repository. It contains the corpus builder, the eval bank, and training plumbing, plus a training plan. No trained checkpoint is in the tree. The declared base constant is `moonshotai/Kimi-K3`. This site documents the harness and that plan.

The lab builds a point-in-time market-state object, scores forecasts with proper scoring rules, and keeps every claim tied to a sealed receipt. Simulated paper and shadow runs exercise those same gates. `dipcatcher live` is the operator desk: shadow by default, an in-process SYNTHETIC rehearsal, and a drop folder for an operator-owned bridge. That desk does not authenticate to a broker and does not claim live performance. See [LIVE_DESK.md](LIVE_DESK.md).

## Honesty and receipts

Research results are proper scores:

- pinball
- CRPS
- PIT
- QLIKE
- Brier
- ECE
- Kupiec
- HMM likelihood

A research claim is reproducible from an immutable receipt. `dipcatcher verify-research` checks the seals — config hash, dataset bytes, git revision, package versions, and the benchmark catalog — and rejects a forged live-trading claim. Receipts live under `receipts/` and `data/metadata/research/`.

Sharpe, Sortino, Calmar, P&L, and NAV are excluded from research headlines. The token list is `FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog`, mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS`.

See [Receipt verification](RECEIPT_VERIFICATION.md) and [Institutional readiness](INSTITUTIONAL_READINESS.md).

## Real data and SYNTHETIC data

| Label | Meaning |
|---|---|
| **SYNTHETIC** | Simulator or planted-factor panel. A passing score is an engine correctness test, and the label stays on the receipt. |
| **public / file** | Local CSV or Parquet, including session-close tapes. Scores stay on proper rules. `available_time` is a publication convention, not a SIP vintage. |
| **public sources** | Registered feeds collected with `dipcatcher collect` (FRED, EDGAR, Treasury, CFTC, Binance klines, Hugging Face `hf_ohlcv_1m`, and others). Each collection writes a JSON receipt. |

SYNTHETIC evidence cannot take a champion or live alias. The full rules are in [Data source labels](DATA_SOURCE_LABELS.md). Minute bars from Hugging Face are specified in [HF OHLCV-1m](HF_OHLCV_1M.md).

## Where to go

- [Production product contract](DIPCATCHER_PRODUCT_CONTRACT.md) — proposed chat-first dipcatcher experience, `/superpower`, and measurable release gates
- [Claude handoff](CLAUDE_DIPCATCHER_HANDOFF.md) — implemented chat entry point and core guards, verification results, and the remaining production work
- [Getting started](getting-started.md) — install, doctor, a SYNTHETIC research smoke, and this site
- [Examples gallery](examples.md) — runnable scripts, each labeled SYNTHETIC or tracked snapshot
- [Evidence](evidence/index.md) — numbers copied from sealed receipts
- [Architecture](ARCHITECTURE.md) — package layout and the decision clock
- [Data contracts](DATA_CONTRACTS.md) — point-in-time fields and lake tables
- [Validation](VALIDATION.md) — walk-forward, purge, embargo, CPCV
- [Mathematical specification](MATH_SPEC.md) — formulas the code implements
- [Research centre](RESEARCH_CENTRE.md) — benchmark families and what a lab headline is
- [fx-1](FX1.md) — corpus, eval, training plumbing, and the training plan
- [API reference](api/index.md) — docstrings for the public harness modules
