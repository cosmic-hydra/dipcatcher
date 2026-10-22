# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version source is `fx1.__version__`.

## [Unreleased]

### Added

- **Release and deploy readiness**: `release.yml` preflight (tag ↔
  `fx1.__version__` ↔ CHANGELOG, same-SHA CI/fx1/proofcore/matrix, plus
  `evidence-audit`), wheel smoke, CycloneDX SBOM, Sigstore/SLSA on tags only,
  container build + `/health` wait (no registry push), `make release-dry-run`,
  `make docs-consistency`, production deploy runbook
  (`docs/PRODUCTION_DEPLOY.md`) covering config/secrets/`deploy/observability`,
  and research-API `GET /ready`. Does not tag, publish, or authorize live
  trading.

- **PROOFCORE wave 2 — causal proven runs + cryptographic replay + IO
  guard**: `quant_fund.proofcore.scheduler` (pure decision-window scheduler),
  `quant_fund.proof.runner.run_proven` (causal walk-forward over a declared
  RunSpec allowlist surface; hash-chained decision trace; trace/env/seeds
  sidecars committed in the config sidecar; atomic no-partial-bundle mint),
  `quant_fund.proof.replay.replay_bundle` (bit-exact re-execution gate,
  env-fingerprinted, first-divergence reporting), and
  `quant_fund.leakage.guard.install_io_guard` (runtime interposition on raw
  `read_parquet`/`open()` inside decision windows; the runner drives it in
  enforce mode). New CLI: `quant proof run --spec … --vault-root …
  --bundle-dir … [--signing-key-env VAR]` and `quant proof replay --bundle …
  --bundle-dir … [--vault-root …]`. `verify_bundle` additionally hash-checks
  the wave-2 sidecars committed in `config["sidecars"]`. Scope limits are
  documented in `docs/proofcore/README.md` (fixed grid, no calendar
  awareness, two reference estimators; replay verdicts are re-executability
  evidence only). The wave-1 `run_backtest_proven` API stays fail-closed by
  design (WAVE2.md amendment A4).

### Fixed

- **reality-filter CI**: `data/metadata/proofcore.duckdb` is not in the
  repository at HEAD or on main, so `quant proofcore export` creates an
  empty database and writes 0 trial rows. `quant reality preflight --db`
  exits 3 with `REALITY_FILTER_SKIP` for that missing file (and for an
  export that has no rows). `make reality-gate` exits 0 and the workflow
  annotates the skip as a notice. Recorded rows are scored with the same
  thresholds, and a verdict other than `pass` fails the job.
- **A1 F1 (PROOFCORE W4)**: `hedge_lab/scoreboard.py` `book_economic_scoreboard`
  no longer feeds the **annualized** Sharpe into `probabilistic_sharpe` /
  `min_track_record_length` with a per-day observation count — that inflated
  the z-statistic by ~sqrt(252) (~15.9x) and short-circuited `psr_vs_zero` to
  ~1.0 for any positive-Sharpe book. PSR/MinTRL now route through the new
  unit-safe returns-only API (`quant_fund.reality.psr.psr_from_returns` /
  `min_trl_from_returns`), which computes the per-period SR internally.
  **Diagnostic values change by design (~15.9x z-deflation)**: `psr_vs_zero`
  and `min_trl_days` in scoreboard receipts now follow the correct Bailey &
  López de Prado per-period convention; `min_trl_days` is a period count. The
  annualized Sharpe remains display-only. Receipts already label these fields
  as research diagnostics.
- **A1 F1-class, second site (PROOFCORE W4)**: `validation/multiple_testing.py`
  `TrialLedger.dsr_for` fed the default (annualized, `periods_per_year=252`)
  `sharpe_ratio` into `deflated_sharpe` with a per-period observation count —
  the same ~15.9x z-inflation as scoreboard F1. It now uses the per-period
  convention (`irregular=True`). Diagnostic values change by design.

### Added

- **PROOFCORE W4 reality filter** (`quant_fund/reality/`): unit-safe PSR /
  MinTRL, Deflated Sharpe with effective-trials clustering (Bailey & López de
  Prado 2014), CSCV/PBO (Bailey–Borwein–López de Prado–Zhu combinatorially
  symmetric CV), SPA / White reality-check drivers, BH-FDR over the trial
  ledger (family-split, never pooled), `RealityReport` assembly, and the
  `quant reality` sub-typer (`trial-report`, `ledger-gate`; mounting owned by
  W5). Shared PROOFCORE schemas live in `quant_fund/proofcore/contracts.py`.
- `metrics.returns.sharpe_ratio_batch`: vectorized canonical Sharpe; the
  ad-hoc Sharpe copies in `paper/sim_live.py` and `hedge_lab/scoreboard.py`
  (bootstrap) now delegate to `metrics/returns.py` (A2 F4). The scoring.py
  IC-based information ratio is re-exported as `ic_information_ratio`.

### Changed

- README opens on dipcatcher: purpose, badges, architecture diagram, and a
  quick start whose default path is labeled synthetic research.
- fx-1 is documented as the in-tree package (corpus, eval, training
  manifests). No trained checkpoint ships in the repository.

### Fixed

- `quant_fund.__version__` follows `fx1.__version__` instead of a second
  literal (`1.0.0` against distribution `0.4.0`).

## [0.4.0] - 2026-09-26

Package version `0.4.0` was set on 2026-09-25 in `src/fx1/__init__.py`.
Entries below are recent conventional commits that landed on that version.

### Added

- Hugging Face OHLCV-1m US minute-bar source (`feat(data)`, #87).
- fx-1 package: corpus builder, eval bank, training-run manifests, model
  cards, SBOM, doctor, and `fx1 --version`.
- Dip Quality bench receipt over 11 crypto series
  (`receipts/dip_bench_crypto_1d_20260925.json`).
- Blocking honesty-inheritance test and corpus-contract smoke in the fx1
  workflow.
- `make fx1-gate` for local parity with that workflow.

### Changed

- Distribution version is read from `fx1.__version__`. Source distributions
  are limited to the buildable tree.
- Coverage floor raised from 70 to 80 (`fail_under` in `pyproject.toml`).
- `warn_return_any` is enforced on `src/quant_fund` and `src/fx1`.
- `make sync` installs all extras, matching CI.

### Fixed

- Worktree fingerprint no longer treats Git LFS pointer drift as a dirty
  tree on Python 3.13 CI (#88).
- Hosted-backend `urlopen` annotated for the Bandit gate.
