# AGENTS.md — working in this repo

## What this is

**fx-1** is the product: a quant research LLM fine-tuned from Kimi K3 open
weights, served as two models, **fx1** and **fx1-lite**
([artificialhedge.co](https://artificialhedge.co)). **dipcatcher** is the
state-of-the-art harness that orchestrates them from a chat-first,
Codex-style CLI: plain text is a conversation with the active model, and
`/superpower` plans and runs the research families that fit the goal (design
and current status: `docs/DIP_CONCIERGE.md`). Bare `dipcatcher` in a terminal
opens the chat via the outer launcher `src/dipcatcher_cli`; `quant` is the lab
CLI without it, and `fxi` stays compatible.
Underneath, `src/quant_fund` is the data engine, evaluation, and verification
that builds and gates fx-1 (`src/fx1`).

## Setup

```bash
make sync          # uv sync --frozen --all-groups --all-extras (uv.lock is
                   # authoritative; --frozen must pass — regenerate with
                   # `uv lock` after touching [project]). Includes the torch
                   # `nn` extra, matching the CI test job.
```

`MOONSHOT_API_KEY` is needed only for hosted eval (`make fx1-eval`); see
`.env.example` for all env vars.

## Gates (run before committing)

| Gate | Command |
|---|---|
| Lint | `make lint` (ruff check + format --check on `src`/`tests`) |
| Types | `make typecheck` (mypy `src/quant_fund`); fx1: `uv run mypy src/fx1` |
| Lab tests | `make test` (PR gate: not network, not slow, xdist). `make test-full` includes slow. |
| fx1 suite | `make fx1-test` — separate lane, ~200 tests |
| fx1 full gate | `make fx1-gate` — lint + types + tests + honesty + corpus smoke |
| Docs vs code | `make docs-consistency` — release/deploy path and honesty checks |
| Release dry-run | `make release-dry-run` — version/changelog, wheel, SBOM, checksums, container health (no tag/sign/publish) |
| CI parity | `.github/workflows/ci.yml` + `fx1.yml` use `uv sync --frozen`. A `v*.*.*` tag signs only after same-SHA green CI (incl. `evidence-audit`) plus readiness gates (`scripts/require_release_gates.py`, `scripts/verify_release_readiness.py`, `release.yml`). |

## Honesty contract (hard rules — do not weaken)

1. Research results are **proper scores** (pinball, CRPS, PIT, QLIKE, Brier,
   ECE, Kupiec, HMM likelihood). Never headline Sharpe/Sortino/Calmar/P&L/NAV.
   `FORBIDDEN_RESEARCH_METRIC_KEYS` in `quant_fund.research.catalog` is
   mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` — change them together;
   `tests/fx1/test_honesty_inheritance.py` blocks drift.
2. SYNTHETIC results are correctness tests, always labeled, never market
   evidence.
3. No live-trading claims — no broker connectivity; see
   `docs/INSTITUTIONAL_READINESS.md` for the five minimum-evidence conditions.
4. Receipts are immutable evidence (`receipts/`, `verify-research`). Every
   research claim should be reproducible from a receipt hash.

## Conventions

- **Commits land directly on `main`** for fx-1 lanes; cross-cutting repo
  changes go through PRs. Match the existing terse conventional-commit style.
- `data/fx1/`, `data/*` derived dirs, `mlflow.db`, coverage files are
  gitignored — regenerate, never commit.
- `tests/fx1` is excluded from the default pytest testpaths on purpose; run it
  via `make fx1-test`. (The `tests/tests` mirror was dropped; if a stray
  reappears, keep it out of default collection.)
- Verifier acceptance history lives in `verifier/vN/`; run receipts in
  `verifier/runs/`. `INFLIGHT` / `day_grind_progress.md` / `docs/DATA_CONTRACTS.md`
  are the lab's work-tracking files — read them before grabbing work.
- `fx1_seed_corpus.jsonl` and the root docs (`APPLY.md`, `MATH_SPEC.md`,
  `RESEARCH_REFERENCES.md`) are tracked inputs/notes — don't delete blindly.

## Remote fleet (Windows box `D:\dipcatcher`)

The SOTA/eval fleet runs on a dedicated Windows host; conventions below are
load-bearing — jobs break silently otherwise.

- **PowerShell-only.** Remote ops scripts are `.ps1`; bash heredocs do not
  exist there. Quote `cmd /c` inner strings with backtick-escaped quotes.
- **WMI spawn, not `Start-Process`.** Jobs are created via
  `Invoke-CimMethod -ClassName Win32_Process -MethodName Create` so they are
  owned by WMI and survive ssh session teardown. Always wrap the payload in
  `cmd /c "... 1> stdout.log 2> stderr.log"` — `Win32_Process` has no built-in
  redirection.
- **Parametrized launcher.** New fleet work goes through
  `scripts/fleet_spawn.ps1 -Manifest jobs.json` (one JSON job list, one
  spawn path) plus `scripts/fleet_watchdog.ps1` (heartbeat file at
  `.dsh-24x7\fleet_heartbeat.json`, auto-respawn bounded by `-MaxRespawns`).
  `scripts/fleet_manifest_sota.ps1` regenerates the canonical SOTA manifest.
  The legacy `spawn_*.ps1` one-offs stay for reference; prefer the launcher.
- **Durable paths.** Long-running artifacts live under `.dsh-24x7\` (gitignored,
  durable remotely); model weights under `data\models\`; bars under
  `data\raw\sources\`. Do not write fleet state into temp dirs. **Caveat: 911
  files under `.dsh-24x7/` were committed before the ignore rule existed and are
  still tracked.** `.gitignore` stops *re-adding* scratch; it does not untrack
  what is already in the index. Untracking is an explicit
  `git rm -r --cached .dsh-24x7` bundled into the history-reconciliation commit
  (`docs/ULTRA_PROD_READINESS.md`, Phase 2) — do not do it piecemeal.
- **Thread pinning.** Jobs set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false` in their `env` block — the manifest schema
  carries this; keep it when authoring new jobs.
- **Defender exclusions.** The remote host needs `D:\dipcatcher` excluded
  from real-time scanning or parquet/bar reads get throttled mid-fleet.

## Key docs

`docs/FX1.md` (model), `docs/FX1_TRAINING.md` (compute ladder + gates),
`docs/FX1_API_STABILITY.md` (versioning — `fx1.__version__` is canonical
semver; pyproject reads it via hatch dynamic version),
`docs/FX1_DATA.md`, `docs/FX1_DATASOURCES.md`, `docs/FX1_ARCHITECTURE.md`,
`docs/OPERATIONS_RUNBOOK.md`.
