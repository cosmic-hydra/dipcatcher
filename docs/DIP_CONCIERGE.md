# dipcatcher CLI — chat-first orchestration for fx1 and fx1-lite

dipcatcher is Artificial Hedge's state-of-the-art harness for orchestrating
its **fx1** and **fx1-lite** models ([artificialhedge.co](https://artificialhedge.co))
from the command line. The CLI is designed like Codex CLI: you open it, you
talk to it, and it does the work. Plain text is a conversation with the
active model; everything else is a slash command. `/superpower` is how the
conversation reaches dipcatcher's research families — it picks the
capabilities that fit your goal, runs them, and explains what it used and
why.

> **Status (2026-10-08, working tree).** The packaged `dipcatcher` launcher
> now opens chat when stdin and stdout are terminals. `dipcatcher chat
> --model fx1-lite` chooses the starting model; `fxi` remains compatible.
> Redirected bare invocation prints help, and explicit lab commands retain
> their existing behavior. Ordinary chat and `/superpower` synthesis use the
> honesty validator; refined plans enforce the same command cap as the
> deterministic planner. Qualified family selection, durable jobs, full
> terminal UX, and production acceptance remain open in
> [the product contract](DIPCATCHER_PRODUCT_CONTRACT.md). These changes are
> not yet a published release or proof of live model availability.

## Design: Codex-style, chat-first

About 99% of the interface is the conversation. The rules:

| Principle | What it means here |
|---|---|
| One command, one conversation | Bare `dipcatcher` opens the chat. There is no subcommand tree to learn first. |
| Plain text is a turn | Ask in your own words; the active model answers and calls tools (memory, web, harness) as needed. |
| Slash commands for control | `/superpower`, `/research`, `/use`, `/status`, `/cancel` — discoverable with `/help`, never required to get started. |
| Approval before consequence | Consequential lab work (`train`, `optimize`, `paper`) asks first; a refusal is recorded, not retried. |
| Long work runs in the background | Jobs stream progress while you keep typing; `/cancel` and `/redirect` take effect at the next step. |
| Honest by default | Ordinary chat and `/superpower` synthesis pass the existing honesty validator before display. This lexical check does not establish semantic correctness or verify a cited receipt. |
| Subcommands are the other 1% | `dipcatcher research --config …` and the rest stay for scripts, CI, and receipts — the non-interactive path. |

New operator-facing features belong in the conversation (a slash command or
a model tool in `src/fx1/interactive`), not as new top-level subcommands.

**Boundary rule.** The chat lives on the fx1 side (`src/fx1/interactive`),
and `quant_fund` never imports `fx1` (enforced by
`tests/unit/test_fx1_dependency_edge.py` and `configs/arch_boundaries.toml`).
The outer launcher is `src/dipcatcher_cli`, outside both libraries. The chat
reaches the lab exclusively through the registered
harness commands in `fx1.harness.HARNESS_REGISTRY`, executed as
subprocesses — the model can never touch an unregistered lab surface.

## Quickstart

```sh
uv sync --frozen --all-groups --all-extras
dipcatcher              # opens chat in an interactive terminal
# Inside chat: /keys set fx1 prompts for a masked API key and endpoint URL.
# Use the completion URL supplied by your service; no endpoint is assumed live.
# Inside chat: /help, /superpower help, /use fx1-lite, /exit
# Or select the starting model explicitly:
dipcatcher chat --model fx1-lite
```

Without an endpoint the console still works: slash commands run, web research
runs, flash context runs — only model turns (chat replies, plan refinement,
synthesis) report honestly that no endpoint is configured and fall back to a
deterministic summary.

## The session

Plain text is a chat turn with the active model (`fx1` or `fx1-lite`, switch
with `/use`). The model gets four tools, executed locally with the lab's
fail-closed posture:

| Tool | What it does | Gate |
|---|---|---|
| `flash_retrieve` | ranks persistent research memory against the query | none (local read) |
| `web_lookup` | one bounded search + up to two page reads | fetch policy |
| `flash_save` | persists a finding with task/uncertainty | none (local write) |
| `harness_run` | runs one registered `dipcatcher` command | approval when consequential |

Slash commands:

```text
/superpower <goal> [--sync]     plan + coordinate research capabilities
/superpower help                list the capability catalog
/research <goal> [--budget-s N] [--max-sources N] [--queries a|b] [--sync]
/status                         background job progress
/cancel                         stop the background job at its next step
/redirect <new goal>            re-target the running research
/flash add|list|show|remove|correct|search|refresh ...
/keys list|set|remove           model endpoints (presence-only)
/harness list|describe|run      registered lab commands
/live                           operator card for the live desk (chat does not transmit)
/trade [readiness|help]         live orders are disabled; paper is the only book
/use <model>  /model  /help  /exit
```

Background jobs run on their own thread; the console polls stdin between
lines, so progress renders while you keep typing, and `/cancel` /
`/redirect` take effect at the job's next step boundary. On piped stdin the
loop degrades to blocking reads (jobs still stream progress).

On Windows the console uses blocking input because console handles are not
supported by `select`; queued background progress appears between submitted
lines. Full cross-platform editing and live progress remain acceptance work.

## /trade — paper book, live orders refused

`/trade` and `/trade readiness` print the five institutional conditions.
Every condition is blocked or not yet collected: live orders stay disabled,
and the report cannot be flipped from inside the process. The shell command
that actually runs a book is simulated:

```sh
dipcatcher trade readiness
dipcatcher trade session --state ./paper-session.json --commands ./orders.jsonl
```

`orders.jsonl` is one command per line: `mark`, `submit`, `fill`, `cancel`,
`kill`, `clear_kill`, `checkpoint`. Prices are caller-supplied. Short sales
are refused. A live endpoint is not a flag on this command. This session is
not a broker connection and not evidence for a live-trading claim.

## /superpower — capability orchestration

`/superpower <goal>` answers "which of dipcatcher's research capabilities
should run for this goal, and why?" The catalog is every registered harness
command — `research` (the scientific benches that make up the research
families; see [RESEARCH_CENTRE.md](RESEARCH_CENTRE.md)), plus `validate`,
`forecast`, `backtest`, `northset`, `kyle-ofi`, and the other lab surfaces —
together with web research, flash retrieve/save, and synthesis.
`/superpower help` prints the live list. It runs in four phases
(`fx1.interactive.superpower`):

1. **Plan.** The deterministic planner scores every capability against the
   goal (trigger-term catalog, data not code) and selects the relevant
   subset — never a blind run-everything. With a model endpoint available,
   the hosted model may refine the selection, but its reply is validated
   against the registry: unknown ids are dropped and any malformed or failed
   model pass falls back to the deterministic plan. Both paths apply
   `max_harness=4`; refinement deduplicates IDs and preserves retrieval first
   and synthesis last. The model may select another registered command that
   the keyword pass missed, within that same cap. Consequential commands
   still stop at the approval gate. This is command-level planning; suitable
   research-family selection remains future work.
2. **Recall.** Flash context is retrieved first (cheap, local) so prior
   findings feed everything downstream.
3. **Execute.** Web research (budgeted, cancellable) runs before harness
   commands. Consequential harness commands — `train`, `optimize`, `paper`
   (the `MODEL_TRAINING` role) — require explicit operator approval and are
   recorded `approval-denied` when refused. Harness output keeps its
   `DATA_LABEL=SYNTHETIC` honesty markers into the synthesis context.
4. **Synthesize.** The final answer is composed from capability results only
   (sources and receipt paths cited), then passes
   `fx1.honesty.validate_fx1_output`: forbidden headline metrics
   (Sharpe/Sortino/Calmar/P&L/NAV), live-performance claims, or unlabeled
   synthetic evidence cause the answer to be **withheld** with the gate's
   reason — violating text is never emitted. The run itself is saved to
   flash context with its goal and capability list.

The plan is printed with a per-capability rationale before anything runs,
and every result carries an explain block (`capabilities used and why`).

## Deep web research

`fx1.webresearch` is a stdlib-only research engine — no API keys, no new
dependencies:

- **Search** (`fx1.webresearch.search`): keyless DuckDuckGo (lite endpoint,
  html fallback). Failures report an error and yield zero hits; hits are
  never fabricated. Queries derive from the user's goal text only — private
  flash context never enters a search query.
- **Fetch** (`fx1.webresearch.fetch`): http/https only; credentials in URLs
  refused; private/loopback/link-local/reserved addresses refused **including
  on redirects** (no SSRF lane into your network); size caps with explicit
  truncation; redirect caps; robots.txt honored (conservative subset,
  per-origin cache); HTML reduced to title/text/links with script, style, and
  site-chrome filtering.
- **Iterate** (`fx1.webresearch.engine`): seed queries include a
  deliberately adversarial angle ("criticism OR contradictions") so
  conflicting evidence is searched for, not avoided. The loop searches,
  ranks hits by goal overlap with domain diversity, fetches, extracts
  goal-relevant sentences as claims, and follows ranked in-page references
  within the depth budget.
- **Verify**: claims are clustered across sources by token overlap; clusters
  report `corroborated` (≥2 sources), `single-source`, or `contested`
  (negation or antonym polarity detected across different sources, listed as
  explicit CONFLICT pairs).

Every report is labeled **HEURISTIC WEB VERIFICATION** — cross-source checks
of public pages, never a sealed dipcatcher receipt.

Budgets are hard and checked mid-loop: wall-clock (`--budget-s`), source
count (`--max-sources`), searches, per-source bytes, reference depth, and an
inter-search pause. Cancellation is cooperative and checked between network
steps; partial results are kept and the report is stamped `(cancelled)`.

## Flash context — persistent research memory

`fx1.flash` stores findings as append-only JSONL under
`$FX1_CONFIG_DIR/flash/entries.jsonl` (default `~/.fx1/flash/`), shared by
every `fxi` process — memory survives sessions.

Each entry (`fx1.flash.store.FlashEntry`) carries: text, task, tags, sources
(URL/title/accessed-at), created/updated timestamps, uncertainty
(`low|medium|high`), conflicting evidence, verified and private flags, an
optional staleness window, refresh state, and usage counts.

- **Retrieve** (`fx1.flash.retrieve`): deterministic hashed-vector cosine
  over text/tags/task, tempered by recency (30-day half-life) and staleness;
  every hit explains itself (`term-match`, `tag-match`, `stale`, `private`,
  ...). Entries below the relevance threshold are left behind instead of
  filling the model's context.
- **Correct/remove**: corrections append a revision and tombstone the old
  line — history is never rewritten; removal tombstones. Torn trailing lines
  are reported via `read_errors()`, never silently dropped.
- **Refresh** (`fx1.flash.refresh`): re-fetches recorded sources and stamps
  reachability honestly; unreachable sources are named in the refresh note so
  retrieval can demote stale findings.
- **Private**: entries flagged `--private` render with a do-not-share marker
  in prompt context; the approval gate covers private-data sharing.

```text
/flash add "Fed held rates in Sept 2026; dot plot shows one cut" \
    --task rates --tags macro,fed --source https://... --uncertainty medium
/flash search fed rate path
/flash correct <id> --uncertainty high --conflict "two members dissented"
/flash refresh <id>
/flash remove <id>
```

## Approvals and safety model

- Approvals default to **deny** whenever no interactive operator can answer
  (piped stdin, scripts, tests); the console prompts `[y/N]` and anything but
  an explicit yes is a refusal. Every decision is recorded
  (`fx1.interactive.approvals.ApprovalGate`).
- Consequential actions: harness `MODEL_TRAINING` commands (`train`,
  `optimize`, `paper`) in chat tools and in `/superpower` plans.
- Private-data sharing: private flash entries are marked in prompt context;
  web-search queries are built from goal text only.
- Credentials: presence-only reporting (`...wxyz` fingerprints), keys travel
  as environment values, never argv; store file mode 0600.

## One-shot mode (scripting)

```sh
fxi superpower "assess conformal coverage regressions" --sync
fxi research "ECB rate path expectations" --budget-s 120 --max-sources 5 --sync
fxi flash add "finding" --task t --tags a,b --uncertainty low
fxi flash search "ecb rates"
fxi flash refresh <id>
```

`--sync` runs inline (no background thread), which is what CI and pipes want.

## Tests and verification

Offline suites (network seams injected; run with `make fx1-test` or
`uv run pytest tests/fx1/test_flash.py tests/fx1/test_webresearch.py
tests/fx1/test_superpower.py tests/fx1/test_concierge.py
tests/fx1/test_interactive.py`):

- `tests/fx1/test_flash.py` — CRUD, tombstones, torn-line reporting, threaded
  appends, relevance ranking, staleness, privacy flags, context blocks,
  refresh, cross-process persistence.
- `tests/fx1/test_webresearch.py` — URL/SSRF policy, redirect re-checks,
  size caps, robots subset + caching, DDG parsing against the live HTML
  shape, claim extraction filters, contradiction detection, budgets
  (time/search/source), cancel with partial results, redirect, fragment
  dedupe, IRI encoding.
- `tests/fx1/test_superpower.py` — registry/harness parity, consequential
  set, planner selection + caps, model-refinement validation and fallbacks,
  coordination order, approval denial, honesty-gate withholding, offline
  summary, cancel semantics.
- `tests/fx1/test_concierge.py` — slash dispatch, chat tool loop (flash
  retrieve/save, harness approval + registry refusal, backend fault
  rollback), background research lifecycle (status/cancel/redirect/error),
  superpower sync/background, flash commands, console driver over scripted
  stdin.

Live verification (performed against the real internet during development):
`fxi research --sync` fetched real sources (DDG lite), extracted real claims
with URLs, and deduped fragments; `fxi superpower --sync` planned, ran live
web research, saved to flash, and produced the offline synthesis; a separate
`fxi flash list/search` process retrieved the saved entry — cross-session
memory confirmed; a full piped REPL session (help → flash add/search/list →
plain chat → exit) ran end-to-end, with the no-endpoint chat turn answering
honestly instead of fabricating a model reply. Live *model* turns were not
verifiable with the stored credentials — see Known limitations.

## Known limitations (honest gaps)

- **Search provider**: keyless DDG HTML parsing breaks if DDG changes markup;
  the failure mode is an explicit error plus zero hits, never fabricated
  results. No paid-SERP integration yet.
- **No JS rendering**: SPA-only pages yield little text.
- **Heuristic claims**: sentence-level extraction and token-overlap grouping
  are lexical, not semantic; mirrors/syndication of one article count as
  multiple corroborating sources (fragment URLs are deduped, distinct hosts
  are not).
- **Cost budgets**: time/search/source/byte limits are enforced; *token*
  spend is tracked by the backend usage tracker but not yet budget-capped.
- **Background jobs** live in-process; they do not survive a console restart
  (flash entries and reports written before the restart do).
- **Retrieval** is hashed bag-of-words — no embeddings, so paraphrased queries
  with no shared vocabulary miss.
- **Contested detection** catches explicit negation/antonymy only; subtle
  numeric disagreement is shown as separate claims, not a conflict.
- **Hosted model turns**: chat replies, model plan refinement, and synthesis
  need a working endpoint. At the time of writing the operator's stored
  endpoint (`inference.artificialhedge.co`) rejected its stored key
  (`invalid_api_key`) and the stored base URL lacks the
  `/v1/chat/completions` path, so live model turns were **not** verifiable —
  they are covered offline by scripted-backend tests on the repo's own
  `HostedK3Backend` wire client. Fix with `fxi keys set fx1` (full chat URL +
  valid key); every non-model capability (web research, flash, planning,
  coordination, approvals) was verified live end-to-end.
