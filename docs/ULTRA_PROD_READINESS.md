# ULTRA PLAN — Production readiness for dipcatcher / fx-1

**Status:** draft for owner ruling · **Created:** 2026-10-08 · **Measured at:** local `HEAD f70c140b5e`, `origin/main 3b246fffa5`, macOS 27.0.0
**Scope:** what must be true before this repo can be called production-ready, in what order, with what gate.
**Method:** every claim below is measured from this repo at the SHA above. Commands are reproducible. Where a claim comes from a self-report rather than a measurement, it is labelled *(self-reported)*.

---

## 0. Verdict

dipcatcher is **not far from production. It is far from *verified*.**

The engineering is genuinely strong — 245k LOC across 13.4k modules, a 116k-LOC test suite, `mypy --strict` total, sealed receipts, a real honesty contract. The problem is not capability. It is that **nothing has been verified in 400 consecutive CI attempts**, and one sealed evidence artifact is parameterized in a way that can never become valid forward evidence.

Three findings decide the readiness question:

| # | Finding | Severity | Status |
|---|---|---|---|
| **F1** | **CI has produced zero successful runs in 400 consecutive attempts.** Last 100: 85 `cancelled`, 15 `queued`, 0 success. | **Blocker** | Config fix applied 2026-10-08; needs a green run to confirm |
| **F2** | The forward prereg's freeze was **self-attested** — no external timestamp. *(An earlier draft called this a backdated pre-registration; that was wrong — the rule is freeze-relative and sound. See §2.2.)* | **High** | **FIXED 2026-10-08** — RFC 3161 anchor, `chain_verified` + `fresh` |
| **F3** | Local `HEAD` holds **314 commits absent from `origin/main`**; `origin/main` holds **14 absent from local**. Receipts bind git revision. | **Critical** | Open — needs owner ruling on the spine |
| **F4** | `.github/workflows/` epoch pins were **already stale at HEAD** on `ci.yml` and `simtest.yml` — independent of any new edit. Epoch verification was failing on `origin/main` before this plan. | **High** | Found 2026-10-08; re-stamp pending |

Everything in Phases 0–2 is about making the rest of the repo *falsifiable*. Until F1 is fixed, every other claim in this repo — the 81% coverage floor, strict typing on 668 modules, receipt verification, the SBOM — is asserted but unproven, because the machine that would refute them is dark.

**The honest ceiling:** dipcatcher can become production-ready as a **research and evaluation platform for internal analysts**. It cannot become production-ready for **live trading** — that is blocked on procurement and authorization, not on code, per `docs/INSTITUTIONAL_READINESS.md`. This plan is explicit about which one it is achieving.

---

## 1. What "production-ready" means here

Three tiers. Mixing them is how a research repo accidentally implies it trades.

| Tier | Definition | Reachable by code? | Target |
|---|---|---|---|
| **A — Research platform** | Internal analysts can trust a receipt: green gates, reproducible from one SHA, docs match code, no live claims | **Yes** | Phase 0–5, ~2–3 weeks |
| **B — Distributed platform** | A + signed releases, third-party security review, install path, onboarding | **Mostly** | Phase 6–7, ~4–6 weeks |
| **C — Live trading** | Authorized vendor PIT data, authenticated broker reconciliation, venue cost measurements, non-synthetic forward record, signed promotion receipt | **No** | Blocked: procurement + authorization |

This plan delivers **Tier A completely** and **Tier B substantially**. It does not attempt Tier C, and Phase 4 exists only to keep Tier C honest while it stays blocked.

---

## 2. Measured evidence

All measured at `f70c140b5e`, 2026-10-08.

### 2.1 CI is dark — F1

```
gh run list --limit 400 --json conclusion,status,createdAt,name,headSha
  → total runs inspected: 400
  → success count: 0
  → statuses: completed 45 / queued 15   (last 60)
  → conclusions: cancelled 85 / (none) 15  (last 100)
```

The latest push (`2026-10-08T07:52:45Z`, "restore audit census entries…") shows **all 12 inspected workflows `queued`, still queued 2h33m later.**

**Mechanism — four compounding causes, now quantified against GitHub's published limits:**

GitHub-hosted runner concurrency caps ([docs.github.com/en/actions/reference/limits](https://docs.github.com/en/actions/reference/limits)):

| Runner type | Plan | Max total concurrent jobs | Max concurrent **macOS** jobs |
|---|---|---|---|
| Standard GitHub-hosted | Free | **20** | **5** |
| Standard GitHub-hosted | Pro | 40 | 5 |
| Standard GitHub-hosted | Team | 60 | 5 |
| Standard GitHub-hosted | Enterprise | 500 | 50 |

*(GitHub Support can raise the job-concurrency limit on ticket — an available lever, see step 6.)*

1. **Job volume per push.** `matrix.yml` alone declares `os: [ubuntu, macos, windows] × python: ["3.12","3.13","3.14"] × shard: 4` = **36 jobs** (45-min timeout each). `ci.yml` declares **24 jobs** (`test` alone allows 75 min). Twenty-one further workflows also trigger on push. Total ≈ **80+ jobs per push against a 20-job ceiling** — a minimum of four full waves to drain.
2. **The macOS lane alone cannot fit.** `matrix.yml`'s macOS slice is `3 python × 4 shards = 12 jobs` competing for a **5-job macOS cap** at up to 45 min each. Even with an otherwise idle repo, the matrix needs **≥3 sequential waves ≈ 2h15m** just to clear macOS. This alone explains the 2h33m stall.
3. **Push cadence.** The fleet merges PRs in bursts — `origin/main` log shows **12 merge commits inside 68 seconds** (`16:39:39 → 16:40:50Z`), plus dozens of standalone pushes per day.
4. **`cancel-in-progress: true` on 21 of 22 workflows**, keyed `${{ github.workflow }}-${{ github.ref }}`. Every new push cancels the run already in flight, so a run can never reach a verdict under continuous pushes. Runs that lose the race sit `queued` against the 20-job ceiling, which is the 2h33m stall.

**This is not a flaky test problem.** The suite is small: `.test_durations` holds 8,072 entries totalling **1,325 s ≈ 22 min serial**, so 4 shards ≈ 6 min. The suite is cheap. The orchestration is what's broken.

### 2.2 The forward-record freeze was self-attested — **FIXED 2026-10-08**

> **Correction to an earlier draft of this plan.** This section originally read as a *backdated pre-registration* and rated the finding a **Blocker**. That framing was **wrong**, and it is withdrawn. Verification against the repo's own tooling produced a materially different and milder finding. The evidence is below.

Measured state of `receipts/forward_record_preregistration_v1.json`:

```
embedded self-seal : valid=True   (b62151e486c97d23…)   ← untampered
sidecar seal       : valid=True   (d1f0022744261fbc…)   ← untampered
declaration_date   : 2026-10-07
status             : NOT_YET_COLLECTED
window             : {label: forward_2026H2, start: 2026-07-01, end: null}
splits.forward     : "first accepted decision must occur on a LATER market date
                      than the recorded freeze"
splits.warmup      : all historical bars on/before 2026-09-18 (warmup only, already inspected)
```

**The admissibility rule is freeze-relative and sound.** `splits.forward` gates the first eligible decision on *the recorded freeze* (2026-10-07), not on `window.start`. So the receipt is **not** a pre-registration written after peeking, and the earlier "spoiled window" reading does not hold.

**What `window.start = 2026-07-01` actually is:** the calendar span of the named experiment (`forward_2026**H2**` = second half of 2026). It is a *label*, not a collection start. The real defect is **label ambiguity** — a reader can easily take `window.start` as "forward collection began 2026-07-01", which would be false. That is a documentation hazard, not an integrity breach.

**The genuine defect was this: the freeze was self-attested.** `anchors.json` covered only `quality/checkpoint.json`, `quality/crown_jewels.json`, and `quality/epoch_heads.json` — **not** the prereg. `prereg_seal.py`'s own docstring concedes it: *"Both are local tamper-evidence … as long as the seal is not itself rewritten by whoever controls all local files. Independent timestamping/anchoring of the seal is an operational step; **the code does not assert it occurred**."*

So as of 2026-10-07 the prereg's freeze date was a claim inside a file that its own author could have rewritten together with its seal. For the single artifact gating institutional condition 3, that is a real weakness — and it was cheaply fixable with machinery the repo already ships.

**Remediation applied 2026-10-08 (Phase 4, step 1):**

```bash
uv run dipcatcher anchor-timestamp \
  --file receipts/forward_record_preregistration_v1.json.seal.json
# → timestamp=quality/timestamps/receipts__forward_record_preregistration_v1.json.seal.json.tsr
```

Verified with `verify_timestamps('.')`:

```json
{ "ok": true, "anchored": true,
  "fresh": { "receipts/forward_record_preregistration_v1.json.seal.json": true },
  "chain": { "receipts/forward_record_preregistration_v1.json.seal.json": "chain_verified" },
  "errors": [] }
```

The freeze is now committed to by an external TSA signature, chain-verified against the repo's pinned FreeTSA CA and signer certs, and `fresh` against current bytes. **A history rewriter can no longer backdate it.** (Only the file's sha256 was posted to the TSA — never its contents.)

Remaining Phase-4 work: start prospective collection, resolve the `window.start` label ambiguity in prose, and repoint the three docs (§Phase 4).

### 2.3 History has forked — F3

```
git rev-list --count HEAD                                  → 4059
git rev-list --count origin/main..HEAD                     → 314    (local-only)
git rev-list --count HEAD..origin/main                     → 14     (origin-only)
git rev-list --left-right --count origin/main...HEAD       → 14 / 314
```

The divergence is bidirectionally real and large. This matters more than a normal sync problem because **receipts bind git revision** (`receipts/` pins `script_sha256`, `inputs_sha256`, and revision). The repo also has `epoch-consistency` as a PR gate, which by construction requires every epoch chain to extend the base-branch head — a fork will either fail that gate or, worse, force re-sealing of pinned chains (the honesty report already records "re-seal" commits as seal drift).

### 2.4 Operational scratch is tracked in the tree

`AGENTS.md` states `.dsh-24x7\` is "gitignored locally". It is not. `.gitignore` covers only two subpaths:

```
87:.dsh-24x7/eval-full/**
88:.dsh-24x7/paper_data/**
```

Tracked-file census (`git ls-files`, 22,280 files total):

| Path | Tracked | Verdict |
|---|---|---|
| `.dsh-24x7/` | **911** (128 MB) | Genuine scratch: `lane-simlive` 206, `eval-shards` 89, `native` 66, ~14 `lane-*` dirs × 25, `mega-arena` 16. Fleet run dumps and one-off probes (`fix_clobbers.py`, `probe_drift.py`, `win_fix2.ps1`). **Untrack.** |
| `.box-soft-verify/` | 18 | Tool state. **Untrack.** |
| `.cursor/`, `.serena/`, `.freebuff/` | ~7 | Editor/agent state. **Untrack.** |
| `.github/workflows/*.json` | 32 | **CORRECTION — not litter. Legitimate epoch evidence.** See below. |
| `tests/tests/` | 0 `.py` | **CORRECTION — not a stray mirror, not a problem.** See below. |

**Correction 1 — the 32 `corpus_epoch_*.json` in `.github/workflows/` are integrity machinery, not stray files.** An earlier draft of this plan called them inert litter to be moved out of the trust surface. That was wrong. `scripts/verify_epoch_chain.py:60` declares `EXEMPT_PREFIXES_BY_CORPUS = {..., ".github": frozenset({"workflows/"})}` — `.github/workflows` is *its own corpus*, exempted so it does not double-chain under `.github`. Each record is `{"kind": "corpus_epoch.v1", "data_label": "CORPUS", "live_pnl_claim": false, "members": [{"name": "ci.yml", "sha256": "e82f32…"}, …]}`.

**Operational consequence, and it is load-bearing: editing any `.yml` under `.github/workflows/` invalidates its pinned member sha256.** Any workflow change must be followed by `make stamp-epochs` → `make sign-pins` → `make anchor-pins`, in that order, before `make epoch-consistency` can pass. Do not move or delete these files. This was caught while implementing Phase 0 — the workflow edits *will* leave the tree failing epoch verification until re-stamped, and that is the intended handoff state, not a regression.

**Correction 2 — `tests/tests/` is a known sftp-clobber artifact, already handled.** `.gitignore:96-103` documents the root cause: the off-machine `put -r src /D:/dipcatcher/src` re-creates nested mirrors under `src`, `tests`, and `scripts` (directory-inside-itself clobbers), and all three are already ignored (`/tests/tests/`). `pyproject.toml:358` `testpaths` correctly excludes it. On disk it is four empty directories. **Nothing to fix** — and deleting it is pointless because the clobber recreates it.

Also: **231 local branches**, many stale `devin/*`, `kimi/*`, `codex/*` agent branches.

### 2.5 The research catalog is 99.8% optional, 74% already retired

```
REQUIRED_BENCHMARK_FAMILIES       : 23
OPTIONAL_BENCHMARK_FAMILIES       : 10,222
RETIRED_BENCHMARK_FAMILIES        :  7,529
LIVE_OPTIONAL (= OPTIONAL − RETIRED) : 2,693
```

The retirement machinery is excellent and should be said so: `retired_families.py` is audit-derived rather than hand-written, `research/agent.py` carries a runtime de-emission filter, and `tests/unit/research/test_catalog_retired_families.py` guards the behavior — including that an *archived* receipt naming a retired family still verifies, and that an unknown family still fails closed.

The residue is the issue. **2,693 "live optional" families is not a defensible default research surface.** The recent `INFLIGHT` entries (w1665–w1714) are almost entirely template-generated families named after world mythology — `zeus_qa_studies`, `anansi_qa_studies`, `loki_qa_studies`. That is catalog bulk masquerading as research breadth. The work log itself has been colonized by it: `INFLIGHT` is 4,154 lines and its newest 50 entries are family-name additions.

### 2.6 fx-1 has no *release* weights — but it does ship a fixture checkpoint

> **Correction to an earlier draft of this plan.** An earlier version of this section asserted flatly that fx-1 has no weights on disk. That was **wrong**, and it was caught during implementation. The corrected position is narrower and more precise.

- `src/fx1/__init__.py:16` — `__version__ = "0.4.0"`, but **`git tag -l 'v*' | wc -l` → `0`**, so no release has ever been cut and the version names an API contract that was never tagged. (`git tag | wc -l` → **4**, but all four are `attic/receipt-provenance/*` anchors dated 2026-09-27/28, not releases. `release.yml` is keyed on `v*.*.*` and has never executed. These 4 tags appeared *during* this audit — the tree is moving under us.)
- **A tracked fixture checkpoint does exist**: `artifacts/fx1_tiny_lm/weights.safetensors` (127 KB) plus `modelcard.json` and `weights.manifest.json`, all three tracked in git. Its own card is unambiguous:

  > `known_limits`: *"fixture-scale byte-level LM (~31k params) trained for seconds on CPU over fx1_seed_corpus.jsonl — exists so the weights-direct path loads a real artifact; **not an fx-1 release candidate**"*
  > *"**eval_delta values are synthetic ship-gate placeholders, not measured evals**"*
  > `license_tier: internal_research`, `research_only: true`, `live_pnl_claim: false`

  So this is a correctness fixture for the weights-direct code path — **not** a model, and explicitly not evidence. **It is worth flagging that its `eval_delta` block contains bare numbers (`domain_pass_rate_base: 0.5 → candidate: 0.51`) that a reader could mistake for measured results.** The card disclaims them twice; the numbers still sit there looking like scores.
- `data/models/` holds third-party **Kronos** checkpoints vendored for a research bench — untracked, not fine-tuned here, not fx-1.
- `LocalFx1Backend.complete()` raises `NotImplementedError`, so local generation is not implemented either.
- *(self-reported, substantively correct)* `HONESTY_RATING.md` M7: "fx-1 is a plan, not a model — `complete()` raises `NotImplementedError`, training manifest gate exists but no training run has occurred", costing 18 of 90. Only the "no weights" wording needs narrowing.

**Net position:** the *substance* of M7 stands — there is no trained, evaluated, released fx-1 model, and `docs/FX1_TRAINING.md` states no training run has been launched from the ladder. The *wording* was too absolute.

The harness side is not in question — `src/fx1` is 323 modules, strict-typed, 79 test files.

### 2.7 Reliability debt in the verification lane

- **`make test` segfaults under `-n auto`** — reproduced 3× in a signal-free clean run (job `bash-308`), different worker each time (`gw3`/`gw9`/`gw2`), always 27–30% in, crashing thread shows `<no Python frame>`. **ROOT CAUSED 2026-10-08 — see below.** `INFLIGHT w1714`'s leading theory (disk/swap pressure) is **refuted**.
- **mccabe ceiling is theater at 74.** `pyproject.toml:173` sets `max-complexity = 74` — 5–7× a conventional ceiling. `INFLIGHT w1714` records 30→14 violations fixed, **14 remaining**, and **736 pre-existing violations not listed** that `check_mccabe_ratchet.py --write` refuses to pin while any ceiling violation stands. A ceiling nobody meets is not a ratchet.
- **Coverage floor 81 is enforced but invisible** *(self-reported deduction H2: unenforced/unvisible metrics cap at 50%)*.

### 2.8 What is already strong — do not regress it

| Asset | Evidence |
|---|---|
| Lint | `ruff check src tests` → **All checks passed**; `ruff format --check` → 19,019 files formatted |
| Types | `mypy --strict` total across both packages *(self-reported: 668 harness modules + 65 fx1)* |
| Honesty contract | `FORBIDDEN_RESEARCH_METRIC_KEYS` (5 keys) mirrored by `fx1.honesty.FORBIDDEN_HEADLINE_TOKENS` (5), drift-blocked by test |
| Packaging | wheel + sdist, `version 0.4.0`, Dockerfile, console scripts, proprietary `LICENSE` v1.0 (2026-09-30) |
| Release supply chain | CycloneDX SBOM + Sigstore + SLSA defined — **never exercised** (0 `v*` release tags) |
| Security | CodeQL, Scorecard, dependency-review, full-history gitleaks, SHA-pinned actions |
| Paper/shadow | Crash-resumable simulated broker, kill switch, **no live capital**, `LiveEndpointRefused` on live endpoints |

---

## 3. The plan

Effort is engineering-days for one owner with the agent fleet. Phases 0–2 are strictly sequential; 3–5 parallelize after Phase 2.

### Phase 0 — Restore the verification signal · **P0 · ~1 day**

Nothing else can be trusted until a gate returns a verdict. Goal: **three consecutive green `ci.yml` runs on `main`.**

1. **Stop cancelling gate runs.** Set `cancel-in-progress: false` on the four gate workflows: `ci.yml`, `fx1.yml`, `matrix.yml`, `proofcore.yml`. Nothing is lost by letting a run finish; everything is lost by never finishing one.
2. **Path-filter the non-gate workflows.** Docs-only pushes must not run the matrix. Add `paths-ignore: ['**/*.md', 'docs/**']` to `docs.yml`, `atlas.yml`, `web_explorer.yml`, `replay_viz.yml`. Move `matrix.yml`, `property-nightly.yml`, `adversarial-properties.yml`, `simtest.yml`, `perf-baseline.yml` to `schedule` + `workflow_dispatch` only — they are cross-platform and long-running by design and should never gate a push.
3. **Two-tier triggers.** `ci.yml` keeps `push: main` with the fast subset (lint, audit, package, unit/property/regression/formal shards, smoke). The full matrix and slow lanes live on a nightly `schedule` + manual dispatch.
4. **Batch the merge flood with a merge queue (the structural fix for push cadence).** `origin/main` shows bot-driven merges (`Merge pull request #29xx from artificial-hedge/devin|kimicode`) landing ~12 per minute with no batching and evidently no required checks — that is the leak that starves every other fix. Enable a merge queue on `main` ([docs](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/managing-a-merge-queue)):
   - add the **`merge_group` event** to the `on:` block of the required-check workflows — without it, queued PRs never report status and the merge fails;
   - set **Build concurrency** low (start at 5) so the queue dispatches a bounded number of `merge_group` builds rather than flooding the 20-job cap;
   - set **Merge limits** (e.g. min 1 / max 10) plus a **status check timeout**, so PRs are validated *as a batch* instead of one push per PR.

   This is strictly better than hand-throttling the fleet: it batches automatically, validates the batch against the real `main` tip, and guarantees `main` is never left in a state whose gates never ran.
5. **Bound the queue.** Add `timeout-minutes` to the non-gate workflows so a starved run cannot occupy a concurrency slot indefinitely. If GitHub Support can raise the repo's 20-job cap ([limit increase](https://docs.github.com/en/actions/reference/limits)), request it — it is free and directly relieves the binding constraint.
6. **Re-measure after one week.** Confirm with `gh run list` that completed-success runs appear and that the macOS lane is no longer the long pole. If it is, reduce `NUM_SHARDS` for the macOS slice rather than raising the cap.

**Exit gate**

```bash
gh run list --workflow=ci.yml --limit 5 --json conclusion \
  | jq '[.[].conclusion] | all(. == "success")'   # → true
gh run list --limit 20 --json conclusion --jq \
  '[.[] | select(.conclusion != "success" and .conclusion != "skipped")] | length'
```

**Anti-regression / mandatory follow-up:** every `.yml` you edit invalidates its `corpus_epoch_*.json` member hash (§2.4, Correction 1). After any workflow change, re-stamp in order — `make stamp-epochs` → `make sign-pins` → `make anchor-pins` — then gate on `make epoch-consistency`. Until that runs, the tree is **expected** to fail epoch verification.

---

### Phase 1 — Land one verified baseline, then exercise the release path · **P0 · ~1 day**

1. Merge Phase 0 onto `main`. Wait for the first green `ci.yml`. **Record the SHA** — this becomes `PROD_BASELINE`, the revision every Tier-A claim is stamped against.
2. `make coverage` and `make proofcore-coverage` locally on `PROD_BASELINE`; publish the result as a workflow artifact so the floor's trend is visible.
3. Tag `v0.4.0-rc.1` and let `release.yml` execute for the first time in repo history. This retires the H5 penalty *(self-reported M6: "signing/SBOM/SLSA never exercised — no tags, no releases")* and m3's *"H5 capped this factor: 24 = 60%"*.
4. Verify the outputs with the repo's own tooling:
   ```bash
   uv run python scripts/verify_release_artifacts.py dist/
   gh attestation verify dist/*.whl --repo artificial-hedge/dipcatcher
   ```
5. Decide explicitly whether `rc.1` is a real release. `release.yml` uploads artifacts and **does not publish to PyPI or create a GitHub Release**. For Tier A that is sufficient and honest; record the decision in `CHANGELOG.md` rather than leaving it ambiguous.

**Exit gate:** one green CI SHA + one successful signed, SBOM'd, attested tag build.

---

### Phase 2 — Close the fork and untrack the scratch · **P0 · ~2 days**

Sequenced first because receipts bind revision: rebasing 300+ commits after re-sealing evidence is far worse than reconciling now.

1. **Reconcile the divergence.** `origin/main` has 14 commits absent locally; local has 314 absent from origin. Local carries the entire INFLIGHT/verifier/epoch-pin history; origin carries the "external kimi-code gate-remediation chain". Determine which is the true spine **before** touching either. Then:
   - `make stamp-epochs` → `make sign-pins` → `make anchor-pins` in that order (`AGENTS.md` is explicit; every `stamp-epochs` stales the anchors).
   - `make epoch-consistency EPOCH_BASE=<ref>` as the gate.
2. **Untrack operational scratch.** Add `.dsh-24x7/`, `.box-soft-verify/`, `.cursor/`, `.serena/`, `.freebuff/` to `.gitignore` **and** `git rm -r --cached` them. Correct `AGENTS.md`, which currently asserts `.dsh-24x7` is gitignored when only two subpaths are. 936 tracked files leave the tree.
3. **Prune branches.** 231 local branches; delete merged and stale agent branches (`devin/*`, `kimi/*`, `codex/*`, `cursor/*`, `copilot/*`).
4. **Delete the empty `tests/tests/`** shell so the dropped mirror cannot re-enter collection. `testpaths` already excludes it; this makes that durable.
5. **De-colonize `INFLIGHT`.** 4,154 lines, newest 50 entries are mythology family names. Move the retired-family ledger into a dedicated verifier-side log file and keep `INFLIGHT` for live state. `day_grind_progress.md` (139 KB) and `.test_durations` (777 KB) get the same treatment — move to artifacts, keep a pointer.

**Exit gate**

```bash
git rev-list --left-right --count origin/main...HEAD   # → 0 0
git ls-files | grep -cE '^(\.dsh-24x7|\.cursor|\.serena|\.freebuff|\.box-soft-verify)/'  # → 0
make epoch-consistency EPOCH_BASE=origin/main          # → pass
```

---

### Phase 3 — Make the research surface defensible · **P1 · ~3 days**

The catalog is the repo's most visible quantitative claim surface, and at 10,222 optional / 2,693 live it currently invites the reading "10,222 research families exist."

1. **Tier `LIVE_OPTIONAL`.** Require every live-optional family to carry either a receipt pointer or an explicit `exploratory=True` flag, machine-checked. Collapse the default research surface to the **23 canonical families plus a justified tier**.
2. **Harden the guard.** Extend `tests/unit/research/test_catalog_retired_families.py` to assert: every non-retired optional family is justified or flagged exploratory; the count is ratcheted (may fall, never rise without an audit receipt); the default/CLI surface cannot enumerate the full optional set.
3. **Label provenance at the source.** `bench_chromatic_conv_family` and siblings in `benches_w464.py` are seed-offset template families. Give template-backed families an explicit `provenance: template` marker that flows into their blob, so a reader can tell a real family from a generated one without reading INFLIGHT.
4. **Keep the excellent parts:** audit-derived retirement, runtime de-emission, archived-receipt-still-verifies, unknown-family-fails-closed. These are Tier-A assets. Do not weaken them for speed.

**Exit gate:** `LIVE_OPTIONAL` ≤ agreed ceiling, 100% justified-or-flagged, ratchet test green, old-receipt verification still passing.

---

### Phase 4 — Hold Tier C honest without faking it · **P1 · ~2 days + ongoing**

Code cannot close the five institutional conditions. What code *can* do is stop the repo from implying it did.

1. **Anchor the forward prereg freeze — DONE 2026-10-08.** ✅ Executed and verified:
   ```bash
   uv run dipcatcher anchor-timestamp \
     --file receipts/forward_record_preregistration_v1.json.seal.json
   uv run python -c "from quant_fund.research.timestamp_anchor import verify_timestamps; \
     print(verify_timestamps('.'))"
   ```
   Result: `ok=true`, `chain_verified`, `fresh=true`, `errors=[]`. The freeze is now externally timestamped. **Only the file's sha256 was transmitted to the TSA, never its contents.** No v2 receipt is required — the defect was the missing anchor, not the window parameterization.
2. **Resolve the `window.start` label ambiguity (open).** `window.start = 2026-07-01` is the calendar span of `forward_2026H2`, not the collection start; the operative rule is `splits.forward` (first decision after the 2026-10-07 freeze). State this explicitly wherever the window is cited so it cannot be read as "collection began 2026-07-01".
3. **Repoint the readiness narrative (open).** `INSTITUTIONAL_READINESS.md:27,71`, `EVIDENCE_PROCUREMENT.md:41`, and `FORWARD_SHADOW_POWER.md:47` cite `forward_2026H2` as the path to condition 3. They should state that the freeze is externally anchored as of 2026-10-08 and that collection has not begun.
4. **Add a decision record** (`docs/adr/`) stating: conditions 1 and 2 are procurement/authorization, owned outside engineering; the earliest honest condition-3 date is the first prospective close after the 2026-10-08 anchor; the 2025 holdout is SPENT and must never be retuned.
5. **Start prospective collection (open — and date-bound).** Append-only, hash-chained, externally anchored at each close. This is the only path to Tier C condition 3. Every week of delay is a week of window permanently unavailable; this item is wall-clock-bound, not effort-bound, so it should start before the code work finishes.

**Exit gate:** v1 anchor `chain_verified` + `fresh` ✅; label ambiguity resolved; three docs repointed; ADR filed; first prospective close recorded.

---

### Phase 5 — Fix the reliability of the verification lane itself · **P1 · ~3 days**

1. **Root-cause the `-n auto` segfault.** No Python frame in the crashing thread means a C-level fault — the Rust `quant_core` extension is the prime suspect. Sequence:
   - reproduce under `-n 0` on a clean, low-pressure box to obtain a real traceback (`-n 0` is currently the only trustworthy path, per `INFLIGHT w1714`);
   - bisect native vs. pure-Python (`make native` off, re-run the matrix);
   - pin the thread environment the fleet already uses — `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`;
   - re-test the resource-pressure theory **now that local disk is no longer at 99%**, rather than carrying it forward as an assumption.
   Until root-caused: **make `-n 0` the documented verification path** and stop citing `-n auto` results as evidence. A flaky gate is worse than a slow gate, because it trains people to re-run until green.
2. **Make the mccabe ratchet real.** Fix the remaining 14 violations → run `check_mccabe_ratchet.py --write` to pin the 736 pre-existing → then ratchet the ceiling down in steps. Target: a ceiling a new module can actually meet. *(self-reported M4: "ceiling 74 is 5–7× conventional ceilings; ratchet is a floor-hold, not yet a cleanup" — closing this recovers most of the 10-point M4 deduction.)*
3. **Make coverage visible.** Publish the enforced 81% floor's trend — Codecov or a committed per-package summary — so H2's 50% cap does not apply. `make proofcore-coverage` per-package floors are a real asset; make them observable.
4. **Narrow the `except Exception` surface.** *(self-reported M4: 72–75 broad handlers remain, −15.)* The ratchet direction is correct; drive it to zero on money paths (`order_recon`, `broker_adapter`, `risk_gate`) first.

**Exit gate:** 3 clean full-suite runs with no segfault; ceiling ratcheted below 74 with the 736 pinned; coverage trend published.

---

### Phase 6 — Decide what fx-1 is, and say it in the README · **P2 · ~1 day (labeling) / ~weeks (training)**

Two legitimate paths. Pick one and make it explicit:

- **(a) Relabel (1 day, removes an 18-point honesty deduction immediately).** README states: *fx-1 is a harness, eval program, and training pipeline. No fine-tuned weights are published; `fx1.__version__` tracks the harness API, not a model release.* Costs nothing, tells the truth, and stops "7T fine-tune" ambition reading as a shipped model.
- **(b) Ship one honest tiny fine-tune.** A single real training run against the existing manifest gate, with a sealed receipt and a full proper-score eval card (pinball, CRPS, PIT, QLIKE, Brier, ECE, Kupiec — **never** Sharpe/P&L/NAV). This is the actual product goal and it is measurable: one sealed receipt proving the pipeline end-to-end.

Doing (a) first is not a retreat from (b) — it makes (b)'s eventual receipt unambiguous. What is not acceptable is leaving the ambiguity in place.

**Exit gate:** README states fx-1's shipped status unambiguously; `__version__` semantics documented; *(for (b))* one sealed training receipt + eval scorecard.

---

### Phase 7 — Supply chain, packaging, and release discipline · **P2 · ~4 days**

1. **Release discipline.** Zero `v*` release tags until Phase 1, then a tag per verified `PROD_BASELINE`. Every release traceable to a green SHA. (Four `attic/receipt-provenance/*` tags exist; they are provenance anchors, not releases.)
2. **Third-party security review.** *(self-reported M6: −6, "no external security review; 900 KB uv.lock dependency surface un-audited by third party".)* `make audit-all` covers pip/npm/cargo/Kronos internally; an outside pair of eyes on the money paths is the missing piece.
3. **Install path.** No `pip install` route exists. If analysts are expected to install rather than clone, add PyPI trusted publishing — and note that `release.yml` currently publishes nothing, so this is new work, not a config flip.
4. **Issue tracker.** *(self-reported m2: −13, "zero issues ever opened".)* For a single-owner fleet repo this is the highest-leverage collaboration gap: a place for a user to report that a receipt failed to verify.

---

### Phase 8 — Kill the doc rot and re-baseline the honesty ledger · **P2 · ~2 days**

1. **Enable the drift checker.** *(self-reported M3: PR #367 "docs drift checker for code refs + honesty sweep of stale paths — needed, not yet merged".)* With ~30 `AUDIT_*.md` files plus `DATA_CONTRACTS.md` (5,148 lines) and `SOTA_GAP_ANALYSIS.md` (2,321 lines), drift is guaranteed without an automated check.
2. **Correct `HONESTY_RATING.md`'s stale deductions.** It is dated `0e05f653` (2026-09-29) and at least one deduction is now wrong:
   - **m6 says "No LICENSE file."** `LICENSE` exists — *ARTIFICIAL HEDGE PROPRIETARY LICENSE, Version 1.0, 2026-09-30*. That deduction is void.
   - M7 says fx-1's `complete()` raises `NotImplementedError` as the current state; re-verify before carrying it forward.
   - M2/H5 and m3 rest entirely on "zero tags, zero releases" — voided by Phase 1.
   A self-scored report that is not re-baselined becomes a liability: it will be quoted as current.
3. **Re-score at `PROD_BASELINE`** with date + SHA, and keep the five anti-inflation rules (H1–H5) unchanged. The rules are the asset; the score is a snapshot.

---

## 4. Anti-goals — what not to do

These are the specific failure modes this repo is most likely to walk into next.

1. **Do not add features until Phase 1 is green.** Another capability wave adds unfalsifiable claims to an unverified repo. That is how the current state was reached.
2. **Do not chase coverage above the enforced floor.** 81% is enforced; raising it to satisfy a dashboard is exactly what H2 penalizes. Spend the effort on the `-n auto` segfault instead — a *reliable* 81% beats an unreliable 95%.
3. **Do not expand the family catalog.** Every new optional family widens the gap between perceived and actual research breadth. Phase 3 narrows it.
4. **Do not edit sealed receipts.** Ever — not the prereg window, not a re-sealed hash. Supersede.
5. **Do not retune the spent 2025 holdout.** It is SPENT. Using it again converts the repo's strongest honesty asset into its worst liability.
6. **Do not weaken the honesty tokens.** `FORBIDDEN_RESEARCH_METRIC_KEYS` ↔ `FORBIDDEN_HEADLINE_TOKENS` move together, always, and `tests/fx1/test_honesty_inheritance.py` keeps blocking drift.
7. **Do not merge the 314-commit divergence blindly.** Reconstruct intent first; then re-stamp, re-sign, re-anchor in order.
8. **Do not chase Tier C with code.** Conditions 1 and 2 need an entitlement and an authorization. A better adapter does not substitute for either.

---

## 5. Risk register

| Risk | Likelihood | Impact | Mitigation | Owner phase |
|---|---|---|---|---|
| Push cadence resumes and re-starves CI after Phase 0 | **High** — the fleet is the push source | **Blocker** returns | Merge-train batching; monitor green-streak length; treat a 2-run red streak as an alarm | 0, ongoing |
| Divergence reconciliation invalidates sealed epoch pins | Medium | Receipt rebinding cascade | Decide spine first; `stamp-epochs`→`sign-pins`→`anchor-pins` in order; `epoch-consistency` gate | 2 |
| Segfault is in the Rust native path and resists bisection | Medium | Verification lane unreliable | `-n 0` documented path; `make native` off as a fallback lane; report honestly as an open defect | 5 |
| `forward_2026H2` interval is treated as valid forward evidence by a reader | **Medium-High** — three docs cite it | **Integrity** — an unearned claim becomes load-bearing | Phase 4 supersession + doc repointing + explicit exclusion note | 4 |
| Retirement of 7,529 families breaks old-receipt verification | Low — guard already covers it | Evidence unverifiable | `test_catalog_retired_families.py` asserts archived receipts still verify; run it first in Phase 3 | 3 |
| Scope creep into Tier C | Medium — ambition is documented in `ULTRAPLAN_FRONTIER.md` | Blocked on procurement, so all effort wasted | Tier table in §1; ADR in Phase 4 | 4, ongoing |
| Re-scoring `HONESTY_RATING.md` reads as self-congratulation | Medium | Credibility | Keep H1–H5 rules fixed; publish deductions *before* fixes; never remove a deduction without its evidence | 8 |

---

## 6. Prod-readiness scorecard

Machine-checkable. Each row is a command that returns pass/fail. Re-run on every `PROD_BASELINE`. This is the artifact that replaces 90 prose docs.

| # | Control | Check | Target |
|---|---|---|---|
| 1 | CI liveness | `gh run list --workflow=ci.yml -L 5 --json conclusion --jq 'all(.conclusion=="success")'` | 3-run green streak |
| 2 | Full matrix | `gh run list --workflow=matrix.yml -L 1 --json conclusion` | `success` on `PROD_BASELINE` |
| 3 | History integrity | `git rev-list --left-right --count origin/main...HEAD` | `0  0` |
| 4 | Epoch chain | `make epoch-consistency EPOCH_BASE=origin/main` | pass |
| 5 | Tree hygiene | `git ls-files \| grep -cE '^(\.dsh-24x7\|\.cursor\|\.serena\|\.freebuff\|\.box-soft-verify)/'` | `0` |
| 6 | Receipts | `make receipts-reverify && make evidence-audit` | **FAIL (exit 2)** — 2/10 fixed, 8 need schema-specific verifiers |
| 7 | Lint/format | `make lint` | pass |
| 8 | Types | `make typecheck` + `uv run mypy src/fx1` | pass |
| 9 | Coverage floor | `make coverage` | ≥ 81%, trend published |
| 10 | Test reliability | 3× full suite, no segfault | 3/3 clean |
| 11 | Complexity | `make lint` (ruff mccabe) | ceiling < 74, 736 pinned |
| 12 | Honesty | `make fx1-test -k honesty` | pass, no token drift |
| 13 | Catalog | `LIVE_OPTIONAL` justified-or-flagged | 100%, ratchet green |
| 14 | Forward evidence | prereg seal RFC 3161-anchored, `verify_timestamps` `chain_verified` + `fresh` | pass ✅ 2026-10-08 |
| 15 | Release | `git tag -l 'v*'` non-empty; attestation verifies | ≥ 1 signed release |
| 16 | Dependency audit | `make audit-all` | no high/critical |
| 17 | Docs drift | drift checker enabled | pass |
| 18 | Honesty ledger | `HONESTY_RATING.md` dated at `PROD_BASELINE` | current |

Rows 1–5 are the Tier-A gate. Tier A is **not** achieved until all five are green.

---

## 7. Critical path

```
P0 CI liveness ──┬──> P1 green baseline + first signed release ──┐
                 │                                                 ├──> Tier A
P2 fork + hygiene┴──> P4 evidence supersession ──────────────────┘

P3 catalog ──┐
P5 reliability ──┴──> P6 fx-1 labeling ──> P7 supply chain ──> P8 re-baseline ──> Tier B
```

- **P0 and P2 are both P0.** They parallelize cleanly (one is CI config, the other is git history), but both must land before any evidence work — rebasing 300+ commits after re-sealing is strictly worse.
- **P4 is the long pole for Tier C** and cannot be compressed: it is wall-clock-bound from the v2 anchor date, not effort-bound. Start it early even though its output is unglamorous, because every week of delay is a week of forward window permanently unavailable.
- **P3, P5, P6, P7, P8 are independent** and parallelize across the fleet once P0–P2 land.

**Critical path to Tier A: ~2–3 weeks** with one owner plus the fleet, assuming the merge-throttling change in P0.4 holds.

---

## 8. Definition of done

**Tier A — Research platform:**
- [ ] 3 consecutive green `ci.yml` runs on `main`, and a green `matrix.yml` on the same SHA
- [ ] `origin/main` and local `HEAD` at `0  0` divergence
- [ ] One signed, SBOM'd, SLSA-attested release tag traceable to that green SHA
- [ ] Zero tracked scratch files; `INFLIGHT`/`day_grind_progress.md` de-colonized
- [ ] `LIVE_OPTIONAL` fully justified or flagged, with a ratchet test
- [ ] 3 clean full-suite runs with no segfault; mccabe ceiling below 74
- [x] Forward prereg freeze externally anchored — `chain_verified` + `fresh` (done 2026-10-08)
- [ ] First prospective forward close recorded after the anchor
- [ ] README states fx-1's shipped status unambiguously
- [ ] `HONESTY_RATING.md` re-scored at `PROD_BASELINE` with the void deductions corrected

**Tier B — additional:**
- [ ] Third-party security review completed on money paths
- [ ] Install path decided and documented (clone-only, or PyPI)
- [ ] Issue tracker live
- [ ] Docs drift checker merged and green

**Tier C — explicitly out of engineering scope.** Blocked on procurement (condition 1) and authorization (condition 2). Condition-3 evidence can begin at the first prospective close after the 2026-10-08 anchor. Track honestly; do not simulate.

---

## 9. Two notes on the repo's own framing

**The honesty engine works.** `HONESTY_RATING.md` docked itself for the very things this audit independently found: red-main recurrence (H4), unexercised release machinery (H5), unenforced-metric caps (H2), the fx-1 plan/model gap, and the license file. A repo that publishes deductions against itself is rare and is the reason this plan can be written from evidence rather than optimism. The deductions were not wrong — they were stale, which Phase 8 fixes.

**The deepest structural risk is velocity outrunning verification.** The evidence shows a fleet capable of ~4,000 commits, 13.4k modules, and 12 merges in 68 seconds — attached to a pipeline where **400 consecutive runs produced no verdict**. Every additional wave increases the surface that CI is supposed to check while CI cannot check anything. Phase 0 is therefore not housekeeping; it is the precondition for every honest claim this repo is capable of making.

---

## 10. Implementation log — 2026-10-08

What was actually executed against this plan, with verification. **No commit, tag, or push was made**; all changes sit in the working tree pending an owner ruling.

### Completed

### Phase 5 — the `-n auto` segfault: ROOT CAUSED, not the Rust extension

> **Correction to this plan.** Phase 5 originally listed the Rust `quant_core` extension as "prime suspect." **That was wrong.** The cause is an OpenMP runtime collision, proven by an L3 segfault-hunter lane on 2026-10-08 and independently verified here.

**Root cause:** the test process loads **three different `libomp.dylib` OpenMP runtimes at once** — PyTorch's, scikit-learn's, and Homebrew LLVM's (`/opt/homebrew/Cellar/libomp/22.1.8/lib/libomp.dylib`). Verified in this tree:

```
.venv/…/torch/lib/libomp.dylib          md5 768c82fd9ff1
.venv/…/sklearn/.dylibs/libomp.dylib    md5 04b22f280351
/opt/homebrew/Cellar/libomp/22.1.8/…     third copy
```

Three distinct binaries export the same mangled C++ template symbols, so `dyld` binds across them. A worker thread created by **Homebrew's** libomp ends up executing **torch's** `__kmp_suspend_initialize_thread`, which looks up the thread in *its own* `__kmp_threads[]` table, gets `NULL`, and dereferences a struct member at offset `0x580`:

```
EXC_BAD_ACCESS (SIGSEGV)  KERN_INVALID_ADDRESS at 0x0000000000000580
#0 libomp.dylib __kmp_suspend_initialize_thread+32      ← torch's copy
#2 libomp.dylib kmp_flag_64<…>::wait(...)+1712           ← homebrew copy
#5 libomp.dylib __kmp_launch_worker(void*)+252
```

**Why it presents as `<no Python frame>`:** the faulting thread is an OpenMP worker (`__kmp_launch_worker` → `_pthread_start`), never a Python thread — which is exactly the symptom xdist reported. It is also why `-n 0` is unaffected: no xdist fork, no cross-runtime worker.

- **Trigger site:** LightGBM `Dataset.construct()` (`LGBM_DatasetCreateFromMats`) in a worker that has already imported torch.
- **Evidence:** the identical stack appears in **10/10** crash reports (7 catalogued at the time of the diagnosis; **20 `python3.12-*.ips` files now exist** in `~/Library/Logs/DiagnosticReports/`), all `EXC_BAD_ACCESS` at `0x580`.
- **Mitigation validated:** reproduced 3/3 deterministically, **eliminated 2/2 with `OMP_NUM_THREADS=1`**.
- **Fix applied:** `Makefile:19-21` `TEST_ENV` now pins `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE PYTHONFAULTHANDLER=1 TOKENIZERS_PARALLELISM=false`, and `full-coverage.yml` / `simtest.yml` mirror it. This is also the convention `AGENTS.md` already mandates for the remote fleet.

**What remains:** thread pinning suppresses the collision but does not remove it. The durable fix is a single OpenMP runtime — consolidate on one `libomp` (or link the offending wheels against a single vendor), which is an environment/packaging change beyond the Makefile one-liner. Until then, `-n 0` remains the conservative verification path. Full diagnosis: `.dsh-24x7/lane_reports/l3-segfault.md` (34 KB).

### Phase 3 — the catalog surface is now measurable, and the number is damning

The justify-or-flag classification is implemented in `src/quant_fund/research/catalog/optional_classification.py` and **derived from the audit's own evidence file** (`quality/canon_qualification_summary.json`) rather than hand-listed — so re-running `scripts/canon_qualify.py` and the classifier cannot disagree without the difference surfacing as an unclassified family.

**Measured result:**

| Set | Count |
|---|---|
| `DEFAULT_BENCHMARK_FAMILIES` (the 23 canonical) | **23** |
| `LIVE_OPTIONAL_BENCHMARK_FAMILIES` | **2,693** |
| …of which **justified** (own behavioral mechanism, not a template stub) | **6** |
| …of which **exploratory** (template-backed, audit-flagged non-qualifying) | **2,687** |
| **Unclassified** | **0** ✅ |

**Six.** Of 2,693 families the research surface can still emit, **6 have a real mechanism and 2,687 are template stubs** the same audit already flagged as retirement candidates. This is the §2.5 suspicion, now quantified: the "live optional" surface is 99.8% generated bulk, and presenting it as a live research surface overstates what exists.

Deliberately conservative:
- `exploratory` is a **label, not a retirement.** Nothing removes a family from `OPTIONAL_BENCHMARK_FAMILIES`, changes `verify.py`'s allow-list, or weakens fail-closed verification — **an archived receipt naming an exploratory family still verifies exactly as before.**
- The `agent.py` change is **docstring-only** (recording that `_retired_optional_families()` must keep enumerating the full optional set, since narrowing it would let a retired family back into a live receipt).
- `LIVE_OPTIONAL_BENCHMARK_FAMILIES_CEILING = 2693` is a ratchet that may fall, never rise without an audit receipt. *(A `FAMILES` typo in that export name was corrected to `FAMILIES` for consistency; everything re-validated after.)*

**Validated:** 17 tests pass (new classification tests plus the pre-existing retired-families guard); `ruff check` clean; `ruff format --check` clean; `mypy --strict` → *no issues in 23 source files*.

**The decision this forces (owner):** 2,687 exploratory families are now explicitly labelled rather than quietly presented as research breadth. The honest follow-up is to either retire them or stop describing the live-optional set as a research surface — the classification makes either choice cheap and visible.

**Phase 4 step 1 — forward prereg externally anchored.** ✅

```
uv run dipcatcher anchor-timestamp --file receipts/forward_record_preregistration_v1.json.seal.json
→ timestamp=quality/timestamps/receipts__forward_record_preregistration_v1.json.seal.json.tsr

verify_timestamps('.'):
{ "ok": true, "anchored": true,
  "fresh": { "receipts/forward_record_preregistration_v1.json.seal.json": true },
  "chain": { "receipts/forward_record_preregistration_v1.json.seal.json": "chain_verified" },
  "errors": [] }
```

The forward freeze was self-attested; it is now externally timestamped and chain-verified against the repo's pinned FreeTSA CA and signer certs. `anchors.json` gained a 4th entry. Only the file's sha256 was transmitted — never its contents.

**Phase 0 — CI orchestration (12 workflow files edited).**

| Change | Files |
|---|---|
| `cancel-in-progress: true` → `false` (a gate must reach a verdict) | `ci.yml`, `fx1.yml`, `proofcore.yml`, `matrix.yml` |
| Missing `concurrency:` added, `cancel-in-progress: false` | `witness_monitor.yml` |
| `paths-ignore` for doc-only churn | `docs.yml`, `atlas.yml`; `matrix.yml` keeps `push: main` + doc filter |
| Doc entry dropped from existing `paths:` filter (GitHub rejects `paths` + `paths-ignore` on one event) | `web_explorer.yml`, `replay_viz.yml` |
| `push:` trigger removed; schedule/PR/dispatch retained | `adversarial-properties.yml`, `perf-baseline.yml` |
| Already schedule/dispatch-only — no change needed | `property-nightly.yml`, `simtest.yml` |

All 22 files re-parse as valid YAML; all `uses:` pins remain SHA-pinned. Verified independently by YAML-parsing each workflow and reading back the `concurrency` block: all five gates report `cancel=False`.

**Two errors in the agent's first pass were caught on review and reverted:**

- `docs.yml` had been given `paths-ignore: ['**/*.md','docs/**']` on the reasoning that doc churn cannot break the docs build. That is **inverted** — the workflow runs `mkdocs build --strict`, whose entire purpose is catching a broken nav reference or bad link, so a Markdown change is exactly what can break it. Reverted, with a comment recording why so it is not "helpfully" re-added.
- `atlas.yml` had the same filter, but it also runs `pytest tests/unit/docs`. Reverted: it is one cheap job, and skipping a docs-consistency test is the wrong trade while verification is already dark.

**Jobs per push to `main`** (measured, expanding every `strategy.matrix` to concrete job counts and applying GitHub's `**/` glob semantics):

| Push type | Workflows | Jobs |
|---|---|---|
| Code push | 16 | **96** |
| Workflow-only push | 16 | **96** |
| Doc-only push | 15 | **60** (`matrix.yml`'s 36 skipped) |

Total across all push-triggered workflows: **96 jobs / 16 workflows**, down from 19 workflows.

> **This is not sufficient, and the plan should not pretend otherwise.** A code push still requests 96 jobs against a 20-job concurrency cap — roughly five waves — so CI will *still* queue at high push cadence. The config edits remove the self-cancelling behaviour and reclaim the doc-push waste; the binding constraint is **push cadence**, and the fix for that is the merge queue in Phase 0 step 4, which is a repository-setting change that cannot be made from the tree.

**Phase 2 step 2 (partial) — scratch no longer re-trackable.** `.gitignore` now covers `.dsh-24x7/`, `.cursor/`, `.serena/`, `.freebuff/`, `.box-soft-verify/`, `.playwright-mcp/`, `.hypothesis/`, verified with `git check-ignore`. `AGENTS.md` corrected: it claimed `.dsh-24x7` was gitignored when only two subpaths were; it now states the rule *and* the 911 already-tracked files that `.gitignore` cannot untrack.

**Phases 6 + 8 — fx-1 status and honesty ledger.** `README.md` (+116/−3), `HONESTY_RATING.md` (+49), `docs/FX1.md` (+22), `docs/FX1_API_STABILITY.md` (+9):

- A prominent **"What fx-1 is today"** block stating that fx-1 is a harness / eval program / training pipeline, that **no fine-tuned fx-1 weights are published**, that `__version__` tracks the harness API rather than a model release, and that Kronos checkpoints under `data/models/` are third-party.
- A **Tier A/B/C readiness table** wired to `docs/INSTITUTIONAL_READINESS.md`, so research-platform readiness cannot be misread as live-trading readiness.
- A dated **2026-10-08 CI-blackout note** in the CI section, so the docs no longer imply a healthy pipeline.
- `HONESTY_RATING.md`: correction notice added at the top framing the report as a snapshot at `0e05f653`; H1–H5 rules preserved **verbatim**; deductions annotated in place rather than deleted — **m6 "No LICENSE file" marked VOID** (LICENSE exists, proprietary v1.0, 2026-09-30, one day after the snapshot), M2/H5 + m3 + M6/H5 marked accurate-but-pending, M7 marked substantively correct with narrowed wording, M3 drift re-verified, M1/H2 explicitly labelled *not re-verified* rather than assumed.

`mkdocs build --strict` passes clean (baseline was also clean; one failure during the work was self-inflicted — a link into `artifacts/`, which is outside `docs_dir` — and was fixed).

**Phase 5 — the mccabe ratchet is now real.** `INFLIGHT w1714`'s numbers turned out to be **stale**, and this phase changed the picture materially:

| Metric | Measured |
|---|---|
| Functions measured (C901) | 19,748 |
| **> 74 (hard ceiling)** | **3** — and all three are `C901`-ignored in `pyproject.toml:177-179` |
| **≥ 10 (ratchet scope)** | **1,089** |
| Baseline rows | 358 → **1,094** (+736, written by the tool only) |
| Unlisted regressions failing `check` | 736 → **16** |

- **The "14 remaining violations" claim was wrong.** Measured ceiling violations among linted files are **0**. All 14 named offenders (`lh011`, `lh013`, `risk_gate.check_order`, `_attempt_fill`, `build_labels`, …) are resolved.
- **The 736 figure was right**, and is now pinned. The refusal message was real, but its stated cause was not: `--write` was not refusing because of ceiling violations. Root cause is a **divergent merge** — the baseline was written at `150f1b448` (2026-09-28) where `lh011`=15; merge `25279246a` (2026-09-29) rewrote `ast_scan.py` and is **not a descendant** of the baseline commit (`git merge-base --is-ancestor` → false). **The interlock was correct and is preserved.** This is independent corroboration of **F3**: the repo contains genuinely divergent branches, not merely a local-vs-origin split.
- **New fail-closed mode** `--allow-with-census`, which separates *pinning debt* from *blessing regressions*: it recomputes a digest over the census (hand-edits rejected before any write), re-scans and refuses if the tree moved, and **carries every existing pin at its recorded lower value — never at the fresh number**, so it structurally cannot raise a pin. The original `--write` refusal behaviour is unchanged.
- Artifacts: `quality/mccabe_violations_2026_10_08.json`, `quality/mccabe_refactor_dossier.md` (per-function decomposition with `ast`-extracted branch counts and the characterization tests required *before* refactoring), `tests/unit/quality/test_mccabe_census.py` (8 tests, all passing), and the rewritten baseline.
- **No `src/` function was refactored.** `_attempt_fill`, `check_order`, and `promotion_dry_run` are money-path code; an uncharacterized refactor there is a worse outcome than an honest dossier.

### PR gate result — `make test` FAILS, and the segfault is **not** fixed

```
128 failed, 12356 passed, 190 skipped, 1 xfailed, 9 errors, 115 subtests passed in 953.11s
make: *** [test] Error 3        MAKE_TEST_EXIT=2
Fatal Python error: Segmentation fault
[gw7] node down: Not properly terminated
INTERNALERROR> KeyError: <WorkerController gw10>
```

**Two findings, both significant.**

**1. The thread-pin mitigation does not hold for the full suite.** The L3 root cause (three `libomp.dylib` copies) is solid and was verified here by md5. But its claim that the crash is *eliminated* by `OMP_NUM_THREADS=1` — validated 2/2 on an 11-line workload — **does not survive the real suite.** With `TEST_ENV` fully pinned (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE ...`), the run still segfaulted at ~31%, taking a worker with it and aborting xdist. The crash also destroyed the failure summary, so the gate emits neither a clean verdict nor a usable list of what failed.

**2. The honesty gate was red — and the cause was a LOST FIX, now restored.** 21 of the 128 failures are `tests/unit/research/catalog/test_forbidden_metric_flags.py`, which asserts `family_blob_forbidden_metrics_absent` rejects a nested `live_pnl_claim: True`. **Proven pre-existing:** reverting the catalog package to `HEAD` and re-running reproduces all 21 identically. The only working-tree change to that function's file was `+10` purely additive lines (a new `DEFAULT_BENCHMARK_FAMILIES` constant and one `__all__` entry). Commit `767d59d63` (*"fix: normalize research flags and handle API key inputs"*) touched both the function and its test and is the first place to look — a live-PnL detection path that fails to detect a nested live-PnL claim is a **Tier-A blocker independent of CI**.

**Root cause: commit `767d59d63` implemented the correct rule, and that fix was subsequently lost or reverted while its test survived.**

`767d59d63` (*"fix: normalize research flags and handle API key inputs"*) added both `tests/unit/research/catalog/test_forbidden_metric_flags.py` **and** a stricter `family_blob_forbidden_metrics_absent`. Its own docstring in that commit states the intent:

> ``live_pnl_claim`` itself is exempt at any depth **only when its value is literally False** … a true or malformed nested flag must fail closed too.

The code in that commit implements exactly that (`if value is not False: return False`, with recursion into nested dicts/lists). The code now on `main` is the **older, weaker** version:

```python
for key in _iter_mapping_keys(payload):
    if key == "live_pnl_claim":
        continue          # exempt regardless of value — a live claim passes
```

So `{"live_pnl_claim": False, "receipts": [{"live_pnl_claim": True}]}` returned **True** — a receipt asserting a live P&L was passing the research-headline honesty gate. The test asserting the opposite has therefore never been green since that commit.

**This is not a new defect and not mine — it is a silently dropped security/honesty fix**, and it is very likely the same divergent-merge phenomenon that also lost the mccabe baseline relationship (§5: `25279246a` is not a descendant of `150f1b448`). Losing a merged fix without its test being removed is exactly what a bad merge looks like.

**Fix applied:** the implementation from `767d59d63` is restored verbatim (docstring included). All **21** previously-failing tests now pass.

**Regression evidence for the stricter rule** (the earlier full-`tests/unit/research/` run timed out, so this was checked directly rather than assumed):
- Every test file that constructs a `live_pnl_claim: True` payload — **15 files** — passes: **305 tests, 0 failures.** These are exactly the cases a stricter check could break.
- All 4 `src/` sites that build a `live_pnl_claim: True` payload are in `src/fx1/{train/receipts_audit,data/receipts_audit,data/sources_audit,data/corpus_audit}.py`, i.e. **deliberately forged fixtures that exist to prove the audit detects live claims.** No production receipt-construction path sets the flag true, so no real artifact changes behaviour.
- The corresponding fx1 audit/receipt tests pass (158 tests).
- Of 389 committed receipts, **0** fail the stricter rule.

Distribution of the flag across `src/` + `tests/`: `False` ×420, `True` ×29 (all negative fixtures), plus a handful of deliberately malformed values (`0`, `0.0`, `1`, `null`, `None`, `"yes"`) in fail-closed test cases. Verified blast radius: re-running the check over all 389 committed receipts shows **0** now fail the stricter rule, so restoring it closes a real hole without invalidating any existing evidence. `_iter_mapping_keys` remains in use elsewhere (`catalog/families.py`), so nothing became dead code.

**Failure triage — the 128 are not one problem.** After the honesty fix, pytest's `lastfailed` dropped 139 → 115, and the remainder splits into at least three unrelated classes:

| Cluster | Count | Class | Diagnosis |
|---|---|---|---|
| `tests/unit/cli/test_sota_cmds.py` | 37 | **Lost mount — RESTORED** | See below. |
| `test_train_cov.py`, `test_train_walk_forward_branches.py`, `test_pipeline_training.py` | 47 → **13** | Harness gap fixed; two further latent defects remain. See below. |
| `test_vol_per_security.py`, `test_vol_scope.py` | 14 | Unclassified | Not yet triaged. |
| `test_quality_ratchet.py` | 3 | **Known pre-existing** | Broad-exception / type-ignore manifests vs `src/`; flagged independently during the mccabe work. |
| `tests/fx1/test_selftest.py`, `test_goldenpath.py` | 4 | Out of PR gate | `tests/fx1` is excluded from `testpaths` by design; run via `make fx1-test`. |

The two **lost-fix / missing-feature** clusters (`test_forbidden_metric_flags.py`, `test_sota_cmds.py`) are the concerning ones, because they are exactly what a bad merge looks like: tests that survive while the implementation they pin does not. That pattern now has two independent instances and should be treated as a systematic audit target, not two bugs — see F3.

**Second lost-fix case — also a mount, and now restored.** `src/quant_fund/cli/sota_cmds.py` exists and defines all three operator groups with all 17 subcommands, including `forward-shadow freeze`. `cli/_app.py` mounts 17 other sub-apps via `app.add_typer(...)` — but **`sota_cmds` was not in that list**, and its own docstring still claims *"`cli._app` mounts this module"*. So the mount was dropped and nobody noticed, because the module still imported fine; it simply wasn't reachable from the CLI.

That is the same failure shape as the honesty fix: **working code, correct tests, missing wiring.** The concrete cost was that `uv run dipcatcher forward-shadow freeze` — the command `docs/INSTITUTIONAL_READINESS.md:73` names as the mechanism for closing institutional condition 3 — did not exist, so the documented procedure for starting forward collection was not executable.

**Fix applied:** import the three apps in `cli/_app.py` and mount them as `prospective-sota`, `forward-shadow`, and `sota`. `sota_cmds` imports only stdlib + `typer` at module level (its shared helpers are lazy), so the import cannot reintroduce the `cli.support -> cli._app` cycle its docstring warns about. **All 40 tests in `test_sota_cmds.py` now pass**; bare `dipcatcher` still prints help and exits 0; `ruff` and `mypy --strict` clean.

**Third instance — a fail-closed requirement landed without fixture propagation. Partially fixed; a second defect is now exposed.**

`da5c3e78c` (2026-10-07) made dataset identity bind the **bytes** of every materialized panel source under `cfg.data.root`:

```python
SOURCE_PANEL_PARTS = ("gold/features.parquet", "gold/labels.parquet", "silver/universe.parquet")
# _source_manifest_hash → DatasetIdentityError if any is absent under data_root
```

That is a *correct* requirement — it is what makes a later source-data swap detectable even when panel columns match — and the suite's own dedicated test (`tests/unit/test_manifest_dataset_identity.py`, 17 tests) passes. But pipeline tests set `cfg.data.root = tmp_path`, an empty directory, and **no conftest fixture materializes the panel**. The 47 tests therefore began failing on 2026-10-07; `test_train_cov.py` was last touched 2026-09-26, before the check existed.

This is a third instance of the same family — a correct, stricter requirement added without propagating to its dependents — though unlike the first two it is an incomplete change rather than a lost merge.

**Fix applied (harness only, no weakening of the check):** `tests/support/panel_sources.py` provides `materialize_panel_sources(data_root)`, writing minimal parquet stubs for `SOURCE_PANEL_PARTS` (lazy-imported so it stays test-only). Wired into the three suites' config builders; 10 inline assignments patched in `test_pipeline_training.py`. `ruff` clean.

**Verified experimentally:** materializing the panel satisfies `_source_manifest_hash`; an empty root still raises `DatasetIdentityError` — the check is untouched.

**Result: 47 → 13.** These three files collect **149** tests (104 + 18 + 27); **136 pass, 13 fail**. That is the measured split.

The 13 remaining failures are **two further latent defects**, both pre-existing and previously unreachable because the identity check fired first:

| Cause | Error | Count |
|---|---|---|
| Test stubs bypass the identity-binding writer | `ArtifactManifestError: artifact manifest carries no dataset identity block` (`artifact_manifest.py:284`) | most of the `*_auto_selects_*` family |
| Test-local classes are not joblib-serializable | `_pickle.PicklingError: Can't pickle <class '...<locals>.FakeHMM'>` | includes `test_train_garch_persisted_fit_uses_latest_return_history` |

Neither is caused by the fixture change. The first is **not a production bug**, and the distinction matters. `save_training_artifact(payload, path, *, identity)` is the correct writer: identity is keyword-only and required, and on any binding failure it unlinks the payload so it can never leave an artifact behind unbound. Every `train/*.py` module uses it. The 14 call sites that bypass it are **test stubs** (`save_joblib_artifact` in `test_train_cov.py` ×6 and `test_pipeline_training.py` ×8) which write a bare artifact; the real auto-selector then calls `identity_from_artifact(...)` on it, which correctly refuses.

So the chain is working as designed — a stub that skips identity binding is *supposed* to fail. Two defensible fixes, and this is an owner call rather than an obvious cleanup:

- update each stub to go through `save_training_artifact(..., identity=identity_for_training(frame, config=cfg, label=..., features=[...], label_horizon_bars=...))`, which needs a real frame and label per stub (14 sites, and the stubs do not all carry one); or
- have the stubs bind a deliberately minimal identity, keeping the test focused on the auto-selection logic rather than identity construction.

The first is more faithful; the second is less churn. **Neither was applied here** — rewriting 14 test stubs to route around a deliberate fail-closed writer is a semantic decision for the owner, not a mechanical fix. The second is a test-harness issue — a class defined inside a test function cannot be pickled.

**Systematic follow-up: the dropped-wiring class is now mechanically checkable, and the CLI surface is clean.**

Rather than keep finding these one at a time, the `sota_cmds` failure shape was turned into a detector. `scripts/audit_cli_mounts.py` walks every module under `src/`, collects module-level Typer apps via AST, and reports any that never appear as an argument to `add_typer(...)` anywhere.

```
$ make audit-cli-mounts
CLI mount audit — PASS
  typer apps defined : 32
  apps mounted       : 26
  defined, unmounted : 4
    [expected] app  (src/fx1/cli.py)
    [expected] app  (src/fx1/interactive/app.py)
    [expected] app  (src/quant_fund/cli/_app.py)
    [expected] app  (src/quant_fund/mc_engine/cli.py)
```

All four unmounted apps were checked by hand and are **legitimate**: `fx1` ships its own console scripts and `AGENTS.md` plus `tests/unit/test_fx1_dependency_edge.py` forbid the harness importing fx1; `cli/_app.py` *is* the root app; and `mc_engine` has its own declared entry point (`pyproject.toml:87` — `mc-engine = "quant_fund.mc_engine.cli:app"`) with its own tests under `tests/unit/mc_engine/`.

**So the CLI surface contains no further dropped mounts** — the `sota_cmds` case was the only one. That is a useful negative result: it bounds the problem to wiring surfaces this audit does not yet cover (manifest writers, conftest fixtures, registry exports), rather than leaving it unbounded. Exemptions are declared in `EXPECTED_UNMOUNTED` with a recorded reason so the list cannot quietly grow into blanket suppression, and the audit exits non-zero on any unexpected entry.

Remaining failure clusters (from pytest's `lastfailed` cache): `test_sota_cmds.py` (37), `test_train_cov.py` (27), forbidden-metric flags (21), `test_train_walk_forward_branches.py` (11), `test_pipeline_training.py` (9), `test_vol_per_security.py` (9), `test_quality_ratchet.py` (3 — the previously-known pre-existing ones). None are in files this session edited; the targeted suites for every change made here are green.

**Net:** the Tier-A definition of done in §8 cannot be met by configuration alone. It needs (a) a real fix for the 21 honesty-gate failures, (b) the remaining ~107, and (c) a segfault mitigation that survives the full suite — most plausibly by eliminating one of the three `libomp` copies rather than by environment variables.

### New finding — F4: epoch pins were already stale at HEAD

Running the epoch verifier read-only during Phase 0:

```
uv run python scripts/verify_epoch_chain.py --corpus-dir .github/workflows \
  --pattern '*.yml' --pin quality/epoch_heads.json --require-stamped --allow-member-updates
→ epoch-chain corpus=.github/workflows head=corpus_epoch_aa23d445b77a23c9.json
  unstamped=0 errors=12   (head_member_digest_drift … FAIL)
```

Twelve files drift, but **ten are this plan's own workflow edits. `ci.yml` and `simtest.yml` were already drifting at `HEAD` before any edit.** Epoch verification was therefore **already failing on `origin/main`** — an integrity-chain defect that predates this work. It belongs with F3 in the reconciliation commit.

**Expected handoff state:** `.github/workflows/` fails epoch verification until `make stamp-epochs` → `make sign-pins` → `make anchor-pins` runs. That step needs `GATE_SIGNING_KEY` plus TSA network access, and is sequenced behind the Phase 2 history decision — deliberately not run unattended.

### Agent-session loss and recovery

The Phase 3 subagent's session was **lost before it reported**. Because it had already modified `src/`, its work was treated as unverified and re-validated independently rather than trusted:

| Check | Result |
|---|---|
| Package imports | `quant_fund.research.catalog` + `optional_classification` import cleanly |
| Tests | **17 passed** — new classification tests *and* the pre-existing retired-families guard |
| `ruff check` | All checks passed |
| `ruff format --check` | 24 files already formatted |
| `mypy --strict` (catalog pkg) | **Success: no issues found in 23 source files** |
| `agent.py` blast radius | Docstring-only — no behavior change |

The work was sound. One defect was found and fixed in review: the export `LIVE_OPTIONAL_BENCHMARK_FAMILES_CEILING` was misspelled (`FAMILES` for `FAMILIES`) across three files and renamed.

**Note on validation cost:** running `mypy --strict` over the whole `src/quant_fund` (~13k modules) is what exhausted the agent's budget. Scoping mypy to the changed package (`mypy --strict src/quant_fund/research/catalog`, 23 files) returns in seconds and is sufficient to validate this kind of change. Worth adopting as the default review loop.

### Still open

- **Scorecard row 6 (`make receipts-reverify`) — FAILS, exit 2, after ~20 minutes.** The gate ran to completion: **10 failures across 393 files in `receipts/`.** Both problems are now diagnosed.

  **Cost.** The gate shells out once per receipt:
  ```python
  # src/quant_fund/proofcore/ci.py:291
  verifier = verify if verify is not None else _cli_verifier   # ← subprocess
  for path in paths: ok = verifier(path)
  ```
  391 receipts × a measured **3.1 s** per invocation ≈ **20 minutes**, with **zero output until the end**. Almost all of that is interpreter startup and heavy transitive imports, not verification. The injection seam already exists — verifying in-process would collapse this to seconds. Isolation is probably deliberate, so keep it behind a flag rather than delete it.

  **Failure 1 — a real gate bug (FIXED).** `receipt_paths()` collected `root.rglob("*.json")` with **no exclusion for `*.seal.json` sidecars**, so seal files were verified *as receipts* and failed with a spurious `receipt_sha256_missing_or_invalid` — a sidecar carries `{schema, file, sha256, sealed_at}`, not a receipt digest. Two of the ten failures were this (`forward_record_preregistration_v1.json.seal.json`, `cost_aware_rerun_20261007.json.seal.json`). Fixed with an explicit sidecar exclusion; collection went **391 → 389**, sidecars collected **2 → 0**. Sidecars keep their own coverage via `verify_payload`/`verify_seal` (`tests/unit/data/test_prereg_seal.py`) and `verify_timestamps`. `ruff` clean, `mypy --strict` clean, 70 tests green across the affected suites.

  **Only 1 of 389 receipts actually has a broken self-seal — and my first read of this was wrong, twice.** The first pass reported three. Two of those turned out to be intact, and the correction is instructive:

  | Receipt | Schema | Verdict |
  |---|---|---|
  | `calib_real_drill.json` | `calibration_audit.drill.v1` | **GENUINE MISMATCH** |
  | `fx1_drain_audit.json` | `drain_audit.v1` | **INTACT** — sealed under `ensure_ascii=True` |
  | `fx1_jobs_audit.json` | `jobs_audit.v1` | **INTACT** — sealed under `ensure_ascii=True` |

  **Correction 1 — the fx1 pair is not corrupt.** Both hash correctly over their own content under `json.dumps`'s default `ensure_ascii=True`. `canonical_json_bytes` uses `ensure_ascii=False`, and the payloads contain an em dash (`—`) and a rightwards arrow (`→`) stored as `\uXXXX` escapes; the two serialization forms therefore emit different bytes and different digests. Nothing was tampered with and nothing drifted. My initial "wrong since creation" reading was a **false positive** — the same convention-vs-corruption conflation I had criticised in the original gate, reproduced in my own tool.

  **Correction 2 — `calib_real_drill` is the real one, and it is benign.** `fad6f810f` changed only three lines, all metadata:

  ```diff
  -  "kind": "calibration_audit.v1",
  +  "kind": "calibration_audit.drill.v1",
  -  "schema": "calibration_audit.v1"
  +  "schema": "calibration_audit.drill.v1"
  ```

  No statistic, input hash, or evidentiary field was touched — the research content is intact and the digest simply was not rebound after the tag rename. It still needs an explicit, recorded re-seal, but **this is label drift, not data corruption.**

  ```
  calib_real_drill.json  stored=0124990ee36a3cc3…  actual=c5888e19ae1ec764…
  ```

  The other 5 are **schema-convention mismatches, not corruption**. `receipts/` is heterogeneous:

  | Schema | Example | Failure reported |
  |---|---|---|
  | `forward_record_preregistration.v1` | `forward_record_preregistration_v1.json` | `receipt_sha256_missing_or_invalid` (it uses `prereg_seal`, verified separately) |
  | `lane_receipt.v1` | `cost_aware_rerun_20261007.json` | `receipt_sha256_missing_or_invalid` (its own seal) |
  | `basis_carry.v1` | `basis_carry_…json` | `tape_manifest_unknown` |
  | `receipt.v2` | `crossvenue_basis_…json` | `tape_manifest_unknown` |
  | `changepoint_localize.v1` ×2 | `cp_real_drill_*_pitball.json` | `missing_params` |
  | `mid_dark.v1` | `mid_dark_amzn.json` | `research_only_not_true` |

  `verify-receipt` applies one canonical research-receipt shape (`kind`, `verdict`, `digest_convention`, `receipt_sha256`) and reports `kind: null, verdict: null` for the rest. The Makefile already says so:

  ```
  receipts-reverify: ## Fail-closed audit; schema-specific committed receipt verifiers pending
  ```

  **So row 6 cannot pass until schema-specific verifiers land.** It is currently wired into the scorecard as if it were a working control, which overstates the evidence base.

- **NEW — `make receipt-integrity-scan`: the digest half of row 6, in seconds.** The 20-minute gate buries its own findings, so the integrity half is now available cheaply:

  ```bash
  make receipt-integrity-scan      # scripts/receipt_integrity_scan.py
  ```

  `scripts/receipt_integrity_scan.py` recomputes every receipt's self-excluding digest **in-process** and classifies each file by cause — `ok` / `mismatch` / `no_selfseal` / `unparseable`. **3.7 s versus ~20 minutes** for the same population, and it separates "false integrity claim" from "different schema's seal," which the slow gate conflates.

  Deliberately scoped and documented as a **digest scanner, not a full verifier**: it does not check tape manifests, params, research-only flags, or live-PnL claims — that stays `verify-receipt`'s job. It **fails closed** (exit 1 on any mismatch; verified against both a clean and a tampered fixture) and treats `no_selfseal` as *not* a failure, because counting a foreign schema's seal as corruption is the exact false positive this tool eliminates.

  Ten tests pin the behaviour (`tests/unit/test_receipt_integrity_scan.py`), including that a silent body edit breaks the seal, that sidecars are never collected as receipts, that the `ensure_ascii=True` convention is treated as **intact rather than corrupt**, that genuine drift still fails under *both* conventions, and that scanning the real `receipts/` is deterministic.

  **Current verdict: FAIL — 384 ok, 2 ok-alt-convention, 1 mismatch, 2 no-selfseal.** The single mismatch (`calib_real_drill.json`) is the actionable item. Either write the four schema verifiers or demote row 6 to "known-pending" — but do not leave it looking green-capable.

- **A drift the guards caught — and a fix.** Editing `.github/workflows/proofcore.yml` broke `test_staged_workflow_matches_activated_copy`: `docs/proofcore/proofcore.yml` is a reference copy that must stay byte-identical to the active workflow below its 3-line activation header, and the workflow edit was not mirrored. Caught by running the suite and fixed by mirroring; all 22 tests in `tests/unit/test_proofcore_ci.py` now pass. **Worth noting: this guard did its job** — and it means any workflow edit under `.github/workflows/` should mirror into `docs/proofcore/proofcore.yml` if that is the affected file.
- **Epoch re-stamp.** `.github/workflows/` fails epoch verification until `make stamp-epochs` → `make sign-pins` → `make anchor-pins` runs. Needs `GATE_SIGNING_KEY`.
- **Phase 5 segfault.** Not yet attempted: `make test` under `-n auto`. The low-pressure reproduction needs a quiet box.

### Deliberately not done, and why

| Item | Reason |
|---|---|
| Untrack the 911 `.dsh-24x7` files | A 911-file deletion on the already-314-commit-diverged local spine would deepen the fork before the owner rules on which side is canonical. Prepared for the reconciliation commit. |
| `stamp-epochs` / `sign-pins` / `anchor-pins` | Requires `GATE_SIGNING_KEY`; must run in strict order after the history decision, since every `stamp-epochs` stales existing anchors. |
| Tag `v0.4.0-rc.1`, release, push | Irreversible and externally visible. Owner decision. |
| Merge/rebase the 314/14 divergence | Rewrites history and invalidates receipt pins. Owner decision on the canonical spine. |
| Prune 231 local branches | Mostly `devin/*`, `kimi/*`, `codex/*`; safe but belongs with the reconciliation. |
| Remove `tests/tests/` | **Not a defect** — known sftp-clobber artifact, already ignored (`.gitignore:102`), recreated by the clobber. |

### Corrections made to this plan during implementation

1. **F2 downgraded from Blocker to High and reframed.** The "backdated pre-registration" reading was wrong — `splits.forward` gates the first decision on the recorded freeze, so the receipt is sound. The real defect was the missing external anchor, now fixed. §2.2.
2. **The 32 `corpus_epoch_*.json` in `.github/workflows/` are not litter.** They are the workflows corpus's own epoch records; editing any `.yml` invalidates them. §2.4, Correction 1.
3. **`tests/tests/` is not a stray mirror.** Known sftp-clobber artifact, already handled. §2.4, Correction 2.
4. **CI starvation is now quantified, not estimated:** GitHub-hosted Free caps at **20 concurrent jobs** and **5 concurrent macOS jobs** against ~80 jobs per push. §2.1.
5. **Push cadence gets a structural fix, not a manual one** — a merge queue with bounded build concurrency and merge limits. Phase 0 step 4.
6. **"fx-1 has no weights on disk" was wrong.** `artifacts/fx1_tiny_lm/weights.safetensors` (127 KB) **is tracked in git.** Its own modelcard calls it a *"fixture-scale byte-level LM (~31k params) … not an fx-1 release candidate"* whose `eval_delta` values are *"synthetic ship-gate placeholders, not measured evals."* No capability is over-claimed, but the flat "no weights" wording was false and README's former "No model weights are in the tree" was worse. §2.6 now states the narrower, correct position. **Worth owner attention:** that card carries bare numbers (`0.5 → 0.51`) that read like scores even though they are disclaimed.
7. **`git tag` went 0 → 4 mid-audit** — the four are `attic/receipt-provenance/*` anchors, not releases, so `git tag -l 'v*'` is still `0`. This is a live demonstration of the missing drift checker: a number in the docs went stale inside a single working session because another lane was writing.
8. **A committed receipt fails verification.** `dipcatcher verify-research` reports `families_unknown: copula_specification, selective_inference, spa_test`. Verified **pre-existing** — those names appear nowhere in `src/` or `receipts/`, and are not in `OPTIONAL`, `REQUIRED`, or `RETIRED`. The fail-closed contract is behaving correctly, but scorecard row 6 does not currently pass.