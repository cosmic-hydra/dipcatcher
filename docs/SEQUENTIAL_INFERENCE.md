# Sequential inference suite

The repo's research lanes all terminate in receipts; this family of
modules makes the *act of reading them over time* statistically honest.
Batch tests rerun on accumulating evidence silently inflate the
false-positive rate — every module here is built on **e-processes** or
**alpha-investing**, which stay valid under arbitrary stopping.

## The e-process core

`research/evalues.py` — `LossEProcess`: a betting e-process over a
per-origin proper-loss difference stream (challenger minus incumbent).
`E_t = Π(1 − λ·g_i)` with clipped predictable bets is a non-negative
test martingale, so Ville's inequality gives `P(ever E ≥ 1/α) ≤ α` —
a promotion decision may be taken at the first origin where `E` crosses
`1/α` without any correction for having peeked. `promotion_report()`
emits `evalue_promotion.v1` (deep-verified by `verify-receipt`).

## Race: sequential elimination

`research/fleet_race.py` + `dipcatcher race` — runs *two* e-processes per
head against a fixed chunk-0 incumbent: a promote process (head beats
incumbent) and a demote process (incumbent beats head). Heads are
eliminated or promoted with anytime-valid evidence; the winner is the
first promoted head or the last survivor (`verdict: anytime|final_mean`).
Sealed `fleet_race.v1` receipts.

## Corpus-level and online multiple testing

- `research/corpus_inference.py` — `corpus_audit` harvests every p-value
  and e-value committed under `receipts/`, pools the p-values under one
  BH-FDR bound, and merges e-values by product (Shafer's rule). Answers:
  *which artifact claims survive family-level correction?*
- `research/online_fdr.py` — Foster–Stine alpha-investing (LORD): a
  wealth-budget controller over an *ordered stream* of tests, so each
  new receipt's findings are tested as they arrive with mFDR ≤ level at
  every arrival time — no pool-and-rerun.

## Selection bias and stability

- `research/winner_curse.py` — the tournament winner's reported score is
  optimistically biased by the act of selection. Paired-bootstrap
  optimism correction + split-half honest control + selection-aware
  CIs; emits `winner_curse.v1`.
- `research/drift_alarm.py` — `EProcessDriftAlarm` fires on a *level
  shift* in a loss stream with P(ever false-alarming) ≤ α (a head whose
  persistent level exists from step 1 is absorbed into its baseline —
  the contract is change detection, not badness detection).
  `PageHinkleyAlarm` rides alongside as the classical CUSUM diagnostic.
  Emits `drift_alarm.v1`.

## The composite verdict

`research/honest_verdict.py` — `honest_verdict(scores)` composes all
three lenses into one sealed `honest_verdict.v1` claim:

| verdict | meaning |
|---|---|
| `confirmed` | bias corrected, e-process promoted at α, no drift |
| `supported_with_caveats` | edge holds; evidence not yet decisive or PH-only alarm |
| `not_supported` | corrected score erases the edge, or drift alarm fired |
| `inconclusive` | inputs missing/mismatched or a component lane unavailable |

Components are lazily imported and degrade gracefully — a missing lane
is named in `unavailable_lanes` and forces `inconclusive`, never a
silent pass.

## Running the suite

`research/verdict_run.py` replays the fleet tournament keeping the
per-origin streams the aggregates discard, then composites the verdict —
`dipcatcher verdict` writes a sealed `honest_verdict.v1` receipt.

`research/monitor_run.py` (`dipcatcher fleet-monitor`) streams every
(shard, head) cell through the whole anytime-valid family — coverage
breach rate, nested tail depth, PIT calibration, conformal
exchangeability, loss drift vs the fleet median — and pools the lane
e-values per cell via `emerge_mean` (valid under arbitrary dependence).
Lanes not yet merged report `lane_missing` on the `monitor_run.v1`
receipt rather than failing silently.

`research/lane_power.py` (`dipcatcher lane-power`) is the suite's own
capability bench: measured alarm rate and time-to-alarm per lane per
injected defect size, with the defect=0 row bounding the false-alarm
rate by α — the sequential claim is only useful if the lanes actually
fire.

`research/suite_health.py` (`dipcatcher suite-health`) audits the whole
evidence trail in one pass: every receipt under `--receipts-dir` is
re-verified (seal + kind contract), harvestable p-values/e-values are
pooled via `emerge_mean` (valid under the arbitrary dependence between
receipts), and a single corrupt artifact withholds the pooled claim —
the `suite_health.v1` receipt is itself sealed and verifiable.

## Extended lane roster

The core trio (coverage / drift / verdict) is joined by lanes that each
attack a defect class the others are structurally blind to. All emit
sealed receipts and all are fail-closed on missing or malformed
observations.

- `research/coverage_watch.py` — anytime-valid audit of the *nominal
  coverage rate*: an e-process over the breach indicator stream
  `1{y_t < q_lo or y_t > q_hi}` against the exact binomial null. Catches
  under-coverage (intervals too thin) and over-coverage equally;
  latched `alarmed` flag is the ever-crossed semantics — Ville's bound
  is about the supremum, so a decayed final e-value does not un-fire
  the alarm.
- `research/coverage_cs.py` — the confidence-sequence twin: maintains a
  time-uniform confidence interval on the breach probability itself,
  so the answer is an interval that excludes the nominal rate *from
  some origin onward*, not just a point alarm.
- `research/tail_watch.py` — nested-quantile consistency: among the
  `τ`-breaches, exactly `τ'/τ` must also breach the deeper `τ'`
  quantile under *any* correct conditional tail — shape-free. Catches
  the failure coverage lanes are blind to: the right breach *rate*
  with the wrong tail *depth* (e.g. a head whose intervals breach at
  the nominal 20% but whose breaches land 73% deep instead of 50%).
- `research/calibration_eprocess.py` — e-process over the PIT
  histogram: any systematic non-uniformity of the probability
  integral transform is a calibration violation; the lane uses GRAPA
  adaptive bets (predictable `λ_t` fitted from past PITs), which stays
  valid under serial dependence and out-grows fixed bets on biased
  streams.
- `research/conformal_monitor.py` — Vovk's conformal martingale on the
  exchangeability of per-origin PIT values: the only lane whose null
  is *the whole predictive law*, not a summary statistic.
- `research/serial_watch.py` — independence audit: PITs can be uniform
  (calibration-clean) yet serially dependent (clusters of high/low
  PITs) — a defect any level-wise lane misses. The lane bets on the
  lag-1 structure of the uniform PIT stream; null = iid Uniform(0,1).
- `research/changepoint_localize.py` — scan for *where* the stream
  shifted: per-candidate-split e-values with a Bonferroni correction
  over overlapping windows (the un-corrected version alarmed on 25%
  of stationary streams; corrected: 0/40 with detection preserved).
- `research/quantile_ladder.py` — multi-level calibration family: a
  per-level e-process on the breach indicator at every
  `τ ∈ {0.05,…,0.95}` merged by the dependence-robust mean (valid
  under arbitrary cross-level dependence); the per-level family is
  reported separately at the Bonferroni threshold — a head can pass
  every single-level audit yet fail the ladder (mid too wide, tails
  too thin, cancelling out).
- `research/xwatch.py` — cross-head lead e-process: whether head A's
  per-origin loss *leads* head B's — catches informational
  look-ahead/clone defects between fleet heads.
- `research/mcs_seq.py` — `AnytimeMCS`: a sequential model confidence
  set. Hansen's MCS is a fixed-sample batch test; this maintains a
  survivor set that contains an optimal head with probability ≥ 1−α
  *uniformly over time* via pairwise e-processes plus a union bound,
  with permanent elimination — publishable-grade machinery.
- `scripts/panel_audit_drill.py` — cross-sectional coverage e-process:
  pools the breach stream across the panel dimension (24-symbol real
  tape), isolating a systematic *across-instrument* calibration miss
  from per-name noise.
- `research/emerge.py` — the valid pooling toolbox: arithmetic mean
  and harmonic mean of e-values (each valid under arbitrary
  dependence), product under independence, Bonferroni and
  Simes-with-`H_k` for p-values, and the p↔e calibrators.
  `lane_power.py` rides on it for the suite's capability bench.

## Per-lane evidence ledger (real tape)

Each lane has a sealed `yahoo_eod` drill receipt committed under
`receipts/` — the 2,693-bar NVDA Yahoo EOD return series through the
full fleet. The convergent verdict across independent lanes:

| lane | real-tape verdict | evidence |
|---|---|---|
| coverage_watch | 10/12 heads alarm; 80% intervals breach at 33–48% | fhs_skew e≈1.8e23 |
| coverage_cs | breach-rate CS exits the nominal band at origins ~47–56 | same heads |
| tail_watch | fhs_skew breaches land 73% deep vs exact 50% | e=220 |
| calibration (PIT) | all 12 heads non-uniform | fhs_skew e≈2.6e22 |
| conformal_monitor | gaussian_pit alarmed at origin 55 | pooled conformal e |
| mcs_seq | 9/12 heads eliminated; survivors {fhs_skew, hstep_t, qar} | pairwise e-processes |
| panel_audit | all 12 heads under-cover across 24 symbols (36–49% breach) | fhs_skew pooled e≈2.8e47 |
| changepoint | τ̂=34 (regime onset) + winner-gap τ̂=220 | scan-corrected |
| drift_alarm | correctly silent on persistent-from-start defects | level-shift contract |
| loss_cs | conf_t beats empirical; CS excludes zero | first confirmed positive |
| honest_verdict | composite = `not_supported` | all lanes composite |

The cross-lane scientific point: `fhs_skew` *wins* the pinball MCS
while being the worst-calibrated head — a score-optimal model is not a
trusted model. That is the claim the suite exists to make honestly.

### Self-drill evidence

`receipts/monitor_run_drill_clean.json` and
`receipts/monitor_run_drill_defect.json` are committed `monitor_run.v1`
receipts from a six-lane live pass over `iid_gaussian`/`regime_switch`
shards. Clean run: the only alarm is `qar` on `regime_switch` — a real
calibration miss the union drill detected (pooled e ≈ 1.7×10⁵). Defect
run: an injected biased quantile head fires coverage, tail, and
calibration on both shards (pooled e ≈ 9.9×10¹⁶) while drift correctly
stays silent on `regime_switch` — a persistent-from-start defect is
absorbed by the level-shift contract, not an alarm.

`receipts/lane_power_drill.json` is the suite's measured power curve:
nine lanes × defect ∈ {0, 0.5, 1.0} × 8 seeds at n=200. Every lane's
defect-0 alarm rate is 0/8 (within the α bound); calibration,
changepoint, loss-CS and promotion all fire at defect 0.5, and the
slower Bernoulli lanes (coverage, tail, conformal, drift) fire by
defect 1.0 — the conservative-null / real-power asymmetry is the
signature of valid e-processes, not a bug.

Two failure modes the drill surfaced and fixed:

- `changepoint_localize` alarmed on 25% of *stationary* streams — the
  per-split e-value threshold was applied to a max over ~75 candidate
  splits. The threshold is now scan-corrected (`log(n_candidates/α)`,
  Bonferroni over the overlapping windows): null alarm 0/40, detection
  of a planted 1.5σ shift preserved.
- All three Bernoulli lanes (`coverage_watch`, `coverage_cs`,
  `tail_watch`) coerced `None`/`2`/`NaN` into a non-breach — a missing
  observation silently deflated the measured breach rate. They now take
  a strict {bool, 0/1 int} flag and raise on anything else.

`test_committed_sealed_receipts_reject_tampering` mutation-tests the
committed corpus itself: every sealed `receipts/*.json` is perturbed
five ways (seal flip, seal removal, kind/schema rename, claim-digit
edit, top-level drop) and must come back invalid — zero survive.

`receipts/monitor_run_real_drill.json` is the suite's first non-synthetic
evidence: the six-lane monitor over 2,693-day Yahoo EOD return series
(8 symbols × 8 fleet heads, sealed with `data_label: yahoo_eod` — the
receipt label is derived from shard configs, never hard-coded). The
lanes discriminate correctly on real data: the calibration e-process
fires on nearly every (symbol, unconditional-head) cell — a static
512-day window cannot track vol clustering — while the conformal
martingale never alarms and the drift lane separates only the heads
that degrade relative to the fleet median.

`receipts/honest_verdict_real_drill.json` carries the composite verdict
on the same tape (`scripts/verdict_real_drill.py`): **not_supported**.
On real fat-tailed returns the gaussian head wins the raw tournament,
but the promotion e-process finds no evidence it beats the empirical
runner-up (E ≈ 0), calibration fires nearly everywhere, and Page–Hinkley
flags the regime structure the drift e-process correctly cannot confirm —
a genuinely honest negative claim, sealed. The bench inputs now carry the
same provenance gate: `SyntheticBook`/`CrossSectionalPanel` constructors
must declare `data_label`, and a mixed corpus fails closed rather than
inheriting a hard-coded SYNTHETIC stamp.

### Tape manifests (`tape_manifest.v1`)

Real-tape receipts used to attest their input only by self-declared
digest (`inputs_sha256`/`dataset_sha256`) — nothing committed proved
which bytes a digest names. `dipcatcher tape-pin` closes that gap: it
writes a sealed `data/manifests/<source_label>.json` recording each tape
file's sha256 and byte count, the frame's row/name/window profile, and
`frame_csv_sha256` — `hash_bytes(frame.write_csv().encode())`, the exact
bytes a lane seals as `inputs_sha256`. The tapes themselves stay
gitignored; `dipcatcher tape-verify` re-hashes them and fails closed on
any drift. `verify-receipt` consults the same registry: a receipt
declaring a non-synthetic `data_label` that binds
`tape_manifest_sha256`/`dataset_sha256` must resolve to a committed
manifest's seal or tape digest, else `tape_manifest_unknown`. Receipts
without a declared binding stay admissible — the check ratchets in, it
does not retroactively seal old evidence. Pinned today:
`yahoo_eod` (5-name 2024 collect, promoted under
`data/raw/sources/yahoo_eod/`) and `yahoo_eod_us_wide` (the 424-name
2016–2026 tape behind the `*_real_drill` receipts — its tape sha256 is
the `dataset_sha256` `e22bf3…` the real_benchmark manifest cites).

## Verifier coverage

Every kind above has a contract check in `research/evalue_contracts.py`
dispatched by `verify-receipt` on the `kind` tag — each embedded claim
is re-derived from the payload (e.g. `anytime_p == min(1, 1/E)`,
`corpus_reject_at_alpha ⇔ E ≥ 1/q`), so a tampered verdict fails
closed even under a valid reseal.

## Reproduction

`verify-receipt` re-derives claims but never re-runs a lane. Replay-declared
receipts carry a `replay` manifest `{argv, artifacts: [{path, sha256}], cwd?}`;
`dipcatcher replay <receipt>` re-executes argv under the repo root, re-hashes
each declared artifact's raw bytes, and seals a `replay_proof.v1` receipt —
fail closed on non-zero exit, timeout, missing artifact, or any digest
mismatch. Coverage today:

| lane | manifest receipt | replayable? |
|---|---|---|
| serial_watch | `serial_watch_78dd891261e2aae9.json` | yes — PIT fixture + argv are byte-deterministic |
| fleet_eval (incl. hstep heads `hstep_t`/`hstep_emp`) | `fleet_eval_cee9aaee0a0f0f36.json` | yes — seeded shards and heads |
| vol_bench | `vol_bench_860a3412278105ca.json` | yes — seeded vol shards and forecasters |
| cross_sectional_rankic | `rankic_eval_b0f9818dfe2b3663.json` | yes — seeded panels and challengers |
| capacity_overlay | `capacity_eval_bdf234110827cd1f.json` | yes — seeded books and AUM ladder |
| monitor_run (coverage_watch, tail_watch, drift_alarm, calibration, conformal, emerge) | `monitor_run_85aece35e89b5b12.json` | yes — seeded tournament; the six monitor lanes are streams inside one receipt |
| honest_verdict | `honest_verdict_d5d9bf60e0d5e1a4.json` | yes — seeded tournament + bootstrap |

The mechanism that makes these replayable is a provenance convention: lane
reports keep wall-clock/revision stamps (`generated_at`, `git_revision`,
`code_revision`) under a top-level `meta` block that writers strip before
sealing — the seal covers every byte written, so the argv-produced artifact
is byte-identical across checkouts, and the `receipt.v2` envelope still
stamps `meta` on wrap. Lanes whose receipt body is not byte-deterministic
(e.g. live-tape lanes binding wall-clock inputs or `data/` files that are
regenerated, never committed) cannot be covered by replay — their byte
identity is the committed file digest, and `verify-receipt` remains the
integrity check. `coverage_cs`, `mcs_seq`, `panel_audit`, `changepoint`,
`loss_cs`, `calibration`, and `conformal_monitor` as standalone CLIs fall in
this group today: they either have no standalone argv (they are sub-lanes of
`fleet-monitor`, covered transitively by the monitor manifest) or their
evidence receipts bind non-deterministic inputs.

## References

Ville (1939); Ramdas, Ruf, Larsson, Koning (2020) betting e-processes;
Waudby-Smith & Ramdas (2024) anytime-valid confidence sequences;
Javanmard & Montanari (2018) LORD; Foster & Stine (2008) alpha-investing;
Fithian, Sun & Taylor (2014) post-selection inference.
