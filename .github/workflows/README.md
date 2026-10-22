# Workflow files are epoch-chained

Every `*.yml` here is a stamped member of the corpus-epoch integrity chain
(`quality/epoch_heads.json` pins the head). After adding/editing/removing a
workflow, re-stamp: `make stamp-epochs` (or `dipcatcher corpus-epoch
--corpus-dir .github/workflows --glob '*.yml' --out-dir
.github/workflows --heads-pin quality/epoch_heads.json`).
`make evidence-audit` fails on `head_member_digest_drift` /
`head_member_missing_live` until the stamp is committed alongside the
workflow change. The `corpus_epoch_*.json` file in this directory is the
chain head — do not delete it.

## Lean-run policy (CI pipeline health)

- **Gate workflows** (`ci.yml`, `fx1.yml`, `proofcore.yml`): cancel superseded
  *pull_request* runs only; main / merge_group / schedule always drain.
- **Platform matrix** stays `schedule` + `workflow_dispatch` (never push/PR).
- **Scorecard** is weekly + main push (no PR). **CodeQL** is PR + weekly
  (no push-to-main fan-out).
- **Path filters** on arch-guards, atlas, replay_viz, web_explorer, witness,
  reproduce_sota, and dependency-audit so unrelated pushes skip those lanes.
- **PR test matrix** in `ci.yml` is Python 3.12 × 4 shards; push/schedule keep
  3.12+3.13. Full OS×Python coverage is nightly `matrix.yml`.

