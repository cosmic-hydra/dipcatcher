# legacy-unsealed/

Artifacts committed here predate a verifiable sealed contract for their
schema, or bind tape digests no committed `tape_manifest.v1` attests.

They are retained for provenance — content is unchanged, git history
intact — but **they cannot be re-verified** under `verify-receipt` /
`receipts-reverify`. Classes:

- **Pre-seal / sidecar-only** — no top-level `receipt_sha256` (bench
  results and the frozen forward-record preregistration + its
  `prereg_seal.v1` sidecar).
- **Pre-deep-contract drills** — sealed `changepoint_localize.v1`
  receipts written before the kind required a `params` block
  (`missing_params`).
- **Tape-legacy** — sealed receipts whose `dataset_sha256` /
  `tape_manifest_sha256` binds provider tapes (kraken / kraken+okx) that
  are not attested by any committed manifest (`tape_manifest_unknown`).
  Return to `receipts/` once those tapes are pinned.

Because of that, this directory is deliberately **outside** the
`receipts/*.json` glob used by `receipts-reverify` and `verify-all`: the
sealed set must be end-to-end re-verifiable, fail-closed. If any of these
experiments are re-run under a sealed, contract-conformant writer, the
new receipt belongs in `receipts/` proper.

Quarantine is itself pinned, not assumed: `quality/legacy_quarantine.json`
records each file's sha256 plus its exact `verify-receipt` error set, and
`tests/unit/research/test_legacy_quarantine.py` fails if either drifts.
