# Evidence integrity

How dipcatcher keeps the evidence corpus tamper-evident — the layers, what each
detects, and the boundary where detection stops.

## The layers

| Layer | Artifact | Command | Detects |
|---|---|---|---|
| Seal | `receipt_sha256` on each receipt body | `dipcatcher verify-receipt` | Content mutation of a single receipt |
| Contract | per-kind deep checks | `verify-receipt` dispatch | Forged aggregates, renamed kinds, inconsistent claims |
| Lattice | claim groups across receipts | `dipcatcher lattice --strict` | Two receipts contradicting each other on the same claim |
| Provenance DAG | `receipt_graph` | — | Orphaned/fabricated citation edges |
| Epoch chain | `corpus_epoch_*.json` per (dir, pattern) | `dipcatcher corpus-epoch --check` | Receipt deletion, reorder, unstamped arrivals, dishonest deltas |
| Head pins | `quality/epoch_heads.json` | `--heads-pin` on check | Rewind attack: deleting the newest epoch to hide later tamper |
| Crown jewels | `quality/crown_jewels.json` | `dipcatcher crown-jewels --check` | Silent edits to the files that *define* the gates (Makefile, pyproject, uv.lock, hooks, gitleaks, conftests, mkdocs, AGENTS.md) **and the verifier's own source** — a `return ok` patch to corpus_epoch/receipt_v2/lane_contracts/etc. fails `jewel_mutated` |
| Pin signature | `gate_pins.sig` + `quality/gate_signing.pub` | `verify-repo` (`pin_signatures` gate) | Forged pins: an editor can re-pin after rewriting a gate file, but cannot re-sign — Ed25519 covers both pin files' bytes |
| Timestamp anchor | `quality/timestamps/*.tsr` + `anchors.json` | `verify-repo` (`timestamp_anchors` gate) | Retroactive history fabrication: an RFC 3161 TSA signature proves the pinned state existed by wall-clock time T — the chain can't be minted after the fact |
| Bitcoin anchor | `quality/timestamps/ots/*.ots` + `*.hdr` + `*.blk` + `ots_anchors.json` | `dipcatcher ots-stamp` / `ots-upgrade [--full]` / `verify-repo` (`timestamp_anchors` gate) | A second, independent trust root: `ots-stamp` POSTs the epoch-heads sha256 (never its contents) to the public OpenTimestamps calendar pool and commits the merged detached proof — `pending:<uri>` proves calendar submission immediately. `ots-upgrade` polls each anchor's own calendars for the confirmed timestamp and commits the claimed 80-byte block header, after which verify runs a pure sha256d<nBits PoW check (`bitcoin:<height>:pow_verified`). `--full` additionally commits the block's txid list + raw coinbase (`*.blk`), and verify then reaches `fully_verified`: coinbase txid = txids[0], the recomputed block merkle root equals `header[36:68]`, and the attestation's committed digest sits in a coinbase OP_RETURN push — end-to-end leaf→block with zero explorer trust. `scripts/verify_ots_auditor.py` is a dependency-free independent reimplementation (differentially pinned against the library in tests); a proof stale to the current target reports `fresh=false`/`[stale]` honestly — a timestamp of superseded bytes is still valid evidence of *those* bytes |
| Inclusion proof | `corpus_proof_*.json` (RFC 6962 merkle path) | `dipcatcher corpus-proof --check` | "Was this file in epoch N?" in O(log n) — the path recomputes the epoch's Merkle root and binds to the named chained receipt. `member_tree_root` is sealed into the epoch payload and mirrored into the heads pin, so `--check --pin epoch_heads.json` verifies offline — the quorum-signed, OTS-anchored pin is the whole trust root. `leaf_index` is shape-bound: the side sequence is a pure function of (index, n_members), so a forged index cannot recompute the pinned root |
| Absence proof | `corpus_absence.v1` sealed receipts (bounding neighbors) | `dipcatcher corpus-absence --member X`; `--check` / `--check --pin` | "Was this name NOT committed at epoch N?" — the two sorted neighbors bracketing the gap carry shape-bound inclusion paths whose leaf_indexes are provably adjacent (`hi == lo + 1`), so no member could sit between them. Verified offline against the heads pin or live via `verify_epoch_absence`; standalone oracle: `scripts/verify_corpus_proof.py --absence` |
| Member-name authenticity | enforced inside `corpus_epoch` / `epoch_merkle` | automatic on every stamp + verify | Filesystem-semantic attacks: NFD spellings alias NFC members on APFS (`member_name_not_nfc`), distinct names colliding under casefold can't coexist on macOS/Windows checkouts (`member_name_alias`), symlink members hash their *target's* bytes (`member_is_symlink`), and control/format/line-separator codepoints can inject fake verdict lines or bidi-rename members (`member_name_not_portable`). Stamp refuses all four; the live chain flags them. Every diagnostic label that interpolates a filename carries its ASCII repr, so even an unstamped hostile name can't inject output |
| Authenticated epoch loading | `_load_epoch_receipt` seal-recompute + `corpus_epoch_<sha16(seal)>` name binding | inside `verify_epoch_proof` / `verify_epoch_absence` | Forged or renamed epoch receipts presented to live proof verification: the receipt body must re-seal (`epoch_receipt_tampered`) and its filename must be the one its seal dictates (`epoch_receipt_name_mismatch`), v2 envelopes unwrapped. `check_epoch_chain` additionally contract-verifies every committed epoch (`epoch_receipt_invalid:<name>`) |
| Consistency proof | `epoch_consistency.v1` (hop digest list) | `dipcatcher corpus-consistency --check [--held]` | History rewrite: proves the live chain *extends* an old head digest a verifier already holds (anchored pin, earlier clone) — a rewritten chain can't reach the held bytes without keeping every real hop verbatim |
| Verifier coverage | derived jewel set | inside `crown-jewels --check` | A new integrity-critical module (`*_epoch*`, `*merkle*`, `*anchor*`, …) that dodges `DEFAULT_JEWELS` fails `verifier_unpinned` — the pin can't be dodged by omission |
| Checkpoint | `quality/checkpoint.json` (signed, anchored) | `dipcatcher verify-checkpoint` | Portable signed tree-head: an auditor needs only checkpoint + pubkey + its TSA token — signature verifies offline, pinned digests bind a clone. `current` flags staleness without invalidating authenticity |
| Checkpoint spine | `quality/checkpoints/*.json` — every historical checkpoint, each carrying `prev_sha256` | `dipcatcher checkpoint-chain` | Pin-state history as a *chain*, not a point: deleting an archive record → `dangling_prev`, injecting a side-chain → `spine_fork`+`spine_orphan`, a re-signed record → `signature_invalid`, Rekor-witnessed digests that vanished → `witnessed_absent`. Terminus resolvable only in Rekor (pre-archive era) is a valid `witness` genesis |
| Transparency log | `quality/witness/*.json` (Rekor hashedrekord entry) | `dipcatcher verify-witness` (offline), `verify-witness --online` (live consistency: committed tree is a prefix of Rekor's *current* signed head) | External non-repudiation: the checkpoint digest + our ECDSA witness sig sit in sigstore's public append-only log — a state *we* can't rewrite either. Proof self-verifies: RFC 6962 inclusion walk to Rekor's signed root + signed-entry-timestamp + checkpoint-note sig under the pinned Rekor pubkey. Proofs minted before the archive era bound `witnessed_absent` vs `grandfathered_witness` (pre-retention history) |
| Auditor bundle | `auditor_bundle.v1` (one JSON doc) | `dipcatcher witness-bundle` / `verify-bundle <file>` | Zero-trust third-party audit: bundle carries checkpoint + pins + pubkeys + **every archived checkpoint and every committed witness proof** — the auditor replays the whole spine (links, signatures, forks, orphans, Rekor order), not just the head. The witness key is authenticated by the **log itself** — Rekor's entry body records the signer pubkey, which must equal the bundled `witness_signing.pub` byte-for-byte. `scripts/verify_auditor_bundle.py` is an independent second implementation (same verdict contract; members outside the pinned prefixes fail `unexpected_member`). No repo access, no trusted inputs beyond Rekor's own key. The bundle also carries `auditor_self_sha256` — the sha256 of the standalone script at build time — so an auditor can diff their local copy before trusting its verdict; a mutated script warns `auditor_self_mismatch` |
| Tamper drill | `tamper_drill.v1` | `dipcatcher tamper-drill` | Self-adversarial: clones the real tree, lands every probe mutation (epoch/merkle/anchor/sig/checkpoint/spine/key classes), and requires `verify-repo` to flag **every** one — a verifier that went blind reports `missed:` instead of `ok` |
| Fuzz drill | `fuzz_drill.v1` | `dipcatcher fuzz-drill --seed N` | Metamorphic two-sided fuzzing of the substrate itself: seeded byte flips/truncations/key-renames across corpus members, jewels, pins, spine records, witness proofs, and checkpoints — plus benign probes (`docs/` write) that must NOT alarm. Grades `escaped`/`false_positive`; a `baseline_dirty` verdict refuses to drill on an already-drifted tree. First campaign: 4 real escapes found and closed (spine extent, checkpoint-absent fail-open, unstamped injection, witness-proof delete/byte-drift) — 6 seeds × 17 mutations (incl. crafted pin-rollback / forged-epoch / removal-laundering / foreign-extension-drop policy attacks) now fully `calibrated` |
| Receipt fuzz | `receipt_fuzz.v1` | `dipcatcher fuzz-receipts --seed N` | Forge-and-reseal: mutates one claim leaf inside each committed receipt, then re-seals *honestly* (sha256 seals are integrity, not authenticity — a self-consistent forgery is free to mint). Maps the verification boundary: escapes = claim fields that are seal-bound but not contract re-derived (mostly `*_real_drill` result fields whose source streams aren't committed — they are *attested*, not *verified*). Deliberately a measurement lane, not a gate — the `replay_proof` lane closes the gap by pinning inputs and re-executing |
| Key rotation | `rotation_*.json` (`key_rotation.v1`) | `dipcatcher verify-rotations` | Key substitution without authorization: each link is dual-signed — the outgoing key proves *authorization*, the incoming proves *possession* — and the first link's `old_pubkey` must re-verify a real spine signature (`unanchored_genesis` otherwise). The spine walks a **keyring** — live key + every authorized retired key — so records verify under their era's key (`signature_key_unknown` for a never-authorized signer). Live `gate_signing.pub` must equal the chain terminus |
| Key reset | `key_reset_*.json` (`key_reset.v1`) | `dipcatcher key-reset` records it; `checkpoint-chain`, `verify-rotations` and `verify-repo` report it; `verify-bundle` and `scripts/verify_auditor_bundle.py` re-verify it from the bundle | Lost keys — the one key-set change the outgoing key cannot sign. The record embeds the last checkpoint signed under the lost set (the **boundary**, which must re-verify under that set), names the new gate key + registry (+ witness key), and is signed by the new registry's threshold including the new gate key (**possession**). On the spine the era may change only out of the boundary (`key_reset_off_boundary`); after it no lost key signs and no record claims the lost registry or any later registry that holds a lost key (`key_reset_revoked_signer` / `key_reset_revoked_registry`); no new key signed anything up to it (`key_reset_resurrects`); a witness key declared lost is never installed again, by a later reset or by hand (`key_reset_reinstates_witness`). Records are strict JSON: a NaN or Infinity makes them `malformed`. Lineage verdicts read `reset`, never `intact`, and `checkpoint-chain` lists every reset in `notes`. An invalid record contributes no key — the new era it names stays unauthorized (fail closed) |
| Custody | `custody_proof.v1` (one JSON doc) | `dipcatcher custody --member X --out bundle.json`; `custody --check bundle.json --member-file F`; `custody --member X --timeline` | Per-member provenance with **zero repo access**: the bundle embeds the member's earliest-chain pinning epoch (proof of first committed state), the epoch→head hop files raw (digests re-derived on verify), inclusion merkle path, signed pins, checkpoint, both witness pubkeys, every committed Rekor proof, and the RFC 3161 timestamp dir — materialized into a synthetic root so the existing file verifiers run unmodified. Eight layers: member→inclusion→chain→pins→signature→checkpoint→witness→timestamps. `--timeline` enumerates a member's committed digests across epochs (byte-evolution lineage for mutable corpora) |
| Admission | `receipt_admission.v1` | `dipcatcher admit-batch --strict` | A new receipt that breaks lattice/FDR on entry |
| Retraction | `receipt_tombstone.v1` (`tombstone_<sha16>.json`) | `dipcatcher tombstone <receipt> --reason ...` | The corpus is append-only — a wrong/superseded receipt can't be deleted (epoch flags removal, lattice flags drift), so it is retracted: a sealed record pinned to the target's bytes excludes its claims from the lattice (`retracted:` block), honors partial `scope` per claim path, and fails closed if the target drifts (`tombstone_target_drift`) or the tombstone is unsealed |
| Release attestation | `release_attestation.v1` (one JSON doc) | `dipcatcher attest-release --artifact W [--witness]` / `verify-release A --artifact W` | Extends the proof to bytes that live *outside* the repo (wheels, sdists, exported bundles): an Ed25519-signed binding of each artifact's sha256+size to the live pin-state digests (`attestation_stale` when they've drifted — staleness ≠ forgery). With `--witness` the attestation is anchored in Rekor and the self-verifying proof travels **inside** the payload — both pubkeys embedded, the witness key authenticated by the logged entry itself — so the attestation alone convinces an auditor with no repo access |
| Capstone | `repo_integrity.v1` | `dipcatcher verify-repo` | One sealed verdict over all of the above — contract-checked so the attestation can't claim `ok` while a gate lists errors, hide a gate entirely, or disagree with its own digest pins. `--evidence-only` verifies an exported evidence bundle (the five evidence dirs + `gate_pins.sig`, no `src/`): crown_jewels reports a skipped marker and the attestation records `mode=evidence_only`, so a partial verdict can never masquerade as a full-tree one. Coverage closure: every file under a corpus dir must match the declared member glob or a per-corpus exemption (`uncovered_member:<rel>` otherwise) — a `.yaml` workflow dropped beside `*.yml` members would still run on GitHub, and now surfaces |

## Corpus discipline

Twenty-eight corpora are epoch-chained under `quality/epoch_heads.json` —
the *entire tracked tree*: after the eight evidence corpora, every code and
doc directory is closed-world chained too (`src`, `tests`, `scripts`,
`docs`, `research`, `replay`, `reports`, `notebooks`, `examples`, `clients`,
`typings`, `spec`, `docker`, `deploy`, `third_party`, `rust`, `web`,
`.box-soft-verify`, `.cursor`, `.github` beyond `workflows/`). A tamper
anywhere in the repo — source, test fixture, vendored code, deploy surface —
drifts a head. Two pins keep the world closed: every tracked top-level dir
must be a corpus or an explicit tool-state opt-out, and the root-level file
set is pinned (root files sit above every corpus dir).

- `receipts/*.json` — append-only, strict: any member digest change between
  epochs is `member_mutated` (tamper evidence). `legacy-unsealed/` is
  quarantined with byte-pins + recorded verify-errors (`quality/legacy_quarantine.json`).
- `verifier/*.md` — append-only acceptance records, strict.
- `quality/*.json` — mutable bookkeeping: `--allow-member-updates` gives ordered
  history attestation; a post-stamp edit still fails as `head_member_digest_drift`
  until `make stamp-epochs` re-stamps.
- `.github/workflows/*.yml` — mutable *and* security-critical:
  `--require-stamped` makes every unstamped `.yml` arrival a CI error plus
  `--allow-member-updates` for edits. The rest of `.github` (templates,
  plans) is its own `*` corpus.
- `configs/*` — mutable: post-hoc config edits would silently rewrite what a
  sealed bench measured, so each change needs a stamp like the quality corpus.
- `artifacts/*` — committed claim artifacts (champion selects, dev grids):
  regenerated between runs → mutable, stamped arrivals only.
- `.dsh-24x7/*` — committed fleet evidence (the SOTA input stream): a live
  dir, so arrivals land between stamps, but a stamped member is immutable —
  an `.npz` mutating post-stamp is tamper.
- The twenty closed-world corpora — mutable (`--allow-member-updates`) with
  stamped arrivals (`--require-stamped`): any member byte drifting between
  epochs is recorded tamper evidence, and a committed file no check covers is
  a coverage hole (`uncovered_member`). Exemptions: `__pycache__/` dirs are
  never members anywhere (machine-local bytecode), and `.github`'s corpus
  exempts `workflows/` (its own chain). Membership exemptions are corpus-scoped
  by basename — a `witness/` dir under `receipts` is an ordinary member.
- `data/metadata/*` — committed dataset manifests + validation outputs: the
  inputs `data_manifest` receipts pin; write-once dated dirs, stamped
  arrivals only.

After touching a covered file: `make stamp-epochs` (advances all chains +
pin) and `dipcatcher crown-jewels --write` if a jewel changed. Both must land
in the same commit as the change. If `GATE_SIGNING_KEY` is provisioned,
finish with `make sign-pins` — the signature must cover the final pin bytes
(sign last, since stamping updates `epoch_heads.json`), then
`make checkpoint` + `dipcatcher witness-checkpoint` to extend the spine and
anchor the new pin-state in Rekor.

Timestamp anchors (`dipcatcher anchor-timestamp`) go **stale by design** on
every stamp — the `.tsr` binds the pin bytes at request time. Re-anchor at
quiet points: after merges land, at release. A stale anchor stays valid
proof of the earlier state; `fresh: false` just says the pin has moved on.
The FreeTSA certs in `quality/timestamps/` make chain verification offline;
only stamping needs the network.

## Trust boundary

Receipt seals are sha256, not signatures — but the *pins* are: when
`gate_pins.sig` + `quality/gate_signing.pub` are committed, rewriting a gate
file, re-pinning, and re-stamping still cannot produce a valid signature
without the private key (held outside the repo). For unsigned trees the
boundary is **git review**: pins and epochs are small, reviewable diffs in
every PR, and `verify-repo` receipts pin the pin files' digests into the
receipts corpus so their history is itself chained. Either way, silent
tamper is impossible — any rewrite leaves a committed, hash-linked trail or
a broken signature.

Standalone-verifier boundary, stated plainly: `verify_epoch_proof` /
`verify_corpus_proof.py --proof` accept a *properly-named, self-sealed,
contract-valid* epoch receipt they are shown — a wholly fabricated chain
link verifies standalone because nothing in the file itself is dishonest.
Provenance is owned one layer up: the heads pin (`epoch_heads.json`) names
the exact chain terminus a proof must bind to, `check_epoch_chain` rejects
orphan/fork/multiple-genesis chains, and the pin is quorum-signed +
OTS-anchored. An auditor should therefore always pass `--pin` (or
`--checkpoint` + `--pubkey`, which additionally binds the pin file's bytes
inside the Ed25519-signed checkpoint) — bare `--proof` mode authenticates
shape, not custody.

Key lineage: when `GATE_SIGNING_KEY` rotates, `dipcatcher rotate-key`
records a dual-signed `key_rotation.v1` link (old key authorizes, new key
proves possession), then the operator installs the new pubkey in
`gate_signing.pub`, re-signs (`make sign-pins`), and checkpoints + witnesses
the transition. The rotation chain's genesis is anchored to the spine: the
outgoing key must have signed a real checkpoint. A substituted key whose
holder never signed — i.e. never held authority — fails
`unanchored_genesis`; a rotation recorded but never installed fails
`live_key_not_terminus`.

Key reset: a rotation needs the outgoing key. When nobody holds it any more
(never escrowed, destroyed, generated somewhere that no longer exists),
`dipcatcher key-reset` records a `key_reset.v1` file instead of a silent key
swap. It binds the current checkpoint as the boundary — embedded byte-exact
and re-verified under the lost key set — names the new gate key, registry and
witness key, and is signed by the new keys. History up to the boundary keeps
verifying under the old keys only; everything after it must be signed by the
new ones, and a lost key that signs again is rejected, so a reset also
retires keys that may have leaked rather than vanished. The witness key is
declared lost only when the reset replaces it (`--witness-pub`); otherwise
the record names no witness key on either side and the installed one stays.
A record that declares a witness key lost must name a different one
(`lost_witness_not_replaced`, `reuses_lost_witness_key`). Lost stays lost
across resets: no later reset, rotation or hand edit may bring back a gate
key, registry member or witness key an earlier reset declared lost
(`key_reset_revoked_registry`, `key_reset_reinstates_witness`), and
`key-reset` refuses to write such a record (`reinstates_lost_key`,
`reinstates_lost_witness_key`). A bundle whose head signs an archive or
witness-proof extent, or that carries a reset, is always walked as a spine
bundle, so stripping its history fails (`spine_archive_count`) instead of
reading as a pre-spine bundle.

What a reset cannot prove is authorization: a lost key cannot sign its own
retirement, so the reset is authorized by whoever can commit to the
repository and `authorized_by` is a recorded claim, not a signature. The
defense is visibility and review. `verify-rotations`, `checkpoint-chain`
and `verify-repo` report the reset (`verdict=reset`, `n_resets`, a
`key_reset:` note), both bundle verifiers re-check it, the record is a
stamped `quality/` member pinned by every later checkpoint, and it cannot
rewrite history — pre-boundary checkpoints stay bound to the old keys and to
their Rekor entries. Treat a reset as a trust-root change: protect
`quality/key_reset_*`, `quality/gate_signing.pub`, `quality/gate_quorum.json`
and `quality/witness_signing.pub` with CODEOWNERS and branch protection, and
review the record's reason and boundary like any other change to them.
Residual risk, stated plainly: someone with commit access and no keys can
forge a reset. It is loud — a new record, new pubkeys, `verdict=reset` in
every report — but only review blocks it.

## Operator quick reference

```bash
make evidence-audit        # all gates (CI runs this)
make stamp-epochs          # re-stamp chains + heads pin after touching a corpus
dipcatcher verify-repo     # compose every gate into one sealed verdict
dipcatcher verify-repo --out quality/repo_integrity.json   # seal the attestation
dipcatcher admit-batch receipts/<new>.json --strict          # gate a new receipt
```
