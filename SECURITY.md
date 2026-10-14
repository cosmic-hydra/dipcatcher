# Security Policy — fx-1 / dipcatcher

## Scope

This repository contains the fx-1 model project and the dipcatcher research
harness. Security-relevant surfaces:

- **fx-1 serving path** (`fx1.serve`): hosted-backend credentials, local
  checkpoint loading, release-signature verification.
- **Harness API** (`quant_fund.api`): API-key auth, loopback fail-closed,
  config allowlisting, request-size caps.
- **Corpus/receipt integrity** (`fx1.data`, `fx1.train.receipts`): hash-chained
  ledgers and tamper-evident receipts.
- **Signing keys** (`GATE_SIGNING_KEY`, `GATE_QUORUM_KEYS`,
  `WITNESS_SIGNING_KEY`, `FX1_SIGNING_KEY`): Ed25519/ECDSA key material for
  pin signatures, rotation receipts, witness anchoring, and checkpoint
  verification. Keys live only in the environment or in files outside the
  repository; only public keys (`quality/gate_signing.pub`) are tracked.

This is **research infrastructure only**. There is no live-trading path: no
broker connectivity, no order placement, no account balances, no P&L
execution. Everything downstream of a decision is simulated or SYNTHETIC
(and labeled as such); a security report that assumes a live trading
account is out of scope by definition.

## Supported versions

| Version | Supported | Security fixes |
|---|---|---|
| `main` (HEAD) | yes | all fixes land here first |
| latest `v*.*.*` tag | yes, when published | backports case-by-case |
| older tags, forks, vendored `third_party/` | no | upgrade to `main` or the latest tag |

Security fixes land on `main`. There is no published GitHub Release or PyPI
upload yet. A tag matching `v*.*.*` runs `.github/workflows/release.yml`
(build, CycloneDX SBOM, Sigstore signatures, SLSA provenance). That workflow
is tag-only: it does not publish to PyPI and does not create a GitHub
Release. Downstream consumers check downloaded artifacts with
`scripts/verify_release_artifacts.py` (checksums fail closed on tamper);
Sigstore and `gh attestation verify` cover signatures and SLSA provenance.

## Hard rules (enforced in code, tested)

- No credentials in source or tests; secrets come from environment variables
  only (`MOONSHOT_API_KEY`, `FX1_SIGNING_KEY`, `QUANT_API_KEY`,
  `RESEARCH_API_KEY`, `FX1_API_KEY`, and the gate/witness signing keys
  `GATE_SIGNING_KEY`, `GATE_SIGNING_KEY_NEW`, `GATE_QUORUM_KEYS`,
  `WITNESS_SIGNING_KEY`). The full
  env-var surface is enumerated in `.env.example`, a staged-diff
  `secret-scan` pre-commit hook and a gitleaks hook are enforced, and
  diagnostics (`fx1 doctor`, `dipcatcher doctor`) report presence flags,
  never values.
- Fail-closed defaults: missing signatures, missing model cards, failing
  honesty gates, and invalid receipts all *block* rather than warn.
- Network access is opt-in (explicit `collect`, explicit API serve), never
  part of default ingest or training paths.
- The paper and simulated brokers still refuse live endpoints
  (`LiveEndpointRefused`). `dipcatcher live` is a separate operator desk.
  It does not embed a broker SDK or a broker credential. Shadow is the
  default and writes no order ticket. Loopback fills stay in-process and
  are labeled SYNTHETIC. The drop folder writes a ticket only when the
  desk was armed with the confirm phrase `SEND-LIVE-ORDERS` and the
  process environment has `DIP_LIVE_ARM=1`. Chat cannot transmit. Reports
  keep `live_pnl_claim` false. A live trading account is still out of
  scope for research claims; see `docs/LIVE_DESK.md`.

## Reporting a vulnerability

Report privately. Use a [GitHub private security advisory](https://github.com/artificial-hedge/dipcatcher/security/advisories/new).
Do not open a public issue, pull request, or discussion for an exploitable
finding. Do not include a proof of concept that would disclose another
user's data.

Please include:

- A description of the issue and the impact you expect.
- The commit SHA you tested, and whether the behavior is on `main`.
- Steps or a minimal test that demonstrates the issue without embedding a
  live credential. If a secret is involved, send only the variable name and
  where it was found (file, commit, rule). Do not send the value.
- Whether you are willing to wait for a coordinated release.

We acknowledge reports within 72 hours, send a status update within 7 days,
and aim to publish a fix or a coordinated disclosure within 90 days of
acknowledgment. We will credit you unless you ask not to be named.

### Safe harbor

We will not pursue legal action against research that is in good faith,
stays within this repository, avoids privacy violations, data destruction,
and service degradation, and is reported privately as above. Stop if you
hit unexpected personal data and report it.

### Out of scope

- Social engineering of maintainers, physical attacks, and denial of service.
- Findings that only affect `third_party/` vendored trees, unless you show
  an impact on dipcatcher or fx-1 as built from this repository.
- Scanner results that are content hashes, receipt ids, or the synthetic
  canary in `tests/fx1/test_sources.py` (`test_probe_never_leaks_secret_values`).
- Requests for live-trading, broker, or order-placement behavior. This
  project does not ship that.

## Supply chain

- Python dependencies are locked (`uv.lock`), npm projects each have a
  `package-lock.json`, and the native extension has `Cargo.lock`. CI audits
  every platform's Python pins, including all groups and optional extras,
  with `pip-audit --strict` and runs Bandit. The dependency-audit workflow
  also checks all three npm projects, Rust, and the independently resolved
  vendored Kronos requirements on dependency changes and weekly.
- Run `make audit-all` locally (requires npm and `cargo-audit`; install the
  latter with `cargo install cargo-audit --version 0.22.2 --locked`). The
  individual targets are `audit`, `audit-js`, `audit-rust`, and `audit-kronos`.
  Kronos still uses requirements files rather than a lockfile; its audit
  checks a fresh Python 3.12 resolution without installing the model stack.
- Dependabot tracks root Python via `uv`, vendored Kronos via `pip`, all npm
  projects, Cargo, and GitHub Actions. Root Python updates have one owner
  so pip update PRs cannot leave `uv.lock` stale.
- Pull requests run dependency review. CodeQL (`security-extended`) analyzes
  `src/`, `tests/`, and `scripts/`. OpenSSF Scorecard runs on pull requests,
  on `main`, and weekly.
- gitleaks runs on the staged diff (pre-commit) and on full git history
  (`.github/workflows/secret-scan.yml`), with `.gitleaks.toml` allowing
  content hashes and the synthetic test canary.
- GitHub Actions are pinned to full commit SHAs. Workflow tokens are
  read-only except where a job must upload code-scanning results or mint an
  OIDC identity. A tag push builds, signs, and attests the distributions and
  uploads them as workflow artifacts only (no PyPI upload from CI).
- Checkpoint signatures for a local fx-1 checkout stay on the attestation
  ladder (`fx1 attestation <checkpoint_dir>` before serving). Package
  releases, when a tag is pushed, are signed with Sigstore and carry SLSA
  build provenance from GitHub artifact attestations. Verify local copies
  with `uv run python scripts/verify_release_artifacts.py <dist-dir>`.
