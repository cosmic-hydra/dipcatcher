# Production deploy (research / simulation only)

This is the production **runbook for research and simulated paper surfaces**.
It does not authorize live trading, enable a live broker, or claim that
verification gates promote capital. The live broker path stays fail-closed;
see [INSTITUTIONAL_READINESS.md](INSTITUTIONAL_READINESS.md) and
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) "Promotion and live boundary".

Owner action after this PR: when CI is green on a candidate `main` SHA, tag
`v$(fx1.__version__)` (today `v0.4.0`) to exercise `.github/workflows/release.yml`.
Do not tag from this branch until the owner decides.

## Surfaces

| Surface | How it runs | Health |
|---|---|---|
| Research API (root `Dockerfile`) | `dipcatcher api` on loopback `:8000` | `GET /health` (liveness, public); `GET /ready` (doctor-backed, auth/loopback) |
| Research evidence API (`docker/research-api.Dockerfile`) | `python -m quant_fund.api.research_api` on `:8010` | `GET /health` (liveness); `GET /ready` (dirs present, auth/loopback) |
| fx-1 harness service | `uv run fx1 harness serve …` | See [FX1_DEPLOY.md](FX1_DEPLOY.md) (`/health`, `/ready`) |
| Observability stack | `deploy/observability/docker-compose.yml` | Prometheus / Grafana / OTLP on loopback only |

## Config

Committed profiles under `configs/` (and nested `configs/configs/`):

| Profile | Purpose |
|---|---|
| `configs/research.yaml` | Default research / doctor profile |
| `configs/paper.yaml` | Simulated paper broker (no live endpoints) |
| `configs/production.yaml` | Stricter research/sim ops knobs — **not** a live-trading enablement |
| `configs/backtest.yaml` / `configs/sim_live.yaml` | Offline / simulated lanes |

Resolve configs only under `configs/` (path allowlist in the research API). Prefer
loopback binds (`127.0.0.1`) unless `QUANT_API_KEY` / `RESEARCH_API_KEY` is set.

## Secrets

Names only — never commit values. Canonical inventory: `.env.example`.

| Class | Examples | Notes |
|---|---|---|
| API auth | `QUANT_API_KEY`, `RESEARCH_API_KEY` | Required for non-loopback binds |
| Evidence signing | `GATE_SIGNING_KEY`, `GATE_QUORUM_KEYS`, `WITNESS_SIGNING_KEY`, `PROOFCORE_SIGNING_KEY` | Owner-held; agents do not mint keys |
| Hosted eval (optional) | `MOONSHOT_API_KEY` | `make fx1-eval` only |
| Observe (optional) | `DIPCATCHER_OBSERVE`, `DIPCATCHER_OTEL_ENDPOINT`, `DIPCATCHER_METRICS_PORT` | Off unless set; see [AUDIT_OBSERVABILITY.md](AUDIT_OBSERVABILITY.md) |

Rotate gate keys with the procedures in [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md)
"Gate-key rotation". Absent signing keys, pin/checkpoint steps fail closed.

## Observability (`deploy/observability`)

```bash
docker compose -f deploy/observability/docker-compose.yml up
export DIPCATCHER_OBSERVE=1
export DIPCATCHER_OTEL_ENDPOINT=http://127.0.0.1:4318/v1/traces
export DIPCATCHER_METRICS_PORT=9464
export DIPCATCHER_METRICS_HOST=0.0.0.0
```

- Compose binds collector `:4318`, Prometheus `:9090`, Grafana `:3000` on **loopback**.
- SLO definitions: `deploy/observability/slo.yml` (missing samples are `no_data`, not a pass).
- Dashboard: `deploy/observability/grafana/dashboards/dipcatcher.json`.
- Full contract: [AUDIT_OBSERVABILITY.md](AUDIT_OBSERVABILITY.md). Simulation gauges are
  operational telemetry, not research scores and not a live P&L claim.

`make audit-obs` exercises the audit/observe unit tests and compose parse checks.

## Health checks

| Probe | Meaning | Fail closed when |
|---|---|---|
| `GET /health` | Process up (liveness) | Almost never — keep Docker HEALTHCHECK here |
| `GET /ready` (lab API) | Doctor report for a config | Manifest/receipt/import/dir checks fail |
| `GET /ready` (research API) | Evidence directories exist | Any configured root missing |
| `dipcatcher doctor` | Pre-run harness check | `live_allowed` unexpectedly true, invalid manifest/receipt |
| Container HEALTHCHECK | Docker wraps `/health` | Unreachable for 3 retries |

Local smokes:

```bash
make health-smoke          # in-process /health + /ready against research API factory
make container-health      # docker build + wait for healthy (mirrors CI/release)
```

## Release pipeline (no publish from this doc)

1. `make release-dry-run` — version/changelog, wheel, SBOM, checksums, container health.
   Skips Sigstore, SLSA, tags, PyPI, and registry push.
2. Ensure the candidate SHA has green `ci.yml` (including `evidence-audit`), `fx1.yml`,
   `proofcore.yml`, and the full platform `matrix.yml`.
3. Owner tags `v<fx1.__version__>` on that SHA. `release.yml` signs wheels, attests
   provenance, uploads workflow artifacts, and re-checks container health.
4. Verify downloads: `uv run python scripts/verify_release_artifacts.py dist/`.

Homebrew `packaging/homebrew/fxi.rb` stays `FILL_AT_RELEASE` until the owner
publishes GitHub Release assets (out of band; this workflow does not create them).

## Rollback

Follow [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) "Rollback": previous image/wheel,
drain fx-1 harness before SIGTERM, revoke/rotate keys rather than editing receipts.
