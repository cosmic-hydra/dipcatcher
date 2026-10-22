.PHONY: help test test-full coverage lint typecheck doctor sync fmt security audit ci examples evidence native audit-obs docs docs-serve formal simtest simtest-large fx1-test fx1-lint fx1-corpus fx1-corpus-full fx1-eval fx1-gate mc-engine-smoke diffbacktest proofcore-test proofcore-coverage proof-integrity proof-verify leakage-scan reality-gate receipts-reverify receipt-integrity-scan audit-cli-mounts pretrade-bench stress-smoke market-sim-test parity-smoke demo-data lattice-check replay-sweep perf-record perf-check evidence-audit admission-gate stamp-epochs sign-pins anchor-pins checkpoint anchor-checkpoint witness-checkpoint verify-witness witness-bundle verify-bundle epoch-consistency verify-rotations rotate-key tamper-drill fuzz-drill fuzz-receipts evidence-bundle bundle-verify audit-js audit-rust audit-kronos audit-all native-lint native-test robustness-smoke fx1-coverage arch-guard

.DEFAULT_GOAL := help

help: ## Show targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

sync: ## Install the locked environment (all groups and extras)
	uv sync --frozen --all-groups --all-extras

# Thread pinning + fork/fault env for every test lane. Validated fix for the
# three-OpenMP-runtime NULL-kmp_info segfault under xdist
# (.dsh-24x7/lane_reports/l3-segfault.md 7.1): OMP_NUM_THREADS=1 keeps libomp
# out of __kmp_suspend_* (the crashing stack); KMP_INIT_AT_FORK=FALSE closes the
# second NULL-kmp_info route via fork(); PYTHONFAULTHANDLER=1 captures any
# future native fault. Matches the fleet job manifests (AGENTS.md "Thread
# pinning"). Mirrored into the CI test workflows — keep both sides identical.
TEST_ENV := OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
	VECLIB_MAXIMUM_THREADS=1 KMP_INIT_AT_FORK=FALSE PYTHONFAULTHANDLER=1 \
	TOKENIZERS_PARALLELISM=false

test: ## PR-gate lab tests (not network, not slow; xdist)
	$(TEST_ENV) uv run pytest -n auto --dist loadfile -m "not network and not slow"

test-full: ## Full offline lab suite, including slow tests
	$(TEST_ENV) uv run pytest -n auto --dist loadfile -m "not network"

test-durations: ## Refresh checked-in .test_durations for pytest-split CI shards
	# PR gate first, then slow tests so schedule/full shards stay balanced too.
	$(TEST_ENV) uv run pytest -n auto --dist loadfile -m "not network and not slow" \
		--store-durations --durations-path .test_durations --clean-durations
	$(TEST_ENV) uv run pytest -n auto --dist loadfile -m "slow and not network" \
		--store-durations --durations-path .test_durations

parity-smoke: ## SYNTHETIC backtest/shadow parity smoke (simulated broker only)
	uv run pytest -q tests/unit/parity
	uv run python -m quant_fund.parity smoke --out data/metadata/parity-smoke

coverage: ## PR-gate tests + coverage (threshold in pyproject)
	# Threshold lives in [tool.coverage.report] (pyproject.toml) — no inline
	# --cov-fail-under so CI and local cannot drift. Sharded CI combines
	# partial data files and applies the same threshold once.
	uv run pytest -n auto --dist loadfile -m "not network and not slow" --cov --cov-report=term-missing --cov-report=xml

lint: ## Ruff check + format check on src/ and tests/
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run python scripts/check_mypy_strict_allowlist.py
	uv run python scripts/check_mccabe_ratchet.py

fmt: ## Auto-fix lint + format
	uv run ruff check --fix src tests
	uv run ruff format src tests

typecheck: ## mypy on the harness (public modules are strict; see pyproject)
	uv run mypy src/quant_fund
	uv run mypy --strict --follow-imports=silent src/quant_fund/__init__.py src/quant_fund/public.py

arch-guard: ## Import-boundary guard (same commands as arch_guards.yml; stdlib-only)
	uv run --no-project python scripts/check_import_boundaries.py
	uv run --no-project python -m unittest tests.unit.arch.test_import_boundaries

diffbacktest: ## Differentiable backtest (optional JAX extra, CPU)
	JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" uv run pytest -q tests/unit/diffbacktest

security: ## Bandit static security analysis on src/
	uvx --from bandit==1.9.4 bandit -q -r src --severity-level medium --confidence-level medium

audit-obs: ## Audit ledger and observability tests
	uv run pytest tests/unit/audit tests/unit/observe -q
	uv run mypy src/quant_fund/audit src/quant_fund/observe

# Remove export markers before scanning so Windows/Linux extras are audited
# on every host. --disable-pip --no-deps queries the exported pins directly.
audit: ## All-platform locked Python dependency vulnerability audit (pip-audit)
	@set -eu; requirements=$$(mktemp); trap 'rm -f "$$requirements" "$$requirements.all"' EXIT; \
		uv export --frozen --quiet --format requirements.txt --no-hashes --no-emit-project \
		--all-extras --all-groups -o "$$requirements"; \
		sed 's/ ;.*//' "$$requirements" > "$$requirements.all"; \
		uvx --from pip-audit==2.10.1 pip-audit --strict --disable-pip --no-deps -r "$$requirements.all"

audit-js: ## Audit all four locked npm dependency trees
	cd web && npm audit
	cd replay && npm audit
	cd clients/typescript && npm audit
	cd clients/typescript/fx1 && npm audit

audit-rust: ## Audit the native extension (requires cargo-audit)
	cargo audit --file rust/quant_core/Cargo.lock

audit-kronos: ## Resolve and audit the independent vendored Kronos dependencies
	@set -eu; requirements=$$(mktemp); trap 'rm -f "$$requirements"' EXIT; \
		uv pip compile third_party/kronos/webui/requirements.txt --python-version 3.12 \
		--quiet -o "$$requirements"; \
		uvx --from pip-audit==2.10.1 pip-audit --strict --disable-pip --no-deps -r "$$requirements"

audit-all: audit audit-js audit-rust audit-kronos ## Repository-wide dependency audits

doctor: ## Harness environment check
	uv run dipcatcher doctor

demo-data: ## Generate labeled-SYNTHETIC offline demo dataset into data/demo/
	uv run python scripts/gen_demo_data.py

native: ## Build optional quant_core (Rust + maturin). NumPy stays the fallback.
	uv pip install "maturin>=1.7,<2"
	uv run maturin develop --release --manifest-path rust/quant_core/Cargo.toml

native-lint: ## Rust fmt + clippy for quant_core (same commands as ci.yml rust-accel)
	cd rust/quant_core && cargo fmt --check
	cd rust/quant_core && cargo clippy --locked --all-targets -- -D warnings

native-test: ## quant_core parity tests (ci.yml rust-accel; run `make native` first)
	uv run pytest -q -m native tests/property tests/unit tests/native

evidence: ## Regenerate docs/evidence/index.md from sealed receipts
	uv run python scripts/build_evidence_report.py

ci: lint typecheck coverage ## Local mirror of the CI gate

code-inventory: ## Report tracked semantic Python LOC and feature/test counts
	uv run python scripts/code_quality_inventory.py --summary-only

formal: ## TLC order-lifecycle check + Z3/conformance/stateful tests
	bash scripts/run_tlc.sh
	uv run pytest tests/formal -q

mc-engine-smoke: ## Monte Carlo engine tests (not slow) and a tiny CLI run
	uv run pytest tests/unit/mc_engine -q -m "not slow"
	# Same invocation as the mc-engine-smoke job in ci.yml (process backend +
	# antithetic shocks; the CI job additionally asserts the JSON contract) —
	# flags stay in lockstep so the local smoke cannot drift from the gate.
	uv run python -m quant_fund.mc_engine run --paths 1500 --steps 8 --workers 2 \
		--backend process --chunk-size 500 --seed 1 --shock-mode antithetic --no-progress

pretrade-bench: ## Pre-trade hot-path latency gate (p50 < 5us, p99 < 20us)
	uv run python -m quant_fund.pretrade.bench --gate

robustness-smoke: ## Robustness/certifier tests and CLI smoke (same commands as ci.yml)
	uv run pytest -q tests/unit/robustness tests/property/test_robustness_bounds.py
	uv run python -m quant_fund.robustness

stress-smoke: ## Fast stress-engine tests and the catalog report CLI
	uv run pytest -q -m "not network and not slow" tests/unit/stress
	uv run dipcatcher stress crises
	uv run dipcatcher stress report --strategy configs/stress_research.yaml --out /tmp/stress-report.md --format markdown

examples: ## Offline examples gallery: ruff, mypy, subprocess runner
	uv run ruff check examples tests/examples
	uv run ruff format --check examples tests/examples
	uv run mypy examples
	uv run pytest tests/examples -q

docs: ## Build the documentation site (strict)
	uv run --only-group docs --frozen mkdocs build --strict

docs-serve: ## Serve the documentation site locally
	uv run --only-group docs --frozen mkdocs serve --dev-addr 127.0.0.1:8000

simtest: ## Bounded deterministic-simulation tests and swarm (CI size)
	uv run pytest tests/unit/simtest tests/regression/test_simtest_duplicate_bar_order.py -q -m "not slow"
	MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/simtest_swarm.py --seeds 64 --days 5 --base-seed 0

simtest-large: ## Large seeded swarm (workflow_dispatch size; not the PR default)
	MLFLOW_DISABLE_AGENT_HINT=1 uv run python scripts/simtest_swarm.py --seeds 4000 --days 8 --base-seed 0

# Timings are machine-local: the CI gate (.github/workflows/perf-baseline.yml)
# records its baseline on CI runners; these targets are for local self-checks.
perf-record: ## Record a local perf baseline over the scoring/inference hot paths
	uv run python scripts/perf_baseline.py record --output data/metadata/perf-baseline.json

perf-check: ## Compare current timings against the stored local baseline (1.5x gate)
	uv run python scripts/perf_baseline.py record --output data/metadata/perf-current.json
	uv run python scripts/perf_baseline.py compare --baseline data/metadata/perf-baseline.json --current data/metadata/perf-current.json

# --- fx-1 (the model) lifecycle — dipcatcher is the harness ---------------
fx1-test: ## fx-1 test suite
	PYTHONPATH=src uv run pytest tests/fx1 -q

fx1-lint: ## fx-1 lint
	uv run ruff check src/fx1 tests/fx1

fx1-corpus: ## Build fx-1 SFT corpus from harness receipts + lab research runs
	# data/metadata/research/runs is host-local (gitignored, regenerable via
	# the lab research pipeline); the build degrades gracefully without it.
	uv run fx1 corpus build --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --out data/fx1/corpus.jsonl

fx1-corpus-full: ## Full corpus: receipts + research runs + notebooks + ledgers
	uv run fx1 corpus build-full --receipts-dir receipts \
		--receipts-dir data/metadata/research/runs --artifacts-dir artifacts \
		--notebooks docs/FX1.md --out data/fx1/corpus_full.jsonl

fx1-eval: ## Run eval task bank (requires MOONSHOT_API_KEY for hosted_k3)
	uv run fx1 eval --backend hosted_k3 --out data/fx1/eval.json

fx1-gate: fx1-lint ## Full fx-1 CI gate locally: lint + types + tests + honesty + corpus smoke
	uv run mypy src/fx1
	# CI's fx1 "Types" step also runs --strict (fx1.yml); the local full gate
	# keeps parity with the CI gate — raise code to strict, never drop this.
	uv run mypy --strict src/fx1
	PYTHONPATH=src uv run pytest tests/fx1 -q
	PYTHONPATH=src uv run pytest tests/fx1 -q -k honesty
	uv run fx1 corpus build --receipts-dir receipts --out data/fx1/corpus_ci_smoke.jsonl

fx1-coverage: ## fx-1 coverage ratchet, floor 80 (same command as proofcore.yml fx1-coverage)
	uv run pytest tests/fx1 -q --cov=fx1 --cov-fail-under=80 --cov-report=term-missing

# --- PROOFCORE gates (DESIGN.md §9.3; additive — no existing target changed) -
# Merge order: W5 lands last; the W1-W4 CLIs below exist once those waves merge.

PROOFCORE_LEDGER ?= data/metadata/proofcore-trials.jsonl
PROOFCORE_DB ?= data/metadata/proofcore.duckdb
DEFAULT_PROOFCORE_DB := data/metadata/proofcore.duckdb
COMMITTED_TRIAL_LEDGER ?= research/reality/trials.jsonl

proofcore-test: ## PROOFCORE tests: W5 contracts/provenance/CI/layering + W6 scheduler/runner/estimators + W7 replay + W8 guard/fixes + wave-2 e2e
	uv run pytest tests/unit/test_proofcore_*.py tests/end_to_end/test_proofcore_smoke.py \
		tests/unit/test_scheduler.py tests/unit/test_proven_runner.py \
		tests/unit/test_estimators.py tests/unit/test_replay_engine.py \
		tests/unit/test_io_guard.py tests/unit/test_cscv_combo_guard.py \
		tests/unit/test_fingerprint_fallback.py tests/unit/test_wave2_e2e.py \
		tests/property/test_replay_determinism.py -q

proofcore-coverage: ## Per-package coverage floors (A3 #2): pit/proof/reality/proofcore 90, leakage 85
	# Subset run over the PROOFCORE test lanes; the global 80% floor still
	# applies to the full `make coverage` lane and is NOT re-asserted here
	# --cov-fail-under=0 defers to the per-package gates below, which read
	# [tool.proofcore.coverage-floors] from pyproject.toml).
	uv run pytest -m "not network" -q \
		-k "pit or proof or leakage or reality or proofcore" \
		--cov --cov-report= --cov-fail-under=0
	uv run python -m quant_fund.proofcore.ci coverage-gate

proof-integrity: ## Current proof signer/recorder integrity checks
	uv run pytest tests/unit/proof/test_integrity.py -q

proof-verify: ## Bundle hash, sidecar, signature, and metric verification tests; replay remains closed
	uv run pytest tests/unit/test_proof_bundle.py tests/unit/test_proof_verify.py tests/property/test_backtest_receipt_identity.py -q

leakage-scan: ## Leakage hunter — WARN MODE this wave (adjudicated: advisory only)
	@echo "leakage-scan is WARN MODE this wave: findings are advisory, the gate"
	@echo "flips to --fail-on error in the migration wave (DESIGN.md §13 phase 3)."
	uv run quant leakage scan --paths src/quant_fund --format json > leakage-report.json || \
		echo "::warning::leakage scan failed or is not yet merged (advisory this wave)"
	@if [ -d tests/leakage_fixtures ]; then \
		uv run pytest tests/unit -q -k "leakage"; \
	else \
		echo "tests/leakage_fixtures not present yet (W3 lands separately); skipped"; \
	fi

reality-gate: ## Reality-filter gate: score trials; absent DB or empty export skips
	if [ ! -f "$(PROOFCORE_DB)" ] && [ "$(PROOFCORE_DB)" = "$(DEFAULT_PROOFCORE_DB)" ] && [ -s "$(COMMITTED_TRIAL_LEDGER)" ]; then \
	  uv run quant reality preflight --ledger $(COMMITTED_TRIAL_LEDGER); code=$$?; \
	  if [ $$code -eq 3 ]; then exit 0; fi; \
	  if [ $$code -ne 0 ]; then exit $$code; fi; \
	  uv run quant reality trial-report --ledger $(COMMITTED_TRIAL_LEDGER); \
	  uv run quant reality ledger-gate --ledger $(COMMITTED_TRIAL_LEDGER); \
	  exit $$?; \
	fi; \
	uv run quant reality preflight --db $(PROOFCORE_DB); code=$$?; \
	if [ $$code -eq 3 ]; then exit 0; fi; \
	if [ $$code -ne 0 ]; then exit $$code; fi; \
	uv run quant proofcore export --db $(PROOFCORE_DB) --out $(PROOFCORE_LEDGER); \
	uv run quant reality preflight --ledger $(PROOFCORE_LEDGER); code=$$?; \
	if [ $$code -eq 3 ]; then exit 0; fi; \
	if [ $$code -ne 0 ]; then exit $$code; fi; \
	uv run quant reality trial-report --ledger $(PROOFCORE_LEDGER) && \
	uv run quant reality ledger-gate --ledger $(PROOFCORE_LEDGER)

receipts-reverify: ## Fail-closed audit; schema-specific committed receipt verifiers pending (~20min, one subprocess per receipt)
	uv run python -m quant_fund.proofcore.ci receipts-reverify receipts

receipt-integrity-scan: ## Fast in-process self-seal digest scan over committed receipts (seconds, not minutes)
	uv run python scripts/receipt_integrity_scan.py receipts

audit-cli-mounts: ## Detect Typer apps defined in src/ but never mounted (dropped-wiring guard)
	uv run python scripts/audit_cli_mounts.py

evidence-audit: ## CI gate: re-verify every committed receipt; fail on any unverifiable non-legacy artifact
	uv run dipcatcher suite-health --strict --out-dir "$${RUNNER_TEMP:-/tmp}/evidence-audit"
	uv run dipcatcher corpus-epoch --corpus-dir receipts --check --heads-pin quality/epoch_heads.json --require-stamped --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir verifier --glob '*.md' --check --heads-pin quality/epoch_heads.json --require-stamped --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir quality --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir .github/workflows --glob '*.yml' --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir configs --glob '*' --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir artifacts --glob '*' --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir .dsh-24x7 --glob '*' --check --heads-pin quality/epoch_heads.json --allowed-removals quality/epoch_allowed_removals.json
	uv run dipcatcher corpus-epoch --corpus-dir data/metadata --glob '*' --check --heads-pin quality/epoch_heads.json --require-stamped --allowed-removals quality/epoch_allowed_removals.json
	for spec in "src" "tests" "scripts" "docs" "research" "replay" "reports" "notebooks" "examples" "clients" "typings" "spec" "docker" "deploy" "third_party" "rust" "web" ".box-soft-verify" ".cursor" ".github"; do \
	  uv run dipcatcher corpus-epoch --corpus-dir "$$spec" --glob '*' --check --heads-pin quality/epoch_heads.json --require-stamped --allow-member-updates --allowed-removals quality/epoch_allowed_removals.json || exit 1; \
	done
	uv run dipcatcher crown-jewels --check
	uv run dipcatcher verify-witness
	uv run dipcatcher verify-repo
	uv run dipcatcher checkpoint-chain
	uv run dipcatcher verify-rotations
	uv run dipcatcher tamper-drill
	uv run dipcatcher fuzz-drill --seed 7

checkpoint-chain: ## Walk the full checkpoint spine — every archived link verifies, no forks/orphans, Rekor order holds
	uv run dipcatcher checkpoint-chain

verify-rotations: ## Verify the gate-key rotation chain — dual-signed links, spine-anchored genesis, live key is terminus
	uv run dipcatcher verify-rotations

rotate-key: ## Record an authorized gate-key rotation (needs GATE_SIGNING_KEY + GATE_SIGNING_KEY_NEW); then re-sign pins + checkpoint
	uv run dipcatcher rotate-key

tamper-drill: ## Self-attack: clone the integrity state, land every probe mutation, require verify-repo flags each
	uv run dipcatcher tamper-drill

fuzz-drill: ## Metamorphic self-fuzz: seeded mutations classified must-fail vs must-pass — catches a verifier that is too strict OR too blind
	uv run dipcatcher fuzz-drill --seed 7

fuzz-receipts: ## Forge-and-reseal drill: mutates one claim per committed receipt, re-seals honestly — maps which claims contracts re-derive vs which stay self-attested
	uv run dipcatcher fuzz-receipts --seed 7

epoch-consistency: ## PR gate: prove every epoch chain extends the base-branch head — a history rewrite can't satisfy it. Needs EPOCH_BASE=<ref>
	@if [ -z "$${EPOCH_BASE:-}" ]; then echo "epoch-consistency: no EPOCH_BASE — skipped"; exit 0; fi; \
	for spec in "receipts:*.json" "verifier:*.md" "quality:*.json" ".github/workflows:*.yml" "configs:*" "artifacts:*" ".dsh-24x7:*" "data/metadata:*" "src:*" "tests:*" "scripts:*" "docs:*" "research:*" "replay:*" "reports:*" "notebooks:*" "examples:*" "clients:*" "typings:*" "spec:*" "docker:*" "deploy:*" "third_party:*" "rust:*" "web:*" ".box-soft-verify:*" ".cursor:*" ".github:*"; do \
	  dir=$${spec%%:*}; glob=$${spec##*:}; \
	  head=$$(git show "$$EPOCH_BASE:quality/epoch_heads.json" 2>/dev/null | uv run python -c "import json,sys; print(json.load(sys.stdin)['heads'].get('$$dir/$$glob',{}).get('receipt',''))"); \
	  if [ -z "$$head" ]; then echo "epoch-consistency skip $$dir: no base head"; continue; fi; \
	  proof="$${RUNNER_TEMP:-/tmp}/consistency_$$(echo $$dir | tr '/.' '__').json"; \
	  uv run dipcatcher corpus-consistency --corpus-dir "$$dir" --glob "$$glob" --from-epoch "$$head" --out "$$proof" >/dev/null || exit 1; \
	  uv run dipcatcher corpus-consistency --corpus-dir "$$dir" --glob "$$glob" --check "$$proof" || exit 1; \
	done

stamp-epochs: ## Re-stamp all corpus-epoch chains + head pin after touching any covered dir
	uv run dipcatcher corpus-epoch --corpus-dir receipts --out-dir receipts --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir verifier --glob '*.md' --out-dir verifier --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir quality --out-dir quality --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir .github/workflows --glob '*.yml' --out-dir .github/workflows --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir configs --glob '*' --out-dir configs --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir artifacts --glob '*' --out-dir artifacts --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir .dsh-24x7 --glob '*' --out-dir .dsh-24x7 --heads-pin quality/epoch_heads.json
	uv run dipcatcher corpus-epoch --corpus-dir data/metadata --glob '*' --out-dir data/metadata --heads-pin quality/epoch_heads.json
	for spec in src tests scripts docs research replay reports notebooks examples clients typings spec docker deploy third_party rust web .box-soft-verify .cursor .github; do \
	  uv run dipcatcher corpus-epoch --corpus-dir "$$spec" --glob '*' --out-dir "$$spec" --heads-pin quality/epoch_heads.json || exit 1; \
	done

sign-pins: ## Ed25519-sign the integrity pins (needs GATE_SIGNING_KEY or --key-file); run LAST, after stamp-epochs
	uv run dipcatcher sign-pins

anchor-pins: ## RFC 3161 timestamp-anchor both pin files via FreeTSA (network); run after sign-pins, at quiet points only — every stamp-epochs stales the anchors
	uv run dipcatcher anchor-timestamp --file quality/epoch_heads.json
	uv run dipcatcher anchor-timestamp --file quality/crown_jewels.json

checkpoint: ## Sign the pin state into quality/checkpoint.json (needs GATE_SIGNING_KEY); run LAST — it binds the current signature
	uv run dipcatcher checkpoint

anchor-checkpoint: ## RFC 3161-anchor the checkpoint (network); one token time-binds the whole pin state
	uv run dipcatcher checkpoint --anchor

witness-checkpoint: ## Witness the checkpoint into the public Rekor transparency log (network; needs WITNESS_SIGNING_KEY); commits a self-verifying proof under quality/witness/
	uv run dipcatcher witness-checkpoint

verify-witness: ## Verify committed Rekor witness proofs offline — RFC 6962 inclusion + Rekor SET/note signatures
	uv run dipcatcher verify-witness

witness-bundle: ## Emit the zero-trust auditor bundle (one JSON: checkpoint + pins + pubkeys + freshest Rekor proof)
	uv run dipcatcher witness-bundle --out auditor_bundle.json

verify-bundle: ## Verify an auditor bundle with zero trusted repo input (BUNDLE=path)
	uv run dipcatcher verify-bundle $(BUNDLE)

evidence-bundle: ## Export the evidence store as a portable third-party bundle (BUNDLE_DIR=path)
	uv run dipcatcher evidence-export --out "$${BUNDLE_DIR:-evidence-bundle}"

bundle-verify: ## Audit an exported evidence bundle — stdlib script, no repo imports (BUNDLE_DIR=path)
	uv run python scripts/verify_evidence_bundle.py --root "$${BUNDLE_DIR:-evidence-bundle}"

lattice-check: ## CI gate: cross-receipt consistency lattice; fails on 'inconsistent' verdicts
	uv run dipcatcher lattice --strict \
		--known-inconsistent quality/lattice_known_inconsistent.json \
		--out-dir "$${RUNNER_TEMP:-/tmp}/lattice"

replay-sweep: ## CI gate: replay every replayable carrier; fail on divergence or all-skip
	uv run dipcatcher replay-all --strict \
		--out "$${RUNNER_TEMP:-/tmp}/replay_coverage.json"

ADMISSION_BASE ?= origin/main
admission-gate: ## CI gate: sequentially admit each diff-changed corpus receipt (BASE vs HEAD)
	@changed=$$(git diff --name-only --diff-filter=ACMRT $(ADMISSION_BASE) HEAD -- 'receipts' 2>/dev/null \
		| grep '^receipts/[^/]*\.json$$' || true); \
	if [ -n "$$changed" ]; then \
		uv run dipcatcher admit-batch $$changed --corpus-dir receipts --strict \
			--known-inconsistent quality/lattice_known_inconsistent.json \
			--out-dir "$${RUNNER_TEMP:-/tmp}/admission"; \
	else \
		echo "admission-gate: no corpus receipt changes vs $(ADMISSION_BASE)"; \
	fi

market-sim-test: ## Matching engine and agent-market tests
	uv run pytest tests/unit/market_sim tests/property/test_lob_invariants.py -m "not slow"
