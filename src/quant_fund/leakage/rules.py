"""Leakage Hunter rule registry (LH001..LH014) and path allowlists.

RULE_REGISTRY is the ONLY source of rule metadata (DESIGN.md §6.1). Rule IDs
are permanent; new rules append with new IDs. Allowlists are per-rule
frozensets of path globs; every entry carries a comment citing why the
existing site is legitimate (or which audit finding / follow-up owns it).
LH001 also has a function-scoped allowlist: the file stays scanned, and
only the named function is exempt.

Residual ceiling (ADVERSARIAL §1a, pinned by the documented-negative fixtures
in tests/leakage_fixtures/adv_*.py): numpy/pandas index arithmetic, dict
lookups at ``dates[i+1]``, lone numbers beyond the proximity window, manual
Sharpe algebra without a ``sharpe_ratio`` call, and cross-file/multi-level
helper indirection remain uncaught. The pack is a tripwire; the runtime
watchdog + proof layer are the barrier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str  # "LH001"
    severity: Literal["error", "warning"]
    title: str
    detail: str  # remediation text shown in report


RULE_REGISTRY: dict[str, RuleSpec] = {
    "LH001": RuleSpec(
        "LH001",
        "error",
        "Backward shift on price-like frames",
        "`.shift(-k)` on a price-like series reads future bars. Forward-return "
        "targets belong under `labels/` with explicit `fwd_*` naming; features "
        "must use trailing (`shift(+k)`) windows only.",
    ),
    "LH002": RuleSpec(
        "LH002",
        "error",
        "Centered rolling window",
        "`center=True` mixes future observations into the window. Use trailing "
        "windows (`center=False`, the default) everywhere in signal paths.",
    ),
    "LH003": RuleSpec(
        "LH003",
        "error",
        "Fit-before-split scaler",
        "Scaler `.fit` must be lexically inside a fold/split loop (or a "
        "`*_fold*` function) so statistics are computed on train folds only. "
        "Fit on the train slice, freeze, then transform.",
    ),
    "LH004": RuleSpec(
        "LH004",
        "error",
        "As-of join on event_time",
        "`join_asof(on='event_time')` joins by event time, not publish time. "
        "PIT joins must key on `known_at`/`available_time` so restatements "
        "published after the decision cannot leak in.",
    ),
    "LH005": RuleSpec(
        "LH005",
        "error",
        "Frozen universe construction",
        "A hard-coded ticker list is survivorship-biased (audit F4). Build "
        "universes from point-in-time membership (`data/universe.py`) instead.",
    ),
    "LH006": RuleSpec(
        "LH006",
        "error",
        "Forward difference under contemporaneous name",
        "`delta_*`/`d_*`/`chg_*` columns that are forward differences "
        "(`X.shift(-k) - X`) read as contemporaneous but encode next-bar "
        "prices (audit F9). Name forward targets with a `fwd_`/`lead_` prefix.",
    ),
    "LH007": RuleSpec(
        "LH007",
        "error",
        "Annualized Sharpe into PSR/MinTRL",
        "PSR/MinTRL/DSR require the Sharpe ratio in PERIOD units, never "
        "annualized (the audit's unit-inflation finding). Use per-period "
        "inputs or the unit-safe `reality.psr_from_returns` API.",
    ),
    "LH008": RuleSpec(
        "LH008",
        "error",
        "Forbidden-metric headline string",
        "String literal headlines a forbidden metric (a sharpe/sortino/calmar/"
        "pnl/nav alias in proximity of a numeric literal — the audit's "
        "headline-bypass finding). Research results are proper scores only.",
    ),
    "LH009": RuleSpec(
        "LH009",
        "warning",
        "Direct parquet read outside data layer",
        "`pl.read_parquet`/`pl.scan_parquet` outside `data/`/`pit/` bypasses "
        "the PIT choke point (audit A2 F8); `PitVault.history()` returns "
        "unfiltered versions and is audit-only. Route reads through "
        "`pit.guarded_read_parquet`. WARNING-only until the call-site "
        "migration wave lands (adjudicated; flips to error afterwards).",
    ),
    "LH010": RuleSpec(
        "LH010",
        "warning",
        "Backfill on time-series frame",
        "`.bfill()`/`fill_null(strategy='backward')` on a frame with an "
        "`event_time` column pulls future values into the past. Use forward "
        "fills or explicit staleness handling.",
    ),
    "LH011": RuleSpec(
        "LH011",
        "error",
        "PROOFCORE layering violation",
        "New packages (pit/proof/leakage/reality/proofcore) may only import "
        "the whitelisted `quant_fund` packages for their layer (DESIGN.md "
        "§1.3). Keep the new stack acyclic above the existing SCC.",
    ),
    "LH012": RuleSpec(
        "LH012",
        "warning",
        "Unparseable file",
        "`ast.parse` raised SyntaxError; the file was not scanned. Fix the "
        "syntax so the linter can check it.",
    ),
    "LH013": RuleSpec(
        "LH013",
        "warning",
        "Headline metric claim in a comment/docstring/f-string/spelled-out form",
        "ADVERSARIAL section 1a hardening: the forbidden-headline channel is "
        "not limited to digit-bearing plain string literals — comments and "
        "docstrings render into published docs, f-string templates disclose "
        "runtime numbers, and spelled-out numerals ('exceeded two') evade "
        "the digit matcher. WARNING-only: prose mentions of metric names are "
        "legitimate; a human reviews these.",
    ),
    "LH014": RuleSpec(
        "LH014",
        "warning",
        "Call to a helper that trips a leakage rule",
        "ADVERSARIAL section 1a hardening: leaky logic hidden in a helper "
        "function is still leaky when strategy code calls the helper. "
        "Single-level, single-file summary pass: flags call sites of "
        "functions whose own body produced a finding. Cross-file and "
        "multi-level indirection are the documented residual ceiling.",
    ),
}

# ---------------------------------------------------------------------------
# Allowlists: per-rule frozensets of path globs (matched against the scanned
# path and any suffix after a leading directory). Every entry cites its
# justification. These codify the LEGITIMATE sites at HEAD so the clean-src
# gate (tests/unit/test_leakage_clean_src.py) can demand zero error findings.
# ---------------------------------------------------------------------------

# LH001: known-legitimate forward-shift sites verified in audit §2
# ("No wrong-sign shifts in features" — all are forward-return *targets*).
LH001_ALLOWLIST: frozenset[str] = frozenset(
    {
        # labels/** builds forward-return targets; the only place shift(-k)
        # on prices is legal (audit §2 clean-check).
        "src/quant_fund/labels/**",
        # data/calendars.py: next-session lookup (`next_open` execution
        # calendar), not a feature (audit §2 clean-check; DESIGN.md §6.1).
        "src/quant_fund/data/calendars.py",
        # northset/sweep_research.py: event-study entry/exit horizons are
        # explicitly forward-looking research targets (audit §2 clean-check).
        "src/quant_fund/northset/sweep_research.py",
        # northset/kyle_ofi.py + benches.py: `fwd_ret_*`/`fwd_delta_mid`
        # Kyle-regression *targets* (audit F9 confirms consumers treat them
        # as targets only). LH006 separately gates the `delta_mid` alias.
        "src/quant_fund/northset/kyle_ofi.py",
        "src/quant_fund/northset/benches.py",
        # microstructure/bench.py: forward candle-return benchmark target
        # (audit §2 clean-check).
        "src/quant_fund/microstructure/bench.py",
    }
)

# LH001 function scope. The file stays scanned; a finding is dropped only
# when its innermost enclosing function is listed. Nested helpers and every
# other function in the file remain checked.
# forward_close_return_labels builds y_{t+1} as column fwd_ret_1 on the full
# bar panel. It is a label, not a feature.
LH001_FUNCTION_ALLOWLIST: dict[str, frozenset[str]] = {
    "src/quant_fund/microstructure/candle_book_features.py": frozenset(
        {"forward_close_return_labels"}
    ),
}

# LH003: scaler `.fit` sites that consume caller-sliced train folds (audit §2:
# "ranker scalers fit inside each train fold"; the fold loop lives in
# pipeline/train.py, above these classes).
LH003_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/quant_fund/models/ranking.py",
        "src/quant_fund/models/asset_pricing.py",
        # GaussianHMMRegime.fit self-fits its StandardScaler on the model's
        # own training input — the caller slices folds, same model-self-fit
        # idiom as ranking.py / asset_pricing.py above.
        "src/quant_fund/models/regime.py",
    }
)

# kit_paths.py accepts caller-supplied training arrays. Keep the file scanned
# and exempt only the reviewed self-fit entry points: fit_ohlcv trains a model
# on its input, while bench_kit_paths is a synthetic benchmark. There is no
# production pipeline caller proving fold isolation yet, so callers must pass
# an already isolated training slice.
LH003_FUNCTION_ALLOWLIST: dict[str, frozenset[str]] = {
    "src/quant_fund/models/kit_paths.py": frozenset({"fit_ohlcv", "bench_kit_paths"}),
}

# LH004: backward as-of joins on event_time over frames that are already
# PIT-filtered upstream (funding/roll calendars, factor exposures) — verified
# publish-time-clean in audit §2. New PIT joins must key on known_at.
LH004_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/quant_fund/backtest/sleeves.py",
        "src/quant_fund/portfolio/pnl_attribution.py",
        "src/quant_fund/portfolio/factor_model.py",
        "src/quant_fund/microstructure/candle_book_features.py",
        # data/universe.py _vectorized_membership_panel: backward asof-join on
        # event_time over bars whose _membership_fast_supported check already
        # proves available_time <= event_time (no late inserts) — same
        # publish-time-clean class as the sites above.
        "src/quant_fund/data/universe.py",
    }
)

# LH005: intentional fixed research universes. lightspeed/specs.py pins the
# momentum spec's reference universe by design (explicit SELECTION_END /
# HOLDOUT_START split — holdout is not for retuning); LH005 gates every
# other site.
LH005_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/quant_fund/lightspeed/specs.py",
    }
)

# LH006: the two known audit-F9 `delta_mid` alias sites. Renaming the column
# touches northset consumers and is deferred per DESIGN.md §14.6 — the gate
# blocks every NEW occurrence while the legacy sites stay inventoried here.
LH006_ALLOWLIST: frozenset[str] = frozenset(
    {
        "src/quant_fund/northset/kyle_ofi.py",
        "src/quant_fund/northset/benches.py",
    }
)

# LH007: hedge_lab/scoreboard.py IS audit F1; the unit-safe fix is owned by
# W4 (DESIGN.md §7.1 scoreboard fix spec). Allowlisted here only until that
# PR lands; LH007 gates all other files immediately.
LH007_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Audit F1 itself; unit-safe fix owned by W4 (§7.1 scoreboard spec).
        "src/quant_fund/hedge_lab/scoreboard.py",
        # Same F1 unit class (annualized SR into deflated_sharpe with
        # per-period n) found by this rule at HEAD — the audit listed only
        # scoreboard.py. Tracked with the W4 unit-safe migration.
        "src/quant_fund/validation/multiple_testing.py",
    }
)

# LH008: exact SHA-256 hashes of existing disclosure literals. File-wide
# exemptions hid new forbidden headlines added to these modules. These hashes
# cover only the known gate notes and paper-book caveats at HEAD; changing a
# literal requires explicit review. Hashes are of UTF-8 literal values.
LH008_LITERAL_ALLOWLIST: dict[str, frozenset[str]] = {
    "src/quant_fund/risk/gates.py": frozenset(
        {
            "c720638fb545bc36175d3031be70b52a8fa2a0aaed5dacaafb723ec13d62f70d",
            "5c6f2c6c6b17419e77d41fd276e959d2555a26cadedc8555f99c44e849ef7437",
            "76c03b2fab26ca028cefd3fbc2cb15f56fd075ad4d9c3ac4443b8feb7eaa12f0",
        }
    ),
    "src/quant_fund/hedge_lab/gated_race.py": frozenset(
        {
            "e1f6e4021a81c1fe1bb085ce890c951eb9240e7a89e056f61a47e507fccb35d7",
            "c438a1925886ed4fdb1a8bc2eb1a07dfebd23d85e47b86affb71cf2d0bbc98e7",
        }
    ),
    "src/quant_fund/hedge_lab/mirror.py": frozenset(
        {
            "e1b51a564a3d38b5f5d609b87bd2860a424501c652d8bc8202476aff305e1c6d",
        }
    ),
    "src/quant_fund/hedge_lab/runner.py": frozenset(
        {
            "44e73ad81e844c41b5aa4f9f133633137d6d2bd7596efec232706c219bdf7048",
            "dfe65ba8ae01e38edc1b1981445f6b78a885a1256302778ea4725f43db8b53ab",
        }
    ),
    "src/quant_fund/hedge_lab/target_hunt.py": frozenset(
        {
            "2771f70229b61aa85b998b4d663846f03fa586070b11bb08ef49b761a8111a42",
        }
    ),
    "src/quant_fund/hedge_lab/v2_slate.py": frozenset(
        {
            "9eaace5b2b62dff19e3db82bac479bbcd04beeb54e87f8e62210029c64bb5472",
            "9191a0aa6d5e9d44401e4ead4855ebea13aefe24547c986a37d700e3d727c26c",
            "94fda2dd11e56e7fb460c49397ed208064e4435cec034ca1f5608665e9ac4fc3",
            "0d6b8449f8d0745b25880e3a6cb9618ba33181adc11ba8b25aafd531c29e4973",
        }
    ),
    # parity_leak_audit.py: scanner test fixtures — these string literals are
    # deliberately constructed forbidden patterns used to verify the LH008
    # scanner catches them. They are NOT production disclosure text.
    "src/quant_fund/parity_leak_audit.py": frozenset(
        {
            "965649c38bb6c4740e1a3fb14c9204aa2b6838481d27134664f7f8e890ebf121",
            "8793cf9737b657b34ede8437b3459ba9c77b8920991487d775d7017cb4b59f5c",
            "4ee52816c7079cce1a82f5030570df6d716fdb9c3a836483418d9e6fa83c7299",
            "db3293fc24a72a39a096494fa85b0a375c1d35841e79e31077d85763a96e4dbf",
            "537b151e6a40c4dc1f5fd9c0fd828a57b0119a4a83db028ddd2092c670253505",
            "5464f85d0ca6adc8995d8328c4006e2f0de72dc84c07f8168f230dd8e07c7554",
            "e722ad9fdbe5e5a3f216748b393d756807099f5c185b57c73c900dd8c6958ef3",
        }
    ),
    # policy_eprocess.py: validation error message containing "pnl" — not a
    # headline metric claim, just a parameter-name reference in an exception.
    "src/quant_fund/research/policy_eprocess.py": frozenset(
        {
            "31cccf51a0ecea0642f135a3f1d6ba96be2edab03362ef6cefaef4e81c300823",
        }
    ),
}

# Fixture clean-controls that deliberately mirror allowlisted production
# idioms; they exercise the allowlist plumbing and the false-positive
# discipline of the seeded-leak gate (DESIGN.md §6.4).
CONTROL_FIXTURE_ALLOWLIST: frozenset[str] = frozenset(
    {
        # Mirrors labels/engine.py forward-target idiom (LH001-exempt there).
        "tests/leakage_fixtures/clean_forward_label.py",
        # Mirrors northset/benches.py `fwd_delta_mid` target idiom.
        "tests/leakage_fixtures/clean_fwd_delta_mid.py",
    }
)

RULE_ALLOWLISTS: dict[str, frozenset[str]] = {
    "LH001": LH001_ALLOWLIST | CONTROL_FIXTURE_ALLOWLIST,
    "LH003": LH003_ALLOWLIST,
    "LH004": LH004_ALLOWLIST,
    "LH005": LH005_ALLOWLIST,
    "LH006": LH006_ALLOWLIST,
    "LH007": LH007_ALLOWLIST,
}

# rule id -> repo-relative path -> function names. Exact path suffix, not a
# glob: a same-named function in another file is still scanned.
FUNCTION_ALLOWLISTS: dict[str, dict[str, frozenset[str]]] = {
    "LH001": LH001_FUNCTION_ALLOWLIST,
    "LH003": LH003_FUNCTION_ALLOWLIST,
}

# LH009 / LH010 scope exemptions (these rules are warning-severity at HEAD but
# the exemptions are structural, not per-site).
LH009_EXEMPT_GLOBS: frozenset[str] = frozenset(
    {
        "src/quant_fund/data/**",  # data layer IS the legacy choke point
        "src/quant_fund/pit/**",  # the vault itself
        "tests/**",  # tests may read fixtures directly
    }
)

# LH011: import whitelist per PROOFCORE package (DESIGN.md §1.3). Top-level
# imports must be in WHITELIST; function-level (lazy) imports may additionally
# use LAZY_WHITELIST.
LH011_WHITELIST: dict[str, frozenset[str]] = {
    "pit": frozenset({"proofcore", "schemas", "utils", "data"}),
    # proof -> config (layer-0, §1.3): the runner binds AppConfig at top level;
    # config imports nothing from the SCC, so the edge stays acyclic.
    "proof": frozenset({"proofcore", "schemas", "utils", "config"}),
    "leakage": frozenset({"proofcore", "schemas"}),
    "reality": frozenset({"proofcore", "metrics", "validation"}),
    "proofcore": frozenset(),
}
# config/utils are layer-0 packages (§1.3): lazy imports of them from CLI
# glue modules (which live inside each new package) cannot create cycles.
_LH011_LAYER0_LAZY = frozenset({"config", "utils"})
LH011_LAZY_WHITELIST: dict[str, frozenset[str]] = {
    "pit": _LH011_LAYER0_LAZY,
    # proof lazily reaches pit (W1 vault seam, §5.2), leakage (W3 watchdog,
    # §5.2 step 2), and metrics (A1 F2 headline recompute). None of pit /
    # leakage import proof back, so the lazy edges cannot create a cycle.
    # promotion_receipt also lazily reaches pipeline (artifact identity),
    # registry (promotion_is_approved), and research (receipt_v2 seal) —
    # composition-only seams; pipeline/registry never import proof, and
    # research only reaches proof.bundle lazily, so no import-time cycle.
    "proof": _LH011_LAYER0_LAZY
    | {"backtest", "pit", "metrics", "leakage", "pipeline", "registry", "research"},
    # §6.2: patterns.py lazily sources FORBIDDEN_HEADLINE_TOKENS from
    # research.catalog.FORBIDDEN_RESEARCH_METRIC_KEYS (no copy); lazy-only so
    # no import-time edge into the SCC.
    "leakage": _LH011_LAYER0_LAZY | {"pit", "research"},
    # reality/cli lazily drives research.reality_sweep and
    # research.reality_survivorship; those lazily import reality.cscv and
    # reality.report back. Mutual lazy edges only — no import-time edge into the
    # SCC, the same reasoning as the leakage entry above.
    "reality": _LH011_LAYER0_LAZY | {"research"},
    "proofcore": frozenset(),
}
LH011_PACKAGES: frozenset[str] = frozenset({"pit", "proof", "leakage", "reality", "proofcore"})
