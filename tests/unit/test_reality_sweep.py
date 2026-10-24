"""Pre-registered reality sweep: grid lock, cost lock, and ledger recording."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from quant_fund.config.loader import load_config
from quant_fund.proofcore.cli import write_trial_csv
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    CodeFingerprint,
    DataManifestSummary,
    EnvFingerprint,
    ProofBundleV1,
    SignatureBlock,
    TrialLedgerRow,
    sha256_hex_bytes,
)
from quant_fund.proofcore.provenance import ProvenanceDB
from quant_fund.research.reality_sweep import (
    Cell,
    ScoredCell,
    assert_cost_lock,
    equal_weight_long,
    grid_cells,
    load_spec,
    persist_trials,
    select_winner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# The 2026-09-27 study is decided (verdict: deflated — see
# docs/REALITY_TRIAL_2026.md), so its frozen spec lives under the archive
# path, not the pending default.
_DECIDED_SPEC = (
    REPO_ROOT
    / "research"
    / "reality"
    / "studies"
    / "reality-us-liquid-daily-2026-09-27"
    / "preregistration.json"
)
_HEX = "ab" * 32


def _scored(trial_id: str, ratio: float, *, degenerate: bool) -> ScoredCell:
    empty = np.asarray([], dtype=float)
    return ScoredCell(
        cell=Cell("sweep_reclaim", "sweep_reclaim", "discovery", {}),
        trial_id=trial_id,
        by_window={"validation": {"periodic_ratio": ratio, "degenerate": degenerate}},
        validation_returns=empty,
        train_returns=empty,
        holdout_returns=np.asarray([9.0], dtype=float),
        dates={},
        turnover={},
        equity=pl.DataFrame(),
    )


def test_grid_matches_the_frozen_preregistration() -> None:
    spec = load_spec(_DECIDED_SPEC)
    cells = grid_cells(spec)
    assert len(cells) == 29
    assert sum(cell.strategy == "sweep_reclaim" for cell in cells) == 18
    assert sum(cell.strategy == "slow_trend" for cell in cells) == 10
    assert sum(cell.strategy == "equal_weight_long" for cell in cells) == 1
    assert len({cell.trial_id(spec["study_id"]) for cell in cells}) == 29


def test_cost_lock_accepts_backtest_yaml_and_rejects_a_change() -> None:
    spec = load_spec(_DECIDED_SPEC)
    config = load_config(REPO_ROOT / "configs" / "backtest.yaml")
    assert_cost_lock(config, spec)
    config.costs.frictionless = True
    with pytest.raises(ValueError, match="frictionless"):
        assert_cost_lock(config, spec)


def test_winner_uses_validation_ratio_and_ignores_holdout_magnitude() -> None:
    rows = [
        _scored("b" * 64, 0.2, degenerate=False),
        _scored("a" * 64, 0.2, degenerate=False),
        _scored("c" * 64, 5.0, degenerate=True),
    ]
    rows[0].holdout_returns = np.asarray([100.0], dtype=float)
    winner = select_winner(rows)
    assert winner.trial_id == "a" * 64


def test_equal_weight_respects_net_and_name_caps() -> None:
    when = datetime(2024, 1, 2, 21, tzinfo=UTC)
    names = [f"N{i}" for i in range(8)]
    bars = pl.DataFrame(
        {
            "event_time": [when] * len(names),
            "security_id": names,
            "close": [10.0] * len(names),
        }
    )
    weights = equal_weight_long(bars, name_cap=0.05, net_cap=0.2)
    assert weights.height == 8
    assert float(weights["target_weight"].sum()) == pytest.approx(0.2)
    assert float(weights["target_weight"].max()) == pytest.approx(0.025)


def test_persist_trials_keeps_a_loser(tmp_path: Path) -> None:
    bundle_id = sha256_hex_bytes(b"reality-sweep-bundle")
    bundle = ProofBundleV1(
        bundle_id=bundle_id,
        created_utc="2026-09-27T00:00:00+00:00",
        run_kind="research",
        code=CodeFingerprint(git_revision="a26be34", worktree_sha256=_HEX, dirty=False),
        data_manifest=DataManifestSummary(reads=(), merkle_root=_HEX, n_reads=0),
        config_sha256=_HEX,
        seed=7,
        env=EnvFingerprint(
            python_version="3.12.12",
            python_implementation="CPython",
            platform="linux",
            machine="x86_64",
            byteorder="little",
            packages={"dipcatcher": "0.1.0"},
        ),
        signal_log_sha256=_HEX,
        trade_log_sha256=_HEX,
        metrics_sha256=_HEX,
        metrics_recompute={"n_trials": 2.0},
        prev_bundle_hash=GENESIS_HASH,
        signature=SignatureBlock(scheme="none", key_id="unsigned", value=""),
    )
    created = "2026-09-27T00:00:00+00:00"
    rows = [
        TrialLedgerRow(
            trial_id=sha256_hex_bytes(b"loser"),
            bundle_hash=bundle_id,
            family="discovery",
            strategy="sweep_reclaim",
            cluster_id="sweep_reclaim",
            created_utc=created,
            n_obs=40,
            periods_per_year=252.0,
            sharpe_periodic=-0.4,
            skew=0.0,
            kurtosis_raw=3.0,
            returns_sha256=_HEX,
        ),
        TrialLedgerRow(
            trial_id=sha256_hex_bytes(b"winner"),
            bundle_hash=bundle_id,
            family="discovery",
            strategy="slow_trend",
            cluster_id="slow_trend",
            created_utc=created,
            n_obs=40,
            periods_per_year=252.0,
            sharpe_periodic=0.1,
            skew=0.0,
            kurtosis_raw=3.0,
            returns_sha256=_HEX,
        ),
    ]
    db = tmp_path / "proofcore.duckdb"
    assert persist_trials(db, bundle, rows) == 2
    csv_path = tmp_path / "trials.csv"
    with ProvenanceDB(db) as prov:
        stored = prov.trials()
    write_trial_csv(stored, csv_path)
    assert len(stored) == 2
    assert any(row.sharpe_periodic < 0 for row in stored)
    assert "sweep_reclaim" in csv_path.read_text(encoding="utf-8")


def test_committed_ledger_branch_scores_when_default_db_is_absent(tmp_path: Path) -> None:
    db = REPO_ROOT / "data" / "metadata" / "proofcore.duckdb"
    if db.exists():
        pytest.skip("local provenance db is present")
    ledger = tmp_path / "trials.jsonl"
    row = TrialLedgerRow(
        trial_id=sha256_hex_bytes(b"gate-row"),
        bundle_hash=_HEX,
        family="discovery",
        strategy="sweep_reclaim",
        cluster_id="sweep_reclaim",
        created_utc="2026-09-27T00:00:00+00:00",
        n_obs=32,
        periods_per_year=252.0,
        sharpe_periodic=0.0,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256=_HEX,
    )
    ledger.write_text(
        json.dumps(row.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [
            "make",
            "SHELL=bash",
            "reality-gate",
            "PROOFCORE_DB=data/metadata/proofcore.duckdb",
            f"COMMITTED_TRIAL_LEDGER={ledger}",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "verdict=insufficient_evidence" in proc.stdout
    assert "REALITY_FILTER_SKIP:" not in proc.stdout
    assert not db.exists()
