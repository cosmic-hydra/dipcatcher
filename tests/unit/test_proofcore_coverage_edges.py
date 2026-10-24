"""Fail-closed edges in proofcore contracts, provenance, CI, and run context."""

from __future__ import annotations

import runpy
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from quant_fund.proofcore import ci as ci_mod
from quant_fund.proofcore.ci import _git_revision, coverage_floors
from quant_fund.proofcore.cli import _fsync_directory
from quant_fund.proofcore.contracts import (
    GENESIS_HASH,
    DataAccessRecord,
    DecisionTrace,
    DecisionTraceRow,
    ProofError,
    ProofVerificationError,
    ProvenanceError,
    SignatureBlock,
)
from quant_fund.proofcore.provenance import ProvenanceDB
from quant_fund.proofcore.run_context import PropagatingThread

_HEX = "ab" * 32


def _trace_row(*, seq: int = 0, skew_ceiling: bool = False) -> DecisionTraceRow:
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    ceiling = stamp + timedelta(seconds=1) if skew_ceiling else stamp
    return DecisionTraceRow(
        seq=seq,
        decision_time=stamp,
        known_at_ceiling=ceiling,
        data_manifest_sha256=_HEX,
        feature_set_sha256=_HEX,
        estimator_state_sha256=_HEX,
        action_sha256=_HEX,
        rng_counter_sha256=_HEX,
        prev_row_sha256=GENESIS_HASH,
    )


def _trace(row: DecisionTraceRow) -> DecisionTrace:
    return DecisionTrace(
        rows=(row,),
        spec_sha256=_HEX,
        code_fingerprint="src-tree",
        env_fingerprint="linux|3.12|dev",
        head_row_sha256=_HEX,
    )


def test_verification_error_keeps_reason_list() -> None:
    err = ProofVerificationError("failed", ["chain:missing"])
    assert err.reasons == ["chain:missing"]
    assert ProofVerificationError("failed").reasons == []


def test_asof_utc_must_be_iso() -> None:
    with pytest.raises(ValidationError, match="ISO-8601"):
        DataAccessRecord(dataset="silver/bars", asof_utc="yesterday", rows=1, content_sha256=_HEX)


def test_signature_block_rejects_dishonest_unsigned_and_bad_hmac() -> None:
    with pytest.raises(ValidationError, match="unsigned"):
        SignatureBlock(scheme="none", key_id="signed", value="")
    with pytest.raises(ValidationError, match="hmac-sha256"):
        SignatureBlock(scheme="hmac-sha256", key_id="not-hex-key-id", value="zz")


def test_trace_times_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        DecisionTraceRow(
            seq=0,
            decision_time=datetime(2024, 1, 1),
            known_at_ceiling=datetime(2024, 1, 1, tzinfo=UTC),
            data_manifest_sha256=_HEX,
            feature_set_sha256=_HEX,
            estimator_state_sha256=_HEX,
            action_sha256=_HEX,
            rng_counter_sha256=_HEX,
            prev_row_sha256=GENESIS_HASH,
        )


def test_trace_chain_rejects_seq_gap_and_ceiling_mismatch() -> None:
    with pytest.raises(ProofError, match="seq gap"):
        _trace(_trace_row(seq=1)).verify_chain()
    with pytest.raises(ProofError, match="known_at_ceiling"):
        _trace(_trace_row(skew_ceiling=True)).verify_chain()


def test_fsync_directory_ignores_missing_path(tmp_path: Path) -> None:
    _fsync_directory(tmp_path / "does-not-exist")


def test_propagating_thread_wraps_positional_target_and_allows_none() -> None:
    seen: list[str] = []

    def _target() -> None:
        seen.append("ran")

    bare = PropagatingThread()
    assert bare._proven_context is not None

    worker = PropagatingThread(None, _target)
    worker.start()
    worker.join(timeout=5)
    assert seen == ["ran"]


def test_coverage_floors_fail_closed(tmp_path: Path) -> None:
    from quant_fund.proofcore.contracts import ProofcoreError

    missing = tmp_path / "absent.toml"
    with pytest.raises(ProofcoreError, match="pyproject not found"):
        coverage_floors(missing)

    bad = tmp_path / "bad.toml"
    bad.write_text("[tool.proofcore.coverage-floors]\npit = true\n")
    with pytest.raises(ProofcoreError, match="bad entry"):
        coverage_floors(bad)

    incomplete = tmp_path / "incomplete.toml"
    incomplete.write_text("[tool.proofcore.coverage-floors]\npit = 90\n")
    with pytest.raises(ProofcoreError, match="missing an int entry"):
        coverage_floors(incomplete)


def test_git_revision_is_none_when_git_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ci_mod.shutil, "which", lambda _name: None)
    assert _git_revision(tmp_path) is None


def test_module_entrypoint_exits_with_the_gate_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["quant_fund.proofcore.ci", "receipts-reverify", str(tmp_path)]
    )
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(ci_mod.__file__, run_name="__main__")
    assert caught.value.code == 1


def _trial(trial_id: str):
    from quant_fund.proofcore.contracts import TrialLedgerRow

    return TrialLedgerRow(
        trial_id=trial_id,
        bundle_hash=_HEX,
        family="discovery",
        strategy="edge",
        cluster_id="c",
        created_utc="2024-01-01T00:00:00+00:00",
        n_obs=8,
        periods_per_year=252.0,
        sharpe_periodic=0.1,
        skew=0.0,
        kurtosis_raw=3.0,
        returns_sha256=_HEX,
    )


def test_empty_trial_chain_is_vacuously_ordered(tmp_path: Path) -> None:
    db = ProvenanceDB(tmp_path / "proofcore.duckdb")
    try:
        assert db.verify_trial_chain() == []
    finally:
        db.close()


def test_relative_db_path_skips_parent_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    db = ProvenanceDB(Path("ledger.duckdb"))
    try:
        assert db.path.name == "ledger.duckdb"
    finally:
        db.close()


def test_unopenable_provenance_db_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="cannot open provenance DB"):
        ProvenanceDB(tmp_path)


def test_verify_trial_chain_reports_a_fork(tmp_path: Path) -> None:
    db = ProvenanceDB(tmp_path / "proofcore.duckdb")
    try:
        columns = [
            row[1] for row in db._con.execute("PRAGMA table_info('proof_bundles')").fetchall()
        ]
        values = {
            "bundle_id": _HEX,
            "created_utc": "2024-01-01T00:00:00+00:00",
            "run_kind": "backtest",
            "git_revision": "0" * 40,
            "config_sha256": _HEX,
            "seed": 1,
            "merkle_root": _HEX,
            "prev_bundle_hash": GENESIS_HASH,
            "signature_scheme": "none",
            "bundle_json": "{}",
            "verified_ok": True,
            "verification_json": None,
        }
        db._con.execute(
            f"INSERT INTO proof_bundles ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [values[name] for name in columns],
        )
        db.insert_trial(_trial("11" * 32))
        db.insert_trial(_trial("22" * 32))
        # The UNIQUE constraint is what insert_trial relies on; the audit still
        # has to recognize a fork if that constraint was bypassed (raw rewrite).
        db._con.execute("CREATE TABLE trial_ledger_fork AS SELECT * FROM trial_ledger")
        db._con.execute("DROP TABLE trial_ledger")
        db._con.execute("ALTER TABLE trial_ledger_fork RENAME TO trial_ledger")
        db._con.execute("UPDATE trial_ledger SET prev_trial_hash = ?", [GENESIS_HASH])
        with pytest.raises(ProvenanceError, match="forked"):
            db.verify_trial_chain()
    finally:
        db.close()
