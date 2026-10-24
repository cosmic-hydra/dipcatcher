"""Crash-resume recovery tests for the ingestion journal.

SYNTHETIC batches only (correctness tests, never market evidence). Proves a
killed ingest resumes from its cursor with no lost or duplicate rows, that
re-ingest is idempotent, and that a corrupted journal fails closed.
"""

from __future__ import annotations

import json

import pytest

from quant_fund.data.ingest_resume import Batch, IngestCheckpoint, IngestJournalError


def test_crash_resume_no_duplicate_rows(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run1 = IngestCheckpoint(path)
    run1.ingest(Batch(batch_id="b1", rows=100))
    run1.ingest(Batch(batch_id="b2", rows=50))
    # crash: new process reloads the journal from disk
    run2 = IngestCheckpoint(path)
    assert run2.cursor == 2
    assert run2.total_rows == 150
    assert run2.n_batches == 2
    assert run2.ingest(Batch(batch_id="b3", rows=25)) == "committed"
    assert run2.total_rows == 175


def test_reingest_is_idempotent(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    assert run.ingest(Batch(batch_id="b1", rows=100)) == "committed"
    assert run.ingest(Batch(batch_id="b1", rows=100)) == "duplicate"
    assert run.total_rows == 100
    # crash + resume: the same batch is still a duplicate after reload
    run2 = IngestCheckpoint(path)
    assert run2.ingest(Batch(batch_id="b1", rows=100)) == "duplicate"
    assert run2.total_rows == 100


def test_resume_state_is_research_only(tmp_path) -> None:
    run = IngestCheckpoint(tmp_path / "ingest.jsonl")
    run.ingest(Batch(batch_id="b1", rows=1))
    state = run.resume()
    assert state["research_only"] is True
    assert state["live_pnl_claim"] is False


def test_journal_skips_blank_lines_and_rejects_broken_records(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    assert IngestCheckpoint(path).cursor == 0

    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(IngestJournalError, match="corrupt ingest journal"):
        IngestCheckpoint(path)

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(IngestJournalError, match="not an object"):
        IngestCheckpoint(path)


def test_corrupt_journal_fails_closed(tmp_path) -> None:
    path = tmp_path / "ingest.jsonl"
    run = IngestCheckpoint(path)
    run.ingest(Batch(batch_id="b1", rows=100))
    # silently tamper with a committed row count -> must fail closed on reload
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["rows"] = 999
    path.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    try:
        IngestCheckpoint(path)
    except IngestJournalError:
        pass
    else:
        raise AssertionError("expected IngestJournalError on tampered journal")
