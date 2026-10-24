"""Cross-process append correctness for the audit ledger.

The ledger is the root of the evidence chain: several processes appending to
one root must serialize through the ``.lock`` file so the hash chain stays
intact. These tests run real child processes (spawn-safe: module-level
worker, picklable arguments) rather than threads, because flock semantics
across processes are the property under test.
"""

from __future__ import annotations

import json
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from quant_fund.audit.canonical import sha256_hex
from quant_fund.audit.ledger import GENESIS_HASH, AuditLedger, read_entries
from quant_fund.audit.signing import Ed25519Signer
from quant_fund.audit.verify import verify_ledger

_WRITERS = 4
_APPENDS_PER_WRITER = 20
# fork copies xdist/coverage thread locks into children that never run those
# threads, so a child can block forever on a lock the parent still holds.
# The worker is module-level and picklable; spawn is the context this file
# was written for.
_POOL = multiprocessing.get_context("spawn")


def _append_worker(root: str, writer: int, count: int) -> int:
    """Append ``count`` entries to the shared ledger. Runs in a child process."""
    ledger = AuditLedger(root, signer=None, sync=True)
    for seq in range(count):
        ledger.append(
            "paper_decision",
            {
                "writer": writer,
                "seq": seq,
                "simulation_only": True,
                "live_pnl_claim": False,
            },
        )
    return count


def _prefix_chain_errors(entries: list[dict[str, object]]) -> list[str]:
    """Verify hash-chain integrity of a possibly-truncated parsed prefix."""
    errors: list[str] = []
    prev = GENESIS_HASH
    for position, entry in enumerate(entries):
        if entry.get("index") != position:
            errors.append(f"index_gap:{position}")
        if entry.get("prev_hash") != prev:
            errors.append(f"prev_hash_mismatch:{position}")
        prev = str(entry.get("entry_hash"))
    return errors


def test_concurrent_process_appends_preserve_the_hash_chain(tmp_path: Path) -> None:
    """N processes × M appends -> exactly N*M entries, one index each, clean chain."""
    root = tmp_path / "ledger"
    with ProcessPoolExecutor(max_workers=_WRITERS, mp_context=_POOL) as pool:
        futures = [
            pool.submit(_append_worker, str(root), writer, _APPENDS_PER_WRITER)
            for writer in range(_WRITERS)
        ]
        for future in futures:
            assert future.result() == _APPENDS_PER_WRITER

    entries, errors = read_entries(root / "entries.jsonl")
    assert errors == []
    assert len(entries) == _WRITERS * _APPENDS_PER_WRITER
    assert _prefix_chain_errors([entry.to_obj() for entry in entries]) == []
    # No lost or duplicated writer/seq pairs: the lock truly serialized appends.
    seen = [(entry.payload["writer"], entry.payload["seq"]) for entry in entries]
    assert sorted(seen) == [
        (writer, seq) for writer in range(_WRITERS) for seq in range(_APPENDS_PER_WRITER)
    ]
    # Unsigned entries are not evidence: attach a signer and checkpoint once.
    signer = Ed25519Signer.generate()
    AuditLedger(root, signer=signer, sync=True).checkpoint()
    report = verify_ledger(root, trust_public_key=signer.public_key)
    assert report["valid"] is True
    assert report["tree_size"] == _WRITERS * _APPENDS_PER_WRITER
    assert report["errors"] == []


def test_concurrent_locked_reads_always_observe_a_valid_prefix(tmp_path: Path) -> None:
    """A reader holding the flock never sees a mid-append ledger state."""
    root = tmp_path / "ledger"
    stop = threading.Event()
    observations: list[str] = []

    def reader() -> None:
        ledger = AuditLedger(root, signer=None, sync=False)
        while not stop.is_set():
            try:
                entries = ledger.entries()
            except Exception as exc:  # noqa: BLE001 - record, classify below
                observations.append(f"error:{exc.__class__.__name__}")
                continue
            objs = [entry.to_obj() for entry in entries]
            chain_errors = _prefix_chain_errors(objs)
            observations.append("corrupt" if chain_errors else "clean")

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        writer = AuditLedger(root, signer=None, sync=False)
        for seq in range(40):
            writer.append(
                "paper_decision",
                {"seq": seq, "simulation_only": True, "live_pnl_claim": False},
            )
    finally:
        stop.set()
        thread.join(timeout=30)

    assert observations, "reader never ran while the writer was appending"
    # Locked reads must be clean or loudly failed — never a corrupt prefix.
    assert "corrupt" not in observations
    assert set(observations) <= {"clean"}


def test_checkpoint_after_concurrent_appends_signs_the_full_tree(tmp_path: Path) -> None:
    """A signer attached post-hoc checkpoints exactly the committed tree."""
    root = tmp_path / "ledger"
    with ProcessPoolExecutor(max_workers=_WRITERS, mp_context=_POOL) as pool:
        list(
            pool.map(
                _append_worker,
                [str(root)] * _WRITERS,
                list(range(_WRITERS)),
                [_APPENDS_PER_WRITER] * _WRITERS,
            )
        )
    signer = Ed25519Signer.generate()
    ledger = AuditLedger(root, signer=signer, sign_every=1, sync=True)
    record = ledger.checkpoint()
    assert record["tree_size"] == _WRITERS * _APPENDS_PER_WRITER
    report = verify_ledger(root, trust_public_key=signer.public_key)
    assert report["valid"] is True
    assert report["signed_tree_size"] == _WRITERS * _APPENDS_PER_WRITER
    checkpoint_lines = [
        json.loads(line) for line in (root / "checkpoints.jsonl").read_text().splitlines() if line
    ]
    assert len(checkpoint_lines) == 1
    assert sha256_hex(checkpoint_lines[0]["merkle_root"].encode()) != ""


def test_append_to_missing_root_creates_layout_under_the_lock(tmp_path: Path) -> None:
    """First writers racing on a fresh root must not corrupt the layout."""
    root = tmp_path / "fresh-ledger"
    with ProcessPoolExecutor(max_workers=_WRITERS, mp_context=_POOL) as pool:
        futures = [pool.submit(_append_worker, str(root), writer, 5) for writer in range(_WRITERS)]
        for future in futures:
            future.result()
    entries, errors = read_entries(root / "entries.jsonl")
    assert errors == []
    assert len(entries) == _WRITERS * 5
    assert _prefix_chain_errors([entry.to_obj() for entry in entries]) == []
