"""Tests for checkpoint_chain — the end-to-end spine verifier."""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.research.checkpoint_chain import (
    chain_contract_errors,
    checkpoint_spine,
    write_chain_receipt,
)
from quant_fund.research.gate_signatures import generate_keypair, sign_pins
from quant_fund.research.integrity_checkpoint import write_checkpoint


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    q = tmp_path / "quality"
    q.mkdir(parents=True)
    (q / "crown_jewels.json").write_text('{"pin": "crown"}\n')
    (q / "epoch_heads.json").write_text(
        json.dumps(
            {
                "schema": "epoch_heads.v1",
                "heads": {
                    "receipts/*.json": {
                        "receipt": "corpus_epoch_abc123.json",
                        "sha256": "0" * 64,
                    }
                },
            }
        )
        + "\n"
    )
    priv, pub = generate_keypair()
    sign_pins(tmp_path, priv, pub)
    return tmp_path, priv, pub


def _spine(root: Path, priv: str, pub: str, n: int) -> None:
    """Write n checkpoints — each lands with the previous archived."""
    for _ in range(n):
        write_checkpoint(root, [(priv, pub)])


def test_genesis_only(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    write_checkpoint(root, [(priv, pub)])
    res = checkpoint_spine(root)
    assert res["ok"] is True
    assert res["spine_length"] == 1
    assert res["verdict"] == "intact"


def test_chain_walks_to_genesis(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    _spine(root, priv, pub, 4)
    res = checkpoint_spine(root)
    assert res["ok"] is True
    assert res["spine_length"] == 4
    assert res["n_records"] == 4
    assert res["genesis_sha256"] != res["head_sha256"]


def test_dangling_prev_fails(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    _spine(root, priv, pub, 3)
    # Remove the archive member the live checkpoint points at.
    live = json.loads((root / "quality/checkpoint.json").read_text())
    prev = live["payload"]["prev_sha256"]
    archive = root / "quality/checkpoints"
    victim = next(
        f
        for f in archive.glob("*.json")
        if __import__("hashlib").sha256(f.read_bytes()).hexdigest() == prev
    )
    victim.unlink()
    res = checkpoint_spine(root)
    assert res["ok"] is False
    assert any(e.startswith("dangling_prev:") for e in res["errors"])


def test_injected_side_chain_is_orphan_and_fork(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    _spine(root, priv, pub, 2)
    forged = json.loads((root / "quality/checkpoint.json").read_text())
    sig = forged["signature"]
    # Always change the first hex digit: a fixed "0" is a no-op whenever the
    # fresh random key's signature already starts with "0" (p = 1/16).
    forged["signature"] = ("1" if sig[:1] == "0" else "0") + sig[1:]
    (root / "quality/checkpoints/zz_injected.json").write_text(json.dumps(forged))
    res = checkpoint_spine(root)
    assert res["ok"] is False
    assert any(e.startswith("fork:") for e in res["errors"])
    assert any(e.startswith("orphan_checkpoint:") for e in res["errors"])
    assert any(e.startswith("signature_invalid:") for e in res["errors"])


def test_flipped_signature_in_archive_fails(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    _spine(root, priv, pub, 2)
    archived = next((root / "quality/checkpoints").glob("*.json"))
    body = json.loads(archived.read_text())
    sig = body["signature"]
    body["signature"] = ("1" if sig[:1] == "0" else "0") + sig[1:]
    archived.write_text(json.dumps(body))
    res = checkpoint_spine(root)
    assert res["ok"] is False
    assert any("signature_invalid" in e for e in res["errors"])


def test_no_checkpoint_is_absent_neutral(tmp_path: Path) -> None:
    root, _priv, _pub = _repo(tmp_path)
    res = checkpoint_spine(root)
    assert res["ok"] is True
    assert res["verdict"] == "absent"
    assert res["signed"] is False


def test_contract_rejects_incoherent_payload() -> None:
    bad = {
        "schema": "checkpoint_chain.v1",
        "ok": True,
        "verdict": "intact",
        "n_records": 5,
        "spine_length": 2,  # ok=True but didn't reach everything
        "n_forks": 0,
        "n_orphans": 0,
        "errors": [],
    }
    assert "ok_implies_full_spine" in chain_contract_errors(bad)
    good = {
        "schema": "checkpoint_chain.v1",
        "ok": True,
        "verdict": "intact",
        "n_records": 3,
        "spine_length": 3,
        "n_forks": 0,
        "n_orphans": 0,
        "errors": [],
    }
    assert chain_contract_errors(good) == []


def test_write_chain_receipt_seals(tmp_path: Path) -> None:
    root, priv, pub = _repo(tmp_path)
    _spine(root, priv, pub, 2)
    out = tmp_path / "chain.json"
    write_chain_receipt(checkpoint_spine(root), out)
    sealed = json.loads(out.read_text())
    assert sealed["receipt_sha256"]
    assert sealed["schema"] == "checkpoint_chain.v1"
