from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_fund.research.corpus_epoch import (
    corpus_epoch,
    write_epoch_receipt,
)
from quant_fund.research.epoch_consistency import (
    chain_index,
    consistency_contract_errors,
    consistency_proof,
    verify_consistency,
)
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def _corpus(tmp_path: Path, n_stamps: int = 3) -> Path:
    c = tmp_path / "receipts"
    c.mkdir()
    (c / "r1.json").write_text(json.dumps({"a": 1}))
    for i in range(n_stamps):
        (c / f"r{i + 2}.json").write_text(json.dumps({"b": i}))
        write_epoch_receipt(corpus_epoch(c), c)
    return c


def _genesis(index: dict) -> str:
    """Digest-named epoch files have no lexical order — genesis is the hop
    whose prev link is absent or points outside the set."""
    return sorted(n for n, (_, e, _) in index.items() if e.get("prev_epoch_receipt") not in index)[
        0
    ]


def test_consistency_proof_verifies(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    index = chain_index(c)
    names = sorted(index)
    proof = consistency_proof(c, _genesis(index))
    assert proof["schema"] == "epoch_consistency.v1"
    assert proof["n_hops"] == len(names)
    assert verify_consistency(proof, c) == []
    # Contract on the emitted payload.
    assert consistency_contract_errors(proof) == []


def test_consistency_binds_held_digest(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    index = chain_index(c)
    first = _genesis(index)
    held = hash_bytes(index[first][0].read_bytes())
    proof = consistency_proof(c, first)
    assert verify_consistency(proof, c, held_sha256=held) == []
    # Wrong held digest = the verifier's copy isn't what the chain extends.
    assert "held_head_digest_mismatch" in verify_consistency(proof, c, held_sha256="0" * 64)


def test_consistency_unreachable_from_fails(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    with pytest.raises(ValueError, match="unknown from_receipt"):
        consistency_proof(c, "corpus_epoch_deadbeefdeadbeef.json")


def test_consistency_rewrite_detected(tmp_path: Path) -> None:
    """Corrupt a hop's file bytes: the proof's recorded digest no longer
    matches the live file, and a verifier holding the real old-head digest
    is unaffected — held binds the proof's claim, not the tampered bytes."""
    c = _corpus(tmp_path)
    index = chain_index(c)
    first = _genesis(index)
    held = hash_bytes(index[first][0].read_bytes())
    proof = consistency_proof(c, first)
    # Corrupt the first hop's bytes while keeping the epoch schema — removal
    # lands as hop_missing; a schema-preserving tamper lands as digest drift.
    payload = json.loads(index[first][0].read_text())
    payload["forged_marker"] = True
    (c / first).write_text(json.dumps(payload))
    errors = verify_consistency(proof, c, held_sha256=held)
    assert any(e.startswith("hop_digest_mismatch") for e in errors)


def test_consistency_held_binds_proof_start(tmp_path: Path) -> None:
    """A verifier whose held digest is for DIFFERENT bytes than the proof
    starts from is told the chain doesn't extend their state."""
    c = _corpus(tmp_path)
    index = chain_index(c)
    first = _genesis(index)
    proof = consistency_proof(c, first)
    assert "held_head_digest_mismatch" in verify_consistency(proof, c, held_sha256="f" * 64)


def test_consistency_link_broken(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    index = chain_index(c)
    names = sorted(index)
    proof = consistency_proof(c, _genesis(index))
    # Swap a mid-chain receipt's prev pointer: linkage claim != reality.
    mid = names[1]
    path, payload, digest = index[mid]
    payload["prev_epoch_receipt"] = "corpus_epoch_" + "f" * 16 + ".json"
    path.write_text(json.dumps(payload))
    errors = verify_consistency(proof, c)
    assert errors  # digest mismatch + link broken both fire


def test_consistency_to_must_be_head(tmp_path: Path) -> None:
    c = _corpus(tmp_path)
    index = chain_index(c)
    names = sorted(index)
    # Extend past the proof's `to` — the proof's endpoint is no longer head.
    proof = consistency_proof(c, _genesis(index), to_receipt=names[-1])
    (c / "r99.json").write_text(json.dumps({"z": 9}))
    write_epoch_receipt(corpus_epoch(c), c)
    errors = verify_consistency(proof, c)
    assert "to_not_chain_head" in errors


def test_consistency_member_digest_binding(tmp_path: Path) -> None:
    """The successor's member map must pin the predecessor's file bytes —
    linkage alone could point at a file that isn't the recorded hop."""
    c = _corpus(tmp_path)
    index = chain_index(c)
    proof = consistency_proof(c, _genesis(index))
    hops = [h["name"] for h in proof["hops"]]
    succ = hops[1]
    path, payload, _ = index[succ]
    for m in payload["members"]:
        if m["name"] == hops[0]:
            m["sha256"] = "0" * 64  # member pin no longer binds the hop bytes
    path.write_text(json.dumps(payload))
    errors = verify_consistency(proof, c)
    # The successor's own bytes changed → its digest mismatches too.
    assert any(e.startswith("hop_digest_mismatch") for e in errors)
    assert f"hop_member_digest_mismatch:{succ}" in errors


def _md_corpus(tmp_path: Path, n_stamps: int = 3) -> Path:
    """``*.md`` corpus (like ``verifier/``): the glob excludes the epoch
    receipts, so no successor can pin its predecessor as a member."""
    c = tmp_path / "verifier"
    c.mkdir()
    (c / "r1.md").write_text("# one\n")
    for i in range(n_stamps):
        (c / f"r{i + 2}.md").write_text(f"# {i}\n")
        write_epoch_receipt(corpus_epoch(c, pattern="*.md"), c)
    return c


def _rewrite_genesis(c: Path, *, reseal: bool) -> tuple[str, str]:
    """Edit the genesis receipt in place; return (genesis, successor)."""
    index = chain_index(c, pattern="*.md")
    first = _genesis(index)
    payload = json.loads((c / first).read_text())
    payload["forged_marker"] = True
    if reseal:
        unsigned = {k: v for k, v in payload.items() if k != "receipt_sha256"}
        payload["receipt_sha256"] = hash_bytes(canonical_json_bytes(unsigned))
    (c / first).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    succ = next(n for n, (_, e, _) in index.items() if e.get("prev_epoch_receipt") == first)
    return first, succ


def test_consistency_non_member_glob_binds_through_seal(tmp_path: Path) -> None:
    c = _md_corpus(tmp_path)
    index = chain_index(c, pattern="*.md")
    proof = consistency_proof(c, _genesis(index), pattern="*.md")
    assert proof["n_hops"] == 3
    assert verify_consistency(proof, c) == []


def test_consistency_non_member_glob_rejects_broken_seal(tmp_path: Path) -> None:
    """A predecessor edited without resealing keeps its name, but its seal
    no longer verifies — even a proof built over the edited tree fails."""
    c = _md_corpus(tmp_path)
    first, succ = _rewrite_genesis(c, reseal=False)
    proof = consistency_proof(c, first, pattern="*.md")
    assert f"hop_member_digest_mismatch:{succ}" in verify_consistency(proof, c)


def test_consistency_non_member_glob_rejects_resealed_rename(tmp_path: Path) -> None:
    """Resealing the edit makes the seal verify, but the file name no longer
    embeds it — the name is what the successor's prev link commits to."""
    c = _md_corpus(tmp_path)
    first, succ = _rewrite_genesis(c, reseal=True)
    proof = consistency_proof(c, first, pattern="*.md")
    assert f"hop_member_digest_mismatch:{succ}" in verify_consistency(proof, c)


def test_consistency_member_glob_never_falls_back_to_seal(tmp_path: Path) -> None:
    """Where the glob covers receipts, a dropped member pin is an error even
    though the predecessor's seal is intact."""
    c = _corpus(tmp_path)
    index = chain_index(c)
    proof = consistency_proof(c, _genesis(index))
    succ = proof["hops"][1]["name"]
    path, payload, _ = index[succ]
    payload["members"] = [m for m in payload["members"] if m["name"] != proof["hops"][0]["name"]]
    path.write_text(json.dumps(payload))
    assert f"hop_member_digest_mismatch:{succ}" in verify_consistency(proof, c)


def test_verifier_coverage_ratchet() -> None:
    """Every integrity-critical research module is a pinned jewel — a new
    verifier file that dodges DEFAULT_JEWELS fails here, not in review."""
    from quant_fund.research.crown_jewels import verifier_coverage_errors

    repo_root = Path(__file__).resolve().parents[3]
    assert verifier_coverage_errors(repo_root) == []


def test_verifier_coverage_catches_unpinned(tmp_path: Path) -> None:
    from quant_fund.research.crown_jewels import verifier_coverage_errors

    src = tmp_path / "src" / "quant_fund" / "research"
    src.mkdir(parents=True)
    (src / "epoch_consistency.py").write_text("# not pinned")
    errs = verifier_coverage_errors(tmp_path, jewels=())
    assert errs == ["verifier_unpinned:src/quant_fund/research/epoch_consistency.py"]
