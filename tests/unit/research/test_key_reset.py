"""Tests for key_reset — the audited custody break when signing keys are lost.

A reset is the one key-set change the outgoing keys cannot sign. These tests
pin what is still checkable: the boundary is real history under the lost
keys, the new keys are held, the spine crosses eras only at the boundary,
lost keys never sign again, new keys never signed before — and the library
and the standalone bundle verifier reach the same verdict every time.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from scripts.verify_auditor_bundle import _parse_reset, _verify_reset_spine, _verify_spine

from quant_fund.research import key_reset as key_reset_module
from quant_fund.research.checkpoint_chain import chain_contract_errors, checkpoint_spine
from quant_fund.research.gate_signatures import (
    generate_keypair,
    init_quorum,
    key_id,
    registry_sha256,
    sign_pins,
)
from quant_fund.research.integrity_checkpoint import (
    checkpoint_state,
    verify_checkpoint,
    write_checkpoint,
)
from quant_fund.research.key_reset import (
    RESET_SCHEMA,
    build_reset,
    lineage_verdict,
    load_resets,
    parse_reset,
    reset_keys,
    reset_spine_errors,
)
from quant_fund.research.key_rotation import rotation_contract_errors, verify_rotations
from quant_fund.research.quorum_rotation import (
    quorum_rotation_contract_errors,
    verify_quorum_rotations,
)
from quant_fund.utils.hashing import canonical_json_bytes

Pair = tuple[str, str]

REASON = "gate key lost in a test sandbox"
WHO = "test lead"
AT = "2026-11-15T00:00:00+00:00"


# -- helpers -----------------------------------------------------------------


def _sig(priv: str, payload: Any) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv))
    return key.sign(canonical_json_bytes(payload)).hex()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _registry(pubs: list[str], threshold: int = 1) -> dict[str, Any]:
    return {
        "schema": "gate_quorum.v1",
        "threshold": threshold,
        "keys": [{"key_id": key_id(p), "pubkey": p} for p in pubs],
    }


def _quorum_claim(registry: dict[str, Any]) -> dict[str, Any]:
    return {
        "registry_sha256": registry_sha256(registry),
        "threshold": registry["threshold"],
        "n_keys": len(registry["keys"]),
    }


def _p256_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pub = ec.generate_private_key(ec.SECP256R1()).public_key()
    return pub.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()


def _p384_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pub = ec.generate_private_key(ec.SECP384R1()).public_key()
    return pub.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()


def _base(root: Path, gate: Pair) -> None:
    """Minimal pinned repo (same fixture shape as test_quorum_rotation)."""
    q = root / "quality"
    q.mkdir(parents=True)
    (q / "crown_jewels.json").write_text('{"pin": "crown"}\n')
    heads = {"receipts/*.json": {"receipt": "corpus_epoch_abc123.json", "sha256": "0" * 64}}
    (q / "epoch_heads.json").write_text(
        json.dumps({"schema": "epoch_heads.v1", "heads": heads}) + "\n"
    )
    sign_pins(root, *gate)


def _strip_quorum_field(root: Path, gate: Pair) -> None:
    """Rewrite the head as a v2 record minted before ``payload.quorum``."""
    head = root / "quality/checkpoint.json"
    body = json.loads(head.read_bytes())
    del body["payload"]["quorum"]
    body["signatures"] = [{"key_id": key_id(gate[1]), "signature": _sig(gate[0], body["payload"])}]
    head.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def _tree(tmp_path: Path, *, v1: int = 0, prefield: int = 0, quorum: int = 3) -> tuple[Path, Pair]:
    """A checkpointed tree signed by one gate key that is also the 1-of-1
    registry: ``v1`` single-signer records, then ``prefield`` v2 records
    without the quorum claim, then ``quorum`` v2 records claiming it — the
    committed tree's own history shape."""
    gate = generate_keypair()
    _base(tmp_path, gate)
    for _ in range(v1):
        write_checkpoint(tmp_path, [gate])
    if prefield or quorum:
        init_quorum(tmp_path, [gate[1]], threshold=1)
    for _ in range(prefield):
        write_checkpoint(tmp_path, [gate])
        _strip_quorum_field(tmp_path, gate)
    for _ in range(quorum):
        write_checkpoint(tmp_path, [gate])
    return tmp_path, gate


def _reset(root: Path, *new: Pair, **kwargs: Any) -> Path:
    return reset_keys(root, list(new), reason=REASON, authorized_by=WHO, at=AT, **kwargs)


def _archive_head(root: Path) -> bytes:
    head = root / "quality/checkpoint.json"
    raw = head.read_bytes()
    digest = _sha(raw)
    archive = root / "quality/checkpoints"
    archive.mkdir(exist_ok=True)
    (archive / f"{digest[:16]}_{digest[16:24]}.json").write_bytes(raw)
    return raw


def _forge_successor(root: Path, pairs: list[Pair], claim: dict[str, Any] | None) -> None:
    """``write_checkpoint`` without its registry checks: archive the head and
    sign a fresh state claiming ``claim``'s registry with arbitrary keys."""
    _archive_head(root)
    state = checkpoint_state(root)
    state["quorum"] = None if claim is None else _quorum_claim(claim)
    body = {
        "schema": "integrity_checkpoint_sig.v2",
        "algorithm": "ed25519",
        "signatures": [
            {"key_id": key_id(pub), "signature": _sig(priv, state)} for priv, pub in pairs
        ],
        "payload": state,
    }
    (root / "quality/checkpoint.json").write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


def _library(root: Path) -> dict[str, dict[str, Any]]:
    return {
        "rotations": verify_rotations(root),
        "quorum": verify_quorum_rotations(root),
        "spine": checkpoint_spine(root),
    }


def _library_errors(root: Path) -> list[str]:
    return [e for result in _library(root).values() for e in result["errors"]]


def _standalone_errors(root: Path, *, archives: bool = True) -> list[str]:
    """The standalone verifier's spine walk over the tree's bundle members
    (``archives=False`` strips the archived predecessors, as a truncated
    bundle would)."""
    q = root / "quality"
    paths = [
        q / "checkpoint.json",
        *(sorted((q / "checkpoints").glob("*.json")) if archives else []),
        *sorted(q.glob("key_reset_*.json")),
        *sorted((q / "quorum_rotations").glob("*.json")),
    ]
    for optional in ("gate_quorum.json", "witness_signing.pub"):
        if (q / optional).is_file():
            paths.append(q / optional)
    files = {
        p.relative_to(root).as_posix(): base64.b64encode(p.read_bytes()).decode() for p in paths
    }
    declared = {rel: _sha(base64.b64decode(b64)) for rel, b64 in files.items()}
    errors: list[str] = []
    gate_pub = (q / "gate_signing.pub").read_text().strip()
    _verify_spine(files, declared, (q / "checkpoint.json").read_bytes(), gate_pub, errors)
    return errors


def _assert_reset_verified(root: Path, n_resets: int = 1) -> None:
    res = _library(root)
    assert res["rotations"]["ok"], res["rotations"]["errors"]
    assert res["rotations"]["verdict"] == "reset"
    assert res["rotations"]["n_resets"] == n_resets
    assert res["rotations"]["n_rotations"] == 0
    assert [link["kind"] for link in res["rotations"]["lineage"]] == ["reset"] * n_resets
    assert rotation_contract_errors(res["rotations"]) == []
    assert res["quorum"]["ok"], res["quorum"]["errors"]
    assert res["quorum"]["verdict"] == "reset"
    assert res["quorum"]["n_resets"] == n_resets
    assert quorum_rotation_contract_errors(res["quorum"]) == []
    assert res["spine"]["ok"], res["spine"]["errors"]
    assert res["spine"]["verdict"] == "intact"
    assert res["spine"]["n_resets"] == n_resets
    assert sum(n.startswith("key_reset:") for n in res["spine"]["notes"]) == n_resets
    assert chain_contract_errors(res["spine"]) == []
    assert _standalone_errors(root) == []


# -- ceremony ----------------------------------------------------------------


def test_reset_round_trip(tmp_path: Path) -> None:
    root, gate = _tree(tmp_path)
    boundary = (root / "quality/checkpoint.json").read_bytes()
    new = generate_keypair()
    out = _reset(root, new)

    assert out.name == f"key_reset_{key_id(new[1])}.json"
    body = json.loads(out.read_bytes())
    assert body["schema"] == RESET_SCHEMA
    payload = body["payload"]
    assert payload["boundary"]["checkpoint_sha256"] == _sha(boundary)
    assert base64.b64decode(payload["boundary"]["checkpoint_b64"]) == boundary
    assert payload["lost"]["gate_key_id"] == key_id(gate[1])
    assert payload["new"]["gate_key_id"] == key_id(new[1])
    # The new key set is installed; the record names the registry exactly.
    assert (root / "quality/gate_signing.pub").read_text().strip() == new[1]
    live = json.loads((root / "quality/gate_quorum.json").read_bytes())
    assert registry_sha256(live) == payload["new"]["registry_sha256"]

    write_checkpoint(root, [new])
    _assert_reset_verified(root)
    assert verify_checkpoint(root)["ok"]


def test_reset_verifies_before_the_first_new_checkpoint(tmp_path: Path) -> None:
    """Right after the reset the head IS the boundary — still consistent."""
    root, _gate = _tree(tmp_path)
    _reset(root, generate_keypair())
    _assert_reset_verified(root)


def test_reset_over_the_committed_tree_shape(tmp_path: Path) -> None:
    """v1 genesis -> pre-field v2 -> quorum-claiming v2 -> reset -> new era."""
    root, _gate = _tree(tmp_path, v1=2, prefield=2, quorum=2)
    new = generate_keypair()
    _reset(root, new)
    write_checkpoint(root, [new])
    write_checkpoint(root, [new])
    _assert_reset_verified(root)


def test_reset_from_a_v1_only_tree(tmp_path: Path) -> None:
    """No registry yet: the lost side is the gate key as a 1-of-1 registry."""
    root, gate = _tree(tmp_path, v1=3, quorum=0)
    assert not (root / "quality/gate_quorum.json").exists()
    new = generate_keypair()
    out = _reset(root, new)
    assert json.loads(out.read_bytes())["payload"]["lost"]["registry"] == _registry([gate[1]])
    _assert_reset_verified(root)
    write_checkpoint(root, [new])  # the first v2 record, under the new registry
    _assert_reset_verified(root)


def test_double_reset_chains(tmp_path: Path) -> None:
    root, _gate = _tree(tmp_path)
    first, second = generate_keypair(), generate_keypair()
    _reset(root, first)
    write_checkpoint(root, [first])
    _reset(root, second)
    write_checkpoint(root, [second])
    _assert_reset_verified(root, n_resets=2)


def test_reset_to_a_two_of_three_registry(tmp_path: Path) -> None:
    root, _gate = _tree(tmp_path)
    new = [generate_keypair() for _ in range(3)]
    _reset(root, *new, threshold=2)
    write_checkpoint(root, new[:2])
    _assert_reset_verified(root)


def test_reset_records_and_installs_the_witness_key(tmp_path: Path) -> None:
    root, _gate = _tree(tmp_path)
    old_pem, new_pem = _p256_pem(), _p256_pem()
    (root / "quality/witness_signing.pub").write_text(old_pem)
    out = _reset(root, generate_keypair(), witness_pubkey_pem=new_pem)
    payload = json.loads(out.read_bytes())["payload"]
    assert payload["lost"]["witness_pubkey_pem"] == old_pem
    assert payload["new"]["witness_pubkey_pem"] == new_pem
    assert (root / "quality/witness_signing.pub").read_text() == new_pem


def test_reset_without_a_witness_key_keeps_the_installed_one(tmp_path: Path) -> None:
    """No replacement means the witness key is not declared lost: the record
    names none on either side and the installed key stays."""
    root, _gate = _tree(tmp_path)
    pem, new = _p256_pem(), generate_keypair()
    (root / "quality/witness_signing.pub").write_text(pem)
    out = _reset(root, new)
    payload = json.loads(out.read_bytes())["payload"]
    assert payload["lost"]["witness_pubkey_pem"] is None
    assert payload["new"]["witness_pubkey_pem"] is None
    assert (root / "quality/witness_signing.pub").read_text() == pem
    write_checkpoint(root, [new])
    _assert_reset_verified(root)


def test_reset_refuses_to_reinstall_the_lost_witness_key(tmp_path: Path) -> None:
    root, gate = _tree(tmp_path)
    pem = _p256_pem()
    (root / "quality/witness_signing.pub").write_text(pem)
    with pytest.raises(ValueError, match="reuses_lost_witness_key"):
        _reset(root, generate_keypair(), witness_pubkey_pem=pem.replace("\n", "\r\n"))
    assert load_resets(root) == []
    assert (root / "quality/gate_signing.pub").read_text().strip() == gate[1]


def test_reset_keys_refusals(tmp_path: Path) -> None:
    root, gate = _tree(tmp_path)
    with pytest.raises(ValueError, match="at least one new key"):
        _reset(root)
    with pytest.raises(ValueError, match="reuses_lost_key"):
        _reset(root, gate)
    with pytest.raises(ValueError, match="new_witness_pubkey_malformed"):
        _reset(root, generate_keypair(), witness_pubkey_pem=_p384_pem())
    # Nothing was written by a refused reset.
    assert load_resets(root) == []
    assert (root / "quality/gate_signing.pub").read_text().strip() == gate[1]

    lost = {
        p: (root / p).read_bytes() for p in ("quality/gate_signing.pub", "quality/gate_quorum.json")
    }
    new = generate_keypair()
    _reset(root, new)
    for rel, raw in lost.items():  # roll the key files back, keep the record
        (root / rel).write_bytes(raw)
    with pytest.raises(ValueError, match="already exists"):
        _reset(root, new)


def test_build_reset_needs_a_checkpoint(tmp_path: Path) -> None:
    gate = generate_keypair()
    _base(tmp_path, gate)
    with pytest.raises(ValueError, match="checkpoint.json"):
        build_reset(tmp_path, [generate_keypair()], reason=REASON, authorized_by=WHO)


def test_cli_key_reset(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from quant_fund.cli.main import app

    root, _gate = _tree(tmp_path / "repo")
    new = generate_keypair()
    key_file = tmp_path / "new.key"
    key_file.write_text(f"{new[0]}:{new[1]}\n")
    args = ["key-reset", "--root", str(root), "--reason", REASON, "--authorized-by", WHO]
    runner = CliRunner()

    missing = runner.invoke(app, args)
    assert missing.exit_code == 2, missing.output

    done = runner.invoke(app, [*args, "--key-file", str(key_file)])
    assert done.exit_code == 0, done.output
    assert f"key_reset_{key_id(new[1])}.json" in done.output

    verified = runner.invoke(app, ["verify-rotations", "--root", str(root)])
    assert verified.exit_code == 0, verified.output
    assert "resets=1 verdict=reset" in verified.output

    # The installed key is now the live one: resetting to it again reuses it.
    again = runner.invoke(app, [*args, "--key-file", str(key_file)])
    assert again.exit_code == 2, again.output
    assert "reuses_lost_key" in again.output


# -- attacks on the tree -----------------------------------------------------


def test_tampered_reset_fails_closed(tmp_path: Path) -> None:
    """An invalid record contributes no key: the new era is unauthorized."""
    root, _gate = _tree(tmp_path)
    new = generate_keypair()
    out = _reset(root, new)
    write_checkpoint(root, [new])
    body = json.loads(out.read_bytes())
    body["payload"]["reason"] = "rewritten after signing"
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")

    res = _library(root)
    assert not res["rotations"]["ok"] and not res["quorum"]["ok"] and not res["spine"]["ok"]
    assert f"key_reset_possession_below_quorum:0/1:{out.name}" in res["spine"]["errors"]
    assert res["rotations"]["verdict"] != "reset"
    standalone = _standalone_errors(root)
    assert f"key_reset_possession_below_quorum:0/1:'quality/{out.name}'" in standalone
    assert any(e.startswith("spine_quorum_registry_unknown:") for e in standalone)


def test_off_boundary_crossing_is_rejected(tmp_path: Path) -> None:
    """A reset bound to an older checkpoint cannot authorize a crossing after
    later lost-key records — the reset never re-authorizes a splice."""
    root, _gate = _tree(tmp_path)
    head = root / "quality/checkpoint.json"
    head_raw = head.read_bytes()
    parent = json.loads(head_raw)["payload"]["prev_sha256"]
    archived = next(
        p for p in (root / "quality/checkpoints").glob("*.json") if _sha(p.read_bytes()) == parent
    )
    head.write_bytes(archived.read_bytes())  # reset bound to the parent
    new = generate_keypair()
    _reset(root, new)
    head.write_bytes(head_raw)
    write_checkpoint(root, [new])

    errors = _library_errors(root)
    child = _sha(head.read_bytes())[:12]
    assert f"key_reset_off_boundary:{child}" in errors
    assert f"key_reset_revoked_signer:{_sha(head_raw)[:12]}" in errors
    standalone = _standalone_errors(root)
    assert f"spine_key_reset_off_boundary:{child}" in standalone
    assert f"spine_key_reset_revoked_signer:{_sha(head_raw)[:12]}" in standalone


def test_lost_key_signing_after_the_boundary_is_revoked(tmp_path: Path) -> None:
    root, gate = _tree(tmp_path)
    lost_registry = json.loads((root / "quality/gate_quorum.json").read_bytes())
    _reset(root, generate_keypair())
    _forge_successor(root, [gate], lost_registry)  # the "lost" key keeps signing

    forged = _sha((root / "quality/checkpoint.json").read_bytes())[:12]
    res = _library(root)
    assert not res["spine"]["ok"]
    assert f"key_reset_revoked_signer:{forged}" in res["spine"]["errors"]
    assert f"key_reset_revoked_registry:{forged}" in res["spine"]["errors"]
    standalone = _standalone_errors(root)
    assert f"spine_key_reset_revoked_signer:{forged}" in standalone
    assert f"spine_key_reset_revoked_registry:{forged}" in standalone


def test_new_key_that_signed_before_the_boundary_is_rejected(tmp_path: Path) -> None:
    """New keys must be fresh: one that co-signed lost-era history is not."""
    root, gate = _tree(tmp_path)
    lost_registry = json.loads((root / "quality/gate_quorum.json").read_bytes())
    new = generate_keypair()
    _forge_successor(root, [gate, new], lost_registry)  # boundary co-signed by new
    _reset(root, new)
    write_checkpoint(root, [new])

    assert f"key_reset_resurrects:{key_id(new[1])}" in checkpoint_spine(root)["errors"]
    assert f"spine_key_reset_resurrects:{key_id(new[1])}" in _standalone_errors(root)


def _relabel_head_signature(root: Path, index: int) -> str:
    """Give one head signature a key id no registry knows; returns the new
    head digest."""
    head = root / "quality/checkpoint.json"
    body = json.loads(head.read_bytes())
    body["signatures"][index]["key_id"] = "f" * 16
    head.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return _sha(head.read_bytes())


def test_relabelled_lost_key_signature_after_the_boundary_is_rejected(tmp_path: Path) -> None:
    """A lost key's signature hidden behind an unknown key id still fails:
    a spine record meeting its quorum may carry no unverifiable signature."""
    root, gate = _tree(tmp_path)
    new = generate_keypair()
    _reset(root, new)
    _forge_successor(
        root, [new, gate], json.loads((root / "quality/gate_quorum.json").read_bytes())
    )
    digest = _relabel_head_signature(root, 1)
    write_checkpoint(root, [new])

    assert (
        f"signature_invalid:{digest[:12]}:signature_key_unknown" in checkpoint_spine(root)["errors"]
    )
    assert f"spine_signature_key_unknown:{digest[:12]}" in _standalone_errors(root)


def test_relabelled_new_key_signature_before_the_boundary_is_rejected(tmp_path: Path) -> None:
    root, gate = _tree(tmp_path)
    lost_registry = json.loads((root / "quality/gate_quorum.json").read_bytes())
    new = generate_keypair()
    _forge_successor(root, [gate, new], lost_registry)
    digest = _relabel_head_signature(root, 1)
    _reset(root, new)
    write_checkpoint(root, [new])

    assert (
        f"signature_invalid:{digest[:12]}:signature_key_unknown" in checkpoint_spine(root)["errors"]
    )
    assert f"spine_signature_key_unknown:{digest[:12]}" in _standalone_errors(root)


def test_boundary_off_spine_is_flagged_by_both_verifiers() -> None:
    reset = {"file": "key_reset_x.json", "rel": "quality/key_reset_x.json", "boundary": "f" * 64}
    assert reset_spine_errors([reset], {}, ["a" * 64]) == [
        "key_reset_boundary_off_spine:key_reset_x.json"
    ]
    errors: list[str] = []
    _verify_reset_spine([reset], ["a" * 64], {}, {}, errors, {}, None)
    assert errors == ["spine_key_reset_boundary_off_spine:'quality/key_reset_x.json'"]


def _members_world(root: Path) -> tuple[Pair, Pair, Pair]:
    """Gate + a dormant member (1-of-2), three checkpoints, a reset to
    ``first`` and one checkpoint under it."""
    gate, dormant, first = (generate_keypair() for _ in range(3))
    _base(root, gate)
    init_quorum(root, [gate[1], dormant[1]], threshold=1)
    for _ in range(3):
        write_checkpoint(root, [gate])
    _reset(root, first)
    write_checkpoint(root, [first])
    return gate, dormant, first


def test_a_key_lost_by_an_earlier_reset_stays_revoked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dormant member was declared lost with the first registry; a later
    reset that registers it again is refused, and a forced one leaves every
    record claiming that registry revoked in both verifiers."""
    _gate, dormant, _first = _members_world(tmp_path)
    second = generate_keypair()
    with pytest.raises(ValueError, match=f"reinstates_lost_key:{key_id(dormant[1])}"):
        _reset(tmp_path, second, dormant)
    assert len(load_resets(tmp_path)) == 1
    monkeypatch.setattr(key_reset_module, "_reinstated_refusals", lambda *_args: [])
    _reset(tmp_path, second, dormant)
    write_checkpoint(tmp_path, [second])
    head = _sha((tmp_path / "quality/checkpoint.json").read_bytes())[:12]
    assert f"key_reset_revoked_registry:{head}" in checkpoint_spine(tmp_path)["errors"]
    assert f"spine_key_reset_revoked_registry:{head}" in _standalone_errors(tmp_path)


def test_two_resets_without_reinstatement_verify(tmp_path: Path) -> None:
    """Control for the scenario above: fresh keys at the second reset."""
    _members_world(tmp_path)
    second = generate_keypair()
    _reset(tmp_path, second)
    write_checkpoint(tmp_path, [second])
    _assert_reset_verified(tmp_path, n_resets=2)


def test_a_witness_key_lost_by_an_earlier_reset_stays_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """W1 -> W2 -> W1 across two resets: each record alone is fine (its new
    key differs from its own lost key), but W1 was declared lost first."""
    root, _gate = _tree(tmp_path)
    w1, w2 = _p256_pem(), _p256_pem()
    (root / "quality/witness_signing.pub").write_text(w1)
    first, second = generate_keypair(), generate_keypair()
    _reset(root, first, witness_pubkey_pem=w2)
    write_checkpoint(root, [first])
    with pytest.raises(ValueError, match="reinstates_lost_witness_key"):
        _reset(root, second, witness_pubkey_pem=w1)
    monkeypatch.setattr(key_reset_module, "_reinstated_refusals", lambda *_args: [])
    out = _reset(root, second, witness_pubkey_pem=w1.replace("\n", "\r\n"))
    write_checkpoint(root, [second])
    library = checkpoint_spine(root)["errors"]
    assert f"key_reset_reinstates_witness:{out.name}" in library
    assert "key_reset_reinstates_witness:live" in library
    script = _standalone_errors(root)
    assert f"spine_key_reset_reinstates_witness:{ascii('quality/' + out.name)}" in script
    assert "spine_key_reset_reinstates_witness:live" in script


@pytest.mark.parametrize(
    "wrap",
    [
        lambda pem: pem.encode(),
        lambda pem: pem.replace("\n", "\r\n").encode(),
        # Not UTF-8, yet the witness verifier still loads the key from it.
        lambda pem: b"\xff\n" + pem.encode(),
        lambda pem: pem.encode() + b"\xfe",
    ],
    ids=["verbatim", "crlf", "leading_non_utf8", "trailing_non_utf8"],
)
def test_reinstalling_a_lost_witness_key_by_hand_is_flagged(
    tmp_path: Path, wrap: Callable[[str], bytes]
) -> None:
    root, _gate = _tree(tmp_path)
    w1, w2 = _p256_pem(), _p256_pem()
    (root / "quality/witness_signing.pub").write_text(w1)
    new = generate_keypair()
    _reset(root, new, witness_pubkey_pem=w2)
    write_checkpoint(root, [new])
    _assert_reset_verified(root)
    (root / "quality/witness_signing.pub").write_bytes(wrap(w1))
    assert "key_reset_reinstates_witness:live" in checkpoint_spine(root)["errors"]
    assert "spine_key_reset_reinstates_witness:live" in _standalone_errors(root)


def test_an_unparseable_installed_witness_key_does_not_crash_the_spine(tmp_path: Path) -> None:
    root, _gate = _tree(tmp_path)
    (root / "quality/witness_signing.pub").write_text(_p256_pem())
    new = generate_keypair()
    _reset(root, new, witness_pubkey_pem=_p256_pem())
    write_checkpoint(root, [new])
    (root / "quality/witness_signing.pub").write_bytes(b"\xff\xfe not a key")
    assert not any("reinstates_witness" in e for e in checkpoint_spine(root)["errors"])
    assert not any("reinstates_witness" in e for e in _standalone_errors(root))


def test_a_bundle_stripped_of_its_archives_is_not_pre_spine(tmp_path: Path) -> None:
    """The head signs its archive extent and a reset is bundled: dropping
    the archived predecessors must not skip the spine walk."""
    root, _gate = _tree(tmp_path)
    new = generate_keypair()
    _reset(root, new)
    write_checkpoint(root, [new])
    assert _standalone_errors(root) == []
    n = len(list((root / "quality/checkpoints").glob("*.json")))
    errors = _standalone_errors(root, archives=False)
    assert f"spine_archive_count:{n}!=0" in errors
    assert "spine_tip_mismatch" in errors
    assert any(e.startswith("spine_dangling_prev:") for e in errors)
    assert any(e.startswith("spine_key_reset_boundary_off_spine:") for e in errors)


# -- record parsing: library and standalone verifier agree ---------------------


def _side(gate_pub: str, registry: dict[str, Any], witness: str | None = None) -> dict[str, Any]:
    return {
        "gate_pubkey": gate_pub,
        "gate_key_id": key_id(gate_pub),
        "registry": registry,
        "registry_sha256": registry_sha256(registry),
        "witness_pubkey_pem": witness,
    }


def _v1(pair: Pair) -> bytes:
    payload = {"schema": "integrity_checkpoint.v1", "prev_sha256": "0" * 64}
    body = {
        "schema": "integrity_checkpoint_sig.v1",
        "algorithm": "ed25519",
        "key_id": key_id(pair[1]),
        "signature": _sig(pair[0], payload),
        "payload": payload,
    }
    return json.dumps(body).encode()


def _v2(pairs: list[Pair], claim: dict[str, Any] | None) -> bytes:
    payload: dict[str, Any] = {"schema": "integrity_checkpoint.v2", "prev_sha256": "0" * 64}
    if claim is not None:
        payload["quorum"] = _quorum_claim(claim)
    body = {
        "schema": "integrity_checkpoint_sig.v2",
        "algorithm": "ed25519",
        "signatures": [
            {"key_id": key_id(pub), "signature": _sig(priv, payload)} for priv, pub in pairs
        ],
        "payload": payload,
    }
    return json.dumps(body).encode()


def _boundary(raw: bytes) -> dict[str, str]:
    return {"checkpoint_sha256": _sha(raw), "checkpoint_b64": base64.b64encode(raw).decode()}


def _record(payload: dict[str, Any], signers: list[Pair], **top: Any) -> bytes:
    body = {
        "schema": RESET_SCHEMA,
        "algorithm": "ed25519",
        "payload": payload,
        "signatures": [
            {"key_id": key_id(pub), "signature": _sig(priv, payload)} for priv, pub in signers
        ],
    }
    body.update(top)
    return (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()


class _World:
    """Lost gate A (1-of-1 registry), new keys B and C, stranger X."""

    def __init__(self) -> None:
        self.a, self.b, self.c, self.x = (generate_keypair() for _ in range(4))
        self.lost_registry = _registry([self.a[1]])
        self.lost = _side(self.a[1], self.lost_registry)
        self.w1, self.w2 = _p256_pem(), _p256_pem()
        self.w1_crlf = self.w1.replace("\n", "\r\n")  # same key, other text

    def payload(
        self, *, boundary: bytes | None = None, new: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "at": AT,
            "reason": REASON,
            "authorized_by": WHO,
            "boundary": _boundary(
                boundary if boundary is not None else _v2([self.a], self.lost_registry)
            ),
            "lost": copy.deepcopy(self.lost),
            "new": new if new is not None else _side(self.b[1], _registry([self.b[1]])),
        }


def _honest(w: _World) -> bytes:
    return _record(w.payload(), [w.b])


def _edit(path: tuple[str, ...], value: Any) -> Callable[[_World], bytes]:
    """An honest record with one payload field replaced, re-signed by B."""

    def build(w: _World) -> bytes:
        payload = w.payload()
        node = payload
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value
        return _record(payload, [w.b])

    return build


def _tampered(w: _World) -> bytes:
    body = json.loads(_honest(w))
    body["payload"]["reason"] = "rewritten after signing"
    return json.dumps(body).encode()


def _witnesses(lost: str | None, new: str | None) -> Callable[[_World], bytes]:
    """An honest record whose witness keys are World attributes (or None)."""

    def build(w: _World) -> bytes:
        new_pem = None if new is None else getattr(w, new)
        payload = w.payload(new=_side(w.b[1], _registry([w.b[1]]), new_pem))
        payload["lost"]["witness_pubkey_pem"] = None if lost is None else getattr(w, lost)
        return _record(payload, [w.b])

    return build


def _registry_key_field(side: str, field: str, value: Any) -> Callable[[_World], bytes]:
    """An honest record whose ``side`` registry's first entry has ``field``
    replaced (unhashable values included), re-signed by B."""

    def build(w: _World) -> bytes:
        payload = w.payload()
        payload[side]["registry"]["keys"][0][field] = value
        return _record(payload, [w.b])

    return build


def _non_finite(value: float) -> Callable[[_World], bytes]:
    """Signed with ``extra: null``, then carrying NaN/Infinity there:
    canonical signing bytes map both to null, so a lax parser would still
    verify the signature over a record that says something else."""

    def build(w: _World) -> bytes:
        payload = w.payload()
        payload["extra"] = None
        body = json.loads(_record(payload, [w.b]))
        body["payload"]["extra"] = value
        return json.dumps(body).encode()

    return build


def _two_of_two(signers: str) -> Callable[[_World], bytes]:
    def build(w: _World) -> bytes:
        new = _side(w.b[1], _registry([w.b[1], w.c[1]], threshold=2))
        return _record(w.payload(new=new), [getattr(w, s) for s in signers])

    return build


CASES: dict[str, tuple[Callable[[_World], bytes], list[str] | Callable[[_World], list[str]]]] = {
    "honest": (_honest, []),
    "honest_v1_boundary": (lambda w: _record(w.payload(boundary=_v1(w.a)), [w.b]), []),
    "honest_prefield_boundary": (
        lambda w: _record(w.payload(boundary=_v2([w.a], None)), [w.b]),
        [],
    ),
    "honest_two_of_two": (_two_of_two("bc"), []),
    "honest_witness_replaced": (_witnesses("w1", "w2"), []),
    "honest_witness_introduced": (_witnesses(None, "w2"), []),
    "lost_witness_not_replaced": (_witnesses("w1", None), ["lost_witness_not_replaced"]),
    "reuses_lost_witness_key": (_witnesses("w1", "w1"), ["reuses_lost_witness_key"]),
    "reuses_lost_witness_key_rewrapped": (
        _witnesses("w1", "w1_crlf"),
        ["reuses_lost_witness_key"],
    ),
    "malformed": (lambda w: b"{not json", ["malformed"]),
    "nan": (_non_finite(float("nan")), ["malformed"]),
    "negative_infinity": (_non_finite(float("-inf")), ["malformed"]),
    "boundary_nan": (
        lambda w: _record(
            w.payload(boundary=b'{"schema": "integrity_checkpoint_sig.v2", "payload": {"x": NaN}}'),
            [w.b],
        ),
        ["boundary_malformed"],
    ),
    "lost_registry_key_id_unhashable": (
        _registry_key_field("lost", "key_id", []),
        ["lost_registry_malformed"],
    ),
    "new_registry_key_id_unhashable": (
        _registry_key_field("new", "key_id", {}),
        ["new_registry_malformed"],
    ),
    "schema_unexpected": (
        lambda w: _record(w.payload(), [w.b], schema="key_reset.v0"),
        ["schema_unexpected"],
    ),
    "algorithm_unexpected": (
        lambda w: _record(w.payload(), [w.b], algorithm="rsa"),
        ["schema_unexpected"],
    ),
    "reason_missing": (_edit(("reason",), "  "), ["reason_missing"]),
    "authorized_by_missing": (_edit(("authorized_by",), None), ["authorized_by_missing"]),
    "lost_missing": (_edit(("lost",), "gone"), ["lost_missing"]),
    "lost_gate_key_id_mismatch": (
        _edit(("lost", "gate_key_id"), "0" * 16),
        ["lost_gate_key_id_mismatch"],
    ),
    "new_gate_pubkey_malformed": (
        _edit(("new", "gate_pubkey"), "AB" * 32),
        ["new_gate_pubkey_malformed"],
    ),
    "new_registry_malformed": (
        _edit(("new", "registry", "threshold"), 0),
        ["new_registry_malformed"],
    ),
    "new_registry_digest_mismatch": (
        _edit(("new", "registry_sha256"), "0" * 64),
        ["new_registry_digest_mismatch"],
    ),
    "new_witness_p384": (
        lambda w: _edit(("new", "witness_pubkey_pem"), _p384_pem())(w),
        ["new_witness_pubkey_malformed"],
    ),
    "new_witness_garbage": (
        _edit(("new", "witness_pubkey_pem"), "not a pem"),
        ["new_witness_pubkey_malformed"],
    ),
    "boundary_missing": (_edit(("boundary",), None), ["boundary_malformed"]),
    "boundary_b64_malformed": (_edit(("boundary", "checkpoint_b64"), "!!"), ["boundary_malformed"]),
    "boundary_digest_mismatch": (
        _edit(("boundary", "checkpoint_sha256"), "0" * 64),
        ["boundary_digest_mismatch"],
    ),
    "boundary_not_checkpoint": (
        lambda w: _record(
            w.payload(boundary=json.dumps({"schema": "x", "payload": {}}).encode()), [w.b]
        ),
        ["boundary_not_checkpoint"],
    ),
    "boundary_signed_by_stranger": (
        lambda w: _record(w.payload(boundary=_v2([w.x], w.lost_registry)), [w.b]),
        ["boundary_not_lost_era"],
    ),
    "boundary_claims_other_registry": (
        lambda w: _record(w.payload(boundary=_v2([w.a], _registry([w.x[1]]))), [w.b]),
        ["boundary_not_lost_era"],
    ),
    "boundary_v1_other_key": (
        lambda w: _record(w.payload(boundary=_v1(w.x)), [w.b]),
        ["boundary_not_lost_era"],
    ),
    "reuses_lost_key": (
        lambda w: _record(w.payload(new=_side(w.b[1], _registry([w.b[1], w.a[1]]))), [w.b]),
        lambda w: [f"reuses_lost_key:{key_id(w.a[1])}"],
    ),
    "new_gate_is_lost_gate": (
        lambda w: _record(w.payload(new=_side(w.a[1], _registry([w.a[1]]))), [w.a]),
        lambda w: [f"reuses_lost_key:{key_id(w.a[1])}"],
    ),
    "new_gate_key_unregistered": (
        lambda w: _record(w.payload(new=_side(w.b[1], _registry([w.c[1]]))), [w.b, w.c]),
        ["new_gate_key_unregistered"],
    ),
    "possession_below_quorum": (_two_of_two("b"), ["possession_below_quorum:1/2"]),
    "possession_duplicate_signer": (_two_of_two("bb"), ["possession_below_quorum:1/2"]),
    "possession_gate_key_unproven": (
        lambda w: _record(w.payload(new=_side(w.b[1], _registry([w.b[1], w.c[1]]))), [w.c]),
        ["possession_gate_key_unproven"],
    ),
    "signed_only_by_lost_key": (
        lambda w: _record(w.payload(), [w.a]),
        ["possession_below_quorum:0/1", "possession_gate_key_unproven"],
    ),
    "tampered_after_signing": (
        _tampered,
        ["possession_below_quorum:0/1", "possession_gate_key_unproven"],
    ),
}


@pytest.fixture(scope="module")
def world() -> _World:
    return _World()


@pytest.mark.parametrize("case", sorted(CASES))
def test_record_verdicts_match_the_standalone_verifier(world: _World, case: str) -> None:
    build, expected = CASES[case]
    raw = build(world)
    want = expected(world) if callable(expected) else expected
    lib = parse_reset(raw, "x")
    summary, script_errors = _parse_reset(raw)
    assert lib["errors"] == want
    assert script_errors == want
    assert (summary is None) == bool(want)
    if summary is not None:
        assert summary["boundary"] == lib["boundary"]
        assert summary["lost_digest"] == lib["lost_digest"]
        assert summary["new_digest"] == lib["new_digest"]
        assert summary["lost_kids"] == lib["lost_kids"]
        assert summary["new_kids"] == lib["new_kids"]


def test_lineage_verdict_and_contracts() -> None:
    assert lineage_verdict([], 0) == "intact"
    assert lineage_verdict([], 2) == "reset"
    assert lineage_verdict(["x"], 2) == "broken"
    bogus: dict[str, Any] = {
        "ok": True,
        "verdict": "reset",
        "n_rotations": 0,
        "n_resets": 0,
        "lineage": [],
        "errors": [],
    }
    assert "reset_without_records" in rotation_contract_errors(bogus)
    bogus["schema"] = "quorum_rotations.v1"
    assert "reset_without_records" in quorum_rotation_contract_errors(bogus)
