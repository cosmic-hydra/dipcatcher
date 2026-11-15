"""auditor_bundle: zero-trust verification against the live Rekor record.

The honest-path tests need the committed quality/ tree (checkpoint, pins,
witness proof) and are skipped when the substrate isn't present — the same
skip pattern as test_integrity_witness.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY = REPO_ROOT / "quality"

_REQUIRES = [
    QUALITY / "checkpoint.json",
    QUALITY / "epoch_heads.json",
    QUALITY / "crown_jewels.json",
    QUALITY / "gate_signing.pub",
    REPO_ROOT / "gate_pins.sig",
    QUALITY / "witness_signing.pub",
    QUALITY / "rekor_pubkey.pem",
]
requires_tree = pytest.mark.skipif(
    not all(p.is_file() for p in _REQUIRES) or not list((QUALITY / "witness").glob("*.json")),
    reason="integrity substrate not committed",
)


def _build(tmp_path: Path) -> Path:
    from quant_fund.research.auditor_bundle import build_bundle

    return build_bundle(REPO_ROOT, tmp_path / "bundle.json")


@requires_tree
def test_build_bundle_contains_all_members(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import (
        BUNDLE_MEMBERS,
        BUNDLE_SCHEMA,
        OPTIONAL_MEMBERS,
    )

    bundle = json.loads(_build(tmp_path).read_text())
    assert bundle["schema"] == BUNDLE_SCHEMA
    assert set(BUNDLE_MEMBERS) <= set(bundle["files"])
    # spine members + declared optionals ride along; nothing else
    assert not (set(bundle["files"]) - set(BUNDLE_MEMBERS) - set(OPTIONAL_MEMBERS)) - {
        r
        for r in bundle["files"]
        if r.startswith(
            (
                "quality/checkpoints/",
                "quality/witness/",
                "quality/rotation_",
                "quality/quorum_rotations/",
                "quality/key_reset_",
            )
        )
    }
    assert bundle["witness_proof"]["schema"] == "integrity_witness.v1"
    # every member hash is declared
    assert set(bundle["files_sha256"]) == set(bundle["files"])


@requires_tree
def test_verify_bundle_honest_pinned_key(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    path = _build(tmp_path)
    res = verify_bundle(path, rekor_url=None)
    assert res["ok"], res["errors"]
    assert res["log_index"]


@requires_tree
def test_verify_bundle_caller_pinned_key(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(
        _build(tmp_path),
        rekor_pubkey_pem=(QUALITY / "rekor_pubkey.pem").read_bytes(),
    )
    assert res["ok"], res["errors"]


@requires_tree
def test_verify_bundle_tampered_checkpoint_fails(tmp_path: Path) -> None:
    """Mutate a member AND fix its declared digest: the deeper chain —
    checkpoint bytes must hash to the logged Rekor digest — must catch it."""
    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    raw = bytearray(base64.b64decode(bundle["files"]["quality/checkpoint.json"]))
    raw[8] ^= 0x01
    mutated = bytes(raw)
    bundle["files"]["quality/checkpoint.json"] = base64.b64encode(mutated).decode()
    import hashlib

    bundle["files_sha256"]["quality/checkpoint.json"] = hashlib.sha256(mutated).hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert any("witness" in e or "checkpoint" in e for e in res["errors"])


@requires_tree
def test_verify_bundle_swapped_witness_key_fails(tmp_path: Path) -> None:
    """A bundle carrying a foreign witness pubkey must fail even when its
    own declared digests are patched — the Rekor entry records the real key."""
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    evil = (
        ec.generate_private_key(ec.SECP256R1())
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    bundle["files"]["quality/witness_signing.pub"] = base64.b64encode(evil).decode()
    import hashlib

    bundle["files_sha256"]["quality/witness_signing.pub"] = hashlib.sha256(evil).hexdigest()
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert "witness_pubkey_diverges_from_log" in res["errors"]


@requires_tree
def test_verify_bundle_missing_member_fails(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    bundle = json.loads(_build(tmp_path).read_text())
    del bundle["files"]["quality/epoch_heads.json"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(bundle))
    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert "missing:quality/epoch_heads.json" in res["errors"]


def test_verify_bundle_rejects_malformed(tmp_path: Path) -> None:
    from quant_fund.research.auditor_bundle import verify_bundle

    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    assert verify_bundle(bad)["errors"] == ["bundle_malformed"]
    bad.write_text(json.dumps({"schema": "x"}))
    assert verify_bundle(bad)["errors"] == ["schema_mismatch"]
    bad.write_text(json.dumps({"schema": "auditor_bundle.v1", "files": {}, "witness_proof": {}}))
    res = verify_bundle(bad)
    assert not res["ok"]
    assert res["errors"][0].startswith("missing:")


def test_bundle_contract_errors() -> None:
    from quant_fund.research.auditor_bundle import bundle_contract_errors

    assert bundle_contract_errors({}) != []
    ok_body = {
        "schema": "auditor_bundle.v1",
        "files": {
            rel: base64.b64encode(b"x").decode()
            for rel in (
                "quality/checkpoint.json",
                "quality/epoch_heads.json",
                "quality/crown_jewels.json",
                "quality/gate_signing.pub",
                "gate_pins.sig",
                "quality/witness_signing.pub",
                "quality/rekor_pubkey.pem",
            )
        },
        "witness_proof": {},
    }
    import hashlib

    ok_body["files_sha256"] = {rel: hashlib.sha256(b"x").hexdigest() for rel in ok_body["files"]}
    assert bundle_contract_errors(ok_body) == []


@requires_tree
def test_standalone_verifier_agrees_with_library(tmp_path: Path) -> None:
    """The independent scripts/ implementation must reach the same verdict as
    the library verifier on honest AND tampered bundles — divergence between
    the two implementations is itself a finding."""
    import subprocess
    import sys

    from quant_fund.research.auditor_bundle import verify_bundle

    script = REPO_ROOT / "scripts/verify_auditor_bundle.py"
    bundle = _build(tmp_path)

    res = verify_bundle(bundle, rekor_url=None)
    proc = subprocess.run(
        [sys.executable, str(script), str(bundle), "--offline"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res["ok"] and proc.returncode == 0, proc.stderr + proc.stdout

    import hashlib

    tampered = json.loads(bundle.read_text())
    raw = bytearray(base64.b64decode(tampered["files"]["quality/epoch_heads.json"]))
    raw[0] ^= 0x01
    tampered["files"]["quality/epoch_heads.json"] = base64.b64encode(bytes(raw)).decode()
    tampered["files_sha256"]["quality/epoch_heads.json"] = hashlib.sha256(bytes(raw)).hexdigest()
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(tampered))
    res2 = verify_bundle(bad, rekor_url=None)
    proc2 = subprocess.run(
        [sys.executable, str(script), str(bad), "--offline"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert not res2["ok"] and proc2.returncode == 1


@requires_tree
def test_bundle_carries_full_spine(tmp_path: Path) -> None:
    """The bundle ships every archived checkpoint + witness proof so an
    auditor verifies the whole chain, not just the head."""
    bundle = _build(tmp_path)
    body = json.loads(bundle.read_text())
    spine = [r for r in body["files"] if r.startswith("quality/checkpoints/")]
    proofs = [r for r in body["files"] if r.startswith("quality/witness/")]
    assert len(spine) >= 3  # archive has history on this branch
    assert len(proofs) >= 3
    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(bundle, rekor_url=None)
    assert res["ok"], res["errors"]
    assert res["spine_members"] == len(spine)
    assert res["witness_proofs"] == len(proofs)


@requires_tree
def test_bundle_interior_spine_drop_fails_both_implementations(tmp_path: Path) -> None:
    """Dropping an interior link orphans everything below it — both the
    library and the standalone script must fail."""
    import base64
    import hashlib
    import subprocess
    import sys

    bundle = _build(tmp_path)
    body = json.loads(bundle.read_text())
    # Pick a NON-genesis archive member (one that has a prev on the spine):
    # walk from the head, drop the second link.
    records = {}
    for rel, b64 in body["files"].items():
        if rel.startswith("quality/checkpoints/") or rel == "quality/checkpoint.json":
            raw = base64.b64decode(b64)
            records[hashlib.sha256(raw).hexdigest()] = (rel, raw)
    head = hashlib.sha256(base64.b64decode(body["files"]["quality/checkpoint.json"])).hexdigest()
    order = []
    cur: str | None = head
    while cur in records:
        order.append(cur)
        cur = json.loads(records[cur][1]).get("payload", {}).get("prev_sha256")
    assert len(order) >= 3
    interior = records[order[1]][0]  # first archived link
    body["files"].pop(interior)
    body["files_sha256"].pop(interior)
    bad = tmp_path / "interior_dropped.json"
    bad.write_text(json.dumps(body))

    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(bad, rekor_url=None)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/verify_auditor_bundle.py"),
            str(bad),
            "--offline",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert not res["ok"]
    assert any("orphan" in e for e in res["errors"])
    assert proc.returncode == 1, proc.stdout


@requires_tree
def test_bundle_rejects_unexpected_member(tmp_path: Path) -> None:
    """Members outside the pinned prefixes are fail-closed."""
    import base64

    bundle = _build(tmp_path)
    body = json.loads(bundle.read_text())
    body["files"]["unexpected/extra.json"] = base64.b64encode(b"{}").decode()
    bad = tmp_path / "extra.json"
    bad.write_text(json.dumps(body))

    from quant_fund.research.auditor_bundle import verify_bundle

    res = verify_bundle(bad, rekor_url=None)
    assert not res["ok"]
    assert any("unexpected_member" in e for e in res["errors"])


# --- Quorum-era drills against the standalone verifier's internals ---------
#
# The bundle's Rekor proof can't be fabricated offline, so these exercises
# _verify_spine / _verify_quorum_rotations directly on synthetic bundle-file
# maps: v1-era genesis -> registry era -> authorized rotation -> swapped
# registry attacks, mirroring the library's checkpoint_chain continuity rules.


def _sign(priv_hex: str, msg: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).sign(msg).hex()


def _kid(pub_hex: str) -> str:
    import hashlib

    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]


def _registry(*pubs: str, threshold: int = 1) -> dict:
    return {
        "schema": "gate_quorum.v1",
        "threshold": threshold,
        "keys": [{"key_id": _kid(p), "pubkey": p} for p in pubs],
    }


def _canon_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _v1_checkpoint(priv: str, pub: str, prev_sha: str | None = None) -> bytes:
    payload: dict = {"schema": "integrity_checkpoint.v1"}
    if prev_sha is not None:
        payload["prev_sha256"] = prev_sha
    return json.dumps(
        {
            "schema": "integrity_checkpoint_sig.v1",
            "algorithm": "ed25519",
            "key_id": _kid(pub),
            "signature": _sign(priv, _canon_bytes(payload)),
            "payload": payload,
        }
    ).encode()


def _v2_checkpoint(pairs: list[tuple[str, str]], prev_sha: str, registry: dict) -> bytes:
    from scripts.verify_auditor_bundle import _registry_sha256

    payload = {
        "schema": "integrity_checkpoint.v2",
        "prev_sha256": prev_sha,
        "quorum": {
            "registry_sha256": _registry_sha256(registry),
            "threshold": registry["threshold"],
            "n_keys": len(registry["keys"]),
        },
    }
    return json.dumps(
        {
            "schema": "integrity_checkpoint_sig.v2",
            "algorithm": "ed25519",
            "signatures": [
                {"key_id": _kid(pub), "signature": _sign(priv, _canon_bytes(payload))}
                for priv, pub in pairs
            ],
            "payload": payload,
        }
    ).encode()


def _rotation(signers: list[tuple[str, str]], prev_reg: dict, new_reg: dict) -> bytes:
    from scripts.verify_auditor_bundle import _registry_sha256

    payload = {
        "schema": "quorum_rotation.v1",
        "prev_registry": prev_reg,
        "registry": new_reg,
        "prev_registry_sha256": _registry_sha256(prev_reg),
        "registry_sha256": _registry_sha256(new_reg),
        "reason": "test rotation",
    }
    return json.dumps(
        {
            "schema": "quorum_rotation.v1",
            "payload": payload,
            "signatures": [
                {"key_id": _kid(pub), "signature": _sign(priv, _canon_bytes(payload))}
                for priv, pub in signers
            ],
        }
    ).encode()


def _spine_files(
    head: bytes,
    archives: dict[str, bytes],
    rotations: dict[str, bytes] | None = None,
    registry: dict | None = None,
) -> tuple[dict[str, str], dict[str, str], bytes]:
    """files/declared/decoded_checkpoint for _verify_spine."""
    import hashlib

    files: dict[str, str] = {"quality/checkpoint.json": base64.b64encode(head).decode()}
    for name, raw in archives.items():
        files[f"quality/checkpoints/{name}"] = base64.b64encode(raw).decode()
    for name, raw in (rotations or {}).items():
        files[f"quality/quorum_rotations/{name}"] = base64.b64encode(raw).decode()
    if registry is not None:
        canon = json.dumps(registry, indent=2, sort_keys=True) + "\n"
        files["quality/gate_quorum.json"] = base64.b64encode(canon.encode()).decode()
    declared = {rel: hashlib.sha256(base64.b64decode(b)).hexdigest() for rel, b in files.items()}
    return files, declared, head


def _rot_name(rotation_raw: bytes) -> str:

    d = json.loads(rotation_raw)["payload"]["registry_sha256"]
    return f"rotation_{d[:16]}.json"


def test_spine_quorum_rotation_era_chain_verifies() -> None:
    """Honest rotation: v1 genesis -> era R0 -> rotation R0->R1 -> era R1."""
    from scripts.verify_auditor_bundle import _registry_sha256, _verify_spine

    from quant_fund.research.gate_signatures import generate_keypair

    g1_priv, g1_pub = generate_keypair()
    q_priv, q_pub = generate_keypair()
    n_priv, n_pub = generate_keypair()
    r0 = _registry(g1_pub, q_pub, threshold=1)
    r1 = _registry(n_pub, threshold=1)
    d0, d1 = _registry_sha256(r0), _registry_sha256(r1)

    cp0 = _v1_checkpoint(g1_priv, g1_pub)
    cp1 = _v2_checkpoint([(g1_priv, g1_pub)], _sha_hex(cp0), r0)
    rot = _rotation([(g1_priv, g1_pub)], r0, r1)
    cp2 = _v2_checkpoint([(n_priv, n_pub)], _sha_hex(cp1), r1)

    files, declared, head = _spine_files(
        cp2,
        {"cp0.json": cp0, "cp1.json": cp1},
        {_rot_name(rot): rot},
        registry=r1,
    )
    errors: list[str] = []
    _verify_spine(files, declared, head, g1_pub, errors)
    assert errors == [], errors
    assert d0 != d1


def test_spine_flags_registry_swap_without_rotation() -> None:
    """Swap the live registry + forge a v2 head under it — no rotation record
    means spine_quorum_unauthorized even though the head self-verifies."""
    from scripts.verify_auditor_bundle import _verify_spine

    from quant_fund.research.gate_signatures import generate_keypair

    g1_priv, g1_pub = generate_keypair()
    n_priv, n_pub = generate_keypair()
    r0 = _registry(g1_pub, threshold=1)
    r1 = _registry(n_pub, threshold=1)

    cp0 = _v1_checkpoint(g1_priv, g1_pub)
    cp1 = _v2_checkpoint([(g1_priv, g1_pub)], _sha_hex(cp0), r0)
    forged = _v2_checkpoint([(n_priv, n_pub)], _sha_hex(cp1), r1)

    files, declared, head = _spine_files(forged, {"cp0.json": cp0, "cp1.json": cp1}, registry=r1)
    errors: list[str] = []
    _verify_spine(files, declared, head, g1_pub, errors)
    assert any(e.startswith("spine_quorum_unauthorized:") for e in errors), errors


def test_spine_flags_rotation_signed_by_wrong_era() -> None:
    """A rotation signed by the NEW quorum (not the outgoing one) can't
    authorize the swap — the record verifies but isn't legitimate."""
    from scripts.verify_auditor_bundle import _verify_spine

    from quant_fund.research.gate_signatures import generate_keypair

    g1_priv, g1_pub = generate_keypair()
    n_priv, n_pub = generate_keypair()
    r0 = _registry(g1_pub, threshold=1)
    r1 = _registry(n_pub, threshold=1)

    cp0 = _v1_checkpoint(g1_priv, g1_pub)
    cp1 = _v2_checkpoint([(g1_priv, g1_pub)], _sha_hex(cp0), r0)
    # Signed by n_priv — the INCOMING key, not in the predecessor registry.
    rot = _rotation([(n_priv, n_pub)], r0, r1)
    cp2 = _v2_checkpoint([(n_priv, n_pub)], _sha_hex(cp1), r1)

    files, declared, head = _spine_files(
        cp2,
        {"cp0.json": cp0, "cp1.json": cp1},
        {_rot_name(rot): rot},
        registry=r1,
    )
    errors: list[str] = []
    _verify_spine(files, declared, head, g1_pub, errors)
    assert any(e.startswith("rotation_signer_unknown:") for e in errors), errors
    assert any(e.startswith("spine_quorum_unauthorized:") for e in errors), errors


def test_spine_flags_registry_reverted_behind_terminus() -> None:
    """Live registry set back to R0 while the chain's tip is R1 — the
    terminus check catches a stale-registry rollback."""
    from scripts.verify_auditor_bundle import _verify_spine

    from quant_fund.research.gate_signatures import generate_keypair

    g1_priv, g1_pub = generate_keypair()
    n_priv, n_pub = generate_keypair()
    r0 = _registry(g1_pub, threshold=1)
    r1 = _registry(n_pub, threshold=1)

    cp0 = _v1_checkpoint(g1_priv, g1_pub)
    cp1 = _v2_checkpoint([(g1_priv, g1_pub)], _sha_hex(cp0), r0)
    rot = _rotation([(g1_priv, g1_pub)], r0, r1)
    cp2 = _v2_checkpoint([(n_priv, n_pub)], _sha_hex(cp1), r1)

    # Live registry presented as the OLD r0 — the chain's tip is r1.
    files, declared, head = _spine_files(
        cp2,
        {"cp0.json": cp0, "cp1.json": cp1},
        {_rot_name(rot): rot},
        registry=r0,
    )
    errors: list[str] = []
    _verify_spine(files, declared, head, g1_pub, errors)
    assert "quorum_registry_not_terminus" in errors, errors


def test_spine_flags_genesis_without_member_overlap() -> None:
    """A registry that shares no signer with the preceding era's records is
    a grafted quorum, not an introduction."""
    from scripts.verify_auditor_bundle import _verify_spine

    from quant_fund.research.gate_signatures import generate_keypair

    g1_priv, g1_pub = generate_keypair()
    alien_priv, alien_pub = generate_keypair()
    alien_reg = _registry(alien_pub, threshold=1)

    cp0 = _v1_checkpoint(g1_priv, g1_pub)
    forged = _v2_checkpoint([(alien_priv, alien_pub)], _sha_hex(cp0), alien_reg)

    files, declared, head = _spine_files(forged, {"cp0.json": cp0}, registry=alien_reg)
    errors: list[str] = []
    _verify_spine(files, declared, head, g1_pub, errors)
    assert any(e.startswith("spine_quorum_genesis_discontinuous:") for e in errors), errors


def _sha_hex(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()
