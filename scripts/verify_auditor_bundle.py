#!/usr/bin/env python3
"""Standalone auditor-bundle verifier — deliberately independent.

Everything in ``quant_fund.research.auditor_bundle`` is verified code, but
a single verifier is itself a trust bottleneck: a subtle bug (or a silent
patch) in the repo's implementation could hide a forgery. This script is
the second implementation — it imports NOTHING from quant_fund, only the
stdlib + ``cryptography``. An auditor can read it top-to-bottom (~300
lines) and verify a bundle against the public Rekor log alone.

Usage:
    python scripts/verify_auditor_bundle.py auditor_bundle.json \
        [--rekor-pubkey rekor.pem] [--offline]

Exit code 0 = every link verified; 1 = any failure (each printed).

Chain verified (same semantics as verify_bundle, reimplemented):
  1. files_sha256 manifest over every b64 member
  2. Rekor inclusion: leaf = sha256(0x00 || entry-body-bytes); audit-path
     walk to the recorded rootHash; UUID tail must equal the leaf hash
  3. Rekor SET: ECDSA/SHA-256 over the JCS-canonical LogEntryAnon
     (sorted-key JSON of {body, integratedTime, logID, logIndex})
  4. Rekor checkpoint note: sig over payload+"\n" (4-byte key-hint prefix
     stripped from the note's signature blob)
  5. checkpoint sha256 == the digest inside the entry body
  6. witness signature over checkpoint bytes under the pubkey the LOG
     recorded as signer — which must equal the bundled witness pubkey
  7. checkpoint Ed25519 signature over its signed ``payload`` object
     (canonical JSON); checkpoint's ``pins`` must match the bundled pin
     files' sha256; crown_jewels must pin the gate pubkey's bytes;
     gate_pins.sig verifies over its own embedded payload whose declared
     digests must match the bundled pin files
  8. key_reset.v1 records (keys lost, new keys from a boundary checkpoint):
     each record re-verified in full — boundary bytes under the lost key
     set, possession under the new registry, disjoint key sets — then the
     boundary must sit on the spine, the first new-era checkpoint must link
     straight to it, and no lost key may sign (nor new key predate) it
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

REKOR_PUBLIC_KEY_URL = "https://rekor.sigstore.dev/api/v1/log/publicKey"

MEMBERS = (
    "quality/checkpoint.json",
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "quality/gate_signing.pub",
    "gate_pins.sig",
    "quality/witness_signing.pub",
    "quality/rekor_pubkey.pem",
)
# Optional literal members — bundled when present, never required.
OPTIONAL_MEMBERS = ("quality/gate_quorum.json",)

# Optional members: the archived checkpoint records and every committed
# Rekor proof. Carrying them lets this script verify the whole spine —
# every historical pin state — not just the head checkpoint.
SPINE_PREFIXES = (
    "quality/checkpoints/",
    "quality/witness/checkpoint.json_",
    "quality/rotation_",
    "quality/quorum_rotations/",
    "quality/key_reset_",
)
QUORUM_ROTATION_PREFIX = "quality/quorum_rotations/"
RESET_PREFIX = "quality/key_reset_"
_CHECKPOINT_SCHEMAS = ("integrity_checkpoint_sig.v1", "integrity_checkpoint_sig.v2")


def _sha(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _canon(value: Any) -> bytes:
    """canonical_json_bytes equivalent for plain JSON types."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _registry_sha256(registry: Any) -> str:
    """Canonical digest of a gate_quorum.v1 body: the on-disk bytes are
    json.dumps(indent=2, sort_keys) + newline — digest matches only when the
    committed bytes are byte-exact."""
    raw = json.dumps(registry, indent=2, sort_keys=True) + "\n"
    return _sha(raw.encode()).hex()


def _ecdsa_pem_verify(pem: bytes, sig_der: bytes, msg: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        pub = load_pem_public_key(pem)
        if not isinstance(pub, ec.EllipticCurvePublicKey):
            return False
        pub.verify(sig_der, msg, ec.ECDSA(hashes.SHA256()))
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _ed25519_verify(pub_hex: str, sig_hex: str, msg: bytes) -> bool:
    """Repo convention: Ed25519 keys/sigs are raw 32/64-byte hex, not PEM."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex.strip()))
        pub.verify(bytes.fromhex(sig_hex), msg)
    except (InvalidSignature, ValueError):
        return False
    return True


def _inclusion_walk(leaf: bytes, log_index: int, tree_size: int, path: list[bytes]) -> bytes:
    """RFC 6962 §2.1.1 audit path → reconstructed root."""
    fn, sn = log_index, tree_size - 1
    r = leaf
    for h in path:
        if (fn & 1) or fn == sn:
            r = _sha(b"\x01" + h + r)
            while fn and not (fn & 1):
                fn >>= 1
                sn >>= 1
        else:
            r = _sha(b"\x01" + r + h)
        fn >>= 1
        sn >>= 1
    return r


def _key_id(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()[:16]


def _verify_rotations(
    files: dict[str, str],
    declared: dict[str, str],
    gate_pub_hex: str,
    records: dict[str, bytes],
    errors: list[str],
    resets: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Dual-signed key lineage: old authorizes, new proves possession.

    Returns the keyring (key_id -> pub hex) covering every authorized key —
    live terminus + genesis + every intermediate link. A verified key reset
    is a link too (lost gate key -> new gate key): it joins the same chain
    shape, terminus and genesis-anchor checks.
    """
    ring = {_key_id(gate_pub_hex): gate_pub_hex}
    recs: list[tuple[str, dict[str, Any]]] = []
    for rel, b64 in files.items():
        if not rel.startswith("quality/rotation_"):
            continue
        try:
            raw = base64.b64decode(b64)
            body = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            errors.append(f"rotation_malformed:{ascii(rel)}")
            continue
        if declared.get(rel) != _sha(raw).hex():
            errors.append(f"files_sha256_mismatch:{rel}")
            continue
        if (
            not isinstance(body, dict)
            or body.get("schema") != "key_rotation.v1"
            or not isinstance(body.get("payload"), dict)
        ):
            errors.append(f"rotation_malformed:{ascii(rel)}")
            continue
        payload = body["payload"]
        ok = True
        for side in ("old", "new"):
            if not _ed25519_verify(
                str(payload.get(f"{side}_pubkey", "")),
                str(body.get(f"{side}_signature", "")),
                _canon(payload),
            ):
                errors.append(f"rotation_signature_invalid:{ascii(rel)}:{side}")
                ok = False
        if str(payload.get("old_key_id", "")) != _key_id(str(payload.get("old_pubkey", ""))) or str(
            payload.get("new_key_id", "")
        ) != _key_id(str(payload.get("new_pubkey", ""))):
            errors.append(f"rotation_key_id_mismatch:{ascii(rel)}")
            ok = False
        if ok:
            recs.append((rel, payload))
            ring.setdefault(_key_id(str(payload["old_pubkey"])), str(payload["old_pubkey"]))
            ring.setdefault(_key_id(str(payload["new_pubkey"])), str(payload["new_pubkey"]))
    for reset in resets or []:
        recs.append(
            (reset["rel"], {"old_pubkey": reset["lost_gate"], "new_pubkey": reset["new_gate"]})
        )
        ring.setdefault(_key_id(reset["lost_gate"]), reset["lost_gate"])
        ring.setdefault(_key_id(reset["new_gate"]), reset["new_gate"])
    if not recs:
        return ring

    claims: dict[str, list[str]] = {}
    for rel, payload in recs:
        claims.setdefault(str(payload.get("old_pubkey", "")), []).append(rel)
    for old, group in claims.items():
        if len(group) > 1:
            errors.append(f"rotation_fork:{_key_id(old)}")
    new_pubs = {str(p["new_pubkey"]) for _, p in recs}
    starts = [(r, p) for r, p in recs if str(p["old_pubkey"]) not in new_pubs]
    if len(starts) != 1:
        errors.append(f"rotation_chain_ambiguous:{len(starts)}")
        return ring
    cur_rel, cur = starts[0]
    seen: set[str] = set()
    terminus = None
    while True:
        new_pub = str(cur["new_pubkey"])
        terminus = new_pub
        if new_pub in seen:
            errors.append(f"rotation_cycle:{ascii(_key_id(new_pub))}")
            break
        seen.add(new_pub)
        nxt = claims.get(new_pub)
        if not nxt:
            break
        cur = next(p for r, p in recs if r == nxt[0])
    if terminus != gate_pub_hex:
        errors.append("rotation_live_key_not_terminus")

    # Genesis anchor: the first outgoing key must verify a spine record (or
    # be a reset's lost gate key that signed the reset's boundary).
    genesis_pub = str(starts[0][1]["old_pubkey"])
    anchored = any(
        r["lost_gate"] == genesis_pub and _signed_by(r["boundary_body"], genesis_pub)
        for r in resets or []
    )
    for raw in records.values():
        try:
            rec = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if _ed25519_verify(genesis_pub, str(rec.get("signature", "")), _canon(rec.get("payload"))):
            anchored = True
            break
    if not anchored:
        errors.append("rotation_unanchored_genesis")
    return ring


def _registered_map(registry: Any) -> tuple[dict[str, str], int]:
    """(key_id -> pub, threshold) from a gate_quorum.v1 body.

    Malformed bodies map to an impossible quorum so downstream signature
    checks fail closed rather than crash.
    """
    try:
        keys = registry["keys"]
        threshold = int(registry["threshold"])
        registered = {str(e["key_id"]): str(e["pubkey"]) for e in keys}
        if threshold < 1 or threshold > len(registered):
            raise ValueError
        for kid, pub in registered.items():
            if kid != _key_id(pub):
                raise ValueError
        return registered, threshold
    except (KeyError, TypeError, ValueError, AttributeError):
        return {}, 10**9


def _count_quorum_sigs(
    signatures: Any,
    registered: dict[str, str],
    threshold: int,
    payload: Any,
    errors: list[str],
    tag: str,
) -> int:
    """Count distinct valid Ed25519 signers over canonical payload bytes."""
    if not isinstance(signatures, list) or not signatures:
        errors.append(f"{tag}_signatures_missing")
        return 0
    valid: set[str] = set()
    for entry in signatures:
        if not isinstance(entry, dict):
            errors.append(f"{tag}_signature_malformed")
            continue
        kid = str(entry.get("key_id", ""))
        pub = registered.get(kid)
        if pub is None:
            errors.append(f"{tag}_signer_unknown:{kid}")
            continue
        if kid in valid:
            continue  # a replayed signer cannot satisfy a quorum twice
        if _ed25519_verify(pub, str(entry.get("signature", "")), _canon(payload)):
            valid.add(kid)
        else:
            errors.append(f"{tag}_signature_invalid:{kid}")
    if len(valid) < threshold:
        errors.append(f"{tag}_below_quorum:{len(valid)}/{threshold}")
    return len(valid)


# -- key resets ---------------------------------------------------------------
# key_reset.v1: the keys that signed history up to a boundary checkpoint are
# lost; a new key set signs everything after it. The lost keys cannot sign, so
# the reset's authority is the commit itself — what CAN be checked is that the
# boundary is real history under the lost keys, the new keys are held, and the
# spine only crosses eras at the boundary. Reimplements, never imports,
# quant_fund.research.key_reset.


def _hex_bytes(value: Any, n_bytes: int) -> bool:
    """Lowercase hex of exactly ``n_bytes`` bytes."""
    if not isinstance(value, str) or len(value) != 2 * n_bytes or value != value.lower():
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _registry_ok(registry: Any) -> bool:
    """A structurally clean gate_quorum.v1 body: 1 <= threshold <= n keys,
    each a 32-byte hex pubkey under its own sha256-derived key_id, no
    duplicates."""
    if not isinstance(registry, dict) or registry.get("schema") != "gate_quorum.v1":
        return False
    keys, threshold = registry.get("keys"), registry.get("threshold")
    if not isinstance(keys, list) or not keys or not isinstance(threshold, int):
        return False
    if not 1 <= threshold <= len(keys):
        return False
    seen: set[str] = set()
    for entry in keys:
        pub = entry.get("pubkey") if isinstance(entry, dict) else None
        if not isinstance(pub, str) or len(pub) != 64:
            return False
        try:
            kid = _key_id(pub)
        except ValueError:
            return False
        if entry.get("key_id") != kid or kid in seen:
            return False
        seen.add(kid)
    return True


def _witness_pem_ok(pem: Any) -> bool:
    """``None`` (no witness key recorded) or a PEM ECDSA P-256 public key."""
    if pem is None:
        return True
    if not isinstance(pem, str):
        return False
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    try:
        pub = load_pem_public_key(pem.encode())
    except (ValueError, TypeError, UnsupportedAlgorithm):
        return False
    return isinstance(pub, ec.EllipticCurvePublicKey) and isinstance(pub.curve, ec.SECP256R1)


def _valid_signer_ids(signatures: Any, registered: dict[str, str], payload: Any) -> set[str]:
    """Distinct registered key_ids whose signature over ``payload`` verifies."""
    if not isinstance(signatures, list):
        return set()
    valid: set[str] = set()
    for entry in signatures:
        kid = str(entry.get("key_id", "")) if isinstance(entry, dict) else ""
        pub = registered.get(kid)
        if pub is not None and _ed25519_verify(pub, str(entry.get("signature")), _canon(payload)):
            valid.add(kid)
    return valid


def _signed_by(body: Any, pub_hex: str) -> bool:
    """Any signature on a checkpoint record body verifies under ``pub_hex``."""
    if not isinstance(body, dict):
        return False
    sigs: list[Any] = [body.get("signature")]
    extra = body.get("signatures")
    if isinstance(extra, list):
        sigs += [e.get("signature") for e in extra if isinstance(e, dict)]
    msg = _canon(body.get("payload"))
    return any(_ed25519_verify(pub_hex, str(s), msg) for s in sigs if s is not None)


def _side_kids(side: dict[str, Any]) -> set[str]:
    return {str(side["gate_key_id"])} | {str(e["key_id"]) for e in side["registry"]["keys"]}


def _side_pubs(side: dict[str, Any]) -> set[str]:
    return {str(side["gate_pubkey"])} | {str(e["pubkey"]) for e in side["registry"]["keys"]}


def _reset_side_errors(side: Any, tag: str) -> list[str]:
    """One key set: gate key + id, registry + digest, optional witness key."""
    if not isinstance(side, dict):
        return [f"{tag}_missing"]
    errors: list[str] = []
    pub = side.get("gate_pubkey")
    if not _hex_bytes(pub, 32):
        errors.append(f"{tag}_gate_pubkey_malformed")
    elif side.get("gate_key_id") != _key_id(str(pub)):
        errors.append(f"{tag}_gate_key_id_mismatch")
    if not _registry_ok(side.get("registry")):
        errors.append(f"{tag}_registry_malformed")
    elif side.get("registry_sha256") != _registry_sha256(side["registry"]):
        errors.append(f"{tag}_registry_digest_mismatch")
    if not _witness_pem_ok(side.get("witness_pubkey_pem")):
        errors.append(f"{tag}_witness_pubkey_malformed")
    return errors


def _reset_boundary(boundary: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Decode the embedded boundary checkpoint; its digest must match."""
    if not isinstance(boundary, dict) or not _hex_bytes(boundary.get("checkpoint_sha256"), 32):
        return None, "boundary_malformed"
    try:
        raw = base64.b64decode(str(boundary.get("checkpoint_b64", "")), validate=True)
        body = json.loads(raw)
    except ValueError:  # binascii.Error, JSONDecodeError, UnicodeDecodeError
        return None, "boundary_malformed"
    if _sha(raw).hex() != boundary["checkpoint_sha256"]:
        return None, "boundary_digest_mismatch"
    if not isinstance(body, dict) or not isinstance(body.get("payload"), dict):
        return None, "boundary_malformed"
    if body.get("schema") not in _CHECKPOINT_SCHEMAS:
        return None, "boundary_not_checkpoint"
    return body, None


def _boundary_in_lost_era(body: dict[str, Any], lost: dict[str, Any]) -> bool:
    """The boundary verifies under the LOST key set: a v2 record reaching the
    lost registry's threshold (naming that registry, or predating the field),
    or a v1 record signed by the lost gate key."""
    payload = body["payload"]
    if body.get("schema") == "integrity_checkpoint_sig.v2":
        q = payload.get("quorum")
        claimed = q.get("registry_sha256") if isinstance(q, dict) else None
        if isinstance(claimed, str) and claimed != lost["registry_sha256"]:
            return False
        registered, threshold = _registered_map(lost["registry"])
        return len(_valid_signer_ids(body.get("signatures"), registered, payload)) >= threshold
    pub = str(lost["gate_pubkey"])
    return body.get("key_id") == _key_id(pub) and _ed25519_verify(
        pub, str(body.get("signature")), _canon(payload)
    )


def _witness_spki(pem: str) -> bytes:
    """DER SubjectPublicKeyInfo of a PEM public key (wrapping-independent)."""
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_public_key,
    )

    return load_pem_public_key(pem.encode()).public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )


def _reset_relation_errors(payload: dict[str, Any]) -> list[str]:
    """The new gate key is registered; the new set shares nothing with the
    lost; a witness key declared lost has a different replacement."""
    lost, new = payload["lost"], payload["new"]
    errors: list[str] = []
    registered, _threshold = _registered_map(new["registry"])
    if registered.get(str(new["gate_key_id"])) != new["gate_pubkey"]:
        errors.append("new_gate_key_unregistered")
    shared = (_side_kids(lost) & _side_kids(new)) | {
        _key_id(pub) for pub in _side_pubs(lost) & _side_pubs(new)
    }
    errors.extend(f"reuses_lost_key:{kid}" for kid in sorted(shared))
    lost_witness, new_witness = lost.get("witness_pubkey_pem"), new.get("witness_pubkey_pem")
    if lost_witness is not None:
        if new_witness is None:
            errors.append("lost_witness_not_replaced")
        elif _witness_spki(lost_witness) == _witness_spki(new_witness):
            errors.append("reuses_lost_witness_key")
    return errors


def _reset_authority_errors(body: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Boundary under the lost set; possession under the new registry —
    its threshold, the new gate key among the signers."""
    errors: list[str] = []
    boundary, _err = _reset_boundary(payload["boundary"])
    if boundary is None or not _boundary_in_lost_era(boundary, payload["lost"]):
        errors.append("boundary_not_lost_era")
    registered, threshold = _registered_map(payload["new"]["registry"])
    valid = _valid_signer_ids(body.get("signatures"), registered, payload)
    if len(valid) < threshold:
        errors.append(f"possession_below_quorum:{len(valid)}/{threshold}")
    if str(payload["new"]["gate_key_id"]) not in valid:
        errors.append("possession_gate_key_unproven")
    return errors


def _parse_reset(raw: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    """Fully verify one key_reset.v1 record -> (summary, []) or (None, errors)."""
    try:
        body = json.loads(raw)
    except ValueError:
        return None, ["malformed"]
    payload = body.get("payload") if isinstance(body, dict) else None
    if (
        not isinstance(payload, dict)
        or body.get("schema") != "key_reset.v1"
        or body.get("algorithm") != "ed25519"
    ):
        return None, ["schema_unexpected"]
    errors = [
        f"{field}_missing"
        for field in ("at", "reason", "authorized_by")
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip()
    ]
    errors += _reset_side_errors(payload.get("lost"), "lost")
    errors += _reset_side_errors(payload.get("new"), "new")
    boundary, boundary_error = _reset_boundary(payload.get("boundary"))
    errors += [boundary_error] if boundary_error is not None else []
    errors = errors or _reset_relation_errors(payload)
    errors = errors or _reset_authority_errors(body, payload)
    if errors:
        return None, errors
    lost, new = payload["lost"], payload["new"]
    return {
        "boundary": payload["boundary"]["checkpoint_sha256"],
        "boundary_body": boundary,
        "lost_gate": str(lost["gate_pubkey"]),
        "new_gate": str(new["gate_pubkey"]),
        "lost_digest": str(lost["registry_sha256"]),
        "new_digest": str(new["registry_sha256"]),
        "lost_registry": lost["registry"],
        "new_registry": new["registry"],
        "lost_kids": _side_kids(lost),
        "new_kids": _side_kids(new),
    }, []


def _verify_resets(
    files: dict[str, str], declared: dict[str, str], errors: list[str]
) -> list[dict[str, Any]]:
    """Every bundled key_reset.v1 record, re-verified; only valid ones count —
    an invalid reset contributes no key and no era (fail closed)."""
    resets: list[dict[str, Any]] = []
    for rel, b64 in sorted(files.items()):
        if not rel.startswith(RESET_PREFIX) or not rel.endswith(".json"):
            continue
        try:
            raw = base64.b64decode(b64)
        except (ValueError, TypeError):
            errors.append(f"key_reset_malformed:{ascii(rel)}")
            continue
        if declared.get(rel) != _sha(raw).hex():
            errors.append(f"files_sha256_mismatch:{rel}")
            continue
        reset, reset_errors = _parse_reset(raw)
        errors.extend(f"key_reset_{e}:{ascii(rel)}" for e in reset_errors)
        if reset is not None:
            reset["rel"] = rel
            resets.append(reset)
    return resets


def _reset_edge(
    resets: list[dict[str, Any]], parent: str, parent_era: str | None, child_era: str
) -> str | None:
    """Classify a spine edge whose registry era changes: ``None`` when no
    reset leads into ``child_era`` (an ordinary rotation must), ``"ok"`` when
    one does and ``parent`` is its boundary, ``"off_boundary"`` otherwise."""
    linked = [
        r for r in resets if r["new_digest"] == child_era and parent_era in (r["lost_digest"], None)
    ]
    if not linked:
        return None
    return "ok" if any(r["boundary"] == parent for r in linked) else "off_boundary"


def _verify_reset_spine(
    resets: list[dict[str, Any]],
    ordered: list[str],
    signers_of: dict[str, set[str]],
    era_of: dict[str, str | None],
    errors: list[str],
) -> None:
    """Each reset's boundary sits on the spine; its new keys signed nothing
    at or before the boundary; past it, no lost key signs and no record
    claims the lost registry."""
    position = {digest: i for i, digest in enumerate(ordered)}
    for r in resets:
        idx = position.get(r["boundary"])
        if idx is None:
            errors.append(f"spine_key_reset_boundary_off_spine:{ascii(r['rel'])}")
            continue
        used: set[str] = set()
        for digest in ordered[: idx + 1]:
            used |= signers_of.get(digest, set())
        errors.extend(f"spine_key_reset_resurrects:{kid}" for kid in sorted(r["new_kids"] & used))
        for digest in ordered[idx + 1 :]:
            if signers_of.get(digest, set()) & r["lost_kids"]:
                errors.append(f"spine_key_reset_revoked_signer:{digest[:12]}")
            if era_of.get(digest) == r["lost_digest"]:
                errors.append(f"spine_key_reset_revoked_registry:{digest[:12]}")


def _prefield_quorum(
    files: dict[str, str],
    resets: list[dict[str, Any]],
    quorum: tuple[dict[str, str], int] | None,
) -> tuple[dict[str, str], int] | None:
    """The registry v2 records minted before the quorum field (no era claim)
    verify under: the genesis of the committed lineage — well-formed quorum
    rotations plus verified resets — when it has exactly one root, else the
    live registry (on a tree that never changed registry, the same set)."""
    edges: list[tuple[str, str, Any]] = [
        (r["lost_digest"], r["new_digest"], r["lost_registry"]) for r in resets
    ]
    for rel, b64 in files.items():
        if not rel.startswith(QUORUM_ROTATION_PREFIX) or not rel.endswith(".json"):
            continue
        try:
            body = json.loads(base64.b64decode(b64))
            payload = body["payload"]
            prev_reg, new_reg = payload["prev_registry"], payload["registry"]
        except (ValueError, TypeError, KeyError):
            continue  # flagged by _verify_quorum_rotations
        if (
            body.get("schema") == "quorum_rotation.v1"
            and _registry_ok(prev_reg)
            and _registry_ok(new_reg)
            and payload.get("prev_registry_sha256") == _registry_sha256(prev_reg)
            and payload.get("registry_sha256") == _registry_sha256(new_reg)
        ):
            edges.append((_registry_sha256(prev_reg), _registry_sha256(new_reg), prev_reg))
    targets = {new_d for _prev, new_d, _reg in edges}
    roots = [reg for prev_d, _new, reg in edges if prev_d not in targets]
    return _registered_map(roots[0]) if len(roots) == 1 else quorum


def _verify_quorum_rotations(
    files: dict[str, str],
    declared: dict[str, str],
    ring: dict[str, str],
    live_registry_digest: str | None,
    records: dict[str, bytes],
    spine: list[str],
    errors: list[str],
    resets: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, tuple[dict[str, str], int]], set[tuple[str, str]]]:
    """quorum_rotation.v1 records: authorization + chain shape + era lineage.

    A registry swap is only valid if a rotation record signed by the
    OUTGOING quorum's threshold links the old digest to the new. Returns
    ``(lineage, authorized_pairs)``: ``digest -> (registered, threshold)``
    covering every registry the bundle can prove governed an era (live
    plus embedded predecessors), and the ``(prev_digest, digest)`` edges
    whose signatures actually reached the predecessor's threshold.

    Verified key resets are lineage edges too (lost -> new registry): they
    join the chain shape and the era lineage but never ``authorized_pairs``
    — a reset crossing is bound to its boundary checkpoint, which the spine
    continuity pass checks.
    """
    lineage: dict[str, tuple[dict[str, str], int]] = {}
    # rel, payload, prev_d, new_d, n_valid, prev_threshold
    recs: list[tuple[str, dict[str, Any], str, str, int, int]] = []
    for rel, b64 in files.items():
        if not rel.startswith(QUORUM_ROTATION_PREFIX) or not rel.endswith(".json"):
            continue
        try:
            raw = base64.b64decode(b64)
        except ValueError:
            errors.append(f"quorum_rotation_malformed:{ascii(rel)}")
            continue
        if declared.get(rel) != _sha(raw).hex():
            errors.append(f"files_sha256_mismatch:{rel}")
            continue
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"quorum_rotation_malformed:{ascii(rel)}")
            continue
        payload = body.get("payload") if isinstance(body, dict) else None
        if (
            not isinstance(body, dict)
            or body.get("schema") != "quorum_rotation.v1"
            or not isinstance(payload, dict)
        ):
            errors.append(f"quorum_rotation_malformed:{ascii(rel)}")
            continue
        prev_reg = payload.get("prev_registry")
        new_reg = payload.get("registry")
        if not isinstance(prev_reg, dict) or not isinstance(new_reg, dict):
            errors.append(f"quorum_rotation_malformed:{ascii(rel)}")
            continue
        prev_d = payload.get("prev_registry_sha256")
        new_d = payload.get("registry_sha256")
        if prev_d != _registry_sha256(prev_reg):
            errors.append(f"quorum_rotation_prev_digest_mismatch:{ascii(rel)}")
            continue
        if new_d != _registry_sha256(new_reg):
            errors.append(f"quorum_rotation_digest_mismatch:{ascii(rel)}")
            continue
        # Authorization: distinct valid signers under the PREDECESSOR registry.
        registered, threshold = _registered_map(prev_reg)
        n_valid = _count_quorum_sigs(
            body.get("signatures"), registered, threshold, payload, errors, "rotation"
        )
        recs.append((rel, payload, str(prev_d), str(new_d), n_valid, threshold))
        lineage.setdefault(str(prev_d), _registered_map(prev_reg))
        lineage.setdefault(str(new_d), _registered_map(new_reg))
    for reset in resets or []:
        # n_valid 0 < threshold 1: a reset edge never enters ``authorized``.
        recs.append((reset["rel"], {}, reset["lost_digest"], reset["new_digest"], 0, 1))
        lineage.setdefault(reset["lost_digest"], _registered_map(reset["lost_registry"]))
        lineage.setdefault(reset["new_digest"], _registered_map(reset["new_registry"]))

    # Chain shape: no two records may extend the same predecessor (fork), no
    # record may loop (cycle), and there must be a single root (a record
    # whose predecessor no other record produced) — else grafted chains
    # sit beside the real one undetected.
    by_prev: dict[str, list[str]] = {}
    for rel, _p, prev_d, new_d, _n, _t in recs:
        if prev_d == new_d:
            errors.append(f"quorum_rotation_cycle:{ascii(rel)}")
        by_prev.setdefault(prev_d, []).append(rel)
    for prev_d, group in by_prev.items():
        if len(group) > 1:
            errors.append(f"quorum_rotation_fork:{prev_d[:16]}")
    new_digests = {new_d for _r, _p, _pd, new_d, _n, _t in recs}
    roots = [r for r, _p, prev_d, _nd, _n, _t in recs if prev_d not in new_digests]
    if len(roots) > 1:
        errors.append(f"quorum_rotation_chain_ambiguous:{len(roots)}")
    # Terminus: the chain tip (a new_digest nobody extends) must be the live
    # registry's canonical digest — else the live file was swapped after the
    # last authorized rotation.
    tips = sorted(new_digests - set(by_prev))
    if live_registry_digest is not None and tips and live_registry_digest not in tips:
        errors.append("quorum_registry_not_terminus")
    # Genesis anchor: the root record's predecessor registry must have
    # actually governed — it appears as a spine record's claimed era, or
    # shares a key_id with the v1 signer ring, or is a verified reset's lost
    # registry (whose boundary reached that registry's threshold).
    spine_eras: set[str] = {r["lost_digest"] for r in resets or []}
    for digest in spine:
        raw = records.get(digest)
        if raw is None:
            continue
        try:
            payload = json.loads(raw).get("payload") or {}
        except (json.JSONDecodeError, AttributeError):
            continue
        quorum = payload.get("quorum")
        if isinstance(quorum, dict) and isinstance(quorum.get("registry_sha256"), str):
            spine_eras.add(str(quorum["registry_sha256"]))
    for rel, _p, prev_d, _nd, _n, _t in recs:
        if prev_d in new_digests:
            continue  # non-root
        if prev_d in spine_eras:
            continue  # a spine record claimed this era
        anchored = bool(set(lineage.get(prev_d, ({}, 0))[0]) & set(ring))
        if not anchored:
            errors.append(f"quorum_rotation_unanchored:{ascii(rel)}")
    authorized = {(prev_d, new_d) for _r, _p, prev_d, new_d, n, threshold in recs if n >= threshold}
    return lineage, authorized


def _verify_spine(
    files: dict[str, str],
    declared: dict[str, str],
    decoded_checkpoint: bytes,
    gate_pub_hex: str,
    errors: list[str],
) -> None:
    """Independent spine walk: chain links, signatures, forks, Rekor order.

    Independent of the library implementation — a divergence between this
    verdict and ``checkpoint_chain.checkpoint_spine`` is itself a finding.
    """
    for rel in files:
        if (
            rel not in MEMBERS
            and rel not in OPTIONAL_MEMBERS
            and not (any(rel.startswith(p) for p in SPINE_PREFIXES) and rel.endswith(".json"))
        ):
            errors.append(f"unexpected_member:{ascii(rel)}")
    records: dict[str, bytes] = {}
    witnessed: dict[str, int] = {}
    wit_times: dict[str, float] = {}
    for rel, b64 in files.items():
        if rel.startswith(SPINE_PREFIXES[1]) and rel.endswith(".json"):
            try:
                raw = base64.b64decode(b64)
            except ValueError:
                errors.append(f"b64_malformed:{ascii(rel)}")
                continue
            if declared.get(rel) != _sha(raw).hex():
                errors.append(f"files_sha256_mismatch:{rel}")
                continue
            try:
                proof = json.loads(raw)
            except json.JSONDecodeError:
                errors.append(f"spine_proof_malformed:{ascii(rel)}")
                continue
            digest = proof.get("target", {}).get("sha256")
            index = proof.get("rekor", {}).get("log_index")
            itime = proof.get("rekor", {}).get("integrated_time")
            if isinstance(digest, str) and isinstance(index, int):
                witnessed[digest] = index
                if isinstance(itime, (int, float)):
                    wit_times[digest] = float(itime)
        elif rel.startswith(SPINE_PREFIXES[0]):
            try:
                raw = base64.b64decode(b64)
            except ValueError:
                errors.append(f"b64_malformed:{ascii(rel)}")
                continue
            if declared.get(rel) != _sha(raw).hex():
                errors.append(f"files_sha256_mismatch:{rel}")
                continue
            records[_sha(raw).hex()] = raw
    # Key resets are re-verified even in a pre-spine bundle: an invalid
    # record is a finding wherever it sits.
    resets = _verify_resets(files, declared, errors)
    if not records and not witnessed:
        return  # pre-spine bundle — nothing more to check

    # The live checkpoint is a record too — it is the spine head.
    records[_sha(decoded_checkpoint).hex()] = decoded_checkpoint

    spine: list[str] = []
    seen: set[str] = set()
    cur: str | None = _sha(decoded_checkpoint).hex()
    while cur is not None:
        if cur in seen:
            errors.append(f"spine_cycle:{cur[:12]}")
            break
        seen.add(cur)
        spine_raw = records.get(cur)
        if spine_raw is None:
            if cur not in witnessed:
                errors.append(f"spine_dangling_prev:{cur[:12]}")
            break
        spine.append(cur)
        try:
            payload = json.loads(spine_raw).get("payload") or {}
        except (json.JSONDecodeError, AttributeError):
            errors.append(f"spine_malformed:{cur[:12]}")
            break
        prev = payload.get("prev_sha256")
        if prev is None:
            break  # genesis
        if not isinstance(prev, str) or len(prev) != 64:
            errors.append(f"spine_prev_malformed:{cur[:12]}")
            break
        cur = prev

    # A committed quorum registry retires the v1 single-signer envelope:
    # v2 records resolve every signature's key_id through the registry era
    # their payload claims — the live registry plus every predecessor a
    # quorum_rotation.v1 record can authenticate.
    quorum: tuple[dict[str, str], int] | None = None
    live_registry_digest: str | None = None
    qraw = files.get("quality/gate_quorum.json")
    if qraw is not None:
        try:
            registry = json.loads(base64.b64decode(qraw))
            qkeys = registry["keys"]
            quorum = (
                {str(e["key_id"]): str(e["pubkey"]) for e in qkeys},
                int(registry["threshold"]),
            )
            live_registry_digest = _registry_sha256(registry)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("quorum_registry_malformed")
            quorum = ({}, 10**9)

    # Every record's signature under the key authorized in its era.
    ring = _verify_rotations(files, declared, gate_pub_hex, records, errors, resets)
    lineage, authorized_rots = _verify_quorum_rotations(
        files, declared, ring, live_registry_digest, records, spine, errors, resets
    )
    if quorum is not None and live_registry_digest is not None:
        lineage.setdefault(live_registry_digest, quorum)
    prefield = _prefield_quorum(files, resets, quorum)

    # Per-record era claims for the continuity pass below: v2 records claim
    # a registry digest in payload.quorum; v1 records claim nothing (their
    # era is the key-rotation ring). signers_of keeps the record's signer
    # key_ids — the v1 key_id or a v2 signatures list — for the
    # genesis-continuity member-overlap check.
    era_of: dict[str, str | None] = {}
    signers_of: dict[str, set[str]] = {}
    for digest, raw in records.items():
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            errors.append(f"spine_malformed:{digest[:12]}")
            continue
        if body.get("schema") == "integrity_checkpoint_sig.v2":
            claimed = None
            qp = body.get("payload")
            if isinstance(qp, dict):
                q = qp.get("quorum")
                if isinstance(q, dict):
                    claimed = q.get("registry_sha256")
            era_of[digest] = claimed if isinstance(claimed, str) else None
            sig_list = body.get("signatures")
            signers_of[digest] = {
                str(e.get("key_id", ""))
                for e in (sig_list if isinstance(sig_list, list) else [])
                if isinstance(e, dict) and e.get("key_id")
            }
            if quorum is None:
                errors.append(f"spine_quorum_registry_missing:{digest[:12]}")
                continue
            # v2 records minted before the quorum field exist: they claim no
            # era and resolve under the lineage's genesis registry (the era
            # that signed them — the live one if the registry never changed);
            # the continuity pass treats them as era=None. ``prefield`` is
            # None only without a live registry, handled just above.
            if claimed is None:
                registered, threshold = prefield
            else:
                if not isinstance(claimed, str) or len(claimed) != 64:
                    errors.append(f"spine_quorum_unbound:{digest[:12]}")
                    continue
                era = lineage.get(claimed)
                if era is None:
                    errors.append(f"spine_quorum_registry_unknown:{claimed[:16]}")
                    continue
                registered, threshold = era
            sigs = body.get("signatures")
            if not isinstance(sigs, list) or not sigs:
                errors.append(f"spine_quorum_signatures_missing:{digest[:12]}")
                continue
            valid: set[str] = set()
            for entry in sigs:
                if not isinstance(entry, dict):
                    errors.append(f"spine_signature_malformed:{digest[:12]}")
                    continue
                kid = str(entry.get("key_id", ""))
                pub = registered.get(kid)
                if pub is None:
                    errors.append(f"spine_signature_key_unknown:{digest[:12]}")
                    continue
                if _ed25519_verify(
                    pub, str(entry.get("signature", "")), _canon(body.get("payload"))
                ):
                    valid.add(kid)
                else:
                    errors.append(f"spine_signature_invalid:{digest[:12]}")
            if len(valid) < threshold:
                errors.append(f"spine_quorum_below:{digest[:12]}:{len(valid)}/{threshold}")
            continue
        kid = str(body.get("key_id", ""))
        era_of[digest] = None
        signers_of[digest] = {kid} if kid else set()
        pub = ring.get(kid)
        if pub is None:
            errors.append(f"spine_signature_key_unknown:{digest[:12]}")
            continue
        if not _ed25519_verify(pub, str(body.get("signature", "")), _canon(body.get("payload"))):
            errors.append(f"spine_signature_invalid:{digest[:12]}")

    # Forks + orphans.
    claims: dict[str, int] = {}
    for raw in records.values():
        try:
            prev = (json.loads(raw).get("payload") or {}).get("prev_sha256")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(prev, str):
            claims[prev] = claims.get(prev, 0) + 1
    for prev, n in claims.items():
        if n > 1:
            errors.append(f"spine_fork:{prev[:12]}")
    head_digest = _sha(decoded_checkpoint).hex()
    for digest in records:
        if digest not in spine and digest != head_digest:
            errors.append(f"spine_orphan:{digest[:12]}")

    # Rekor order must follow the chain direction on witnessed links.
    ordered = list(reversed(spine))
    for parent, child in zip(ordered, ordered[1:], strict=False):  # adjacent pairs
        if parent in witnessed and child in witnessed and witnessed[child] <= witnessed[parent]:
            errors.append(f"spine_rekor_order:{child[:12]}")

    # Quorum era continuity: every registry-digest change along the spine
    # must be backed by a quorum_rotation.v1 record the OUTGOING registry's
    # threshold authorized. A registry swapped in without a rotation fails
    # here even when every record self-verifies under its own era.
    # A key reset's crossing is legal only out of its boundary checkpoint.
    for parent, child in zip(ordered, ordered[1:], strict=False):
        pd, cd = era_of.get(parent), era_of.get(child)
        if pd == cd or cd is None:
            continue
        reset_edge = _reset_edge(resets, parent, pd, cd)
        if reset_edge is not None:
            if reset_edge != "ok":
                errors.append(f"spine_key_reset_off_boundary:{child[:12]}")
            continue
        if pd is None:
            # v1 -> v2 transition (or a pre-field v2 parent): the child's
            # registry must contain one of the parent's signers — registry
            # introduction requires member overlap with the prior era.
            if not (signers_of.get(parent, set()) & set(lineage.get(cd, ({}, 0))[0])):
                errors.append(f"spine_quorum_genesis_discontinuous:{child[:12]}")
            continue
        if (pd, cd) not in authorized_rots:
            errors.append(f"spine_quorum_unauthorized:{child[:12]}")
    _verify_reset_spine(resets, ordered, signers_of, era_of, errors)

    # Witnessed-but-absent digests: grandfather pre-retention history
    # (integrated before the earliest recorded proof), flag the rest.
    recorded_times = [wit_times[d] for d in spine if d in wit_times]
    era_start = min(recorded_times) if recorded_times else None
    for digest in set(witnessed) - set(records):
        itime = wit_times.get(digest)
        if era_start is not None and itime is not None and itime < era_start:
            continue
        errors.append(f"spine_witnessed_absent:{digest[:12]}")

    # Witness extent: the live checkpoint's signed ``witness.proofs`` map
    # names every committed proof (name -> sha256). A dropped or byte-drifted
    # proof must surface even though its own file verified standalone.
    try:
        live_payload = json.loads(decoded_checkpoint).get("payload") or {}
    except json.JSONDecodeError:
        live_payload = {}
    witness_claim = live_payload.get("witness")
    if isinstance(witness_claim, dict):
        claimed_raw = witness_claim.get("proofs")
        claimed = (
            dict(claimed_raw)
            if isinstance(claimed_raw, dict)
            else {n: None for n in (claimed_raw or [])}
        )
        present = {
            rel.rsplit("/", 1)[-1]: rel
            for rel in files
            if rel.startswith(SPINE_PREFIXES[1]) and rel.endswith(".json")
        }
        missing = sorted(set(claimed) - set(present))
        if missing:
            errors.append(f"witness_proof_deleted:{','.join(missing)}")
        for name, claimed_digest in sorted(claimed.items()):
            member = present.get(name)
            if member is None or not isinstance(claimed_digest, str):
                continue
            try:
                raw_b = base64.b64decode(files[member])
            except ValueError:
                continue  # already flagged b64_malformed
            if _sha(raw_b).hex() != claimed_digest:
                errors.append(f"witness_proof_drift:{name}")
        extra = sorted(set(present) - set(claimed))
        if len(extra) > 1:
            errors.append(f"witness_proof_unpinned:{','.join(extra)}")
        elif len(extra) == 1:
            # The one legal extra is the proof for the live head.
            try:
                extra_body = json.loads(base64.b64decode(files[present[extra[0]]]))
                if (extra_body.get("target") or {}).get("sha256") != head_digest:
                    errors.append(f"witness_proof_unpinned:{extra[0]}")
            except (ValueError, json.JSONDecodeError):
                errors.append(f"witness_proof_unpinned:{extra[0]}")


def verify(bundle_path: Path, rekor_pem: bytes | None) -> list[str]:
    errors: list[str] = []
    try:
        bundle = json.loads(bundle_path.read_text())
    except (OSError, json.JSONDecodeError):
        return ["bundle_unreadable"]
    if bundle.get("schema") != "auditor_bundle.v1":
        return ["schema_mismatch"]

    files: dict[str, str] = bundle.get("files", {})
    declared: dict[str, str] = bundle.get("files_sha256", {})
    proof: dict[str, Any] = bundle.get("witness_proof", {})
    decoded: dict[str, bytes] = {}
    for rel in (*MEMBERS, *OPTIONAL_MEMBERS):
        b64 = files.get(rel)
        if b64 is None:
            if rel not in OPTIONAL_MEMBERS:
                errors.append(f"missing:{rel}")
            continue
        try:
            raw = base64.b64decode(b64)
        except ValueError:
            errors.append(f"b64_malformed:{ascii(rel)}")
            continue
        decoded[rel] = raw
        if declared.get(rel) != _sha(raw).hex():
            errors.append(f"files_sha256_mismatch:{rel}")
    if errors:
        return sorted(errors)

    rekor = proof.get("rekor", {})
    body_b64 = str(rekor.get("body_b64", ""))
    ip = rekor.get("inclusion_proof", {})
    if rekor_pem is None:
        rekor_pem = decoded["quality/rekor_pubkey.pem"]

    # -- Rekor entry -------------------------------------------------
    try:
        entry_body = json.loads(base64.b64decode(body_b64))
        spec = entry_body["spec"]
        logged_digest = spec["data"]["hash"]["value"]
        logged_sig = base64.b64decode(spec["signature"]["content"])
        logged_pub = base64.b64decode(spec["signature"]["publicKey"]["content"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        return errors + [f"entry_body_malformed:{exc}"]

    leaf = _sha(b"\x00" + base64.b64decode(body_b64))
    if not str(rekor.get("uuid", "")).endswith(leaf.hex()):
        errors.append("leaf_uuid_mismatch")
    try:
        root = _inclusion_walk(
            leaf,
            int(ip["log_index"]),
            int(ip["tree_size"]),
            [bytes.fromhex(h) for h in ip["hashes"]],
        )
        if root.hex() != ip["root_hash"]:
            errors.append("inclusion_root_mismatch")
    except (KeyError, TypeError, ValueError):
        errors.append("inclusion_proof_malformed")

    canon = json.dumps(
        {
            "body": body_b64,
            "integratedTime": rekor.get("integrated_time"),
            "logID": rekor.get("log_id"),
            "logIndex": rekor.get("log_index"),
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    try:
        set_sig = base64.b64decode(str(rekor.get("signed_entry_timestamp", "")))
    except ValueError:
        set_sig = b""
    if not _ecdsa_pem_verify(rekor_pem, set_sig, canon):
        errors.append("set_signature_invalid")

    note = str(ip.get("checkpoint_note", ""))
    note_payload, _, sig_block = note.partition("\n\n")
    try:
        note_sig = base64.b64decode(sig_block.strip().split(" ")[-1])[4:]
    except (ValueError, IndexError):
        note_sig = b""
    if not sig_block or not _ecdsa_pem_verify(rekor_pem, note_sig, (note_payload + "\n").encode()):
        errors.append("checkpoint_note_signature_invalid")

    # -- our checkpoint ----------------------------------------------
    checkpoint_raw = decoded["quality/checkpoint.json"]
    if _sha(checkpoint_raw).hex() != logged_digest:
        errors.append("checkpoint_digest_mismatch")
    if not _ecdsa_pem_verify(logged_pub, logged_sig, checkpoint_raw):
        errors.append("witness_signature_invalid")
    if logged_pub != decoded["quality/witness_signing.pub"]:
        errors.append("witness_pubkey_diverges_from_log")

    gate_pub_hex = decoded["quality/gate_signing.pub"].decode().strip()
    # Parse the committed quorum registry once: it governs both the head
    # checkpoint's signature set and the pins' gate_signatures.v2 envelope.
    quorum_registered: dict[str, str] | None = None
    quorum_threshold = 0
    quorum_raw = decoded.get("quality/gate_quorum.json")
    if quorum_raw is not None:
        try:
            registry = json.loads(quorum_raw)
            qkeys = registry["keys"]
            quorum_registered = {str(e["key_id"]): str(e["pubkey"]) for e in qkeys}
            quorum_threshold = int(registry["threshold"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            errors.append("registry_malformed")
            quorum_registered, quorum_threshold = {}, 10**9
    try:
        checkpoint = json.loads(checkpoint_raw)
        payload = checkpoint["payload"]
        if quorum_registered is not None:
            # Under a committed registry the head must be the multisig
            # envelope — a v1 checkpoint can't satisfy the quorum.
            if checkpoint.get("schema") != "integrity_checkpoint_sig.v2":
                errors.append("checkpoint_below_quorum")
            else:
                sigs = checkpoint.get("signatures")
                if not isinstance(sigs, list) or not sigs:
                    errors.append("checkpoint_below_quorum")
                else:
                    valid_signers: set[str] = set()
                    for entry in sigs:
                        if not isinstance(entry, dict):
                            errors.append("checkpoint_signature_malformed")
                            continue
                        kid = str(entry.get("key_id", ""))
                        pub = quorum_registered.get(kid)
                        if pub is None:
                            errors.append(f"checkpoint_signature_key_unknown:{kid}")
                            continue
                        if _ed25519_verify(pub, str(entry.get("signature", "")), _canon(payload)):
                            valid_signers.add(kid)
                        else:
                            errors.append(f"checkpoint_signature_invalid:{kid}")
                    if len(valid_signers) < quorum_threshold:
                        errors.append(
                            f"checkpoint_below_quorum:{len(valid_signers)}/{quorum_threshold}"
                        )
        elif not _ed25519_verify(
            gate_pub_hex, str(checkpoint.get("signature", "")), _canon(payload)
        ):
            errors.append("checkpoint_signature_invalid")
        pins = payload.get("pins", {}) if isinstance(payload, dict) else {}
        for rel in (
            "quality/crown_jewels.json",
            "quality/epoch_heads.json",
            "gate_pins.sig",
            "quality/gate_quorum.json",
        ):
            if pins.get(rel) != _sha(decoded[rel]).hex():
                errors.append(f"checkpoint_pin_drift:{rel}")
        # Quorum binding: under a registry the v2 payload must name the
        # registry's canonical digest — a swapped registry can't satisfy
        # quorum AND claim the honest digest.
        if quorum_registered is not None and isinstance(payload, dict):
            claimed = payload.get("quorum")
            claimed_digest = claimed.get("registry_sha256") if isinstance(claimed, dict) else None
            if claimed_digest is None:
                errors.append("quorum_unbound")
            else:
                try:
                    disk_digest = _registry_sha256(json.loads(quorum_raw))
                except json.JSONDecodeError:
                    disk_digest = _sha(quorum_raw).hex()
                if str(claimed_digest) != disk_digest:
                    errors.append("quorum_registry_drift")
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError):
        errors.append("checkpoint_malformed")

    # -- pins ---------------------------------------------------------
    try:
        crown = json.loads(decoded["quality/crown_jewels.json"])
        sigfile = json.loads(decoded["gate_pins.sig"])
    except json.JSONDecodeError:
        return sorted(set(errors + ["pins_malformed"]))

    crown_files = crown.get("files", {})
    if (
        crown_files.get("quality/gate_signing.pub")
        != _sha(decoded["quality/gate_signing.pub"]).hex()
    ):
        errors.append("gate_pubkey_not_pinned")

    sig_payload = sigfile.get("payload", {})
    sig_files = sig_payload.get("files", {}) if isinstance(sig_payload, dict) else {}
    for rel, want in sig_files.items():
        member = decoded.get(rel)
        if member is None:
            errors.append(f"pinned_file_missing:{rel}")
        elif _sha(member).hex() != want:
            errors.append(f"pin_drift:{rel}")
    if quorum_registered is not None:
        # gate_signatures.v2: M-of-N registered signers; each entry verifies
        # under the pubkey the committed registry assigns to its key_id.
        registered, threshold = quorum_registered, quorum_threshold
        if sigfile.get("schema") != "gate_signatures.v2":
            errors.append("quorum_sig_missing")
        sigs = sigfile.get("signatures")
        if not isinstance(sigs, list) or not sigs:
            errors.append("quorum_sig_missing")
            sigs = []
        seen: set[str] = set()
        valid = 0
        for entry in sigs:
            if not isinstance(entry, dict):
                errors.append("signature_malformed")
                continue
            kid = str(entry.get("key_id", ""))
            pub = registered.get(kid)
            if pub is None:
                errors.append(f"unknown_signer:{kid}")
                continue
            if kid in seen:
                errors.append(f"duplicate_signer:{kid}")
                continue
            seen.add(kid)
            if _ed25519_verify(pub, str(entry.get("signature", "")), _canon(sig_payload)):
                valid += 1
            else:
                errors.append(f"signature_invalid:{kid}")
        if valid < threshold:
            errors.append(f"quorum_not_met:{valid}/{threshold}")
    elif not _ed25519_verify(gate_pub_hex, str(sigfile.get("signature", "")), _canon(sig_payload)):
        errors.append("gate_pins_signature_invalid")

    _verify_spine(files, declared, checkpoint_raw, gate_pub_hex, errors)

    return sorted(set(errors))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--rekor-pubkey", type=Path, default=None)
    ap.add_argument(
        "--offline",
        action="store_true",
        help="use the bundle's pinned Rekor key instead of fetching it live",
    )
    args = ap.parse_args()

    rekor_pem: bytes | None = None
    if args.rekor_pubkey is not None:
        rekor_pem = args.rekor_pubkey.read_bytes()
    elif not args.offline:
        try:
            with urllib.request.urlopen(  # noqa: S310
                REKOR_PUBLIC_KEY_URL, timeout=30
            ) as resp:
                rekor_pem = resp.read()
        except OSError as exc:
            print(f"warn: live key fetch failed ({exc}); using bundle pin", file=sys.stderr)

    errors = verify(args.bundle, rekor_pem)
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  {e}")
        return 1
    # Self-binding: the bundle pins the bytes of the script that produced
    # its verdict context. A drifted local copy still verifies correctly
    # (the field is advisory) — but the auditor is told to diff.
    try:
        declared_self = json.loads(args.bundle.read_text()).get("auditor_self_sha256")
        if (
            isinstance(declared_self, str)
            and declared_self != _sha(Path(__file__).read_bytes()).hex()
        ):
            print(
                "warn: auditor_self_mismatch — this script differs from the "
                "pinned copy; diff before trusting this verdict",
                file=sys.stderr,
            )
    except OSError:
        pass
    print("ok — every link verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
