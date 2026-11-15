"""Gate-key rotation receipts: an auditable key lineage over the spine.

The crown-jewels pin protects ``quality/gate_signing.pub`` from *silent*
substitution, and git review is the human check — but nothing proves a new
pubkey was *authorized by the holder of the old key*. A rotation record
closes that gap cryptographically: each link is signed by the outgoing key
(authorization) and countersigned by the incoming key (proof of possession),
chained ``old_pubkey → new_pubkey`` like the checkpoint spine's
``prev_sha256`` links.

Genesis anchor: the first rotation's ``old_pubkey`` must have actually
*signed a checkpoint on the spine* — the signature is re-verified, so the
key's historical authority is proven, not asserted. A rotation whose
outgoing key never signed anything is ``unanchored_genesis``. The live
``gate_signing.pub`` must equal the chain terminus — a rotation that was
recorded but whose key was never installed is ``live_key_not_terminus``.

The verifier exports ``load_keyring``: every key that was ever authorized
(genesis + each rotation), so ``checkpoint_spine`` can verify records signed
by retired keys instead of rejecting them under the live pubkey.

Key resets (``key_reset``) are the one exception to "the old key signs the
new": when the outgoing key is lost, a verified reset record joins the chain
as a link of kind ``reset``, its boundary checkpoint anchors the lost key,
and the verdict says ``reset`` instead of ``intact`` so the custody break is
never silent.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes

ROTATION_SCHEMA = "key_rotation.v1"
ROTATION_GLOB = "rotation_*.json"
ROTATION_DIR = Path("quality")


def _record(path: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"file": path.name, "errors": []}
    try:
        body = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        rec["errors"].append(f"malformed:{exc.__class__.__name__}")
        return rec
    rec["body"] = body
    if not isinstance(body, dict) or body.get("schema") != ROTATION_SCHEMA:
        rec["errors"].append("schema_unexpected")
        return rec
    payload = body.get("payload")
    if not isinstance(payload, dict):
        rec["errors"].append("payload_missing")
        return rec
    rec["payload"] = payload
    return rec


def _key_id(pub_hex: str) -> str:
    from quant_fund.research.gate_signatures import key_id

    try:
        return key_id(pub_hex)
    except (ValueError, TypeError):
        return "malformed"


def _verify_dual_signatures(rec: dict[str, Any], errors: list[str], tag: str) -> bool:
    """Both signatures must verify against the pubkeys inside the payload."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    body = rec.get("body") or {}
    payload = rec.get("payload") or {}
    old_pub, new_pub = str(payload.get("old_pubkey", "")), str(payload.get("new_pubkey", ""))
    ok = True
    for side, pub_hex in (("old", old_pub), ("new", new_pub)):
        sig = str(body.get(f"{side}_signature", ""))
        try:
            pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
            pub.verify(bytes.fromhex(sig), canonical_json_bytes(payload))
        except (InvalidSignature, ValueError):
            errors.append(f"{tag}_signature_invalid:{side}")
            ok = False
    declared_old = str(payload.get("old_key_id", ""))
    declared_new = str(payload.get("new_key_id", ""))
    if declared_old != _key_id(old_pub):
        errors.append(f"{tag}_old_key_id_mismatch")
        ok = False
    if declared_new != _key_id(new_pub):
        errors.append(f"{tag}_new_key_id_mismatch")
        ok = False
    return ok


def _live_pubkey_hex(root: Path) -> str | None:
    from quant_fund.research.gate_signatures import DEFAULT_PUBKEY_PATH

    path = root / DEFAULT_PUBKEY_PATH
    if not path.is_file():
        return None
    try:
        return path.read_text().strip().lower()
    except OSError:
        return None


def load_keyring(root: str | Path) -> dict[str, str]:
    """Every authorized gate key: live + each rotation terminus + genesis."""
    root_path = Path(root)
    ring: dict[str, str] = {}
    live = _live_pubkey_hex(root_path)
    if live:
        ring[_key_id(live)] = live
    for f in sorted(root_path.glob(f"{ROTATION_DIR}/{ROTATION_GLOB}")):
        rec = _record(f)
        payload = rec.get("payload") or {}
        for key in ("old_pubkey", "new_pubkey"):
            pub = str(payload.get(key, ""))
            if pub:
                ring.setdefault(_key_id(pub), pub)
    from quant_fund.research.key_reset import gate_keys, load_resets

    for kid, pub in gate_keys(load_resets(root_path)).items():
        ring.setdefault(kid, pub)
    return ring


def _genesis_anchor(
    root_path: Path,
    genesis_pub: str,
    resets: list[dict[str, Any]],
    errors: list[str],
    notes: list[str],
) -> None:
    """The first outgoing key must have signed a spine member — or, when it
    is a reset's lost key, the boundary checkpoint that reset embeds."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    from quant_fund.research.checkpoint_chain import collect_spine_records
    from quant_fund.research.key_reset import boundary_anchors_gate, signed_by

    anchored = False
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(genesis_pub))
        records = collect_spine_records(root_path)
        anchored = any(
            rec.get("payload") is not None and signed_by(rec.get("body"), genesis_pub)
            for rec in records.values()
        ) or boundary_anchors_gate(resets, genesis_pub)
    except ValueError:
        errors.append("genesis_pubkey_malformed")
    if anchored:
        return
    if root_path.joinpath("quality/checkpoint.json").is_file() or any(
        root_path.glob("quality/checkpoints/*.json")
    ):
        errors.append("unanchored_genesis")
    else:
        notes.append("no_spine_anchor_available")


def verify_rotations(root: str | Path = ".") -> dict[str, Any]:
    """Verify the rotation chain: dual signatures, links, genesis anchor.

    Verified key resets join the chain as ``reset`` links; their authority is
    the reset record (boundary under the lost key, possession under the new),
    not an outgoing-key signature. Invalid reset records are errors.
    """
    from quant_fund.research.key_reset import (
        gate_links,
        lineage_verdict,
        load_resets,
        reset_record_errors,
    )

    root_path = Path(root)
    errors: list[str] = []
    notes: list[str] = []
    recs = [
        rec
        for f in sorted(root_path.glob(f"{ROTATION_DIR}/{ROTATION_GLOB}"))
        if not (rec := _record(f))["errors"]
    ]
    n_files = len(list(root_path.glob(f"{ROTATION_DIR}/{ROTATION_GLOB}")))
    malformed = n_files - len(recs)
    if malformed:
        errors.append(f"rotation_malformed:{malformed}")
    for rec in recs:
        _verify_dual_signatures(rec, errors, "rotation")
    n_rotations = len(recs)
    resets = load_resets(root_path)
    errors.extend(reset_record_errors(resets))
    recs.extend(gate_links(resets))

    if not recs:
        return {
            "ok": not errors,
            "verdict": "no_rotations",
            "n_rotations": 0,
            "n_resets": 0,
            "current_key_id": _key_id(_live_pubkey_hex(root_path) or ""),
            "lineage": [],
            "errors": errors,
            "notes": notes,
        }

    # Chain links: order by old->new pubkey equality, detect forks.
    claims: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:
        claims.setdefault(str(rec["payload"].get("old_pubkey", "")), []).append(rec)
    forks = {k: v for k, v in claims.items() if len(v) > 1}
    for old_pub in forks:
        errors.append(f"rotation_fork:{_key_id(old_pub)}")

    # Walk from each unclaimed-as-new start; the canonical chain is the one
    # reaching the live key.
    new_pubs = {str(r["payload"].get("new_pubkey", "")) for r in recs}
    starts = [r for r in recs if str(r["payload"].get("old_pubkey", "")) not in new_pubs]
    live = _live_pubkey_hex(root_path)
    lineage: list[dict[str, Any]] = []
    genesis_pub: str | None = None
    terminus_pub: str | None = None
    if len(starts) != 1:
        errors.append(f"rotation_chain_ambiguous:{len(starts)}")
    else:
        cur = starts[0]
        seen: set[str] = set()
        while True:
            payload = cur["payload"]
            old_pub = str(payload.get("old_pubkey", ""))
            new_pub = str(payload.get("new_pubkey", ""))
            if genesis_pub is None:
                genesis_pub = old_pub
            terminus_pub = new_pub
            lineage.append(
                {
                    "file": cur["file"],
                    "kind": cur.get("kind", "rotation"),
                    "at": payload.get("at"),
                    "old_key_id": _key_id(old_pub),
                    "new_key_id": _key_id(new_pub),
                    "reason": payload.get("reason"),
                }
            )
            if new_pub in seen:
                errors.append(f"rotation_cycle:{_key_id(new_pub)}")
                break
            seen.add(new_pub)
            nxt = claims.get(new_pub)
            if not nxt:
                break
            cur = nxt[0]
        if live and terminus_pub != live:
            errors.append("live_key_not_terminus")
        elif not live:
            errors.append("live_pubkey_missing")

    # Genesis anchor: the first outgoing key must have signed a spine member.
    if genesis_pub:
        _genesis_anchor(root_path, genesis_pub, resets, errors, notes)

    return {
        "ok": not errors,
        "verdict": lineage_verdict(errors, len(recs) - n_rotations),
        "n_rotations": n_rotations,
        "n_resets": len(recs) - n_rotations,
        "current_key_id": _key_id(terminus_pub or live or ""),
        "genesis_key_id": _key_id(genesis_pub) if genesis_pub else None,
        "lineage": lineage,
        "errors": errors,
        "notes": notes,
    }


def rotate_key(
    root: str | Path,
    old_private_seed_hex: str,
    new_private_seed_hex: str,
    reason: str = "scheduled rotation",
) -> Path:
    """Write a dual-signed rotation record. Both private keys are required:
    the old proves authorization, the new proves possession."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
    )

    root_path = Path(root)
    live = _live_pubkey_hex(root_path)
    old_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(old_private_seed_hex))
    new_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(new_private_seed_hex))
    old_pub = old_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    new_pub = new_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()
    if live and live != old_pub:
        raise ValueError("old key does not match the committed gate_signing.pub")

    payload = {
        "at": datetime.now(UTC).isoformat(),
        "reason": reason,
        "old_key_id": _key_id(old_pub),
        "new_key_id": _key_id(new_pub),
        "old_pubkey": old_pub,
        "new_pubkey": new_pub,
    }
    body = {
        "schema": ROTATION_SCHEMA,
        "payload": payload,
        "old_signature": old_key.sign(canonical_json_bytes(payload)).hex(),
        "new_signature": new_key.sign(canonical_json_bytes(payload)).hex(),
    }
    out = root_path / ROTATION_DIR / f"rotation_{_key_id(new_pub)[:8]}.json"
    atomic_write_text(out, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return out


def rotation_contract_errors(payload: dict[str, Any]) -> list[str]:
    """Contract checks on the verify payload (keeps receipts honest)."""
    errors: list[str] = []
    if payload.get("ok") and payload.get("errors"):
        errors.append("ok_with_errors")
    verdict = payload.get("verdict")
    if verdict in ("intact", "reset") and not payload.get("ok"):
        errors.append(f"{verdict}_not_ok")
    if verdict == "no_rotations" and payload.get("n_rotations"):
        errors.append("no_rotations_with_records")
    n_links = payload.get("n_rotations")
    if isinstance(n_links, int):
        n_links += int(payload.get("n_resets") or 0)
    if verdict in ("intact", "reset") and len(payload.get("lineage") or []) != n_links:
        errors.append("lineage_length_mismatch")
    if verdict == "reset" and not payload.get("n_resets"):
        errors.append("reset_without_records")
    return errors
