"""Quorum-registry rotation: an auditable lineage over the M-of-N signer set.

``gate_quorum.v1`` (``quality/gate_quorum.json``) decides which key_ids may
sign v2 checkpoints and the quorum pin signature. But the registry file is
only *bytes on disk* — nothing in a v2 checkpoint by itself proves that the
registry authorizing it was ever legitimate: an attacker with tree access
could swap in their own key set and sign a fresh head that verifies under
the forged registry.

The rotation ceremony closes that gap the same way ``key_rotation`` closes
single-key substitution: each rotation is a committed record whose
``signatures`` must reach the **predecessor** registry's threshold —
authorization flows from the outgoing quorum. The record embeds both
registry bodies so the verifier can resolve signers without trusting disk
state, and the digests chain ``prev_registry_sha256 → registry_sha256``
like the checkpoint spine's ``prev_sha256``.

Genesis anchor: the first rotation's ``prev_registry`` must have actually
*authorized history* — its digest appears as ``quorum.registry_sha256`` in
a spine checkpoint, or its key_ids overlap signers on the checkpoint spine.
A rotation whose predecessor never governed anything is
``unanchored_genesis``. The live ``gate_quorum.json`` must equal the chain
terminus — ``registry_not_terminus``.

Key resets (``key_reset``) join the lineage as edges of kind ``reset``: the
lost registry cannot sign, so a verified reset record (boundary under the
lost registry, possession under the new) stands in for the predecessor's
signatures, and the verdict reads ``reset`` so the break is never silent.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.gate_signatures import (
    key_id,
    quorum_registry_errors,
    registry_file_bytes,
    registry_sha256,
)
from quant_fund.utils.atomicio import atomic_write_bytes, atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes

QUORUM_ROTATION_SCHEMA = "quorum_rotation.v1"
QUORUM_ROTATION_DIR = Path("quality/quorum_rotations")
QUORUM_ROTATION_GLOB = "rotation_*.json"
DEFAULT_QUORUM_PATH = Path("quality/gate_quorum.json")


def _record(file: Path) -> dict[str, Any]:
    rec: dict[str, Any] = {"file": file.name, "errors": []}
    try:
        body = json.loads(file.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        rec["errors"].append(f"malformed:{exc.__class__.__name__}")
        return rec
    rec["body"] = body
    if not isinstance(body, dict) or body.get("schema") != QUORUM_ROTATION_SCHEMA:
        rec["errors"].append("schema_unexpected")
        return rec
    payload = body.get("payload")
    if not isinstance(payload, dict):
        rec["errors"].append("payload_missing")
        return rec
    rec["payload"] = payload
    prev_reg = payload.get("prev_registry")
    new_reg = payload.get("registry")
    if not isinstance(prev_reg, dict) or not isinstance(new_reg, dict):
        rec["errors"].append("registry_embed_missing")
        return rec
    if quorum_registry_errors(prev_reg):
        rec["errors"].append("prev_registry_malformed")
    if quorum_registry_errors(new_reg):
        rec["errors"].append("registry_malformed")
    if payload.get("prev_registry_sha256") != registry_sha256(prev_reg):
        rec["errors"].append("prev_registry_digest_mismatch")
    if payload.get("registry_sha256") != registry_sha256(new_reg):
        rec["errors"].append("registry_digest_mismatch")
    rec["prev_digest"] = payload.get("prev_registry_sha256")
    rec["digest"] = payload.get("registry_sha256")
    return rec


def _registered_map(registry: dict[str, Any]) -> tuple[dict[str, str], int]:
    return (
        {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]},
        int(registry["threshold"]),
    )


def _count_valid(
    signatures: Any,
    registered: dict[str, str],
    threshold: int,
    payload: dict[str, Any],
    errors: list[str],
    tag: str,
) -> int:
    """Count distinct valid signers under a registry; append per-sig errors."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

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
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
                bytes.fromhex(str(entry.get("signature", ""))),
                canonical_json_bytes(payload),
            )
            valid.add(kid)
        except (InvalidSignature, ValueError):
            errors.append(f"{tag}_signature_invalid:{kid}")
    if len(valid) < threshold:
        errors.append(f"{tag}_below_quorum:{len(valid)}/{threshold}")
    return len(valid)


def rotation_authorizes(rot_recs: list[dict[str, Any]], prev_digest: str, digest: str) -> bool:
    """Some verified-shape rotation record chains ``prev_digest → digest``
    with signatures reaching the predecessor registry's threshold."""
    for rot in rot_recs:
        payload = rot["payload"]
        if (
            str(payload.get("prev_registry_sha256")) != prev_digest
            or str(payload.get("registry_sha256")) != digest
        ):
            continue
        registered, threshold = _registered_map(payload["prev_registry"])
        sigs = rot["body"].get("signatures")
        if _count_valid(sigs, registered, threshold, payload, [], "rotation") >= threshold:
            return True
    return False


def _rotation_records(root_path: Path) -> list[dict[str, Any]]:
    """Every committed rotation record, parsed (``errors`` empty = usable)."""
    rot_dir = root_path / QUORUM_ROTATION_DIR
    if not rot_dir.is_dir():
        return []
    return [_record(f) for f in sorted(rot_dir.glob(QUORUM_ROTATION_GLOB))]


def lineage_edges(root: str | Path) -> list[dict[str, Any]]:
    """Usable rotation records + verified key resets, as lineage edges."""
    from quant_fund.research.key_reset import load_resets, registry_edges

    root_path = Path(root)
    edges = [rec for rec in _rotation_records(root_path) if not rec["errors"]]
    return edges + registry_edges(load_resets(root_path))


def prefield_registry(root: str | Path) -> tuple[dict[str, str], int] | None:
    """The registry pre-field v2 checkpoints (no ``quorum`` claim) verify under.

    That is the genesis of the committed lineage (rotations + key resets) —
    pre-field records predate every registry change. With no lineage, or an
    ambiguous one (reported by ``verify_quorum_rotations``), it is the live
    registry: on a tree that never rotated, the same set.
    """
    from quant_fund.research.gate_signatures import load_quorum_registry

    edges = lineage_edges(root)
    targets = {str(e["digest"]) for e in edges}
    starts = [e for e in edges if str(e["prev_digest"]) not in targets]
    if len(starts) != 1:
        return load_quorum_registry(root)
    return _registered_map(starts[0]["payload"]["prev_registry"])


def rotate_quorum(
    root: str | Path,
    new_registry: dict[str, Any],
    signers: list[tuple[str, str]],
    *,
    reason: str = "quorum rotation",
) -> Path:
    """Commit a quorum rotation record + install the new registry.

    ``signers`` are ``(private_seed_hex, pubkey_hex)`` pairs of *outgoing*
    quorum members — authorization comes from the predecessor registry, so
    fewer than its threshold of distinct members raises without writing.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root_path = Path(root)
    prev_file = root_path / DEFAULT_QUORUM_PATH
    if not prev_file.is_file():
        raise ValueError("no committed quorum registry — genesis uses quorum-init")
    try:
        prev_registry = json.loads(prev_file.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"committed registry unreadable: {exc}") from exc
    if errs := quorum_registry_errors(prev_registry):
        raise ValueError(f"committed registry malformed: {errs}")
    if errs := quorum_registry_errors(new_registry):
        raise ValueError(f"new registry malformed: {errs}")

    registered, threshold = _registered_map(prev_registry)
    seen: set[str] = set()
    signatures = []
    for priv_hex, pub_hex in signers:
        kid = key_id(pub_hex)
        if kid not in registered or registered[kid] != pub_hex:
            raise ValueError(f"signer not in current quorum registry: {kid}")
        if kid in seen:
            continue
        seen.add(kid)
        signatures.append((kid, priv_hex))
    if len(signatures) < threshold:
        raise ValueError(
            f"quorum rotation unattainable: {len(signatures)} signers < threshold {threshold}"
        )

    payload = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "prev_registry": prev_registry,
        "prev_registry_sha256": registry_sha256(prev_registry),
        "registry": new_registry,
        "registry_sha256": registry_sha256(new_registry),
    }
    msg = canonical_json_bytes(payload)
    body = {
        "schema": QUORUM_ROTATION_SCHEMA,
        "algorithm": "ed25519",
        "payload": payload,
        "signatures": [
            {
                "key_id": kid,
                "signature": Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
                .sign(msg)
                .hex(),
            }
            for kid, priv_hex in signatures
        ],
    }
    out_dir = root_path / QUORUM_ROTATION_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"rotation_{payload['registry_sha256'][:16]}.json"
    atomic_write_text(out, json.dumps(body, indent=2, sort_keys=True) + "\n")
    # Install the new registry byte-exactly as the digests bind it.
    atomic_write_bytes(prev_file, registry_file_bytes(new_registry))
    return out


def load_registry_lineage(root: str | Path) -> dict[str, tuple[dict[str, str], int]]:
    """``{registry_sha256: (registered_map, threshold)}`` for every registry
    the committed rotation chain, verified key resets and the live registry
    admit."""
    root_path = Path(root)
    lineage: dict[str, tuple[dict[str, str], int]] = {}
    live_file = root_path / DEFAULT_QUORUM_PATH
    if live_file.is_file():
        try:
            live = json.loads(live_file.read_bytes())
            if not quorum_registry_errors(live):
                lineage[registry_sha256(live)] = _registered_map(live)
        except (OSError, json.JSONDecodeError):
            pass
    for edge in lineage_edges(root_path):
        payload = edge["payload"]
        lineage[str(payload["prev_registry_sha256"])] = _registered_map(payload["prev_registry"])
        lineage[str(payload["registry_sha256"])] = _registered_map(payload["registry"])
    return lineage


def _genesis_anchored(
    root_path: Path, genesis_digest: str, genesis_reg: Any, notes: list[str]
) -> bool:
    """The chain's first predecessor governed real history: its digest is a
    spine checkpoint's ``quorum.registry_sha256`` (post-field era) or its
    key_ids overlap a spine signer (pre-field era continuity)."""
    try:
        from quant_fund.research.checkpoint_chain import collect_spine_records

        records = collect_spine_records(root_path)
        genesis_kids = (
            {str(e["key_id"]) for e in genesis_reg["keys"]}
            if isinstance(genesis_reg, dict) and isinstance(genesis_reg.get("keys"), list)
            else set()
        )
        spine_kids: set[str] = set()
        for rec in records.values():
            payload = rec.get("payload")
            if not isinstance(payload, dict):
                continue
            q = payload.get("quorum")
            if isinstance(q, dict) and q.get("registry_sha256") == genesis_digest:
                return True
            body = rec.get("body") or {}
            if body.get("key_id"):
                spine_kids.add(str(body["key_id"]))
            sigs = body.get("signatures")
            if isinstance(sigs, list):
                spine_kids.update(
                    str(s["key_id"]) for s in sigs if isinstance(s, dict) and s.get("key_id")
                )
        return bool(genesis_kids & spine_kids)
    except Exception:  # noqa: BLE001 — anchor is best-effort evidence
        notes.append("genesis_anchor_unverifiable")
        return False


def verify_quorum_rotations(root: str | Path = ".") -> dict[str, Any]:
    """Verify the rotation chain: per-record quorum, links, genesis anchor,
    live-terminus equality.

    Verified key resets join the chain as ``reset`` edges (their authority is
    the reset record, not predecessor signatures); invalid reset records are
    errors.
    """
    from quant_fund.research.key_reset import (
        boundary_anchors_registry,
        lineage_verdict,
        load_resets,
        registry_edges,
        reset_record_errors,
    )

    root_path = Path(root)
    errors: list[str] = []
    notes: list[str] = []
    recs = []
    for rec in _rotation_records(root_path):
        errors.extend(f"rotation_{e}:{rec['file']}" for e in rec["errors"])
        if not rec["errors"]:
            recs.append(rec)

    # Per-record authorization: signatures under the embedded predecessor.
    for rec in recs:
        payload = rec["payload"]
        registered, threshold = _registered_map(payload["prev_registry"])
        _count_valid(
            rec["body"].get("signatures"), registered, threshold, payload, errors, "rotation"
        )
    n_rotations = len(recs)
    resets = load_resets(root_path)
    errors.extend(reset_record_errors(resets))
    recs.extend(registry_edges(resets))

    live_file = root_path / DEFAULT_QUORUM_PATH
    live_digest: str | None = None
    if live_file.is_file():
        try:
            live_reg = json.loads(live_file.read_bytes())
            if quorum_registry_errors(live_reg):
                errors.append("registry_malformed")
            else:
                live_digest = registry_sha256(live_reg)
        except (OSError, json.JSONDecodeError):
            errors.append("registry_malformed")

    if not recs:
        return {
            "schema": "quorum_rotations.v1",
            "ok": not errors,
            "verdict": "no_rotations" if not errors else "broken",
            "n_rotations": 0,
            "n_resets": 0,
            "errors": sorted(errors),
            "notes": notes,
        }

    # Links: order by prev -> new digest; forks and orphans surface.
    claims: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:
        claims.setdefault(str(rec["prev_digest"]), []).append(rec)
    for prev, group in claims.items():
        if len(group) > 1:
            errors.append(f"rotation_fork:{prev[:16]}")
    new_digests = {str(r["digest"]) for r in recs}
    starts = [r for r in recs if str(r["prev_digest"]) not in new_digests]
    chain: list[dict[str, Any]] = []
    genesis_digest: str | None = None
    terminus_digest: str | None = None
    if len(starts) != 1:
        errors.append(f"rotation_chain_ambiguous:{len(starts)}")
    else:
        cur = starts[0]
        seen: set[str] = set()
        while True:
            genesis_digest = genesis_digest or str(cur["prev_digest"])
            terminus_digest = str(cur["digest"])
            chain.append(cur)
            if str(cur["digest"]) in seen:
                errors.append(f"rotation_cycle:{str(cur['digest'])[:16]}")
                break
            seen.add(str(cur["digest"]))
            nxt = claims.get(str(cur["digest"]))
            if not nxt:
                break
            cur = nxt[0]
    on_chain = {id(r) for r in chain}
    for rec in recs:
        if id(rec) not in on_chain:
            errors.append(f"rotation_orphan:{rec['file']}")

    if live_digest is not None and terminus_digest is not None and terminus_digest != live_digest:
        errors.append("registry_not_terminus")

    # Genesis anchor: the chain's first predecessor must have governed real
    # history on the spine — or be a verified reset's lost registry, whose
    # embedded boundary checkpoint reached that registry's threshold.
    if genesis_digest and not (
        _genesis_anchored(
            root_path, genesis_digest, starts[0]["payload"].get("prev_registry"), notes
        )
        or boundary_anchors_registry(resets, genesis_digest)
    ):
        errors.append("unanchored_genesis")

    return {
        "schema": "quorum_rotations.v1",
        "ok": not errors,
        "verdict": lineage_verdict(errors, len(recs) - n_rotations),
        "n_rotations": n_rotations,
        "n_resets": len(recs) - n_rotations,
        "genesis_registry_sha256": genesis_digest,
        "terminus_registry_sha256": terminus_digest,
        "lineage": [
            {
                "file": r["file"],
                "kind": r.get("kind", "rotation"),
                "at": (r.get("payload") or {}).get("at"),
                "prev": str(r["prev_digest"])[:16],
                "registry": str(r["digest"])[:16],
            }
            for r in chain
        ],
        "errors": sorted(errors),
        "notes": sorted(notes),
    }


def quorum_rotation_contract_errors(payload: Any) -> list[str]:
    """Contract checks on a ``quorum_rotations.v1`` verdict payload."""
    if not isinstance(payload, dict) or payload.get("schema") != "quorum_rotations.v1":
        return ["schema_mismatch"]
    errors: list[str] = []
    if not isinstance(payload.get("ok"), bool):
        errors.append("ok_not_bool")
    n = payload.get("n_rotations")
    if not isinstance(n, int) or n < 0:
        errors.append("n_rotations_malformed")
    if payload.get("ok") and payload.get("errors"):
        errors.append("ok_with_errors")
    if payload.get("verdict") == "reset" and not payload.get("n_resets"):
        errors.append("reset_without_records")
    return errors
