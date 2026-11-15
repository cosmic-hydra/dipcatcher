"""Key reset: an explicit, audited break in signing-key custody.

``key_rotation`` and ``quorum_rotation`` both assume the outgoing key still
exists — the old key authorizes the new. When every outgoing key is *lost*
(nobody can sign anything with it) that ceremony is impossible, and the only
other moves are a silent key swap, which every verifier rightly rejects, or
rewriting history. A reset record is the honest third option: a committed,
epoch-chained, checkpoint-pinned file that says "the keys that signed history
up to checkpoint B are lost; these new keys sign everything after B", which
``verify-rotations``, ``checkpoint-chain`` and ``verify-repo`` all report.

What a reset proves, and what it cannot:

- **Boundary.** The record embeds the exact bytes of the last checkpoint
  signed under the lost keys (``boundary.checkpoint_sha256`` is their
  digest). The boundary must verify under the *lost* key set, so a reset
  cannot invent the history it claims to succeed. Spine verifiers also
  require the boundary to sit on the checkpoint spine and the first new-era
  checkpoint to link directly to it — a reset never re-authorizes a side
  chain or a mid-history splice.
- **Possession.** The new registry's threshold, including the new gate key,
  signs the record, so the named new keys are held by someone.
- **Revocation.** No spine checkpoint after the boundary may be signed by,
  or claim, a lost key or the lost registry — a reset also retires keys that
  may have leaked.
- **Freshness.** New keys share nothing with the lost set and never signed
  anything at or before the boundary.
- **Authorization is not cryptographic.** The lost keys cannot sign, by
  definition, so the reset is authorized by whoever can commit to the
  repository; ``authorized_by`` is a claim, not a proof. The protection is
  visibility: checkpoint-chain, verify-rotations, verify-repo, the auditor
  bundle and the standalone bundle script all report the reset, and the
  record is reviewed like any other change to the trust root.

An invalid reset record never contributes a key to any keyring or lineage:
it fails closed, so the spine signed by the keys it names stays broken.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import base64
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.gate_signatures import (
    DEFAULT_PUBKEY_PATH,
    DEFAULT_QUORUM_PATH,
    QUORUM_REGISTRY_SCHEMA,
    key_id,
    quorum_registry_errors,
    registry_file_bytes,
    registry_sha256,
)
from quant_fund.utils.atomicio import atomic_write_bytes, atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

RESET_SCHEMA = "key_reset.v1"
RESET_DIR = Path("quality")
RESET_GLOB = "key_reset_*.json"
RESET_PREFIX = "quality/key_reset_"
CHECKPOINT_PATH = Path("quality/checkpoint.json")
WITNESS_PUBKEY_PATH = Path("quality/witness_signing.pub")
_CHECKPOINT_SCHEMAS = ("integrity_checkpoint_sig.v1", "integrity_checkpoint_sig.v2")
_QUORUM_CHECKPOINT_SCHEMA = "integrity_checkpoint_sig.v2"
_TEXT_FIELDS = ("at", "reason", "authorized_by")


# --------------------------------------------------------------- primitives


def _is_hex(value: Any, n_bytes: int) -> bool:
    """Lowercase hex of exactly ``n_bytes`` bytes."""
    if not isinstance(value, str) or len(value) != 2 * n_bytes or value != value.lower():
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _ed25519_ok(pub_hex: str, sig_hex: Any, message: bytes) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex)).verify(
            bytes.fromhex(str(sig_hex)), message
        )
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _sign(private_seed_hex: str, message: bytes) -> str:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    return key.sign(message).hex()


def _registered(registry: dict[str, Any]) -> tuple[dict[str, str], int]:
    return (
        {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]},
        int(registry["threshold"]),
    )


def _valid_signers(signatures: Any, registered: dict[str, str], payload: Any) -> set[str]:
    """Distinct registered key_ids whose signature over ``payload`` verifies."""
    if not isinstance(signatures, list):
        return set()
    message = canonical_json_bytes(payload)
    valid: set[str] = set()
    for entry in signatures:
        kid = str(entry.get("key_id", "")) if isinstance(entry, dict) else ""
        pub = registered.get(kid)
        if pub is not None and _ed25519_ok(pub, entry.get("signature"), message):
            valid.add(kid)
    return valid


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


def _witness_spki(pem: str) -> bytes:
    """DER SubjectPublicKeyInfo of a PEM public key: equal keys compare
    equal however the PEM text is wrapped."""
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_public_key,
    )

    return load_pem_public_key(pem.encode()).public_bytes(
        Encoding.DER, PublicFormat.SubjectPublicKeyInfo
    )


def signed_by(body: Any, pub_hex: str) -> bool:
    """Any signature on a checkpoint record body verifies under ``pub_hex``."""
    if not isinstance(body, dict):
        return False
    message = canonical_json_bytes(body.get("payload"))
    sigs: list[Any] = [body.get("signature")]
    extra = body.get("signatures")
    if isinstance(extra, list):
        sigs += [e.get("signature") for e in extra if isinstance(e, dict)]
    return any(_ed25519_ok(pub_hex, s, message) for s in sigs if s is not None)


def side_key_ids(side: dict[str, Any]) -> set[str]:
    """Every key_id one side of a reset names: gate key + registry members."""
    return {str(side["gate_key_id"])} | {str(e["key_id"]) for e in side["registry"]["keys"]}


def side_pubkeys(side: dict[str, Any]) -> set[str]:
    return {str(side["gate_pubkey"])} | {str(e["pubkey"]) for e in side["registry"]["keys"]}


# ------------------------------------------------------------------ parsing


def _side_errors(side: Any, tag: str) -> list[str]:
    """One key set: gate key + id, registry + digest, optional witness key."""
    if not isinstance(side, dict):
        return [f"{tag}_missing"]
    errors: list[str] = []
    pub = side.get("gate_pubkey")
    if not _is_hex(pub, 32):
        errors.append(f"{tag}_gate_pubkey_malformed")
    elif side.get("gate_key_id") != key_id(str(pub)):
        errors.append(f"{tag}_gate_key_id_mismatch")
    registry: Any = side.get("registry")
    if quorum_registry_errors(registry):
        errors.append(f"{tag}_registry_malformed")
    elif side.get("registry_sha256") != registry_sha256(registry):
        errors.append(f"{tag}_registry_digest_mismatch")
    if not _witness_pem_ok(side.get("witness_pubkey_pem")):
        errors.append(f"{tag}_witness_pubkey_malformed")
    return errors


def _reject_constant(token: str) -> Any:
    raise ValueError(f"non-finite number: {token}")


def _finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):  # 1e999 overflows to inf without a constant token
        raise ValueError(f"non-finite number: {token}")
    return value


def _strict_loads(raw: bytes) -> Any:
    """``json.loads`` refusing what canonical signing bytes cannot carry:
    NaN/Infinity and numbers that overflow to infinity (signed as null, so
    the record would verify while saying something else) and unpaired
    surrogates (not encodable as UTF-8). The standalone verifier refuses the
    same; ``UnicodeEncodeError`` is a ``ValueError`` like the others."""
    value = json.loads(raw, parse_constant=_reject_constant, parse_float=_finite_float)
    json.dumps(value, ensure_ascii=False).encode("utf-8")
    return value


def _boundary_body(boundary: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Decode the embedded boundary checkpoint; its digest must match."""
    if not isinstance(boundary, dict) or not _is_hex(boundary.get("checkpoint_sha256"), 32):
        return None, "boundary_malformed"
    try:
        raw = base64.b64decode(str(boundary.get("checkpoint_b64", "")), validate=True)
        body = _strict_loads(raw)
    except ValueError:  # binascii.Error, JSONDecodeError, Unicode{De,En}codeError
        return None, "boundary_malformed"
    if hash_bytes(raw) != boundary["checkpoint_sha256"]:
        return None, "boundary_digest_mismatch"
    if not isinstance(body, dict) or not isinstance(body.get("payload"), dict):
        return None, "boundary_malformed"
    if body.get("schema") not in _CHECKPOINT_SCHEMAS:
        return None, "boundary_not_checkpoint"
    return body, None


def _claimed_registry(payload: dict[str, Any]) -> str | None:
    q = payload.get("quorum")
    if isinstance(q, dict) and isinstance(q.get("registry_sha256"), str):
        return str(q["registry_sha256"])
    return None


def _boundary_era_ok(body: dict[str, Any], lost: dict[str, Any]) -> bool:
    """The boundary verifies under the LOST key set: a v2 record reaching the
    lost registry's threshold (naming that registry, or predating the field),
    or a v1 record signed by the lost gate key."""
    payload = body["payload"]
    if body.get("schema") == _QUORUM_CHECKPOINT_SCHEMA:
        claimed = _claimed_registry(payload)
        if claimed is not None and claimed != lost["registry_sha256"]:
            return False
        registered, threshold = _registered(lost["registry"])
        return len(_valid_signers(body.get("signatures"), registered, payload)) >= threshold
    pub = str(lost["gate_pubkey"])
    return body.get("key_id") == key_id(pub) and _ed25519_ok(
        pub, body.get("signature"), canonical_json_bytes(payload)
    )


def _relation_errors(payload: dict[str, Any]) -> list[str]:
    """The new gate key is registered; the new set shares nothing with the
    lost. A witness key declared lost needs a different replacement — the
    reset installs the new side, so a lost key must not stay live."""
    lost, new = payload["lost"], payload["new"]
    errors: list[str] = []
    registered, _ = _registered(new["registry"])
    if registered.get(str(new["gate_key_id"])) != new["gate_pubkey"]:
        errors.append("new_gate_key_unregistered")
    shared = (side_key_ids(lost) & side_key_ids(new)) | {
        key_id(pub) for pub in side_pubkeys(lost) & side_pubkeys(new)
    }
    errors.extend(f"reuses_lost_key:{kid}" for kid in sorted(shared))
    lost_witness, new_witness = lost.get("witness_pubkey_pem"), new.get("witness_pubkey_pem")
    if lost_witness is not None:
        if new_witness is None:
            errors.append("lost_witness_not_replaced")
        elif _witness_spki(lost_witness) == _witness_spki(new_witness):
            errors.append("reuses_lost_witness_key")
    return errors


def _payload_errors(payload: dict[str, Any]) -> list[str]:
    errors = [
        f"{field}_missing"
        for field in _TEXT_FIELDS
        if not isinstance(payload.get(field), str) or not str(payload[field]).strip()
    ]
    errors += _side_errors(payload.get("lost"), "lost")
    errors += _side_errors(payload.get("new"), "new")
    _, boundary_error = _boundary_body(payload.get("boundary"))
    if boundary_error is not None:
        errors.append(boundary_error)
    return errors or _relation_errors(payload)


def _authority_errors(body: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    """Boundary under the lost set; possession under the new registry."""
    errors: list[str] = []
    boundary, _ = _boundary_body(payload["boundary"])
    if boundary is None or not _boundary_era_ok(boundary, payload["lost"]):
        errors.append("boundary_not_lost_era")
    registered, threshold = _registered(payload["new"]["registry"])
    valid = _valid_signers(body.get("signatures"), registered, payload)
    if len(valid) < threshold:
        errors.append(f"possession_below_quorum:{len(valid)}/{threshold}")
    if str(payload["new"]["gate_key_id"]) not in valid:
        errors.append("possession_gate_key_unproven")
    return errors


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    lost, new = payload["lost"], payload["new"]
    return {
        "at": payload["at"],
        "reason": payload["reason"],
        "authorized_by": payload["authorized_by"],
        "boundary": payload["boundary"]["checkpoint_sha256"],
        "boundary_body": _boundary_body(payload["boundary"])[0],
        "lost": lost,
        "new": new,
        "lost_digest": lost["registry_sha256"],
        "new_digest": new["registry_sha256"],
        "lost_gate_pub": lost["gate_pubkey"],
        "new_gate_pub": new["gate_pubkey"],
        "lost_kids": side_key_ids(lost),
        "new_kids": side_key_ids(new),
    }


def parse_reset(raw: bytes, name: str) -> dict[str, Any]:
    """Parse + fully verify one reset record; ``errors == []`` means usable."""
    rec: dict[str, Any] = {"file": name, "errors": []}
    try:
        body = _strict_loads(raw)
    except ValueError:
        rec["errors"] = ["malformed"]
        return rec
    payload = body.get("payload") if isinstance(body, dict) else None
    if (
        not isinstance(payload, dict)
        or body.get("schema") != RESET_SCHEMA
        or body.get("algorithm") != "ed25519"
    ):
        rec["errors"] = ["schema_unexpected"]
        return rec
    rec["errors"] = _payload_errors(payload) or _authority_errors(body, payload)
    if not rec["errors"]:
        rec.update(_summary(payload))
    return rec


def reset_paths(root: str | Path) -> list[Path]:
    return sorted((Path(root) / RESET_DIR).glob(RESET_GLOB))


def load_resets(root: str | Path) -> list[dict[str, Any]]:
    """Every committed reset record, parsed and verified."""
    out: list[dict[str, Any]] = []
    for path in reset_paths(root):
        try:
            raw = path.read_bytes()
        except OSError:
            out.append({"file": path.name, "errors": ["unreadable"]})
            continue
        out.append(parse_reset(raw, path.name))
    return out


def valid_resets(resets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in resets if not r["errors"]]


def reset_record_errors(resets: list[dict[str, Any]]) -> list[str]:
    return [f"key_reset_{e}:{r['file']}" for r in resets for e in r["errors"]]


def lineage_verdict(errors: list[str], n_resets: int) -> str:
    """``broken`` on any error; ``reset`` when a custody break is on record."""
    if errors:
        return "broken"
    return "reset" if n_resets else "intact"


# ------------------------------------------------- lineage adapters (keyrings)


def gate_links(resets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valid resets as ``key_rotation``-shaped links (lost → new gate key)."""
    return [
        {
            "file": r["file"],
            "kind": "reset",
            "payload": {
                "at": r["at"],
                "reason": r["reason"],
                "old_pubkey": r["lost_gate_pub"],
                "new_pubkey": r["new_gate_pub"],
            },
        }
        for r in valid_resets(resets)
    ]


def gate_keys(resets: list[dict[str, Any]]) -> dict[str, str]:
    """``{key_id: pubkey}`` for the lost + new gate key of every valid reset."""
    ring: dict[str, str] = {}
    for r in valid_resets(resets):
        ring.setdefault(key_id(r["lost_gate_pub"]), r["lost_gate_pub"])
        ring.setdefault(key_id(r["new_gate_pub"]), r["new_gate_pub"])
    return ring


def boundary_anchors_gate(resets: list[dict[str, Any]], pub_hex: str) -> bool:
    """``pub_hex`` is a valid reset's lost gate key and signed its boundary."""
    return any(
        r["lost_gate_pub"] == pub_hex and signed_by(r["boundary_body"], pub_hex)
        for r in valid_resets(resets)
    )


def registry_edges(resets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Valid resets as ``quorum_rotation``-shaped edges (lost → new registry)."""
    return [
        {
            "file": r["file"],
            "kind": "reset",
            "prev_digest": r["lost_digest"],
            "digest": r["new_digest"],
            "payload": {
                "at": r["at"],
                "prev_registry": r["lost"]["registry"],
                "prev_registry_sha256": r["lost_digest"],
                "registry": r["new"]["registry"],
                "registry_sha256": r["new_digest"],
            },
        }
        for r in valid_resets(resets)
    ]


def boundary_anchors_registry(resets: list[dict[str, Any]], digest: str) -> bool:
    """``digest`` is a valid reset's lost registry — its boundary checkpoint
    reached that registry's threshold when the record was parsed."""
    return any(r["lost_digest"] == digest for r in valid_resets(resets))


# ----------------------------------------------------------- spine checks


def reset_edge_verdict(
    resets: list[dict[str, Any]],
    parent: str,
    parent_era: str | None,
    child_era: str,
) -> str | None:
    """Classify a spine edge whose registry era changes.

    ``None``: no valid reset leads into ``child_era`` from ``parent_era`` —
    the edge needs an ordinary rotation. ``"ok"``: a reset does, and its
    boundary is exactly ``parent``. ``"off_boundary"``: a reset names the
    eras but the spine crosses them anywhere else.
    """
    linked = [
        r for r in resets if r["new_digest"] == child_era and parent_era in (r["lost_digest"], None)
    ]
    if not linked:
        return None
    return "ok" if any(r["boundary"] == parent for r in linked) else "off_boundary"


def _resurrected(reset: dict[str, Any], before: list[dict[str, Any]]) -> list[str]:
    """New keys must be fresh: none signed anything up to the boundary."""
    from quant_fund.research.checkpoint_chain import _signer_key_ids

    used: set[str] = set()
    for rec in before:
        used |= _signer_key_ids(rec)
    return [f"key_reset_resurrects:{kid}" for kid in sorted(reset["new_kids"] & used)]


def _revoked(
    reset: dict[str, Any],
    after: list[str],
    records: dict[str, Any],
    lineage: dict[str, tuple[dict[str, str], int]],
) -> list[str]:
    """Past the boundary the lost keys are revoked: none signs, and no record
    claims the lost registry or any later registry that holds a lost key
    again (a later reset or rotation cannot reinstate one)."""
    from quant_fund.research.checkpoint_chain import _quorum_digest, _signer_key_ids

    errors: list[str] = []
    for digest in after:
        rec = records[digest]
        if _signer_key_ids(rec) & reset["lost_kids"]:
            errors.append(f"key_reset_revoked_signer:{digest[:12]}")
        claimed = _quorum_digest(rec)
        if claimed is not None and (
            claimed == reset["lost_digest"]
            or (claimed in lineage and set(lineage[claimed][0]) & reset["lost_kids"])
        ):
            errors.append(f"key_reset_revoked_registry:{digest[:12]}")
    return errors


def _spki_or_none(pem: Any) -> bytes | None:
    """DER SubjectPublicKeyInfo of a PEM key given as text or raw bytes, or
    ``None`` when it does not parse. Raw bytes matter for the installed key:
    the witness verifier loads the file's bytes, and a PEM with stray
    non-UTF-8 bytes around it still loads there."""
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        PublicFormat,
        load_pem_public_key,
    )

    try:
        data = pem.encode() if isinstance(pem, str) else pem
        if not isinstance(data, bytes):
            return None
        return load_pem_public_key(data).public_bytes(
            Encoding.DER, PublicFormat.SubjectPublicKeyInfo
        )
    except (ValueError, TypeError, UnsupportedAlgorithm):
        return None  # a malformed installed key is the witness checks' finding


def _keys_reinstated(ordered_resets: list[dict[str, Any]]) -> list[str]:
    """A key declared lost stays lost: no later reset names it in its new
    set, whether or not anything has been signed under that set yet."""
    retired: set[str] = set()
    errors: list[str] = []
    for r in ordered_resets:
        if r["new_kids"] & retired:
            errors.append(f"key_reset_reinstates_key:{r['file']}")
        retired |= r["lost_kids"]
    return errors


def _witness_reinstated(
    ordered_resets: list[dict[str, Any]], live_pem: str | bytes | None
) -> list[str]:
    """A witness key declared lost stays lost: no later reset installs it
    again, and it is not the installed key."""
    retired: set[bytes] = set()
    errors: list[str] = []
    for r in ordered_resets:
        new_spki = _spki_or_none(r["new"].get("witness_pubkey_pem"))
        if new_spki is not None and new_spki in retired:
            errors.append(f"key_reset_reinstates_witness:{r['file']}")
        lost_spki = _spki_or_none(r["lost"].get("witness_pubkey_pem"))
        if lost_spki is not None:
            retired.add(lost_spki)
    live = _spki_or_none(live_pem)
    if live is not None and live in retired:
        errors.append("key_reset_reinstates_witness:live")
    return errors


def reset_spine_errors(
    resets: list[dict[str, Any]],
    records: dict[str, Any],
    ordered: list[str],
    lineage: dict[str, tuple[dict[str, str], int]] | None = None,
    live_witness_pem: str | bytes | None = None,
) -> list[str]:
    """Boundary on the spine, fresh new keys, revoked lost keys, and no lost
    gate, registry or witness key reinstated.

    ``ordered`` is the walked spine, genesis first; ``records`` maps digest
    to ``checkpoint_chain`` record dicts; ``lineage`` maps registry digests
    to ``(key_id -> pubkey, threshold)``; ``live_witness_pem`` is the
    installed ``witness_signing.pub`` (its raw bytes).
    """
    position = {digest: i for i, digest in enumerate(ordered)}
    errors: list[str] = []
    placed: list[tuple[int, str, dict[str, Any]]] = []
    for r in resets:
        idx = position.get(r["boundary"])
        if idx is None:
            errors.append(f"key_reset_boundary_off_spine:{r['file']}")
            continue
        placed.append((idx, r["file"], r))
        errors += _resurrected(r, [records[d] for d in ordered[: idx + 1]])
        errors += _revoked(r, ordered[idx + 1 :], records, lineage or {})
    placed.sort(key=lambda item: (item[0], item[1]))
    in_order = [r for _idx, _file, r in placed]
    errors += _keys_reinstated(in_order)
    errors += _witness_reinstated(in_order, live_witness_pem)
    return errors


def reset_notes(resets: list[dict[str, Any]]) -> list[str]:
    return [f"key_reset:{r['file']}@{r['boundary'][:12]}" for r in resets]


# ------------------------------------------------------------------ writer


def _lost_side(root: Path, *, witness_replaced: bool) -> dict[str, Any]:
    """The key set as committed now — the one being declared lost. The
    witness key counts as lost only when the reset replaces it; otherwise it
    stays installed and the record names no witness on either side."""
    pub = (root / DEFAULT_PUBKEY_PATH).read_text().strip().lower()
    quorum_file = root / DEFAULT_QUORUM_PATH
    registry = (
        json.loads(quorum_file.read_bytes())
        if quorum_file.is_file()
        else {
            "schema": QUORUM_REGISTRY_SCHEMA,
            "threshold": 1,
            "keys": [{"key_id": key_id(pub), "pubkey": pub}],
        }
    )
    witness = root / WITNESS_PUBKEY_PATH
    return {
        "gate_pubkey": pub,
        "gate_key_id": key_id(pub),
        "registry": registry,
        "registry_sha256": registry_sha256(registry),
        "witness_pubkey_pem": (
            witness.read_text() if witness_replaced and witness.is_file() else None
        ),
    }


def _new_side(
    signers: list[tuple[str, str]], threshold: int, witness_pubkey_pem: str | None
) -> dict[str, Any]:
    gate_pub = signers[0][1].strip().lower()
    registry = {
        "schema": QUORUM_REGISTRY_SCHEMA,
        "threshold": threshold,
        "keys": [{"key_id": key_id(pub), "pubkey": pub.strip().lower()} for _, pub in signers],
    }
    return {
        "gate_pubkey": gate_pub,
        "gate_key_id": key_id(gate_pub),
        "registry": registry,
        "registry_sha256": registry_sha256(registry),
        "witness_pubkey_pem": witness_pubkey_pem,
    }


def build_reset(
    root: str | Path,
    signers: list[tuple[str, str]],
    *,
    reason: str,
    authorized_by: str,
    threshold: int = 1,
    witness_pubkey_pem: str | None = None,
    at: str | None = None,
) -> tuple[str, bytes, dict[str, Any]]:
    """Build and verify a reset record; nothing is written.

    ``signers`` are ``(private_seed_hex, pubkey_hex)`` pairs of the NEW keys;
    the first is the new gate key. Together they form the new registry at
    ``threshold``, and every one of them signs the record (possession). The
    lost side is read from the tree as it stands — ``gate_signing.pub``,
    ``gate_quorum.json`` and, when ``witness_pubkey_pem`` replaces it,
    ``witness_signing.pub`` — and the current ``checkpoint.json`` becomes the
    boundary. Returns ``(file_name,
    record_bytes, payload)`` once the same parser every verifier uses
    accepts the record.
    """
    root_path = Path(root)
    if not signers:
        raise ValueError("key reset needs at least one new key")
    for needed in (CHECKPOINT_PATH, DEFAULT_PUBKEY_PATH):
        if not (root_path / needed).is_file():
            raise ValueError(f"key reset needs the committed {needed}")
    boundary_raw = (root_path / CHECKPOINT_PATH).read_bytes()
    payload: dict[str, Any] = {
        "at": at or datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "authorized_by": authorized_by,
        "boundary": {
            "checkpoint_sha256": hash_bytes(boundary_raw),
            "checkpoint_b64": base64.b64encode(boundary_raw).decode("ascii"),
        },
        "lost": _lost_side(root_path, witness_replaced=witness_pubkey_pem is not None),
        "new": _new_side(signers, threshold, witness_pubkey_pem),
    }
    message = canonical_json_bytes(payload)
    body: dict[str, Any] = {
        "schema": RESET_SCHEMA,
        "algorithm": "ed25519",
        "payload": payload,
        "signatures": [
            {"key_id": key_id(pub), "signature": _sign(priv, message)} for priv, pub in signers
        ],
    }
    raw = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode()
    errors = parse_reset(raw, "pending")["errors"] or _reinstated_refusals(root_path, payload)
    if errors:
        raise ValueError(f"key reset refused: {errors}")
    return f"key_reset_{payload['new']['gate_key_id']}.json", raw, payload


def _reinstated_refusals(root: Path, payload: dict[str, Any]) -> list[str]:
    """A key an earlier reset declared lost stays lost; refuse to name one in
    the new key set rather than write a record the spine check rejects."""
    retired_kids: set[str] = set()
    retired_witness: set[bytes] = set()
    for r in valid_resets(load_resets(root)):
        retired_kids |= r["lost_kids"]
        spki = _spki_or_none(r["lost"].get("witness_pubkey_pem"))
        if spki is not None:
            retired_witness.add(spki)
    errors = [
        f"reinstates_lost_key:{kid}" for kid in sorted(side_key_ids(payload["new"]) & retired_kids)
    ]
    new_witness = _spki_or_none(payload["new"].get("witness_pubkey_pem"))
    if new_witness is not None and new_witness in retired_witness:
        errors.append("reinstates_lost_witness_key")
    return errors


def reset_keys(
    root: str | Path,
    signers: list[tuple[str, str]],
    *,
    reason: str,
    authorized_by: str,
    threshold: int = 1,
    witness_pubkey_pem: str | None = None,
    at: str | None = None,
) -> Path:
    """Record a key reset (see ``build_reset``) and install the new key set:
    ``gate_signing.pub``, ``gate_quorum.json`` and, when given, the new
    ``witness_signing.pub``. Refuses to overwrite an existing record."""
    root_path = Path(root)
    name, raw, payload = build_reset(
        root_path,
        signers,
        reason=reason,
        authorized_by=authorized_by,
        threshold=threshold,
        witness_pubkey_pem=witness_pubkey_pem,
        at=at,
    )
    out = root_path / RESET_DIR / name
    if out.exists():
        raise ValueError(f"key reset record already exists: {out.name}")
    atomic_write_bytes(out, raw)
    atomic_write_text(root_path / DEFAULT_PUBKEY_PATH, payload["new"]["gate_pubkey"] + "\n")
    atomic_write_bytes(
        root_path / DEFAULT_QUORUM_PATH, registry_file_bytes(payload["new"]["registry"])
    )
    if witness_pubkey_pem is not None:
        atomic_write_text(root_path / WITNESS_PUBKEY_PATH, witness_pubkey_pem)
    return out
