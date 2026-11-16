"""Ed25519 signatures over the integrity pin files.

The sha256 pins (``quality/epoch_heads.json``, ``quality/crown_jewels.json``)
detect tamper, but a writer with tree access can re-pin after rewriting a
gate file. Signing the pin *manifest* with an Ed25519 key upgrades detection
to attribution: a forged pin needs the private key, and verification needs
only the committed public key.

Scope: the signature covers exactly the two pin files' *bytes* — one Ed25519
signature over ``canonical_json_bytes({"files": {path: sha256}})``. The
signature file lives at the repo root (``gate_pins.sig``), outside every
chained corpus dir, so stamping a chain never invalidates the signature it
attests (the stamp-order circularity).

Honesty contract: ``signed`` reports state, not truth — ``verify_pin_signatures``
returns ``unsigned`` when no pubkey/sig exists (a neutral verdict, not a
pass), ``ok`` only when both are present and verify.
"""

from __future__ import annotations

import json
from pathlib import Path

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

GATE_SIGNATURES_SCHEMA = "gate_signatures.v1"
QUORUM_SIG_SCHEMA = "gate_signatures.v2"
QUORUM_REGISTRY_SCHEMA = "gate_quorum.v1"
DEFAULT_PUBKEY_PATH = Path("quality/gate_signing.pub")
DEFAULT_QUORUM_PATH = Path("quality/gate_quorum.json")
DEFAULT_SIG_PATH = Path("gate_pins.sig")
SIGNED_FILES = ("quality/epoch_heads.json", "quality/crown_jewels.json")


def _pin_manifest(root: Path) -> dict[str, dict[str, str]]:
    return {"files": {rel: hash_bytes((root / rel).read_bytes()) for rel in SIGNED_FILES}}


def manifest_bytes(root: Path) -> bytes:
    """Canonical signed payload — byte-exact for sign and verify."""
    return canonical_json_bytes(_pin_manifest(root))


def generate_keypair() -> tuple[str, str]:
    """Return (private_raw_seed_hex, public_raw_hex) — 32-byte seeds."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    key = Ed25519PrivateKey.generate()
    priv = key.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption())
    pub = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv.hex(), pub.hex()


def key_id(pubkey_hex: str) -> str:
    return hash_bytes(bytes.fromhex(pubkey_hex))[:16]


def sign_pins(
    root: str | Path,
    private_seed_hex: str,
    pubkey_hex: str,
) -> Path:
    """Write ``gate_pins.sig`` (and the pubkey file if absent). Atomic."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.atomicio import atomic_write_text

    root_path = Path(root)
    manifest = _pin_manifest(root_path)
    key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(private_seed_hex))
    sig = key.sign(canonical_json_bytes(manifest))
    body = {
        "schema": GATE_SIGNATURES_SCHEMA,
        "algorithm": "ed25519",
        "key_id": key_id(pubkey_hex),
        "payload": manifest,
        "signature": sig.hex(),
    }
    atomic_write_text(
        root_path / DEFAULT_PUBKEY_PATH,
        pubkey_hex + "\n",
    )
    sig_path = root_path / DEFAULT_SIG_PATH
    atomic_write_text(sig_path, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return sig_path


def verify_pin_signatures(
    root: str | Path,
    *,
    pubkey_path: Path = DEFAULT_PUBKEY_PATH,
    sig_path: Path = DEFAULT_SIG_PATH,
    quorum_path: Path = DEFAULT_QUORUM_PATH,
) -> dict[str, object]:
    """Verify the pin manifest signature.

    ``errors`` is empty on a valid signature; ``signed`` distinguishes
    "unsigned repo" (no key material — neutral) from "verified" and "forged".
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    root_path = Path(root)
    errors: list[str] = []
    pub_file = root_path / pubkey_path
    sig_file = root_path / sig_path
    quorum_present = (root_path / quorum_path).exists()
    if not pub_file.exists() and not sig_file.exists() and not quorum_present:
        return {"ok": True, "signed": False, "errors": []}
    if quorum_present:
        return _verify_quorum(root_path, sig_file, quorum_path=quorum_path)
    if not pub_file.exists():
        return {"ok": False, "signed": True, "errors": ["pubkey_missing"]}
    if not sig_file.exists():
        return {"ok": False, "signed": True, "errors": ["signature_file_missing"]}
    pubkey_hex = pub_file.read_text().strip()
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
    except ValueError:
        return {"ok": False, "signed": True, "errors": ["pubkey_malformed"]}
    try:
        body = json.loads(sig_file.read_text())
    except (OSError, ValueError):
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if body.get("schema") != GATE_SIGNATURES_SCHEMA or body.get("algorithm") != "ed25519":
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if body.get("key_id") != key_id(pubkey_hex):
        errors.append("key_id_mismatch")
    errors.extend(_pin_drift_errors(root_path, body.get("payload", {})))
    if errors:
        return {"ok": False, "signed": True, "errors": sorted(errors)}
    try:
        pubkey.verify(
            bytes.fromhex(str(body.get("signature", ""))),
            canonical_json_bytes(body.get("payload", {})),
        )
    except InvalidSignature:
        errors.append("signature_invalid")
    except ValueError:
        errors.append("signature_malformed")
    return {"ok": not errors, "signed": True, "errors": sorted(errors)}


def _pin_drift_errors(root_path: Path, payload: object) -> list[str]:
    """The signed manifest must cover exactly ``SIGNED_FILES`` and match bytes."""
    files = payload.get("files", {}) if isinstance(payload, dict) else {}
    if set(files) != set(SIGNED_FILES):
        return ["signed_file_set_drift"]
    errors: list[str] = []
    for rel, declared in files.items():
        member = root_path / rel
        if not member.exists():
            errors.append(f"pinned_file_missing:{rel}")
        elif hash_bytes(member.read_bytes()) != declared:
            errors.append(f"pin_drift:{rel}")
    return errors


def _verify_quorum(
    root_path: Path,
    sig_file: Path,
    *,
    quorum_path: Path,
) -> dict[str, object]:
    """M-of-N quorum verification under a committed ``gate_quorum.v1`` registry.

    A quorum registry makes the lone-key format fail closed: once committed,
    a ``gate_signatures.v1`` file reports ``quorum_sig_missing`` — a downgrade
    attack cannot strip the quorum by writing the old format.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        registry = json.loads((root_path / quorum_path).read_text())
    except (OSError, ValueError):
        return {"ok": False, "signed": sig_file.exists(), "errors": ["registry_malformed"]}
    reg_errors = quorum_registry_errors(registry)
    if reg_errors:
        return {"ok": False, "signed": sig_file.exists(), "errors": reg_errors}
    threshold = int(registry["threshold"])
    registered = {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]}
    if not sig_file.exists():
        return {"ok": False, "signed": True, "errors": ["signature_file_missing"]}
    try:
        body = json.loads(sig_file.read_text())
    except (OSError, ValueError):
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if not isinstance(body, dict) or body.get("algorithm") != "ed25519":
        return {"ok": False, "signed": True, "errors": ["signature_file_malformed"]}
    if body.get("schema") != QUORUM_SIG_SCHEMA:
        return {"ok": False, "signed": True, "errors": ["quorum_sig_missing"]}
    payload = body.get("payload", {})
    errors = _pin_drift_errors(root_path, payload)
    if errors:
        return {"ok": False, "signed": True, "errors": sorted(errors)}
    msg = canonical_json_bytes(payload)
    sigs = body.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        return {"ok": False, "signed": True, "errors": ["quorum_sig_missing"]}
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
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(
                bytes.fromhex(str(entry.get("signature", ""))), msg
            )
            valid += 1
        except InvalidSignature:
            errors.append(f"signature_invalid:{kid}")
        except ValueError:
            errors.append(f"signature_malformed:{kid}")
    if valid < threshold:
        errors.append(f"quorum_not_met:{valid}/{threshold}")
    return {
        "ok": not errors,
        "signed": True,
        "errors": sorted(errors),
        "quorum": f"{valid}/{threshold}",
    }


def init_quorum(
    root: str | Path,
    pubkeys: list[str],
    *,
    threshold: int,
    labels: list[str] | None = None,
) -> Path:
    """Write the ``gate_quorum.v1`` signer registry (``quality/gate_quorum.json``).

    The registry is public — it lists which Ed25519 keys may co-sign the pin
    manifest and how many valid signatures are required. Its bytes are covered
    by the quality-corpus epoch chain, the checkpoint, and custody, so a
    committed registry is tamper-evident; forging *signatures* still needs M
    private keys.
    """
    from quant_fund.utils.atomicio import atomic_write_bytes

    if threshold < 1 or threshold > len(pubkeys):
        raise ValueError("threshold must be in [1, n_keys]")
    seen: set[str] = set()
    keys = []
    for i, pub in enumerate(pubkeys):
        bytes.fromhex(pub)  # raises on malformed
        kid = key_id(pub)
        if kid in seen:
            raise ValueError(f"duplicate pubkey: {kid}")
        seen.add(kid)
        entry = {"key_id": kid, "pubkey": pub}
        if labels is not None:
            entry["label"] = labels[i]
        keys.append(entry)
    registry = {"schema": QUORUM_REGISTRY_SCHEMA, "threshold": threshold, "keys": keys}
    path = Path(root) / DEFAULT_QUORUM_PATH
    atomic_write_bytes(path, registry_file_bytes(registry))
    return path


def registry_file_bytes(registry: dict) -> bytes:
    """The canonical on-disk bytes for a ``gate_quorum.v1`` body.

    Every registry digest — rotation records, checkpoint ``quorum`` claims,
    lineage maps — is ``sha256`` over these bytes, so on-disk equality is
    byte-exact, not formatting-dependent.
    """
    return (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode()


def registry_sha256(registry: dict) -> str:
    """Canonical digest of a ``gate_quorum.v1`` body as committed to disk."""
    return hash_bytes(registry_file_bytes(registry))


def quorum_registry_errors(registry: object) -> list[str]:
    """Structural checks on a ``gate_quorum.v1`` document; ``[]`` when clean."""
    if not isinstance(registry, dict) or registry.get("schema") != QUORUM_REGISTRY_SCHEMA:
        return ["registry_malformed"]
    keys = registry.get("keys")
    threshold = registry.get("threshold")
    errors: list[str] = []
    if not isinstance(keys, list) or not keys:
        errors.append("registry_keys_missing")
        keys = []
    if (
        not isinstance(threshold, int)
        or threshold < 1
        or (isinstance(keys, list) and keys and threshold > len(keys))
    ):
        errors.append("registry_threshold_invalid")
    seen: set[str] = set()
    for entry in keys if isinstance(keys, list) else []:
        if not isinstance(entry, dict):
            errors.append("registry_key_malformed")
            continue
        pub = entry.get("pubkey")
        kid = entry.get("key_id")
        if not isinstance(pub, str) or len(pub) != 64:
            errors.append("registry_key_malformed")
            continue
        try:
            bytes.fromhex(pub)
        except ValueError:
            errors.append("registry_key_malformed")
            continue
        if kid != key_id(pub):
            # Not necessarily a string (or hashable): never use it as a key.
            errors.append(f"registry_key_id_mismatch:{kid}")
        elif kid in seen:
            errors.append(f"registry_key_duplicate:{kid}")
        else:
            seen.add(kid)
    return sorted(errors)


def _load_quorum_registry(root: Path) -> dict[str, str] | None:  # kept for callers
    """Return ``{key_id: pubkey_hex}`` for a clean committed registry, else None."""
    reg_file = root / DEFAULT_QUORUM_PATH
    if not reg_file.exists():
        return None
    try:
        registry = json.loads(reg_file.read_text())
    except (OSError, ValueError):
        return {"__malformed__": ""}
    if quorum_registry_errors(registry):
        return {"__malformed__": ""}
    return {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]}


def load_quorum_registry(root: str | Path) -> tuple[dict[str, str], int] | None:
    """Validated committed ``gate_quorum.v1`` → ``({key_id: pubkey}, threshold)``.

    ``None`` when the file is absent OR malformed — callers that must
    distinguish the two check the registry file's existence first.
    """
    reg_file = Path(root) / DEFAULT_QUORUM_PATH
    if not reg_file.exists():
        return None
    try:
        registry = json.loads(reg_file.read_text())
    except (OSError, ValueError):
        return None
    if quorum_registry_errors(registry):
        return None
    return {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]}, int(
        registry["threshold"]
    )


def sign_pins_quorum(
    root: str | Path,
    signers: list[tuple[str, str]],
    *,
    registry_path: Path = DEFAULT_QUORUM_PATH,
    sig_path: Path = DEFAULT_SIG_PATH,
) -> Path:
    """Multi-sign the pin manifest under the committed quorum registry.

    ``signers`` are ``(private_seed_hex, pubkey_hex)`` pairs. Fails closed at
    *emission*: refuses to write a signature file that cannot satisfy the
    registry threshold — a partially-signed tree is a brick, not evidence.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from quant_fund.utils.atomicio import atomic_write_text

    root_path = Path(root)
    reg_file = root_path / registry_path
    if not reg_file.exists():
        raise ValueError("no quorum registry — run quorum-init first")
    registry = json.loads(reg_file.read_text())
    errors = quorum_registry_errors(registry)
    if errors:
        raise ValueError(f"malformed quorum registry: {errors}")
    threshold = int(registry["threshold"])
    registered = {str(e["key_id"]): str(e["pubkey"]) for e in registry["keys"]}
    manifest = _pin_manifest(root_path)
    msg = canonical_json_bytes(manifest)
    signatures = []
    for priv_hex, pub_hex in signers:
        kid = key_id(pub_hex)
        if kid not in registered or registered[kid] != pub_hex:
            raise ValueError(f"signer not in quorum registry: {kid}")
        sig = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex)).sign(msg)
        signatures.append({"key_id": kid, "signature": sig.hex()})
    if len(signatures) < threshold:
        raise ValueError(f"quorum unattainable: {len(signatures)} signers < threshold {threshold}")
    body = {
        "schema": QUORUM_SIG_SCHEMA,
        "algorithm": "ed25519",
        "payload": manifest,
        "signatures": signatures,
    }
    out = root_path / sig_path
    atomic_write_text(out, json.dumps(body, indent=2, sort_keys=True) + "\n")
    return out
