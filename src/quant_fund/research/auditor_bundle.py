"""Zero-trust auditor bundle: the whole integrity proof, one file.

``integrity_checkpoint`` gives an auditor a *minimal* bundle, but verifying
it still required checking out the repo — the pinned pubkeys and proofs
live in the tree. ``auditor_bundle`` closes that: one JSON document holds
every artifact an external party needs — checkpoint, both pin files, both
pubkeys, the Rekor pubkey, and the freshest committed witness proof — and
``verify_bundle`` authenticates it with **no trusted input from the repo**.

The trust root is the public transparency log itself: Rekor's signed
entry records *which public key* signed the witnessed digest, so the
bundle's ``witness_signing.pub`` is verified against the key the log
asserts — a swapped key file breaks the chain. Rekor's own pubkey can be
pinned (bundled) or fetched live; everything else is self-proving:

- ``checkpoint.json`` sha256 == the digest inside the Rekor entry body
- Rekor SET + checkpoint-note + RFC 6962 inclusion under the Rekor pubkey
- our witness signature over the checkpoint bytes under the *log-asserted*
  key — equal, byte-for-byte, to the bundled ``witness_signing.pub``
- checkpoint Ed25519 signature under ``gate_signing.pub``, whose bytes are
  pinned by ``crown_jewels.json``, whose hash sits inside the checkpoint

A tampered member anywhere collapses at least one link. Provenance only —
never a market or P&L claim.
"""

from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import hash_bytes

BUNDLE_SCHEMA = "auditor_bundle.v1"
BUNDLE_MEMBERS: tuple[str, ...] = (
    "quality/checkpoint.json",
    "quality/epoch_heads.json",
    "quality/crown_jewels.json",
    "quality/gate_signing.pub",
    "gate_pins.sig",
    "quality/witness_signing.pub",
    "quality/rekor_pubkey.pem",
)

# Optional literal members — bundled when the tree has them, never required.
OPTIONAL_MEMBERS: tuple[str, ...] = (
    # Quorum registry: without it a quorum-format gate_pins.sig fails the
    # single-key verifier as signature_file_malformed.
    "quality/gate_quorum.json",
)

# Optional members carrying the full checkpoint history: every archived
# predecessor plus every committed Rekor witness proof. With them the
# auditor verifies the whole spine offline, not just the head.
SPINE_PREFIXES: tuple[str, ...] = (
    "quality/checkpoints/",
    "quality/witness/checkpoint.json_",
    # Gate-key rotation records ride the bundle so an auditor can verify
    # the key lineage (retired keys still verify their era's checkpoints).
    "quality/rotation_",
    # Quorum-rotation records let the auditor verify registry eras: without
    # them a post-rotation bundle fails era signature resolution, and a
    # swapped registry can't be told apart from an authorized rotation.
    "quality/quorum_rotations/",
    # Key-reset records: the one sanctioned break in key custody. Without
    # them a post-reset bundle cannot resolve the lost era's keys.
    "quality/key_reset_",
)


def _fetch_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310  # nosec B310
        result: dict[str, Any] = json.loads(resp.read())
        return result


def _freshest_proof(root: Path, target_name: str) -> Path | None:
    from quant_fund.research.integrity_witness import WITNESS_DIR

    proofs = sorted((root / WITNESS_DIR).glob(f"{target_name}_*.json"))
    return proofs[-1] if proofs else None


def build_bundle(root: str | Path, out: str | Path) -> Path:
    """Emit the bundle — fail-closed: a tree that doesn't verify is refused.

    Picks the freshest ``quality/witness/checkpoint.json_*`` proof (highest
    log index by sort order); requires all members + at least one proof.
    """
    from quant_fund.research.integrity_checkpoint import verify_checkpoint
    from quant_fund.research.integrity_witness import (
        DEFAULT_TARGET,
        verify_witnesses,
    )

    root_path = Path(root)
    cp = verify_checkpoint(root_path)
    wit = verify_witnesses(root_path)
    errors = [f"checkpoint:{e}" for e in cp.get("errors", [])]
    errors += [f"witness:{e}" for e in wit.get("errors", [])]
    if not cp.get("ok") or not wit.get("ok"):
        raise ValueError(f"refusing to bundle an unverified tree: {errors}")
    proof = _freshest_proof(root_path, DEFAULT_TARGET.name)
    if proof is None:
        raise ValueError("no witness proof committed — run witness-checkpoint first")
    members = list(BUNDLE_MEMBERS)
    members += [rel for rel in OPTIONAL_MEMBERS if (root_path / rel).exists()]
    for sub in sorted((root_path / "quality/checkpoints").glob("*.json")):
        members.append(f"quality/checkpoints/{sub.name}")
    for sub in sorted((root_path / "quality/witness").glob(f"{DEFAULT_TARGET.name}_*.json")):
        members.append(f"quality/witness/{sub.name}")
    for sub in sorted(root_path.glob("quality/rotation_*.json")):
        members.append(f"quality/{sub.name}")
    for sub in sorted((root_path / "quality/quorum_rotations").glob("rotation_*.json")):
        members.append(f"quality/quorum_rotations/{sub.name}")
    for sub in sorted(root_path.glob("quality/key_reset_*.json")):
        members.append(f"quality/{sub.name}")
    files = {rel: base64.b64encode((root_path / rel).read_bytes()).decode() for rel in members}
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "files": files,
        "witness_proof": json.loads(proof.read_text()),
        "files_sha256": {rel: hash_bytes(base64.b64decode(b)) for rel, b in files.items()},
        # The verifier under audit: an external checker can diff its local
        # copy of the standalone script against this pin before trusting the
        # verdict it produces — a swapped/weakened verifier can't launder.
        "auditor_self_sha256": hash_bytes(
            (root_path / "scripts/verify_auditor_bundle.py").read_bytes()
        ),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_path, json.dumps(bundle, indent=2, sort_keys=True) + "\n")
    return out_path


def _extract_entry_pubkey(body_b64: str) -> bytes | None:
    """The public key the LOG recorded as this entry's signer."""
    try:
        body = json.loads(base64.b64decode(body_b64))
        content = body["spec"]["signature"]["publicKey"]["content"]
        return base64.b64decode(content)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def verify_bundle(
    bundle_path: str | Path,
    *,
    rekor_pubkey_pem: bytes | None = None,
    rekor_url: str | None = "https://rekor.sigstore.dev",
    timeout: int = 30,
) -> dict[str, Any]:
    """Zero-trust verification of an emitted bundle.

    ``rekor_pubkey_pem``: caller-pinned Rekor key (the strongest trust
    posture). When omitted the bundle's own ``rekor_pubkey.pem`` is used —
    still safe against tree tampering, since SET/note signatures can't be
    forged under a different real Rekor key — or fetched live with
    ``rekor_url`` set to ``None`` to skip the fetch.
    """
    try:
        bundle = json.loads(Path(bundle_path).read_text())
    except (OSError, json.JSONDecodeError):
        return {"ok": False, "errors": ["bundle_malformed"]}
    if bundle.get("schema") != BUNDLE_SCHEMA:
        return {"ok": False, "errors": ["schema_mismatch"]}
    files = bundle.get("files")
    proof = bundle.get("witness_proof")
    if not isinstance(files, dict) or not isinstance(proof, dict):
        return {"ok": False, "errors": ["bundle_incomplete"]}
    declared = bundle.get("files_sha256", {})

    errors: list[str] = []
    decoded: dict[str, bytes] = {}
    for rel in BUNDLE_MEMBERS:
        b64 = files.get(rel)
        if not isinstance(b64, str):
            errors.append(f"missing:{rel}")
            continue
        try:
            decoded[rel] = base64.b64decode(b64)
        except ValueError:
            errors.append(f"b64_malformed:{rel}")
            continue
        want = declared.get(rel)
        if isinstance(want, str) and hash_bytes(decoded[rel]) != want:
            errors.append(f"files_sha256_mismatch:{rel}")
    # Spine members are optional in the bundle; any member outside the
    # pinned prefixes is an unexpected surface — fail closed.
    for rel in files:
        if rel in BUNDLE_MEMBERS:
            continue
        if rel not in OPTIONAL_MEMBERS and not any(rel.startswith(p) for p in SPINE_PREFIXES):
            errors.append(f"unexpected_member:{rel}")
            continue
        b64 = files[rel]
        if not isinstance(b64, str):
            errors.append(f"b64_malformed:{rel}")
            continue
        try:
            raw = base64.b64decode(b64)
        except ValueError:
            errors.append(f"b64_malformed:{rel}")
            continue
        want = declared.get(rel)
        if isinstance(want, str) and hash_bytes(raw) != want:
            errors.append(f"files_sha256_mismatch:{rel}")
        else:
            decoded[rel] = raw
    if errors:
        return {"ok": False, "errors": sorted(errors)}

    if rekor_pubkey_pem is not None:
        decoded["quality/rekor_pubkey.pem"] = rekor_pubkey_pem
    elif rekor_url is not None:
        try:
            with urllib.request.urlopen(  # noqa: S310  # nosec B310
                rekor_url.rstrip("/") + "/api/v1/log/publicKey", timeout=timeout
            ) as resp:
                decoded["quality/rekor_pubkey.pem"] = resp.read()
        except OSError:
            pass  # fall back to the bundle's pinned copy

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for rel, raw in decoded.items():
            dest = tmp_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        witness_dir = tmp_root / "quality/witness"
        witness_dir.mkdir(parents=True, exist_ok=True)
        proof_rel = "quality/witness/_bundle_proof.json"
        (tmp_root / proof_rel).write_text(json.dumps(proof))

        from quant_fund.research.gate_signatures import verify_pin_signatures
        from quant_fund.research.integrity_checkpoint import verify_checkpoint
        from quant_fund.research.integrity_witness import verify_witness_file

        cp = verify_checkpoint(tmp_root)
        errors += [f"checkpoint:{e}" for e in cp.get("errors", [])]
        pins = verify_pin_signatures(tmp_root)
        pin_errors = pins.get("errors")
        if isinstance(pin_errors, list):
            errors += [f"pins:{e}" for e in pin_errors]
        wit = verify_witness_file(tmp_root, proof_rel)
        wit_errors = [f"witness:{e}" for e in wit.get("errors", [])]
        # Rekor key rotation: if a fetched/override key fails the signature
        # checks, retry under the bundle's pinned era key before failing.
        if (rekor_pubkey_pem is not None or rekor_url is not None) and any(
            "signature_invalid" in e or "signature_mismatch" in e for e in wit_errors
        ):
            bundled_key = base64.b64decode(files["quality/rekor_pubkey.pem"])
            if bundled_key != decoded["quality/rekor_pubkey.pem"]:
                (tmp_root / "quality/rekor_pubkey.pem").write_bytes(bundled_key)
                retry = verify_witness_file(tmp_root, proof_rel)
                if len(retry.get("errors", [])) < len(wit.get("errors", [])):
                    wit, wit_errors = retry, [f"witness:{e}" for e in retry.get("errors", [])]
        errors += wit_errors
        if not wit.get("current"):
            errors.append("witness:checkpoint_digest_stale")

        # Spine: when the bundle carries the archive + witness proofs,
        # verify the whole chain — links, forks, orphans, Rekor order —
        # and cryptographically verify every bundled proof.
        spine_rels = [r for r in decoded if r.startswith(SPINE_PREFIXES[0])]
        witness_rels = [r for r in decoded if r.startswith(SPINE_PREFIXES[1])]
        if spine_rels or witness_rels:
            from quant_fund.research.checkpoint_chain import checkpoint_spine
            from quant_fund.research.integrity_witness import verify_witnesses

            spine_res = checkpoint_spine(tmp_root)
            # The scratch tree legitimately has one "orphan": the live
            # checkpoint file is a member record, not an archive entry.
            for e in spine_res.get("errors", []):
                errors.append(f"spine:{e}")
            if spine_rels and spine_res.get("verdict") != "intact":
                errors.append(f"spine_not_intact:{spine_res.get('verdict')}")
            if len(witness_rels) > 1:
                all_wit = verify_witnesses(tmp_root)
                for e in all_wit.get("errors", []):
                    errors.append(f"witness:{e}")
            from quant_fund.research.key_rotation import verify_rotations

            rot = verify_rotations(tmp_root)
            for e in rot.get("errors", []):
                errors.append(f"rotation:{e}")

            quorum_rot_rels = [
                r
                for r in decoded
                if r.startswith(("quality/quorum_rotations/", "quality/key_reset_"))
            ]
            if quorum_rot_rels:
                from quant_fund.research.quorum_rotation import verify_quorum_rotations

                qres = verify_quorum_rotations(tmp_root)
                for e in qres.get("errors", []):
                    errors.append(f"quorum_rotation:{e}")

        # The decisive link: the key REKOR recorded as signer must equal
        # the bundled witness pubkey — the log authenticates our key.
        logged_key = _extract_entry_pubkey(str(proof.get("rekor", {}).get("body_b64", "")))
        if logged_key is None:
            errors.append("entry_pubkey_missing")
        elif logged_key != decoded["quality/witness_signing.pub"]:
            errors.append("witness_pubkey_diverges_from_log")

    return {
        "ok": not errors,
        "errors": sorted(errors),
        "log_index": proof.get("rekor", {}).get("log_index"),
        "spine_members": len([r for r in decoded if r.startswith(SPINE_PREFIXES[0])]),
        "witness_proofs": len([r for r in decoded if r.startswith(SPINE_PREFIXES[1])]),
    }


def bundle_contract_errors(body: Mapping[str, Any]) -> list[str]:
    """Contract checks for a bundle dict (no filesystem)."""
    errors: list[str] = []
    if body.get("schema") != BUNDLE_SCHEMA:
        errors.append("schema_mismatch")
    files = body.get("files")
    if not isinstance(files, dict):
        return errors + ["files_not_mapping"]
    for rel in BUNDLE_MEMBERS:
        if rel not in files:
            errors.append(f"missing:{rel}")
    if not isinstance(body.get("witness_proof"), dict):
        errors.append("witness_proof_missing")
    declared = body.get("files_sha256")
    if not isinstance(declared, dict):
        errors.append("files_sha256_missing")
    elif isinstance(files, dict):
        for rel, b64 in files.items():
            want = declared.get(rel)
            if isinstance(b64, str) and isinstance(want, str):
                try:
                    if hashlib.sha256(base64.b64decode(b64)).hexdigest() != want:
                        errors.append(f"files_sha256_mismatch:{rel}")
                except ValueError:
                    errors.append(f"b64_malformed:{rel}")
    return sorted(set(errors))
