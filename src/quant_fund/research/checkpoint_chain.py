"""Spine verification for the checkpoint archive.

``verify_checkpoint`` proves the *latest* checkpoint authentic. This module
proves the whole *sequence* coherent: every archived predecessor resolves,
every link's signature verifies, timestamps never regress along the spine,
no two records claim the same predecessor (a fork = a rewind attempt
committed to disk), and no witnessed or archived checkpoint escapes the
spine (an orphan = a side chain someone tried to hide or inject).

When Rekor witness proofs are present they supply an external total order:
consecutive spine members must appear in the log in chain order — a
replayed or re-sequenced chain violates the logIndex ordering Rekor
recorded.

A verified key reset (``key_reset``) lets the spine cross from a lost key
set to a new one at exactly one place — its boundary checkpoint. The
boundary must be on the spine, the first new-era checkpoint must link
directly to it, no lost key or lost registry may sign or be claimed after
it, and no new key may have signed anything up to it. Every reset is listed
in ``notes`` and counted in ``n_resets``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.utils.atomicio import atomic_write_text
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

CHAIN_SCHEMA = "checkpoint_chain.v1"

CHECKPOINT_PATH = Path("quality/checkpoint.json")
ARCHIVE_DIR = Path("quality/checkpoints")
_PUBKEY_PATH = Path("quality/gate_signing.pub")
_WITNESS_DIR = Path("quality/witness")
_WITNESS_PUBKEY_PATH = Path("quality/witness_signing.pub")


def _record(file: Path, rel: str) -> dict[str, Any]:
    data = file.read_bytes()
    rec: dict[str, Any] = {
        "sha256": hash_bytes(data),
        "path": rel,
        "body": None,
        "payload": None,
        "sig_ok": None,
        "errors": [],
    }
    try:
        body = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError):
        rec["errors"].append("malformed_json")
        return rec
    rec["body"] = body
    payload = body.get("payload")
    if isinstance(payload, dict):
        rec["payload"] = payload
    else:
        rec["errors"].append("payload_missing")
    return rec


def _verify_signature(rec: dict[str, Any], pubkey: Any, pub_hex: str) -> None:
    """Verify the record's Ed25519 signature in place."""
    from cryptography.exceptions import InvalidSignature

    from quant_fund.research.gate_signatures import key_id

    body = rec.get("body") or {}
    payload = rec.get("payload")
    if not isinstance(body, dict) or payload is None:
        rec["sig_ok"] = False
        return
    if body.get("algorithm") != "ed25519":
        rec["sig_ok"] = False
        rec["errors"].append("algorithm_unexpected")
        return
    if body.get("key_id") != key_id(pub_hex):
        rec["sig_ok"] = False
        rec["errors"].append("key_id_mismatch")
        return
    try:
        pubkey.verify(bytes.fromhex(str(body.get("signature", ""))), canonical_json_bytes(payload))
        rec["sig_ok"] = True
    except (InvalidSignature, ValueError):
        rec["sig_ok"] = False
        rec["errors"].append("signature_invalid")


def _quorum_digest(rec: dict[str, Any]) -> str | None:
    """The registry digest a checkpoint's payload claims authorized it."""
    payload = rec.get("payload") or {}
    q = payload.get("quorum")
    if isinstance(q, dict) and isinstance(q.get("registry_sha256"), str):
        return str(q["registry_sha256"])
    return None


def _signer_key_ids(rec: dict[str, Any]) -> set[str]:
    body = rec.get("body") or {}
    kids: set[str] = set()
    if isinstance(body.get("key_id"), str):
        kids.add(body["key_id"])
    for entry in body.get("signatures") or []:
        if isinstance(entry, dict) and isinstance(entry.get("key_id"), str):
            kids.add(entry["key_id"])
    return kids


def _verify_quorum_signature(
    rec: dict[str, Any],
    quorum: tuple[dict[str, str], int] | None,
    lineage: dict[str, tuple[dict[str, str], int]] | None = None,
) -> None:
    """Verify a ``integrity_checkpoint_sig.v2`` record against its registry.

    The record's own ``payload.quorum.registry_sha256`` names the registry
    that authorized it; post-field records resolve through the lineage map
    (rotation records + live), pre-field archives fall back to the live
    registry — on a tree that never rotated, they are the same set. Distinct
    valid signers must reach that registry's threshold.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    rec["sig_ok"] = False
    body = rec.get("body") or {}
    payload = rec.get("payload")
    if body.get("algorithm") != "ed25519":
        rec["errors"].append("algorithm_unexpected")
        return
    claimed = _quorum_digest(rec)
    if claimed is not None and lineage is not None and claimed in lineage:
        registered, threshold = lineage[claimed]
    elif claimed is not None:
        rec["errors"].append(f"quorum_registry_unknown:{claimed[:16]}")
        return
    elif quorum is None:
        rec["errors"].append("quorum_registry_missing")
        return
    else:
        registered, threshold = quorum
    sigs = body.get("signatures")
    if not isinstance(sigs, list) or not sigs:
        rec["errors"].append("quorum_signatures_missing")
        return
    valid: set[str] = set()
    for entry in sigs:
        if not isinstance(entry, dict):
            rec["errors"].append("signature_malformed")
            continue
        kid = entry.get("key_id")
        sig_hex = entry.get("signature")
        if kid not in registered:
            rec["errors"].append("signature_key_unknown")
            continue
        if not isinstance(sig_hex, str):
            rec["errors"].append("signature_malformed")
            continue
        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(registered[str(kid)])).verify(
                bytes.fromhex(sig_hex), canonical_json_bytes(payload)
            )
            valid.add(str(kid))
        except (InvalidSignature, ValueError):
            rec["errors"].append("signature_invalid")
    if len(valid) < threshold:
        rec["errors"].append(f"quorum_below:{len(valid)}/{threshold}")
    else:
        rec["sig_ok"] = True


def collect_spine_records(root: Path) -> dict[str, dict[str, Any]]:
    """All known checkpoint records keyed by content digest."""
    records: dict[str, dict[str, Any]] = {}
    live = root / CHECKPOINT_PATH
    if live.is_file():
        rec = _record(live, CHECKPOINT_PATH.as_posix())
        records[rec["sha256"]] = rec
    archive = root / ARCHIVE_DIR
    if archive.is_dir():
        for f in sorted(archive.glob("*.json")):
            rec = _record(f, (ARCHIVE_DIR / f.name).as_posix())
            # Digest-addressed archive: a duplicate digest is the same record.
            records.setdefault(rec["sha256"], rec)
    return records


def checkpoint_spine(root: str | Path = ".") -> dict[str, Any]:
    """Verify the full checkpoint chain end-to-end.

    Returns a ``checkpoint_chain.v1`` verdict payload; ``ok`` requires every
    signature valid, the live checkpoint reachable from itself through the
    archive to a genesis record, no forks, no orphans, no timestamp or
    Rekor-order regressions.
    """
    root_path = Path(root)
    errors: list[str] = []
    live = root_path / CHECKPOINT_PATH
    if not live.is_file():
        # Neutral only on trees that never checkpointed. An archive record
        # or witness proof means the checkpoint existed — deleting it must
        # fail closed. (A pins-signed tree without checkpoints is legal.)
        archive_dir0 = root_path / ARCHIVE_DIR
        has_archive = archive_dir0.is_dir() and any(archive_dir0.glob("*.json"))
        wdir0 = root_path / _WITNESS_DIR
        has_witness = wdir0.is_dir() and any(wdir0.glob(f"{CHECKPOINT_PATH.name}_*.json"))
        if has_archive or has_witness:
            return {
                "schema": CHAIN_SCHEMA,
                "ok": False,
                "signed": True,
                "n_records": 0,
                "spine_length": 0,
                "errors": ["checkpoint_absent"],
                "verdict": "truncated",
            }
        return {
            "schema": CHAIN_SCHEMA,
            "ok": True,
            "signed": False,
            "n_records": 0,
            "spine_length": 0,
            "errors": [],
            "verdict": "absent",
        }
    pub_file = root_path / _PUBKEY_PATH
    if not pub_file.is_file():
        return {
            "schema": CHAIN_SCHEMA,
            "ok": False,
            "signed": True,
            "n_records": 0,
            "spine_length": 0,
            "errors": ["pubkey_missing"],
            "verdict": "unsigned_pub",
        }
    pub_hex = pub_file.read_text().strip()
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex))
    except ValueError:
        return {
            "schema": CHAIN_SCHEMA,
            "ok": False,
            "signed": True,
            "n_records": 0,
            "spine_length": 0,
            "errors": ["pubkey_malformed"],
            "verdict": "unsigned_pub",
        }

    records = collect_spine_records(root_path)

    # Keyring: the live key plus every retired key a rotation chain
    # authorized — records verify under the key that was current in *their*
    # era, so a legitimate rotation doesn't retroactively invalidate the
    # spine, and a record signed by a never-authorized key flags.
    from quant_fund.research.integrity_checkpoint import CHECKPOINT_SIG_SCHEMA_V2
    from quant_fund.research.key_reset import (
        load_resets,
        reset_edge_verdict,
        reset_notes,
        reset_record_errors,
        reset_spine_errors,
        valid_resets,
    )
    from quant_fund.research.key_rotation import load_keyring
    from quant_fund.research.quorum_rotation import load_registry_lineage, prefield_registry

    ring = load_keyring(root_path)
    # Pre-field v2 records (no quorum claim) verify under the lineage's
    # genesis registry — the live one unless a rotation or reset moved it.
    quorum = prefield_registry(root_path)
    lineage = load_registry_lineage(root_path)
    all_resets = load_resets(root_path)
    resets = valid_resets(all_resets)
    errors.extend(reset_record_errors(all_resets))
    for rec in records.values():
        if (rec.get("body") or {}).get("schema") == CHECKPOINT_SIG_SCHEMA_V2:
            _verify_quorum_signature(rec, quorum, lineage)
            continue
        kid = str((rec.get("body") or {}).get("key_id", ""))
        rec_pub_hex = ring.get(kid, pub_hex)
        if kid and kid not in ring:
            rec["sig_ok"] = False
            rec["errors"].append("signature_key_unknown")
            continue
        try:
            rec_pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(rec_pub_hex))
        except ValueError:
            rec["sig_ok"] = False
            rec["errors"].append("signature_key_malformed")
            continue
        _verify_signature(rec, rec_pub, rec_pub_hex)

    # Rekor proofs: digest -> (log_index, integrated_time). Loaded before
    # the walk because a terminus that resolves only in Rekor is a valid
    # genesis — the bytes predate the archive, the digest was witnessed.
    witnessed_idx: dict[str, int] = {}
    witnessed_at: dict[str, float] = {}
    wdir = root_path / _WITNESS_DIR
    if wdir.is_dir():
        for proof in wdir.glob(f"{CHECKPOINT_PATH.name}_*.json"):
            try:
                rec = json.loads(proof.read_text())
                digest = rec.get("target", {}).get("sha256")
                index = rec.get("rekor", {}).get("log_index")
                itime = rec.get("rekor", {}).get("integrated_time")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(digest, str) and isinstance(index, int):
                witnessed_idx[digest] = index
                if isinstance(itime, (int, float)):
                    witnessed_at[digest] = float(itime)

    # Extent pin: the live checkpoint signs the archive set's count and tip,
    # so deleting even an orphaned record moves a signed value.
    live_rec = records.get(head_digest_claim := hash_bytes(live.read_bytes()))
    live_payload = (live_rec or {}).get("payload") or {}
    spine_claim = live_payload.get("spine")
    archive_dir = root_path / ARCHIVE_DIR
    archive_names = (
        sorted(p.name for p in archive_dir.glob("*.json")) if archive_dir.is_dir() else []
    )
    if isinstance(spine_claim, dict):
        expected_tip = archive_names[-1] if archive_names else None
        if spine_claim.get("n_archives") != len(archive_names):
            errors.append(
                f"spine_archive_count:{spine_claim.get('n_archives')}!={len(archive_names)}"
            )
        if spine_claim.get("tip") != expected_tip:
            errors.append("spine_tip_mismatch")

    # Witness extent: the checkpoint signs the proof name set at write
    # time; the live proof lands right after it, so the legal set is the
    # claim or the claim plus one new name that must witness this head.
    live_files = (
        {p.name: p for p in wdir.glob(f"{CHECKPOINT_PATH.name}_*.json")} if wdir.is_dir() else {}
    )
    witness_claim = live_payload.get("witness")
    if isinstance(witness_claim, dict):
        claimed_raw = witness_claim.get("proofs")
        # Current schema: name -> sha256 map; tolerate the earlier
        # name-list shape (digests unchecked) for legacy checkpoints.
        claimed = (
            dict(claimed_raw)
            if isinstance(claimed_raw, dict)
            else {n: None for n in (claimed_raw or [])}
        )
        missing_claimed = sorted(set(claimed) - set(live_files))
        extra = sorted(set(live_files) - set(claimed))
        if missing_claimed:
            errors.append(f"witness_proof_deleted:{','.join(missing_claimed)}")
        for name, claimed_digest in sorted(claimed.items()):
            if (
                claimed_digest is not None
                and name in live_files
                and hash_bytes(live_files[name].read_bytes()) != claimed_digest
            ):
                errors.append(f"witness_proof_drift:{name}")
        if len(extra) > 1:
            errors.append(f"witness_proof_unpinned:{','.join(extra)}")
        elif len(extra) == 1:
            # The one legal extra must be the proof for the live head.
            try:
                extra_body = json.loads((wdir / extra[0]).read_text())
                extra_target = (extra_body.get("target") or {}).get("sha256")
            except (OSError, ValueError):
                extra_target = None
            if extra_target != head_digest_claim:
                errors.append(f"witness_proof_unpinned:{extra[0]}")
    head_digest = head_digest_claim
    if witnessed_idx and head_digest not in witnessed_idx:
        errors.append("head_unwitnessed")

    # Walk head -> genesis through payload.prev_sha256.
    spine: list[str] = []
    seen: set[str] = set()
    cur: str | None = head_digest
    notes: list[str] = []
    genesis_source = "self"
    while cur is not None:
        if cur in seen:
            errors.append(f"chain_cycle:{cur[:12]}")
            break
        seen.add(cur)
        recd = records.get(cur)
        if recd is None:
            if cur in witnessed_idx:
                genesis_source = "witness"
                notes.append(f"terminus_witnessed_only:{cur[:12]}")
            else:
                errors.append(f"dangling_prev:{cur[:12]}")
            break
        spine.append(cur)
        payload = recd.get("payload") or {}
        prev = payload.get("prev_sha256")
        if prev is None:
            cur = None  # genesis reached
        elif not isinstance(prev, str) or len(prev) != 64:
            errors.append(f"prev_malformed:{cur[:12]}")
            cur = None
        else:
            cur = prev

    on_spine = set(spine)

    # Forks: two records claiming the same predecessor.
    claims: dict[str, list[str]] = {}
    for digest, rec in records.items():
        payload = rec.get("payload") or {}
        prev = payload.get("prev_sha256")
        if isinstance(prev, str):
            claims.setdefault(prev, []).append(digest)
    n_forks = 0
    for prev, claimants in claims.items():
        if len(claimants) > 1:
            n_forks += 1
            errors.append(f"fork:{prev[:12]}")

    # Orphans: archived records unreachable from the live head — a side
    # chain written to disk but never witnessed into the running sequence.
    orphans = sorted(
        digest
        for digest, rec in records.items()
        if digest not in on_spine and rec["path"] != CHECKPOINT_PATH.as_posix()
    )
    for digest in orphans:
        errors.append(f"orphan_checkpoint:{digest[:12]}")

    # Signature failures anywhere in the record set. A quorum record that
    # meets its threshold still fails on any other signature that does not
    # verify under its era (unknown key, malformed, invalid): the standalone
    # verifier rejects those too, and an unattributable signature must not
    # slip past the key-reset signer checks, which read key ids.
    sig_failures = sorted(
        digest
        for digest, rec in records.items()
        if rec["sig_ok"] is False or (rec["sig_ok"] is True and rec["errors"])
    )
    for digest in sig_failures:
        rec = records[digest]
        errors.append(f"signature_invalid:{digest[:12]}:{','.join(rec['errors'])}")

    # Timestamp monotonicity genesis -> head (spine list is head-first).
    ordered = list(reversed(spine))

    # Key resets: boundary on the spine, new keys fresh up to it, lost keys
    # and the lost registry revoked after it.
    witness_pub = root_path / _WITNESS_PUBKEY_PATH
    errors.extend(
        reset_spine_errors(
            resets,
            records,
            ordered,
            lineage=lineage,
            live_witness_pem=witness_pub.read_text() if witness_pub.is_file() else None,
        )
    )
    notes.extend(reset_notes(resets))

    # Quorum-registry continuity: adjacent members must name the same
    # authorizing registry, or the change needs a quorum_rotation.v1 record
    # signed to the predecessor registry's threshold. Registry introduction
    # (parent predates the quorum field) requires key continuity — at least
    # one child-registered signer must have signed the parent.
    from quant_fund.research.quorum_rotation import (
        QUORUM_ROTATION_DIR,
        QUORUM_ROTATION_GLOB,
        rotation_authorizes,
    )
    from quant_fund.research.quorum_rotation import (
        _record as _rotation_record,
    )

    rot_recs = (
        [
            rec
            for f in sorted((root_path / QUORUM_ROTATION_DIR).glob(QUORUM_ROTATION_GLOB))
            if not (rec := _rotation_record(f))["errors"]
        ]
        if (root_path / QUORUM_ROTATION_DIR).is_dir()
        else []
    )
    unbound_noted = False
    for parent, child in zip(ordered, ordered[1:], strict=False):
        p_rec, c_rec = records[parent], records[child]
        c_is_v2 = (c_rec.get("body") or {}).get("schema") == CHECKPOINT_SIG_SCHEMA_V2
        if not c_is_v2:
            continue
        cd = _quorum_digest(c_rec)
        pd = _quorum_digest(p_rec)
        if cd is None:
            if not unbound_noted:
                notes.append("quorum_unbound:pre-field v2 record(s)")
                unbound_noted = True
            continue
        if pd == cd:
            continue
        if cd not in lineage:
            errors.append(f"quorum_registry_unknown:{child[:12]}")
            continue
        # Key reset: the era may change only from the reset's boundary.
        reset_verdict = reset_edge_verdict(resets, parent, pd, cd)
        if reset_verdict is not None:
            if reset_verdict != "ok":
                errors.append(f"key_reset_off_boundary:{child[:12]}")
            continue
        if pd is None:
            # Registry introduction: the new quorum must share a signer with
            # the record it succeeds — an attacker-built set can't include a
            # live key it can't sign under.
            child_kids = set(lineage[cd][0])
            if not child_kids & _signer_key_ids(p_rec):
                errors.append(f"quorum_genesis_discontinuous:{child[:12]}")
            continue
        # Rotation edge: find a record chaining pd -> cd, authorized under
        # the predecessor registry the spine itself recorded.
        if not rotation_authorizes(rot_recs, pd, cd):
            errors.append(f"quorum_registry_unauthorized:{child[:12]}")

    order_ok = True
    for parent, child in zip(ordered, ordered[1:], strict=False):  # adjacent pairs
        p_at = (records[parent].get("payload") or {}).get("at")
        c_at = (records[child].get("payload") or {}).get("at")
        if isinstance(p_at, str) and isinstance(c_at, str) and c_at < p_at:
            order_ok = False
            errors.append(f"timestamp_regression:{child[:12]}")

    # Rekor order: where consecutive spine members are both witnessed the
    # log indexes must follow the chain direction.
    witnessed = witnessed_idx
    rekor_order_ok = True
    n_witnessed_on_spine = 0
    for digest in spine:
        if digest in witnessed:
            n_witnessed_on_spine += 1
    for parent, child in zip(ordered, ordered[1:], strict=False):  # adjacent pairs
        if parent in witnessed and child in witnessed and witnessed[child] <= witnessed[parent]:
            rekor_order_ok = False
            errors.append(f"rekor_order_violation:{child[:12]}")
    # A proof for a checkpoint digest we hold no record of is either
    # pre-archive history (witnessed before retention began — grandfathered
    # to a note) or a post-era anomaly (witnessed yet not archived — error).
    # The era boundary is the earliest integrated_time over witnessed
    # digests that ARE recorded locally.
    recorded_times = [witnessed_at[d] for d in spine if d in witnessed_at]
    era_start = min(recorded_times) if recorded_times else None
    for digest in sorted(set(witnessed) - set(records)):
        itime = witnessed_at.get(digest)
        if era_start is not None and itime is not None and itime < era_start:
            notes.append(f"grandfathered_witness:{digest[:12]}")
        else:
            errors.append(f"witnessed_absent:{digest[:12]}")

    ok = not errors
    return {
        "schema": CHAIN_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "ok": ok,
        "signed": True,
        "n_records": len(records),
        "spine_length": len(spine),
        "head_sha256": head_digest,
        "genesis_sha256": ordered[0] if ordered else None,
        "genesis_source": genesis_source,
        "n_witnessed_on_spine": n_witnessed_on_spine,
        "n_forks": n_forks,
        "n_orphans": len(orphans),
        "n_resets": len(resets),
        "order_ok": order_ok,
        "rekor_order_ok": rekor_order_ok,
        "errors": sorted(errors),
        "notes": sorted(notes),
        "verdict": "intact" if ok else "broken",
    }


def chain_contract_errors(payload: Any) -> list[str]:
    """Contract for a ``checkpoint_chain.v1`` verdict payload."""
    if not isinstance(payload, dict) or payload.get("schema") != CHAIN_SCHEMA:
        return ["schema_mismatch"]
    errors: list[str] = []
    for key in ("n_records", "spine_length", "n_forks", "n_orphans"):
        v = payload.get(key)
        if not isinstance(v, int) or v < 0:
            errors.append(f"{key}_malformed")
    if payload.get("verdict") not in {"intact", "broken", "absent", "unsigned_pub"}:
        errors.append("verdict_malformed")
    if (
        payload.get("signed")
        and payload.get("verdict") in {"intact", "broken"}
        and payload.get("genesis_source") not in {"self", "witness"}
    ):
        errors.append("genesis_source_malformed")
    if not isinstance(payload.get("ok"), bool):
        errors.append("ok_not_bool")
    spine_len = payload.get("spine_length")
    n_records = payload.get("n_records")
    if isinstance(spine_len, int) and isinstance(n_records, int):
        if spine_len > n_records:
            errors.append("spine_exceeds_records")
        if payload.get("ok") and spine_len != n_records:
            errors.append("ok_implies_full_spine")
    errs = payload.get("errors")
    if not isinstance(errs, list):
        errors.append("errors_missing")
    elif payload.get("ok") and errs:
        errors.append("ok_with_errors")
    elif not payload.get("ok") and isinstance(errs, list) and not errs:
        errors.append("broken_without_errors")
    return errors


def write_chain_receipt(payload: dict[str, Any], out: Path) -> Path:
    """Seal a spine verdict — even a broken one is evidence."""
    from quant_fund.research.receipt_v2 import seal_receipt

    errs = chain_contract_errors(payload)
    if errs:
        raise ValueError(f"checkpoint_chain contract: {errs}")
    atomic_write_text(out, json.dumps(seal_receipt(payload), indent=2, sort_keys=True) + "\n")
    return out
