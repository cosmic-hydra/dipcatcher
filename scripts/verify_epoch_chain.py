#!/usr/bin/env python3
"""Standalone epoch-chain auditor — no repo imports, stdlib only.

Reimplements ``quant_fund.research.corpus_epoch.check_epoch_chain`` from
scratch so a third-party auditor can verify a corpus's committed epoch
chain — seal authenticity, prev links, fork/genesis/head topology, delta
honesty, and head-vs-live-tree consistency — without running (or trusting)
the repository's own code. The verdict vocabulary mirrors the library so
the two implementations can be diffed: a divergence is itself a finding.

Usage:

    python scripts/verify_epoch_chain.py \
        --corpus-dir quality --pattern '*.json' \
        --pin quality/epoch_heads.json

    # Pin authenticated by the signed checkpoint instead of trusting the
    # pin file (Ed25519 needs the `cryptography` backend; absent →
    # crypto_backend_unavailable):
    python scripts/verify_epoch_chain.py --corpus-dir quality \
        --checkpoint quality/checkpoint.json --pubkey quality/gate_signing.pub

    # Mutable corpora / strict corpora:
    --allow-member-updates   digest drift between epochs is bookkeeping
    --require-stamped        unstamped arrivals are errors, not pending
    --allowed-removals FILE  {member_name: sha256} removal allowances

Exit 0 clean, 1 errors, 2 usage. Prints one error per line, then the
verdict, matching ``verify-repo`` output shape.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

EPOCH_SCHEMA = "corpus_epoch.v1"
GENESIS_PREV = "0" * 64
HEADS_PIN_BASENAME = "epoch_heads.json"

# Mirrors corpus_epoch.py — chain bookkeeping, never members.
EXEMPT_BASENAMES = frozenset({HEADS_PIN_BASENAME, ".epoch_stamp.lock"})
EXEMPT_RELPATHS = frozenset(
    {"timestamps/anchors.json", "timestamps/ots_anchors.json", "checkpoint.json"}
)
# Corpus-scoped (root basename == "quality"): self-authenticating artifacts
# that churn by rule. ``quorum_rotations/`` carries quorum-era lineage —
# checkpoint-pinned but churning by the same rule.
EXEMPT_PREFIXES = frozenset({"witness/", "checkpoints/", "quorum_rotations/"})
# Exempt prefixes are corpus-scoped by corpus basename: ``.github/workflows``
# is its own corpus and must not double-chain under the ``.github`` corpus.
EXEMPT_PREFIXES_BY_CORPUS = {
    "quality": EXEMPT_PREFIXES,
    ".github": frozenset({"workflows/"}),
    "third_party": frozenset({"kronos_src/", "kronos_weights/"}),
}
# Gitignored lake sidecars under data/metadata. Exact names only; nested
# manifests stay members. Mirrors corpus_epoch.EXEMPT_RELPATHS_BY_CORPUS.
EXEMPT_RELPATHS_BY_CORPUS = {
    "metadata": frozenset(
        {
            "data_manifest.json",
            "resolved_config.json",
            "proofcore.duckdb",
        }
    ),
}
# Machine-local build/cache artifacts are never members in any corpus
# (bytecode, JS deps, notebook checkpoints, tool caches, bundler output).
EXEMPT_DIRNAMES = frozenset(
    {
        "__pycache__",
        "node_modules",
        ".ipynb_checkpoints",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "dist",
        "build",
        ".next",
        "test-results",
        "playwright-report",
    }
)
EXEMPT_DIR_SUFFIXES = (".egg-info",)


def _machine_local(relname: str) -> bool:
    parts = Path(relname).parts
    return any(part in EXEMPT_DIRNAMES or part.endswith(EXEMPT_DIR_SUFFIXES) for part in parts[:-1])


_PORTABLE_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})

_LEAF_PREFIX = b"\x00"
_NODE_PREFIX = b"\x01"


# ---------------------------------------------------------------- digest


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonicalize(value: Any) -> Any:
    """Minimal mirror of quant_fund.utils.hashing._canonicalize for
    JSON-parsed payloads: dict keys str-recursed, non-finite floats → None.
    Tuples/sets/datetimes/numpy cannot appear in a parsed JSON document, so
    those branches are unreachable here by construction."""
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if (
        isinstance(value, float)
        and value != value
        or (isinstance(value, float) and value in (float("inf"), float("-inf")))
    ):
        return None
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    """Mirror of canonical_json_bytes: sorted compact UTF-8, no NaN."""
    return json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=str,
    ).encode("utf-8")


def _canonical_digest(body: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(dict(body)))


def _strict_digest(body: Mapping[str, Any]) -> str:
    """The alternate seal convention: sorted compact JSON, ASCII-escaped."""
    return _sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _canon(value: Any) -> bytes:
    """canonical_json_bytes equivalent for plain JSON types — no NaN scrub,
    ASCII allowed to differ from canonical (matches verify_corpus_proof)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


# ---------------------------------------------------------------- merkle


def _leaf_hash(name: str, sha256: str) -> bytes:
    body = _canonical_json_bytes({"name": name, "sha256": sha256})
    return hashlib.sha256(_LEAF_PREFIX + body).digest()


def _node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(_NODE_PREFIX + left + right).digest()


def _merkle_root(members: Mapping[str, str]) -> str:
    """Domain-separated root over ``{name: sha256}`` — promote, never dup."""
    if not members:
        raise ValueError("empty member map has no merkle root")
    level = [_leaf_hash(n, s) for n, s in sorted(members.items())]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(_node_hash(level[i], level[i + 1]))
        if len(level) % 2 == 1:
            nxt.append(level[-1])
        level = nxt
    return level[0].hex()


def _member_tree_root(members: Mapping[str, str]) -> str:
    return _merkle_root(members) if members else _sha256(b"")


def _epoch_root(members: Mapping[str, str]) -> str:
    for name, digest in members.items():
        if not isinstance(name, str) or not name:
            raise ValueError("member name must be a non-empty string")
        if not _is_sha256(digest):
            raise ValueError(f"member {name} digest must be 64-hex")
    return _sha256(_canonical_json_bytes(dict(sorted(members.items()))))


# ------------------------------------------------------------- live tree


def _exempt_member(root: Path, relname: str) -> bool:
    """Membership exemptions — must mirror member_digests exactly."""
    return (
        Path(relname).name in EXEMPT_BASENAMES
        or relname in EXEMPT_RELPATHS
        or relname in EXEMPT_RELPATHS_BY_CORPUS.get(root.name, ())
        or _machine_local(relname)
        or any(
            relname.startswith(prefix) for prefix in EXEMPT_PREFIXES_BY_CORPUS.get(root.name, ())
        )
    )


def _member_digests(corpus_dir: Path, pattern: str) -> dict[str, str]:
    """``{rel-path: sha256-of-bytes}`` — recursive over ``rglob(pattern)``."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"corpus dir {root} does not exist")
    return {
        path.relative_to(root).as_posix(): _sha256(path.read_bytes())
        for path in sorted(root.rglob(pattern))
        if path.is_file()
        and path.name not in EXEMPT_BASENAMES
        and path.relative_to(root).as_posix() not in EXEMPT_RELPATHS
        and not _exempt_member(root, path.relative_to(root).as_posix())
    }


def _nfc_errors(members: Mapping[str, str]) -> list[str]:
    errors = [f"member_name_not_nfc:{n!a}" for n in members if unicodedata.normalize("NFC", n) != n]
    seen: dict[str, str] = {}
    for n in members:
        folded = unicodedata.normalize("NFC", n).casefold()
        other = seen.get(folded)
        if other is not None and other != n:
            errors.append(f"member_name_alias:{other!a}|{n!a}")
        else:
            seen[folded] = n
    return errors


def _symlink_errors(root: Path) -> list[str]:
    """Every entry, not only glob matches — a symlinked dir hides content."""
    return [
        f"member_is_symlink:{p.relative_to(root).as_posix()!a}"
        for p in sorted(root.rglob("*"))
        if p.is_symlink()
    ]


def _portable_name_errors(members: Mapping[str, str]) -> list[str]:
    return [
        f"member_name_not_portable:{n!a}"
        for n in members
        if any(unicodedata.category(c) in _PORTABLE_FORBIDDEN_CATEGORIES for c in n)
    ]


# ------------------------------------------------------------ receipt io


def _seal_errors(payload: Mapping[str, Any]) -> list[str]:
    """``receipt_sha256`` over the body under either digest convention."""
    seal = payload.get("receipt_sha256")
    if not _is_sha256(seal):
        return ["receipt_sha256_missing_or_invalid"]
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    computable = False
    for digest in (_canonical_digest, _strict_digest):
        try:
            actual = digest(body)
        except (TypeError, ValueError):
            continue
        computable = True
        if actual == seal:
            return []
    return ["receipt_sha256_mismatch" if computable else "receipt_sha256_uncomputable"]


def _member_maps(payload: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in payload.get("members") or []:
        if isinstance(entry, Mapping):
            name, sha = entry.get("name"), entry.get("sha256")
            if isinstance(name, str) and isinstance(sha, str):
                out[name] = sha
    return out


def _epoch_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """Independent mirror of ``corpus_epoch.epoch_contract_errors``."""
    errors: list[str] = []
    if payload.get("kind") != EPOCH_SCHEMA:
        errors.append("kind_not_corpus_epoch")
    if payload.get("schema") != EPOCH_SCHEMA:
        errors.append("schema_not_corpus_epoch")
    if payload.get("research_only") is not True:
        errors.append("research_only_not_true")
    if payload.get("live_pnl_claim") is not False:
        errors.append("live_pnl_claim_not_false")
    members = payload.get("members")
    if not isinstance(members, list):
        errors.append("members_not_list")
        members = []
    member_map = _member_maps(payload)
    if len(member_map) != len(members):
        errors.append("members_malformed_or_dup_names")
    names = [
        m["name"] for m in members if isinstance(m, Mapping) and isinstance(m.get("name"), str)
    ]
    if len(names) != len(members):
        errors.append("member_names_not_strings")
    elif names != sorted(names):
        errors.append("members_not_sorted")
    n_members = payload.get("n_members")
    if not isinstance(n_members, int) or n_members != len(members):
        errors.append("n_members_mismatch")
    root = payload.get("epoch_root_sha256")
    if not _is_sha256(root):
        errors.append("epoch_root_sha256")
    elif member_map:
        try:
            if _epoch_root(member_map) != root:
                errors.append("epoch_root_mismatch")
        except ValueError:
            errors.append("member_digest_not_hex")
    tree = payload.get("member_tree_root")
    if tree is not None:
        if not _is_sha256(tree):
            errors.append("member_tree_root")
        elif member_map:
            try:
                if _member_tree_root(member_map) != tree:
                    errors.append("member_tree_root_mismatch")
            except (ValueError, KeyError):
                errors.append("member_digest_not_hex")
    prev = payload.get("prev_epoch_sha256")
    if not _is_sha256(prev):
        errors.append("prev_epoch_sha256")
    for field in ("members_added", "members_removed"):
        lst = payload.get(field)
        if not isinstance(lst, list) or not all(isinstance(x, str) for x in lst):
            errors.append(f"{field}_not_name_list")
            continue
        if lst != sorted(lst):
            errors.append(f"{field}_not_sorted")
        if field == "members_removed" and set(lst) & set(names):
            errors.append("members_removed_still_present")
    verdict = payload.get("verdict")
    if verdict not in ("genesis", "advancing", "shrinking"):
        errors.append("verdict_unknown")
    else:
        removed = payload.get("members_removed") or []
        prev_name = payload.get("prev_epoch_receipt")
        if verdict == "genesis" and prev_name is not None:
            errors.append("verdict_genesis_with_prev")
        if verdict == "shrinking" and not removed:
            errors.append("verdict_shrinking_without_removals")
        if verdict == "advancing" and (removed or prev_name is None):
            errors.append("verdict_advancing_invalid")
    params = payload.get("params")
    if params is not None and not isinstance(params, Mapping):
        errors.append("params_not_mapping")
    elif isinstance(params, Mapping):
        pat = params.get("pattern")
        if pat is not None and not isinstance(pat, str):
            errors.append("params_pattern_not_str")
        head_sha = params.get("head_sha")
        if head_sha is not None:
            hex40 = (
                isinstance(head_sha, str)
                and len(head_sha) == 40
                and all(c in "0123456789abcdef" for c in head_sha)
            )
            if not hex40:
                errors.append("params_head_sha_not_sha1")
    inputs = payload.get("inputs_sha256")
    if not _is_sha256(inputs):
        errors.append("inputs_sha256")
    return errors


def _epoch_receipts(
    corpus_dir: Path,
) -> tuple[list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]], list[str]]:
    """Epoch candidates among top-level ``*.json``: (path, body, envelope).

    Mirrors ``corpus_epoch._epoch_receipts``: any file whose body (or the
    v2 ``payload`` envelope body) claims ``schema``/``kind`` =
    ``corpus_epoch.v1`` is a candidate — the ``corpus_epoch_*`` filename
    convention is a writer-side choice, not a reader-side filter.

    Plus the squat leg the library gained alongside this script: files
    *named* ``corpus_epoch_*.json`` that fail to yield an epoch payload
    squat the reserved prefix and must not pass silently as ordinary
    members (``epoch_prefix_squat``).
    """
    out: list[tuple[Path, Mapping[str, Any], Mapping[str, Any]]] = []
    squatters: list[str] = []
    for path in sorted(corpus_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_bytes())
        except (OSError, UnicodeError, ValueError):
            if path.name.startswith("corpus_epoch_"):
                squatters.append(path.name)
            continue
        if not isinstance(doc, Mapping):
            if path.name.startswith("corpus_epoch_"):
                squatters.append(path.name)
            continue
        body = doc.get("payload")
        candidate = body if isinstance(body, Mapping) else doc
        if isinstance(candidate, Mapping) and (
            candidate.get("schema") == EPOCH_SCHEMA or candidate.get("kind") == EPOCH_SCHEMA
        ):
            out.append((path, candidate, doc))
        elif path.name.startswith("corpus_epoch_"):
            squatters.append(path.name)
    return out, squatters


# --------------------------------------------------------------- crypto


def _ed25519_verify(pub_hex: str, sig_hex: str, msg: bytes) -> bool | None:
    """Ed25519 over raw hex — None when the backend is unavailable."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:
        return None
    try:
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub_hex.strip()))
        pub.verify(bytes.fromhex(sig_hex), msg)
    except (InvalidSignature, ValueError, TypeError):
        return False
    return True


def _ed25519_key_id(pubkey_hex: str) -> str:
    """kid = sha256(pubkey bytes)[:16] — mirrors gate_signatures.key_id."""
    return _sha256(bytes.fromhex(pubkey_hex.strip()))[:16]


def _checkpoint_heads(
    checkpoint: dict[str, Any],
    pubkey: Path,
    quorum: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Authenticate the checkpoint envelope → (heads map, pins map, errs).

    v1: ``signature`` verified under ``pubkey``. v2: a ``signatures`` list of
    ``{key_id, signature}`` — with ``--quorum <registry.json>`` every signature
    must come from a registered key and at least ``threshold`` must verify;
    without it, at least one signature must verify under ``pubkey`` and carry
    that key's id (a swapped-in key can't launder a signed payload).
    """
    errors: list[str] = []
    schema = checkpoint.get("schema")
    payload = checkpoint.get("payload")
    if not isinstance(payload, dict):
        return {}, {}, ["checkpoint_payload_missing"]
    if checkpoint.get("algorithm") != "ed25519":
        errors.append("checkpoint_algorithm")
    msg = _canon(payload)
    if schema == "integrity_checkpoint_sig.v1":
        sig = checkpoint.get("signature")
        if not isinstance(sig, str):
            errors.append("checkpoint_signature_malformed")
        else:
            # Signature message uses the plain canon (sorted compact JSON,
            # no NaN scrub) — byte-identical to verify_corpus_proof._canon.
            verdict = _ed25519_verify(pubkey.read_text().strip(), sig, msg)
            if verdict is None:
                errors.append("crypto_backend_unavailable")
            elif verdict is False:
                errors.append("checkpoint_signature_invalid")
    elif schema == "integrity_checkpoint_sig.v2":
        sigs = checkpoint.get("signatures")
        if not isinstance(sigs, list) or not sigs:
            errors.append("checkpoint_signature_malformed")
            sigs = []
        registered: dict[str, str] = {}
        threshold = 1
        if quorum is not None:
            try:
                reg = json.loads(quorum.read_bytes())
                if (
                    reg.get("schema") != "gate_quorum.v1"
                    or not isinstance(reg.get("keys"), list)
                    or not isinstance(reg.get("threshold"), int)
                ):
                    errors.append("quorum_registry_malformed")
                else:
                    registered = {
                        str(k["key_id"]): str(k["pubkey"])
                        for k in reg["keys"]
                        if isinstance(k, dict)
                    }
                    threshold = int(reg["threshold"])
                    # The presented --pubkey must itself be a registry
                    # member — else an auditor's trust anchor silently
                    # passes under keys it never chose.
                    presented = _ed25519_key_id(pubkey.read_text())
                    if presented not in registered:
                        errors.append(f"pubkey_not_in_quorum:{presented}")
                    # The signed payload names the authorizing registry's
                    # canonical digest — mirrors registry_sha256. A swapped
                    # --quorum file must fail even when it lists the keys.
                    claimed_q = payload.get("quorum")
                    claimed_digest = (
                        claimed_q.get("registry_sha256") if isinstance(claimed_q, dict) else None
                    )
                    if claimed_digest is None:
                        errors.append("quorum_unbound")
                    else:
                        canon = (json.dumps(reg, indent=2, sort_keys=True) + "\n").encode()
                        if claimed_digest != _sha256(canon):
                            errors.append("quorum_registry_drift")
            except (OSError, ValueError):
                errors.append("quorum_registry_unreadable")
        else:
            registered = {_ed25519_key_id(pubkey.read_text()): pubkey.read_text().strip()}
        seen: set[str] = set()
        valid = 0
        for entry in sigs:
            if not isinstance(entry, dict):
                errors.append("checkpoint_signature_malformed")
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
            verdict = _ed25519_verify(pub.strip(), str(entry.get("signature", "")), msg)
            if verdict is None:
                errors.append("crypto_backend_unavailable")
            elif verdict is True:
                valid += 1
            else:
                errors.append(f"checkpoint_signature_invalid:{kid}")
        if valid < threshold:
            errors.append(f"checkpoint_quorum_not_met:{valid}/{threshold}")
    else:
        errors.append("checkpoint_schema")
    heads = payload.get("heads")
    pins = payload.get("pins")
    if not isinstance(heads, dict):
        errors.append("checkpoint_heads_missing")
        heads = {}
    if not isinstance(pins, dict):
        pins = {}
    return heads, pins, errors


# ---------------------------------------------------------------- chain


def check_chain(
    corpus_dir: Path,
    *,
    pattern: str,
    allowed_removals: Mapping[str, str],
    expected_head: Mapping[str, str] | None,
    require_stamped: bool,
    allow_member_updates: bool,
) -> dict[str, Any]:
    """Mirror of ``check_epoch_chain`` — same error vocabulary, live-tree
    legs included."""
    root = Path(corpus_dir)
    errors: list[str] = []
    unstamped: list[str] = []

    exp_name: str | None = None
    if expected_head is not None:
        exp_name = str(expected_head.get("receipt") or "") or None
        exp_sha = str(expected_head.get("sha256") or "")
        exp_path = root / exp_name if exp_name else root / ""
        if exp_name is None or not exp_path.is_file():
            errors.append(f"epoch_head_missing:{(exp_name or '')!a}")
        elif _sha256(exp_path.read_bytes()) != exp_sha:
            errors.append(f"epoch_head_mutated:{exp_name!a}")

    candidates, squatters = _epoch_receipts(root)
    errors.extend(f"epoch_prefix_squat:{name!a}" for name in squatters)
    epochs = [
        (p, body, env)
        for p, body, env in candidates
        if isinstance(body.get("params") or {}, Mapping)
        and (body.get("params") or {}).get("pattern", "*.json") == pattern
    ]
    if not epochs:
        if exp_name is not None:
            errors.append(f"epoch_head_rollback:{exp_name!a}")
        errors.append("no_epoch_receipts")
        return {"errors": errors, "unstamped": [], "head": None}

    # Seal-verify every epoch envelope before trusting its claims. The v2
    # receipt seals the *envelope*; the v1 receipt seals the flat body.
    sealed: dict[str, Mapping[str, Any]] = {}
    for path, body, envelope in epochs:
        errs = _seal_errors(envelope)
        if errs or _epoch_contract_errors(body):
            errors.append(f"epoch_receipt_invalid:{path.name!a}")
            continue
        sealed[path.name] = body
    if not sealed:
        errors.append("no_valid_epoch_receipts")
        return {"errors": errors, "unstamped": unstamped, "head": None}

    by_name = sealed
    child_of: dict[str, str] = {}
    genesis: list[str] = []
    for name, payload in by_name.items():
        prev = payload.get("prev_epoch_receipt")
        if prev is None:
            genesis.append(name)
            continue
        if prev not in by_name:
            errors.append(f"epoch_orphan:{name!a}")
            continue
        if payload.get("prev_epoch_sha256") != by_name[prev].get("epoch_root_sha256"):
            errors.append(f"epoch_prev_root_mismatch:{name!a}")
        if prev in child_of:
            errors.append(f"epoch_fork:{prev!a}->{child_of[prev]!a},{name!a}")
        child_of[prev] = name
    if len(genesis) > 1:
        errors.append(f"epoch_multiple_genesis:{','.join(sorted(ascii(g) for g in genesis))}")

    allowed = dict(allowed_removals)
    for prev_name, cur_name in child_of.items():
        prev_members = {
            k: v for k, v in _member_maps(by_name[prev_name]).items() if not _exempt_member(root, k)
        }
        cur_members = {
            k: v for k, v in _member_maps(by_name[cur_name]).items() if not _exempt_member(root, k)
        }
        for gone in sorted(set(prev_members) - set(cur_members)):
            if allowed.get(gone) != prev_members[gone]:
                errors.append(f"member_removed:{gone!a}@{cur_name!a}")
        declared_removed = {
            n
            for n in (by_name[cur_name].get("members_removed") or [])
            if not _exempt_member(root, n)
        }
        if declared_removed != set(prev_members) - set(cur_members):
            errors.append(f"members_removed_dishonest:{cur_name!a}")
        declared_added = {
            n for n in (by_name[cur_name].get("members_added") or []) if not _exempt_member(root, n)
        }
        if declared_added != set(cur_members) - set(prev_members):
            errors.append(f"members_added_dishonest:{cur_name!a}")
        for kept in set(prev_members) & set(cur_members):
            if prev_members[kept] != cur_members[kept] and not allow_member_updates:
                errors.append(f"member_mutated:{kept!a}@{cur_name!a}")

    heads = set(by_name) - set(child_of)
    head_name: str | None = None
    if len(heads) == 1:
        head_name = next(iter(heads))
        head = by_name[head_name]
        live = _member_digests(root, pattern)
        errors += _nfc_errors(live)
        errors += _symlink_errors(root)
        errors += _portable_name_errors(live)
        head_members = {k: v for k, v in _member_maps(head).items() if not _exempt_member(root, k)}
        stamped = set(head_members)
        for name in stamped - set(live):
            if allowed.get(name) != head_members[name]:
                errors.append(f"head_member_missing_live:{name!a}")
        for name, sha in head_members.items():
            if name in live and live[name] != sha:
                errors.append(f"head_member_digest_drift:{name!a}")
        unstamped = sorted(set(live) - stamped - set(by_name))
        if require_stamped:
            errors.extend(f"unstamped_member:{name!a}" for name in unstamped)
    elif len(heads) > 1:
        errors.append(f"epoch_multiple_heads:{','.join(sorted(ascii(h) for h in heads))}")

    if exp_name is not None:
        cur = head_name
        seen: set[str] = set()
        ancestor = False
        while cur is not None and cur not in seen:
            if cur == exp_name:
                ancestor = True
                break
            seen.add(cur)
            prev_ref = by_name.get(cur)
            prev = prev_ref.get("prev_epoch_receipt") if isinstance(prev_ref, Mapping) else None
            cur = prev if isinstance(prev, str) else None
        if not ancestor:
            errors.append(f"epoch_head_rollback:{exp_name!a}")
    return {"errors": errors, "unstamped": unstamped, "head": head_name}


# ------------------------------------------------------------------ cli


def _load_allowed(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_bytes())
    if not isinstance(raw, Mapping):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, required=True)
    ap.add_argument("--pattern", type=str, default="*.json")
    ap.add_argument("--pin", type=Path, default=None, help="epoch_heads.json")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Signed checkpoint carrying the heads pin map",
    )
    ap.add_argument("--pubkey", type=Path, default=None)
    ap.add_argument(
        "--quorum",
        type=Path,
        default=None,
        help="gate_quorum.v1 registry for v2 checkpoints (default: --pubkey only)",
    )
    ap.add_argument("--key", type=str, default=None, help="pin key e.g. 'quality/*.json'")
    ap.add_argument("--allowed-removals", type=Path, default=None)
    ap.add_argument("--require-stamped", action="store_true")
    ap.add_argument("--allow-member-updates", action="store_true")
    args = ap.parse_args()

    if args.checkpoint is not None and args.pubkey is None:
        print("--checkpoint requires --pubkey", file=sys.stderr)
        return 2
    if not args.corpus_dir.is_dir():
        print(f"corpus dir {args.corpus_dir} does not exist", file=sys.stderr)
        return 2

    pre_errors: list[str] = []
    pin_heads: dict[str, Any] = {}
    if args.checkpoint is not None:
        checkpoint = json.loads(args.checkpoint.read_bytes())
        assert args.pubkey is not None
        heads, pins, errs = _checkpoint_heads(checkpoint, args.pubkey, args.quorum)
        pre_errors += errs
        pin_heads = heads
        if args.pin is not None:
            rel = args.pin.as_posix().removeprefix("./")
            want = pins.get(rel)
            if want is None:
                want = next((v for k, v in pins.items() if rel.endswith(k)), None)
            got = _sha256(args.pin.read_bytes())
            if want != got:
                pre_errors.append(f"checkpoint_pin_mismatch:{args.pin.name}")
    elif args.pin is not None:
        raw = json.loads(args.pin.read_bytes())
        pin_heads = raw.get("heads", {}) if isinstance(raw, dict) else {}

    key = args.key or f"{args.corpus_dir.as_posix().removeprefix('./')}/{args.pattern}"
    expected_head = pin_heads.get(key) if pin_heads else None
    if pin_heads and expected_head is None:
        # Mirror _pin_entry fallback: match by basename of the corpus dir.
        alt = {k: v for k, v in pin_heads.items() if k == f"{args.corpus_dir.name}/{args.pattern}"}
        if len(alt) == 1:
            expected_head = next(iter(alt.values()))
    if pin_heads and expected_head is None:
        pre_errors.append(f"heads_pin_key_missing:{key!a}")

    result = check_chain(
        args.corpus_dir,
        pattern=args.pattern,
        allowed_removals=_load_allowed(args.allowed_removals),
        expected_head=expected_head,
        require_stamped=args.require_stamped,
        allow_member_updates=args.allow_member_updates,
    )
    errors = pre_errors + result["errors"]
    print(
        f"epoch-chain corpus={args.corpus_dir} head={result['head']} "
        f"unstamped={len(result['unstamped'])} errors={len(errors)}"
    )
    for e in errors:
        print(f"  {e}")
    for u in result["unstamped"]:
        print(f"  unstamped:{u!a}")
    if errors:
        print("FAIL")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
