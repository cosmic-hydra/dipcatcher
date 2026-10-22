"""Corpus epochs: a hash-chained integrity root over the evidence store.

``verify-receipt`` seals individual receipts; ``receipt_lattice`` cross-checks
claims inside them. Neither watches the *container*: today a receipt can be
deleted or rewritten under ``receipts/`` and nothing in the receipt format
notices — only git history does, and git is outside the sealed-evidence model.

A ``corpus_epoch.v1`` receipt stamps the corpus's full membership:

- ``members`` — every ``*.json`` in the corpus dir as ``{name, sha256}``
  (digest over raw file bytes, sealing-agnostic — pre-seal artifacts count),
  sorted by name;
- ``epoch_root_sha256`` — sha256 over the canonical member map, a Merkle-style
  root for the corpus at this instant;
- ``prev_epoch_sha256`` / ``prev_epoch_receipt`` — link to the previous epoch
  receipt, forming a chain over corpus *states*;
- ``members_added`` / ``members_removed`` — the delta vs that previous epoch.

``check_epoch_chain`` walks the committed chain and enforces the corpus
discipline: epochs form one fork-free chain, membership is non-decreasing
(a deleted receipt flips ``members_removed``), and the head epoch's root
matches the live corpus — so a receipt modified or dropped after the last
epoch stamp surfaces as ``corpus_drift_since_head_epoch``.

The chain alone cannot see its *own* head being deleted: rewind to an older
epoch still verifies internally while unstamping any members committed since.
``quality/epoch_heads.json`` therefore pins each chain's newest receipt —
stamping updates the pin in the same commit, and ``--check --heads-pin``
turns head deletion into ``epoch_head_missing``/``epoch_head_rollback``
instead of a silent rewind.

Verdicts: ``genesis`` (no previous epoch), ``advancing`` (monotone growth),
``shrinking`` (members removed — recorded fact, honest when intentional).
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from quant_fund.research.epoch_merkle import merkle_root
from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes


def member_tree_root(members: Mapping[str, str]) -> str:
    """RFC 6962-style Merkle root (domain-separated) over ``{name: sha256}``.

    Bound into the epoch payload and the heads pin so a ``corpus_proof.v1``
    verifies against the pin alone — no epoch-receipt fetch needed.
    """
    return merkle_root(members) if members else hash_bytes(b"")


EPOCH_SCHEMA = "corpus_epoch.v1"
GENESIS_PREV = "0" * 64

# ``epoch_heads.json`` is the chain's own bookkeeping (see check_epoch_chain):
# stamping updates it, so as a member it would show permanent post-stamp drift.
# Reserved name — it is never a member under any pattern in any corpus dir.
HEADS_PIN_BASENAME = "epoch_heads.json"

# Basenames never admitted as members — bookkeeping that the chain governs
# rather than measures: the heads pin is rewritten on every stamp.
EXEMPT_BASENAMES = frozenset({HEADS_PIN_BASENAME, ".epoch_stamp.lock"})

# Member rel-paths exempt per corpus dir — exact paths, not basenames, so a
# real member can never hide behind a shared filename: the timestamp-anchor
# manifests are rewritten by each anchor request and authenticate themselves —
# anchors.json via the .tsr imprints + pinned TSA certs, ots_anchors.json via
# the committed digests inside each referenced .ots proof — and the integrity
# checkpoint is a
# digest-of-pins (chained membership would stale itself instantly).
EXEMPT_RELPATHS: frozenset[str] = frozenset(
    {"timestamps/anchors.json", "timestamps/ots_anchors.json", "checkpoint.json"}
)

# Member rel-path *prefixes* exempt per corpus dir. Rekor witness proofs are
# self-authenticating (RFC 6962 inclusion + log-signed timestamps inside the
# file) and churn with every checkpoint rewrite — a new proof replaces the
# retired one each cycle, so chain membership would force an endless
# restamp-rewrite loop. Exemption is prefix-wide because filenames embed the
# per-entry log index.
EXEMPT_PREFIXES: frozenset[str] = frozenset({"witness/", "checkpoints/", "quorum_rotations/"})

# Exempt prefixes are corpus-scoped by the corpus dir's basename: a
# ``witness/`` drop under ``receipts`` is a normal member, and a
# subdirectory that is its own corpus (``.github/workflows``) must not be
# double-chained by the parent corpus.
EXEMPT_PREFIXES_BY_CORPUS: dict[str, frozenset[str]] = {
    "quality": EXEMPT_PREFIXES,
    ".github": frozenset({"workflows/"}),
    # Gitignored vendored clones inside the third_party corpus — the
    # committed vendored tree is chained; locally re-fetched copies are not.
    "third_party": frozenset({"kronos_src/", "kronos_weights/"}),
}

# Directory names never admitted as members in ANY corpus: machine-local
# build/cache artifacts (bytecode, JS deps, notebook checkpoints, tool
# caches, bundler output) — chaining them would break the chain across
# hosts and flag ordinary dev state as tamper. Verified: no tracked file
# lives under any of these names.
EXEMPT_DIRNAMES: frozenset[str] = frozenset(
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

# Directory-name *suffixes* exempt in every corpus (packaging metadata dirs
# like ``dipcatcher.egg-info/`` embed the project name in the dir name).
EXEMPT_DIR_SUFFIXES: tuple[str, ...] = (".egg-info",)


def _machine_local(relname: str) -> bool:
    parts = Path(relname).parts
    return any(part in EXEMPT_DIRNAMES or part.endswith(EXEMPT_DIR_SUFFIXES) for part in parts[:-1])


def member_digests(corpus_dir: Path | str, *, pattern: str = "*.json") -> dict[str, str]:
    """``{rel-path: sha256-of-bytes}`` for every file matching ``pattern``.

    Recursive — member keys are POSIX relative paths so nested evidence dirs
    (``verifier/runs/*.md``) are covered and digests are stable across OSes.
    On a flat dir the keys equal the plain filenames, so existing chains are
    unchanged. ``corpus_epoch_*.json`` receipts are members like any other —
    epochs stamp each other, which is what lets the chain detect a stamped
    epoch's own deletion or mutation. ``EXEMPT_BASENAMES`` and
    ``EXEMPT_RELPATHS`` are skipped: they are chain bookkeeping, not corpus
    content. Exempt rel-paths are exact corpus-relative paths — a basename
    exemption would let a real member hide under a shared filename.
    """
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ValueError(f"corpus dir {root} does not exist")
    return {
        path.relative_to(root).as_posix(): hash_bytes(path.read_bytes())
        for path in sorted(root.rglob(pattern))
        if path.is_file()
        and path.name not in EXEMPT_BASENAMES
        and path.relative_to(root).as_posix() not in EXEMPT_RELPATHS
        and not _exempt_member(root, path.relative_to(root).as_posix())
    }


def _exempt_member(root: Path, relname: str) -> bool:
    """True for membership exemptions — must mirror ``member_digests`` exactly.

    Basenames and relpaths exempt unconditionally (chain bookkeeping); the
    prefix set is corpus-scoped — a ``witness/`` drop under ``receipts`` stays
    a normal member. Machine-local dir segments (bytecode caches, JS deps,
    tool caches, bundler output) are exempt in every corpus — they are not
    content, and ``parts[:-1]`` keeps a like-named *file* a member."""
    return (
        Path(relname).name in EXEMPT_BASENAMES
        or relname in EXEMPT_RELPATHS
        or _machine_local(relname)
        or any(
            relname.startswith(prefix) for prefix in EXEMPT_PREFIXES_BY_CORPUS.get(root.name, ())
        )
    )


def _nfc_errors(members: Mapping[str, str]) -> list[str]:
    """Member names must be NFC-canonical — Unicode-equivalent spellings
    (NFC ``café`` vs NFD ``café``) alias to one file on APFS/HFS+ and
    Windows-lookup filesystems while remaining distinct strings on Linux,
    which would make inclusion/absence semantics platform-dependent.

    Also rejects distinct names that collide under NFC+casefold
    (``A.json``/``a.json``) — they can't coexist on default macOS/Windows
    checkouts, so a corpus admitting both wouldn't be portable."""
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


def _symlink_errors(root: Path, pattern: str) -> list[str]:
    """Symlink members are refused: ``is_file()`` follows links, so a
    link's *target* content is hashed — retargeting is a mutation vector
    (and the target may live outside the corpus entirely). A corpus member
    must be a real file whose bytes the name unambiguously owns."""
    # Scan every entry, not only glob matches: a symlinked *directory* never
    # matches the member pattern, but could hide content under the corpus.
    return [
        f"member_is_symlink:{p.relative_to(root).as_posix()!a}"
        for p in sorted(root.rglob("*"))
        if p.is_symlink()
    ]


_PORTABLE_FORBIDDEN_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})


def _portable_name_errors(members: Mapping[str, str]) -> list[str]:
    """Member names must not carry control, format, or line-separator
    characters — C0/DEL inject fake lines into human-facing verifier output,
    bidi/format codepoints (U+202E…) visually rename members, and Zl/Zp break
    line-oriented tooling. ASCII-repr'd in labels so the name can't inject
    even while being reported."""
    return [
        f"member_name_not_portable:{n!a}"
        for n in members
        if any(unicodedata.category(c) in _PORTABLE_FORBIDDEN_CATEGORIES for c in n)
    ]


def epoch_root(members: Mapping[str, str]) -> str:
    """Merkle-style root over the member map — order-free via canonical JSON."""
    for name, digest in members.items():
        if not isinstance(name, str) or not name:
            raise ValueError("member name must be a non-empty string")
        if not (isinstance(digest, str) and len(digest) == 64):
            raise ValueError(f"member {name} digest must be 64-hex")
        int(digest, 16)  # raises on non-hex
    return hash_bytes(canonical_json_bytes(dict(sorted(members.items()))))


def _member_maps(payload: Mapping[str, Any]) -> dict[str, str]:
    """`members` list → `{name: sha256}`; malformed entries map to empty."""
    out: dict[str, str] = {}
    for entry in payload.get("members") or []:
        if isinstance(entry, Mapping):
            name, sha = entry.get("name"), entry.get("sha256")
            if isinstance(name, str) and isinstance(sha, str):
                out[name] = sha
    return out


def _epoch_receipts(
    corpus_dir: Path,
) -> tuple[list[tuple[Path, Mapping[str, Any]]], list[str]]:
    """Epoch candidates among top-level ``*.json``, plus prefix squatters.

    Any file whose body (or v2 ``payload``) claims ``schema``/``kind`` =
    ``corpus_epoch.v1`` is a candidate — the ``corpus_epoch_*`` filename
    convention is writer-side, not a reader filter. Files *named*
    ``corpus_epoch_*.json`` that fail to yield an epoch payload squat the
    reserved prefix: they can't forge a chain, but they must not pass
    silently as ordinary members."""
    out: list[tuple[Path, Mapping[str, Any]]] = []
    squatters: list[str] = []
    for path in sorted(corpus_dir.glob("*.json")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, UnicodeError, ValueError):
            if path.name.startswith("corpus_epoch_"):
                squatters.append(path.name)
            continue
        body: object = doc.get("payload") if isinstance(doc, Mapping) else None
        candidate = body if isinstance(body, Mapping) else doc
        if isinstance(candidate, Mapping) and (
            candidate.get("schema") == EPOCH_SCHEMA or candidate.get("kind") == EPOCH_SCHEMA
        ):
            out.append((path, candidate))
        elif path.name.startswith("corpus_epoch_"):
            squatters.append(path.name)
    return out, squatters


class EpochStampLocked(RuntimeError):
    """Raised when another process holds the corpus's stamp lock."""


def _acquire_stamp_lock(root: Path) -> Any:
    """Exclusive non-blocking lock over the head-read -> write -> pin sequence.

    Two concurrent stamps that both observe the same head would each write
    ``prev = head`` — a chain fork (``epoch_multiple_heads``) created by
    correct software. The lock serializes the whole stamp; a contended lock
    fails closed instead of forking. Advisory (fcntl flock): a process that
    never takes the lock is unaffected, which is fine — the chain checker
    still catches any fork.
    """
    import fcntl

    lock_path = root / ".epoch_stamp.lock"
    fd = lock_path.open("a")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fd.close()
        raise EpochStampLocked(
            f"another corpus-epoch stamp is in flight on {root} (lock: {lock_path})"
        ) from exc
    return fd


def corpus_epoch(
    corpus_dir: Path | str,
    *,
    head_sha: str | None = None,
    pattern: str = "*.json",
    allow_new_pattern: bool = False,
) -> dict[str, Any]:
    """Build the epoch receipt over the corpus's current membership.

    Links to the newest committed epoch receipt (highest chain position) as
    ``prev``; membership delta is computed against it. ``pattern`` selects
    which files count as members (default ``*.json``); evidence dirs that
    aren't JSON (e.g. ``verifier/`` markdown reports) stamp the same way.
    """
    root = Path(corpus_dir)
    members = member_digests(root, pattern=pattern)
    nfc = _nfc_errors(members)
    if nfc:
        raise ValueError(
            f"refusing to stamp {root}: non-NFC member names "
            f"{[e.split(':', 1)[1] for e in nfc]} — rename to the NFC form"
        )
    links = _symlink_errors(root, pattern)
    if links:
        raise ValueError(
            f"refusing to stamp {root}: symlink members "
            f"{[e.split(':', 1)[1] for e in links]} — replace with real files"
        )
    unportable = _portable_name_errors(members)
    if unportable:
        raise ValueError(
            f"refusing to stamp {root}: non-portable member names "
            f"{[e.split(':', 1)[1] for e in unportable]}"
        )
    candidates, squatters = _epoch_receipts(root)
    if squatters:
        raise ValueError(
            f"refusing to stamp {root}: files squat the corpus_epoch_* prefix "
            f"without carrying an epoch payload: {[ascii(s) for s in squatters]}"
        )
    # Chains are per-(dir, pattern): only epochs stamped with the same member
    # glob participate. Absent params.pattern means the default "*.json". A
    # receipt claiming corpus_epoch.v1 with non-mapping params can't carry a
    # chain pattern — the checker flags it; it neither parents a stamp nor
    # constrains patterns.
    epochs: list[tuple[Path, Mapping[str, Any]]] = []
    foreign_patterns: set[str] = set()
    for p, e in candidates:
        params = e.get("params")
        if params is not None and not isinstance(params, Mapping):
            continue
        pat = str((params or {}).get("pattern", "*.json"))
        if pat == pattern:
            epochs.append((p, e))
        else:
            foreign_patterns.add(pat)
    # Fail closed on a *new* pattern in a dir that already has an established
    # chain under another glob — a mismatched --glob mints a parallel
    # (dir, pattern) chain whose records then read as unstamped members of
    # the real chain, exactly the drift the heads pin exists to catch.
    if foreign_patterns and not allow_new_pattern:
        raise ValueError(
            f"refusing to stamp {root}: corpus already has epoch chains under "
            f"{sorted(foreign_patterns)}; a new pattern needs --allow-new-pattern"
        )
    # The chain head is the epoch no other epoch names as prev.
    prevs = {e.get("prev_epoch_receipt") for _, e in epochs}
    heads = [(p, e) for p, e in epochs if p.name not in prevs]
    prev_name: str | None = None
    prev_root = GENESIS_PREV
    prev_members: dict[str, str] = {}
    if heads:
        # Deterministic pick if a fork already exists (chain check flags it).
        prev_path, prev_payload = sorted(heads, key=lambda t: t[0].name)[-1]
        prev_name = prev_path.name
        prev_members = _member_maps(prev_payload)
        prev_root = str(prev_payload.get("epoch_root_sha256") or GENESIS_PREV)
    added = sorted(set(members) - set(prev_members))
    removed = sorted(set(prev_members) - set(members))
    if prev_name is None:
        verdict = "genesis"
    elif removed:
        verdict = "shrinking"
    else:
        verdict = "advancing"
    return {
        "kind": EPOCH_SCHEMA,
        "schema": EPOCH_SCHEMA,
        "research_only": True,
        "live_pnl_claim": False,
        "data_label": "CORPUS",
        "simulated_only": False,
        "inputs_sha256": hash_bytes(
            canonical_json_bytes({"corpus_dir": str(root), "pattern": pattern})
        ),
        "params": ({"head_sha": head_sha} if head_sha else {}) | {"pattern": pattern},
        "epoch_root_sha256": epoch_root(members),
        "member_tree_root": member_tree_root(members),
        "members": [{"name": n, "sha256": s} for n, s in members.items()],
        "n_members": len(members),
        "prev_epoch_sha256": prev_root,
        "prev_epoch_receipt": prev_name,
        "members_added": added,
        "members_removed": removed,
        "verdict": verdict,
    }


def epoch_contract_errors(payload: Mapping[str, Any]) -> list[str]:
    """``corpus_epoch.v1`` internal consistency; ``[]`` when clean."""
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
    if not (isinstance(root, str) and len(root) == 64):
        errors.append("epoch_root_sha256")
    elif member_map:
        try:
            if epoch_root(member_map) != root:
                errors.append("epoch_root_mismatch")
        except ValueError:
            errors.append("member_digest_not_hex")
    tree = payload.get("member_tree_root")
    if tree is not None:
        if not (isinstance(tree, str) and len(tree) == 64):
            errors.append("member_tree_root")
        elif member_map:
            try:
                if member_tree_root(member_map) != tree:
                    errors.append("member_tree_root_mismatch")
            except (ValueError, KeyError):
                errors.append("member_digest_not_hex")
    prev = payload.get("prev_epoch_sha256")
    if not (isinstance(prev, str) and len(prev) == 64):
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
    if not (isinstance(inputs, str) and len(inputs) == 64):
        errors.append("inputs_sha256")
    return errors


def epoch_heads_key(corpus_dir: Path | str, pattern: str) -> str:
    """Pin-file key for one (dir, pattern) chain, e.g. ``receipts/*.json``."""
    return f"{Path(corpus_dir).as_posix()}/{pattern}"


def load_heads_pin(pin_path: Path | str) -> dict[str, dict[str, str]]:
    """Load ``quality/epoch_heads.json`` → ``{key: {"receipt", "sha256"}}``."""
    path = Path(pin_path)
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("heads"), dict):
        raise ValueError("heads pin must be an object with a 'heads' map")
    heads: dict[str, dict[str, Any]] = {}
    for key, entry in raw["heads"].items():
        if (
            isinstance(key, str)
            and isinstance(entry, Mapping)
            and isinstance(entry.get("receipt"), str)
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
        ):
            record = {"receipt": entry["receipt"], "sha256": entry["sha256"]}
            tree_root = entry.get("tree_root")
            if tree_root is not None:
                if not (isinstance(tree_root, str) and len(tree_root) == 64):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["tree_root"] = tree_root
            chain_root = entry.get("chain_root")
            if chain_root is not None:
                if not (isinstance(chain_root, str) and len(chain_root) == 64):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["chain_root"] = chain_root
            n_epochs = entry.get("n_epochs")
            if n_epochs is not None:
                if not (isinstance(n_epochs, int) and n_epochs > 0):
                    raise ValueError(f"heads pin entry {key!r} malformed")
                record["n_epochs"] = n_epochs
            heads[key] = record
        else:
            raise ValueError(f"heads pin entry {key!r} malformed")
    return heads


def load_allowed_removals(root: Path | str) -> dict[str, str]:
    """Load ``<root>/quality/epoch_allowed_removals.json`` → name→sha256 map.

    Absent or malformed file yields an empty map — declared removals are a
    courtesy layer on top of the chain, not a prerequisite for checking it.
    ``root`` is the repo root; corpus-relative callers pass the corpus dir's
    parent.
    """
    ar_path = Path(root) / "quality" / "epoch_allowed_removals.json"
    if not ar_path.is_file():
        return {}
    try:
        raw = json.loads(ar_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if isinstance(v, str)}


def update_heads_pin(
    pin_path: Path | str,
    corpus_dir: Path | str,
    pattern: str,
    head_receipt: Path,
) -> Path:
    """Record ``head_receipt`` as the pinned chain head for (dir, pattern).

    Called by the stamping path so the pin and the epoch receipt land in the
    same commit — the committed pin is then the authoritative statement of
    which epoch is the newest, making head deletion a detectable rollback.
    """
    path = Path(pin_path)
    key = epoch_heads_key(corpus_dir, pattern)
    heads: dict[str, dict[str, Any]] = (
        {k: dict(v) for k, v in load_heads_pin(path).items()} if path.is_file() else {}
    )
    try:
        head_doc = json.loads(head_receipt.read_bytes())
    except (OSError, ValueError):
        head_doc = {}
    payload = head_doc.get("payload") if isinstance(head_doc, Mapping) else None
    body = payload if isinstance(payload, Mapping) else head_doc
    tree_root = body.get("member_tree_root") if isinstance(body, Mapping) else None
    heads[key] = {
        "receipt": head_receipt.name,
        "sha256": hash_bytes(head_receipt.read_bytes()),
    }
    if isinstance(tree_root, str) and len(tree_root) == 64:
        heads[key]["tree_root"] = tree_root
    # Position-tree commitment: a Merkle root over the ordered epoch chain
    # (leaves bind (position, name, file-sha)) so `epoch_position.v1`
    # proofs verify offline against this pin in O(log n).
    from quant_fund.research.epoch_merkle import chain_tree_root, ordered_epoch_chain

    try:
        ordered, chain_errors = ordered_epoch_chain(corpus_dir, pattern=pattern)
    except (OSError, ValueError):
        ordered, chain_errors = [], ["chain_compute_failed"]
    if not chain_errors and ordered and ordered[-1][0] == head_receipt.name:
        names_sha = [(name, file_sha) for name, file_sha, _, _, _ in ordered]
        heads[key]["chain_root"] = chain_tree_root(names_sha)
        heads[key]["n_epochs"] = len(names_sha)
    from quant_fund.utils.atomicio import atomic_write_text

    payload = {"schema": "epoch_heads.v1", "heads": heads}
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def check_epoch_chain(
    corpus_dir: Path | str,
    *,
    allowed_removals: Mapping[str, str] | None = None,
    pattern: str = "*.json",
    expected_head: Mapping[str, Any] | None = None,
    require_stamped: bool = False,
    allow_member_updates: bool = False,
) -> dict[str, Any]:
    """Walk the committed epoch chain against the live corpus.

    Returns ``{"errors": [...], "unstamped": [...], "head": <name|None>,
    "head_epoch_root": <hex|None>}``. Errors are integrity violations the
    chain is authoritative over: forked/disconnected/invalid epoch receipts,
    stamped members removed (outside ``allowed_removals`` name→sha256 pins),
    stamped members mutated, and dishonest delta fields. ``unstamped`` lists
    corpus members not covered by the head epoch — arrivals between stamps
    are the normal state, recorded not flagged.

    ``expected_head`` (from ``load_heads_pin``) pins which epoch receipt is
    the committed head: missing file → ``epoch_head_missing``, byte-drift →
    ``epoch_head_mutated``, and an observed head that isn't the pin or one
    of its descendants → ``epoch_head_rollback`` — closing the hole where
    deleting the newest epoch receipts silently reverts the chain to an
    older (pre-tamper) head.

    ``require_stamped`` promotes ``unstamped`` members to
    ``unstamped_member:`` errors — for corpora whose *every* member is
    security-critical (e.g. CI workflow definitions), an unstamped arrival
    is itself the tamper, not a pending stamp.

    ``allow_member_updates`` relaxes ``member_mutated`` for *mutable*
    corpora (quality manifests, workflow definitions) whose members
    legitimately change between stamps — the chain then provides ordered
    history attestation (every committed state, honestly declared deltas,
    pinned head) while post-stamp edits still surface as
    ``head_member_digest_drift`` until re-stamped. Append-only corpora
    (receipts, verifier records) must leave it False: a digest change
    between epochs there is tamper evidence.
    """
    root = Path(corpus_dir)
    errors: list[str] = []
    unstamped: list[str] = []

    # Committed head pin (phase 1, chain-free): the pin file states which
    # epoch receipt is the newest — a missing or byte-drifted pinned file is
    # a violation even when no chain is left to check. The ancestry walk runs
    # below once the chain resolves, catching rollback to an older head.
    exp_name: str | None = None
    if expected_head is not None:
        exp_name = str(expected_head.get("receipt") or "") or None
        exp_sha = str(expected_head.get("sha256") or "")
        exp_path = root / exp_name if exp_name else root / ""
        if exp_name is None or not exp_path.is_file():
            errors.append(f"epoch_head_missing:{(exp_name or '')!a}")
        elif hash_bytes(exp_path.read_bytes()) != exp_sha:
            errors.append(f"epoch_head_mutated:{exp_name!a}")

    # Chains are per-(dir, pattern) — epochs stamped under a different member
    # glob form their own chain and are ignored here. Files squatting the
    # corpus_epoch_* reserved prefix (no epoch payload) are flagged, not
    # ignored: they can't forge a chain but must not pass silently.
    candidates, squatters = _epoch_receipts(root)
    errors.extend(f"epoch_prefix_squat:{name!a}" for name in squatters)
    epochs: list[tuple[Path, Mapping[str, Any]]] = []
    for epoch_path, epoch_payload in candidates:
        params = epoch_payload.get("params")
        if params is not None and not isinstance(params, Mapping):
            errors.append(f"epoch_params_malformed:{epoch_path.name}")
            continue
        if (params or {}).get("pattern", "*.json") == pattern:
            epochs.append((epoch_path, epoch_payload))
    if not epochs:
        if exp_name is not None:
            errors.append(f"epoch_head_rollback:{exp_name!a}")
        errors.append("no_epoch_receipts")
        return {
            "errors": errors,
            "unstamped": [],
            "head": None,
            "head_epoch_root": None,
        }

    # Seal-verify every epoch receipt before trusting its claims.
    from quant_fund.research.receipt_v2 import verify_receipt_file

    sealed: dict[str, Mapping[str, Any]] = {}
    for path, payload in epochs:
        ver = verify_receipt_file(path)
        if not ver["valid"]:
            errors.append(f"epoch_receipt_invalid:{path.name!a}")
            continue
        sealed[path.name] = payload
    if not sealed:
        errors.append("no_valid_epoch_receipts")
        return {
            "errors": errors,
            "unstamped": unstamped,
            "head": None,
            "head_epoch_root": None,
        }

    # Link the chain: each non-genesis epoch names prev_epoch_receipt.
    by_name = sealed
    child_of: dict[str, str] = {}  # prev_name -> epoch name (for fork check)
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

    # Membership must be non-decreasing along the chain. Members under a
    # corpus-scoped exempt prefix churn by rule (self-authenticating files
    # like Rekor witness proofs replace per checkpoint rewrite) — their
    # epoch-to-epoch presence delta is bookkeeping, not evidence loss.
    allowed = dict(allowed_removals or {})
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
        actual_removed = set(prev_members) - set(cur_members)
        if declared_removed != actual_removed:
            errors.append(f"members_removed_dishonest:{cur_name!a}")
        declared_added = {
            n for n in (by_name[cur_name].get("members_added") or []) if not _exempt_member(root, n)
        }
        actual_added = set(cur_members) - set(prev_members)
        if declared_added != actual_added:
            errors.append(f"members_added_dishonest:{cur_name!a}")
        # Mutated members: same name, different digest.
        for kept in set(prev_members) & set(cur_members):
            if prev_members[kept] != cur_members[kept] and not allow_member_updates:
                errors.append(f"member_mutated:{kept!a}@{cur_name!a}")

    # Head vs live corpus: stamped membership must hold exactly; files added
    # after the head stamp are unstamped (normal), not violations.
    heads = set(by_name) - set(child_of)
    head_name: str | None = None
    head_root: str | None = None
    if len(heads) == 1:
        head_name = next(iter(heads))
        head = by_name[head_name]
        root_val = head.get("epoch_root_sha256")
        head_root = root_val if isinstance(root_val, str) else None
        live = member_digests(root, pattern=pattern)
        errors += _nfc_errors(live)
        errors += _symlink_errors(root, pattern)
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

    # Committed head pin (phase 2): the observed head must be the pinned head
    # or one of its descendants — a shorter or forked chain means head epochs
    # were deleted to rewind past a committed corpus state.
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

    # Bound chain fields: when the pin carries the position-commitment
    # fields written by ``update_heads_pin`` (member ``tree_root``, ordered
    # ``chain_root``, ``n_epochs``), recompute them against the observed
    # chain truncated at the pinned head — a pin naming a live head while
    # lying about its committed position root must fail closed, since
    # offline position proofs anchor on these values.
    if exp_name is not None and ancestor and expected_head is not None:
        from quant_fund.research.epoch_merkle import (
            chain_tree_root,
            ordered_epoch_chain,
        )

        ordered, order_errors = ordered_epoch_chain(root, pattern=pattern)
        if not order_errors:
            tip_idx = next((i for i, e in enumerate(ordered) if e[0] == exp_name), None)
            if tip_idx is not None:
                sub = ordered[: tip_idx + 1]
                names_sha = [(e[0], e[1]) for e in sub]
                pinned_head = by_name[exp_name]
                exp_tree = expected_head.get("tree_root")
                if isinstance(exp_tree, str) and pinned_head.get("member_tree_root") != exp_tree:
                    errors.append(f"epoch_head_tree_root_mismatch:{exp_name!a}")
                exp_chain = expected_head.get("chain_root")
                if isinstance(exp_chain, str) and chain_tree_root(names_sha) != exp_chain:
                    errors.append(f"epoch_head_chain_root_mismatch:{exp_name!a}")
                exp_n = expected_head.get("n_epochs")
                if isinstance(exp_n, int) and len(names_sha) != exp_n:
                    errors.append(f"epoch_pin_n_epochs_mismatch:{exp_name!a}")
    return {
        "errors": errors,
        "unstamped": unstamped,
        "head": head_name,
        "head_epoch_root": head_root,
    }


def write_epoch_receipt(
    receipt: Mapping[str, Any],
    receipts_dir: Path | str = Path("receipts"),
    *,
    receipt_version: int = 1,
) -> Path:
    """Seal + atomically write a corpus-epoch receipt; fails closed on contract violation."""
    errors = epoch_contract_errors(receipt)
    if errors:
        raise ValueError(f"corpus_epoch receipt violates contract: {errors}")
    dirpath = Path(receipts_dir)
    dirpath.mkdir(parents=True, exist_ok=True)

    from quant_fund.research.fleet_eval import _atomic_write_text
    from quant_fund.research.receipt_v2 import seal_receipt, wrap_receipt_v2

    if receipt_version == 1:
        canonical = json.loads(canonical_json_bytes(dict(receipt)))
        digest = hash_bytes(canonical_json_bytes(canonical))
        payload = {**canonical, "receipt_sha256": digest}
    elif receipt_version == 2:
        payload = seal_receipt(
            wrap_receipt_v2(
                receipt,
                code_files=(Path(__file__),),
                verdict="pass" if receipt.get("verdict") in ("genesis", "advancing") else "fail",
            )
        )
        digest = str(payload["receipt_sha256"])
    else:
        raise ValueError(f"receipt_version must be 1 or 2, got {receipt_version!r}")
    path = dirpath / f"corpus_epoch_{digest[:16]}.json"
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
