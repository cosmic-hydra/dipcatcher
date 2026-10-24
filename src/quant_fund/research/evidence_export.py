"""Evidence-bundle export — the producer half of ``verify-repo --evidence-only``.

``verify-repo --evidence-only`` verifies a tree that carries the evidence
store and nothing else — but nothing produced that tree. This module writes
it: every member of the evidence corpora (``EVIDENCE_CORPORA`` — the
receipts, verifier records, quality pins/checkpoints/witness proofs,
workflows, configs, artifacts, the ops log, and data metadata) plus the
root-level ``gate_pins.sig``, laid out at the same repo-relative paths.

An auditor copies the bundle out of band — a tarball, a thumb drive, a
release artifact — and runs ``dipcatcher verify-repo --evidence-only``
against it with no clone of the repo and no trust in the exporter: every
exported file is authenticated by the epoch chains, the signed pin
manifest, and (when witnessed) the transparency-log proofs, all of which
travel inside the bundle itself.

Membership-exempt bookkeeping is a different matter: the epoch chain
records (``corpus_epoch_*.json``), the heads pin
(``quality/epoch_heads.json``), and the signing pubkey
(``quality/gate_signing.pub``) are not corpus *members* — they are the
metadata the verifier needs to authenticate the members. The bundle
carries exactly those bookkeeping files plus the attested member set;
unaudited files (machine-local dirs like ``__pycache__``) never export.

Provenance evidence only; never a market or P&L claim.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.research.corpus_epoch import member_digests
from quant_fund.research.repo_integrity import CORPORA, EVIDENCE_CORPORA
from quant_fund.utils.hashing import hash_bytes

EVIDENCE_BUNDLE_SCHEMA = "evidence_bundle.v1"
BUNDLE_MANIFEST = "BUNDLE.json"
# The signature authenticates the pin files inside quality/; it is the one
# integrity-relevant root file the bundle must carry.
_SIGNATURE_FILE = "gate_pins.sig"
# Chain bookkeeping exempted from corpus membership but required to verify:
# the heads pin the epoch gates check, the pubkey pin_signatures needs, and
# the latest signed checkpoint — the self-contained ``--checkpoint`` auth path.
_BOOKKEEPING_FILES = (
    "quality/epoch_heads.json",
    "quality/gate_signing.pub",
    "quality/checkpoint.json",
)


def _corpus_glob(root: Path, corpus_dir: str) -> str:
    for name, pattern, *_ in CORPORA:
        if name == corpus_dir:
            return pattern
    raise KeyError(corpus_dir)


def _git_visible_relpaths(root: Path) -> set[str] | None:
    """Tracked paths, plus untracked paths git does not ignore.

    ``None`` when ``root`` is not the worktree top (synthetic trees, or git
    missing): export then copies every on-disk member. Tracked files stay
    visible even when a later ignore rule would hide them — ``git ls-files
    -c`` lists the index — so committed evidence under an ignore rule still
    exports. Untracked ignored scratch (runtime manifests, regenerated
    reports) does not: it is not corpus content, and folding it into the
    bundle makes ``require_stamped`` corpora fail ``unstamped_member``.
    """
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    if Path(top).resolve() != root.resolve():
        return None
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "-c", "-o", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    return {entry.decode("utf-8", "surrogateescape") for entry in listed.stdout.split(b"\0") if entry}


def _exportable(visible: set[str] | None, corpus_dir: str, rel: str) -> bool:
    if visible is None:
        return True
    return f"{corpus_dir}/{rel}" in visible


def _git_blob_sha1(path: Path) -> str:
    """The git blob object id for ``path``'s bytes — ``sha1("blob <n>\\0" + data)``.

    Recording it lets an auditor prove bundle member == committed blob without
    trusting the exporter: ``git ls-tree -r <source_revision>`` emits the same
    ids for the same content, so the manifest binds the bundle to the commit's
    object database, not just to a directory that happened to be present.
    """
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def _copy_corpus(
    root: Path,
    out: Path,
    corpus_dir: str,
    visible: set[str] | None,
) -> dict[str, Any]:
    """Copy one evidence corpus. Missing dir fails closed; ignored scratch stays out."""
    src = root / corpus_dir
    if not src.is_dir():
        raise FileNotFoundError(f"evidence corpus missing from tree: {corpus_dir}")
    pattern = _corpus_glob(root, corpus_dir)
    members = member_digests(src, pattern=pattern)
    copied = 0
    git_members: dict[str, str] = {}
    for rel in members:
        if not _exportable(visible, corpus_dir, rel):
            continue
        src_file = src / rel
        dst_file = out / corpus_dir / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_file, dst_file)
        git_members[rel] = _git_blob_sha1(src_file)
        copied += 1
    # Epoch receipts are exempt bookkeeping under some globs (``*.md`` /
    # ``*.yml`` don't match ``corpus_epoch_*.json``). Copy the visible ones:
    # without the chain records the heads pin is unverifiable.
    for receipt in sorted(src.glob("corpus_epoch_*.json")):
        if receipt.name in members or not _exportable(visible, corpus_dir, receipt.name):
            continue
        dst_file = out / corpus_dir / receipt.name
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(receipt, dst_file)
    return {"glob": pattern, "members": copied, "git_members": git_members}


def export_evidence_bundle(
    root: Path | str = ".", out_dir: Path | str = "evidence_bundle"
) -> dict[str, Any]:
    """Write the evidence bundle under ``out_dir``; returns the manifest.

    Copies the git-visible member set of each evidence corpus — tracked
    files, including ones an ignore rule would hide, plus untracked files
    that are not ignored. Untracked ignored scratch is left behind so a
    dirty worktree does not fail ``require_stamped`` on runtime output.
    The exported tree re-derives the pinned epoch heads byte-for-byte or
    it was wrong to call itself a bundle. Fails closed: refuses to
    overwrite a non-empty dir, and a missing corpus dir is an error
    (export is a full-tree operation; there is no partial bundle). When
    ``root`` is not its own git worktree, every on-disk member is copied.
    """
    root = Path(root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"export target not empty: {out}")

    visible = _git_visible_relpaths(root)
    corpora = {
        corpus_dir: _copy_corpus(root, out, corpus_dir, visible)
        for corpus_dir in sorted(EVIDENCE_CORPORA)
    }

    bookkeeping: dict[str, str] = {}
    for rel in _BOOKKEEPING_FILES:
        src_file = root / rel
        if src_file.is_file():
            dst_file = out / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, dst_file)
            bookkeeping[rel] = _git_blob_sha1(src_file)

    sig_src = root / _SIGNATURE_FILE
    sig_sha = ""
    if sig_src.is_file():
        shutil.copyfile(sig_src, out / _SIGNATURE_FILE)
        sig_sha = hash_bytes(sig_src.read_bytes())
        bookkeeping[_SIGNATURE_FILE] = _git_blob_sha1(sig_src)

    commit = ""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = proc.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass  # not a git tree — the manifest records an empty revision

    manifest: dict[str, Any] = {
        "schema": EVIDENCE_BUNDLE_SCHEMA,
        "exported_at": datetime.now(UTC).isoformat(),
        "source_revision": commit,
        "verify_with": "dipcatcher verify-repo --evidence-only",
        "corpora": corpora,
        "bookkeeping": bookkeeping,
        "gate_pins_sig_sha256": sig_sha,
    }
    (out / BUNDLE_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
