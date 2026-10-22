"""Shared receipt-seal primitive.

One convention repo-wide: ``receipt_sha256`` binds the canonical JSON bytes of
the payload, computed with a small exclusion set for keys attached post-seal
(the digest itself and any filesystem-path metadata). Lane packages wrap this
with their own exclusion set and honesty contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

ALWAYS_EXCLUDED = ("receipt_sha256",)

# Subdirectories under a receipts root whose contents are governed by a
# separate authority (``quality/legacy_quarantine.json`` byte-pins
# ``receipts/legacy-unsealed/``) rather than per-file seal verification.
# Corpus scanners (evidence audit, suite health, lattice, receipts-reverify)
# skip these; the epoch chain still hashes them as members.
QUARANTINED_SUBDIRS = frozenset({"legacy-unsealed"})


def is_quarantined(path: Path, root: Path) -> bool:
    """True iff ``path`` sits under a quarantined top-level subdir of ``root``."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    return len(rel.parts) > 1 and rel.parts[0] in QUARANTINED_SUBDIRS


def verified_corpus_files(root: Path, *, pattern: str = "*.json") -> list[Path]:
    """Recursive corpus listing minus quarantined subdirs, sorted.

    Per-file verification must match the epoch chain's ``rglob`` member
    semantics — a claim receipt dropped in a subdirectory is still corpus
    evidence, not a blind spot.

    ``*.seal.json`` sidecars (``prereg_seal.v1``) are seal evidence, not
    receipts — same exclusion as ``proofcore.ci.receipt_paths``. Feeding one
    to ``verify-receipt`` reports a spurious ``receipt_sha256_missing_or_invalid``.
    """
    return sorted(
        p
        for p in root.rglob(pattern)
        if p.is_file()
        and not is_quarantined(p, root)
        and not p.name.endswith(".seal.json")
    )


def seal_receipt(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> dict[str, Any]:
    """Copy of *receipt* with ``receipt_sha256`` bound over canonical JSON."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    return {**receipt, "receipt_sha256": hash_bytes(canonical_json_bytes(unsigned))}


def seal_errors(receipt: Mapping[str, Any], *, exclude: Iterable[str] = ()) -> list[str]:
    """Digest errors for a sealed receipt; ``[]`` when the seal matches."""
    dropped = set(exclude) | set(ALWAYS_EXCLUDED)
    unsigned = {key: value for key, value in receipt.items() if key not in dropped}
    expected = hash_bytes(canonical_json_bytes(unsigned))
    return [] if receipt.get("receipt_sha256") == expected else ["receipt_sha256"]
