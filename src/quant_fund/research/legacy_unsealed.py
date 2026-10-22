"""Hash-pinned allowlists of byte-exact legacy receipts.

Every committed ``receipts/*.json`` must seal-verify. ``KNOWN_UNSEALED``
previously exempted the seven pre-seal-era artifacts; those migrated to
``receipts/legacy-unsealed/`` (a quarantined subdir — see
``utils.receipt.QUARANTINED_SUBDIRS``), so the map is
empty — it must stay empty: a new unsealed receipt in ``receipts/`` fails.

``KNOWN_CONTRACT_LEGACY`` pins sealed receipts written by drill scripts
before their kind's deep contract existed — they pass the seal check but
fail contract re-derivation on shape alone. Exempt ONLY for the exact
error listed; any other failure still fails.

Shrink-only: removing an entry is always safe; adding one is a defect
(commit the new receipt sealed and contract-conformant instead).
Shared by the committed-receipts test ratchet and ``suite-health --strict``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# filename → sha256 of the committed file bytes. Empty: the seven pre-seal
# artifacts live under `receipts/legacy-unsealed/`, a quarantined subdir.
KNOWN_UNSEALED: dict[str, str] = {}

# Sealed receipts that predate their kind's deep contract (drill artifacts
# written before `changepoint_localize.v1` required a `params` block).
# Exempt ONLY for the exact `missing_params` contract error — any other
# failure still fails. Values pin the file bytes.
KNOWN_CONTRACT_LEGACY: dict[str, str] = {}


def _bytes_match(path: Path, expected: str) -> bool:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False


def is_known_unsealed(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` failed verification solely because it is a
    byte-pinned legacy receipt that predates the seal format."""
    expected = KNOWN_UNSEALED.get(path.name)
    if expected is None or errors != ["receipt_sha256_missing_or_invalid"]:
        return False
    return _bytes_match(path, expected)


def is_known_contract_legacy(path: Path, errors: list[str]) -> bool:
    """True iff ``path`` is a byte-pinned sealed receipt whose only failure
    is the `missing_params` contract error of a contract that postdates it.

    The error list may carry duplicates when multiple contract paths flag the
    same gap — the exemption key is the error *set*, not the multiset."""
    expected = KNOWN_CONTRACT_LEGACY.get(path.name)
    if expected is None or set(errors) != {"missing_params"}:
        return False
    return _bytes_match(path, expected)
