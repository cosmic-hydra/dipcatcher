"""Every committed receipt in ``receipts/`` must pass ``verify-receipt``.

Append-only evidence is only evidence when it verifies. ``receipts/legacy-
unsealed/`` holds the pre-seal-era artifacts quarantined as *unverifiable*;
anything in ``receipts/`` root is claimed live and must seal-verify.
``KNOWN_UNSEALED`` is a shrink-only ratchet for the legacy receipts that have
not migrated yet — a new unsealed receipt in ``receipts/`` fails this test.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_fund.research.legacy_unsealed import (
    KNOWN_UNSEALED,
    is_known_contract_legacy,
    is_known_unsealed,
)
from quant_fund.research.receipt_v2 import verify_receipt_file, verify_receipt_payload

REPO_ROOT = Path(__file__).resolve().parents[3]
RECEIPTS = REPO_ROOT / "receipts"

# KNOWN_UNSEALED / KNOWN_CONTRACT_LEGACY live in
# quant_fund.research.legacy_unsealed — byte-pinned shrink-only ratchets
# shared with `suite-health --strict`. A new unverifiable receipt in
# receipts/ fails this test; the maps may only ever shrink.


def test_committed_receipts_verify_or_are_known_legacy() -> None:
    unsealed: list[str] = []
    failures: list[str] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        # prereg_seal.v1 sidecars are seal evidence, not receipts — same
        # exclusion as proofcore.ci.receipt_paths / verified_corpus_files.
        if path.name.endswith(".seal.json"):
            continue
        result = verify_receipt_file(path)
        if result["valid"]:
            continue
        if is_known_unsealed(path, result["errors"]):
            unsealed.append(path.name)
            continue
        if is_known_contract_legacy(path, result["errors"]):
            continue
        failures.append(f"{path.name}: {result['errors']}")
    assert failures == [], "committed receipts fail verification:\n" + "\n".join(failures)
    assert not set(KNOWN_UNSEALED) - set(unsealed), (
        "KNOWN_UNSEALED has entries no longer unsealed in receipts/ — shrink the set"
    )


def _sealed_committed() -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(RECEIPTS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("receipt_sha256"), str):
            out.append((path.name, payload))
    return out


def _first_leaf(payload: dict[str, Any], *, want: type | tuple[type, ...]) -> list[Any] | None:
    stack: list[tuple[dict[str, Any], str]] = [(payload, k) for k in payload]
    while stack:
        parent, key = stack.pop()
        value = parent[key]
        if isinstance(value, want) and not isinstance(value, bool):
            return [parent, key]
        if isinstance(value, dict):
            stack.extend((value, k) for k in value)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            stack.extend((value[0], k) for k in value[0])
    return None


def test_committed_sealed_receipts_reject_tampering() -> None:
    """The seal must catch post-hoc edits on every committed receipt.

    Mutations are in-memory: the receipt is re-verified after each edit and
    every variant must come back invalid. A committed receipt that survives
    a claim edit would be evidence the seal does not cover that surface.
    """
    sealed = _sealed_committed()
    assert sealed, "no sealed committed receipts found — fixture drift"
    misses: list[str] = []
    for name, payload in sealed:
        variants: dict[str, dict[str, Any]] = {}
        # 1. seal flip
        v = dict(payload)
        sha = str(v["receipt_sha256"])
        v["receipt_sha256"] = ("0" if sha[0] != "0" else "1") + sha[1:]
        variants["seal_flip"] = v
        # 2. seal removed (a live receipt must not pass unsealed)
        v = dict(payload)
        v.pop("receipt_sha256")
        variants["seal_removed"] = v
        # 3. rename the claimed kind/schema (dispatch evasion)
        for k in ("kind", "schema"):
            if isinstance(payload.get(k), str):
                v = dict(payload)
                v[k] = "forged_kind"
                variants[f"rename_{k}"] = v
        # 4. perturb the first numeric leaf (a claim digit) — relative, so
        #    huge floats where +1.0 is below epsilon still change value
        loc = _first_leaf(payload, want=(int, float))
        if loc is not None:
            v = json.loads(json.dumps(payload))
            dst = _first_leaf(v, want=(int, float))
            assert dst is not None
            parent, key = dst
            parent[key] = float(parent[key]) * 1.5 + 1.0
            variants["numeric_claim"] = v
        # 5. drop each top-level key that exists
        for k in payload:
            if k == "receipt_sha256":
                continue
            v = dict(payload)
            v.pop(k)
            variants[f"drop_{k}"] = v
        for label, variant in variants.items():
            result = verify_receipt_payload(variant)
            if result["valid"]:
                misses.append(f"{name}:{label}")
    assert misses == [], "tampered receipts still verify — seal/contract hole:\n" + "\n".join(
        misses
    )
