"""Hash-sealed (sidecar) pre-registration + tamper detection.

A forward-record pre-registration must be frozen **before** the window runs and
must make later *silent* edits detectable. This module provides two
content-addressed seals:

1. **Self-seal** — :func:`seal_payload` embeds a ``_seal.content_sha256`` over
   the canonical payload (excluding the seal block) inside the JSON. Editing
   any declared field breaks the embedded hash: :func:`verify_payload` fails.
2. **Sidecar seal** — :func:`seal_file` writes ``<file>.seal.json`` binding the
   file's SHA-256. Editing the file without re-sealing breaks
   :func:`verify_seal`.

Both are *local* tamper-evidence: they detect accidental or silent edits as
long as the seal is not itself rewritten by whoever controls all local files.
Independent timestamping/anchoring of the seal is an operational step (see
`docs/REALITY_PREREGISTRATION.md`); the code does not assert it occurred.
``live_pnl_claim=False`` and ``research_only=True`` throughout.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.utils.hashing import canonical_json_bytes, hash_bytes

SEAL_SCHEMA = "prereg_seal.v1"
_SEAL_KEY = "_seal"


class PreregSealError(ValueError):
    """Malformed seal input (never a tamper verdict)."""


def content_hash(payload: dict[str, Any]) -> str:
    """Canonical SHA-256 over ``payload`` excluding any embedded ``_seal``."""
    body = {k: v for k, v in payload.items() if k != _SEAL_KEY}
    return hash_bytes(canonical_json_bytes(body))


def seal_payload(payload: dict[str, Any], *, sealed_at: datetime | None = None) -> dict[str, Any]:
    """Return ``payload`` with an embedded self-seal over its content."""
    if not isinstance(payload, dict):
        raise PreregSealError("payload must be a dict")
    stamp = (sealed_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
    sealed = dict(payload)
    sealed[_SEAL_KEY] = {
        "schema": SEAL_SCHEMA,
        "content_sha256": content_hash(payload),
        "sealed_at": stamp,
    }
    return sealed


def verify_payload(sealed: dict[str, Any]) -> dict[str, Any]:
    """Verify an embedded self-seal; fail closed on any content edit."""
    seal = sealed.get(_SEAL_KEY)
    if not isinstance(seal, dict):
        return {"valid": False, "errors": ["seal_missing"]}
    errors: list[str] = []
    # The content hash excludes the seal block, so a renamed schema would
    # otherwise verify. Sidecar seals already require this schema.
    if seal.get("schema") != SEAL_SCHEMA:
        errors.append("seal_schema_mismatch")
    expected = seal.get("content_sha256")
    actual = content_hash(sealed)
    if not isinstance(expected, str) or expected != actual:
        errors.append("seal_content_hash_mismatch")
    return {"valid": not errors, "errors": errors, "expected": expected, "actual": actual}


def seal_path_for(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(f"{p.name}.seal.json")


def seal_file(path: str | Path, *, sealed_at: datetime | None = None) -> dict[str, Any]:
    """Write ``<file>.seal.json`` binding the file's SHA-256; return the seal."""
    p = Path(path)
    if not p.is_file():
        raise PreregSealError(f"cannot seal missing file: {p}")
    digest = hash_bytes(p.read_bytes())
    stamp = (sealed_at or datetime.now(tz=UTC)).astimezone(UTC).isoformat()
    seal = {"schema": SEAL_SCHEMA, "file": p.name, "sha256": digest, "sealed_at": stamp}
    side = seal_path_for(p)
    tmp = side.with_name(f".{side.name}.tmp")
    tmp.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, side)
    return seal


def verify_seal(path: str | Path, *, seal_path: str | Path | None = None) -> dict[str, Any]:
    """Verify the sidecar seal against the file bytes; fail closed on edit."""
    p = Path(path)
    side = Path(seal_path) if seal_path is not None else seal_path_for(p)
    if not side.is_file():
        return {"valid": False, "errors": ["seal_sidecar_missing"]}
    try:
        seal = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"valid": False, "errors": [f"seal_unreadable:{type(exc).__name__}"]}
    if not isinstance(seal, dict) or seal.get("schema") != SEAL_SCHEMA:
        return {"valid": False, "errors": ["seal_schema_mismatch"]}
    if not p.is_file():
        return {"valid": False, "errors": ["sealed_file_missing"]}
    expected = seal.get("sha256")
    actual = hash_bytes(p.read_bytes())
    errors: list[str] = []
    if not isinstance(expected, str) or expected != actual:
        errors.append("seal_sha256_mismatch")
    return {"valid": not errors, "errors": errors, "expected": expected, "actual": actual}


__all__ = [
    "SEAL_SCHEMA",
    "PreregSealError",
    "content_hash",
    "seal_file",
    "seal_path_for",
    "seal_payload",
    "verify_payload",
    "verify_seal",
]
