"""Arm file for the broker gateway. This is not the operator-desk arming record.

The desk arm file (``live_arming.v1``) selects shadow, loopback, or drop.
This file (``live_arm.v1``) arms one gateway state directory. ``alpaca-live``
still requires the confirmation phrase. The broker then refuses that host;
the phrase does not authorize live capital.
"""

from __future__ import annotations

import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quant_fund.live.config import Venue
from quant_fund.live.errors import LiveGateError

LIVE_CAPITAL_PHRASE = "I AUTHORIZE LIVE CAPITAL"
_ARM_SCHEMA = "live_arm.v1"
_ZERO = "0" * 64


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Replace ``path`` with mode 0600 JSON. A crash leaves the previous file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _read_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LiveGateError(f"{path.name} is not a JSON object")
    return raw


def load_hmac_key(path: Path) -> bytes:
    """Return the on-disk HMAC key, creating a 32-byte key the first time."""
    if path.exists():
        data = path.read_bytes()
        if len(data) < 16:
            raise LiveGateError("live integrity key is unusable")
        return data
    path.parent.mkdir(parents=True, exist_ok=True)
    data = os.urandom(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    except OSError:
        os.close(fd)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(fd)
    os.chmod(path, 0o600)
    return data


def _actor(actor: str) -> str:
    cleaned = actor.strip()
    if not cleaned or len(cleaned) > 80 or any(ord(ch) < 32 for ch in cleaned):
        raise LiveGateError("actor must be 1..80 visible characters")
    return cleaned


def arm_session(
    path: Path,
    *,
    venue: Venue,
    actor: str,
    pretrade_sha256: str,
    confirm_live_capital: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write the arm file. ``alpaca-live`` requires the exact confirmation phrase."""
    who = _actor(actor)
    if len(pretrade_sha256) != 64:
        raise LiveGateError("pretrade digest is missing")
    confirmed = False
    if venue == "alpaca-live":
        phrase = confirm_live_capital.encode("utf-8")
        expected = LIVE_CAPITAL_PHRASE.encode("utf-8")
        if not hmac.compare_digest(phrase, expected):
            raise LiveGateError(
                "alpaca-live requires --confirm-live-capital to match the published phrase"
            )
        confirmed = True
    elif confirm_live_capital.strip():
        raise LiveGateError("the live-capital phrase is only used for venue alpaca-live")
    stamp = (now or datetime.now(UTC)).isoformat()
    payload: dict[str, Any] = {
        "schema": _ARM_SCHEMA,
        "venue": venue,
        "actor": who,
        "armed_at": stamp,
        "pretrade_sha256": pretrade_sha256,
        "live_capital_confirmed": confirmed,
        "live_pnl_claim": False,
        "research_evidence": False,
    }
    _atomic_json(path, payload)
    return payload


def disarm_session(path: Path, *, actor: str) -> dict[str, Any]:
    """Remove the arm file. Open positions are left where they are."""
    who = _actor(actor)
    if path.exists():
        path.unlink()
    return {
        "ok": True,
        "armed": False,
        "actor": who,
        "positions_unchanged": True,
        "live_pnl_claim": False,
        "research_evidence": False,
    }


def read_arm(path: Path, *, venue: Venue, pretrade_sha256: str) -> dict[str, Any] | None:
    """Return the arm record, or None when no arm file exists.

    A file that claims a profit result, names another venue, or was signed
    for a different pre-trade config is refused rather than treated as armed.
    """
    if not path.exists():
        return None
    raw = _read_object(path)
    if raw.get("schema") != _ARM_SCHEMA:
        raise LiveGateError("arm file schema is not supported")
    if raw.get("live_pnl_claim") is not False or raw.get("research_evidence") is not False:
        raise LiveGateError("arm file must not claim research evidence or live profit")
    if raw.get("venue") != venue:
        raise LiveGateError("arm file venue does not match the live config")
    if raw.get("pretrade_sha256") != pretrade_sha256:
        raise LiveGateError("arm file is stale for the current pre-trade config; arm again")
    if venue == "alpaca-live" and raw.get("live_capital_confirmed") is not True:
        raise LiveGateError("alpaca-live arm file is missing live-capital confirmation")
    if not isinstance(raw.get("actor"), str) or not raw["actor"].strip():
        raise LiveGateError("arm file has no actor")
    return raw


def zero_audit_head() -> str:
    return _ZERO
