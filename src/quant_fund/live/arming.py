"""Venue selection for the live desk. Transmission is a separate switch.

Shadow never transmits. Loopback stays inside the process and labels fills
SYNTHETIC. Drop writes tickets only when the process environment has
``DIP_LIVE_ARM=1``. An arming file cannot declare a live-performance claim
or mark the institutional evidence conditions ready.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

VENUES: frozenset[str] = frozenset({"shadow", "loopback", "drop"})
CONFIRM_DROP = "SEND-LIVE-ORDERS"
ENV_ARM = "DIP_LIVE_ARM"
ENV_ROOT = "DIP_LIVE_ROOT"
_SCHEMA = "live_arming.v1"


class ArmingError(ValueError):
    """The arming file is missing a required refusal or names an unknown venue."""


@dataclass(frozen=True)
class Arming:
    """Persisted venue. The evidence flags are fixed; callers cannot set them."""

    venue: str

    def __post_init__(self) -> None:
        if self.venue not in VENUES:
            raise ArmingError(f"unknown venue {self.venue!r}")

    def as_dict(self) -> dict[str, object]:
        capital = "operator_bridge" if self.venue == "drop" else "none"
        return {
            "schema": _SCHEMA,
            "venue": self.venue,
            "capital": capital,
            "live_pnl_claim": False,
            "would_promote_live": False,
            "evidence_conditions": "blocked",
        }


def transmits(arming: Arming, env: Mapping[str, str], *, halted: bool, trace_ok: bool) -> bool:
    """True only when this process may hand a ticket to a bridge."""
    if halted or not trace_ok:
        return False
    if arming.venue == "loopback":
        return True
    if arming.venue == "drop":
        return env.get(ENV_ARM) == "1"
    return False


def held_reason(arming: Arming, env: Mapping[str, str]) -> str:
    """Why a risk-approved order would not leave the process."""
    if arming.venue == "shadow":
        return "shadow venue does not transmit"
    if arming.venue == "drop" and env.get(ENV_ARM) != "1":
        return f"{ENV_ARM} is not 1"
    return ""


def write_arming(path: Path, arming: Arming) -> None:
    _atomic_text(path, json.dumps(arming.as_dict(), sort_keys=True) + "\n")


def read_arming(path: Path) -> Arming:
    """Load an arming file. A missing file is shadow. A forged claim raises."""
    if not path.exists():
        return Arming("shadow")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ArmingError("arming file must be a JSON object")
    _require_closed(raw)
    venue = raw.get("venue")
    if not isinstance(venue, str):
        raise ArmingError("arming venue must be a string")
    return Arming(venue)


def _require_closed(raw: Mapping[str, Any]) -> None:
    if raw.get("schema") != _SCHEMA:
        raise ArmingError("arming schema is not live_arming.v1")
    if raw.get("live_pnl_claim") is not False:
        raise ArmingError("arming record must set live_pnl_claim false")
    if raw.get("would_promote_live") is not False:
        raise ArmingError("arming record must set would_promote_live false")
    if raw.get("evidence_conditions") != "blocked":
        raise ArmingError("evidence conditions stay blocked")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)
