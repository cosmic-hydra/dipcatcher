"""fxi — the interactive front door for fx-1.

fx-1 ships two hosted model names (``fx1`` and ``fx1-lite``); each has one
endpoint (API key + base URL) managed by ``fx1.interactive.profiles`` and
injected into the dipcatcher harness. See ``fx1.interactive.app`` for the
CLI entry point and ``fx1.interactive.wizard`` for first-run onboarding.

The orchestration surface lives alongside the classic shell:

- ``concierge`` — chat, /superpower, background web research, /flash.
- ``superpower`` — capability registry, planner, coordinator.
- ``approvals`` — fail-closed gates for consequential actions and
  private-data sharing.
"""

from typing import Any

from fx1.interactive import actions, approvals, concierge, orb, profiles, shell, superpower, wizard

__all__ = [
    "actions",
    "app",
    "approvals",
    "concierge",
    "main",
    "orb",
    "profiles",
    "shell",
    "superpower",
    "wizard",
]


def __getattr__(name: str) -> Any:
    # Lazy so ``python -m fx1.interactive.app`` is not a second import of app.
    if name in {"app", "main"}:
        from fx1.interactive.app import app, main

        return {"app": app, "main": main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
