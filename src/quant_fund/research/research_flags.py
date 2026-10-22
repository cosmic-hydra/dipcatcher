"""Runtime flags for the research lab runner (no market claims)."""

from __future__ import annotations

import os


def research_required_only() -> bool:
    """Skip optional family benches when set (examples/CI smoke stay bounded)."""
    raw = os.environ.get("DIPCATCHER_RESEARCH_REQUIRED_ONLY", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
