"""Receipt pin-map option helpers shared by research commands.

Split out of ``research_cmds`` to keep it within its line budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer


def _sha_pin_map(path: Path | None, flag: str) -> dict[str, str] | None:
    """Load a ``--<flag>`` JSON map of receipt filename -> 64-hex sha256."""
    if path is None:
        return None
    if not path.is_file():
        raise typer.BadParameter(f"{flag} file {path} does not exist")
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"{flag} is not JSON: {exc}") from exc
    if not isinstance(raw, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and len(v) == 64 for k, v in raw.items()
    ):
        raise typer.BadParameter(f"{flag} must be a JSON object mapping filename -> 64-hex sha256")
    return dict(raw)


_ALLOWED_REMOVALS_HELP = (
    "JSON map of receipt filename -> sha256 whose removal from the corpus's "
    "epoch chain is acknowledged (quality/epoch_allowed_removals.json), as "
    "corpus-epoch --check and verify-repo apply it."
)
