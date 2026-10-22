"""Local release metadata and dry-run helpers (no docker, no network)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_verify_release_readiness_local_only_on_repo() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "verify_release_readiness.py"), "--local-only"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "changelog ok" in result.stdout


def test_release_yml_declares_dry_run_dispatch() -> None:
    text = (REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch" in text
    assert "dry_run" in text
    assert "scripts/release_dry_run.py" not in text  # local make target, not Actions script
    assert "make release-dry-run" in text or "Prefer `make release-dry-run`" in text
