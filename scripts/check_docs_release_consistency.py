#!/usr/bin/env python3
"""Docs-vs-code consistency for release and deploy surfaces.

Fail-closed checks (stdlib only):

1. ``release.yml`` preflight invokes ``verify_release_readiness.py`` and
   ``require_release_gates.py``; build invokes wheel smoke, artifact verify,
   and a container health wait.
2. Documented make targets ``release-dry-run``, ``docs-consistency``,
   ``container-health``, and ``health-smoke`` exist in the root Makefile.
3. Deploy docs that mention observability point at ``deploy/observability/``
   paths that exist on disk.
4. Production deploy docs must not claim live broker connectivity or authorize
   live trading (honesty contract).

Run via ``make docs-consistency`` or directly.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_YML = ROOT / ".github" / "workflows" / "release.yml"
MAKEFILE = ROOT / "Makefile"
DEPLOY_DOC = ROOT / "docs" / "PRODUCTION_DEPLOY.md"
RUNBOOK = ROOT / "docs" / "OPERATIONS_RUNBOOK.md"
OBS_DOC = ROOT / "docs" / "AUDIT_OBSERVABILITY.md"

_REQUIRED_SCRIPTS = (
    "scripts/verify_release_readiness.py",
    "scripts/require_release_gates.py",
    "scripts/smoke_installed_wheel.py",
    "scripts/verify_release_artifacts.py",
    "scripts/release_dry_run.py",
    "scripts/check_docs_release_consistency.py",
)
_REQUIRED_MAKE = (
    "release-dry-run",
    "docs-consistency",
    "container-health",
    "health-smoke",
)
_REQUIRED_OBS_PATHS = (
    "deploy/observability/docker-compose.yml",
    "deploy/observability/prometheus.yml",
    "deploy/observability/otel-collector.yaml",
    "deploy/observability/slo.yml",
)
# Affirmative live-trading claims only. Explicit refusals on the same line
# are allowed (honesty contract).
_FORBIDDEN_CLAIM = re.compile(
    r"(?i)\b("
    r"live\s+broker\s+connectivity\s+is\s+(?:enabled|supported|ready)|"
    r"authorizes?\s+live\s+trading|"
    r"production[- ]ready\s+for\s+live\s+trading|"
    r"submits?\s+live\s+orders"
    r")\b"
)
_REFUSAL_PHRASES = (
    "does not authorize live trading",
    "do not authorize live trading",
    "not authorize live trading",
    "never authorize live trading",
    "no live-trading",
    "no live trading",
    "live host is refused",
    "fail-closed",
    "fails closed",
)


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)


def check_scripts_exist() -> list[str]:
    errors: list[str] = []
    for rel in _REQUIRED_SCRIPTS:
        if not (ROOT / rel).is_file():
            errors.append(f"missing required script {rel}")
    return errors


def check_release_workflow() -> list[str]:
    errors: list[str] = []
    if not RELEASE_YML.is_file():
        return ["missing .github/workflows/release.yml"]
    text = RELEASE_YML.read_text(encoding="utf-8")
    for needle in (
        "scripts/verify_release_readiness.py",
        "scripts/require_release_gates.py",
        "scripts/smoke_installed_wheel.py",
        "scripts/verify_release_artifacts.py",
        "docker build",
        "State.Health.Status",
        "Tag version and same-commit evidence",
    ):
        if needle not in text:
            errors.append(f"release.yml missing required fragment: {needle!r}")
    if "publish to PyPI" not in text and "does not publish" not in text.lower():
        # Soft: workflow header must stay honest about non-publishing.
        if "does not publish to PyPI" not in text:
            errors.append("release.yml must state it does not publish to PyPI")
    return errors


def check_makefile_targets() -> list[str]:
    errors: list[str] = []
    text = MAKEFILE.read_text(encoding="utf-8")
    for target in _REQUIRED_MAKE:
        if not re.search(rf"^{re.escape(target)}\s*:", text, re.MULTILINE):
            errors.append(f"Makefile missing target {target}")
    return errors


def check_observability_paths() -> list[str]:
    errors: list[str] = []
    for rel in _REQUIRED_OBS_PATHS:
        if not (ROOT / rel).is_file():
            errors.append(f"missing observability path {rel}")
    for doc in (DEPLOY_DOC, RUNBOOK, OBS_DOC):
        if not doc.is_file():
            errors.append(f"missing deploy/runbook doc {doc.relative_to(ROOT)}")
            continue
        body = doc.read_text(encoding="utf-8")
        if "deploy/observability" not in body:
            errors.append(f"{doc.relative_to(ROOT)} must reference deploy/observability")
    return errors


def check_no_live_trading_claims() -> list[str]:
    errors: list[str] = []
    for doc in (DEPLOY_DOC, RUNBOOK):
        if not doc.is_file():
            continue
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), start=1):
            # Strip markdown emphasis so "does **not** authorize" still counts
            # as an explicit refusal.
            normalized = re.sub(r"[*_`]", "", line).lower()
            if any(phrase in normalized for phrase in _REFUSAL_PHRASES):
                continue
            if _FORBIDDEN_CLAIM.search(normalized):
                errors.append(f"{doc.relative_to(ROOT)}:{lineno}: forbidden live-trading claim")
    return errors


def main() -> int:
    errors: list[str] = []
    for checker in (
        check_scripts_exist,
        check_release_workflow,
        check_makefile_targets,
        check_observability_paths,
        check_no_live_trading_claims,
    ):
        errors.extend(checker())
    if errors:
        for err in errors:
            _fail(err)
        return 1
    print("OK: docs-vs-code release/deploy consistency")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
