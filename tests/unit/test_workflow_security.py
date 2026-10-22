"""Regression guards for fail-closed, least-privilege GitHub workflows."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_ci_does_not_grant_workflow_wide_actions_write() -> None:
    """Caches and artifacts must not widen the GITHUB_TOKEN permission set."""

    workflow = _workflow("ci.yml")

    top_level = workflow.split("\njobs:\n", maxsplit=1)[0]
    assert "actions: write" not in top_level


def test_dependency_review_fails_closed() -> None:
    """Unavailable dependency metadata must not produce a green review."""

    workflow = _workflow("dependency-review.yml")

    assert "actions/dependency-review-action@" in workflow
    assert "continue-on-error:" not in workflow
    assert "steps.review.outcome" not in workflow
    assert 'if [ "$code" = "404" ]' not in workflow


def test_codeql_covers_supported_source_languages() -> None:
    """Python and the repository's JavaScript/TypeScript must be analyzed."""

    workflow = _workflow("codeql.yml")
    config = (REPO_ROOT / ".github" / "codeql" / "codeql-config.yml").read_text(encoding="utf-8")

    assert "language: [python, javascript-typescript]" in workflow
    assert "languages: ${{ matrix.language }}" in workflow
    assert "build-mode: none" in workflow
    for path in ("src", "tests", "scripts", "web", "replay", "clients/typescript"):
        assert f"  - {path}\n" in config


def test_secret_scan_pins_gitleaks_and_scans_full_history() -> None:
    """Secret scan must stay sha256-pinned and history-scoped to HEAD."""

    workflow = _workflow("secret-scan.yml")

    assert 'version="8.30.1"' in workflow
    assert "gitleaks_${version}_linux_x64.tar.gz" in workflow
    assert "551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb" in workflow
    assert "--log-opts=--full-history" in workflow
    assert "continue-on-error:" not in workflow


def test_dependabot_covers_all_npm_audit_trees() -> None:
    """Dependabot must track the same four npm trees `make audit-js` audits."""

    text = (REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    for directory in ("/web", "/replay", "/clients/typescript", "/clients/typescript/fx1"):
        assert f"- {directory}" in text
