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


def test_fast_gates_cancel_only_superseded_pull_requests() -> None:
    """Main/schedule must drain; pull_request may cancel superseded runs."""

    for name in ("ci.yml", "fx1.yml", "proofcore.yml"):
        text = _workflow(name)
        assert "cancel-in-progress:" in text, name
        assert "${{ github.event_name == 'pull_request' }}" in text, name


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
    for path in ("src", "tests", "scripts", "web", "replay", "clients/typescript"):
        assert f"  - {path}\n" in config
