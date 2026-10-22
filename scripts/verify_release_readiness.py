#!/usr/bin/env python3
"""Require tag/version/changelog consistency and same-commit CI evidence.

``.github/workflows/release.yml`` runs this on tag pushes before Sigstore
signing. ``require_release_gates.py`` additionally requires a green
``evidence-audit`` job on the latest completed ``ci.yml`` run.

Local dry-run (no network)::

    uv run python scripts/verify_release_readiness.py --local-only
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class ReleaseGateError(RuntimeError):
    """The tag does not have the required release evidence."""


# Workflow file → optional required job name → event filter.
# ``None`` job means every job on the latest matching run must succeed.
# ``matrix.yml`` uses the exact 36-job name set (no inventory job on current main).
_REQUIRED = (
    ("ci.yml", None, "push"),
    ("fx1.yml", None, "push"),
    ("proofcore.yml", None, "push"),
    ("matrix.yml", "__matrix__", None),
)
_PLATFORMS = ("ubuntu-latest", "macos-latest", "windows-latest")
_PYTHONS = ("3.12", "3.13", "3.14")
_MATRIX_JOBS = {
    f"{os_name} / py{python} / shard{shard}"
    for os_name in _PLATFORMS
    for python in _PYTHONS
    for shard in range(4)
}
_VERSION_SOURCE = Path("src/fx1/__init__.py")
_CHANGELOG = Path("CHANGELOG.md")
_TAG_REF = re.compile(
    r"refs/tags/v(?P<version>(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?)\Z"
)
# Keepachangelog heading: ## [0.4.0] - 2026-09-26  or  ## [0.4.0]
_CHANGELOG_HEADING = re.compile(
    r"^## \[(?P<version>[^\]\s]+)\](?:\s+-\s+\S+)?\s*$", re.MULTILINE
)


def _parse_tag_ref(ref: str) -> str:
    match = _TAG_REF.fullmatch(ref)
    if match is None:
        raise ReleaseGateError(f"release ref is not a SemVer tag: {ref!r}")
    tag_version = match.group("version")
    prerelease = tag_version.split("+", 1)[0].split("-", 1)
    if len(prerelease) == 2 and any(
        part.isascii() and part.isdigit() and len(part) > 1 and part[0] == "0"
        for part in prerelease[1].split(".")
    ):
        raise ReleaseGateError(f"release tag has a noncanonical numeric prerelease: {ref!r}")
    return tag_version


def read_package_version(root: Path) -> str:
    """Return the literal ``fx1.__version__`` string from the Hatch source."""

    try:
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ReleaseGateError(f"cannot read Hatch version configuration: {exc}") from exc
    project = config.get("project", {})
    if (
        "version" not in project.get("dynamic", [])
        or "version" in project
        or config.get("tool", {}).get("hatch", {}).get("version", {}).get("path")
        != _VERSION_SOURCE.as_posix()
    ):
        raise ReleaseGateError("pyproject must derive its version from fx1.__version__")
    try:
        source = ast.parse((root / _VERSION_SOURCE).read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        raise ReleaseGateError(f"cannot read fx1 version source: {exc}") from exc
    assignments = [
        node
        for node in ast.walk(source)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__version__"
                for target in node.targets
            )
        )
        or (
            isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr))
            and isinstance(node.target, ast.Name)
            and node.target.id == "__version__"
        )
    ]
    if (
        len(assignments) != 1
        or assignments[0] not in source.body
        or not isinstance(assignments[0], ast.Assign)
        or len(assignments[0].targets) != 1
        or not isinstance(assignments[0].targets[0], ast.Name)
        or not isinstance(assignments[0].value, ast.Constant)
    ):
        raise ReleaseGateError("fx1.__version__ must be one literal string assignment")
    package_version = assignments[0].value.value
    if not isinstance(package_version, str):
        raise ReleaseGateError("fx1.__version__ must be a string literal")
    return package_version


def verify_tag_version(ref: str, root: Path) -> str:
    """Match an exact SemVer tag to Hatch's literal fx1 version source."""

    tag_version = _parse_tag_ref(ref)
    package_version = read_package_version(root)
    if package_version != tag_version:
        raise ReleaseGateError(
            f"release tag version {tag_version!r} does not match "
            f"fx1.__version__ {package_version!r}"
        )
    return package_version


def verify_changelog(version: str, root: Path) -> None:
    """Require a Keep-a-Changelog heading for the release version.

    Full releases (no ``-`` prerelease segment) must have ``## [X.Y.Z]``.
    Prereleases may ship under ``## [Unreleased]`` only when that section is
    non-empty; a matching ``## [X.Y.Z-rc…]`` heading also satisfies the gate.
    """

    path = root / _CHANGELOG
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReleaseGateError(f"cannot read {_CHANGELOG}: {exc}") from exc
    headings = {match.group("version"): match for match in _CHANGELOG_HEADING.finditer(text)}
    base = version.split("+", 1)[0]
    is_prerelease = "-" in base
    if base in headings:
        return
    if not is_prerelease:
        raise ReleaseGateError(
            f"CHANGELOG.md missing heading ## [{base}] for release version {version!r}"
        )
    unreleased = headings.get("Unreleased")
    if unreleased is None:
        raise ReleaseGateError(
            f"CHANGELOG.md missing ## [{base}] and has no ## [Unreleased] "
            f"for prerelease {version!r}"
        )
    # Non-empty Unreleased body: content between this heading and the next ##.
    start = unreleased.end()
    next_heading = _CHANGELOG_HEADING.search(text, start)
    body = text[start : next_heading.start() if next_heading else len(text)].strip()
    if not body:
        raise ReleaseGateError(
            f"CHANGELOG.md ## [Unreleased] is empty; cannot cover prerelease {version!r}"
        )


def verify_local_release_metadata(root: Path, *, ref: str | None = None) -> str:
    """Version + changelog checks used by both tag preflight and dry-run."""

    package_version = read_package_version(root)
    if ref:
        tagged = verify_tag_version(ref, root)
        assert tagged == package_version
        verify_changelog(package_version, root)
    else:
        # Dry-run: package version must already have a changelog home
        # (released section or non-empty Unreleased for an upcoming cut).
        verify_changelog(package_version, root)
    return package_version


_API_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+\Z")


def _repo_parts(repo: str) -> tuple[str, str]:
    parts = repo.split("/")
    if len(parts) != 2 or any(
        part in {".", ".."} or _API_SEGMENT.fullmatch(part) is None for part in parts
    ):
        raise ReleaseGateError("GITHUB_REPOSITORY must be owner/repo")
    return parts[0], parts[1]


def _api(repo: str, path: str, token: str) -> dict[str, Any]:
    owner, name = _repo_parts(repo)
    route, _separator, query = path.partition("?")
    route_parts = route.split("/")
    if (
        any(part in {"", ".", ".."} or _API_SEGMENT.fullmatch(part) is None for part in route_parts)
        or "#" in query
        or any(ord(char) < 32 for char in query)
    ):
        raise ReleaseGateError("GitHub API path must be repository-relative")
    encoded_path = "/repos/" + "/".join(
        urllib.parse.quote(part, safe="") for part in (owner, name, *route_parts)
    )
    url = urllib.parse.urlunsplit(("https", "api.github.com", encoded_path, query, ""))
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except (OSError, ValueError) as exc:
        raise ReleaseGateError(f"GitHub API request failed for {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ReleaseGateError(f"GitHub API returned a non-object for {path}")
    return result


def _complete_list(data: dict[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    rows = data.get(key)
    if not isinstance(rows, list) or not isinstance(data.get("total_count"), int):
        raise ReleaseGateError(f"{label}: malformed GitHub API response")
    if data["total_count"] > len(rows):
        raise ReleaseGateError(f"{label}: GitHub API response was paginated")
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseGateError(f"{label}: malformed list entry")
    return rows


def _workflow_run(
    repo: str, sha: str, token: str, workflow: str, event: str | None
) -> dict[str, Any]:
    params = {"head_sha": sha, "per_page": 100}
    if event:
        params["event"] = event
        params["branch"] = "main"
    query = urllib.parse.urlencode(params)
    data = _api(repo, f"actions/workflows/{workflow}/runs?{query}", token)
    runs = _complete_list(data, "workflow_runs", workflow)
    candidates = [
        run
        for run in runs
        if run.get("head_sha") == sha
        and run.get("event") in ({event} if event else {"schedule", "workflow_dispatch", "push"})
        and (event is None or run.get("head_branch") == "main")
    ]
    if not candidates:
        raise ReleaseGateError(f"{workflow}: no same-SHA run with the required event")
    latest = max(
        candidates,
        key=lambda run: (
            str(run.get("updated_at") or run.get("created_at") or ""),
            int(run.get("id", 0)),
        ),
    )
    if latest.get("status") != "completed" or latest.get("conclusion") != "success":
        raise ReleaseGateError(
            f"{workflow}: latest same-SHA run {latest.get('id')} is "
            f"{latest.get('status')}/{latest.get('conclusion')}"
        )
    return latest


def _jobs(repo: str, token: str, workflow: str, run_id: int) -> list[dict[str, Any]]:
    data = _api(repo, f"actions/runs/{run_id}/jobs?per_page=100&filter=latest", token)
    return _complete_list(data, "jobs", f"{workflow} jobs")


def verify_release(repo: str, sha: str, token: str) -> list[str]:
    """Return audit references if the tag has all required same-SHA evidence."""

    _repo_parts(repo)
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ReleaseGateError("GITHUB_SHA must be a commit SHA")
    if not token:
        raise ReleaseGateError("GITHUB_TOKEN is required for release evidence checks")

    ancestry = _api(repo, f"compare/{sha}...main", token)
    if ancestry.get("status") not in {"ahead", "identical"}:
        raise ReleaseGateError(f"tag commit {sha} is not an ancestor of main")

    evidence = []
    for workflow, required_job, event in _REQUIRED:
        run = _workflow_run(repo, sha, token, workflow, event)
        run_id = run.get("id")
        if not isinstance(run_id, int):
            raise ReleaseGateError(f"{workflow}: run has no numeric ID")
        jobs = _jobs(repo, token, workflow, run_id)
        if required_job == "__matrix__":
            names = {job.get("name") for job in jobs}
            if names != _MATRIX_JOBS or len(jobs) != len(_MATRIX_JOBS):
                raise ReleaseGateError(
                    f"{workflow}: run {run_id} lacks the complete {len(_MATRIX_JOBS)}-job matrix"
                )
            if any(job.get("conclusion") != "success" for job in jobs):
                raise ReleaseGateError(f"{workflow}: run {run_id} has a non-success matrix job")
        elif required_job:
            gate = [job for job in jobs if job.get("name") == required_job]
            if len(gate) != 1 or gate[0].get("conclusion") != "success":
                raise ReleaseGateError(f"{workflow}: {required_job} did not pass in run {run_id}")
        else:
            if not jobs:
                raise ReleaseGateError(f"{workflow}: run {run_id} has no jobs")
            if any(job.get("conclusion") != "success" for job in jobs):
                raise ReleaseGateError(f"{workflow}: run {run_id} has a non-success job")
        evidence.append(f"{workflow}: {run.get('html_url') or run_id}")
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Check fx1.__version__ and CHANGELOG only (no GitHub API)",
    )
    parser.add_argument(
        "--ref",
        default="",
        help="Override GITHUB_REF (refs/tags/v…). Ignored with --local-only "
        "unless provided explicitly for changelog matching.",
    )
    # argv=None → sys.argv (CLI). Tests pass an explicit list (often empty).
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        if args.local_only:
            ref = args.ref or None
            package_version = verify_local_release_metadata(root, ref=ref)
            print(f"release local preflight: fx1.__version__={package_version}; changelog ok")
            return 0
        ref = args.ref or os.environ.get("GITHUB_REF", "")
        package_version = verify_local_release_metadata(root, ref=ref)
        evidence = verify_release(
            os.environ.get("GITHUB_REPOSITORY", ""),
            os.environ.get("GITHUB_SHA", ""),
            os.environ.get("GITHUB_TOKEN", ""),
        )
    except ReleaseGateError as exc:
        print(f"release preflight failed: {exc}")
        return 1
    for row in evidence:
        print(row)
    print(f"release preflight: v{package_version} matches fx1; same-SHA CI and matrix passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
