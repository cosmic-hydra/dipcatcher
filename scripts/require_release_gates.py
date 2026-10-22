#!/usr/bin/env python3
"""Fail closed unless a SHA has a green CI run with a green evidence-audit job.

``.github/workflows/release.yml`` calls this before Sigstore signing. A tag on
a red SHA must not be signed. Unit tests evaluate the same predicate against
fixture runs so the gate is proven without firing GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
EVIDENCE_AUDIT_JOB = "evidence-audit"
COMPLETED = "completed"
SUCCESS = "success"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
API_VERSION = "2022-11-28"
USER_AGENT = "dipcatcher-require-release-gates"
MAX_PAGES = 10
PER_PAGE = 100


class ReleaseGateError(Exception):
    """Raised when the SHA must not be signed."""


def normalize_sha(sha: str) -> str:
    value = sha.strip().lower()
    if not SHA_RE.fullmatch(value):
        raise ReleaseGateError(f"invalid commit sha {sha!r}")
    return value


def _is_evidence_audit_job(job: Mapping[str, Any]) -> bool:
    name = str(job.get("name") or "")
    return name == EVIDENCE_AUDIT_JOB or name.startswith(EVIDENCE_AUDIT_JOB + " ")


def _is_ci_run(run: Mapping[str, Any], sha: str) -> bool:
    path = str(run.get("path") or "")
    head = str(run.get("head_sha") or "").lower()
    status = str(run.get("status") or "")
    return path == CI_WORKFLOW_PATH and head == sha and status == COMPLETED


def _job_succeeded(job: Mapping[str, Any]) -> bool:
    return str(job.get("status") or "") == COMPLETED and str(job.get("conclusion") or "") == SUCCESS


def _run_recency(run: Mapping[str, Any]) -> tuple[str, int]:
    """Newest completed run wins. Missing timestamps sort as oldest."""

    created = str(run.get("created_at") or "")
    try:
        run_id = int(run.get("id") or 0)
    except (TypeError, ValueError):
        run_id = 0
    return created, run_id


def evaluate_release_gates(sha: str, runs: Sequence[Mapping[str, Any]]) -> tuple[bool, str]:
    """The latest completed CI run is the verdict.

    An older green run does not authorize a tag whose newest completed CI
    run failed, was cancelled, or lacks a successful evidence-audit job.
    """

    try:
        digest = normalize_sha(sha)
    except ReleaseGateError as exc:
        return False, str(exc)
    ci_runs = [run for run in runs if _is_ci_run(run, digest)]
    if not ci_runs:
        return False, f"no completed CI run for {digest}"
    latest = max(ci_runs, key=_run_recency)
    run_id = latest.get("id")
    label = f"run {run_id}" if run_id is not None else "latest CI run"
    conclusion = str(latest.get("conclusion") or "")
    if conclusion != SUCCESS:
        return False, (
            f"tagged commit {digest} is not signable: "
            f"latest completed CI {label} conclusion={conclusion or 'none'}"
        )
    jobs = list(latest.get("jobs") or [])
    audit = next((job for job in jobs if _is_evidence_audit_job(job)), None)
    if audit is None:
        return False, (
            f"tagged commit {digest} is not signable: "
            f"latest completed CI {label} has no evidence-audit job"
        )
    if not _job_succeeded(audit):
        return False, (
            f"tagged commit {digest} is not signable: evidence-audit on {label} "
            f"conclusion={audit.get('conclusion') or 'none'} "
            f"status={audit.get('status') or 'none'}"
        )
    return True, f"CI and evidence-audit succeeded on {digest} ({label})"


def _link_next(headers: Mapping[str, str]) -> str | None:
    link = headers.get("Link") or headers.get("link")
    if not link:
        return None
    for part in link.split(","):
        if 'rel="next"' not in part:
            continue
        start = part.find("<")
        end = part.find(">", start + 1)
        if start >= 0 and end > start:
            return part[start + 1 : end]
    return None


def github_get_json(
    url: str, token: str, params: Mapping[str, str] | None = None
) -> tuple[Any, str | None]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": USER_AGENT,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            headers = {k: v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        body = exc.read()[:300]
        raise ReleaseGateError(f"GitHub API {exc.code} for {url}: {body!r}") from exc
    except urllib.error.URLError as exc:
        raise ReleaseGateError(f"GitHub API unreachable: {exc.reason}") from exc
    return payload, _link_next(headers)


def _paginate(
    url: str, token: str, params: Mapping[str, str] | None, item_key: str, getter: Any
) -> list[Any]:
    items: list[Any] = []
    next_url: str | None = url
    query: Mapping[str, str] | None = params
    for _ in range(MAX_PAGES):
        if next_url is None:
            break
        payload, next_url = getter(next_url, token, query)
        query = None
        if not isinstance(payload, dict):
            raise ReleaseGateError(f"GitHub API returned a non-object payload for {item_key}")
        chunk = payload.get(item_key)
        if not isinstance(chunk, list):
            raise ReleaseGateError(f"GitHub API missing {item_key!r} list")
        items.extend(chunk)
        if len(chunk) < PER_PAGE:
            break
    return items


def load_runs_from_github(
    repo: str, sha: str, token: str, *, getter: Any = github_get_json
) -> list[dict[str, Any]]:
    digest = normalize_sha(sha)
    owner_repo = repo.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", owner_repo):
        raise ReleaseGateError(f"invalid repository {repo!r}")
    base = f"https://api.github.com/repos/{owner_repo}"
    raw_runs = _paginate(
        f"{base}/actions/workflows/ci.yml/runs",
        token,
        {"head_sha": digest, "status": COMPLETED, "per_page": str(PER_PAGE)},
        "workflow_runs",
        getter,
    )
    loaded: list[dict[str, Any]] = []
    for raw in raw_runs:
        run_id = raw.get("id")
        jobs: list[Any] = []
        if run_id is not None:
            jobs = _paginate(
                f"{base}/actions/runs/{run_id}/jobs",
                token,
                {"per_page": str(PER_PAGE)},
                "jobs",
                getter,
            )
        loaded.append(
            {
                "path": raw.get("path") or CI_WORKFLOW_PATH,
                "head_sha": raw.get("head_sha") or digest,
                "status": raw.get("status"),
                "conclusion": raw.get("conclusion"),
                "jobs": jobs,
            }
        )
    return loaded


def load_runs_from_json(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "runs" in payload:
        payload = payload["runs"]
    if not isinstance(payload, list):
        raise ReleaseGateError("runs JSON must be a list or an object with a runs list")
    return [dict(item) for item in payload]


def _token_from_env() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    if not token.strip():
        raise ReleaseGateError("GH_TOKEN or GITHUB_TOKEN is required to query GitHub")
    return token.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--repo", default="")
    parser.add_argument("--runs-json", default="")
    args = parser.parse_args(argv)
    try:
        if args.runs_json and args.repo:
            raise ReleaseGateError("pass only one of --repo or --runs-json")
        if args.runs_json:
            raw = args.runs_json
            if raw.lstrip().startswith(("[", "{")):
                payload = json.loads(raw)
            else:
                payload = json.loads(Path(raw).read_text(encoding="utf-8"))
            runs = load_runs_from_json(payload)
        elif args.repo:
            runs = load_runs_from_github(args.repo, args.sha, _token_from_env())
        else:
            raise ReleaseGateError("pass --repo (live) or --runs-json (dry-run)")
        ok, reason = evaluate_release_gates(args.sha, runs)
        if not ok:
            print(f"FAIL: {reason}", file=sys.stderr)
            return 1
        print(f"OK: {reason}")
        return 0
    except ReleaseGateError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"FAIL: invalid runs JSON: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
