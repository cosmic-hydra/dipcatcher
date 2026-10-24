"""A release tag must carry successful evidence from its exact commit."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs
from urllib.request import Request

import pytest
import yaml
from scripts import verify_release_readiness as gate

_ROOT = Path(__file__).resolve().parents[3]
_SHA = "a" * 40
_REPO = "artificial-hedge/dipcatcher"


def _version_fixture(root: Path, version: str = "0.4.0") -> None:
    (root / "src" / "fx1").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[project]\nname = "fx-1"\ndynamic = ["version"]\n'
        '[tool.hatch.version]\npath = "src/fx1/__init__.py"\n',
        encoding="utf-8",
    )
    (root / "src" / "fx1" / "__init__.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (root / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [Unreleased]\n\n- pending\n\n## [{version.split('-', 1)[0]}] - 2026-01-01\n\n- shipped\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("version", "tag"),
    [("0.4.0", "refs/tags/v0.4.0"), ("1.2.3-rc1", "refs/tags/v1.2.3-rc1")],
)
def test_tag_matches_hatch_version_source(tmp_path: Path, version: str, tag: str) -> None:
    _version_fixture(tmp_path, version)
    assert gate.verify_tag_version(tag, tmp_path) == version


@pytest.mark.parametrize(
    "tag",
    [
        "refs/heads/main",
        "refs/tags/0.4.0",
        "refs/tags/v0.4.1",
        "refs/tags/v00.4.0",
        "refs/tags/v0.4.0-rc.01",
        "refs/tags/v0.4.0-rc1/extra",
        "refs/tags/v0.4.0 ",
    ],
)
def test_tag_rejects_wrong_version_or_malformed_semver(tmp_path: Path, tag: str) -> None:
    _version_fixture(tmp_path)
    with pytest.raises(gate.ReleaseGateError):
        gate.verify_tag_version(tag, tmp_path)


def test_tag_rejects_nonliteral_or_changed_hatch_source(tmp_path: Path) -> None:
    _version_fixture(tmp_path)
    source = tmp_path / "src" / "fx1" / "__init__.py"
    source.write_text('__version__ = "0.4." + "0"\n', encoding="utf-8")
    with pytest.raises(gate.ReleaseGateError, match="literal string"):
        gate.verify_tag_version("refs/tags/v0.4.0", tmp_path)
    source.write_text('__version__ = "0.4.0"\n__version__: str = "0.4.1"\n', encoding="utf-8")
    with pytest.raises(gate.ReleaseGateError, match="literal string"):
        gate.verify_tag_version("refs/tags/v0.4.0", tmp_path)
    source.write_text('__version__ = "0.4.0"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "fx-1"\ndynamic = ["version"]\n'
        '[tool.hatch.version]\npath = "src/other.py"\n',
        encoding="utf-8",
    )
    with pytest.raises(gate.ReleaseGateError, match="derive its version"):
        gate.verify_tag_version("refs/tags/v0.4.0", tmp_path)


def test_changelog_requires_version_heading(tmp_path: Path) -> None:
    _version_fixture(tmp_path, "0.4.0")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n\n", encoding="utf-8")
    with pytest.raises(gate.ReleaseGateError, match="missing heading"):
        gate.verify_changelog("0.4.0", tmp_path)


def test_changelog_prerelease_accepts_nonempty_unreleased(tmp_path: Path) -> None:
    _version_fixture(tmp_path, "0.4.1-rc1")
    (tmp_path / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [Unreleased]\n\n- rc work\n\n## [0.4.0] - 2026-01-01\n\n- prior\n",
        encoding="utf-8",
    )
    gate.verify_changelog("0.4.1-rc1", tmp_path)


def test_local_only_main_checks_package_and_changelog(tmp_path: Path) -> None:
    _version_fixture(tmp_path, "0.4.0")
    assert gate.verify_local_release_metadata(tmp_path) == "0.4.0"


def test_main_checks_tag_before_any_api_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_REF", "refs/tags/v0.4.1")
    monkeypatch.setenv("GITHUB_REPOSITORY", _REPO)
    monkeypatch.setenv("GITHUB_SHA", _SHA)
    monkeypatch.setenv("GITHUB_TOKEN", "token")
    monkeypatch.setattr(
        gate,
        "verify_release",
        lambda *_args: pytest.fail("remote evidence checked before tag version"),
    )
    assert gate.main([]) == 1


@pytest.mark.parametrize("repo", ["../dipcatcher", "owner/..", "./repo", "owner/."])
def test_api_rejects_repository_dot_segments_before_request(
    monkeypatch: pytest.MonkeyPatch, repo: str
) -> None:
    monkeypatch.setattr(
        gate.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid repository reached the network"),
    )
    with pytest.raises(gate.ReleaseGateError, match="GITHUB_REPOSITORY"):
        gate._api(repo, f"compare/{_SHA}...main", "token")


@pytest.mark.parametrize(
    "path",
    [
        "../compare",
        "actions/../runs",
        "/actions/runs",
        "actions/%2e%2e/runs",
        "actions/runs?x=1#frag",
    ],
)
def test_api_rejects_traversal_in_route_before_request(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setattr(
        gate.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid route reached the network"),
    )
    with pytest.raises(gate.ReleaseGateError, match="repository-relative"):
        gate._api(_REPO, path, "token")


def test_api_calls_only_fixed_github_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    urls: list[str] = []

    def fake_urlopen(request: Request, *, timeout: int) -> BytesIO:
        assert timeout == 30
        urls.append(request.full_url)
        return BytesIO(b'{"status":"ahead"}')

    monkeypatch.setattr(gate.urllib.request, "urlopen", fake_urlopen)
    assert gate._api(_REPO, f"compare/{_SHA}...main", "token") == {"status": "ahead"}
    assert urls == [f"https://api.github.com/repos/{_REPO}/compare/{_SHA}...main"]


def _matrix_job_names() -> set[str]:
    config = yaml.load(
        (_ROOT / ".github" / "workflows" / "matrix.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    matrix = config["jobs"]["test"]["strategy"]["matrix"]
    return {
        f"{os_name} / py{python} / shard{shard}"
        for os_name in matrix["os"]
        for python in matrix["python"]
        for shard in matrix["shard"]
    }


def _responses() -> dict[str, dict]:
    responses: dict[str, dict] = {f"compare/{_SHA}...main": {"status": "ahead"}}
    for index, (workflow, required_job, event) in enumerate(gate._REQUIRED, start=1):
        run_event = event or "workflow_dispatch"
        responses[f"actions/workflows/{workflow}/runs"] = {
            "total_count": 1,
            "workflow_runs": [
                {
                    "id": index,
                    "head_sha": _SHA,
                    "event": run_event,
                    "head_branch": "main",
                    "status": "completed",
                    "conclusion": "success",
                    "created_at": "2026-10-16T00:00:00Z",
                    "updated_at": "2026-10-16T00:01:00Z",
                    "html_url": f"https://github.com/{_REPO}/actions/runs/{index}",
                }
            ],
        }
        if required_job == "__matrix__":
            names = sorted(_matrix_job_names())
        else:
            names = ["lint", "evidence-audit", "test"]
        responses[f"actions/runs/{index}/jobs"] = {
            "total_count": len(names),
            "jobs": [{"name": name, "conclusion": "success"} for name in names],
        }
    return responses


def _verify(monkeypatch: pytest.MonkeyPatch, responses: dict[str, dict]) -> list[str]:
    def fake_api(repo: str, path: str, token: str) -> dict:
        assert repo == _REPO
        assert token == "token"
        prefix = path.split("?", 1)[0]
        if prefix.startswith("actions/workflows/"):
            query = parse_qs(path.split("?", 1)[1])
            assert query["head_sha"] == [_SHA]
            if not prefix.endswith("matrix.yml/runs"):
                assert query["event"] == ["push"]
                assert query["branch"] == ["main"]
        return responses[prefix]

    monkeypatch.setattr(gate, "_api", fake_api)
    return gate.verify_release(_REPO, _SHA, "token")


def test_release_preflight_requires_complete_same_sha_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _verify(monkeypatch, _responses())
    assert len(evidence) == 4
    assert evidence == [
        f"{workflow}: https://github.com/{_REPO}/actions/runs/{index}"
        for index, (workflow, _job, _event) in enumerate(gate._REQUIRED, start=1)
    ]
    assert _matrix_job_names() == gate._MATRIX_JOBS


def test_release_preflight_uses_latest_execution_including_reruns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = _responses()
    ci_runs = responses["actions/workflows/ci.yml/runs"]
    ci_runs["total_count"] = 2
    ci_runs["workflow_runs"].append(
        {
            **ci_runs["workflow_runs"][0],
            "id": 101,
            "created_at": "2026-10-16T00:02:00Z",
            "updated_at": "2026-10-16T00:03:00Z",
            "conclusion": "failure",
        }
    )
    ci_runs["workflow_runs"][0]["updated_at"] = "2026-10-16T00:04:00Z"
    assert len(_verify(monkeypatch, responses)) == 4


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("diverged", "not an ancestor"),
        ("wrong_sha", "no same-SHA run"),
        ("wrong_branch", "no same-SHA run"),
        ("red_run", "failure"),
        ("red_job", "non-success job"),
        ("missing_matrix_job", "lacks the complete"),
        ("skipped_matrix_job", "non-success matrix job"),
        ("paginated", "response was paginated"),
    ],
)
def test_release_preflight_rejects_missing_or_red_evidence(
    monkeypatch: pytest.MonkeyPatch, change: str, message: str
) -> None:
    responses = _responses()
    ci_run = responses["actions/workflows/ci.yml/runs"]["workflow_runs"][0]
    if change == "diverged":
        responses[f"compare/{_SHA}...main"]["status"] = "diverged"
    elif change == "wrong_sha":
        ci_run["head_sha"] = "b" * 40
    elif change == "wrong_branch":
        ci_run["head_branch"] = "fx-1/other"
    elif change == "red_run":
        ci_run["conclusion"] = "failure"
    elif change == "red_job":
        responses["actions/runs/1/jobs"]["jobs"][0]["conclusion"] = "failure"
    elif change == "missing_matrix_job":
        matrix = responses["actions/runs/4/jobs"]
        matrix["jobs"].pop()
        matrix["total_count"] -= 1
    elif change == "skipped_matrix_job":
        responses["actions/runs/4/jobs"]["jobs"][0]["conclusion"] = "skipped"
    elif change == "paginated":
        responses["actions/workflows/ci.yml/runs"]["total_count"] = 2
    with pytest.raises(gate.ReleaseGateError, match=message):
        _verify(monkeypatch, responses)


def test_release_workflow_places_read_only_preflight_before_signing() -> None:
    config = yaml.load(
        (_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    preflight = config["jobs"]["preflight"]
    assert preflight["permissions"] == {"contents": "read", "actions": "read"}
    runs = "\n".join(
        step.get("run", "") for step in preflight["steps"] if isinstance(step, dict)
    )
    assert "scripts/verify_release_readiness.py" in runs
    assert "scripts/require_release_gates.py" in runs
    assert config["jobs"]["build"]["needs"] == "preflight"
    assert preflight["name"] == "Tag version and same-commit evidence"
    assert "Build" in config["jobs"]["build"]["name"]
    assert _matrix_job_names() == gate._MATRIX_JOBS
