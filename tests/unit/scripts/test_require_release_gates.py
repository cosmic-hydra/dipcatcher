"""Release signing must fail closed on a red SHA (CI or evidence-audit)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.require_release_gates import (
    CI_WORKFLOW_PATH,
    evaluate_release_gates,
    load_runs_from_github,
    load_runs_from_json,
    main,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "require_release_gates.py"
RED_SHA = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
GREEN_SHA = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _job(name: str, *, conclusion: str = "success", status: str = "completed") -> dict[str, Any]:
    return {"name": name, "conclusion": conclusion, "status": status}


def _run(
    *,
    sha: str,
    conclusion: str,
    jobs: list[dict[str, Any]],
    path: str = CI_WORKFLOW_PATH,
    status: str = "completed",
    created_at: str = "2026-10-19T00:00:00Z",
    run_id: int = 1,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "created_at": created_at,
        "path": path,
        "head_sha": sha,
        "status": status,
        "conclusion": conclusion,
        "jobs": jobs,
    }


def test_dry_run_red_sha_fails_release_gate() -> None:
    ok, reason = evaluate_release_gates(
        RED_SHA,
        [
            _run(
                sha=RED_SHA,
                conclusion="failure",
                jobs=[_job("evidence-audit", conclusion="failure")],
            )
        ],
    )
    assert ok is False
    assert RED_SHA in reason


def test_no_ci_run_fails_closed() -> None:
    ok, reason = evaluate_release_gates(RED_SHA, [])
    assert ok is False
    assert "no completed CI run" in reason


def test_green_ci_without_evidence_audit_fails_closed() -> None:
    ok, reason = evaluate_release_gates(
        GREEN_SHA,
        [_run(sha=GREEN_SHA, conclusion="success", jobs=[_job("lint")])],
    )
    assert ok is False
    assert "evidence-audit" in reason


def test_skipped_evidence_audit_fails_closed() -> None:
    ok, reason = evaluate_release_gates(
        GREEN_SHA,
        [
            _run(
                sha=GREEN_SHA,
                conclusion="success",
                jobs=[_job("evidence-audit", conclusion="skipped")],
            )
        ],
    )
    assert ok is False
    assert "evidence-audit" in reason


def test_other_workflow_success_does_not_sign() -> None:
    ok, _reason = evaluate_release_gates(
        GREEN_SHA,
        [
            _run(
                sha=GREEN_SHA,
                conclusion="success",
                path=".github/workflows/fx1.yml",
                jobs=[_job("evidence-audit")],
            )
        ],
    )
    assert ok is False


def test_green_ci_and_evidence_audit_is_signable() -> None:
    ok, reason = evaluate_release_gates(
        GREEN_SHA,
        [_run(sha=GREEN_SHA, conclusion="success", jobs=[_job("lint"), _job("evidence-audit")])],
    )
    assert ok is True
    assert GREEN_SHA in reason


def test_older_green_run_does_not_excuse_a_later_red_run() -> None:
    """The latest completed CI run is the verdict, not any historical success."""

    ok, reason = evaluate_release_gates(
        RED_SHA,
        [
            _run(
                sha=RED_SHA,
                conclusion="success",
                jobs=[_job("evidence-audit")],
                created_at="2026-10-18T00:00:00Z",
                run_id=1,
            ),
            _run(
                sha=RED_SHA,
                conclusion="failure",
                jobs=[_job("evidence-audit", conclusion="failure")],
                created_at="2026-10-20T00:00:00Z",
                run_id=2,
            ),
        ],
    )
    assert ok is False
    assert "latest completed CI" in reason


def test_invalid_sha_fails_closed() -> None:
    ok, reason = evaluate_release_gates("deadbeef", [])
    assert ok is False
    assert "invalid commit sha" in reason


def test_cli_dry_run_red_sha_exits_nonzero(tmp_path: Path) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps(
            [
                _run(
                    sha=RED_SHA,
                    conclusion="failure",
                    jobs=[_job("evidence-audit", conclusion="failure")],
                )
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--sha", RED_SHA, "--runs-json", str(fixture)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "FAIL" in proc.stderr


def test_cli_dry_run_green_sha_exits_zero(tmp_path: Path) -> None:
    fixture = tmp_path / "runs.json"
    fixture.write_text(
        json.dumps(
            {"runs": [_run(sha=GREEN_SHA, conclusion="success", jobs=[_job("evidence-audit")])]}
        ),
        encoding="utf-8",
    )
    assert main(["--sha", GREEN_SHA, "--runs-json", str(fixture)]) == 0


def test_load_runs_from_github_uses_same_predicate() -> None:
    def getter(
        url: str, token: str, params: dict[str, str] | None = None
    ) -> tuple[dict[str, Any], None]:
        assert token == "tok"
        if "workflows/ci.yml/runs" in url:
            assert params is not None and params["head_sha"] == GREEN_SHA
            return (
                {
                    "workflow_runs": [
                        {
                            "id": 9,
                            "path": CI_WORKFLOW_PATH,
                            "head_sha": GREEN_SHA,
                            "status": "completed",
                            "conclusion": "success",
                        }
                    ]
                },
                None,
            )
        assert "/jobs" in url
        return (
            {"jobs": [{"name": "evidence-audit", "status": "completed", "conclusion": "success"}]},
            None,
        )

    runs = load_runs_from_github("owner/repo", GREEN_SHA, "tok", getter=getter)
    ok, _reason = evaluate_release_gates(GREEN_SHA, load_runs_from_json(runs))
    assert ok is True
