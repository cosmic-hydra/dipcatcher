"""W5 CI-helper tests (DESIGN.md §12 W5 row).

- coverage floor config parses and matches the pyproject single source of truth
- receipts re-verification loop: fail-closed on bit-rot, empty dir, exceptions
- proofcore.yml (staged at docs/proofcore/) is valid YAML, declares the §9.3
  gate matrix, and every `make <target>` it invokes exists in the Makefile
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from quant_fund.proofcore.ci import (
    REQUIRED_FLOOR_PACKAGES,
    _cli_verifier,
    coverage_floors,
    coverage_gate,
    receipt_paths,
    reverify_receipts,
)
from quant_fund.proofcore.contracts import ProofcoreError

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
# Canonical staged copy (always tracked). The activated path
# (.github/workflows/proofcore.yml) is installed by a workflow-scoped token.
WORKFLOW_STAGED = REPO_ROOT / "docs" / "proofcore" / "proofcore.yml"
WORKFLOW_ACTIVATED = REPO_ROOT / ".github" / "workflows" / "proofcore.yml"
MAKEFILE = REPO_ROOT / "Makefile"

# §9.3/§12 (lead-adjudicated): pit/proof/reality/proofcore 90, leakage 85.
# Main raised the package floors +1 (2026-09-28 measurements); the wave-2
# amendment adds per-MODULE floors (dotted keys) for the new wave-2 modules,
# all 90 (WAVE2.md §1.6: every new module >= 90%).
# 2026-10-24: proof, proofcore, leakage, and their wave-2 module floors
# ratcheted to the measured 100% line+branch. Pit and reality are unchanged.
EXPECTED_FLOORS = {
    "pit": 91,
    "proof": 100,
    "leakage": 100,
    "reality": 91,
    "proofcore": 100,
    "proofcore.scheduler": 100,
    "proof.runner": 100,
    "proof.estimators": 100,
    "proof.replay": 100,
    "leakage.guard": 100,
}


# ---------------------------------------------------------------------------
# Coverage floor config (A3 #2)
# ---------------------------------------------------------------------------


def test_coverage_floors_parse_from_pyproject() -> None:
    floors = coverage_floors(PYPROJECT)
    assert floors == EXPECTED_FLOORS
    assert set(REQUIRED_FLOOR_PACKAGES) <= set(floors)


def test_coverage_floors_fail_closed_on_missing_table(tmp_path: Path) -> None:
    fake = tmp_path / "pyproject.toml"
    fake.write_text("[project]\nname = 'x'\n")
    with pytest.raises(ProofcoreError, match="coverage-floors"):
        coverage_floors(fake)


def test_coverage_floors_reject_floor_below_global_ratchet(tmp_path: Path) -> None:
    fake = tmp_path / "pyproject.toml"
    rows = "\n".join(
        f"{json.dumps(pkg) if '.' in pkg else pkg} = {70 if pkg == 'leakage' else floor}"
        for pkg, floor in EXPECTED_FLOORS.items()
    )
    fake.write_text(f"[tool.proofcore.coverage-floors]\n{rows}\n")
    with pytest.raises(ProofcoreError, match="raise-never-lower"):
        coverage_floors(fake)


def test_coverage_gate_reports_failures_via_runner() -> None:
    """The gate shells out per floor entry and collects every failure."""
    calls: list[list[str]] = []

    import subprocess

    def fake_runner(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        rc = 1 if "leakage/*" in cmd[4] else 0
        return subprocess.CompletedProcess(cmd, rc, stdout="TOTAL 42%", stderr="")

    failures = coverage_gate(PYPROJECT, runner=fake_runner)
    assert len(calls) == len(EXPECTED_FLOORS)
    assert len(failures) == 1 and "leakage" in failures[0]
    for cmd in calls:
        include = cmd[4]
        floor = int(cmd[5].split("=")[1])
        name = include.removeprefix("--include=src/quant_fund/")
        # Package entries glob the tree; dotted module entries name the file.
        name = name.removesuffix("/*").replace("/", ".").removesuffix(".py")
        assert floor == EXPECTED_FLOORS[name]


def test_coverage_floors_reject_bad_module_entry(tmp_path: Path) -> None:
    """Wave-2 module floors are validated too: non-int or sub-80 fails closed."""
    fake = tmp_path / "pyproject.toml"
    rows = "\n".join(
        f"{json.dumps(pkg) if '.' in pkg else pkg} = {floor}"
        for pkg, floor in EXPECTED_FLOORS.items()
    )
    fake.write_text(f"[tool.proofcore.coverage-floors]\n{rows}\n'replay.broken' = 75\n")
    with pytest.raises(ProofcoreError, match="raise-never-lower"):
        coverage_floors(fake)


def test_module_floor_include_pattern() -> None:
    """Dotted floor keys map to exactly one module file, packages to a tree."""
    from quant_fund.proofcore import ci

    assert ci._include_pattern("src/quant_fund", "pit") == "src/quant_fund/pit/*"
    assert ci._include_pattern("src/quant_fund", "proof.runner") == "src/quant_fund/proof/runner.py"


# ---------------------------------------------------------------------------
# Receipts re-verification loop (A3 #4)
# ---------------------------------------------------------------------------


def _write_receipt(path: Path, payload: dict[str, object]) -> None:
    body = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(body.encode()).hexdigest()
    path.write_text(json.dumps({"payload": payload, "sha256": digest}))


def _checksum_verifier(path: Path) -> bool:
    data = json.loads(path.read_text())
    body = json.dumps(data["payload"], sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest() == data["sha256"]


def test_receipt_paths_sorted(tmp_path: Path) -> None:
    for name in ("b.json", "a.json", "c.json"):
        _write_receipt(tmp_path / name, {"name": name})
    (tmp_path / "notes.txt").write_text("not a receipt")
    assert [p.name for p in receipt_paths(tmp_path)] == ["a.json", "b.json", "c.json"]


def test_reverify_all_clean(tmp_path: Path) -> None:
    for i in range(3):
        _write_receipt(tmp_path / f"r{i}.json", {"i": i})
    assert reverify_receipts(tmp_path, verify=_checksum_verifier) == []


def test_reverify_bit_rot_detected(tmp_path: Path) -> None:
    """§12 W5 adversarial: flip one byte in one committed receipt -> failure."""
    for i in range(3):
        _write_receipt(tmp_path / f"r{i}.json", {"i": i})
    victim = tmp_path / "r1.json"
    data = json.loads(victim.read_text())
    data["sha256"] = "0" * 64  # one corrupted field — the bit-rot simulation
    victim.write_text(json.dumps(data))
    failures = reverify_receipts(tmp_path, verify=_checksum_verifier)
    assert len(failures) == 1
    assert "r1.json" in failures[0]


def test_reverify_fail_closed_on_empty_dir_and_raising_verifier(tmp_path: Path) -> None:
    assert reverify_receipts(tmp_path, verify=_checksum_verifier) != []
    _write_receipt(tmp_path / "r.json", {"i": 0})

    def boom(path: Path) -> bool:
        raise RuntimeError("verifier crashed")

    failures = reverify_receipts(tmp_path, verify=boom)
    assert len(failures) == 1 and "verifier raised" in failures[0]


def test_committed_receipts_dir_nonempty() -> None:
    """The gate only means something while receipts/ carries tracked receipts."""
    assert receipt_paths(REPO_ROOT / "receipts"), "receipts/ must not be empty"


def test_default_verifier_is_verify_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: the default verifier shelled to verify-research (the
    notebook verifier), which rejects every receipt — the gate reported
    10/10 failures on sealed receipts that pass verify-receipt."""
    seen: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd: list[str], **_kw: object) -> _Proc:
        seen.append(cmd)
        return _Proc()

    monkeypatch.setattr("subprocess.run", fake_run)
    receipt = tmp_path / "r.json"
    receipt.write_text("{}")
    assert _cli_verifier(receipt) is True
    assert len(seen) == 1 and "verify-receipt" in seen[0]
    assert "verify-research" not in seen[0]


def test_cli_verifier_respects_exit_status(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    from quant_fund.proofcore import ci

    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert cmd[-2:] == ["verify-receipt", str(receipt)]
        return subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    assert not ci._cli_verifier(receipt)


def test_gate_main_dispatch_and_failure_reporting(monkeypatch, capsys, tmp_path: Path) -> None:
    from quant_fund.proofcore import ci

    monkeypatch.setattr(ci, "coverage_gate", lambda _path: ["pit below floor"])
    assert ci.main(["coverage-gate", "--pyproject", str(tmp_path / "pyproject.toml")]) == 1
    assert "pit below floor" in capsys.readouterr().err

    monkeypatch.setattr(ci, "reverify_receipts", lambda _path: [])
    assert ci.main(["receipts-reverify", str(tmp_path)]) == 0
    assert "GATE OK" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Workflow YAML validity + make-target cross-reference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def workflow() -> dict:
    assert WORKFLOW_STAGED.is_file(), "docs/proofcore/proofcore.yml (staged workflow) missing"
    text = WORKFLOW_STAGED.read_text()
    # Strip the 3-line ACTIVATION header comment before parsing.
    body = text.split("\n", 3)[3]
    data = yaml.safe_load(body)
    assert isinstance(data, dict) and "jobs" in data
    return data


def test_staged_workflow_matches_activated_copy() -> None:
    """Once activated, .github/workflows/proofcore.yml must equal the staged copy."""
    if not WORKFLOW_ACTIVATED.is_file():
        pytest.skip("workflow not yet activated (needs workflow-scoped token)")
    staged_body = WORKFLOW_STAGED.read_text().split("\n", 3)[3]
    assert staged_body == WORKFLOW_ACTIVATED.read_text()


def test_workflow_declares_gate_matrix(workflow: dict) -> None:
    """§9.3 CI gate matrix, all present as jobs."""
    expected = {
        "proof-integrity",
        "leakage-scan",
        "reality-filter",
        "layering",
        "coverage-floors",
        "fx1-coverage",
        "proofcore-test",
    }
    assert expected <= set(workflow["jobs"])


def test_pull_request_jobs_do_not_receive_repository_signing_key(workflow: dict) -> None:
    """PR-controlled commands must never receive the long-lived HMAC key."""
    assert "PROOFCORE_SIGNING_KEY" not in yaml.safe_dump(workflow)


def test_workflow_references_existing_make_targets(workflow: dict) -> None:
    makefile = MAKEFILE.read_text()
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):(?:\s|.*##)", makefile, flags=re.M))
    invoked: set[str] = set()
    for job in workflow["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run", "")
            invoked.update(re.findall(r"\bmake\s+([a-z0-9-]+)", run))
    assert invoked, "workflow must invoke make targets"
    missing = invoked - targets
    assert not missing, f"workflow invokes undefined make targets: {sorted(missing)}"


def test_workflow_warn_and_advisory_modes_as_adjudicated(workflow: dict) -> None:
    jobs = workflow["jobs"]
    # Adjudicated: leakage-scan runs in WARN mode this wave — the job never
    # gates the build (no --fail-on, report archived as artifact).
    leakage_runs = "\n".join(step.get("run", "") for step in jobs["leakage-scan"]["steps"])
    assert "--fail-on error" not in leakage_runs
    assert "warn" in yaml.safe_dump(jobs["leakage-scan"]).lower()
    # Empty provenance exports skip with a notice. The job itself is blocking:
    # a scored ledger that is not 'pass' fails the build. continue-on-error
    # used to hide the empty-ledger exit 2.
    reality = jobs["reality-filter"]
    assert not reality.get("continue-on-error", False)
    reality_run = "\n".join(step.get("run", "") for step in reality["steps"])
    assert "make reality-gate" in reality_run
    assert "REALITY_FILTER_SKIP:" in reality_run
    assert "::notice title=reality-filter::" in reality_run
    for name in ("proof-integrity", "layering", "coverage-floors", "reality-filter"):
        assert not jobs[name].get("continue-on-error", False), name


def test_workflow_floors_match_pyproject(workflow: dict) -> None:
    """No second copy of the floor numbers may drift into the workflow."""
    floors = coverage_floors(PYPROJECT)
    for value in floors.values():
        assert f"--fail-under={value}" not in yaml.safe_dump(workflow), (
            "floors live in pyproject [tool.proofcore.coverage-floors]; "
            "the workflow must go through `make proofcore-coverage`"
        )


def test_receipt_verifier_dispatch(tmp_path: Path) -> None:
    """v2 envelopes and sealed v1 receipts go to verify-receipt; everything
    else (unsealed, unparseable) goes to the verify-research notebook path."""
    from quant_fund.proofcore import ci

    v2 = tmp_path / "v2.json"
    v2.write_text(json.dumps({"schema": "receipt.v2", "schema_version": 2}))
    sealed_v1 = tmp_path / "sealed.json"
    sealed_v1.write_text(json.dumps({"schema": "fleet_eval.v1", "receipt_sha256": "ab" * 32}))
    unsealed = tmp_path / "unsealed.json"
    unsealed.write_text(json.dumps({"schema": "incumbent_bench.v1"}))
    garbage = tmp_path / "garbage.json"
    garbage.write_text("{ not json")
    not_object = tmp_path / "arr.json"
    not_object.write_text("[1, 2]")

    assert ci._receipt_verifier_command(v2) == "verify-receipt"
    assert ci._receipt_verifier_command(sealed_v1) == "verify-receipt"
    assert ci._receipt_verifier_command(unsealed) == "verify-research"
    assert ci._receipt_verifier_command(garbage) == "verify-research"
    assert ci._receipt_verifier_command(not_object) == "verify-research"


def test_cli_verifier_routes_sealed_to_verify_receipt(tmp_path, monkeypatch) -> None:
    import subprocess

    from quant_fund.proofcore import ci

    receipt = tmp_path / "sealed.json"
    receipt.write_text(json.dumps({"schema": "x.v1", "receipt_sha256": "ab" * 32}))
    seen: list[str] = []

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(cmd[-2])
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(ci.subprocess, "run", fake_run)
    assert ci._cli_verifier(receipt)
    assert seen == ["verify-receipt"]
