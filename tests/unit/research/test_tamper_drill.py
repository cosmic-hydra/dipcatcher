"""Tests for tamper_drill — the mechanized fail-closed proof."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quant_fund.research.tamper_drill import (
    DRILL_SCHEMA,
    _await_fresh_ctime,
    _probe_scope,
    _tree_state,
    drill_contract_errors,
    tamper_drill,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
QUALITY = REPO_ROOT / "quality"
requires_tree = pytest.mark.skipif(
    not (QUALITY / "crown_jewels.json").is_file(),
    reason="committed integrity state not present",
)


@requires_tree
@pytest.mark.slow
def test_tamper_drill_fail_closed_on_committed_tree() -> None:
    """Every probe over the real integrity state must be caught."""
    result = tamper_drill(REPO_ROOT)
    assert result["schema"] == DRILL_SCHEMA
    assert result["baseline_errors"] == []
    assert result["n_probes"] >= 10, "drill needs real surface to be meaningful"
    assert result["n_caught"] == result["n_probes"], [
        p["probe"] for p in result["probes"] if not p.get("caught")
    ]
    assert result["verdict"] == "fail_closed"
    assert drill_contract_errors(result) == []


def test_drill_contract_clean_and_strict() -> None:
    good = {
        "schema": "tamper_drill.v1",
        "research_only": True,
        "live_pnl_claim": False,
        "probes": [
            {"probe": "flip_x", "caught": True, "errors": ["gate:x"]},
            {"probe": "delete_y", "caught": True, "errors": []},
        ],
        "n_probes": 2,
        "n_caught": 2,
        "verdict": "fail_closed",
    }
    assert drill_contract_errors(good) == []

    assert drill_contract_errors({"schema": "other"}) == ["schema_mismatch"]
    bad = {**good, "n_caught": 1}
    assert "n_caught_mismatch" in drill_contract_errors(bad)
    escaped = {**good, "verdict": "fail_closed", "n_caught": 1, "n_probes": 2}
    assert "verdict_fail_closed_but_escaped" in drill_contract_errors(escaped)


def test_probe_scope_names_every_touched_corpus(tmp_path: Path) -> None:
    for rel in (
        "src/pkg/a.py",
        ".github/workflows/ci.yml",
        ".github/dependabot.yml",
        "quality/epoch_heads.json",
        "quality/x.json",
        "README.md",
    ):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    base = _tree_state(tmp_path)

    def scope_after(change: Any) -> frozenset[str] | None:
        # Same-size rewrites right after creation are the hard case: only
        # the ctime moves, and only once the timestamp clock has ticked.
        before = _tree_state(tmp_path)
        _await_fresh_ctime(before)
        change()
        return _probe_scope(before, _tree_state(tmp_path))

    assert _probe_scope(base, base) == frozenset()
    # Nested corpora both claim a workflow; a sibling only the parent.
    assert scope_after(lambda: (tmp_path / ".github/workflows/ci.yml").write_text("y")) == {
        ".github",
        ".github/workflows",
    }
    assert scope_after(lambda: (tmp_path / ".github/dependabot.yml").write_text("yy")) == {
        ".github"
    }
    # Deletes, drops, symlinks and mode flips count; prefixes don't
    # (``srcx/`` is not ``src/``).
    assert scope_after(lambda: (tmp_path / "src/pkg/a.py").unlink()) == {"src"}
    assert scope_after(lambda: (tmp_path / "docs").mkdir()) == {"docs"}
    assert scope_after(lambda: (tmp_path / "srcx").mkdir()) == frozenset()
    assert scope_after(lambda: (tmp_path / "web").symlink_to(tmp_path / "src")) == {"web"}
    assert scope_after(lambda: (tmp_path / "quality/x.json").chmod(0o600)) == {"quality"}
    # Root files belong to no corpus: the always-run gates cover them.
    assert scope_after(lambda: (tmp_path / "README.md").write_text("changed")) == frozenset()
    # A shared epoch input re-runs every epoch gate.
    assert scope_after(lambda: (tmp_path / "quality/epoch_heads.json").write_text("{}")) is None


def test_scoped_drill_verdicts_match_full_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every scoped re-verify gives the full verifier's verdict, with a
    subset of its errors — the speedup can't turn an escape into a catch."""
    import quant_fund.research.repo_integrity as ri
    from tests.unit.research.test_repo_integrity import _git_repo

    root = _git_repo(tmp_path)
    real = ri.verify_repo
    compared: list[str] = []

    def errors(res: dict[str, Any]) -> set[str]:
        return {f"{g}:{e}" for g, gate in res["gates"].items() for e in gate["errors"]}

    def scoped_and_full(clone: Path, **kw: Any) -> dict[str, Any]:
        res = real(clone, **kw)
        if kw.get("epoch_scope") is not None:
            full = real(clone)
            assert res["ok"] == full["ok"], kw
            assert errors(res) <= errors(full), kw
            compared.append(",".join(sorted(kw["epoch_scope"])))
        return res

    monkeypatch.setattr(ri, "verify_repo", scoped_and_full)
    result = tamper_drill(root)
    assert result["verdict"] == "fail_closed", result
    assert drill_contract_errors(result) == []
    assert len(compared) >= 50
    by_probe = {p["probe"]: p["epoch_scope"] for p in result["probes"]}
    assert by_probe["flip_epoch_heads_pin"] == "all"
    assert by_probe["flip_member:docs"] == ["docs"]
