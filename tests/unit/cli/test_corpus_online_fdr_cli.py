"""CLI bridge for the corpus/stream inference lanes (PRs #382/#383).

``corpus_inference`` and ``online_fdr`` land via sibling branches — the
commands must fail closed with a typed ``BadParameter`` when the import is
unavailable and drive the lane interfaces when it is. Tests inject fake
modules into ``sys.modules`` so the seal/write/replay wiring is exercised
without the real implementations.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from quant_fund.cli.main import app
from quant_fund.research.receipt_v2 import verify_receipt_file

runner = CliRunner()


def _error_text(text: str) -> str:
    """Join a Rich error panel so a long temp path cannot split a phrase.

    Typer wraps ``BadParameter`` to the console width. A pytest tmp path
    under ``/tmp`` or macOS ``/private/var`` pushes ``does not exist`` onto
    the next line, so the words are no longer contiguous in ``result.output``.
    """
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in {"╭", "╰"}:
            continue
        if stripped.startswith("│") and stripped.endswith("│"):
            stripped = stripped[1:-1].strip()
        if stripped:
            lines.append(stripped)
    return " ".join(lines)


CORPUS_MODULE = "quant_fund.research.corpus_inference"
ONLINE_MODULE = "quant_fund.research.online_fdr"


def _write_receipt(path: Path, payload: Any, *, mtime: float) -> Path:
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    os.utime(path, (mtime, mtime))
    return path


def _fake_corpus_audit(receipts_dir: Path, *, q: float = 0.05, glob: str = "*.json") -> dict:
    root = Path(receipts_dir)
    if not root.is_dir():
        raise ValueError(f"receipts dir {root} does not exist")
    if not (0.0 < q < 1.0):
        raise ValueError("q must lie in (0, 1)")
    files = sorted(p for p in root.glob(glob) if p.is_file())
    return {
        "kind": "corpus_inference.v1",
        "schema": "corpus_inference.v1",
        "data_label": "SYNTHETIC",
        "research_only": True,
        "live_pnl_claim": False,
        "inputs_sha256": "ab" * 32,
        "params": {"q": q, "glob": glob},
        "n_receipts": len(files),
        "n_parse_errors": 0,
        "parse_errors": [],
        "n_p_findings": 0,
        "n_e_findings": 0,
        "n_survivors": 0,
        "surviving_claims": [],
        "corpus_evalue": 1.0,
        "corpus_reject_at_alpha": False,
    }


def _install_corpus_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    harvest=None,
) -> None:
    mod = types.ModuleType(CORPUS_MODULE)
    mod.corpus_audit = _fake_corpus_audit
    mod.harvest_findings = harvest or (lambda payload, source: list(payload.get("findings", [])))

    def _write(receipt: Any, receipts_dir: Any, *, receipt_version: int = 1) -> Path:
        from quant_fund.research.receipt_v2 import seal_receipt

        sealed = seal_receipt(dict(receipt))
        digest = str(receipt.get("inputs_sha256", ""))[:16]
        path = Path(receipts_dir) / f"corpus_inference_{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sealed))
        return path

    mod.write_corpus_receipt = _write
    monkeypatch.setitem(sys.modules, CORPUS_MODULE, mod)


class _FakeOnlineFDR:
    """Minimal Foster–Stine stand-in: records the replayed p sequence."""

    instances: list[_FakeOnlineFDR] = []

    def __init__(self, level: float = 0.05, **_kw: Any) -> None:
        if not (0.0 < level < 1.0):
            raise ValueError("level must lie in (0, 1)")
        self.level = float(level)
        self.ps: list[float] = []
        _FakeOnlineFDR.instances.append(self)

    def update(self, p_value: float) -> float:
        p = float(p_value)
        self.ps.append(p)
        return p

    def stream_report(self) -> dict[str, Any]:
        rejected = [i for i, p in enumerate(self.ps) if p <= self.level]
        return {
            "kind": "online_fdr.v1",
            "n_tests": len(self.ps),
            "n_rejections": len(rejected),
            "rejection_indices": rejected,
            "final_wealth": self.level,
            "level": self.level,
            "evidence": ["foster_stine_alpha_investing"],
        }


def _install_online_module(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOnlineFDR.instances = []
    mod = types.ModuleType(ONLINE_MODULE)
    mod.OnlineFDR = _FakeOnlineFDR

    def _write(receipt: Any, receipts_dir: Any, *, receipt_version: int = 1) -> Path:
        from quant_fund.research.receipt_v2 import seal_receipt

        sealed = seal_receipt(dict(receipt))
        digest = str(receipt.get("inputs_sha256", ""))[:16]
        path = Path(receipts_dir) / f"online_fdr_{digest}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sealed))
        return path

    mod.write_online_fdr_receipt = _write
    monkeypatch.setitem(sys.modules, ONLINE_MODULE, mod)


def _last_online() -> _FakeOnlineFDR:
    assert _FakeOnlineFDR.instances, "OnlineFDR was never constructed"
    return _FakeOnlineFDR.instances[-1]


def test_corpus_writes_sealed_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _write_receipt(receipts / "a.json", {"kind": "x.v1", "p_value": 0.04}, mtime=1000)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        ["corpus", "--receipts-dir", str(receipts), "--out-dir", str(out), "--q", "0.1"],
    )
    assert result.exit_code == 0, result.output
    assert "DATA_LABEL=SYNTHETIC" in result.output
    written = list(out.glob("corpus_inference_*.json"))
    assert len(written) == 1
    assert written[0].name == f"corpus_inference_{'ab' * 8}.json"
    sealed = json.loads(written[0].read_text())
    assert sealed["schema"] == "corpus_inference.v1"
    assert sealed["params"]["q"] == 0.1
    assert sealed["live_pnl_claim"] is False
    assert verify_receipt_file(written[0])["valid"] is True
    assert f"receipt={written[0]}" in result.output


def test_corpus_fails_closed_without_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, CORPUS_MODULE, None)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    result = runner.invoke(app, ["corpus", "--receipts-dir", str(receipts)])
    assert result.exit_code != 0
    assert "requires corpus_inference (PR #382)" in result.output


def test_corpus_rejects_bad_q(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    result = runner.invoke(app, ["corpus", "--receipts-dir", str(receipts), "--q", "1.5"])
    assert result.exit_code != 0
    assert "q must lie in (0, 1)" in result.output


def test_corpus_missing_dir_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    result = runner.invoke(app, ["corpus", "--receipts-dir", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "does not exist" in _error_text(result.output)


def test_online_fdr_per_finding_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _write_receipt(
        receipts / "first.json",
        {"kind": "x.v1", "findings": [{"stat": "p", "value": 0.4}, {"stat": "e", "value": 9.0}]},
        mtime=100,
    )
    _write_receipt(
        receipts / "second.json",
        {"kind": "x.v1", "findings": [{"stat": "p", "value": 0.01}]},
        mtime=200,
    )
    out = tmp_path / "out"

    result = runner.invoke(
        app, ["online-fdr", "--receipts-dir", str(receipts), "--out-dir", str(out)]
    )
    assert result.exit_code == 0, result.output
    # e-values are not fed to the controller; per-finding => every p is a test.
    assert _last_online().ps == [0.4, 0.01]
    written = list(out.glob("online_fdr_*.json"))
    assert len(written) == 1
    sealed = json.loads(written[0].read_text())
    assert sealed["schema"] == "online_fdr.v1"
    assert sealed["n_tests"] == 2
    assert sealed["n_p_findings"] == 2
    assert sealed["params"]["per_receipt"] is False
    assert sealed["live_pnl_claim"] is False
    assert verify_receipt_file(written[0])["valid"] is True
    assert f"receipt={written[0]}" in result.output


def test_online_fdr_per_receipt_uses_min_p(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _write_receipt(
        receipts / "one.json",
        {"kind": "x.v1", "findings": [{"stat": "p", "value": 0.5}, {"stat": "p", "value": 0.02}]},
        mtime=100,
    )
    _write_receipt(
        receipts / "two.json",
        {"kind": "x.v1", "findings": [{"stat": "e", "value": 4.0}]},
        mtime=200,
    )
    _write_receipt(
        receipts / "three.json",
        {"kind": "x.v1", "findings": [{"stat": "p", "value": 0.3}]},
        mtime=300,
    )

    result = runner.invoke(
        app,
        [
            "online-fdr",
            "--receipts-dir",
            str(receipts),
            "--out-dir",
            str(tmp_path / "o"),
            "--per-receipt",
        ],
    )
    assert result.exit_code == 0, result.output
    # One test per receipt carrying a p; the receipt with only e-values is not a test.
    assert _last_online().ps == [0.02, 0.3]
    sealed = json.loads(next((tmp_path / "o").glob("online_fdr_*.json")).read_text())
    assert sealed["n_tests"] == 2
    assert sealed["n_p_findings"] == 3


def test_online_fdr_replays_in_mtime_name_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    # Name order disagrees with mtime order: mtime wins.
    _write_receipt(receipts / "a_late.json", {"findings": [{"stat": "p", "value": 0.9}]}, mtime=300)
    _write_receipt(
        receipts / "z_early.json", {"findings": [{"stat": "p", "value": 0.1}]}, mtime=100
    )
    # Same-mtime ties break on filename.
    _write_receipt(receipts / "b_tie.json", {"findings": [{"stat": "p", "value": 0.5}]}, mtime=200)
    _write_receipt(receipts / "a_tie.json", {"findings": [{"stat": "p", "value": 0.4}]}, mtime=200)

    result = runner.invoke(
        app,
        ["online-fdr", "--receipts-dir", str(receipts), "--out-dir", str(tmp_path / "o")],
    )
    assert result.exit_code == 0, result.output
    assert _last_online().ps == [0.1, 0.4, 0.5, 0.9]


def test_online_fdr_skips_unparseable_and_non_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _write_receipt(receipts / "bad.json", "not json{", mtime=100)
    _write_receipt(receipts / "list.json", [1, 2, 3], mtime=200)
    _write_receipt(receipts / "good.json", {"findings": [{"stat": "p", "value": 0.2}]}, mtime=300)

    result = runner.invoke(
        app,
        ["online-fdr", "--receipts-dir", str(receipts), "--out-dir", str(tmp_path / "o")],
    )
    assert result.exit_code == 0, result.output
    assert _last_online().ps == [0.2]
    sealed = json.loads(next((tmp_path / "o").glob("online_fdr_*.json")).read_text())
    assert sealed["n_skipped"] == 2
    assert sorted(sealed["skipped_files"]) == ["bad.json", "list.json"]


def test_online_fdr_tolerates_attribute_findings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_corpus_module(
        monkeypatch,
        harvest=lambda payload, source: [
            SimpleNamespace(stat="p", value=0.25),
            SimpleNamespace(stat="e", value=7.0),
        ],
    )
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    _write_receipt(receipts / "r.json", {"kind": "x.v1"}, mtime=100)

    result = runner.invoke(
        app,
        ["online-fdr", "--receipts-dir", str(receipts), "--out-dir", str(tmp_path / "o")],
    )
    assert result.exit_code == 0, result.output
    assert _last_online().ps == [0.25]


@pytest.mark.parametrize("missing", [CORPUS_MODULE, ONLINE_MODULE])
def test_online_fdr_fails_closed_without_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    # The corpus guard runs first — install its fake so the online_fdr
    # guard is what fires when that is the missing module.
    if missing == ONLINE_MODULE:
        _install_corpus_module(monkeypatch)
    monkeypatch.setitem(sys.modules, missing, None)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    result = runner.invoke(app, ["online-fdr", "--receipts-dir", str(receipts)])
    assert result.exit_code != 0
    expected = (
        "requires corpus_inference (PR #382)"
        if missing == CORPUS_MODULE
        else ("requires online_fdr (PR #383)")
    )
    assert expected in result.output


def test_online_fdr_rejects_bad_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    result = runner.invoke(app, ["online-fdr", "--receipts-dir", str(receipts), "--level", "0"])
    assert result.exit_code != 0
    assert "level must lie in (0, 1)" in result.output


def test_online_fdr_missing_dir_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_corpus_module(monkeypatch)
    _install_online_module(monkeypatch)
    result = runner.invoke(app, ["online-fdr", "--receipts-dir", str(tmp_path / "nope")])
    assert result.exit_code != 0
    assert "does not exist" in _error_text(result.output)
