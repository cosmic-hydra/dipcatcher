"""Fail-closed edges in the leakage scanner, IO guard, and report CLI."""

from __future__ import annotations

import ast
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_fund.leakage import ast_scan
from quant_fund.leakage.ast_scan import (
    _innermost_function_name,
    _scope_assignments,
    _shift_call_parts,
    collect_py_files,
    scan_file,
)
from quant_fund.leakage.cli import leakage_app
from quant_fund.leakage.guard import IOGuard
from quant_fund.leakage.patterns import _TOKEN_FALLBACK, _load_forbidden_tokens
from quant_fund.leakage.report import report_to_text, scan_paths
from quant_fund.leakage.watchdog import LeakageWatchdog


def _write(tmp_path: Path, source: str, name: str = "sample.py") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


def _deep_unary(depth: int) -> str:
    expr = "1"
    for _ in range(depth):
        expr = f"+({expr})"
    return expr


def test_shift_folding_and_receiver_shapes(tmp_path: Path) -> None:
    source = f"""
def scan_shapes():
    close = frame.close
    frame.close.shift(-1)
    pl.col("close").shift(-1)
    col("close").shift(-1)
    col(close).shift(-1)
    col(bars.mean()).shift(-1)
    obj["col"]("close").shift(-1)
    close.shift(periods=-1)
    close.shift(foo=1, periods=-1)
    close.shift()
    close.shift(-2 + 1)
    close.shift((-1) * 2)
    close.shift((-1) / 1)
    close.shift(+(-1))
    close.shift(~1)
    close.shift({_deep_unary(70)})
    delta_mid, other = mid.shift(-1) - mid, mid
    self.delta_mid = mid.shift(-1)
"""
    findings = scan_file(_write(tmp_path, source))
    assert any(item.rule_id == "LH001" for item in findings)


def test_fold_fit_without_arguments_and_anonymous_receivers(tmp_path: Path) -> None:
    source = """
def train_fold(X):
    StandardScaler().fit()
    thing["StandardScaler"].fit(X)
    (lookup["StandardScaler"])().fit(X)
"""
    findings = scan_file(_write(tmp_path, source), rules={"LH003"})
    assert any(item.rule_id == "LH003" for item in findings)


def test_shift_parts_reject_non_calls() -> None:
    assign = ast.parse("x = 1").body[0]
    assert _shift_call_parts(assign) is None


def test_scope_assignments_work_outside_a_scan() -> None:
    tree = ast.parse("x = 1\n")
    assigns = _scope_assignments(tree)
    assert "x" in assigns


def test_innermost_function_keeps_the_tighter_span() -> None:
    spans = [(10, 20, "inner"), (5, 40, "outer")]
    assert _innermost_function_name(spans, 12) == "inner"


def test_unparse_failure_does_not_invent_a_backward_fill_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_node: ast.AST) -> str:
        raise RuntimeError("unparse failed")

    monkeypatch.setattr(ast_scan.ast, "unparse", _boom)
    source = """
def fill(frame):
    frame.bfill()
    event_time = frame["event_time"]
"""
    findings = scan_file(_write(tmp_path, source), rules={"LH010"})
    assert findings == []


def test_layering_relative_and_dynamic_imports(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "src" / "quant_fund" / "proof",
        """
from .... import sibling
import_module()
importlib.import_module("quant_fund.proof.inner")
""",
        name="edge.py",
    )
    findings = scan_file(path, rules={"LH011"})
    assert any("relative" in item.message or "non-whitelisted" in item.message for item in findings)
    malformed = ast.ImportFrom(module=None, names=[ast.alias(name="x")], level=0)
    assert ast_scan._lh011_imported_modules(malformed, path.as_posix()) == []


def test_comment_tokenizer_errors_are_ignored() -> None:
    out: list[object] = []
    ast_scan._lh013_comment_findings("def f(\n", "sample.py", out, ast_scan._lh013_patterns())
    assert out == []


def test_fstring_mention_without_a_number_is_not_a_headline(tmp_path: Path) -> None:
    source = 'def claim():\n    return f"the sharpe"\n'
    findings = scan_file(_write(tmp_path, source), rules={"LH013"})
    assert findings == []


def test_recursive_helper_call_does_not_double_count(tmp_path: Path) -> None:
    source = """
def helper():
    close.shift(-1)
    helper()
"""
    findings = scan_file(_write(tmp_path, source), rules={"LH001", "LH014"})
    assert any(item.rule_id == "LH001" for item in findings)
    assert not any(item.rule_id == "LH014" for item in findings)


def test_collect_py_files_drops_duplicate_paths(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = 1\n")
    assert collect_py_files([path, path]) == [path]


def test_unknown_fail_on_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "x = 1\n")
    result = CliRunner().invoke(
        leakage_app, ["scan", "--paths", str(path), "--fail-on", "sometimes"]
    )
    assert result.exit_code == 2
    assert "unknown --fail-on" in result.output


def test_unreadable_source_still_reports_the_parse_finding(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_bytes(b"\xff\xfe not utf-8")
    report = scan_paths([path])
    assert any(item.rule_id == "LH012" for item in report.findings)
    text = report_to_text(report)
    assert "LH012" in text
    assert "        |" not in text


def test_token_catalog_falls_back_when_the_constant_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("quant_fund.research.catalog")
    monkeypatch.setitem(sys.modules, "quant_fund.research.catalog", fake)
    assert _load_forbidden_tokens() == _TOKEN_FALLBACK


def test_watchdog_exposes_strict_mode() -> None:
    assert LeakageWatchdog(strict=False).strict is False


def test_vault_lookup_skips_a_longer_unrelated_root(tmp_path: Path) -> None:
    short = tmp_path / "data"
    long = short / "nested"
    long.mkdir(parents=True)
    target = short / "bars.parquet"
    target.write_bytes(b"")
    short_vault = object()
    guard = IOGuard(mode="enforce", vault_map={long: object(), short: short_vault})
    found = guard._vault_for(target)
    assert found is not None
    assert found[0] is short_vault
    stamp = datetime(2024, 1, 1, tzinfo=UTC)
    assert stamp.tzinfo is UTC
