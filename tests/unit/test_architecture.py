"""Layering and module-size guards for the quant_fund and fx1 libraries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path("src")
MAX_MODULE_LINES = 2000
LIBRARY_ROOTS = (SRC / "quant_fund", SRC / "fx1")

# CLI may depend on the library. The library must not depend on the CLI.
CLI_PREFIX = "quant_fund.cli"
# Research and the rest of the core may use execution impact math.
# They must not import the simulated broker or the paper/live loop.
# simtest, pretrade, formal, and parity are the simulation boundary that
# drives SimulatedBroker (parity replays one tape into that broker only).
# They are not on the research/pipeline/models import path.
BROKER_MODULES = ("quant_fund.execution.simulated_broker",)
PAPER_PREFIX = "quant_fund.paper"
LIVE_PREFIX = "quant_fund.live"
BROKER_ALLOWED_PREFIXES = (
    "quant_fund.cli",
    "quant_fund.paper",
    "quant_fund.execution",
    "quant_fund.simtest",
    "quant_fund.pretrade",
    "quant_fund.formal",
    "quant_fund.parity",
    "quant_fund.live",
)


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    current = _module_name(path)
    parts = current.split(".")
    package = parts if path.name == "__init__.py" else parts[:-1]
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    found.append(node.module)
                continue
            base = package[: len(package) - (node.level - 1)]
            if node.module:
                base = [*base, *node.module.split(".")]
            found.append(".".join(base))
    return found


def _under(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _line_budgets() -> dict[str, int]:
    """Pinned budgets for legacy oversized modules; the pin may only shrink."""
    manifest = Path("quality/module_line_budgets.txt")
    budgets: dict[str, int] = {}
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise AssertionError(f"{manifest}:{line_number}: expected PATH COUNT")
        rel, count_text = fields
        if rel in budgets:
            raise AssertionError(f"{manifest}:{line_number}: duplicate budget for {rel}")
        count = int(count_text)
        if count <= MAX_MODULE_LINES:
            raise AssertionError(f"{manifest}:{line_number}: remove unnecessary budget for {rel}")
        budgets[rel] = count
    return budgets


def test_modules_stay_under_max_lines() -> None:
    budgets = _line_budgets()
    offenders: list[str] = []
    seen: set[str] = set()
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            rel = path.relative_to(SRC).as_posix()
            lines = len(path.read_text(encoding="utf-8").splitlines())
            budget = budgets.get(rel)
            if budget is not None:
                seen.add(rel)
                if lines > budget:
                    offenders.append(f"{rel}:{lines} over pinned budget {budget}")
                elif lines < budget:
                    action = (
                        "remove the pin"
                        if lines <= MAX_MODULE_LINES
                        else f"lower the pin to {lines}"
                    )
                    offenders.append(f"{rel}:{lines} below pinned budget {budget}; {action}")
            elif lines > MAX_MODULE_LINES:
                offenders.append(f"{rel}:{lines}")
    offenders.extend(f"stale budget pin: {rel}" for rel in sorted(set(budgets) - seen))
    assert offenders == []


def test_line_budget_manifest_rejects_duplicate_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "module_line_budgets.txt").write_text(
        "quant_fund/legacy.py 2001\nquant_fund/legacy.py 2002\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match="duplicate budget"):
        _line_budgets()


def test_module_line_budget_must_follow_shrinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = tmp_path / "src" / "quant_fund" / "legacy.py"
    module.parent.mkdir(parents=True)
    module.write_text("line\n" * 2001, encoding="utf-8")
    quality = tmp_path / "quality"
    quality.mkdir()
    (quality / "module_line_budgets.txt").write_text(
        "quant_fund/legacy.py 2002\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(
        AssertionError,
        match="below pinned budget 2002; lower the pin to 2001",
    ):
        test_modules_stay_under_max_lines()


def test_library_does_not_import_cli() -> None:
    offenders: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            module = _module_name(path)
            if _under(module, CLI_PREFIX):
                continue
            for imported in _imported_modules(path):
                if _under(imported, CLI_PREFIX):
                    offenders.append(f"{module} imports {imported}")
    assert offenders == []


def test_core_does_not_import_broker_or_paper() -> None:
    offenders: list[str] = []
    for root in LIBRARY_ROOTS:
        for path in root.rglob("*.py"):
            module = _module_name(path)
            if any(_under(module, prefix) for prefix in BROKER_ALLOWED_PREFIXES):
                continue
            for imported in _imported_modules(path):
                if (
                    _under(imported, PAPER_PREFIX)
                    or _under(imported, LIVE_PREFIX)
                    or any(_under(imported, broker) for broker in BROKER_MODULES)
                ):
                    offenders.append(f"{module} imports {imported}")
    assert offenders == []
