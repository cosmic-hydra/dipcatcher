"""PROOFCORE layering gate (DESIGN.md §1.3 — A2 F1 local fix).

AST-walks ``src/quant_fund/{pit,proof,leakage,reality,proofcore}/**`` and
fails on any import outside the per-package whitelist. Workstreams land in
parallel, so a package directory that does not exist yet is reported as
pending (skip), not failed — but every directory that DOES exist is fully
gated.

Also gates the reverse direction: no existing SCC package may import a
PROOFCORE package at module top level, except the layer-4 CLI glue
(``src/quant_fund/cli/**``), the sanctioned mount point (§9.2).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "quant_fund"
PROOFCORE_PACKAGES = ("pit", "proof", "leakage", "reality", "proofcore")

# §1.3 whitelist: quant_fund subpackages each PROOFCORE package may import at
# module top level (intra-package imports are always allowed).
TOP_LEVEL_WHITELIST: dict[str, frozenset[str]] = {
    "proofcore": frozenset({"proofcore"}),
    "pit": frozenset({"pit", "proofcore", "schemas", "utils", "data"}),
    # Adjudicated edge (LH011 whitelist in leakage/rules.py): proof -> config
    # top-level is allowed — config is layer-0, so the edge stays acyclic.
    "proof": frozenset({"proof", "proofcore", "schemas", "utils", "config"}),
    "leakage": frozenset({"leakage", "proofcore", "schemas"}),
    "reality": frozenset({"reality", "proofcore", "metrics", "validation"}),
}

# Additional quant_fund roots allowed ONLY inside function bodies (lazy
# imports — §1.3: integration goes through function-level imports).
LAZY_WHITELIST: dict[str, frozenset[str]] = {
    # proofcore-standalone (configs/arch_boundaries.toml) and LH011 both bar
    # every quant_fund edge from proofcore, lazy included — its cli carries an
    # inlined atomic-writer instead of reaching utils.atomicio.
    "proofcore": frozenset(),
    "pit": frozenset(),
    # Adjudicated lazy edges (LH011_LAZY_WHITELIST in leakage/rules.py):
    # proof lazily reaches pit (W1 vault seam), leakage (W3 watchdog), and
    # metrics (A1 F2 headline recompute); none import proof back, so the lazy
    # edges cannot create a cycle. promotion_receipt also lazily reaches
    # pipeline / registry / research (identity + approval + seal) — same
    # no-import-time-cycle reasoning.
    "proof": frozenset(
        {
            "backtest",
            "cli",
            "leakage",
            "metrics",
            "pit",
            "pipeline",
            "registry",
            "research",
        }
    ),
    "leakage": frozenset({"pit", "cli", "config", "utils", "research"}),
    # Adjudicated lazy edge (this durability sweep): reality/cli writes its
    # outputs via utils.atomicio — same layer-0 reasoning as proofcore.
    # reality/cli also lazily drives research.reality_sweep and
    # research.reality_survivorship (the `dipcatcher reality` commands). Those
    # two modules lazily import reality.cscv / reality.report back, so the pair
    # is a mutual *lazy* dependency: neither edge exists at import time, which
    # is this codebase's documented cycle-breaking mechanism
    # (configs/arch_boundaries.toml). Mirrors LH011_LAZY_WHITELIST in
    # leakage/rules.py.
    "reality": frozenset({"cli", "utils", "research"}),
}

# Third-party roots each package may use (stdlib is always allowed).
THIRD_PARTY_WHITELIST: dict[str, frozenset[str]] = {
    "proofcore": frozenset({"duckdb", "pydantic", "typer"}),
    "pit": frozenset({"polars", "pydantic", "numpy", "typer"}),
    "proof": frozenset({"polars", "pydantic", "numpy", "typer"}),
    # pandas/polars are lazy-only inside guard.py's hook install/remove —
    # interposing their readers is the IO guard's purpose (W8).
    "leakage": frozenset({"pandas", "polars", "pydantic", "typer"}),
    "reality": frozenset({"numpy", "scipy", "pydantic", "polars", "typer"}),
}

# Third-party roots allowed ONLY inside function bodies — same lazy-edge
# reasoning as LAZY_WHITELIST. Adjudicated: leakage/guard.py monkey-patches
# pandas/polars readers at runtime, so it must import them lazily to avoid
# paying the import cost (and hard dep) for callers that never install the
# IO guard.
LAZY_THIRD_PARTY_WHITELIST: dict[str, frozenset[str]] = {
    "proofcore": frozenset(),
    "pit": frozenset(),
    "proof": frozenset(),
    "leakage": frozenset({"pandas", "polars"}),
    "reality": frozenset(),
}


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    """(full dotted module, line, is_top_level) for every import in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lazy_nodes: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lazy_nodes.update(id(inner) for inner in ast.walk(node))
    out: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append((alias.name, node.lineno, id(node) not in lazy_nodes))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append((node.module, node.lineno, id(node) not in lazy_nodes))
    return out


def _violations(pkg: str) -> list[str]:
    problems: list[str] = []
    for path in sorted((SRC / pkg).rglob("*.py")):
        for module, line, top_level in _imports(path):
            root = module.split(".")[0]
            rel = path.relative_to(SRC)
            if root == "__future__":
                continue
            if root == "quant_fund":
                parts = module.split(".")
                sub = parts[1] if len(parts) > 1 else ""
                if sub in TOP_LEVEL_WHITELIST[pkg]:
                    continue
                if not top_level and sub in LAZY_WHITELIST[pkg]:
                    continue
                problems.append(f"{rel}:{line}: {module} not whitelisted for {pkg}/")
                continue
            if root == "fx1":
                problems.append(f"{rel}:{line}: PROOFCORE packages never import fx1")
                continue
            if (
                root not in sys.stdlib_module_names
                and root not in THIRD_PARTY_WHITELIST[pkg]
                and not (not top_level and root in LAZY_THIRD_PARTY_WHITELIST.get(pkg, frozenset()))
            ):
                problems.append(f"{rel}:{line}: third-party import {root!r} not whitelisted")
    return problems


@pytest.mark.parametrize("pkg", PROOFCORE_PACKAGES)
def test_package_import_whitelist(pkg: str) -> None:
    if not (SRC / pkg).is_dir():
        pytest.skip(f"quant_fund/{pkg} not merged yet (parallel workstream)")
    problems = _violations(pkg)
    assert problems == [], "layering violations:\n" + "\n".join(problems)


def test_contracts_imports_only_stdlib_and_pydantic() -> None:
    """§1.3 hard rule: proofcore/contracts.py is layer 1 — no quant_fund, ever."""
    path = SRC / "proofcore" / "contracts.py"
    for module, line, _top in _imports(path):
        root = module.split(".")[0]
        assert root in {"__future__", "pydantic"} or root in sys.stdlib_module_names, (
            f"contracts.py:{line} imports {module!r} — only stdlib + pydantic allowed"
        )


def test_scc_does_not_import_proofcore_at_top_level() -> None:
    """Reverse direction: only cli/** (layer 4) may top-level-import new packages."""
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        if rel.parts[0] in PROOFCORE_PACKAGES or rel.parts[0] == "cli":
            continue
        for module, line, top_level in _imports(path):
            parts = module.split(".")
            if (
                top_level
                and len(parts) > 1
                and parts[0] == "quant_fund"
                and parts[1] in PROOFCORE_PACKAGES
            ):
                offenders.append(f"{rel}:{line} imports {module}")
    assert offenders == [], "SCC top-level imports of PROOFCORE packages:\n" + "\n".join(offenders)
