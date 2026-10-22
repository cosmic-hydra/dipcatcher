"""Docs <-> code drift gate: every code-form reference in docs must resolve.

Inline-code spans (`` `...` ``) and fenced code blocks in ``docs/**/*.md``,
``README.md`` and ``AGENTS.md`` are scanned for four reference classes:

1. ``quant_fund.<dotted.path>`` / ``fx1.<dotted.path>`` — must resolve to a real
   module (``importlib.util.find_spec``) or to an attribute chain off the
   deepest importable prefix (``getattr``, with a module-file/AST fallback for
   modules whose optional deps are absent, e.g. numba on macOS arm64).
2. ``dipcatcher <cmd> [sub]`` — must be a registered Typer command or group of
   ``quant_fund.cli.main.app``; group invocations with a bare second token must
   name a real sub-command.
3. ``<root>/<path>`` repo paths — must exist as a tracked file or directory
   (``git ls-files`` is the truth, so results do not depend on local generated
   state). ``src/quant_fund/`` and ``src/fx1/`` abbreviations and ``cd`` inside
   a fenced block are honoured. Glob-ish references (``foo_*``, ``vN``, ``NNN``)
   must match at least one tracked path.
4. ``make <target>`` — must be a target declared in the root ``Makefile``.

``data/`` is intentionally absent from the checked path roots: that tree is
gitignored/generated, so its contents are not a committed-state property.

``KNOWN_MISSES`` is the shrink-only manifest for documented-but-not-landed
names; every entry must carry a justification and the dict must equal the
current miss set exactly — landed work removes entries, never adds them.
"""

from __future__ import annotations

import ast
import contextlib
import fnmatch
import importlib
import importlib.util
import re
import subprocess
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
DOC_FILES = sorted(REPO.glob("docs/**/*.md")) + [REPO / "README.md", REPO / "AGENTS.md"]

INLINE_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)

# Path-boundary lookbehind: a leading `/` means the token is a URL route
# (`/research/latest`) or the tail of a longer path (`metadata/research/x.json`).
SYMBOL_RE = re.compile(r"(?<![\w/.-])((?:quant_fund|fx1)(?:\.[A-Za-z_]\w*)+)")
PATH_RE = re.compile(
    r"(?<![\w/.-])((?:scripts|configs|docs|src|tests|examples|spec|third_party|rust|web|"
    r"deploy|docker|verifier|quality|clients|notebooks|replay|research|artifacts|receipts)"
    r"(?:/[\w.-]+)+/?)"
)
MAKE_RE = re.compile(r"(?<![\w/.-])make\s+([a-zA-Z][a-zA-Z0-9_-]*)")
CD_RE = re.compile(r"^cd\s+([A-Za-z0-9_./-]+)")
ENV_PREFIX_RE = re.compile(r"^(?:[A-Z_][A-Z0-9_]*=\S+\s+)+")
CODEWORD = re.compile(r"^[a-z][a-z0-9_-]*$")
GLOB_SEGMENT = re.compile(r"(?:^|/)(?:vN|NNN|YYYY)(?:[/.-]|$)")

# A dotted ref ending in one of these is a file reference (`fx1.yml`,
# `fx1.pt`), not a module/attribute path — skip the symbol check for it.
FILE_TAILS = frozenset(
    {
        "yml",
        "yaml",
        "py",
        "md",
        "txt",
        "json",
        "jsonl",
        "toml",
        "cfg",
        "ini",
        "csv",
        "parquet",
        "pt",
        "pth",
        "onnx",
        "safetensors",
        "db",
        "sqlite",
        "sh",
        "ps1",
        "mjs",
        "ts",
        "tsx",
        "js",
        "jsx",
        "cff",
        "lock",
        "png",
        "svg",
        "html",
        "css",
        "mmd",
        "zip",
        "gz",
        "whl",
        "env",
        "example",
    }
)

# English words that follow "make" in prose but are never Makefile targets.
MAKE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "do",
        "it",
        "its",
        "so",
        "the",
        "this",
        "that",
        "sure",
        "use",
        "up",
        "way",
        "work",
        "sense",
        "note",
        "progress",
        "haste",
        "money",
        "things",
        "matters",
        "count",
        "ready",
        "peace",
        "room",
        "time",
    }
)

# Documented-but-not-landed references. Shrink-only: deleting a stale doc line
# or landing the named thing is the only way this dict changes.
KNOWN_MISSES: dict[str, str] = {
    # hedge_lab training runs through the research pipeline; the dedicated
    # `dipcatcher hedge-lab` entry point is specified in docs/decisions/024 but
    # no Typer command is registered yet.
    "dipcatcher hedge-lab": "hedge_lab has no dedicated CLI yet (decision 024).",
    # robinhood_plus is a research-catalog family (quant_fund.models.robinhood_plus);
    # the `dipcatcher robinhood-plus --compare` card CLI from decision 023 is unlanded.
    "dipcatcher robinhood-plus": "compare CLI specified in decision 023, not landed.",
    # Pending reality-ledger export written by run_sweep; absent in a clean
    # checkout by design — decided studies archive to research/reality/studies/.
    "research/reality/trials.jsonl": "generated pending ledger (decision 038).",
    # Receipt emitted when the ranker-probability experiment is run.
    "receipts/ranker_probability_wide_20260927.json": "experiment output receipt.",
    # Carry-pipeline output dirs populated by carry runs, not committed.
    "artifacts/carry_champion": "generated carry-pipeline output dir.",
    "artifacts/carry_equity": "generated carry-pipeline output dir.",
    # Gitignored vendored weights dir; fetched out-of-band, never committed.
    "third_party/kronos_weights": "gitignored vendored weights dir.",
    # ADR filename template documented in docs/adr/README.md; not a real file.
    "docs/decisions/NNN-slug.md": "ADR filename template, intentionally absent.",
    # `tests/tests` is named only to say the mirror was dropped (ADR-0003).
    "tests/tests": "historical mention of the deleted tests mirror (ADR-0003).",
    # `web/dist` is the vite build output dir — generated, never committed.
    "web/dist": "vite build output, generated by `npm run build`.",
    # `fx1.operation-result/v1` is a schema tag, not an import path.
    "fx1.operation": "schema tag prefix in FX1_OPERATIONS.md, not a module.",
    # `fx1.backend` and `fx1.byok` are API body-block / header names in
    # AUDIT_LEDGER.md, not Python module paths.
    "fx1.backend": "API body-block name in AUDIT_LEDGER.md, not a module.",
    "fx1.byok": "API body-block / header name in AUDIT_LEDGER.md, not a module.",
    # Isolated-branch lanes referenced by planning docs on main (three-branch
    # retention policy); entries shrink away when the lanes land.
    "dipcatcher blueprint": "blueprint lane lives on isolated branch; EXECUTIVE_BLUEPRINT_IMPLEMENTATION.md documents the planned landing.",
}


def _tracked_index() -> tuple[frozenset[str], frozenset[str]]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    # git ls-files always emits forward slashes; PurePosixPath keeps the
    # parent strings slash-separated on Windows too (Path would emit
    # backslashes there and every directory ref would miss).
    dirs = {str(p) for f in out for p in PurePosixPath(f).parents if str(p) != "."}
    return frozenset(out), frozenset(dirs)


TRACKED_FILES, TRACKED_DIRS = _tracked_index()


def _code_regions(text: str) -> list[tuple[int, str, str | None]]:
    """(lineno, content, cwd) for inline spans and fenced lines.

    ``cwd`` is the last ``cd <dir>`` seen inside the enclosing fence, so paths
    written relative to the block's working directory resolve.
    """
    regions: list[tuple[int, str, str | None]] = []
    fences = list(FENCE_RE.finditer(text))
    for m in fences:
        start = text[: m.start()].count("\n") + 1
        cwd: str | None = None
        for j, line in enumerate(m.group(1).splitlines()):
            stripped = line.strip()
            cd = CD_RE.match(stripped)
            if cd:
                cwd = cd.group(1).rstrip("/")
            regions.append((start + j, stripped, cwd))
    for m in INLINE_RE.finditer(text):
        if any(f.start() <= m.start() < f.end() for f in fences):
            continue
        regions.append((text[: m.start()].count("\n") + 1, m.group(1).strip(), None))
    return regions


def _iter_refs():
    for doc in DOC_FILES:
        rel = doc.relative_to(REPO).as_posix()
        for lineno, region, cwd in _code_regions(doc.read_text(encoding="utf-8")):
            yield rel, lineno, region, cwd


def _cli_tokens(region: str) -> list[str] | None:
    """Tokens following `dipcatcher` when it is the command being invoked."""
    s = ENV_PREFIX_RE.sub("", region.strip())
    for prefix in ("$ ", "uv run --frozen ", "uv run "):
        if s.startswith(prefix):
            s = s[len(prefix) :].strip()
            break
    toks = s.split()
    if not toks or toks[0] != "dipcatcher":
        return None
    # Stop at the first token that is not a command word or flag: a trailing
    # `# comment`, a path, or prose such as `dipcatcher · fx1` is not argv.
    out: list[str] = []
    for t in toks[1:]:
        if not (CODEWORD.fullmatch(t) or t.startswith("-")):
            break
        out.append(t)
    return out


def _root_commands() -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    from dipcatcher_cli.app import LAUNCHER_COMMANDS
    from quant_fund.cli.main import app

    def name(c) -> str:
        return c.name or c.callback.__name__.replace("_", "-")

    # `dipcatcher` is the outer launcher: its own commands (e.g. `chat`) sit
    # beside the lab app's registered commands.
    roots = frozenset(name(c) for c in app.registered_commands) | LAUNCHER_COMMANDS
    groups: dict[str, frozenset[str]] = {}
    stack = list(app.registered_groups)
    while stack:
        g = stack.pop()
        groups[g.name or g.typer_instance.info.name] = frozenset(
            name(c) for c in g.typer_instance.registered_commands
        )
        stack.extend(g.typer_instance.registered_groups)
    return roots, groups


def _ast_names(module_file: Path) -> frozenset[str]:
    try:
        tree = ast.parse(module_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return frozenset()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    return frozenset(names)


def _module_file_for(spec, tail: str) -> Path | None:
    """Locate ``<pkg>/<tail>.py`` or ``<pkg>/<tail>/__init__.py`` without importing."""
    for loc in spec.submodule_search_locations or []:
        for cand in (Path(loc) / f"{tail}.py", Path(loc) / tail / "__init__.py"):
            if cand.exists():
                return cand
    return None


def _resolve_dotted(ref: str) -> bool:
    parts = ref.split(".")
    spec = None
    depth = 0
    for i in range(len(parts), 0, -1):
        try:
            spec = importlib.util.find_spec(".".join(parts[:i]))
        except Exception:  # noqa: BLE001 - a parent __init__ may raise on optional deps
            spec = None
        if spec is not None:
            depth = i
            break
    if spec is None:
        return False
    if depth == len(parts):
        return True
    mod = None
    # e.g. numba absent in the shared venv on macOS arm64
    with contextlib.suppress(Exception):
        mod = importlib.import_module(".".join(parts[:depth]))
    tail = parts[depth:]
    for j, name in enumerate(tail):
        last = j == len(tail) - 1
        if mod is not None and hasattr(mod, name):
            if last:
                return True
            mod = getattr(mod, name)
            spec = None
            continue
        sub = ".".join(parts[: depth + j + 1])
        try:
            sub_spec = importlib.util.find_spec(sub)
        except Exception:  # noqa: BLE001
            sub_spec = None
        if sub_spec is not None:
            if last:
                return True
            spec = sub_spec
            try:
                mod = importlib.import_module(sub)
            except Exception:  # noqa: BLE001
                mod = None
            continue
        module_file = _module_file_for(spec, name) if spec is not None else None
        if module_file is not None:
            return last or _ast_names(module_file).issuperset(tail[j + 1 :])
        origin = getattr(spec, "origin", None)
        if origin and name in _ast_names(Path(origin)):
            return last
        return False
    return True


def _path_candidates(ref: str, cwd: str | None) -> list[str]:
    cands = [ref, f"src/quant_fund/{ref}", f"src/fx1/{ref}"]
    if cwd:
        cands.insert(1, f"{cwd}/{ref}")
    return cands


def _is_glob(ref: str) -> bool:
    return (
        any(h in ref for h in ("*", "<", ">", "{", "}", "…", "..."))
        or bool(GLOB_SEGMENT.search(ref))
        or ref.endswith(("_", "-", "/"))
    )


def _as_glob(ref: str) -> str:
    pat = re.sub(r"(?:^|/)vN(?=[/.-]|$)", "/*", ref)
    pat = re.sub(r"(?:^|/)NNN(?=[/.-]|$)", "/*", pat)
    pat = pat.replace("YYYY", "*").replace("...", "*").replace("…", "*")
    pat = re.sub(r"<[^>]*>", "*", pat)
    pat = re.sub(r"\{[^}]*\}", "*", pat)
    if pat.endswith(("_", "-", "/")):
        pat += "*"
    return pat


def _path_resolves(ref: str, cwd: str | None) -> bool:
    globby = _is_glob(ref)
    for cand in _path_candidates(ref, cwd):
        if globby:
            pat = _as_glob(cand)
            if any(fnmatch.fnmatch(f, pat) for f in TRACKED_FILES):
                return True
        else:
            c = cand.rstrip("/.")
            if c in TRACKED_FILES or c in TRACKED_DIRS:
                return True
            if c.endswith(".py") and c[:-3] in TRACKED_DIRS:
                return True  # module ref that has since become a package
    return False


def _make_targets() -> frozenset[str]:
    text = (REPO / "Makefile").read_text(encoding="utf-8")
    return frozenset(re.findall(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*:", text, re.M))


def _scan() -> tuple[dict[str, list[str]], frozenset[str]]:
    """(unresolved ref -> locations, all referenced spellings)."""
    misses: dict[str, list[str]] = {}
    refs: set[str] = set()
    roots, groups = _root_commands()
    targets = _make_targets()

    for rel, lineno, region, cwd in _iter_refs():
        loc = f"{rel}:{lineno}"
        for m in SYMBOL_RE.finditer(region):
            ref = m.group(1)
            refs.add(ref)
            if region[m.end() :].startswith("/"):  # namespaced id e.g. fx1.dip_bench/v1
                continue
            if ref.rsplit(".", 1)[-1] in FILE_TAILS:  # file ref, not a symbol
                continue
            if not _resolve_dotted(ref):
                misses.setdefault(ref, []).append(loc)
        toks = _cli_tokens(region)
        if toks is not None:
            cmd = toks[0] if toks else ""
            sub = toks[1] if len(toks) > 1 else ""
            if cmd and CODEWORD.fullmatch(cmd):
                refs.add(f"dipcatcher {cmd}")
                if cmd in groups:
                    if sub and CODEWORD.fullmatch(sub):
                        refs.add(f"dipcatcher {cmd} {sub}")
                        if sub not in groups[cmd]:
                            misses.setdefault(f"dipcatcher {cmd} {sub}", []).append(loc)
                elif cmd not in roots:
                    misses.setdefault(f"dipcatcher {cmd}", []).append(loc)
        for m in PATH_RE.finditer(region):
            ref = m.group(1).rstrip("/")
            refs.add(ref)
            if ref and not _path_resolves(ref, cwd):
                misses.setdefault(ref, []).append(loc)
        for m in MAKE_RE.finditer(region):
            ref = f"make {m.group(1)}"
            refs.add(ref)
            if m.group(1) not in targets and m.group(1) not in MAKE_STOPWORDS:
                misses.setdefault(ref, []).append(loc)
    return misses, frozenset(refs)


def test_docs_code_references_resolve() -> None:
    misses, refs = _scan()
    details = []
    for ref, locs in sorted(misses.items()):
        if ref not in KNOWN_MISSES:
            details.append(f"{ref} <- {locs}")
    for ref in sorted(KNOWN_MISSES):
        if ref not in refs:
            details.append(f"{ref} <- KNOWN_MISSES entry no longer referenced: remove it")
        elif ref not in misses:
            details.append(f"{ref} <- KNOWN_MISSES entry now resolves: remove it")
    assert not details, "docs reference drift:\n" + "\n".join(details)
