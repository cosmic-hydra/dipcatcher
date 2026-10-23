"""AST linter engine (stdlib ast only — no third-party parser).

Implements the LH001..LH014 rule pack (DESIGN.md §6.1 + ADVERSARIAL §1a
hardening). Detection is purely syntactic and deliberately conservative: a
rule fires only on the exact pattern its spec row describes, and per-rule
path allowlists in ``leakage/rules.py`` codify the legitimate sites at HEAD.
A function-scoped allowlist exempts one named function; the rest of that
file is still scanned.

Residual ceiling (documented, ADVERSARIAL §1a): numpy/pandas index arithmetic
(`px[1:] - px[:-1]`), dict lookups at `dates[i+1]`, single numbers far outside
the proximity window, manually computed Sharpe fed to PSR without a
``sharpe_ratio`` call, cross-file helper indirection, and runtime-only
values remain out of reach of a syntactic tripwire. The runtime watchdog
(leakage/watchdog.py) and the proof layer are the barrier; this linter is
the tripwire.
"""

from __future__ import annotations

import ast
import contextvars
import fnmatch
import hashlib
import io
import re
import tokenize
from collections.abc import Callable
from pathlib import Path

from quant_fund.leakage.rules import (
    FUNCTION_ALLOWLISTS,
    LH008_LITERAL_ALLOWLIST,
    LH009_EXEMPT_GLOBS,
    LH011_LAZY_WHITELIST,
    LH011_PACKAGES,
    LH011_WHITELIST,
    RULE_ALLOWLISTS,
)

_PRICE_NAME_RE = re.compile(r"close|mid|price|bars|nav", re.IGNORECASE)
_PRICE_EXACT_NAMES = frozenset({"close", "mid", "price", "bars", "px", "nav"})
_UNIVERSE_NAME_RE = re.compile(r"universe|members|tickers|tape_ids", re.IGNORECASE)
# `delta_*`/`d_*`/`chg_*`/`ret_*`/`r_*` all read as contemporaneous names;
# a forward difference stored under any of them is mislabeled the same way.
_FWD_DIFF_NAME_RE = re.compile(r"^(delta|d_|chg_|ret_|r_)", re.IGNORECASE)
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.]{0,6}$")
# ADVERSARIAL §1a-E16: lowercase ticker tuples evade the uppercase-only
# ticker regex; match case-insensitively. Separators (`-`, `_`, `/`) are
# allowed so exchange-style symbols (BTC-USD, ETH/USDT) cannot launder a
# frozen universe past the shape check.
_TICKER_CI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._/-]{0,9}$")
# ADVERSARIAL §1a-E6: `normalizer`/`preprocessor` attribute names evaded the
# scaler-name regex. `norm` also matches scipy.stats.norm.fit; that receiver
# is excluded in `_check_lh003` because it is a distribution MLE.
_SCALER_NAME_RE = re.compile(r"scal|rank_gauss|norm|preproc", re.IGNORECASE)
_FOLD_FUNC_RE = re.compile(r"fold", re.IGNORECASE)
_PSR_CALL_NAMES = frozenset({"probabilistic_sharpe", "min_track_record_length", "deflated_sharpe"})
_SHARPE_CALL_NAMES = frozenset({"sharpe_ratio"})
_BLOCKED_IO_NAMES = frozenset({"read_parquet", "scan_parquet"})
# ADVERSARIAL §1a-E13: parquet reads hidden inside SQL strings (duckdb.sql).
_SQL_PARQUET_RE = re.compile(r"read_parquet\s*\(|read_csv\s*\(|\.parquet", re.IGNORECASE)


def _concat_literal_parts(node: ast.AST) -> list[str]:
    """String-literal leaves of a ``+`` concat tree (order-independent —
    used only for token matching, not reconstruction)."""
    parts: list[str] = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, ast.BinOp) and isinstance(cur.op, ast.Add):
            stack.extend((cur.left, cur.right))
        elif isinstance(cur, ast.Constant) and isinstance(cur.value, str):
            parts.append(cur.value)
    return parts


def _is_allowlisted(rule_id: str, path_str: str) -> bool:
    globs = RULE_ALLOWLISTS.get(rule_id, frozenset())
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _function_allowlist_names(rule_id: str, path_str: str) -> frozenset[str]:
    """Function names exempt for this rule on this path.

    Keys are repo-relative paths. An absolute scan path matches only when it
    equals the key or ends with ``/`` + key, so the same function name in
    another file is not exempt.
    """
    table = FUNCTION_ALLOWLISTS.get(rule_id)
    if not table:
        return frozenset()
    matched: set[str] = set()
    for rel, names in table.items():
        if path_str == rel or path_str.endswith("/" + rel):
            matched.update(names)
    return frozenset(matched)


def _function_spans(tree: ast.AST) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            spans.append((node.lineno, end, node.name))
    return spans


def _innermost_function_name(spans: list[tuple[int, int, str]], line: int) -> str | None:
    """Name of the tightest function span containing *line*, if any."""
    best_width: int | None = None
    best_name: str | None = None
    for start, end, name in spans:
        if start <= line <= end:
            width = end - start
            if best_width is None or width < best_width:
                best_width = width
                best_name = name
    return best_name


def _drop_function_allowlisted(
    rule_id: str,
    path_str: str,
    tree: ast.AST,
    findings: list[_Finding],
) -> list[_Finding]:
    """Drop findings whose innermost enclosing function is allowlisted.

    Module-level findings and findings in any other function, including a
    nested helper inside an allowlisted function, are kept.
    """
    allowed = _function_allowlist_names(rule_id, path_str)
    if not allowed or not findings:
        return findings
    spans = _function_spans(tree)
    kept: list[_Finding] = []
    for finding in findings:
        name = _innermost_function_name(spans, finding.line)
        if name is not None and name in allowed:
            continue
        kept.append(finding)
    return kept


def _exempt_by_globs(path_str: str, globs: frozenset[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, g) or fnmatch.fnmatch(path_str, f"*/{g}") for g in globs)


def _fold_str(
    node: ast.AST,
    assigns: _ScopeAssigns | None = None,
    use_line: int = 0,
    _depth: int = 0,
) -> str | None:
    """Constant-fold a string expression: literal, ``+``-concatenation of
    literals (ADVERSARIAL §1a-F4/E12 — string concatenation evasion), or a
    name bound to either when *assigns* is given (concat hidden behind a
    variable is the same evasion one hop down)."""
    # Self-referential binds (`x = x + "y"`) make resolve → fold a cycle;
    # cap depth so they fail conservative instead of recursing forever.
    if _depth > 64:
        return None
    if assigns is not None:
        node = _resolve_expr(node, assigns, use_line)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _fold_str(node.left, assigns, use_line, _depth + 1)
        right = _fold_str(node.right, assigns, use_line, _depth + 1)
        if left is not None and right is not None:
            return left + right
    return None


def _fold_num(
    node: ast.AST, assigns: _ScopeAssigns, use_line: int = 0, _depth: int = 0
) -> float | None:
    """Constant-fold a numeric expression: literals, resolved names, unary
    +/- and ``+ - *`` over foldable operands. ``None`` = not provably
    constant (conservatively clean)."""
    # Same self-referential-bind cycle guard as _fold_str.
    if _depth > 64:
        return None
    resolved = _resolve_expr(node, assigns, use_line)
    if (
        isinstance(resolved, ast.Constant)
        and isinstance(resolved.value, (int, float))
        and not isinstance(resolved.value, bool)
    ):
        return float(resolved.value)
    if isinstance(resolved, ast.UnaryOp):
        operand = _fold_num(resolved.operand, assigns, use_line, _depth + 1)
        if operand is None:
            return None
        if isinstance(resolved.op, ast.USub):
            return -operand
        if isinstance(resolved.op, ast.UAdd):
            return operand
        return None
    if isinstance(resolved, ast.BinOp):
        left = _fold_num(resolved.left, assigns, use_line, _depth + 1)
        right = _fold_num(resolved.right, assigns, use_line, _depth + 1)
        if left is None or right is None:
            return None
        if isinstance(resolved.op, ast.Add):
            return left + right
        if isinstance(resolved.op, ast.Sub):
            return left - right
        if isinstance(resolved.op, ast.Mult):
            return left * right
    return None


def _call_receiver_price_like(receiver: ast.Call) -> bool:
    """Price-like check for a call used as a ``.shift`` receiver."""
    # pl.col("close").shift(-1) — inspect the column-name literal.
    func = receiver.func
    func_name = ""
    if isinstance(func, ast.Attribute):
        func_name = func.attr
    elif isinstance(func, ast.Name):
        func_name = func.id
    if func_name == "col" and receiver.args:
        first = receiver.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            return bool(_PRICE_NAME_RE.search(first.value))
        if isinstance(first, ast.Name):
            name = first.id
            return name.lower() in _PRICE_EXACT_NAMES or bool(_PRICE_NAME_RE.search(name))
    # ADVERSARIAL §1a-E4: a negative shift on a rolling statistic
    # (`rolling_mean(21).shift(k)`, k negative) centers the window on the
    # future regardless of the inner receiver's name.
    if func_name.startswith("rolling"):
        return True
    # Chained transforms still read the underlying series:
    # `close.fill_null(0).shift(-1)` (method receiver) and
    # `np.log(close).shift(-1)` (function argument) were silent misses.
    if _chained_method_receiver_price_like(func):
        return True
    return any(_receiver_price_like(arg) for arg in receiver.args)


def _receiver_price_like(receiver: ast.AST) -> bool:
    """True if the `.shift` receiver looks price-like (close/mid/px/...)."""
    if isinstance(receiver, ast.Name):
        name = receiver.id
        return name.lower() in _PRICE_EXACT_NAMES or bool(_PRICE_NAME_RE.search(name))
    if isinstance(receiver, ast.Attribute):
        return bool(_PRICE_NAME_RE.search(receiver.attr))
    if isinstance(receiver, ast.Subscript):
        key = receiver.slice
        return (
            isinstance(key, ast.Constant)
            and isinstance(key.value, str)
            and bool(_PRICE_NAME_RE.search(key.value))
        )
    if isinstance(receiver, ast.Call):
        return _call_receiver_price_like(receiver)
    return False


def _chained_method_receiver_price_like(func: ast.expr) -> bool:
    """True when a method call's own receiver chain is price-like."""
    return isinstance(func, ast.Attribute) and _receiver_price_like(func.value)


def _shift_call_parts(node: ast.AST) -> tuple[ast.AST, ast.AST] | None:
    """``(receiver, period_expr)`` for a shift-shaped call, else ``None``.

    Covers ``x.shift(k)``, ``getattr(x, "shift")(k)`` (dynamic dispatch is
    the same read one hop down), and bare functional ``shift(x, k)``.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "shift":
        period = _shift_period_expr(node)
        return (func.value, period) if period is not None else None
    if (
        isinstance(func, ast.Call)
        and _call_name(func) == "getattr"
        and len(func.args) >= 2
        and _fold_str(func.args[1]) == "shift"
    ):
        period = node.args[0] if node.args else _shift_period_expr(node)
        return (func.args[0], period) if period is not None else None
    if isinstance(func, ast.Name) and func.id == "shift" and len(node.args) >= 2:
        return (node.args[0], node.args[1])
    return None


def _contains_negative_shift(node: ast.AST, assigns: _ScopeAssigns) -> bool:
    """True when the expression tree contains any provably-negative shift."""
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        parts = _shift_call_parts(sub)
        if parts is None:
            continue
        _, period = parts
        if _is_negative_shift_period(period, assigns, sub.lineno):
            return True
    return False


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _build_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    parents: dict[int, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[id(child)] = node
    return parents


def _enclosing_scopes(
    node: ast.AST, parents: dict[int, ast.AST]
) -> tuple[list[ast.For | ast.AsyncFor], list[ast.FunctionDef | ast.AsyncFunctionDef]]:
    fors: list[ast.For | ast.AsyncFor] = []
    funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    cur = parents.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.For, ast.AsyncFor)):
            fors.append(cur)
        elif isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(cur)
        cur = parents.get(id(cur))
    return fors, funcs


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    """ids of string Constants used as module/class/function docstrings."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (
            isinstance(body, list) and body
        ):
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


class _Finding:
    __slots__ = ("rule_id", "line", "col", "message")

    def __init__(self, rule_id: str, line: int, col: int, message: str) -> None:
        self.rule_id = rule_id
        self.line = line
        self.col = col
        self.message = message


def _shift_period_expr(call: ast.Call) -> ast.AST | None:
    """The period argument of a ``.shift`` call: positional or ``periods=``
    keyword form (ADVERSARIAL §1a-E3)."""
    if call.args:
        return call.args[0]
    for kw in call.keywords:
        if kw.arg == "periods":
            return kw.value
    return None


def _is_negative_shift_period(expr: ast.AST, assigns: _ScopeAssigns, use_line: int = 0) -> bool:
    """True if the shift period provably folds to a negative number.

    Handles literals (``-1``, ``-1.0``), constant arithmetic (``0 - 1``), and
    assignment-bound names (``k = -HALF``, ``h = -horizon`` — ADVERSARIAL
    §1a-E4/E14) via scope-level constant resolution. Unresolvable
    expressions are conservatively clean.
    """
    folded = _fold_num(expr, assigns, use_line)
    return folded is not None and folded < 0


def _merged_assigns(tree: ast.AST, scope: ast.AST) -> dict[str, list[tuple[int, ast.AST]]]:
    """Module-level + enclosing-scope assignment candidates.

    Module constants (``HALF = 10``) must resolve inside functions; scope
    names are appended so use-line-aware resolution picks the live binding.
    Copies the cached lists: the cache itself must stay immutable.
    """
    assigns = {name: list(entries) for name, entries in _scope_assignments(tree).items()}
    for name, entries in _scope_assignments(scope).items():
        assigns.setdefault(name, []).extend(entries)
    return assigns


def _check_lh001(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        parts = _shift_call_parts(node)
        if parts is None:
            continue
        receiver, period = parts
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _merged_assigns(tree, scope)
        if _is_negative_shift_period(period, assigns, node.lineno) and _receiver_price_like(
            receiver
        ):
            out.append(
                _Finding(
                    "LH001",
                    node.lineno,
                    node.col_offset,
                    "negative shift on a price-like series reads future bars",
                )
            )
    return out


def _check_lh002(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if not name.startswith("rolling"):
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _merged_assigns(tree, scope)
        for kw in node.keywords:
            if kw.arg != "center":
                continue
            # Any provably-truthy constant centers the window — `center=c`
            # with c=True and `center=1` evaded the literal-`True` check.
            resolved = _resolve_expr(kw.value, assigns, node.lineno)
            if isinstance(resolved, ast.Constant) and bool(resolved.value):
                out.append(
                    _Finding(
                        "LH002",
                        node.lineno,
                        node.col_offset,
                        "centered rolling window mixes future rows into the window",
                    )
                )
    return out


def _fit_consumes_fold_param(node: ast.Call, func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if any fit argument (positional or keyword) references one of
    *func*'s parameters (i.e. the caller sliced the fold; ADVERSARIAL §1a-E7
    — a function merely NAMED `*fold*` that fits a global/frame attribute is
    not a fold scope).
    """
    if not node.args and not node.keywords:
        return False
    params = {a.arg for a in (*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs)}
    names = {
        n.id
        for arg in (*node.args, *(kw.value for kw in node.keywords))
        for n in ast.walk(arg)
        if isinstance(n, ast.Name)
    }
    return bool(names & params)


def _is_scipy_stats_distribution(receiver: ast.AST) -> bool:
    """True for ``stats.norm`` / ``scipy.stats.lognorm`` (distribution MLE, not a scaler)."""
    if not isinstance(receiver, ast.Attribute):
        return False
    base = receiver.value
    if isinstance(base, ast.Name) and base.id == "stats":
        return True
    return (
        isinstance(base, ast.Attribute)
        and base.attr == "stats"
        and isinstance(base.value, ast.Name)
        and base.value.id == "scipy"
    )


_FIT_ATTRS = frozenset({"fit", "fit_transform", "partial_fit"})


def _scipy_stats_bound_names(tree: ast.AST) -> set[str]:
    """Local names bound by ``from scipy.stats import ...`` — a `norm` from
    there is a distribution MLE (like `stats.norm.fit`), not a scaler."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "scipy.stats":
            bound.update(alias.asname or alias.name for alias in node.names)
    return bound


def _check_lh003(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    scipy_stats_names = _scipy_stats_bound_names(tree)
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _FIT_ATTRS
        ):
            continue
        attr = node.func.attr
        receiver = node.func.value
        recv_name = ""
        if isinstance(receiver, ast.Name):
            recv_name = receiver.id
        elif isinstance(receiver, ast.Attribute):
            recv_name = receiver.attr
        elif isinstance(receiver, ast.Call):
            # `StandardScaler().fit(X)` / `make_scaler().fit(X)` — the
            # constructor callee carries the scaler semantics an anonymous
            # receiver used to hide.
            ctor = receiver.func
            if isinstance(ctor, ast.Attribute):
                recv_name = ctor.attr
            elif isinstance(ctor, ast.Name):
                recv_name = ctor.id
        if not _SCALER_NAME_RE.search(recv_name):
            continue
        # ``scal`` also matches the Scaled*Distribution/Scaled*Tail MLE
        # family — parametric distribution fits (same class as
        # scipy.stats.norm.fit), not feature scalers.
        if re.search(r"(Distribution|Tail)$", recv_name):
            continue
        # ``norm`` in the scaler pattern also matches scipy.stats.norm.fit.
        if not re.search(r"scal|rank_gauss|preproc", recv_name, re.IGNORECASE) and (
            _is_scipy_stats_distribution(receiver) or recv_name in scipy_stats_names
        ):
            continue
        fors, funcs = _enclosing_scopes(node, parents)
        in_fold_loop = any(
            re.search(
                r"fold|split", ast.unparse(f.target) + " " + ast.unparse(f.iter), re.IGNORECASE
            )
            for f in fors
        )
        # A fold-NAMED function is a safe scope only if the fit consumes one
        # of the function's own parameters (caller-sliced fold data).
        in_fold_func = any(
            _FOLD_FUNC_RE.search(f.name) and _fit_consumes_fold_param(node, f) for f in funcs
        )
        if not in_fold_loop and not in_fold_func:
            out.append(
                _Finding(
                    "LH003",
                    node.lineno,
                    node.col_offset,
                    f"scaler-like `{recv_name}.{attr}` is not lexically inside a fold/split scope",
                )
            )
    return out


def _check_lh004(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) == "join_asof"):
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _merged_assigns(tree, scope)
        for kw in node.keywords:
            # `right_on` keys the joined frame the same way `left_on`/`on`
            # key the grid — checking only the left side failed open.
            if kw.arg not in ("on", "left_on", "right_on"):
                continue
            # ADVERSARIAL §1a-E17: the keyword value may hide behind a
            # single-assignment name (`key = "event_time"`). Case and
            # surrounding words don't change the semantics (`Event_Time`,
            # `my_event_time`, `adjusted_event_time`).
            folded = _fold_str(kw.value, assigns, node.lineno)
            if folded is not None and "event_time" in folded.casefold():
                out.append(
                    _Finding(
                        "LH004",
                        node.lineno,
                        node.col_offset,
                        "join_asof keyed on event_time; PIT joins must key on known_at/available_time",
                    )
                )
    return out


_CONFIG_STEM_RE = re.compile(r"(^|[^a-z0-9])configs?([^a-z0-9]|$)")


def _universe_elts(value: ast.AST) -> list[ast.AST] | None:
    """Candidate universe members: list/tuple/set literals, dict keys, and
    single-argument container calls (``list([...])``, ``sorted((...))``)."""
    if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        return list(value.elts)
    if isinstance(value, ast.Dict):
        return [k for k in value.keys if k is not None]
    if (
        isinstance(value, ast.Call)
        and _call_name(value) in ("list", "tuple", "set", "frozenset", "sorted")
        and len(value.args) == 1
    ):
        return _universe_elts(value.args[0])
    return None


def _check_lh005(tree: ast.AST, path_str: str) -> list[_Finding]:
    # tests/ and config files may carry fake ticker lists — EXCEPT the
    # seeded-leak fixture suite, which must stay scannable (DESIGN.md §6.4).
    # Path-component match, not substring: a `mytests/` or `latests/`
    # directory (or a `myconfig.py`/`reconfigure.py` file) is not exempt.
    stem = Path(path_str).stem.lower()
    if "leakage_fixtures" not in path_str and (
        "tests" in Path(path_str).parts or _CONFIG_STEM_RE.search(stem)
    ):
        return []
    out: list[_Finding] = []
    for node in ast.walk(tree):
        target: ast.AST | None = None
        value: ast.AST | None = None
        line = col = 0
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
            line, col = node.lineno, node.col_offset
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
            line, col = node.lineno, node.col_offset
        if not (isinstance(target, ast.Name) and _UNIVERSE_NAME_RE.search(target.id)):
            continue
        # ADVERSARIAL §1a-E16: tuples and lowercase tickers are the same
        # leak; so are sets, dict-of-weights keys, and `list(...)` wrappers.
        elts = _universe_elts(value) if value is not None else None
        if elts is None or len(elts) < 5:
            continue
        if all(
            isinstance(e, ast.Constant)
            and isinstance(e.value, str)
            and _TICKER_CI_RE.match(e.value)
            for e in elts
        ):
            out.append(
                _Finding(
                    "LH005",
                    line,
                    col,
                    f"frozen ticker list assigned to `{target.id}` (survivorship-biased universe)",
                )
            )
    return out


def _fwd_diff_value(value: ast.AST, assigns: _ScopeAssigns) -> bool:
    """True when the value expression reads a forward bar — `X.shift(-k)`
    anywhere in the tree (difference, ratio, or bare shift), with the period
    resolved through scope assignments like LH001."""
    return _contains_negative_shift(value, assigns)


def _contemp_target_names(target: ast.AST) -> list[str]:
    """Contemporaneous names a value is stored under: plain names, subscript
    keys (``frame['delta_mid'] = ...``), and destructured elements."""
    if isinstance(target, ast.Name):
        return [target.id]
    if (
        isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and isinstance(target.slice.value, str)
    ):
        return [target.slice.value]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _contemp_target_names(elt)]
    return []


def _check_lh006(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        # (a) ``delta_mid = mid.shift(-1) - mid`` assignment form (also
        # annotated targets, subscript keys, and tuple destructuring).
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            _, funcs = _enclosing_scopes(node, parents)
            scope: ast.AST = funcs[-1] if funcs else tree
            assigns = _merged_assigns(tree, scope)
            if _fwd_diff_value(node.value, assigns):
                targets: list[ast.AST] = (
                    list(node.targets) if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    for name in _contemp_target_names(target):
                        if _FWD_DIFF_NAME_RE.match(name) and not re.match(r"(?i)^(fwd|lead)", name):
                            out.append(
                                _Finding(
                                    "LH006",
                                    node.lineno,
                                    node.col_offset,
                                    f"forward difference assigned to contemporaneous name `{name}`",
                                )
                            )
        # (b) polars expression form ``(...shift(-1) - ...).alias("delta_mid")``.
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "alias"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            _, funcs = _enclosing_scopes(node, parents)
            scope = funcs[-1] if funcs else tree
            assigns = _merged_assigns(tree, scope)
            name = node.args[0].value
            if (
                _FWD_DIFF_NAME_RE.match(name)
                and not re.match(r"(?i)^(fwd|lead)", name)
                and _fwd_diff_value(node.func.value, assigns)
            ):
                out.append(
                    _Finding(
                        "LH006",
                        node.lineno,
                        node.col_offset,
                        f"forward difference aliased to contemporaneous name `{name}`",
                    )
                )
    return out


def _sharpe_call_unit_safe(
    call: ast.Call, assigns: _ScopeAssigns, use_line: int = 0
) -> bool | None:
    """True if a sharpe_ratio call is provably per-period; None if not a sharpe call."""
    if _call_name(call) not in _SHARPE_CALL_NAMES:
        return None
    for kw in call.keywords:
        resolved = _resolve_expr(kw.value, assigns, use_line)
        if (
            kw.arg == "periods_per_year"
            and isinstance(resolved, ast.Constant)
            and resolved.value in (1, 1.0)
        ):
            return True
        if kw.arg == "irregular" and isinstance(resolved, ast.Constant) and resolved.value is True:
            return True
    return False


# name -> [(lineno, value-expr)] candidates in a scope subtree. Several
# candidates per name are kept: resolution picks the one nearest before the
# use site, so reassignment (``k = 1; k = -1; shift(k)``) cannot launder a
# negative period and unannotated/destructured/walrus bindings cannot hide
# one either.
_ScopeAssigns = dict[str, list[tuple[int, ast.AST]]]
# Per-scan cache. AST node ids are only unique while that tree is alive, so
# scan_file installs a fresh map and drops it before the tree can be freed.
_SCOPE_CACHE: contextvars.ContextVar[dict[int, _ScopeAssigns] | None] = contextvars.ContextVar(
    "dipcatcher_leakage_scope_cache", default=None
)


def _iter_binds(node: ast.AST) -> list[tuple[str, ast.expr]]:
    """(name, value) pairs bound by *node*: ``x = v``, ``x: T = v``,
    ``x := v``, and flat tuple destructuring ``a, b = v, w``."""
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            return []
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return [(target.id, node.value)]
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(node.value, (ast.Tuple, ast.List))
            and len(target.elts) == len(node.value.elts)
            and all(not isinstance(e, ast.Starred) for e in target.elts)
        ):
            return [
                (t.id, v)
                for t, v in zip(target.elts, node.value.elts, strict=True)
                if isinstance(t, ast.Name)
            ]
        return []
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return [(node.target.id, node.value)]
    if isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
        return [(node.target.id, node.value)]
    return []


def _scope_assignments(func: ast.AST) -> _ScopeAssigns:
    cache = _SCOPE_CACHE.get()
    if cache is not None:
        cached = cache.get(id(func))
        if cached is not None:
            return cached
    assigns: _ScopeAssigns = {}
    for node in ast.walk(func):
        binds = _iter_binds(node)
        if not binds:
            continue
        line = node.lineno if isinstance(node, (ast.stmt, ast.expr)) else 0
        for name, value in binds:
            assigns.setdefault(name, []).append((line, value))
    if cache is not None:
        cache[id(func)] = assigns
    return assigns


def _resolve_expr(
    expr: ast.AST, assigns: _ScopeAssigns, use_line: int = 0, depth: int = 0
) -> ast.AST:
    """Follow name -> assigned expr, unwrapping float(...) and subscripts.

    The candidate assignment nearest BEFORE *use_line* wins — that is the
    binding live at the use site. When nothing precedes it, fall back to
    the earliest candidate (a use before assignment cannot run anyway).
    """
    if depth > 8:
        return expr
    if isinstance(expr, ast.Name) and expr.id in assigns:
        candidates = assigns[expr.id]
        before = [c for c in candidates if c[0] < use_line]
        chosen = (
            max(before, key=lambda c: c[0]) if before else min(candidates, key=lambda c: c[0])
        )[1]
        return _resolve_expr(chosen, assigns, use_line, depth + 1)
    if isinstance(expr, ast.Call) and _call_name(expr) in ("float", "int") and expr.args:
        return _resolve_expr(expr.args[0], assigns, use_line, depth + 1)
    if isinstance(expr, ast.Subscript):
        return _resolve_expr(expr.value, assigns, use_line, depth + 1)
    return expr


def _check_lh007(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) in _PSR_CALL_NAMES):
            continue
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _merged_assigns(tree, scope)
        # The sr argument arrives positionally (psr(sr, ...)) or by keyword
        # (psr(sr=..., sharpe=...)) — checking only args[0] failed open.
        exprs: list[ast.AST] = list(node.args[:1])
        exprs.extend(kw.value for kw in node.keywords if kw.arg in ("sr", "sharpe"))
        for expr in exprs:
            resolved = _resolve_expr(expr, assigns, node.lineno)
            if isinstance(resolved, ast.Call):
                safe = _sharpe_call_unit_safe(resolved, assigns, node.lineno)
                if safe is False:
                    out.append(
                        _Finding(
                            "LH007",
                            node.lineno,
                            node.col_offset,
                            f"{_call_name(node)} fed a sharpe_ratio result that is not "
                            "provably per-period (needs an explicit per-period convention)",
                        )
                    )
                    break
    return out


def _lh008_allowed(path_str: str, value: str) -> bool:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return any(
        (path_str == path or path_str.endswith(f"/{path}")) and digest in digests
        for path, digests in LH008_LITERAL_ALLOWLIST.items()
    )


def _parent_is_add_binop(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    parent = parents.get(id(node))
    return isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add)


def _headline_literal(value: str) -> bool:
    """False for identifier-like strings (dict keys such as ``"sharpe"`` or
    ``"sharpe_p05"`` are names, not rendered headline claims — scanning them
    spammed every percentile/metric key in analytics output)."""
    return not value.isidentifier()


def _check_lh008(tree: ast.AST, path_str: str) -> list[_Finding]:
    from quant_fund.leakage.patterns import find_forbidden_headline

    out: list[_Finding] = []
    docstrings = _docstring_constant_ids(tree)
    parents = _build_parent_map(tree)
    module_assigns = _scope_assignments(tree)
    for node in ast.walk(tree):
        value: str | None = None
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and len(node.value) > 0
            # Skip bare identifier keys while still checking compact output
            # labels such as "Sharpe:2.1" and "P&L=$4,200".
            and _headline_literal(node.value)
        ):
            value = node.value
        elif (
            # ADVERSARIAL §1a-F4: constant-fold "a" + "b" string concatenation
            # (top-most Add node only; nested parts are visited as Constants).
            # A concat whose parent is a *non-Add* BinOp is still the
            # top-most foldable concat and must be checked.
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and not _parent_is_add_binop(node, parents)
        ):
            folded = _fold_str(node, module_assigns, node.lineno)
            if folded and _headline_literal(folded):
                value = folded
        if value is None or not isinstance(node, ast.expr):
            continue
        tokens = find_forbidden_headline(value)
        if tokens and not _lh008_allowed(path_str, value):
            out.append(
                _Finding(
                    "LH008",
                    node.lineno,
                    node.col_offset,
                    f"string literal headlines forbidden metric token(s): {', '.join(tokens)}",
                )
            )
    return out


def _blocked_io_aliases(tree: ast.AST) -> set[str]:
    """Local names bound to parquet readers via ``from X import read_parquet
    [as rp]`` — the aliased call `rp(path)` is the same bypass one hop down."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _BLOCKED_IO_NAMES:
                    bound.add(alias.asname or alias.name)
    return bound


# SQL-capable entry points whose string argument can embed a parquet read.
_SQL_ENTRY_NAMES = frozenset({"sql", "execute", "read_sql", "read_sql_query"})
# String-evaluating builtins: `eval("pl.read_parquet(p)")` executes the
# bypass without ever naming it in the AST.
_EVAL_NAMES = frozenset({"eval", "exec", "compile"})


def _lh009_getattr_findings(node: ast.Call, module_assigns: _ScopeAssigns) -> list[_Finding]:
    """ADVERSARIAL §1a-E12: findings for ``getattr(pl, "read_" + "parquet")(path)``."""
    attr = _fold_str(node.args[1], module_assigns, node.lineno)
    if attr is not None and attr in _BLOCKED_IO_NAMES:
        return [
            _Finding(
                "LH009",
                node.lineno,
                node.col_offset,
                f"dynamic getattr(.., {attr!r}) parquet read bypasses the PIT choke point",
            )
        ]
    return []


def _check_lh009(tree: ast.AST, path_str: str) -> list[_Finding]:
    # tests/** is exempt EXCEPT the seeded-leak fixture suite, which must stay
    # scannable (same carve-out as LH005, DESIGN.md §6.4).
    if "leakage_fixtures" not in path_str and _exempt_by_globs(path_str, LH009_EXEMPT_GLOBS):
        return []
    out: list[_Finding] = []
    module_assigns = _scope_assignments(tree)
    io_aliases = _blocked_io_aliases(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in _BLOCKED_IO_NAMES or name in io_aliases:
            out.append(
                _Finding(
                    "LH009",
                    node.lineno,
                    node.col_offset,
                    "direct parquet read outside the data layer bypasses the PIT choke point",
                )
            )
        elif name == "history":
            out.append(
                _Finding(
                    "LH009",
                    node.lineno,
                    node.col_offset,
                    "PitVault.history() returns every version including future "
                    "known_at — audit-only API must not appear in strategy code paths",
                )
            )
        # ADVERSARIAL §1a-E12: getattr(pl, "read_" + "parquet")(path).
        elif name == "getattr" and len(node.args) >= 2:
            out.extend(_lh009_getattr_findings(node, module_assigns))
        # ADVERSARIAL §1a-E13: duckdb.sql("select * from read_parquet(...)"),
        # conn.execute(...), eval/exec of a generated call string.
        elif name in _SQL_ENTRY_NAMES or name in _EVAL_NAMES:
            for arg in node.args:
                folded = _fold_str(arg, module_assigns, node.lineno)
                if folded is not None and _SQL_PARQUET_RE.search(folded):
                    out.append(
                        _Finding(
                            "LH009",
                            node.lineno,
                            node.col_offset,
                            f"parquet read embedded in a {name}(...) string bypasses the "
                            "PIT choke point",
                        )
                    )
                    break
    return out


def _check_lh010(tree: ast.AST, path_str: str) -> list[_Finding]:
    out: list[_Finding] = []
    parents = _build_parent_map(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        _, funcs = _enclosing_scopes(node, parents)
        scope: ast.AST = funcs[-1] if funcs else tree
        assigns = _merged_assigns(tree, scope)
        is_bfill = name == "bfill"
        # strategy= may be bound to a variable; pandas fillna(method=...)
        # is the same backward fill.
        resolved_kws = {kw.arg: _fold_str(kw.value, assigns, node.lineno) for kw in node.keywords}
        is_backward_fill = name == "fill_null" and resolved_kws.get("strategy") == "backward"
        is_pandas_bfill = name == "fillna" and resolved_kws.get("method") in (
            "bfill",
            "backfill",
        )
        if not (is_bfill or is_backward_fill or is_pandas_bfill):
            continue
        try:
            segment = ast.unparse(scope)
        except Exception:
            segment = ""
        if '"event_time"' in segment or "'event_time'" in segment or "event_time" in segment:
            out.append(
                _Finding(
                    "LH010",
                    node.lineno,
                    node.col_offset,
                    "backward fill on a frame with an event_time column in scope",
                )
            )
    return out


def _lh011_package(path_str: str) -> str | None:
    """PROOFCORE package name if path is inside one, else None."""
    parts = Path(path_str).parts
    for i, part in enumerate(parts):
        if part == "quant_fund" and i + 1 < len(parts) and parts[i + 1] in LH011_PACKAGES:
            return parts[i + 1]
    return None


def _lh011_imported_modules(node: ast.Import | ast.ImportFrom, path_str: str) -> list[str]:
    """Module names an import statement binds, including relative forms."""
    if isinstance(node, ast.ImportFrom):
        if node.level:
            parts = Path(path_str).parts
            quant_index = parts.index("quant_fund")
            package_parts = list(parts[quant_index:-1])
            if node.level > len(package_parts):
                return ["quant_fund.__invalid_relative_import__"]
            base = package_parts[: len(package_parts) - node.level + 1]
            return [".".join([*base, *([node.module] if node.module else [])])]
        if node.module == "quant_fund":
            # `from quant_fund import backtest` — the bare root prefix
            # let a top-level sibling-package edge slip past
            # startswith("quant_fund.").
            return [f"quant_fund.{alias.name}" for alias in node.names]
        if node.module:
            return [node.module]
        return []
    return [alias.name for alias in node.names]


def _lh011_lazy(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """True when *node* sits inside a function (lazy-import allowance)."""
    _, funcs = _enclosing_scopes(node, parents)
    return bool(funcs)


def _lh011_dynamic_findings(
    tree: ast.AST,
    package: str,
    whitelist: frozenset[str],
    lazy_whitelist: frozenset[str],
    parents: dict[int, ast.AST],
) -> list[_Finding]:
    """Same whitelist check for ``import_module`` / ``__import__`` calls.

    Dynamic imports (``importlib.import_module("quant_fund.backtest")``,
    ``__import__(...)``) are the same edge smuggled through a call — a
    provably-constant target gets the same whitelist check, at the same
    function-level lazy allowance.
    """
    out: list[_Finding] = []
    module_assigns = _scope_assignments(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _call_name(node) in ("import_module", "__import__")):
            continue
        if not node.args:
            continue
        target = _fold_str(node.args[0], module_assigns, node.lineno)
        if target is None or not target.startswith("quant_fund."):
            continue
        sub = target.split(".")[1]
        if sub == package:
            continue
        nested = _lh011_lazy(node, parents)
        allowed = whitelist | (lazy_whitelist if nested else frozenset())
        if sub not in allowed:
            out.append(
                _Finding(
                    "LH011",
                    node.lineno,
                    node.col_offset,
                    f"quant_fund.{package} dynamically imports non-whitelisted "
                    f"quant_fund.{sub} ({'lazy ' if nested else ''}import; "
                    f"whitelist: {sorted(allowed) or 'none'})",
                )
            )
    return out


def _check_lh011(tree: ast.AST, path_str: str) -> list[_Finding]:
    package = _lh011_package(path_str)
    if package is None:
        return []
    whitelist = LH011_WHITELIST[package]
    lazy_whitelist = LH011_LAZY_WHITELIST[package]
    parents = _build_parent_map(tree)
    out: list[_Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        modules = _lh011_imported_modules(node, path_str)
        nested = _lh011_lazy(node, parents)
        allowed = whitelist | (lazy_whitelist if nested else frozenset())
        for module in modules:
            if not module.startswith("quant_fund."):
                continue
            sub = module.split(".")[1]
            if sub == package:
                continue  # intra-package imports are always fine
            if sub not in allowed:
                out.append(
                    _Finding(
                        "LH011",
                        node.lineno,
                        node.col_offset,
                        f"quant_fund.{package} imports non-whitelisted quant_fund.{sub} "
                        f"({'lazy ' if nested else ''}import; whitelist: {sorted(allowed) or 'none'})",
                    )
                )
    out.extend(_lh011_dynamic_findings(tree, package, whitelist, lazy_whitelist, parents))
    return out


class _Lh013Patterns:
    """Pattern matchers bound once per LH013 scan, not once per syntax node."""

    __slots__ = ("headline", "mentions", "spelled")

    def __init__(
        self,
        headline: Callable[[str], list[str]],
        mentions: Callable[[str], list[str]],
        spelled: Callable[[str], list[str]],
    ) -> None:
        self.headline = headline
        self.mentions = mentions
        self.spelled = spelled


def _lh013_patterns() -> _Lh013Patterns:
    from quant_fund.leakage.patterns import (
        find_forbidden_headline,
        find_forbidden_token_mentions,
        find_spelled_out_headline,
    )

    return _Lh013Patterns(
        find_forbidden_headline,
        find_forbidden_token_mentions,
        find_spelled_out_headline,
    )


def _lh013_emit(out: list[_Finding], line: int, col: int, tokens: list[str], channel: str) -> None:
    out.append(
        _Finding(
            "LH013",
            line,
            col,
            f"{channel} headlines forbidden metric token(s): {', '.join(tokens)}",
        )
    )


def _lh013_comment_findings(
    source: str, path_str: str, out: list[_Finding], patterns: _Lh013Patterns
) -> None:
    """Flag forbidden headline tokens in comments (invisible to the AST)."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type != tokenize.COMMENT:
                continue
            text = tok.string.lstrip("#").strip()
            tokens = patterns.headline(text) or patterns.spelled(text)
            if tokens and not _lh008_allowed(path_str, text):
                _lh013_emit(out, tok.start[0], tok.start[1], tokens, "comment")
    except tokenize.TokenError:
        pass


def _lh013_constant_findings(
    node: ast.AST,
    path_str: str,
    docstrings: set[int],
    out: list[_Finding],
    patterns: _Lh013Patterns,
) -> None:
    """Docstrings and spelled-out numerals in other string literals."""
    if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
        return
    if id(node) in docstrings:
        tokens = patterns.headline(node.value) or patterns.spelled(node.value)
        if tokens and not _lh008_allowed(path_str, node.value):
            _lh013_emit(out, node.lineno, node.col_offset, tokens, "docstring")
        return
    # Spelled-out numerals in non-docstring string literals (digit forms
    # stay with LH008 at error severity).
    if node.value and _headline_literal(node.value) and not patterns.headline(node.value):
        tokens = patterns.spelled(node.value)
        if tokens and not _lh008_allowed(path_str, node.value):
            _lh013_emit(
                out,
                node.lineno,
                node.col_offset,
                tokens,
                "string literal (spelled-out numeral)",
            )


def _lh013_fstring_findings(
    node: ast.AST, path_str: str, out: list[_Finding], patterns: _Lh013Patterns
) -> None:
    """f-string templates whose number arrives at runtime."""
    if not isinstance(node, ast.JoinedStr):
        return
    literal = "".join(
        part.value
        for part in node.values
        if isinstance(part, ast.Constant) and isinstance(part.value, str)
    )
    if not literal or not patterns.mentions(literal):
        return
    has_runtime_value = any(isinstance(part, ast.FormattedValue) for part in node.values)
    tokens = (
        patterns.headline(literal)
        or (patterns.mentions(literal) if has_runtime_value else [])
        or patterns.spelled(literal)
    )
    if tokens and not _lh008_allowed(path_str, literal):
        _lh013_emit(out, node.lineno, node.col_offset, tokens, "f-string template")


def _lh013_format_findings(
    node: ast.AST,
    path_str: str,
    module_assigns: _ScopeAssigns,
    out: list[_Finding],
    patterns: _Lh013Patterns,
) -> None:
    """``"Sharpe %s" % sr`` and ``"Sharpe {}".format(sr)`` templates."""
    template: str | None = None
    channel = ""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
        template = _fold_str(node.left, module_assigns, node.lineno)
        channel = "percent-format template"
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("format", "format_map")
    ):
        template = _fold_str(node.func.value, module_assigns, node.lineno)
        channel = "str.format template"
    else:
        return
    if template is None:
        return
    tokens = (
        patterns.headline(template) or patterns.mentions(template) or patterns.spelled(template)
    )
    if tokens and not _lh008_allowed(path_str, template):
        _lh013_emit(out, node.lineno, node.col_offset, tokens, channel)


def _lh013_concat_findings(
    node: ast.AST,
    path_str: str,
    module_assigns: _ScopeAssigns,
    parents: dict[int, ast.AST],
    out: list[_Finding],
    patterns: _Lh013Patterns,
) -> None:
    """``"Sharpe " + str(sr)``: a concat that does not fully constant-fold.

    Fully-foldable concats are LH008's job (error severity).
    """
    if not (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and not _parent_is_add_binop(node, parents)
        and _fold_str(node, module_assigns, node.lineno) is None
    ):
        return
    literal = "".join(_concat_literal_parts(node))
    if not literal:
        return
    tokens = patterns.headline(literal) or patterns.mentions(literal) or patterns.spelled(literal)
    if tokens and not _lh008_allowed(path_str, literal):
        _lh013_emit(out, node.lineno, node.col_offset, tokens, "string concat template")


def _check_lh013(tree: ast.AST, path_str: str, source: str) -> list[_Finding]:
    """Warning-severity headline channels (ADVERSARIAL §1a E8/E9/E10/F2/F6):

    - COMMENTS via tokenize (comments are invisible to the AST);
    - DOCSTRINGS (LH008 excludes them, but they render into published docs);
    - f-string templates whose number arrives at runtime
      (``f"Sharpe was {sr:.2f}"`` — a forbidden token plus a numeric format
      spec or spelled-out numeral);
    - spelled-out numerals in any string channel
      (``"Sharpe, which exceeded two"``).
    """
    out: list[_Finding] = []
    patterns = _lh013_patterns()
    _lh013_comment_findings(source, path_str, out, patterns)
    docstrings = _docstring_constant_ids(tree)
    parents = _build_parent_map(tree)
    module_assigns = _scope_assignments(tree)
    for node in ast.walk(tree):
        _lh013_constant_findings(node, path_str, docstrings, out, patterns)
        _lh013_fstring_findings(node, path_str, out, patterns)
        _lh013_format_findings(node, path_str, module_assigns, out, patterns)
        _lh013_concat_findings(node, path_str, module_assigns, parents, out, patterns)
    return out


def _check_lh014(tree: ast.AST, path_str: str, findings: list[_Finding]) -> list[_Finding]:
    """Interprocedural-lite (ADVERSARIAL §1a-E2 shape): flag call sites of
    functions whose own body produced a finding. Single-level, single-file:
    cross-file helpers are the documented residual ceiling.
    """
    out: list[_Finding] = []
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    leaky: dict[str, set[str]] = {}
    for func in functions:
        end = func.end_lineno or func.lineno
        hits = {f.rule_id for f in findings if func.lineno <= f.line <= end}
        # the helper's own LH014 call-site findings don't propagate upward
        hits.discard("LH014")
        if hits:
            leaky.setdefault(func.name, set()).update(hits)
    if not leaky:
        return out
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # `helper()` and `self.helper()`/`obj.helper()` alike — a leaky
        # helper carries its finding through a dotted call too.
        callee = node.func
        if isinstance(callee, ast.Name):
            name = callee.id
        elif isinstance(callee, ast.Attribute):
            name = callee.attr
        else:
            continue
        if name not in leaky:
            continue
        # skip recursive self-calls
        enclosing = [
            f
            for f in functions
            if f.lineno <= node.lineno <= (f.end_lineno or f.lineno) and f.name == name
        ]
        if enclosing:
            continue
        out.append(
            _Finding(
                "LH014",
                node.lineno,
                node.col_offset,
                f"call to helper `{name}` whose body trips {', '.join(sorted(leaky[name]))}; "
                "the leakage travels with the helper",
            )
        )
    return out


_CHECKERS = {
    "LH001": _check_lh001,
    "LH002": _check_lh002,
    "LH003": _check_lh003,
    "LH004": _check_lh004,
    "LH005": _check_lh005,
    "LH006": _check_lh006,
    "LH007": _check_lh007,
    "LH008": _check_lh008,
    "LH009": _check_lh009,
    "LH010": _check_lh010,
    "LH011": _check_lh011,
}

_ALL_RULES = frozenset(_CHECKERS) | {"LH012", "LH013", "LH014"}


def scan_file(path: Path, *, rules: set[str] | None = None) -> list[_Finding]:
    """Scan one .py file; never raises on parse errors (LH012 instead)."""
    cache_token = _SCOPE_CACHE.set({})
    try:
        return _scan_file_cached(path, rules=rules)
    finally:
        _SCOPE_CACHE.reset(cache_token)


def _scan_file_cached(path: Path, *, rules: set[str] | None = None) -> list[_Finding]:
    path_str = path.as_posix()
    enabled = rules if rules is not None else set(_ALL_RULES)
    findings: list[_Finding] = []
    try:
        # utf-8-sig: a BOM-bearing file parses cleanly instead of degrading
        # to an LH012 warning that leaves the file's contents unscanned.
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
    except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
        if "LH012" not in enabled:
            return []
        line = getattr(exc, "lineno", None) or 1
        return [
            _Finding("LH012", int(line), 0, f"unparseable file: {exc.__class__.__name__}: {exc}")
        ]
    for rule_id, checker in _CHECKERS.items():
        if rule_id not in enabled:
            continue
        if _is_allowlisted(rule_id, path_str):
            continue
        raw = checker(tree, path_str)
        findings.extend(_drop_function_allowlisted(rule_id, path_str, tree, raw))
    if "LH013" in enabled and not _is_allowlisted("LH013", path_str):
        findings.extend(_check_lh013(tree, path_str, source))
    if "LH014" in enabled and not _is_allowlisted("LH014", path_str):
        findings.extend(_check_lh014(tree, path_str, findings))
    findings.sort(key=lambda f: (f.line, f.col, f.rule_id))
    return findings


def collect_py_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file() and p.suffix == ".py":
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
        else:
            raise ValueError(f"scan target is missing or is not a Python file/directory: {p}")
    # de-duplicate, stable order
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        key = str(f)
        if key not in seen:
            seen.add(key)
            out.append(f)
    return sorted(out, key=lambda f: f.as_posix())
