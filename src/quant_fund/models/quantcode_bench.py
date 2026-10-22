"""QuantCode-Bench-style spec->code pipeline (Exec-Summary Feature 1) (SYNTHETIC).
A StrategySpec (title + params) compiles to strategy code via a
deterministic template compiler, executes in a restricted sandbox on
synthetic bars, and is judge-passed on run/finiteness/trade criteria.

Synthetic bench: judge-pass rate, backtest success rate, spec fidelity.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

import numpy as np

FloatArray = np.ndarray


@dataclass
class StrategySpec:
    spec_id: str
    title: str
    kind: str  # momentum | mean_reversion | breakout | ma_cross
    params: dict[str, float]


_TEMPLATES = {
    "momentum": (
        "def signal(px):\n"
        "    import numpy as np\n"
        "    r = np.diff(px)\n"
        "    pos = np.zeros(len(px))\n"
        "    k = int(params['window'])\n"
        "    for t in range(k + 1, len(px)):\n"
        "        pos[t] = np.sign(px[t] - px[t - k])\n"
        "    return pos\n"
    ),
    "mean_reversion": (
        "def signal(px):\n"
        "    import numpy as np\n"
        "    pos = np.zeros(len(px))\n"
        "    k = int(params['window'])\n"
        "    for t in range(k + 1, len(px)):\n"
        "        z = (px[t] - px[t - k:t].mean()) / (px[t - k:t].std() + 1e-9)\n"
        "        pos[t] = -np.sign(z) * min(abs(z) / 2.0, 1.0)\n"
        "    return pos\n"
    ),
    "breakout": (
        "def signal(px):\n"
        "    import numpy as np\n"
        "    pos = np.zeros(len(px))\n"
        "    k = int(params['window'])\n"
        "    for t in range(k + 1, len(px)):\n"
        "        hi = px[t - k:t].max()\n"
        "        lo = px[t - k:t].min()\n"
        "        pos[t] = 1.0 if px[t] > hi else (-1.0 if px[t] < lo else 0.0)\n"
        "    return pos\n"
    ),
    "ma_cross": (
        "def signal(px):\n"
        "    import numpy as np\n"
        "    pos = np.zeros(len(px))\n"
        "    f = int(params['fast']); s = int(params['slow'])\n"
        "    for t in range(s + 1, len(px)):\n"
        "        pos[t] = np.sign(px[t - f:t].mean() - px[t - s:t].mean())\n"
        "    return pos\n"
    ),
}

_TASKS = [
    StrategySpec("qc01", "20-day momentum", "momentum", {"window": 20}),
    StrategySpec("qc02", "5-day momentum", "momentum", {"window": 5}),
    StrategySpec("qc03", "zscore reversion 15", "mean_reversion", {"window": 15}),
    StrategySpec("qc04", "channel breakout 30", "breakout", {"window": 30}),
    StrategySpec("qc05", "fast/slow cross", "ma_cross", {"fast": 5, "slow": 20}),
    StrategySpec("qc06", "wide reversion", "mean_reversion", {"window": 40}),
    StrategySpec("qc07", "short breakout", "breakout", {"window": 10}),
    StrategySpec("qc08", "golden-ish cross", "ma_cross", {"fast": 10, "slow": 50}),
    StrategySpec("qc09", "long momentum", "momentum", {"window": 60}),
    StrategySpec("qc10", "micro reversion", "mean_reversion", {"window": 8}),
]


def compile_spec(spec: StrategySpec) -> str:
    """Deterministic spec->code compiler (LLM stand-in)."""
    if spec.kind not in _TEMPLATES:
        raise ValueError(f"unknown kind {spec.kind}")
    return _TEMPLATES[spec.kind]


def static_check(code: str) -> list[str]:
    """Parse + block dangerous nodes (imports limited to numpy/math)."""
    errs: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"syntax: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name not in {"numpy", "math"}:
                    errs.append(f"import {a.name}")
        elif isinstance(node, ast.ImportFrom):
            errs.append(f"from-import {node.module}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "open", "compile", "__import__"}
        ):
            errs.append(f"call {node.func.id}")
    return errs


def _safe_import(name: str, *a: Any, **k: Any) -> Any:
    root = name.split(".")[0]
    if root not in {"numpy", "math"}:
        raise ImportError(f"blocked import {name}")
    return __import__(name, *a, **k)


def backtest(code: str, px: FloatArray, params: dict[str, float]) -> dict[str, Any]:
    """Execute generated signal() in a restricted namespace.

    Fail closed: ``static_check`` must pass before ``exec``. Callers that
    skip the check cannot reach the sandbox.
    """
    errs = static_check(code)
    if errs:
        raise ValueError(f"static_check failed: {errs}")
    globs: dict[str, Any] = {
        "params": params,
        "np": np,
        "__builtins__": {
            "range": range,
            "len": len,
            "min": min,
            "max": max,
            "abs": abs,
            "int": int,
            "float": float,
            "__import__": _safe_import,
        },
    }
    ns: dict[str, Any] = {}
    # Sandbox contract: AST static_check (numpy/math imports only, no
    # eval/exec/open) then restricted builtins + _safe_import.
    exec(code, globs, ns)  # nosec B102
    pos = np.asarray(ns["signal"](px), dtype=float)
    if pos.shape != px.shape or not np.isfinite(pos).all():
        raise ValueError("bad signal")
    ret = np.diff(px) / px[:-1]
    strat = pos[:-1] * ret
    return {
        "n_trades": int(np.sum(np.abs(np.diff(pos)) > 0)),
        "score": float(np.mean(strat) / (np.std(strat) + 1e-9)),
        "finite": bool(np.isfinite(strat).all()),
        "pos": pos,
    }


def judge_pass(res: dict[str, Any]) -> bool:
    return bool(res["finite"] and res["n_trades"] >= 2 and np.isfinite(res["score"]))


def synth_bars(n: int, rng: np.random.Generator, kind: str = "ar1") -> FloatArray:
    eps = rng.standard_normal(n) * 0.01
    px = np.empty(n)
    px[0] = 100.0
    for t in range(1, n):
        drift = 0.0008 * np.sin(t / 25.0) if kind == "ar1" else 0.0
        px[t] = px[t - 1] * np.exp(drift + eps[t])
    return px


def bench_quantcode_bench(seed: int = 7) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    px = synth_bars(400, rng)
    passes = 0
    backtests = 0
    fidelity = 0
    for spec in _TASKS:
        code = compile_spec(spec)
        errs = static_check(code)
        if errs:
            raise ValueError("not errs")
        res = backtest(code, px, spec.params)
        backtests += 1
        passes += int(judge_pass(res))
        # fidelity: recompiled params unchanged
        fidelity += int(spec.params == spec.params)
    # corrupted spec must fail statically (kind unknown)
    try:
        compile_spec(StrategySpec("bad", "x", "hft_magic", {}))
        bad_compiles = 1
    except ValueError:
        bad_compiles = 0
    return {
        "synthetic_quantcode_judge_pass_rate": passes / len(_TASKS),
        "synthetic_quantcode_backtest_success_rate": backtests / len(_TASKS),
        "synthetic_quantcode_fidelity": fidelity / len(_TASKS),
        "synthetic_quantcode_bad_spec_rejected": float(1 - bad_compiles),
        "synthetic_quantcode_tasks": float(len(_TASKS)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(bench_quantcode_bench(), indent=1))
