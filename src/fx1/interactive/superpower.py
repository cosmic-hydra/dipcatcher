"""/superpower — plan, select, coordinate, and explain research capabilities.

The registry is built from ``fx1.harness.HARNESS_REGISTRY`` (the exact
fail-closed set of lab surfaces fx-1 may invoke) plus three front-door
capabilities: web research, flash-context retrieval, and flash save. A
planner maps a goal to the *relevant subset* — deterministic keyword scoring,
optionally refined by a model pass — and the coordinator runs that subset in
dependency order (memory → web → harness → save → synthesis), gating
consequential harness commands behind explicit approval.

Two contracts never bend here:
- harness execution happens only through registered commands (subprocess,
  injectable runner), so the model can never reach an unregistered surface;
- the synthesized answer passes ``fx1.honesty.validate_fx1_output`` — on
  violation the answer is withheld and the gate's reason reported instead.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from fx1.harness import HARNESS_REGISTRY, Harness, HarnessRole
from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive.approvals import ApprovalGate

# ---------------------------------------------------------------------------
# Capability registry
# ---------------------------------------------------------------------------

KIND_HARNESS = "harness"
KIND_WEB = "web"
KIND_FLASH = "flash"
KIND_REASON = "reason"

# Trigger terms that make a harness command relevant to a goal. Keys are
# capability ids; the catalog is data, not code, so it can be listed and
# audited (``/superpower help``) without executing anything.
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "doctor": ("doctor", "manifest", "readiness", "health check"),
    "verify-research": ("verify", "receipt", "provenance", "audit", "immutable"),
    "validate": ("validate", "promotion gate", "walk-forward", "causal"),
    "monitor": ("monitor", "operational", "runtime"),
    "ingest": ("ingest", "bronze", "silver", "lake"),
    "collect": ("collect", "public feed", "opt-in"),
    "build-features": ("features", "pit", "point-in-time"),
    "build-labels": ("labels", "target", "labeling"),
    "research": (
        "benchmark",
        "proper score",
        "pinball",
        "crps",
        "qlike",
        "conformal",
        "e-values",
        "jackknife",
        "crc",
        "volatility",
        "regime",
        "tail risk",
        "drawdown",
        "liquidity",
        "ranking",
        "alpha",
        "distribution",
        "quantile bandit",
        "hypothesis",
        "family",
        "sweep",
    ),
    "northset": ("order book", "lob", "northset", "uncrossed", "identity"),
    "candle-book": ("candle", "ohlc", "candlestick"),
    "kyle-ofi": ("kyle", "lambda", "ofi", "flow imbalance"),
    "session-book": ("session", "book reconstruction"),
    "vendor-book-map": ("vendor", "book mapping"),
    "book-panel": ("book panel", "depth"),
    "backtest": ("backtest", "fill", "slippage", "participation", "cost model"),
    "forecast": ("forecast", "predict"),
    "kronos-forecast": ("kronos",),
    "report": ("report", "notebook", "write up", "writeup"),
    "tearsheet": ("tearsheet", "analytics export"),
    "train": ("train", "fine-tune", "finetune", "lora", "qlora"),
    "optimize": ("optimize", "hyperparameter", "optuna"),
    "paper": ("paper trading", "shadow run", "simulated broker"),
    "live": ("live desk", "live trading", "arm live", "drop folder"),
}

_CONSEQUENTIAL = frozenset({"train", "optimize", "paper"})

_WEB_TRIGGERS = (
    "web",
    "internet",
    "online",
    "search",
    "latest",
    "current",
    "news",
    "recent",
    "look up",
    "what does",
    "literature",
    "sources say",
    "find out",
    "compare publicly",
)

_HARNESS_CAPABILITIES: dict[str, dict[str, Any]] = {
    command.name: {
        "id": command.name,
        "name": command.name,
        "kind": KIND_HARNESS,
        "description": command.description,
        "role": command.role.value,
        "timeout_s": command.timeout_s,
        "consequential": command.name in _CONSEQUENTIAL
        or command.role == HarnessRole.MODEL_TRAINING,
        "cost": "heavy" if command.timeout_s >= 1800 else "moderate",
    }
    for command in HARNESS_REGISTRY
}

CAPABILITIES: dict[str, dict[str, Any]] = {
    **_HARNESS_CAPABILITIES,
    "web_research": {
        "id": "web_research",
        "name": "deep web research",
        "kind": KIND_WEB,
        "description": (
            "Iterative public-web research: search, read sources, follow "
            "references, cross-check claims, report conflicts."
        ),
        "role": "research",
        "timeout_s": 600,
        "consequential": False,
        "cost": "moderate",
    },
    "flash_retrieve": {
        "id": "flash_retrieve",
        "name": "flash context (retrieve)",
        "kind": KIND_FLASH,
        "description": (
            "Retrieve the relevant portions of persistent research memory for the current goal."
        ),
        "role": "research",
        "timeout_s": 5,
        "consequential": False,
        "cost": "cheap",
    },
    "flash_save": {
        "id": "flash_save",
        "name": "flash context (save)",
        "kind": KIND_FLASH,
        "description": "Persist findings with sources and uncertainty to flash context.",
        "role": "research",
        "timeout_s": 5,
        "consequential": False,
        "cost": "cheap",
    },
    "synthesize": {
        "id": "synthesize",
        "name": "synthesis",
        "kind": KIND_REASON,
        "description": (
            "Compose the final answer from capability results, explaining "
            "which capabilities were used and why."
        ),
        "role": "research",
        "timeout_s": 300,
        "consequential": False,
        "cost": "cheap",
    },
}

ModelFn = Callable[[str], str]


def _score_goal(goal: str, terms: tuple[str, ...]) -> int:
    lowered = goal.lower()
    return sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))


@dataclass
class PlannedCapability:
    """One selected capability and why it was selected."""

    id: str
    rationale: str


@dataclass
class SuperpowerPlan:
    goal: str
    capabilities: list[PlannedCapability] = field(default_factory=list)
    source: str = "deterministic"

    def ids(self) -> list[str]:
        return [cap.id for cap in self.capabilities]

    def describe(self) -> str:
        lines = [f"plan for: {self.goal}", f"planner: {self.source}"]
        for cap in self.capabilities:
            meta = CAPABILITIES.get(cap.id, {})
            lines.append(
                f"- {cap.id} ({meta.get('kind', '?')}, {meta.get('cost', '?')}): {cap.rationale}"
            )
        return "\n".join(lines)


def plan_goal(
    goal: str,
    *,
    capabilities: dict[str, dict[str, Any]] | None = None,
    model_fn: ModelFn | None = None,
    max_harness: int = 4,
) -> SuperpowerPlan:
    """Select the relevant capability subset for *goal*.

    Deterministic pass: score every harness capability by trigger terms and
    web/flash by their own triggers. Optional model pass: a hosted turn is
    asked for a JSON selection from the registry, including capabilities the
    keyword pass did not select. Unknown ids are dropped, duplicates are
    removed, and the same harness limit applies after refinement. Malformed
    or failed model replies fall back to the bounded deterministic plan.
    """
    if max_harness < 0:
        raise ValueError("max_harness must be nonnegative")
    registry = capabilities or CAPABILITIES
    scored: list[tuple[int, str]] = []
    for cap_id, meta in registry.items():
        if meta.get("kind") != KIND_HARNESS:
            continue
        terms = _TRIGGERS.get(cap_id, ())
        score = _score_goal(goal, terms)
        if score:
            scored.append((score, cap_id))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    harness_ids = [cap_id for _, cap_id in scored[:max_harness]]

    selected: list[PlannedCapability] = [
        PlannedCapability("flash_retrieve", "memory is cheap and may already hold this research"),
    ]
    for cap_id in harness_ids:
        meta = registry[cap_id]
        terms = _TRIGGERS.get(cap_id, ())
        matched = [t for t in terms if re.search(rf"\b{re.escape(t)}\b", goal.lower())]
        selected.append(
            PlannedCapability(
                cap_id,
                f"goal mentions {', '.join(repr(t) for t in matched[:4])}; "
                f"{meta.get('description', '')}",
            )
        )
    if _score_goal(goal, _WEB_TRIGGERS) or not harness_ids:
        selected.append(
            PlannedCapability(
                "web_research",
                "goal asks for current/external information, or no lab surface "
                "matched — public web research fills the gap",
            )
        )
    selected.append(
        PlannedCapability(
            "flash_save",
            "new findings belong in flash context with sources and uncertainty",
        )
    )
    selected.append(
        PlannedCapability(
            "synthesize", "compose the final answer and explain the capability choices"
        )
    )

    plan = SuperpowerPlan(goal=goal, capabilities=selected, source="deterministic")
    if model_fn is not None:
        plan = _model_refine(plan, goal, registry, model_fn, max_harness=max_harness)
    return plan


def _model_refine(
    fallback: SuperpowerPlan,
    goal: str,
    registry: dict[str, dict[str, Any]],
    model_fn: ModelFn,
    *,
    max_harness: int,
) -> SuperpowerPlan:
    """Validate and bound registered model selections, else keep fallback."""
    listing = "\n".join(
        f"- {cap_id}: {meta.get('description', '')}" for cap_id, meta in sorted(registry.items())
    )
    prompt = (
        "You select dipcatcher research capabilities for a goal. Reply with "
        'JSON only: {"capabilities": ["<id>"...], "rationale": "one '
        f'sentence per id"}}. Select at most {max_harness} harness capabilities; '
        f"do not repeat ids. Available: {listing}\n\nGoal: {goal}"
    )
    try:
        raw = model_fn(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON object in model reply")
        payload = json.loads(match.group(0))
        ids = payload.get("capabilities")
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise ValueError("capabilities must be a list of strings")
        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            raise ValueError("rationale must be a string")
        valid: list[str] = []
        harness_count = 0
        for cap_id in dict.fromkeys(ids):
            if cap_id not in registry or cap_id in {"flash_retrieve", "synthesize"}:
                continue
            if registry[cap_id].get("kind") == KIND_HARNESS:
                if harness_count >= max_harness:
                    continue
                harness_count += 1
            valid.append(cap_id)
        # Model ordering cannot remove or relocate the mandatory spine.
        valid = ["flash_retrieve", *valid, "synthesize"]
        capabilities = [
            PlannedCapability(
                cap_id,
                f"model rationale: {rationale}" if rationale else "model-selected",
            )
            for cap_id in valid
        ]
        return SuperpowerPlan(goal=goal, capabilities=capabilities, source="model+validated")
    except Exception:  # noqa: BLE001 — model failure preserves the bounded fallback
        return fallback


# ---------------------------------------------------------------------------
# Coordination
# ---------------------------------------------------------------------------


@dataclass
class CapabilityRun:
    id: str
    name: str
    status: str  # ran | failed | skipped | approval-denied
    detail: str = ""
    artifacts: list[str] = field(default_factory=list)


@dataclass
class SuperpowerResult:
    goal: str
    plan: SuperpowerPlan
    runs: list[CapabilityRun] = field(default_factory=list)
    summary: str = ""
    honesty_violation: str | None = None
    errors: list[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = ["capabilities used and why:", ""]
        for run in self.runs:
            lines.append(f"[{run.status}] {run.name}: {run.detail}")
        if self.honesty_violation:
            lines += ["", f"honesty gate: {self.honesty_violation}"]
        return "\n".join(lines)


@dataclass
class SuperpowerContext:
    """Everything a run needs — all seams injectable for offline tests."""

    model_fn: ModelFn | None = None
    harness: Harness | None = None
    approvals: ApprovalGate | None = None
    researcher: Any | None = None
    research_budget: Any | None = None
    flash_store: Any | None = None
    on_event: Callable[[str, str], None] | None = None
    cancel_event: threading.Event | None = None
    max_stdout_chars: int = 12000
    save_to_flash: bool = True


def _emit(ctx: SuperpowerContext, event: str, detail: str) -> None:
    if ctx.on_event is not None:
        ctx.on_event(event, detail)


def _cancelled(ctx: SuperpowerContext) -> bool:
    return ctx.cancel_event is not None and ctx.cancel_event.is_set()


def _harness_run(
    cap_id: str, ctx: SuperpowerContext, extra_args: list[str] | None = None
) -> tuple[int, str, str]:
    harness = ctx.harness if ctx.harness is not None else Harness()
    result = harness.run(cap_id, extra_args)
    return result.exit_code, result.stdout, result.stderr


def _data_label_of(stdout: str) -> str:
    """Preserve the harness's honesty label lines into the synthesis context."""
    labels = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("DATA_LABEL=") or line.strip() == "SYNTHETIC"
    ]
    return "; ".join(labels[:4])


def run_plan(plan: SuperpowerPlan, ctx: SuperpowerContext) -> SuperpowerResult:
    result = SuperpowerResult(goal=plan.goal, plan=plan)
    selected = list(dict.fromkeys(plan.ids()))
    context_parts: list[str] = []

    # 1 — flash retrieval (cheap, first, feeds everything else)
    if "flash_retrieve" in selected and ctx.flash_store is not None:
        try:
            from fx1.flash.retrieve import context_block, retrieve

            hits = retrieve(ctx.flash_store, plan.goal, k=4)
            if hits:
                context_parts.append(context_block(hits))
                for hit in hits:
                    ctx.flash_store.mark_used(hit.entry.id)
                result.runs.append(
                    CapabilityRun(
                        "flash_retrieve",
                        "flash context (retrieve)",
                        "ran",
                        f"{len(hits)} relevant memory entries retrieved",
                    )
                )
            else:
                result.runs.append(
                    CapabilityRun(
                        "flash_retrieve",
                        "flash context (retrieve)",
                        "ran",
                        "no stored context scored above threshold for this goal",
                    )
                )
        except Exception as exc:  # noqa: BLE001 — memory must not block the run
            result.runs.append(
                CapabilityRun("flash_retrieve", "flash context (retrieve)", "failed", str(exc))
            )
            result.errors.append(f"flash_retrieve: {exc}")

    # 2 — web research (budgeted, cancellable)
    if "web_research" in selected and ctx.researcher is not None and not _cancelled(ctx):
        try:
            report = ctx.researcher.run(
                plan.goal,
                budget=ctx.research_budget,
                cancel_event=ctx.cancel_event,
                on_progress=lambda p: _emit(ctx, f"web:{p.event}", p.detail[:200]),
            )
            context_parts.append(report.to_markdown())
            result.runs.append(
                CapabilityRun(
                    "web_research",
                    "deep web research",
                    "ran",
                    f"{len([s for s in report.sources if s.ok])} sources; "
                    f"{len(report.groups)} claim topics"
                    + (" (cancelled)" if report.cancelled else ""),
                    artifacts=[s.url for s in report.sources[:6]],
                )
            )
        except Exception as exc:  # noqa: BLE001
            result.runs.append(
                CapabilityRun("web_research", "deep web research", "failed", str(exc))
            )
            result.errors.append(f"web_research: {exc}")

    # 3 — harness commands (approval for consequential, cancellable between steps)
    for cap_id in selected:
        meta = CAPABILITIES.get(cap_id)
        if not meta or meta.get("kind") != KIND_HARNESS:
            continue
        if _cancelled(ctx):
            result.runs.append(
                CapabilityRun(cap_id, cap_id, "skipped", "cancelled before execution")
            )
            continue
        gate = ctx.approvals or ApprovalGate()
        if meta.get("consequential") and not gate.ask(
            f"run harness command {cap_id!r} (consequential: {meta.get('description', '')})?",
            kind="consequential",
        ):
            result.runs.append(
                CapabilityRun(
                    cap_id,
                    cap_id,
                    "approval-denied",
                    "operator did not approve this consequential command",
                )
            )
            continue
        _emit(ctx, "harness", f"running {cap_id} (registered command)")
        try:
            code, stdout, stderr = _harness_run(cap_id, ctx)
        except Exception as exc:  # noqa: BLE001
            result.runs.append(CapabilityRun(cap_id, cap_id, "failed", str(exc)))
            result.errors.append(f"{cap_id}: {exc}")
            continue
        status = "ran" if code == 0 else "failed"
        labels = _data_label_of(stdout)
        detail = stdout.strip().splitlines()[-1][:160] if stdout.strip() else stderr[:160]
        if labels:
            detail = f"{labels}; {detail}"
        context_parts.append(f"# harness {cap_id} (exit {code})\n{stdout[-ctx.max_stdout_chars :]}")
        result.runs.append(CapabilityRun(cap_id, cap_id, status, detail))

    # 4 — flash save
    if "flash_save" in selected and ctx.save_to_flash and ctx.flash_store is not None:
        try:
            entry = ctx.flash_store.add(
                text=(f"/superpower run on: {plan.goal} — capabilities: {', '.join(selected)}"),
                task=plan.goal,
                tags=["superpower"],
                uncertainty="medium",
            )
            result.runs.append(
                CapabilityRun(
                    "flash_save",
                    "flash context (save)",
                    "ran",
                    f"saved run summary as flash entry {entry.id[:8]}",
                    artifacts=[entry.id],
                )
            )
        except Exception as exc:  # noqa: BLE001
            result.runs.append(
                CapabilityRun("flash_save", "flash context (save)", "failed", str(exc))
            )
            result.errors.append(f"flash_save: {exc}")

    # 5 — synthesis
    if "synthesize" in selected:
        if ctx.model_fn is None:
            result.summary = _offline_summary(result)
            result.runs.append(
                CapabilityRun(
                    "synthesize",
                    "synthesis",
                    "ran",
                    "no model endpoint available — offline summary of collected results",
                )
            )
        else:
            try:
                answer = ctx.model_fn(_synthesis_prompt(plan, result, context_parts))
                validate_fx1_output(answer)
                result.summary = answer
                result.runs.append(
                    CapabilityRun("synthesize", "synthesis", "ran", "honesty gate passed")
                )
            except Fx1HonestyError as exc:
                result.honesty_violation = str(exc)
                result.summary = (
                    "synthesis withheld by the fx-1 honesty gate "
                    f"({result.honesty_violation}). Raw capability outputs are "
                    "reported above; no violating text was emitted."
                )
                result.runs.append(
                    CapabilityRun(
                        "synthesize", "synthesis", "failed", "honesty gate rejected output"
                    )
                )
            except Exception as exc:  # noqa: BLE001
                result.runs.append(CapabilityRun("synthesize", "synthesis", "failed", str(exc)))
                result.errors.append(f"synthesize: {exc}")
    return result


def _offline_summary(result: SuperpowerResult) -> str:
    """Deterministic summary when no model endpoint is available — real
    results, no fabrication: what ran, what failed, what was found."""
    lines = [f"/superpower goal: {result.goal}", ""]
    for run in result.runs:
        if run.id == "synthesize":
            continue
        lines.append(f"- {run.name} ({run.id}): {run.status} — {run.detail}")
    if result.errors:
        lines.append("")
        lines.append("errors: " + "; ".join(result.errors))
    lines.append("")
    lines.append(
        "note: no model endpoint was available, so this is the raw capability "
        "summary — run `fxi keys set` to enable model synthesis."
    )
    return "\n".join(lines)


def _synthesis_prompt(
    plan: SuperpowerPlan, result: SuperpowerResult, context_parts: list[str]
) -> str:
    runs = "\n".join(f"- {run.name}: {run.status} — {run.detail}" for run in result.runs)
    evidence = "\n\n".join(context_parts) or "(no capability output)"
    return (
        "You are dipcatcher's /superpower coordinator for the fx-1 research "
        "lab. Answer the goal using ONLY the capability results below; do not "
        "invent sources, numbers, or receipts. Cite web sources by URL and "
        "harness results by receipt/artifact path when given. Keep every "
        "SYNTHETIC / DATA_LABEL marker visible. State uncertainty and "
        "conflicting evidence explicitly. Never headline Sharpe/Sortino/"
        "Calmar/P&L/NAV — research results are proper scores.\n\n"
        f"GOAL: {plan.goal}\n\n"
        f"CAPABILITIES USED AND WHY:\n{runs}\n\n"
        f"CAPABILITY RESULTS:\n{evidence}\n\n"
        "Final answer (concise, honest, sources attached):"
    )


def describe_capabilities() -> str:
    lines = ["available /superpower capabilities:", ""]
    for cap_id, meta in sorted(CAPABILITIES.items()):
        flags = []
        if meta.get("consequential"):
            flags.append("approval-required")
        flags.append(str(meta.get("cost", "?")))
        lines.append(
            f"- {cap_id} [{meta.get('kind')}; {', '.join(flags)}]: {meta.get('description', '')}"
        )
    return "\n".join(lines)
