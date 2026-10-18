"""The concierge — dipcatcher's conversational orchestration shell.

This is the front door the product goal describes: chat with the active
fx-series model, ask follow-up questions, and delegate real work through
slash commands:

- ``/superpower <goal>`` — plan, select, coordinate, and explain research
  capabilities (background by default; ``--sync`` for scripts).
- ``/research <goal>`` — deep background web research with budgets,
  progress, ``/cancel`` and ``/redirect``.
- ``/flash ...`` — inspect, correct, remove, search, and refresh persistent
  research memory.
- ``/trade`` — live-trading readiness. Live orders stay disabled; the only
  executable book is the paper session.
- ``/live`` — operator card for the live desk. The chat does not transmit.
- plain text — a conversational turn with tool access to flash context,
  a bounded web lookup, and approval-gated harness commands.

Responsiveness contract: background jobs run on their own thread. On POSIX
terminals the console polls stdin so progress renders while waiting for a
line. Windows terminals and piped (non-TTY) stdin use blocking reads;
background work continues, but its queued progress renders between lines.

Every seam is injectable (backend, harness, researcher, store, approvals,
input/output) so the whole surface is testable offline; nothing here
fabricates results when a seam is missing — it says so.
"""

from __future__ import annotations

import json
import select
import shlex
import sys
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import typer

from fx1.honesty import Fx1HonestyError, validate_fx1_output
from fx1.interactive import actions, profiles
from fx1.interactive.approvals import ApprovalGate, console_ask_fn
from fx1.interactive.orb import OrbAnimator, OrbState, orbs_enabled
from fx1.interactive.orb import banner as orb_banner
from fx1.interactive.profiles import DEFAULT_MODEL
from fx1.interactive.superpower import (
    SuperpowerContext,
    SuperpowerResult,
    describe_capabilities,
    plan_goal,
    run_plan,
)
from fx1.serve.backends import HostedK3Backend

HELP_TEXT = """\
conversation          plain text chats with the active model (tools: flash, web, harness)
/superpower <goal>    plan + run the relevant research capabilities (--sync to run inline)
/research <goal>      deep web research in the background (--budget-s, --max-sources, --queries)
/status               show the background job's progress
/cancel               cancel the background job (research or superpower)
/redirect <goal>      re-target the background research at a new goal
/flash add|list|show|remove|correct|search|refresh ...
/superpower help      list every capability the planner may select
/use <model>          activate fx1 | fx1-lite
/model                show the active model
/keys list|set|remove  manage model endpoints (presence-only)
/harness list|describe|run  registered dipcatcher harness commands (fail-closed)
/live                 how to run the live desk (this chat does not transmit orders)
/trade [readiness|help]  live orders are disabled; paper session is the only book
/help                 this help
/exit                 leave
"""


# ---------------------------------------------------------------------------
# Background jobs
# ---------------------------------------------------------------------------


@dataclass
class BackgroundJob:
    """A cancellable background run with an event queue the console drains."""

    label: str
    run_fn: Callable[[threading.Event, Callable[[str, str], None]], Any]
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)
    events: deque[tuple[str, str]] = field(default_factory=deque)
    result: Any = None
    error: str = ""
    announced: bool = False
    thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def start(self) -> BackgroundJob:
        def on_event(event: str, detail: str) -> None:
            self.events.append((event, detail))

        def target() -> None:
            try:
                self.result = self.run_fn(self.cancel_event, on_event)
            except Exception as exc:  # noqa: BLE001 — report, never crash the shell
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.done_event.set()

        self.thread = threading.Thread(target=target, daemon=True, name=f"dip-{self.label}")
        self.thread.start()
        return self

    @property
    def running(self) -> bool:
        return not self.done_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()

    def join(self, timeout_s: float | None = None) -> bool:
        """Wait for the worker and report whether it actually stopped."""
        if self.thread is not None:
            self.thread.join(timeout_s)
        return self.done_event.is_set()

    def drain(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        while self.events:
            out.append(self.events.popleft())
        return out


# ---------------------------------------------------------------------------
# The concierge
# ---------------------------------------------------------------------------


class Concierge:
    """Conversational orchestrator; one instance per session."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        store: Any | None = None,
        researcher: Any | None = None,
        research_budget: Any | None = None,
        harness: Any | None = None,
        approvals: ApprovalGate | None = None,
        backend_factory: Callable[..., Any] | None = None,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[..., None] = typer.echo,
        animated: bool | None = None,
    ) -> None:
        from fx1.flash.store import FlashStore
        from fx1.webresearch.engine import ResearchBudget, Researcher

        self.model = model
        self._store = store if store is not None else FlashStore()
        self._researcher = researcher if researcher is not None else Researcher()
        self._research_budget = research_budget or ResearchBudget()
        self._harness = harness
        self._approvals = approvals if approvals is not None else ApprovalGate()
        self._backend_factory = backend_factory or HostedK3Backend
        self._input = input_fn
        self._output = output_fn
        self._animated = True if animated is None else animated
        self._messages: list[dict[str, Any]] = []
        self._job: BackgroundJob | None = None

    # -- output helpers ------------------------------------------------------

    def _say(self, text: str = "") -> None:
        self._output(text)

    def _say_error(self, exc: Exception) -> None:
        typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)

    def _backend(self) -> Any | None:
        resolved = profiles.resolve_endpoint(self.model)
        if not resolved:
            return None
        api_key, base_url = resolved
        try:
            return self._backend_factory(api_key=api_key, model=self.model, api_url=base_url)
        except Exception:  # noqa: BLE001
            return None

    # -- session loop --------------------------------------------------------

    def handle_line(self, line: str) -> bool:
        """Dispatch one input line. Returns True when the session should end."""
        text = line.strip()
        if not text:
            return False
        if text.startswith("/"):
            return self._slash(text)
        if text.lower() in {"exit", "quit"}:
            self.shutdown()
            return True
        self._chat_turn(text)
        return False

    def tick(self) -> None:
        """Drain background progress and announce completion — call between
        typed lines so the conversation stays responsive during research."""
        if self._job is None:
            return
        for event, detail in self._job.drain():
            self._say(f"[{self._job.label}] {event}: {detail}")
        if self._job.done_event.is_set() and not self._job.announced:
            self._job.announced = True
            self._announce_job(self._job)

    def _announce_job(self, job: BackgroundJob) -> None:
        if job.error:
            self._say(f"[{job.label}] failed: {job.error}")
            return
        result = job.result
        if isinstance(result, SuperpowerResult):
            self._say("")
            self._say(result.summary or "(no summary)")
            self._say("")
            self._say(result.explain())
            return
        if result is not None and hasattr(result, "to_markdown"):
            self._say(result.to_markdown())
            return
        self._say(f"[{job.label}] finished")

    def join_job(self, timeout_s: float | None = None) -> None:
        """Block until the background job settles (used by --sync and tests)."""
        if self._job is not None and self._job.running:
            self._job.join(timeout_s)
        self.tick()

    def shutdown(self, timeout_s: float = 5.0) -> bool:
        """Cancel and join the session-owned worker before the session exits."""
        job = self._job
        if job is None:
            return True
        if job.running:
            job.cancel()
            stopped = job.join(timeout_s)
            if not stopped:
                self._say(f"[{job.label}] did not stop within {timeout_s:.1f}s")
                return False
        self.tick()
        return True

    def _job_slot_available(self) -> bool:
        """Refuse to orphan an active worker by replacing its only handle."""
        if self._job is not None and self._job.running:
            self._say(
                f"[{self._job.label}] already running — wait for it or use /cancel before "
                "starting another background job"
            )
            return False
        # Announce/drain a completed predecessor before replacing its handle.
        self.tick()
        return True

    # -- slash commands ------------------------------------------------------

    def _slash(self, text: str) -> bool:
        command, _, rest = text[1:].partition(" ")
        handler = getattr(self, f"_cmd_{command.replace('-', '_')}", None)
        if handler is None:
            self._say_error(ValueError(f"unknown command /{command}; try /help"))
            return False
        try:
            return bool(handler(rest))
        except ValueError as exc:
            self._say_error(exc)
            return False

    def _cmd_help(self, rest: str) -> bool:
        del rest
        self._say(HELP_TEXT)
        return False

    def _cmd_exit(self, rest: str) -> bool:
        del rest
        self.shutdown()
        return True

    def _cmd_model(self, rest: str) -> bool:
        del rest
        self._say(f"active model: {self.model}")
        return False

    def _cmd_use(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /use <fx1|fx1-lite>")
        model = tokens[0]
        profiles.apply_profile(model)
        self.model = model
        self._messages = []
        self._say(
            f"active model {model!r}; {profiles.ENV_API_KEY} + {profiles.ENV_BASE_URL} "
            "injected into this session"
        )
        return False

    def _cmd_keys(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens or tokens[0] == "list":
            self._say(actions.render_table(profiles.list_models()))
            return False
        sub = tokens[0]
        if sub == "remove" and len(tokens) == 2:
            removed = profiles.remove_endpoint(tokens[1])
            self._say(
                f"removed endpoint {tokens[1]!r}"
                if removed
                else f"endpoint {tokens[1]!r} was not stored"
            )
            return False
        if sub == "set" and len(tokens) in {1, 2}:
            model = tokens[1] if len(tokens) == 2 else self.model
            profiles.check_model(model)
            import getpass

            endpoint = profiles.resolve_endpoint(model)
            default_url = endpoint[1] if endpoint else profiles.default_base_url()
            try:
                api_key = getpass.getpass(f"› {model} API key: ")
                base_url = self._input(f"› {model} endpoint URL [{default_url}]: ").strip()
            except (EOFError, KeyboardInterrupt):
                self._say(f"endpoint setup for {model!r} cancelled; no changes saved")
                return False
            base_url = base_url or default_url
            path = profiles.set_endpoint(model, api_key, base_url)
            self._say(
                f"stored endpoint for {model!r} -> {profiles.host_of(base_url)} "
                f"in {path} (mode 0600)"
            )
            return False
        raise ValueError("usage: /keys [list | set [model] | remove <model>]")

    def _cmd_harness(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens or tokens[0] == "list":
            self._say(actions.render_table(actions.harness_rows()))
            return False
        if tokens[0] == "describe" and len(tokens) == 2:
            self._say(actions.status_json(actions.harness_get(tokens[1])))
            return False
        if tokens[0] == "run" and len(tokens) >= 2:
            self._run_harness(tokens[1], tokens[2:])
            return False
        raise ValueError("usage: /harness [list | describe <name> | run <name> [args...]]")

    def _cmd_live(self, rest: str) -> bool:
        """Operator card. The chat never transmits an order."""
        del rest
        self._say(
            "\n".join(
                (
                    "Live desk — this chat does not transmit orders.",
                    "  dipcatcher live status",
                    "  dipcatcher live arm --venue loopback",
                    "  dipcatcher live submit --symbol AAPL --side BUY --qty 1 --mark 100",
                    "  dipcatcher live arm --venue drop --confirm SEND-LIVE-ORDERS",
                    "  DIP_LIVE_ARM=1 dipcatcher live submit --symbol AAPL --side BUY --qty 1 --mark 100",
                    "Shadow is the default and transmits nothing.",
                    "Loopback fills are SYNTHETIC.",
                    "The drop folder is the operator bridge. Evidence conditions stay blocked.",
                    "The desk does not report a live-performance claim.",
                    "Approved status only: /harness run live",
                )
            )
        )
        return False

    def _cmd_trade(self, rest: str) -> bool:
        tokens = shlex.split(rest) if rest.strip() else []
        if tokens and tokens[0] not in {"readiness", "help"}:
            raise ValueError("usage: /trade [readiness|help]")
        if tokens and tokens[0] == "help":
            self._say(
                "Live orders are disabled. /trade readiness prints the five blocked "
                "conditions. A paper book is the shell command "
                "`dipcatcher trade session --state <path> --commands <jsonl>`."
            )
            return False
        from quant_fund.validation.live_readiness import format_readiness, live_readiness_report

        self._say(format_readiness(live_readiness_report()))
        return False

    def _run_harness(self, name: str, extra_args: list[str]) -> str:
        """Approval-gated, registry-constrained harness execution."""
        from fx1.harness import HARNESS_REGISTRY, Harness

        command = next((c for c in HARNESS_REGISTRY if c.name == name), None)
        if command is None:
            raise KeyError(f"{name!r} is not a registered harness command")
        consequential = command.role.value == "model_training" or command.name in {
            "train",
            "optimize",
            "paper",
        }
        if consequential and not self._approvals.ask(
            f"run harness command {name!r} (consequential)?", kind="consequential"
        ):
            self._say(f"[harness] {name}: not approved — skipped")
            return ""
        harness = self._harness if self._harness is not None else Harness()
        result = harness.run(name, extra_args)
        if result.stdout:
            self._say(result.stdout.rstrip()[-4000:])
        if result.stderr:
            self._say_error(RuntimeError(result.stderr.strip()[-4000:]))
        return str(result.stdout)

    # -- /superpower ---------------------------------------------------------

    def _cmd_superpower(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /superpower <goal> [--sync]")
        if tokens[0] == "help":
            self._say(describe_capabilities())
            return False
        sync = "--sync" in tokens
        goal = " ".join(t for t in tokens if t != "--sync")
        if not goal.strip():
            raise ValueError("usage: /superpower <goal> [--sync]")
        if not self._job_slot_available():
            return False
        backend = self._backend()
        model_fn = None
        if backend is not None:

            def model_fn(prompt: str) -> str:
                with (
                    OrbAnimator(OrbState.COMPOSING, preset="inline") if self._animated else _null()
                ):
                    return str(backend.complete([{"role": "user", "content": prompt}]))

        plan = plan_goal(goal, model_fn=model_fn)
        self._say(plan.describe())
        self._say("")
        if sync:
            ctx = self._superpower_context(model_fn)
            result = run_plan(plan, ctx)
            job = BackgroundJob(label="superpower", run_fn=lambda *_: result)
            job.result = result
            job.done_event.set()
            job.announced = True
            self._job = job
            self._announce_job(job)
            return False

        def run_fn(
            cancel_event: threading.Event, on_event: Callable[[str, str], None]
        ) -> SuperpowerResult:
            job_ctx = self._superpower_context(
                model_fn, on_event=on_event, cancel_event=cancel_event
            )
            return run_plan(plan, job_ctx)

        self._job = BackgroundJob(label="superpower", run_fn=run_fn).start()
        self._say("[superpower] running in the background — /status, /cancel; chat stays open")
        return False

    def _superpower_context(
        self,
        model_fn: Callable[[str], str] | None,
        *,
        on_event: Callable[[str, str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> SuperpowerContext:
        return SuperpowerContext(
            model_fn=model_fn,
            harness=self._harness,
            approvals=self._approvals,
            researcher=self._researcher,
            research_budget=self._research_budget,
            flash_store=self._store,
            on_event=on_event,
            cancel_event=cancel_event,
        )

    # -- /research -----------------------------------------------------------

    def _cmd_research(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError(
                "usage: /research <goal> [--budget-s N] [--max-sources N] "
                "[--queries q1|q2|q3] [--sync]"
            )
        options: dict[str, Any] = {}
        positional: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in {"--budget-s", "--max-sources"} and i + 1 < len(tokens):
                try:
                    options[token[2:].replace("-", "_")] = int(tokens[i + 1])
                except ValueError:
                    raise ValueError(f"{token} needs an integer") from None
                i += 2
                continue
            if token == "--queries" and i + 1 < len(tokens):
                options["queries"] = [q for q in tokens[i + 1].split("|") if q]
                i += 2
                continue
            if token == "--sync":
                options["sync"] = True
                i += 1
                continue
            positional.append(token)
            i += 1
        goal = " ".join(positional)
        if not goal.strip():
            raise ValueError("usage: /research <goal> ...")
        if not self._job_slot_available():
            return False
        from fx1.webresearch.engine import ResearchBudget

        base = self._research_budget
        budget = ResearchBudget(
            max_duration_s=options.get("budget_s", base.max_duration_s),
            max_sources=options.get("max_sources", base.max_sources),
            max_searches=base.max_searches,
            max_depth=base.max_depth,
            search_pause_s=base.search_pause_s,
        )
        queries = options.get("queries")

        def run_fn(cancel_event: threading.Event, on_event: Callable[[str, str], None]) -> Any:
            return self._researcher.run(
                goal,
                budget=budget,
                queries=queries,
                cancel_event=cancel_event,
                on_progress=lambda p: on_event(p.event, p.detail),
            )

        job = BackgroundJob(label="research", run_fn=run_fn)
        if options.get("sync"):
            job.start()
            job.done_event.wait()
            self._job = job
            self.tick()
            return False
        self._job = job.start()
        self._say(
            f"[research] {goal!r} running in the background "
            f"(budget {budget.max_duration_s:.0f}s / {budget.max_sources} sources) — "
            "/status, /cancel, /redirect; chat stays open"
        )
        return False

    def _cmd_status(self, rest: str) -> bool:
        del rest
        if self._job is None:
            self._say("no background job")
            return False
        state = "running" if self._job.running else "finished"
        self._say(f"[{self._job.label}] {state}")
        for event, detail in self._job.drain():
            self._say(f"[{self._job.label}] {event}: {detail}")
        if not self._job.running and not self._job.announced:
            self._job.announced = True
            self._announce_job(self._job)
        return False

    def _cmd_cancel(self, rest: str) -> bool:
        del rest
        if self._job is None or not self._job.running:
            self._say("no running background job")
            return False
        self._job.cancel()
        self._say(f"[{self._job.label}] cancel requested — will stop at the next step")
        return False

    def _cmd_redirect(self, rest: str) -> bool:
        goal = rest.strip()
        if not goal:
            raise ValueError("usage: /redirect <new goal>")
        if self._job is None or not self._job.running:
            self._say("no running background job to redirect")
            return False
        if hasattr(self._researcher, "redirect"):
            self._researcher.redirect(goal)
            self._say(f"[{self._job.label}] redirect requested -> {goal!r}")
        else:
            self._say_error(RuntimeError("active job cannot be redirected"))
        return False

    # -- /flash --------------------------------------------------------------

    def _cmd_flash(self, rest: str) -> bool:
        tokens = shlex.split(rest)
        if not tokens:
            raise ValueError("usage: /flash add|list|show|remove|correct|search|refresh ...")
        sub = tokens[0]
        args = tokens[1:]
        if sub == "add":
            return self._flash_add(args)
        if sub == "list":
            self._flash_list(args)
            return False
        if sub == "show" and len(args) == 1:
            self._flash_show(args[0])
            return False
        if sub == "remove" and len(args) == 1:
            removed = self._store.remove(args[0])
            self._say(f"removed {args[0]!r}" if removed else f"no entry {args[0]!r}")
            return False
        if sub == "correct":
            return self._flash_correct(args)
        if sub == "search" and args:
            self._flash_search(" ".join(args))
            return False
        if sub == "refresh" and len(args) == 1:
            self._flash_refresh(args[0])
            return False
        raise ValueError("usage: /flash add|list|show|remove|correct|search|refresh ...")

    def _flash_add(self, args: list[str]) -> bool:
        options: dict[str, Any] = {}
        positional: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token in {
                "--task",
                "--tags",
                "--source",
                "--uncertainty",
                "--conflict",
            } and i + 1 < len(args):
                options[token[2:]] = args[i + 1]
                i += 2
                continue
            if token in {"--private", "--verified"}:
                options[token[2:]] = True
                i += 1
                continue
            positional.append(token)
            i += 1
        text = " ".join(positional)
        if not text.strip():
            raise ValueError("usage: /flash add <text> [--task T] [--tags a,b] [--source URL] ...")
        entry = self._store.add(
            text=text,
            task=options.get("task", ""),
            tags=[t.strip() for t in options.get("tags", "").split(",") if t.strip()],
            sources=[{"url": options["source"]}] if options.get("source") else [],
            uncertainty=options.get("uncertainty"),
            conflicts=[options["conflict"]] if options.get("conflict") else [],
            private=bool(options.get("private")),
            verified=bool(options.get("verified")),
        )
        self._say(f"saved flash entry {entry.id} ({entry.id[:8]})")
        return False

    def _flash_list(self, args: list[str]) -> None:
        from fx1.flash.retrieve import retrieve

        if args and args[0] == "all":
            entries = self._store.all()
        elif args:
            entries = [hit.entry for hit in retrieve(self._store, " ".join(args), k=10)]
        else:
            entries = self._store.all()
        if not entries:
            self._say("(no flash entries)")
            return
        for entry in entries[:30]:
            flags = []
            if entry.private:
                flags.append("private")
            if entry.verified:
                flags.append("verified")
            if entry.uncertainty:
                flags.append(entry.uncertainty)
            self._say(
                f"{entry.id[:8]}  {entry.updated_at[:10]}  "
                f"{('[' + ','.join(flags) + '] ') if flags else ''}{entry.text[:90]}"
            )

    def _flash_show(self, entry_id: str) -> None:
        entry = self._store.get(entry_id)
        if entry is None:
            self._say_error(KeyError(f"no flash entry {entry_id!r}"))
            return
        self._say(entry.model_dump_json(indent=2))

    def _flash_correct(self, args: list[str]) -> bool:
        if not args or not args[0]:
            raise ValueError("usage: /flash correct <id> --text ... --uncertainty ... --tags ...")
        entry_id = args[0]
        patch: dict[str, Any] = {}
        i = 1
        while i < len(args):
            token = args[i]
            if token in {"--text", "--task", "--uncertainty"} and i + 1 < len(args):
                patch[token[2:]] = args[i + 1]
                i += 2
                continue
            if token == "--tags" and i + 1 < len(args):
                patch["tags"] = [t.strip() for t in args[i + 1].split(",") if t.strip()]
                i += 2
                continue
            if token == "--verified":
                patch["verified"] = True
                i += 1
                continue
            if token == "--conflict" and i + 1 < len(args):
                entry = self._store.get(entry_id)
                if entry is None:
                    raise KeyError(f"no flash entry {entry_id!r}")
                patch["conflicts"] = [*entry.conflicts, args[i + 1]]
                i += 2
                continue
            raise ValueError(f"unknown /flash correct option {token!r}")
        revised = self._store.update(entry_id, patch)
        if revised is None:
            raise KeyError(f"no flash entry {entry_id!r}")
        self._say(f"corrected {entry_id} (revision kept; history preserved)")
        return False

    def _flash_search(self, query: str) -> None:
        from fx1.flash.retrieve import retrieve

        hits = retrieve(self._store, query, k=8)
        if not hits:
            self._say(f"no flash context above threshold for {query!r}")
            return
        for hit in hits:
            self._say(
                f"{hit.entry.id[:8]}  score={hit.score:.3f}  {hit.entry.text[:100]}"
                f"  [{', '.join(hit.reasons)}]"
            )

    def _flash_refresh(self, entry_id: str) -> None:
        from fx1.flash.refresh import refresh

        report = refresh(self._store, entry_id, fetch_fn=None)
        if report is None:
            self._say_error(KeyError(f"no flash entry {entry_id!r}"))
            return
        self._say(report.note)

    # -- plain chat ----------------------------------------------------------

    def _chat_turn(self, text: str) -> None:
        backend = self._backend()
        if backend is None:
            self._say(
                f"no model endpoint configured — set one with `/keys set {self.model}`, or export "
                "MOONSHOT_API_KEY / FX1_BASE_URL. Slash commands still work."
            )
            return
        start = len(self._messages)
        self._messages.append({"role": "user", "content": text})
        tools = self._chat_tools()
        try:
            for _ in range(6):
                with (
                    OrbAnimator(OrbState.COMPOSING, preset="inline") if self._animated else _null()
                ):
                    completion = backend.complete_with_tools(
                        self._messages, tools=tools, tool_choice="auto"
                    )
                content = validate_fx1_output(completion.content or "")
                if not completion.tool_calls:
                    self._messages.append({"role": "assistant", "content": content})
                    self._say(content)
                    return
                self._messages.append(
                    {
                        "role": "assistant",
                        "content": completion.content,
                        "tool_calls": list(completion.tool_calls),
                    }
                )
                for call in completion.tool_calls:
                    result_text = self._execute_tool(call)
                    self._messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": result_text,
                        }
                    )
            self._say("(tool loop budget exhausted — ask a follow-up to continue)")
        except Fx1HonestyError as exc:
            refusal = (
                f"answer withheld by the fx-1 honesty gate ({exc}). Ask a follow-up to continue."
            )
            if any(message.get("role") == "tool" for message in self._messages[start:]):
                # A blocked answer cannot undo completed tool actions. Keep
                # their validated call/result transcript for the next turn.
                self._messages.append({"role": "assistant", "content": refusal})
            else:
                del self._messages[start:]
            self._say(refusal)
        except Exception as exc:  # noqa: BLE001 — a dead network must not kill the shell
            del self._messages[start:]
            self._say_error(exc)

    def _chat_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "flash_retrieve",
                    "description": (
                        "Search persistent research memory (flash context) for "
                        "prior findings relevant to a query. Returns findings with "
                        "sources, uncertainty, and conflicts."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_lookup",
                    "description": (
                        "A bounded public-web lookup: search and read up to two "
                        "sources. Use for facts outside the conversation; deep "
                        "research belongs to /research."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "flash_save",
                    "description": "Persist a finding to flash context for later sessions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "task": {"type": "string"},
                            "uncertainty": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "harness_run",
                    "description": (
                        "Run one registered dipcatcher harness command "
                        "(e.g. doctor, verify-research, research). Consequential "
                        "commands require operator approval. Fail-closed on "
                        "unknown names."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "args": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name"],
                    },
                },
            },
        ]

    def _execute_tool(self, call: dict[str, Any]) -> str:
        fn = call.get("function", {})
        name = fn.get("name", "")
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
            if not isinstance(arguments, dict):
                arguments = {}
        except json.JSONDecodeError:
            arguments = {}
        if name == "flash_retrieve":
            from fx1.flash.retrieve import context_block, retrieve

            hits = retrieve(self._store, str(arguments.get("query", "")), k=4)
            if not hits:
                return "no relevant flash context found"
            for hit in hits:
                self._store.mark_used(hit.entry.id)
            return context_block(hits)
        if name == "flash_save":
            entry = self._store.add(
                text=str(arguments.get("text", "")),
                task=str(arguments.get("task", "")),
                uncertainty=arguments.get("uncertainty"),
                tags=["model-saved"],
            )
            return f"saved flash entry {entry.id}"
        if name == "web_lookup":
            return self._web_lookup(str(arguments.get("query", "")))
        if name == "harness_run":
            harness_name = str(arguments.get("name", ""))
            args_list = [
                str(a) for a in arguments.get("args", []) if isinstance(a, (str, int, float))
            ]
            try:
                stdout = self._run_harness(harness_name, args_list)
            except KeyError as exc:
                return f"harness refused: {exc}"
            return stdout[-6000:] or "(command produced no output)"
        return f"unknown tool {name!r}"

    def _web_lookup(self, query: str) -> str:
        """One search + at most two fetches, capped — the chat-scale lookup."""
        from fx1.webresearch.engine import ResearchBudget

        if not query.strip():
            return "empty query"
        budget = ResearchBudget(
            max_duration_s=30,
            max_searches=1,
            max_sources=2,
            max_depth=0,
            search_pause_s=0.0,
        )
        report = self._researcher.run(query, budget=budget, queries=[query])
        parts: list[str] = []
        for source in report.sources:
            if source.ok:
                parts.append(f"SOURCE {source.url}\n{source.text[:2500]}")
        if not parts:
            return "web lookup found nothing (or the network is unavailable)"
        return "\n\n".join(parts)


class _null:
    """Context manager that does nothing (animation disabled / non-TTY)."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None


# ---------------------------------------------------------------------------
# Console driver
# ---------------------------------------------------------------------------


def _tty_polling_read(input_fn: Callable[[str], str], prompt: str, timeout_s: float) -> str | None:
    """Poll a POSIX TTY; Windows and non-TTY input use blocking reads.

    Returns None on timeout. The caller supplies an empty prompt while
    waiting for a previously prompted line.
    """
    if sys.platform == "win32" or not sys.stdin.isatty():
        return input_fn(prompt)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    line = sys.stdin.readline()
    if line == "":  # EOF
        raise EOFError
    return line.rstrip("\n")


def run_console(
    model: str = DEFAULT_MODEL,
    *,
    concierge_factory: Callable[..., Concierge] | None = None,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = typer.echo,
    poll_timeout_s: float = 0.15,
) -> None:
    """Read chat input with idle progress ticks on POSIX terminals.

    Windows and non-TTY input render queued progress between submitted lines.
    """
    concierge = (
        concierge_factory()
        if concierge_factory is not None
        else Concierge(
            model=model,
            input_fn=input_fn,
            output_fn=output_fn,
            approvals=ApprovalGate(ask_fn=console_ask_fn(input_fn)),
        )
    )
    model = concierge.model
    endpoint = profiles.resolve_endpoint(model)
    host = profiles.host_of(endpoint[1]) if endpoint else "unconfigured"
    output_fn(orb_banner(model, host, color=orbs_enabled()))
    output_fn(
        f"dipcatcher concierge — model {model}. Plain text chats; "
        "/superpower, /research, /flash, /help. Ctrl-D or /exit to leave."
    )
    if endpoint is None:
        output_fn(f"No endpoint configured. Run /keys set {model} to add your API key and URL.")
    if sys.platform == "win32":
        output_fn("On Windows, background progress updates after you submit a line.")
    prompt_pending = True
    while True:
        try:
            line = _tty_polling_read(input_fn, "dip › " if prompt_pending else "", poll_timeout_s)
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            concierge.shutdown()
            return
        if line is None:  # poll timeout — render background progress
            prompt_pending = False
            concierge.tick()
            continue
        prompt_pending = True
        try:
            if concierge.handle_line(line):
                return
        except Exception as exc:  # noqa: BLE001 — a bad turn must not kill the shell
            typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
        concierge.tick()
