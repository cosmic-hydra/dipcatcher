"""``fxi`` — the interactive front door for fx-1.

fx-1 ships two hosted model names — ``fx1`` and ``fx1-lite`` — and each has
one endpoint here: an API key plus a base URL, stored in
``~/.fx1/credentials.json`` (mode 0600) and injected into the harness as
``MOONSHOT_API_KEY`` + ``FX1_BASE_URL``.

Bare ``fxi`` opens the interactive shell; every command is also available
one-shot for scripting. First-time setup lives in ``fx1.interactive.wizard``
(bare ``dipcatcher`` launches it too).
"""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import NoReturn

import typer

import fx1
from fx1.cli_errors import FxiTyperGroup
from fx1.interactive import actions, concierge, profiles, shell, wizard

app = typer.Typer(
    cls=FxiTyperGroup,
    name="fxi",
    help=(
        "Interactive CLI for fx-1. Bare `fxi` opens the concierge — chat, "
        "/superpower orchestration, background web research, and flash "
        "context. Store fx1 / fx1-lite model endpoints (API key + base URL) "
        "and use them across the dipcatcher harness. "
        "--version on any command prints the version and exits."
    ),
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

keys_app = typer.Typer(help="Manage fx-1 model endpoints (presence-only output).")
harness_app = typer.Typer(help="Registered dipcatcher harness commands (fail-closed).")
flash_app = typer.Typer(
    help="Persistent research memory: add, list, show, correct, remove, search, refresh."
)
app.add_typer(keys_app, name="keys")
app.add_typer(harness_app, name="harness")
app.add_typer(flash_app, name="flash")

state: dict[str, str] = {"model": profiles.DEFAULT_MODEL}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"fxi {fx1.__version__}")
        raise typer.Exit()


def _fail(exc: Exception, code: int = 2) -> NoReturn:
    typer.secho(f"error: {exc}", err=True, fg=typer.colors.RED)
    raise typer.Exit(code=code)


@app.callback(invoke_without_command=True)
def callback(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None,
        "--model",
        "--profile",
        "-p",
        help="Active model for this invocation: fx1 | fx1-lite.",
    ),
    version: bool | None = typer.Option(
        None,
        "--version",
        "-V",
        help="Show the fx-1 version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    if model is not None:
        try:
            profiles.check_model(model)
        except KeyError as exc:
            raise typer.BadParameter(str(exc)) from None
        state["model"] = model
    if ctx.invoked_subcommand is None:
        # Bare `fxi` (or `fxi --model ...`) opens the concierge console:
        # chat, /superpower, background web research, flash context.
        concierge.run_console(state["model"])


@app.command("shell")
def shell_cmd() -> None:
    """Open the classic fxi shell (keys/harness/eval one-liners)."""
    shell.run_shell(state["model"])


@app.command("setup")
def setup_cmd() -> None:
    """Run the endpoint setup wizard (API key, base URL), then enter the console."""
    wizard.enter(force=True)


@keys_app.command("list")
def keys_list() -> None:
    """List model endpoints: model, host, status, fingerprint, set_at."""
    typer.echo(actions.render_table(profiles.list_models()))


@keys_app.command("set")
def keys_set(
    model: str = typer.Argument(..., help="fx1 | fx1-lite"),
    value: str | None = typer.Option(
        None,
        "--value",
        help=(
            "The API key. Omit to be prompted (input not echoed). Prefer the "
            "prompt: --value lands in shell history."
        ),
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Endpoint base URL (default: the pinned Moonshot chat-completions URL).",
    ),
) -> None:
    """Store an endpoint for a model (~/.fx1/credentials.json, mode 0600)."""
    try:
        profiles.check_model(model)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from None
    if value is None:
        value = getpass.getpass(f"› {model} API key: ")
    try:
        path = profiles.set_endpoint(model, value, base_url)
    except ValueError as exc:
        _fail(exc)
    resolved = profiles.resolve_endpoint(model)
    host = profiles.host_of(resolved[1]) if resolved else "unknown"
    typer.echo(f"stored endpoint for {model!r} -> {host} in {path} (mode 0600)")


@keys_app.command("remove")
def keys_remove(model: str = typer.Argument(..., help="fx1 | fx1-lite")) -> None:
    """Delete a stored endpoint."""
    try:
        removed = profiles.remove_endpoint(model)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(f"removed endpoint {model!r}" if removed else f"endpoint {model!r} was not stored")


@app.command("doctor")
def doctor_cmd() -> None:
    """Readiness: fx-1 state, endpoints (presence-only), store, API reachability."""
    typer.echo(actions.status_json(actions.doctor(state["model"])))


@app.command("eval")
def eval_cmd(
    out: Path | None = typer.Option(None, "--out", help="Defaults to data/fx1/eval.json."),
) -> None:
    """Run the fx-1 eval bank against the active model's endpoint."""
    try:
        code = actions.run_eval(out, state["model"])
    except (KeyError, RuntimeError) as exc:
        _fail(exc)
    raise typer.Exit(code=code)


@app.command("chat")
def chat_cmd(
    prompt: str | None = typer.Option(
        None, "--prompt", "-m", help="One-shot prompt; omit for a multi-turn session."
    ),
) -> None:
    """Chat with the active model (hosted endpoint)."""
    try:
        actions.chat(state["model"], prompt)
    except RuntimeError as exc:
        _fail(exc)


@harness_app.command("list")
def harness_list() -> None:
    """List registered dipcatcher harness commands."""
    typer.echo(actions.render_table(actions.harness_rows()))


@harness_app.command("describe")
def harness_describe(
    name: str = typer.Argument(..., help="Registered harness command name."),
) -> None:
    """Show one registered harness command's metadata."""
    try:
        meta = actions.harness_get(name)
    except KeyError as exc:
        _fail(exc)
    typer.echo(actions.status_json(meta))


@harness_app.command("run")
def harness_run_cmd(
    name: str = typer.Argument(..., help="Registered harness command name."),
    extra_args: list[str] = typer.Argument(None, help="Extra argv for the command."),
) -> None:
    """Execute a registered harness command with the active model's endpoint in env."""
    try:
        code = actions.harness_run(name, extra_args, state["model"])
    except (KeyError, RuntimeError) as exc:
        _fail(exc)
    raise typer.Exit(code=code)


@app.command("verify")
def verify_cmd(
    path: Path = typer.Argument(..., help="Receipt file, artifact, or directory."),
) -> None:
    """Verify research provenance via `dipcatcher verify-research`."""
    try:
        code = actions.verify(path, state["model"])
    except (KeyError, RuntimeError) as exc:
        _fail(exc)
    raise typer.Exit(code=code)


@app.command("run")
def run_cmd(
    args: list[str] = typer.Argument(..., metavar="TOOL ARGS..."),
) -> None:
    """Launch a shipped tool (fx1 | dipcatcher | quant | ...) with the session env."""
    tool, rest = args[0], args[1:]
    try:
        code = actions.run_tool(tool, rest, state["model"])
    except (KeyError, RuntimeError) as exc:
        _fail(exc)
    raise typer.Exit(code=code)


@app.command("superpower")
def superpower_cmd(
    goal: list[str] = typer.Argument(None, help="The research goal (plain words)."),
    sync: bool = typer.Option(False, "--sync", help="Run inline instead of in the background."),
    help_catalog: bool = typer.Option(
        False, "--capabilities", help="List the capabilities the planner may select."
    ),
) -> None:
    """Plan, select, coordinate, and explain research capabilities for a goal."""
    from fx1.interactive.superpower import describe_capabilities

    if help_catalog:
        typer.echo(describe_capabilities())
        raise typer.Exit()
    if not goal:
        raise typer.BadParameter("a research goal is required (or use --capabilities)")
    goal_text = " ".join(goal)
    operator = concierge.Concierge(model=state["model"])
    operator._slash(f"/superpower {goal_text}{' --sync' if sync else ''}")  # noqa: SLF001
    operator.join_job(timeout_s=3600)


@app.command("research")
def research_cmd(
    goal: list[str] = typer.Argument(..., help="The research goal."),
    budget_s: int = typer.Option(300, "--budget-s", help="Wall-clock budget in seconds."),
    max_sources: int = typer.Option(8, "--max-sources", help="Maximum sources to fetch."),
    queries: str | None = typer.Option(
        None, "--queries", help="Pipe-separated seed queries (default: derived from goal)."
    ),
    sync: bool = typer.Option(False, "--sync", help="Run inline instead of in the background."),
) -> None:
    """Deep background web research: search, read, cross-check, report."""
    goal_text = " ".join(goal)
    extra = [f"--budget-s {budget_s}", f"--max-sources {max_sources}"]
    if queries:
        extra.append(f"--queries {queries}")
    if sync:
        extra.append("--sync")
    operator = concierge.Concierge(model=state["model"])
    operator._slash(f"/research {goal_text} {' '.join(extra)}")  # noqa: SLF001
    operator.join_job(timeout_s=budget_s + 30)


@flash_app.command("add")
def flash_add(
    text: list[str] = typer.Argument(..., help="The finding (plain words)."),
    task: str = typer.Option("", "--task", help="Task/goal this finding serves."),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags."),
    source: str = typer.Option("", "--source", help="Source URL."),
    uncertainty: str = typer.Option("", "--uncertainty", help="low | medium | high."),
    private: bool = typer.Option(False, "--private", help="Never share without approval."),
    verified: bool = typer.Option(False, "--verified", help="Mark as verified."),
) -> None:
    """Persist a finding with provenance into flash context."""
    from fx1.flash.store import FlashStore, Uncertainty

    if uncertainty and uncertainty not in {"low", "medium", "high"}:
        raise typer.BadParameter("--uncertainty must be low | medium | high")
    from typing import cast

    uncertainty_value = cast(
        "Uncertainty | None",
        uncertainty if uncertainty in {"low", "medium", "high"} else None,
    )
    entry = FlashStore().add(
        text=" ".join(text),
        task=task,
        tags=[t.strip() for t in tags.split(",") if t.strip()],
        sources=[{"url": source}] if source else [],
        uncertainty=uncertainty_value,
        private=private,
        verified=verified,
    )
    typer.echo(f"saved flash entry {entry.id}")


@flash_app.command("list")
def flash_list(
    query: str | None = typer.Argument(None, help="Optional query to rank by relevance."),
) -> None:
    """List flash entries (optionally ranked against a query)."""
    from fx1.flash.store import FlashStore

    store = FlashStore()
    if query:
        from fx1.flash.retrieve import retrieve

        entries = [hit.entry for hit in retrieve(store, query, k=20, min_score=0.0)]
    else:
        entries = store.all()
    for entry in entries:
        typer.echo(
            f"{entry.id}  {entry.updated_at[:10]}  "
            f"{('[' + entry.uncertainty + '] ') if entry.uncertainty else ''}{entry.text[:100]}"
        )


@flash_app.command("show")
def flash_show(entry_id: str = typer.Argument(..., help="Entry id (or prefix).")) -> None:
    """Show one flash entry with full provenance."""
    from fx1.flash.store import FlashStore

    entry = FlashStore().get(entry_id)
    if entry is None:
        _fail(KeyError(f"no flash entry {entry_id!r}"))
    typer.echo(entry.model_dump_json(indent=2))


@flash_app.command("remove")
def flash_remove(entry_id: str = typer.Argument(..., help="Entry id (or prefix).")) -> None:
    """Remove a flash entry (tombstoned, not rewritten)."""
    from fx1.flash.store import FlashStore

    removed = FlashStore().remove(entry_id)
    typer.echo(f"removed {entry_id!r}" if removed else f"no flash entry {entry_id!r}")


@flash_app.command("search")
def flash_search(query: list[str] = typer.Argument(..., help="Query terms.")) -> None:
    """Rank flash entries by relevance to a query."""
    from fx1.flash.retrieve import retrieve
    from fx1.flash.store import FlashStore

    hits = retrieve(FlashStore(), " ".join(query), k=10)
    for hit in hits:
        typer.echo(
            f"{hit.entry.id}  score={hit.score:.3f}  {hit.entry.text[:100]}"
            f"  [{', '.join(hit.reasons)}]"
        )


@flash_app.command("refresh")
def flash_refresh(entry_id: str = typer.Argument(..., help="Entry id (or prefix).")) -> None:
    """Re-fetch an entry's sources and record reachability honestly."""
    from fx1.flash.refresh import refresh
    from fx1.flash.store import FlashStore

    report = refresh(FlashStore(), entry_id)
    if report is None:
        _fail(KeyError(f"no flash entry {entry_id!r}"))
    typer.echo(report.note)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
