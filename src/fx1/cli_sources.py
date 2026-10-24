"""Datasource commands for the fx1 CLI.

Split out of ``fx1.cli`` so that module stays inside its line budget.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer


def register_source_commands(sources_app: typer.Typer, corpus_app: typer.Typer) -> None:
    """Attach the datasource commands to the fx1 Typer groups."""
    sources_app.command("list")(sources_list)
    sources_app.command("probe")(sources_probe)
    sources_app.command("describe")(sources_describe)
    sources_app.command("fetch")(sources_fetch)
    sources_app.command("route")(sources_route)
    sources_app.command("scenarios")(sources_scenarios)
    corpus_app.command("ingest-source")(corpus_ingest_source)


def sources_list() -> None:
    """List every registered datasource with its live availability probe."""
    from fx1.data.sources.registry import list_sources
    from fx1.data.sources.router import probe_names

    specs = list_sources()
    probes = probe_names([spec.name for spec in specs])
    for spec, probe in zip(specs, probes, strict=True):
        typer.echo(
            f"{spec.name:<16} [{probe.status.value:<17}] "
            f"{spec.display} — {','.join(spec.markets)} / "
            f"{','.join(spec.assets)}"
        )


def sources_probe(
    name: str | None = typer.Argument(None, help="Source name (default: all)."),
) -> None:
    """Probe availability (script + credentials) without leaking secrets."""
    from fx1.data.sources.registry import get_spec, list_sources, roots_status
    from fx1.data.sources.router import probe_names

    specs = [get_spec(name)] if name else list_sources()
    probes = probe_names([spec.name for spec in specs])
    report = {
        "roots": roots_status(),
        "probes": [probe.model_dump() for probe in probes],
    }
    typer.echo(json.dumps(report, indent=2))


def sources_describe(name: str = typer.Argument(...)) -> None:
    """Print the source's own capability docs (straight from its CLI)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


def sources_fetch(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}", help="JSON object of API params."),
    as_of: str | None = typer.Option(None, help="Observation date YYYY-MM-DD."),
    timeout: float = typer.Option(120.0, help="Timeout seconds (≤600)."),
) -> None:
    """Fetch from one datasource. Failure is reported, never patched over."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.registry import get_spec

    params = json.loads(params_json)
    if not isinstance(params, dict):
        typer.echo("--params-json must be a JSON object", err=True)
        raise typer.Exit(code=2)
    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=params, as_of=as_of, timeout_s=timeout)
    )
    if result.ok:
        typer.echo(result.text)
        typer.echo(
            json.dumps(
                {
                    "payload_sha256": result.payload_sha256,
                    "fetched_at": result.fetched_at,
                    "elapsed_ms": result.elapsed_ms,
                },
                indent=2,
            ),
            err=True,
        )
    else:
        typer.echo(f"fetch failed: {result.error}", err=True)
        raise typer.Exit(code=1)


def sources_route(
    question: str = typer.Option(..., help="Natural-language research question."),
    market: str | None = typer.Option(None, help="cn | hk | us | global | crypto"),
    need: str | None = typer.Option(None, help="Override need classification."),
) -> None:
    """Show the routing plan for a question (classified need + probed candidates)."""
    from fx1.data.sources.router import route

    plan = route(question, market=market, need=need)
    typer.echo(plan.model_dump_json(indent=2))


def sources_scenarios() -> None:
    """List finance-fetch scenario coverage (statements, consensus, peers…)."""
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec("finance_fetch")).describe()
    if result.ok:
        typer.echo(result.text)
    else:
        typer.echo(f"unavailable: {result.error}", err=True)
        raise typer.Exit(code=1)


def corpus_ingest_source(
    name: str = typer.Argument(..., help="Source name."),
    api: str = typer.Option(..., help="Source-native API/tool name."),
    params_json: str = typer.Option("{}"),
    as_of: str | None = typer.Option(None, help="Observation date (gated)."),
    ledger_path: Path = typer.Option(Path("data/fx1/corpus_ledger.jsonl")),
    out: Path = typer.Option(Path("data/fx1/corpus.jsonl")),
) -> None:
    """Fetch → gate → append to corpus → chain into the ledger.

    Refuses (exit 1) when the fetch fails or the payload lacks an as_of date
    for time-stamped sources — no leakage, no fabrication.
    """
    from fx1.data import CorpusLedger
    from fx1.data.corpus import _system_prompt
    from fx1.data.sources.adapters import build_adapter
    from fx1.data.sources.base import FetchRequest
    from fx1.data.sources.ingest import fetch_to_example, record_fetch_in_ledger
    from fx1.data.sources.registry import get_spec

    result = build_adapter(get_spec(name)).fetch(
        FetchRequest(api=api, params=json.loads(params_json), as_of=as_of)
    )
    decision = fetch_to_example(result, _system_prompt())
    ledger = CorpusLedger(ledger_path)
    record_fetch_in_ledger(ledger, decision)
    if not decision.accepted or decision.example is None:
        typer.echo(f"ingest refused: {decision.reason}", err=True)
        raise typer.Exit(code=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(decision.example.model_dump_json() + "\n")
    typer.echo(
        json.dumps(
            {
                "accepted": True,
                "negative": decision.negative,
                "reason": decision.reason,
                "payload_sha256": decision.payload_sha256,
                "ledger": ledger.audit_export(),
            },
            indent=2,
        )
    )
