"""Command line for the scenario-risk engine.

Research simulation only. The report is JSON on stdout. Progress lines go to
stderr and are not part of the document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from quant_fund.mc_engine.benchmark import scaling_benchmark
from quant_fund.mc_engine.engine import EngineConfig, resume_simulation, run_simulation
from quant_fund.mc_engine.scenario import (
    GbmPortfolioGenerator,
    ScenarioGenerator,
    VolTargetStrategyGenerator,
    generator_from_spec,
)
from quant_fund.utils.cli_errors import McEngineTyperGroup

app = typer.Typer(
    cls=McEngineTyperGroup,
    help=(
        "Monte Carlo scenario risk. Research simulation only. "
        "Does not submit orders and does not claim live profit. "
        "--version on any command prints the version and exits."
    ),
    no_args_is_help=True,
)


def _floats(text: str) -> list[float]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("expected a comma-separated list of numbers")
    return [float(part) for part in parts]


def _dump(document: dict[str, object]) -> None:
    typer.echo(json.dumps(document, indent=2, sort_keys=True, allow_nan=False))


def _progress(event: dict[str, float | int]) -> None:
    typer.echo(
        "progress "
        + json.dumps(
            {key: event[key] for key in sorted(event)},
            sort_keys=True,
            allow_nan=False,
        ),
        err=True,
    )


@app.command("run")
def run_cmd(
    paths: int = typer.Option(100_000, "--paths", min=1),
    steps: int = typer.Option(252, "--steps", min=1),
    workers: int = typer.Option(1, "--workers", min=1),
    chunk_size: int = typer.Option(4096, "--chunk-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    backend: str = typer.Option("process", "--backend"),
    shock_mode: str = typer.Option("crude", "--shock-mode"),
    memory_mode: str = typer.Option("exact", "--memory-mode"),
    strategy: str = typer.Option("portfolio", "--strategy"),
    mu: str = typer.Option("0.0", "--mu"),
    vol: str = typer.Option("0.16", "--vol"),
    corr: float = typer.Option(0.0, "--corr"),
    weights: str = typer.Option("", "--weights"),
    importance_shift: float | None = typer.Option(None, "--importance-shift"),
    ruin_level: float = typer.Option(0.5, "--ruin-level"),
    control_variate: bool = typer.Option(False, "--control-variate"),
    scrambles: int = typer.Option(1, "--scrambles", min=1),
    evt_threshold: float | None = typer.Option(None, "--evt-threshold"),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
    target_vol: float = typer.Option(0.10, "--target-vol"),
    lookback: int = typer.Option(21, "--lookback", min=2),
    progress: bool = typer.Option(True, "--progress/--no-progress"),
) -> None:
    """Simulate scenario paths and print a tail-risk report."""
    drifts = _floats(mu)
    vols = _floats(vol)
    if len(drifts) != len(vols):
        raise typer.BadParameter("--mu and --vol must have the same length")
    generator: ScenarioGenerator
    if strategy == "vol-target":
        if len(drifts) != 1:
            raise typer.BadParameter("vol-target uses one asset")
        generator = VolTargetStrategyGenerator(
            mu=drifts[0],
            sigma=vols[0],
            n_steps=steps,
            target_vol=target_vol,
            lookback=lookback,
        )
    elif strategy == "portfolio":
        if abs(corr) >= 1.0 and len(vols) > 1:
            raise typer.BadParameter("--corr must be in (-1, 1)")
        covariance = [
            [vols[i] * vols[j] * (1.0 if i == j else corr) for j in range(len(vols))]
            for i in range(len(vols))
        ]
        if weights:
            weight_values = _floats(weights)
        else:
            weight_values = [1.0 / len(vols)] * len(vols)
        generator = GbmPortfolioGenerator(
            mu=np.asarray(drifts, dtype=np.float64),
            covariance=np.asarray(covariance, dtype=np.float64),
            weights=np.asarray(weight_values, dtype=np.float64),
            n_steps=steps,
        )
    else:
        raise typer.BadParameter("--strategy must be portfolio or vol-target")
    config = EngineConfig(
        n_paths=paths,
        chunk_size=chunk_size,
        seed=seed,
        workers=workers,
        backend=backend,
        shock_mode=shock_mode,
        importance_shift=importance_shift,
        qmc_scramble=True,
        n_scrambles=scrambles,
        control_variate=control_variate,
        memory_mode=memory_mode,
        ruin_level=ruin_level,
        evt_threshold=evt_threshold,
        checkpoint_dir=None if checkpoint is None else str(checkpoint),
    )
    report = run_simulation(generator, config, progress=_progress if progress else None)
    _dump(report)
    if report.get("status") != "complete":
        raise typer.Exit(code=2)


@app.command("resume")
def resume_cmd(
    checkpoint: Path = typer.Option(..., "--checkpoint", exists=True, file_okay=False),
    workers: int = typer.Option(1, "--workers", min=1),
    backend: str = typer.Option("process", "--backend"),
) -> None:
    """Continue a built-in generator checkpoint and print the report."""
    manifest_path = checkpoint / "manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"no manifest in {checkpoint}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spec = manifest.get("generator_spec")
    if not isinstance(spec, dict):
        raise typer.BadParameter("checkpoint has no generator_spec")
    generator = generator_from_spec(spec)
    report = resume_simulation(
        checkpoint,
        generator,
        workers=workers,
        backend=backend,
        progress=_progress,
    )
    _dump(report)
    if report.get("status") != "complete":
        raise typer.Exit(code=2)


@app.command("bench")
def bench_cmd(
    paths: int = typer.Option(20_000, "--paths", min=1),
    steps: int = typer.Option(32, "--steps", min=1),
    workers: str = typer.Option("1,2", "--workers"),
    repeats: int = typer.Option(2, "--repeats", min=1),
    chunk_size: int = typer.Option(4096, "--chunk-size", min=1),
    seed: int = typer.Option(0, "--seed", min=0),
    backend: str = typer.Option("process", "--backend"),
) -> None:
    """Measure wall-clock throughput at each worker count. Prints JSON."""
    counts = [int(part) for part in _floats(workers)]
    document = scaling_benchmark(
        n_paths=paths,
        n_steps=steps,
        worker_counts=counts,
        repeats=repeats,
        seed=seed,
        chunk_size=chunk_size,
        backend=backend,
    )
    _dump(document)


@app.command("help-interface")
def help_interface_cmd() -> None:
    """Print the scenario-generator contract for plug-in engines."""
    typer.echo(
        "\n".join(
            [
                "ScenarioGenerator.generate(path_indices, *, shocks, seed) -> ScenarioBatch",
                "returns: float64 array (n_paths, n_steps) of simple portfolio returns",
                "accepts_external_shocks True: consume shocks, do not draw your own",
                "accepts_external_shocks False: only shock_mode=crude; draw with philox_* ",
                "and stream_id >= USER_STREAM_ID_MIN, keyed by path_indices",
                "spec_dict() must be JSON-stable; it is the checkpoint fingerprint",
                "data_source: SYNTHETIC or the real source name; market_evidence stays false",
                "research_only: this CLI never submits orders",
            ]
        )
    )


if __name__ == "__main__":
    app()
