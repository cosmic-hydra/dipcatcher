"""Typer app, sub-CLI mounts, and shared helpers for the quant CLI.

``quant_fund.cli.app.app`` is this same ``app`` object.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from quant_fund.cli.benchmark_cmds import (
    net_tournament_app,
    ranker_probability_app,
    real_benchmark_app,
)

# SOTA / forward-evidence operator surface. Restored on this branch after the
# mount was found missing; independently re-added upstream by 8eb627ff2. Kept
# here because ``dipcatcher forward-shadow freeze`` -- the command
# ``docs/INSTITUTIONAL_READINESS.md`` names for closing institutional
# condition 3 -- is unreachable without it. ``sota_cmds`` imports only stdlib +
# typer at module level (shared helpers are lazy), so this cannot reintroduce the
# ``cli.support -> cli._app`` cycle its docstring warns about.
from quant_fund.cli.sota_cmds import (
    forward_shadow_app,
    prospective_sota_app,
    sota_app,
)
from quant_fund.hmm.cli import hmm_app as hmm_app
from quant_fund.leakage.cli import leakage_app
from quant_fund.lightspeed.cli import ls_app as ls_app
from quant_fund.pit.cli import pit_app
from quant_fund.proof.cli import proof_app
from quant_fund.proofcore.cli import proofcore_app
from quant_fund.quant_models.cli import qm_app as qm_app
from quant_fund.reality.cli import reality_app
from quant_fund.research.allocation_cli import allocation_app
from quant_fund.research.explainability_cli import explainability_app
from quant_fund.research.research100_cli import research100_app
from quant_fund.stress.cli import stress_app
from quant_fund.utils.cli_errors import DipcatcherTyperGroup

if TYPE_CHECKING:
    from quant_fund.config.models import AppConfig


def format_data_label(*, synthetic: bool, data_source: str) -> str:
    """Always-printed DATA_LABEL line for research / paper CLI output."""
    return f"DATA_LABEL={'SYNTHETIC' if synthetic else data_source}"


def format_fdr_families(hypotheses: Sequence[object]) -> str:
    """BH-FDR family split summary — calibration/discovery/bound never pooled."""
    cal_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "calibration")
    disc_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "discovery")
    bound_n = sum(1 for h in hypotheses if getattr(h, "family", None) == "bound")
    cal = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "calibration" and getattr(h, "reject_fdr", False)
    )
    disc = sum(
        1
        for h in hypotheses
        if getattr(h, "family", None) == "discovery" and getattr(h, "reject_fdr", False)
    )
    return (
        "BH-FDR families (split, never pooled): "
        f"calibration n={cal_n} rejects={cal}; "
        f"discovery n={disc_n} rejects={disc}; "
        f"bound n={bound_n} (not FDR-adjusted)"
    )


app = typer.Typer(
    cls=DipcatcherTyperGroup,
    help=(
        "Dipcatcher — Artificial Hedge's proprietary research lab. "
        "Default mode is research, never live. "
        "--version on any command prints the version and exits."
    ),
)

# The fx-1 interactive front door is the `fxi` console script (pyproject.toml:
# fxi = "fx1.interactive.app:main"; see docs/FXI.md). It is deliberately NOT
# wired into bare `dipcatcher`: doing so requires a quant_fund -> fx1 import,
# which configs/arch_boundaries.toml denies under `harness-no-fx1-imports` and
# tests/unit/test_fx1_dependency_edge.py enforces. The harness gates the model;
# it never imports it. The only sanctioned seam is fx1.__version__ in
# src/quant_fund/__init__.py, which hatch reads for the distribution version.


def _emit_version(value: bool) -> None:
    if not value:
        return
    from quant_fund import __version__

    typer.echo(f"dipcatcher {__version__}")
    raise typer.Exit()


@app.callback(invoke_without_command=True)
def _bare_help(
    ctx: typer.Context,
    _version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the dipcatcher version and exit.",
        is_eager=True,
        callback=_emit_version,
    ),
) -> None:
    """Bare ``dipcatcher`` prints help and exits 0.

    Click exits 2 when a root Typer app has only sub-typers and no direct
    commands, even with ``no_args_is_help=True``. This callback restores the
    expected help-and-zero behavior without importing any fx1 code.
    ``--version`` is also accepted on every subcommand.
    """
    del _version
    if ctx.invoked_subcommand is None:
        ctx.get_help()
        raise typer.Exit()


train_app = typer.Typer(help="Train a forecast family.")
app.add_typer(train_app, name="train")
app.add_typer(hmm_app, name="hmm")
app.add_typer(ls_app, name="ls")
app.add_typer(pit_app, name="pit")
app.add_typer(qm_app, name="qm")
app.add_typer(research100_app, name="research100")
app.add_typer(proof_app, name="proof")
app.add_typer(leakage_app, name="leakage")
app.add_typer(reality_app, name="reality")
app.add_typer(proofcore_app, name="proofcore")
app.add_typer(stress_app, name="stress")
app.add_typer(allocation_app, name="allocation")
app.add_typer(explainability_app, name="explain")
app.add_typer(prospective_sota_app, name="prospective-sota")
app.add_typer(forward_shadow_app, name="forward-shadow")
app.add_typer(sota_app, name="sota")
app.add_typer(real_benchmark_app, name="real-benchmark")
app.add_typer(net_tournament_app, name="net-tournament")
app.add_typer(ranker_probability_app, name="ranker-probability")


def _cfg(config: Path) -> AppConfig:
    from quant_fund.config import dump_resolved, load_config
    from quant_fund.utils.logging import configure_logging

    cfg = load_config(config)
    configure_logging()
    dump_resolved(cfg, Path(cfg.data.root) / "metadata" / "resolved_config.json")
    return cfg


def _collect_param_value(raw: str) -> object:
    """Coerce a --param value to int/float when it cleanly parses, else str."""
    text = raw.strip()
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
