"""quant CLI."""

# Command modules register Typer callbacks at import time. Keep this order.
# ruff: noqa: I001

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from quant_fund.cli._app import (
    hmm_app as hmm_app,
    ls_app as ls_app,
    qm_app as qm_app,
)
from quant_fund.cli.app import app, train_app
from quant_fund.cli.support import (
    _cfg as _cfg,
    _collect_param_value as _collect_param_value,
    format_data_label,
    format_fdr_families,
)
from quant_fund.cli.audit_cmds import (
    audit_record,
    audit_trace,
    verify_ledger_cmd,
)
from quant_fund.cli.data_cmds import (
    doctor,
    ingest,
    collect,
    build_features_cmd,
    build_labels_cmd,
    membership_coverage_cmd,
    total_return_cmd,
)
from quant_fund.cli.forecast_cmds import (
    validate,
    forecast,
    kronos_forecast,
    optimize,
    backtest,
)
from quant_fund.cli.micro_cmds import (
    candle_book,
    kyle_ofi,
)
from quant_fund.cli.research_cmds import (
    research,
    execution_sensitivity_cmd,
    verify_all_cmd,
    verify_identities,
    verify_receipt_cmd,
    fleet,
    calibration_eval_cmd,
    vol_bench,
    capacity,
    pairs,
    serial_watch_cmd,
    mcs,
    lane_power,
    suite_health_cmd,
    cost_calibration,
    corpus,
    lattice_cmd,
    graph_cmd,
    online_fdr_cmd,
    replay_cmd,
    tape_pin_cmd,
    tape_verify_cmd,
    coherence,
    concordance,
    multih_fleet_cmd,
    expert_mixture_cmd,
    compare,
)
from quant_fund.cli.book_cmds import (
    session_book_cmd,
    vendor_book_map_cmd,
    book_panel_cmd,
    northset,
)
from quant_fund.cli.report_cmds import (
    verify_research,
    lab,
    report,
    tearsheet_cmd,
    regime_performance_cmd,
)
from quant_fund.cli.lake_cmds import (
    lake_import,
    lake_quality,
    lineage_show,
    lineage_verify,
)
from quant_fund.cli.ops_cmds import (
    api,
    paper,
    monitor,
    sim_live,
    trade,
)
from quant_fund.cli.live_cmds import (
    live_app as live_app,
)
from quant_fund.cli.train_cmds import (
    _train as _train,
    train_callback,
    train_family,
    train_ranking,
    train_distribution,
    train_calibration,
    train_volatility,
    train_alpha,
    train_covariance,
    train_regime,
    train_tail,
    train_reinforcement,
    train_liquidity,
)


def dump_resolved(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export. Importing the CLI must not load config."""
    from quant_fund.config import dump_resolved as _dump_resolved

    return _dump_resolved(*args, **kwargs)


def load_config(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export. Importing the CLI must not load config."""
    from quant_fund.config import load_config as _load_config

    return _load_config(*args, **kwargs)


def configure_logging(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export. Importing the CLI must not load logging backends."""
    from quant_fund.utils.logging import configure_logging as _configure_logging

    return _configure_logging(*args, **kwargs)


def get_logger(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export. Tests patch ``quant_fund.cli.data_cmds.get_logger``."""
    from quant_fund.utils.logging import get_logger as _get_logger

    return _get_logger(*args, **kwargs)


def run_doctor(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export. Importing the CLI must not load the doctor pipeline."""
    from quant_fund.pipeline.doctor import doctor as _doctor

    return _doctor(*args, **kwargs)


__all__ = [
    "format_data_label",
    "format_fdr_families",
    "audit_record",
    "audit_trace",
    "verify_ledger_cmd",
    "app",
    "train_app",
    "doctor",
    "ingest",
    "collect",
    "build_features_cmd",
    "build_labels_cmd",
    "membership_coverage_cmd",
    "total_return_cmd",
    "train_callback",
    "train_ranking",
    "train_distribution",
    "train_calibration",
    "train_volatility",
    "train_alpha",
    "train_covariance",
    "train_regime",
    "train_tail",
    "train_reinforcement",
    "train_liquidity",
    "validate",
    "forecast",
    "kronos_forecast",
    "optimize",
    "backtest",
    "candle_book",
    "kyle_ofi",
    "research",
    "execution_sensitivity_cmd",
    "verify_all_cmd",
    "verify_identities",
    "verify_receipt_cmd",
    "fleet",
    "calibration_eval_cmd",
    "vol_bench",
    "capacity",
    "pairs",
    "serial_watch_cmd",
    "mcs",
    "lane_power",
    "suite_health_cmd",
    "cost_calibration",
    "corpus",
    "lattice_cmd",
    "graph_cmd",
    "online_fdr_cmd",
    "replay_cmd",
    "tape_pin_cmd",
    "tape_verify_cmd",
    "coherence",
    "concordance",
    "multih_fleet_cmd",
    "expert_mixture_cmd",
    "compare",
    "session_book_cmd",
    "vendor_book_map_cmd",
    "book_panel_cmd",
    "northset",
    "verify_research",
    "lab",
    "report",
    "tearsheet_cmd",
    "regime_performance_cmd",
    "lake_import",
    "lake_quality",
    "lineage_show",
    "lineage_verify",
    "api",
    "paper",
    "monitor",
    "sim_live",
    "live_app",
    "trade",
    "Path",
    "typer",
    "dump_resolved",
    "load_config",
    "hmm_app",
    "ls_app",
    "run_doctor",
    "train_family",
    "qm_app",
    "configure_logging",
    "get_logger",
]


def _maybe_install_observation_hooks() -> None:
    """Wrap pipeline entry points only when the operator opted in."""
    import os

    flag = os.environ.get("DIPCATCHER_OBSERVE", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return
    from quant_fund.observe.install import install_passive_hooks

    install_passive_hooks()


_maybe_install_observation_hooks()

if __name__ == "__main__":
    app()
