"""Agent-based limit-order-book simulator.

Research and simulation only. Orders stay inside the process. Reports are
simulation diagnostics, not research-catalog headlines and not live results.

Public names load on first use so ``python -m quant_fund.market_sim`` can
answer ``--help`` and ``--version`` without importing the simulator.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "SCENARIOS",
    "EcologyConfig",
    "OrderBook",
    "core_version",
    "drought_config",
    "fit_impact_law",
    "lightspeed_momentum_weight",
    "matching_benchmark",
    "mean_reversion_weight",
    "measure_impact",
    "run_ecology",
    "run_scenario",
    "run_stylized_validation",
    "stress_strategy",
    "target_weight_replay",
    "validate_stylized_facts",
    "validation_config",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "OrderBook": ("quant_fund.market_sim.book", "OrderBook"),
    "EcologyConfig": ("quant_fund.market_sim.config", "EcologyConfig"),
    "drought_config": ("quant_fund.market_sim.config", "drought_config"),
    "validation_config": ("quant_fund.market_sim.config", "validation_config"),
    "lightspeed_momentum_weight": (
        "quant_fund.market_sim.harness",
        "lightspeed_momentum_weight",
    ),
    "mean_reversion_weight": ("quant_fund.market_sim.harness", "mean_reversion_weight"),
    "stress_strategy": ("quant_fund.market_sim.harness", "stress_strategy"),
    "target_weight_replay": ("quant_fund.market_sim.harness", "target_weight_replay"),
    "fit_impact_law": ("quant_fund.market_sim.impact", "fit_impact_law"),
    "measure_impact": ("quant_fund.market_sim.impact", "measure_impact"),
    "core_version": ("quant_fund.market_sim.native", "core_version"),
    "matching_benchmark": ("quant_fund.market_sim.native", "matching_benchmark"),
    "SCENARIOS": ("quant_fund.market_sim.scenarios", "SCENARIOS"),
    "run_scenario": ("quant_fund.market_sim.scenarios", "run_scenario"),
    "run_ecology": ("quant_fund.market_sim.simulator", "run_ecology"),
    "run_stylized_validation": (
        "quant_fund.market_sim.stylized",
        "run_stylized_validation",
    ),
    "validate_stylized_facts": (
        "quant_fund.market_sim.stylized",
        "validate_stylized_facts",
    ),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
