"""Small helpers.

Public names load on first use. Importing a submodule such as
``quant_fund.utils.cli_errors`` must not pull ``numpy`` through this
package, because the CLI cold-start budget forbids heavy libraries.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "atomic_write_bytes",
    "atomic_write_parquet",
    "atomic_write_text",
    "publish_text_once",
    "clip_positive",
    "configure_logging",
    "fingerprint",
    "get_logger",
    "hash_bytes",
    "hash_file",
    "require_finite",
    "set_global_seed",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "atomic_write_bytes": ("quant_fund.utils.atomicio", "atomic_write_bytes"),
    "atomic_write_parquet": ("quant_fund.utils.atomicio", "atomic_write_parquet"),
    "atomic_write_text": ("quant_fund.utils.atomicio", "atomic_write_text"),
    "publish_text_once": ("quant_fund.utils.atomicio", "publish_text_once"),
    "clip_positive": ("quant_fund.utils.numeric", "clip_positive"),
    "configure_logging": ("quant_fund.utils.logging", "configure_logging"),
    "fingerprint": ("quant_fund.utils.hashing", "fingerprint"),
    "get_logger": ("quant_fund.utils.logging", "get_logger"),
    "hash_bytes": ("quant_fund.utils.hashing", "hash_bytes"),
    "hash_file": ("quant_fund.utils.hashing", "hash_file"),
    "require_finite": ("quant_fund.utils.numeric", "require_finite"),
    "set_global_seed": ("quant_fund.utils.seeds", "set_global_seed"),
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
