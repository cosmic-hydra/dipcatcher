from quant_fund.utils.atomicio import (
    atomic_write_bytes,
    atomic_write_parquet,
    atomic_write_text,
    publish_text_once,
)
from quant_fund.utils.hashing import fingerprint, hash_bytes, hash_file
from quant_fund.utils.logging import configure_logging, get_logger

# ``numeric`` and ``seeds`` import numpy. Keep them off ``import quant_fund.utils``
# so the CLI cold-start path, which loads this package for user-error handling,
# does not pull a heavy library.


def __getattr__(name: str) -> object:
    if name in {"clip_positive", "require_finite"}:
        from quant_fund.utils import numeric

        value = getattr(numeric, name)
        globals()[name] = value
        return value
    if name == "set_global_seed":
        from quant_fund.utils.seeds import set_global_seed

        globals()[name] = set_global_seed
        return set_global_seed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
