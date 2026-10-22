"""Call a Numba kernel, or its original Python body when the compiler is broken.

Numba 0.68 on CPython 3.13 can type ``np.empty`` as a plain function during
nopython inference (``ol_np_empty.<locals>.impl`` has no
``get_call_template``). That is a compiler defect, not bad market data.
ValueError and other domain errors still propagate. ``Dispatcher.py_func``
is the undecorated function, so the fallback is the same recurrence.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_COMPILER_ATTR = "get_call_template"


def compiler_defect(exc: BaseException) -> bool:
    """True only for the known Numba overload-dispatcher crash."""
    if not isinstance(exc, AttributeError):
        return False
    cause: BaseException | None = exc
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, AttributeError) and _COMPILER_ATTR in str(cause):
            return True
        cause = cause.__cause__ or cause.__context__
    return False


def python_twin(kernel: Callable[..., Any]) -> Callable[..., Any]:
    """The function Numba wrapped, or ``kernel`` itself when it is already Python."""
    twin = getattr(kernel, "py_func", None)
    if twin is None:
        return kernel
    return twin


def call_kernel(kernel: Callable[..., Any], /, *args: object) -> Any:
    """Run ``kernel``. On the known compiler defect, run its Python body."""
    try:
        return kernel(*args)
    except AttributeError as exc:
        if not compiler_defect(exc):
            raise
        return python_twin(kernel)(*args)


__all__ = ["call_kernel", "compiler_defect", "python_twin"]
