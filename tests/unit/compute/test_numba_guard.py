"""numba_guard: only the known compiler defect falls back to py_func."""

from __future__ import annotations

import pytest

from quant_fund.compute.numba_guard import call_kernel, compiler_defect, python_twin


def test_compiler_defect_matches_get_call_template_only() -> None:
    assert compiler_defect(AttributeError("'function' object has no attribute 'get_call_template'"))
    assert not compiler_defect(AttributeError("'dict' object has no attribute 'items'"))
    assert not compiler_defect(ValueError("domain"))


def test_compiler_defect_walks_cause_chain() -> None:
    inner = AttributeError("'function' object has no attribute 'get_call_template'")
    outer = AttributeError("wrapper")
    outer.__cause__ = inner
    assert compiler_defect(outer)


def test_call_kernel_falls_back_only_for_compiler_defect() -> None:
    calls: list[str] = []

    def broken(*_a: object) -> int:
        calls.append("broken")
        raise AttributeError("'function' object has no attribute 'get_call_template'")

    def twin(*_a: object) -> int:
        calls.append("twin")
        return 7

    broken.py_func = twin  # type: ignore[attr-defined]
    assert call_kernel(broken, 1) == 7
    assert calls == ["broken", "twin"]


def test_call_kernel_propagates_other_attribute_errors() -> None:
    def boom(*_a: object) -> int:
        raise AttributeError("'x' object has no attribute 'y'")

    boom.py_func = lambda *_a: 1  # type: ignore[attr-defined]
    with pytest.raises(AttributeError, match="'x' object"):
        call_kernel(boom)


def test_python_twin_without_py_func_is_identity() -> None:
    def plain(x: int) -> int:
        return x + 1

    assert python_twin(plain) is plain
    assert python_twin(plain)(2) == 3
