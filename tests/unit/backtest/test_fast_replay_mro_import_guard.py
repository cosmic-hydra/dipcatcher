"""``fast_replay`` optional forecast import catches ImportError only.

Older lineages may lack ``quant_fund.pipeline.forecast.market_risk_overlay_asof``.
That is an ``ImportError``. A non-import failure while loading forecast must
not silently disable the market overlay (``_mro_asof = None``).
"""

from __future__ import annotations

import builtins
import importlib
import sys

import pytest

_MOD = "quant_fund.backtest.fast_replay"
_FORECAST = "quant_fund.pipeline.forecast"


def _gate_forecast_import_from_fast_replay(
    monkeypatch: pytest.MonkeyPatch, exc: BaseException
) -> None:
    """Raise ``exc`` only for forecast imports whose caller is ``fast_replay``.

    ``backtest.engine`` also imports forecast without a guard; blocking every
    forecast import would fail the parent package load before ``fast_replay``
    runs its own try/except.
    """
    real_import = builtins.__import__

    def gated(
        name: str,
        globals: dict[str, object] | None = None,  # noqa: A002 - import protocol
        locals: dict[str, object] | None = None,  # noqa: A002
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        caller = (globals or {}).get("__name__", "")
        if caller == _MOD and (name == _FORECAST or name.endswith(".pipeline.forecast")):
            raise exc
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", gated)


def _reimport_fast_replay() -> object:
    sys.modules.pop(_MOD, None)
    return importlib.import_module(_MOD)


def _restore_fast_replay() -> None:
    """Drop ``fast_replay`` from ``sys.modules`` and reload a clean copy.

    Gated ``builtins.__import__`` failures can leave a partial module entry.
    Later tests then see ImportError on symbols that exist after a healthy
    import — an order-dependent flake under xdist/loadfile. Callers undo the
    gate in ``finally`` before restore; the autouse fixture also restores.
    """
    sys.modules.pop(_MOD, None)
    importlib.import_module(_MOD)


@pytest.fixture(autouse=True)
def _fast_replay_module_isolation() -> object:
    """Ensure each test in this module starts and ends with a healthy import."""
    _restore_fast_replay()
    yield
    _restore_fast_replay()


def test_fast_replay_forecast_import_error_sets_mro_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gate_forecast_import_from_fast_replay(monkeypatch, ImportError("simulated older lineage"))
    try:
        fr = _reimport_fast_replay()
        assert fr._mro_asof is None
    finally:
        monkeypatch.undo()
        _restore_fast_replay()


def test_fast_replay_forecast_non_import_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _gate_forecast_import_from_fast_replay(monkeypatch, RuntimeError("not an ImportError"))
    try:
        with pytest.raises(RuntimeError, match="not an ImportError"):
            _reimport_fast_replay()
    finally:
        monkeypatch.undo()
        _restore_fast_replay()
