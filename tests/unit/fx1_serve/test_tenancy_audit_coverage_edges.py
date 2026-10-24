"""Edge arms in tenancy_audit: env restore, poll deadlines, and the embed stub."""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from fx1.serve import tenancy_audit as ten


class _Resp:
    def __init__(self, body: dict[str, Any], status: int = 200) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


class _Client:
    def get(self, url: str, **_kwargs: Any) -> _Resp:
        del url
        return _Resp({"status": "running", "processing_status": "in_progress"})


def _exec_main(mod: Any, bench: str) -> None:
    path = mod.__file__
    assert isinstance(path, str)
    source = Path(path).read_text(encoding="utf-8")

    def _stub_bench() -> dict[str, bool]:
        return {"ok": True}

    class _NS(dict[str, Any]):
        def __setitem__(self, key: str, value: object) -> None:
            if key == bench and callable(value):
                value = _stub_bench
            dict.__setitem__(self, key, value)

    ns = _NS()
    ns.update({"__name__": "__main__", "__file__": path, "__package__": mod.__package__})
    with redirect_stdout(StringIO()):
        exec(compile(source, path, "exec"), ns, ns)  # noqa: S102 — runs the module __main__ line


def test_embeddings_counts_lists_and_scalars() -> None:
    backend = ten._StubBackend()
    listed = backend.embeddings(["a", "b"], model="m")
    scalar = backend.embeddings("solo", model="m", extra=1)
    assert len(listed.data) == 2
    assert len(scalar.data) == 1
    assert listed.model == "m"


def test_api_key_env_pops_and_restores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ten._API_KEY_ENV, "tenancy-prev")
    with ten._api_key_env(None):
        assert ten._API_KEY_ENV not in os.environ
    assert os.environ[ten._API_KEY_ENV] == "tenancy-prev"


def test_list_ids_and_wait_for_cover_each_arm() -> None:
    assert ten._list_ids(_Resp({"data": [{"id": "a"}]}, status=503)) == set()
    assert ten._wait_for(lambda: True, timeout_s=1.0) is True
    calls = {"n": 0}

    def _later() -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    assert ten._wait_for(_later, timeout_s=1.0) is True
    assert ten._wait_for(lambda: False, timeout_s=0.03) is False


def test_resource_waits_sleep_once_then_time_out(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def _mono() -> float:
        calls["n"] += 1
        return 1000.0 if calls["n"] % 3 == 0 else 0.0

    monkeypatch.setattr(ten.time, "monotonic", _mono)
    monkeypatch.setattr(ten.time, "sleep", lambda _seconds: None)
    client = _Client()
    with pytest.raises(AssertionError, match="batch"):
        ten._wait_batch(client, "auth", "batch-1")
    with pytest.raises(AssertionError, match="eval run"):
        ten._wait_eval_run(client, "auth", "ev", "run")
    with pytest.raises(AssertionError, match="ft job"):
        ten._wait_ft(client, "auth", "ftjob")
    with pytest.raises(AssertionError, match="response"):
        ten._wait_response(client, "auth", "resp")


def test_main_prints_stubbed_bench() -> None:
    _exec_main(ten, "tenancy_audit_bench")
