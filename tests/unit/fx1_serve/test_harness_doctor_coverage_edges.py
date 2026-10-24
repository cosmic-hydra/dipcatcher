"""Edge arms in harness_doctor that a healthy doctor report never takes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from fx1.sdk import Fx1Harness
from fx1.serve.client import HarnessAuthError
from fx1.serve.harness_doctor import (
    _DoctorBackend,
    _local_commands,
    _local_completion_roundtrip,
    _local_health,
    _local_honesty_gate,
    _local_key_lifecycle,
    _local_receipt_roundtrip,
    _remote_capabilities,
    _remote_commands,
    _remote_health,
    _remote_key_lifecycle,
    _remote_receipt_roundtrip,
    _remote_unauth_refused,
    _remote_version_compat,
    _Skip,
    _state_dir_replay,
)


def _health(**kw: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "status": "ok",
        "registered_commands": 2,
        "backends": {"byok": True},
        "version": "0",
    }
    base.update(kw)
    return SimpleNamespace(**base)


def test_remote_checks_raise_on_empty_payloads() -> None:
    bad = SimpleNamespace(health=lambda: _health(status="down", registered_commands=0))
    with pytest.raises(RuntimeError, match="registered_commands"):
        _remote_health(bad)
    mismatch = {
        "compatible": False,
        "server_api_version": "0",
        "client_api_version": "1",
    }
    with pytest.raises(RuntimeError, match="api_version"):
        _remote_version_compat(SimpleNamespace(check_compat=lambda strict=False: mismatch))
    with pytest.raises(RuntimeError, match="capabilities"):
        _remote_capabilities(SimpleNamespace(capabilities=lambda: {}))
    with pytest.raises(RuntimeError, match="command registry"):
        _remote_commands(SimpleNamespace(commands=lambda: []))


def test_bogus_key_that_authenticates_is_a_fault() -> None:
    probe = SimpleNamespace(commands=lambda: ["doctor"])
    with pytest.raises(RuntimeError, match="bogus key"):
        _remote_unauth_refused(lambda _key: probe, False)


def test_key_admin_refusal_skips_and_rotate_that_still_works_fails() -> None:
    class _Denied:
        def key_create(self, **_kw: Any) -> dict[str, str]:
            raise HarnessAuthError("nope", code="forbidden")

    with pytest.raises(_Skip, match="key-admin"):
        _remote_key_lifecycle(_Denied(), lambda _key: None, False)

    class _Mint:
        def key_create(self, **_kw: Any) -> dict[str, str]:
            return {"id": "k1", "key": "secret"}

        def key_usage(self, _kid: str) -> dict[str, str]:
            return {}

        def key_rotate(self, _kid: str) -> dict[str, Any]:
            return {"key": {"id": "k2", "key": "secret2"}}

        def key_revoke(self, _kid: str) -> None:
            return None

    still_ok = SimpleNamespace(commands=lambda: ["doctor"])
    with pytest.raises(RuntimeError, match="still authenticates"):
        _remote_key_lifecycle(_Mint(), lambda _key: still_ok, False)


def test_lifecycle_finally_skips_a_missing_successor() -> None:
    revoked: list[str] = []

    class _Mint:
        def key_create(self, **_kw: Any) -> dict[str, str]:
            return {"id": "k1", "key": "secret"}

        def key_revoke(self, kid: str) -> None:
            revoked.append(kid)

    class _Probe:
        def commands(self) -> list[str]:
            raise RuntimeError("minted key rejected")

    with pytest.raises(RuntimeError, match="minted key"):
        _remote_key_lifecycle(_Mint(), lambda _key: _Probe(), False)
    assert revoked == ["k1"]


def test_remote_receipt_skip_and_reverify_fault() -> None:
    with pytest.raises(_Skip, match="no sealed"):
        _remote_receipt_roundtrip(SimpleNamespace(receipts=lambda: []))
    ref = SimpleNamespace(sha256="ab" * 32, name="parity")
    stored = SimpleNamespace(valid=False, document={"schema": "x"})
    verdict = SimpleNamespace(valid=False)
    client = SimpleNamespace(
        receipts=lambda: [ref],
        receipt=lambda _sha: stored,
        verify_receipt=lambda _doc: verdict,
    )
    with pytest.raises(RuntimeError, match="re-verify"):
        _remote_receipt_roundtrip(client)


def test_local_health_commands_and_gate_faults() -> None:
    down = SimpleNamespace(health=lambda: _health(status="down", registered_commands=0))
    with pytest.raises(RuntimeError, match="registered_commands"):
        _local_health(down)
    with pytest.raises(RuntimeError, match="command registry"):
        _local_commands(SimpleNamespace(commands=lambda: []))
    accepted = SimpleNamespace(ok=True)
    with pytest.raises(RuntimeError, match="misclassified"):
        _local_honesty_gate(SimpleNamespace(check_text=lambda _text: accepted))


def test_completion_roundtrip_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Wrong(_DoctorBackend):
        def complete(self, messages: list[dict[str, Any]], *, sampling: Any = None) -> str:
            del messages, sampling
            return "not-the-marker"

    monkeypatch.setattr("fx1.serve.harness_doctor._DoctorBackend", _Wrong)
    with pytest.raises(RuntimeError, match="marker"):
        _local_completion_roundtrip()

    class _Open(_DoctorBackend):
        def close(self) -> None:
            return None

    monkeypatch.setattr("fx1.serve.harness_doctor._DoctorBackend", _Open)
    with pytest.raises(RuntimeError, match="not closed"):
        _local_completion_roundtrip()

    monkeypatch.setattr("fx1.serve.harness_doctor._DoctorBackend", _DoctorBackend)
    _real_complete = Fx1Harness.complete

    def _clear(self: Fx1Harness, *args: Any, **kwargs: Any) -> Any:
        result = _real_complete(self, *args, **kwargs)
        self._last_response_headers.clear()
        return result

    monkeypatch.setattr(Fx1Harness, "complete", _clear)
    with pytest.raises(RuntimeError, match="completion-id"):
        _local_completion_roundtrip()

    def _bad_log(self: Fx1Harness, *args: Any, **kwargs: Any) -> Any:
        result = _real_complete(self, *args, **kwargs)
        self.completion = lambda _cid: SimpleNamespace(ok=False, completion_id="other")  # type: ignore[method-assign]
        return result

    monkeypatch.setattr(Fx1Harness, "complete", _bad_log)
    with pytest.raises(RuntimeError, match="log record"):
        _local_completion_roundtrip()


def test_local_revoke_faults_and_bad_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = Fx1Harness()
    real_get = sdk.key_get

    def _enabled(kid: str) -> dict[str, Any]:
        rec = dict(real_get(kid))
        rec["enabled"] = True
        return rec

    monkeypatch.setattr(sdk, "key_get", _enabled)
    with pytest.raises(RuntimeError, match="tombstoned"):
        _local_key_lifecycle(sdk)

    live = Fx1Harness()
    monkeypatch.setattr(live._key_store, "authenticate", lambda _key: object())
    with pytest.raises(RuntimeError, match="still authenticates"):
        _local_key_lifecycle(live)

    ref = SimpleNamespace(sha256="cd" * 32, name="local")
    stored = SimpleNamespace(valid=True, document={"schema": "x"})
    verdict = SimpleNamespace(valid=False)
    fake = SimpleNamespace(
        receipts=lambda: [ref],
        receipt=lambda _sha: stored,
        verify_receipt=lambda _doc: verdict,
    )
    with pytest.raises(RuntimeError, match="re-verify"):
        _local_receipt_roundtrip(fake, "receipts")


def test_state_dir_replay_wraps_a_lost_key(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(self: Fx1Harness, kid: str) -> dict[str, Any]:
        del self, kid
        raise OSError("disk")

    monkeypatch.setattr(Fx1Harness, "key_get", _boom)
    with pytest.raises(RuntimeError, match="journal replay"):
        _state_dir_replay(tmp_path / "state")
