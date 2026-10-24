"""Lane-157 idempotency audit: battery holds + the receipt verifies."""

from __future__ import annotations

import pytest

from fx1.serve.idem_audit import idem_audit, idem_audit_bench
from quant_fund.research.receipt_v2 import verify_receipt_payload

_FORMER_DEFECTS = frozenset(
    {
        # The six sync _IdemStore routes ran lookup -> execute -> put
        # with no claim lock, so parallel same-key submits could
        # double-execute — claim deps now hold the key across the whole
        # handler span.
        "conc.same_key_one_execute",
        "conc.diff_body_one_wins",
        # /v1/uploads, /v1/files, and /harness/keys mint/rotate/patch/
        # revoke ignored Idempotency-Key entirely — a retried mint
        # fabricated a second credential.
        "coverage.keys_mint.replay",
        "coverage.keys_rotate.replay",
        "coverage.keys_patch.replay",
        "coverage.keys_revoke.replay",
        "coverage.uploads_create.replay",
        "coverage.uploads_part.replay",
        "coverage.uploads_complete.replay",
        "coverage.uploads_cancel.replay",
        "coverage.files.replay",
        # Key reuse was credential-blind — a key under credential A
        # could replay under B; _idem_scope pins the claim to the
        # authenticated fingerprint.
        "namespace.credential_isolated",
        "namespace.credential_own_conflict",
        # The 409 envelope code drifted by route — unified on
        # idempotency_conflict in both error grammars.
        "conflict.openai_code",
        "conflict.harness_code",
    }
)


@pytest.fixture(scope="module")
def results() -> dict[str, object]:
    return idem_audit()


def test_all_probes_hold(results: dict[str, object]) -> None:
    assert len(results) >= 90, "the battery must stay deep — do not thin it out"
    assert results, "audit returned no probes"
    failures = {k: v for k, v in results.items() if v is not True}
    assert not failures, f"probes that must hold: {sorted(failures)}"


def test_former_defects_now_hold(results: dict[str, object]) -> None:
    for name in _FORMER_DEFECTS:
        assert results.get(name) is True, f"{name} regressed — the former defect is back"


def test_receipt_verifies() -> None:
    bench = idem_audit_bench()
    res = verify_receipt_payload(bench)
    assert res["valid"], res
    assert bench["claim"]["ok"] is True, bench["claim"]["defects"]
    assert bench["claim"]["defects"] == []


def test_receipt_deterministic() -> None:
    a = idem_audit_bench()
    b = idem_audit_bench()
    assert a == b, "the sealed receipt must be byte-deterministic"
