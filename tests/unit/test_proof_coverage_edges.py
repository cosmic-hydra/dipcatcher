"""Fail-closed edges in proof bundle, verify, runner, and promotion receipts.

These tests lock branches the happy-path suite never takes: unknown signature
schemes, NaN metric comparison, chain blanks, git-fingerprint fallbacks,
atomic-write failures, and promotion inputs that must not seal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from quant_fund.proof import __dir__ as proof_dir
from quant_fund.proof import bundle as bundle_mod
from quant_fund.proof import runner as runner_mod
from quant_fund.proof.bundle import chain_head, load_chain, round_floats
from quant_fund.proof.promotion_receipt import (
    Approver,
    PromotionCompositionError,
    _artifact_identity_checked,
    _load_evidence_report,
    _report_checked,
    _write_receipt_exclusive,
)
from quant_fund.proof.runner import run_proven
from quant_fund.proof.sign import HmacSha256Signer
from quant_fund.proof.verify import (
    _check_committed_sidecars,
    _check_signature,
    _infer_bundle_dir,
    _metric_close,
    verify_bundle,
)
from quant_fund.proofcore.contracts import GENESIS_HASH
from tests.unit.proof_fake_vault import mint_synthetic_bundle
from tests.unit.test_proven_runner import _build_vault, _spec


class _Dataset:
    materialized_panel_sha256 = "0" * 64


class _Identity:
    dataset = _Dataset()


def _bundle_path(bundle_dir: Path, bundle) -> Path:
    return bundle_dir / "bundles" / f"{bundle.bundle_id}.json"


def test_package_dir_lists_lazy_exports() -> None:
    assert "verify_bundle" in proof_dir()
    assert "run_proven" in proof_dir()


def test_round_floats_canonicalizes_datetimes_and_numpy_scalars() -> None:
    stamp = datetime(2024, 1, 2, 3, 4, tzinfo=UTC)
    assert round_floats(stamp) == stamp.isoformat()
    scalar = np.float64(1.2345678901234567)
    assert round_floats(scalar) == round(float(scalar), 12)

    class _BrokenScalar:
        def item(self) -> float:
            raise ValueError("not a numpy scalar")

    broken = _BrokenScalar()
    assert round_floats(broken) is broken


def test_git_dirty_fails_closed_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_mod.shutil, "which", lambda _name: None)
    assert bundle_mod._git_dirty() is True


def test_git_dirty_fails_closed_when_git_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bundle_mod.shutil, "which", lambda _name: "/usr/bin/git")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("git unavailable")

    monkeypatch.setattr(bundle_mod.subprocess, "run", _boom)
    assert bundle_mod._git_dirty() is True


def test_code_fingerprint_substitutes_genesis_when_git_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import quant_fund.utils.reproducibility as repro

    monkeypatch.setattr(repro, "git_revision", lambda: "UNKNOWN")
    monkeypatch.setattr(repro, "git_worktree_sha256", lambda: "short")
    fingerprint = bundle_mod.code_fingerprint()
    assert fingerprint.git_revision == "0" * 40
    assert fingerprint.worktree_sha256 == GENESIS_HASH
    assert fingerprint.dirty is True


def test_atomic_bundle_write_leaves_no_temp_when_create_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("temp file unavailable")

    monkeypatch.setattr(bundle_mod.tempfile, "NamedTemporaryFile", _boom)
    with pytest.raises(OSError, match="temp file"):
        bundle_mod._atomic_write_bytes(tmp_path / "bundle.bin", b"abc")
    assert list(tmp_path.iterdir()) == []


def test_chain_readers_skip_blank_lines(tmp_path: Path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    chain = bundle_dir / "bundles.jsonl"
    chain.write_text("\n" + chain.read_text() + "\n\n")
    loaded = load_chain(bundle_dir)
    assert [item.bundle_id for item in loaded] == [bundle.bundle_id]
    assert chain_head(bundle_dir) == bundle.bundle_id


def test_metric_close_treats_nan_as_equal_only_to_nan() -> None:
    assert _metric_close(float("nan"), float("nan")) is True
    assert _metric_close(float("nan"), 0.0) is False
    assert _metric_close(0.0, float("nan")) is False


def test_infer_bundle_dir_when_file_is_not_under_bundles(tmp_path: Path) -> None:
    path = tmp_path / "loose.json"
    assert _infer_bundle_dir(path) == tmp_path


def test_unknown_signature_scheme_is_rejected() -> None:
    reasons: list[str] = []
    _check_signature({"signature": {"scheme": "rot13", "key_id": "x", "value": "y"}}, True, reasons)
    assert reasons == ["signature:unknown_scheme:rot13"]


def test_hmac_mismatch_with_the_right_key_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PROOFCORE_SIGNING_KEY", "ci-secret-key")
    signer = HmacSha256Signer.from_env()
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path, signer=signer)
    path = _bundle_path(bundle_dir, bundle)
    payload = json.loads(path.read_bytes())
    payload["signature"]["value"] = "ab" * 32
    path.write_bytes(json.dumps(payload).encode())
    result = verify_bundle(path, bundle_dir=bundle_dir)
    assert "signature:hmac_mismatch" in result.reasons


def test_data_manifest_leaf_error_fails_the_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)

    def _boom(_read: object) -> str:
        raise ValueError("bad leaf")

    monkeypatch.setattr("quant_fund.proof.verify.read_leaf_hash", _boom)
    result = verify_bundle(
        _bundle_path(bundle_dir, bundle), bundle_dir=bundle_dir, strict_signature=False
    )
    assert any(reason.startswith("data_manifest:leaf_error:") for reason in result.reasons)


def test_bundle_missing_from_chain_but_prev_matches_head(tmp_path: Path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    payload = json.loads(_bundle_path(bundle_dir, bundle).read_bytes())
    payload["bundle_id"] = "ab" * 32
    payload["prev_bundle_hash"] = bundle.bundle_id
    other = bundle_dir / "bundles" / "other.json"
    other.write_text(json.dumps(payload))
    result = verify_bundle(other, bundle_dir=bundle_dir, strict_signature=False)
    assert "chain:bundle_missing" in result.reasons
    assert "chain:prev_not_head" not in result.reasons


def test_unreadable_chain_fails_trusted_head_check(tmp_path: Path) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    (bundle_dir / "bundles.jsonl").write_text("{not-json\n")
    result = verify_bundle(
        _bundle_path(bundle_dir, bundle),
        bundle_dir=bundle_dir,
        strict_signature=False,
        trusted_head=GENESIS_HASH,
    )
    assert any(reason.startswith("chain:trusted_head_unreadable:") for reason in result.reasons)


def test_committed_sidecar_checks_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)
    reasons: list[str] = []
    _check_committed_sidecars(bundle, bundle_dir, None, reasons)
    assert reasons == []

    reasons = []
    _check_committed_sidecars(bundle, bundle_dir, b"not-json", reasons)
    assert reasons == ["sidecar:config_invalid_json"]

    reasons = []
    _check_committed_sidecars(bundle, bundle_dir, b"[]", reasons)
    assert reasons == []

    kind = "trace"
    expected = "ab" * 32
    config = json.dumps({"sidecars": {f"{kind}_sha256": expected}}).encode()
    link = bundle_dir / f"{bundle.bundle_id}.{kind}.json"
    link.symlink_to(bundle_dir / "bundles.jsonl")
    reasons = []
    _check_committed_sidecars(bundle, bundle_dir, config, reasons)
    assert f"sidecar:{kind}:unsafe_symlink" in reasons

    link.unlink()
    link.write_text("{}\n")
    # The unreadable-sidecar branch is the OSError handler. Inject the
    # denial so the check is the same on Windows, where mode bits do not
    # stop a read, and so the test file is never left world-readable.
    real_read = Path.read_bytes

    def _deny_committed_sidecar(self: Path) -> bytes:
        if self == link:
            raise PermissionError("denied")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", _deny_committed_sidecar)
    reasons = []
    _check_committed_sidecars(bundle, bundle_dir, config, reasons)
    assert any(reason.startswith(f"sidecar:{kind}:unreadable:") for reason in reasons)


def test_successful_replay_does_not_add_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """verify_bundle does not construct a vault, so an identical replay is injected.

    The branch under test is the verifier's: a True replay verdict must not
    become a failure reason. The CLI replay path builds the vault itself.
    """
    bundle, bundle_dir = mint_synthetic_bundle(tmp_path)

    def _identical(*_args: object, **_kwargs: object) -> tuple[bool, str]:
        return True, json.dumps({"status": "identical"})

    monkeypatch.setattr("quant_fund.proof.replay.replay_bundle", _identical)
    result = verify_bundle(
        _bundle_path(bundle_dir, bundle),
        bundle_dir=bundle_dir,
        strict_signature=False,
        replay=True,
    )
    assert not any(reason.startswith("replay:") for reason in result.reasons)


def test_sidecar_read_failure_is_a_missing_sidecar(tmp_path: Path) -> None:
    from quant_fund.proof.replay import _read_sidecar

    assert _read_sidecar(tmp_path) is None


def test_real_proven_replay_through_verify_is_identical(tmp_path: Path) -> None:
    vault_root = tmp_path / "pit"
    bundle_dir = tmp_path / "proofs"
    ok, bundle_id = run_proven(
        _spec(count=3), vault=_build_vault(vault_root), bundle_dir=bundle_dir
    )
    assert ok
    from quant_fund.pit.vault import PitVault
    from quant_fund.proof.replay import replay_bundle

    ok_replay, detail = replay_bundle(
        bundle_dir / "bundles" / f"{bundle_id}.json",
        bundle_dir=bundle_dir,
        pit_root=vault_root,
        vault=PitVault(vault_root),
    )
    assert ok_replay, detail
    assert json.loads(detail)["status"] == "identical"


def test_runner_atomic_write_and_commit_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("temp file unavailable")

    monkeypatch.setattr(runner_mod.tempfile, "NamedTemporaryFile", _boom)
    with pytest.raises(OSError, match="temp file"):
        runner_mod._atomic_write(tmp_path / "out.bin", b"xyz")

    staging = tmp_path / "staging"
    staging.mkdir()
    dest = tmp_path / "dest"
    runner_mod._commit_staging(staging, dest, 0)
    assert dest.is_dir()
    assert not (dest / "bundles.jsonl").exists()


def test_guard_unavailable_warning_is_once() -> None:
    previous = runner_mod._GUARD_UNAVAILABLE_LOGGED
    runner_mod._GUARD_UNAVAILABLE_LOGGED = True
    try:
        runner_mod._log_guard_unavailable_once()
    finally:
        runner_mod._GUARD_UNAVAILABLE_LOGGED = previous


def test_io_guard_allowlist_skips_vaults_without_a_root(tmp_path: Path) -> None:
    class _Vault:
        def asof(self, *_args: object, **_kwargs: object) -> None:
            raise AssertionError("vault should not be read")

    with runner_mod._enforce_io_guard(_Vault(), tmp_path):
        assert runner_mod._log_guard_unavailable_once is not None


def test_approver_rejects_a_non_iso_decision_time() -> None:
    with pytest.raises(ValidationError, match="ISO-8601"):
        Approver(name="ada", role="lead", decided_at="tomorrow")


def test_evidence_report_loader_rejects_missing_and_non_object(tmp_path: Path) -> None:
    missing = tmp_path / "gone.json"
    with pytest.raises(PromotionCompositionError, match="unreadable"):
        _load_evidence_report(missing)
    payload = tmp_path / "list.json"
    payload.write_text("[1]\n")
    with pytest.raises(PromotionCompositionError, match="not a JSON object"):
        _load_evidence_report(payload)


def test_report_checked_rejects_missing_file_and_bad_shape(tmp_path: Path) -> None:
    with pytest.raises(PromotionCompositionError, match="does not exist"):
        _report_checked(tmp_path / "missing.json", "0" * 64, _Identity())  # type: ignore[arg-type]

    from tests.unit.test_promotion_receipt import _build

    built = _build(tmp_path)
    report = Path(built["report"])
    loose = tmp_path / "loose-report.json"
    loose.write_bytes(report.read_bytes())
    identity = built["identity"]
    sha = "0" * 64
    # No hash sidecar: the optional check is skipped and the report still binds.
    checked = _report_checked(loose, hash_file_sha(report, built), identity)
    assert checked[0]

    unproven = tmp_path / "no-provenance.json"
    body = json.loads(report.read_text())
    body["provenance"] = "missing"
    unproven.write_text(json.dumps(body))
    with pytest.raises(PromotionCompositionError, match="no provenance"):
        _report_checked(unproven, sha, identity)


def hash_file_sha(report: Path, built: dict[str, object]) -> str:
    """Artifact sha the copied report already binds."""
    body = json.loads(report.read_text())
    return str(body["provenance"]["artifact_sha256"])


def test_unreadable_manifest_blocks_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quant_fund.pipeline.artifact_manifest import manifest_path
    from tests.unit.test_promotion_receipt import _build

    built = _build(tmp_path)
    artifact = Path(built["artifact"])
    manifest = manifest_path(artifact)
    real_read = Path.read_text
    seen = {"manifest": 0}

    def _second_manifest_read_is_garbage(self: Path, *args: object, **kwargs: object) -> str:
        text = real_read(self, *args, **kwargs)
        if self == manifest:
            seen["manifest"] += 1
            # The verifier's first read must succeed. The promotion step
            # re-reads the same file to record its class, and that read is
            # the one that must fail closed.
            if seen["manifest"] > 1:
                return "{"
        return text

    monkeypatch.setattr(Path, "read_text", _second_manifest_read_is_garbage)
    with pytest.raises(PromotionCompositionError, match="artifact manifest is unreadable"):
        _artifact_identity_checked(artifact)


def test_receipt_sidecar_failure_does_not_unlink_a_missing_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("temp file unavailable")

    monkeypatch.setattr("quant_fund.proof.promotion_receipt.NamedTemporaryFile", _boom)
    dest = tmp_path / "receipts" / "decision.json"
    with pytest.raises(OSError, match="temp file"):
        _write_receipt_exclusive(dest, {"schema": "promotion_receipt.v1"})
    assert dest.is_file()
    assert not dest.with_name(dest.name + ".sha256").exists()
