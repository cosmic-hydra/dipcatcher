"""Hardening tests: adapter describe paths, runner failure modes,
contamination probes, and the attestation ladder."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from fx1.data.sources.adapters import (
    AgentGwAdapter,
    FinanceFetchAdapter,
    XhcjAdapter,
)
from fx1.data.sources.base import (
    FetchRequest,
    RunOutcome,
    SourceStatus,
    default_runner,
)
from fx1.data.sources.registry import SourceKind, get_spec, list_sources
from fx1.eval.contamination import (
    min_k_percent_probe,
    ngram_containment_scan,
    rephrased_gap_probe,
    run_contamination_audit,
)
from fx1.serve.attestation import (
    OperatorProofManifest,
    TEEQuote,
    attestation_ladder_status,
    verify_quote,
)


@pytest.fixture(autouse=True)
def _stub_plugin_scripts(tmp_path, monkeypatch):
    root = tmp_path / "plugins"
    for spec in list_sources():
        if spec.kind is SourceKind.MCP or not spec.script_candidates:
            continue
        script = root / spec.script_candidates[0]
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("# test stub\n", encoding="utf-8")
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(root))
    for name in ("KIMI_API_KEY", "AGENT_GW_TOKEN", "DATASOURCE_BASE_URL", "DATASOURCE_API_KEY"):
        monkeypatch.setenv(name, "test-dummy")


class _Recorder:
    def __init__(self, outcome: RunOutcome) -> None:
        self.outcome = outcome
        self.calls: list[list[str]] = []

    def __call__(self, argv, *, timeout_s, extra_env=None):
        self.calls.append(list(argv))
        return self.outcome


# --------------------------------------------------------------------------
# Adapter describe/fetch paths
# --------------------------------------------------------------------------
def test_agent_gw_describe_uses_describe_verb():
    rec = _Recorder(RunOutcome(returncode=0, stdout="capabilities", stderr=""))
    result = AgentGwAdapter(get_spec("imf"), runner=rec).describe()
    assert result.ok and result.text == "capabilities"
    assert rec.calls[0][-1] == "describe"


def test_xhcj_describe_and_fetch_grammar():
    rec = _Recorder(RunOutcome(returncode=0, stdout="news", stderr=""))
    adapter = XhcjAdapter(get_spec("xhcj"), runner=rec)
    assert adapter.describe().ok
    assert rec.calls[0][-1] == "desc"
    adapter.fetch(FetchRequest(api="getPolicy", params={"keyword": "降准"}))
    assert rec.calls[1][2:] == ["call", "getPolicy", "keyword=降准"]


def test_finance_fetch_describe_lists_scenarios():
    rec = _Recorder(RunOutcome(returncode=0, stdout="scenarios", stderr=""))
    result = FinanceFetchAdapter(get_spec("finance_fetch"), runner=rec).describe()
    assert result.ok
    assert rec.calls[0][-1] == "--list-scenarios"


def test_finance_fetch_non_json_stdout_passes_through():
    rec = _Recorder(RunOutcome(returncode=0, stdout="plain text table", stderr=""))
    result = FinanceFetchAdapter(get_spec("finance_fetch"), runner=rec).fetch(
        FetchRequest(api="quote", params={"ticker": "NVDA"})
    )
    assert result.ok and result.text == "plain text table"


def test_finance_fetch_flag_args_and_bool_params():
    rec = _Recorder(RunOutcome(returncode=0, stdout="ok", stderr=""))
    FinanceFetchAdapter(get_spec("finance_fetch"), runner=rec).fetch(
        FetchRequest(api="us.income", params={"ticker": "AAPL", "periods": "5Q", "annual": True})
    )
    argv = rec.calls[0]
    assert argv[2:] == ["us.income", "AAPL", "--periods", "5Q", "--annual"]


# --------------------------------------------------------------------------
# Runner failure modes (base.py)
# --------------------------------------------------------------------------
def test_default_runner_lang_is_an_installed_utf8_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not force glibc's C.UTF-8 on a host that rejects that locale name."""
    import fx1.data.sources.base as base

    captured: dict[str, dict[str, str]] = {}

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv: list[str], **kwargs: object) -> _Proc:
        env = kwargs["env"]
        assert isinstance(env, dict)
        captured["env"] = env
        return _Proc()

    monkeypatch.setattr(base.subprocess, "run", fake_run)
    monkeypatch.delenv("LANG", raising=False)
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.setattr(base, "_preferred_utf8_locale", lambda: None)
    base.default_runner(["/bin/true"], timeout_s=1)
    assert "LANG" not in captured["env"]
    assert captured["env"]["PYTHONUTF8"] == "1"

    monkeypatch.setattr(base, "_preferred_utf8_locale", lambda: "en_US.UTF-8")
    base.default_runner(["/bin/true"], timeout_s=1)
    assert captured["env"]["LANG"] == "en_US.UTF-8"

    monkeypatch.setattr(base, "_preferred_utf8_locale", lambda: "C.UTF-8")
    base.default_runner(["/bin/true"], timeout_s=1)
    assert captured["env"]["LANG"] == "C.UTF-8"


def test_default_runner_missing_binary_is_honest():
    outcome = default_runner(["/nonexistent/fx1-binary-xyz"], timeout_s=5)
    assert outcome.returncode == 127
    assert outcome.stderr  # real OS error, not a placeholder


def test_default_runner_timeout_is_honest():
    outcome = default_runner([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.5)
    assert outcome.timed_out and outcome.returncode == 124


def test_default_runner_truncates_and_marks():
    outcome = default_runner([sys.executable, "-c", "print('x' * 300_000)"], timeout_s=30)
    assert outcome.returncode == 0
    assert len(outcome.stdout) < 300_000
    assert outcome.stdout.endswith("[fx1: output truncated]")


def test_injected_runner_failure_surfaces_probe_status():
    rec = _Recorder(RunOutcome(returncode=3, stdout="", stderr="db down"))
    result = AgentGwAdapter(get_spec("imf"), runner=rec).fetch(FetchRequest(api="weo_query"))
    assert not result.ok
    assert "db down" in (result.error or "")
    assert result.status in (SourceStatus.READY, SourceStatus.READY_UNVERIFIED)


# --------------------------------------------------------------------------
# Contamination probes
# --------------------------------------------------------------------------
def test_ngram_scan_flags_overlapping_example():
    prompts = ["what is the dipcatcher honesty contract for receipts"]
    corpus = ["what is the dipcatcher honesty contract for receipts exactly"]
    hits = ngram_containment_scan(corpus, prompts, threshold=0.5)
    assert hits and hits[0].example_index == 0
    assert hits[0].eval_index == 0
    assert not ngram_containment_scan(["completely unrelated text"], prompts)


def test_ngram_scan_catches_eval_item_embedded_in_long_doc():
    """The dominant contamination shape: a long training doc containing one
    verbatim eval item. Doc-side denominators miss it (the item is a small
    fraction of the doc); per-item containment must flag it."""
    prompt = " ".join(f"evaltok{i}" for i in range(12))
    padding = " ".join(f"benign{i}" for i in range(300))
    doc = f"{padding} {prompt} {padding}"
    hits = ngram_containment_scan([doc], [prompt], threshold=0.9)
    assert hits and hits[0].containment == pytest.approx(1.0)


def test_ngram_scan_unrelated_tokens_do_not_flag():
    """No shared 8-gram windows → no hit, however much vocab overlaps."""
    doc = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
    prompts = [" ".join(f"q{i}" for i in range(12))]
    assert not ngram_containment_scan([doc], prompts, threshold=0.3)


def test_audit_digest_binds_multiset_not_join():
    """[\"a\\nb\"] and [\"a\", \"b\"] must produce different corpus digests —
    the join-with-newline scheme collides on embedded newlines."""
    a = run_contamination_audit(["ab\ncd"], ["prompt x"])
    b = run_contamination_audit(["ab", "cd"], ["prompt x"])
    assert a.corpus_sha256 != b.corpus_sha256


def test_min_k_probe_inert_without_logprobs():
    result = min_k_percent_probe([])
    assert result.value is None and not result.flagged
    valued = min_k_percent_probe([[-5.0, -1.0, -0.5], [-4.0, -2.0]])
    assert valued.value is not None and valued.value < 0


def test_rephrased_gap_probe_flags_memorization():
    flagged = rephrased_gap_probe([True] * 10, [False] * 9 + [True], budget=0.3)
    assert flagged.flagged and flagged.value == pytest.approx(0.9)
    calm = rephrased_gap_probe([True] * 4, [True, True, True, False])
    assert not calm.flagged  # gap 0.25 below the 0.3 budget
    with pytest.raises(ValueError, match="paired"):
        rephrased_gap_probe([True], [])


def test_audit_report_is_hash_bound():
    report = run_contamination_audit(["some corpus text"], ["an eval prompt"])
    assert len(report.corpus_sha256) == 64
    assert len(report.eval_bank_sha256) == 64
    again = run_contamination_audit(["some corpus text"], ["an eval prompt"])
    assert report.corpus_sha256 == again.corpus_sha256


# --------------------------------------------------------------------------
# Attestation ladder
# --------------------------------------------------------------------------
def _quote(**kw) -> TEEQuote:
    base = {
        "platform": "sev-snp",
        "checkpoint_sha256": "a" * 64,
        "measurement": "ab" * 48,
        "report_data": "nonce-123:" + "a" * 64,
        "signature": "vendor-sig",
    }
    base.update(kw)
    return TEEQuote(**base)


def test_verify_quote_fails_closed_on_mismatch():
    quote = _quote()
    assert verify_quote(quote, expected_checkpoint_sha256="a" * 64, nonce="nonce-123")
    assert not verify_quote(quote, expected_checkpoint_sha256="b" * 64, nonce="nonce-123")
    assert not verify_quote(quote, expected_checkpoint_sha256="a" * 64, nonce="other-nonce")
    assert not verify_quote(
        _quote(signature=""), expected_checkpoint_sha256="a" * 64, nonce="nonce-123"
    )


def test_operator_manifest_requires_proofs_for_claimed_coverage():
    with pytest.raises(ValueError, match="without proof artifacts"):
        OperatorProofManifest(
            checkpoint_sha256="a" * 64,
            covered_operators=["honesty_gate_logits"],
            proof_artifacts={},
        )


def test_attestation_ladder_status(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FX1_SIGNING_KEY", raising=False)
    status = attestation_ladder_status(tmp_path)
    assert status == {"signed_release": False, "tee": False, "selective_zkml": False}
    (tmp_path / "attestation.quote.json").write_text(_quote().model_dump_json(), encoding="utf-8")
    proof = tmp_path / "proof.bin"
    proof.write_bytes(b"zk")
    (tmp_path / "zkml.manifest.json").write_text(
        OperatorProofManifest(
            checkpoint_sha256="a" * 64,
            covered_operators=["calibration_head"],
            proof_artifacts={"calibration_head": str(proof)},
        ).model_dump_json(),
        encoding="utf-8",
    )
    status = attestation_ladder_status(tmp_path)
    assert status["tee"] and status["selective_zkml"]
    assert not status["signed_release"]  # unsigned still fails closed
