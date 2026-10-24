"""SYNTHETIC edge tests for fx1 modules the fx1 suite left partially covered.

Each case asserts a fail-closed or identity outcome. No network, no sleeps.
"""

from __future__ import annotations

import importlib
import types
from pathlib import Path

import pytest

from fx1.capabilities import owner_references, resolve_seed_id
from fx1.data.ledger import CorpusLedger
from fx1.data.ledgers import ledger_examples
from fx1.data.sources.base import FetchRequest, FetchResult, SourceProbe, SourceStatus
from fx1.data.sources.router import RoutingPlan, fetch_routed
from fx1.doctor import collect_status
from fx1.extensions.contracts import FeatureExtension, PluginExtension, SkillExtension
from fx1.extensions.registry import extension_manifest, get_extension
from fx1.forecast.config import load_harness_config
from fx1.harness import HarnessResult, _subprocess_runner
from fx1.interactive.approvals import ApprovalGate
from fx1.sbom import generate_sbom
from fx1.train.run import _validate_corpus

_NO_RAISE_MODULES = (
    "fx1.bench.dip_audit",
    "fx1.eval.calibration_audit",
    "fx1.eval.capability_audit",
    "fx1.eval.contamination_audit",
    "fx1.eval.core_audit",
    "fx1.eval.masking_audit",
    "fx1.eval.ts_reasoning_audit",
    "fx1.forecast.infra_audit",
    "fx1.harness_audit",
    "fx1.honesty_audit",
    "fx1.modelcard_audit",
    "fx1.sbom_audit",
    "fx1.train.infra_audit",
    "fx1.train.run_audit",
)

_UNLOADED_WRAPPERS = (
    "fx1.extensions.features.z_vs_ma20",
    "fx1.extensions.plugins.binance_crypto",
    "fx1.extensions.plugins.caixin",
    "fx1.extensions.plugins.cls",
    "fx1.extensions.plugins.dongcai",
    "fx1.extensions.plugins.finance_fetch",
    "fx1.extensions.plugins.finance_research",
    "fx1.extensions.plugins.finenter",
    "fx1.extensions.plugins.gildata",
    "fx1.extensions.skills.tearsheet",
    "fx1.extensions.skills.train",
    "fx1.extensions.skills.validate",
    "fx1.extensions.skills.vendor_book_map",
    "fx1.extensions.skills.verify_research",
)


def _boom() -> None:
    raise ValueError("synthetic")


@pytest.mark.parametrize("module_name", _NO_RAISE_MODULES)
def test_audit_raise_helper_reports_success_and_failure(module_name: str) -> None:
    helper = importlib.import_module(module_name)._raises
    assert helper(lambda: None) == "no-raise"
    assert "ValueError" in helper(_boom)


@pytest.mark.parametrize("module_name", _UNLOADED_WRAPPERS)
def test_unloaded_extension_wrapper_binds_its_identity(module_name: str) -> None:
    module = importlib.import_module(module_name)
    extension = module.MODULE
    assert extension.module == module_name
    assert extension.kind in {"plugin", "skill", "feature"}
    assert extension.owner
    assert extension.references


def test_get_extension_refuses_a_module_without_an_extension_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = types.ModuleType("fx1.extensions.plugins.caixin")
    fake.MODULE = object()  # type: ignore[attr-defined]
    monkeypatch.setattr("fx1.extensions.registry.import_module", lambda _path: fake)
    with pytest.raises(TypeError, match="does not export an ExtensionModule"):
        get_extension("plugin", "caixin")


def test_get_extension_refuses_an_identity_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    extension = PluginExtension(
        kind="plugin",
        owner="caixin",
        references=[1],
        module="fx1.extensions.plugins.caixin",
    )
    object.__setattr__(extension, "owner", "not-caixin")
    fake = types.ModuleType("fx1.extensions.plugins.caixin")
    fake.MODULE = extension  # type: ignore[attr-defined]
    monkeypatch.setattr("fx1.extensions.registry.import_module", lambda _path: fake)
    with pytest.raises(ValueError, match="does not match its registered extension identity"):
        get_extension("plugin", "caixin")


def test_extension_manifest_refuses_an_unknown_kind_after_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extension = PluginExtension(
        kind="plugin",
        owner="caixin",
        references=[1],
        module="fx1.extensions.plugins.caixin",
    )
    monkeypatch.setattr("fx1.extensions.registry.get_extension", lambda _kind, _owner: extension)
    with pytest.raises(ValueError, match="unknown extension kind"):
        extension_manifest("other", "caixin")  # type: ignore[arg-type]


def test_records_and_verify_reject_a_foreign_capability() -> None:
    foreign = owner_references("feature", "log_price")[0]
    extension = FeatureExtension(
        kind="feature",
        owner="ret_1",
        references=[foreign],
        module="fx1.extensions.features.ret_1",
    )
    with pytest.raises(ValueError, match="foreign capability"):
        list(extension.records())
    with pytest.raises(ValueError, match="foreign capability"):
        extension.verify()


def test_plugin_and_feature_kinds_are_not_interchangeable() -> None:
    with pytest.raises(ValueError, match="PluginExtension requires kind"):
        PluginExtension(
            kind="feature",
            owner="ret_1",
            references=[1],
            module="fx1.extensions.features.ret_1",
        )
    with pytest.raises(ValueError, match="FeatureExtension requires kind"):
        FeatureExtension(
            kind="plugin",
            owner="caixin",
            references=[1],
            module="fx1.extensions.plugins.caixin",
        )
    with pytest.raises(ValueError, match="SkillExtension requires kind"):
        SkillExtension(
            kind="plugin",
            owner="caixin",
            references=[1],
            module="fx1.extensions.plugins.caixin",
        )


def test_plugin_wrapper_routes_probe_describe_and_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    probe = SourceProbe(name="caixin", status=SourceStatus.READY, detail="synthetic")
    described = FetchResult.failure(
        source="caixin", api="describe", error="synthetic", status=SourceStatus.READY
    )
    fetched = FetchResult(
        ok=True, source="caixin", api="stories", text="SYNTHETIC", status=SourceStatus.READY
    )

    class _Adapter:
        def probe(self) -> SourceProbe:
            return probe

        def describe(self) -> FetchResult:
            return described

        def fetch(self, request: FetchRequest) -> FetchResult:
            assert request.api == "stories"
            return fetched

    monkeypatch.setattr(PluginExtension, "adapter", lambda _self: _Adapter())
    extension = PluginExtension(
        kind="plugin",
        owner="caixin",
        references=[1],
        module="fx1.extensions.plugins.caixin",
    )
    assert extension.probe() == probe
    assert extension.describe() == described
    assert extension.fetch(FetchRequest(api="stories")) == fetched
    assert extension._record_owner({"source": "caixin"}) == "caixin"
    assert extension._record_owner({"source": 1}) is None


def test_feature_build_uses_the_injected_harness() -> None:
    seen: dict[str, object] = {}

    class _Harness:
        def run(self, name: str, config: Path | None = None) -> HarnessResult:
            seen["name"] = name
            seen["config"] = config
            return HarnessResult(command=name, exit_code=0, stdout="SYNTHETIC", stderr="")

    extension = FeatureExtension(
        kind="feature",
        owner="ret_1",
        references=list(owner_references("feature", "ret_1")[:1]),
        module="fx1.extensions.features.ret_1",
    )
    config = Path("configs/research.yaml")
    result = extension.build(config=config, harness=_Harness())  # type: ignore[arg-type]
    assert result.ok is True
    assert seen == {"name": "build-features", "config": config}
    assert extension._record_owner({"feature": "ret_1"}) == "ret_1"
    assert extension._record_owner({"feature": None}) is None
    assert resolve_seed_id(extension.references[0])["owner"] == "ret_1"


def test_doctor_reports_an_unavailable_eval_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Boom:
        def __len__(self) -> int:
            raise RuntimeError("synthetic bank failure")

    monkeypatch.setattr("fx1.eval.DEFAULT_BANK", _Boom())
    status = collect_status(tmp_path)
    assert status["eval_bank_tasks"] == "unavailable"


def test_seed_ids_outside_the_layout_fail_closed() -> None:
    with pytest.raises(KeyError, match="outside the registered layout"):
        resolve_seed_id(-1)
    with pytest.raises(KeyError, match="outside the registered layout"):
        resolve_seed_id(1_000_001)


def test_sbom_refuses_a_non_object_package_and_a_non_list_of_wheels(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text("package = [1]\n")
    with pytest.raises(ValueError, match="malformed"):
        generate_sbom(lock)
    lock.write_text('[[package]]\nname = "dep"\nversion = "1.0"\nwheels = "nope"\n')
    with pytest.raises(ValueError, match="malformed wheels"):
        generate_sbom(lock)
    good = "ab" * 32
    lock.write_text(
        "[[package]]\n"
        'name = "dep"\n'
        'version = "1.0"\n'
        "wheels = ["
        '{hash = "sha256:not-a-digest"}, '
        f'{{hash = "sha256:{good}"}}]\n'
    )
    sbom = generate_sbom(lock)
    assert sbom.entries[0].sha256 == good


def test_non_scalar_live_claim_values_are_asserted(tmp_path: Path) -> None:
    artifact = tmp_path / "ledger.json"
    artifact.write_text('{"live_pnl_claim": {"asserted": true}}\n', encoding="utf-8")
    examples = ledger_examples(artifact, system="SYNTHETIC")
    assert examples
    assert "forbids" in examples[0].messages[-1]["content"]


def test_harness_config_must_be_a_mapping(tmp_path: Path) -> None:
    path = tmp_path / "harness.json"
    path.write_text("[1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_harness_config(path)


def test_corpus_skips_blank_lines_and_requires_a_receipt(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '\n{"receipt_sha256": "abc", "negative": true}\n\n{"receipt_sha256": "def"}\n',
        encoding="utf-8",
    )
    stats = _validate_corpus(corpus)
    assert stats == {"lines": 2, "positive": 1, "negative": 1}
    corpus.write_text('{"text": "no receipt"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="receipt_sha256"):
        _validate_corpus(corpus)


def test_paired_bootstrap_refuses_unequal_or_empty_inputs() -> None:
    from fx1.eval.compare import bootstrap_delta_ci

    with pytest.raises(ValueError, match="equal length"):
        bootstrap_delta_ci([], [])
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_delta_ci([True], [True, False])


def test_subprocess_runner_maps_the_completed_process(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def _run(argv: list[str], **kwargs: object) -> types.SimpleNamespace:
        seen["argv"] = argv
        seen["timeout"] = kwargs["timeout"]
        seen["check"] = kwargs["check"]
        return types.SimpleNamespace(returncode=3, stdout="out", stderr="err")

    monkeypatch.setattr("fx1.harness.subprocess.run", _run)
    code, stdout, stderr = _subprocess_runner(["doctor"], 5)
    assert (code, stdout, stderr) == (3, "out", "err")
    assert seen["argv"] == ["dipcatcher", "doctor"]
    assert seen["timeout"] == 5
    assert seen["check"] is False


def test_approval_interrupt_is_a_denial() -> None:
    def _interrupt(_question: str) -> bool:
        raise KeyboardInterrupt

    gate = ApprovalGate(ask_fn=_interrupt)
    assert gate.ask("transmit?") is False
    assert gate.records[-1].approved is False

    def _eof(_question: str) -> bool:
        raise EOFError

    gate = ApprovalGate(ask_fn=_eof)
    assert gate.ask("again?") is False


def test_routed_fetch_with_no_candidates_names_the_plan_note() -> None:
    plan = RoutingPlan(
        question="SYNTHETIC",
        need="unknown",
        market="unknown",
        candidates=[],
        note="no routing rule",
    )
    result = fetch_routed(plan, api_by_source={})
    assert result.ok is False
    assert "no routing rule" in (result.error or "")


def test_ledger_reload_skips_blank_lines_and_a_broken_link_fails(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = CorpusLedger(path)
    digest = "a" * 64
    ledger.record_example(source_sha256=digest, transform_sha256=digest, example_sha256=digest)
    text = path.read_text(encoding="utf-8")
    path.write_text("\n" + text, encoding="utf-8")
    reloaded = CorpusLedger(path)
    assert reloaded.verify_chain() is True
    broken = reloaded._entries[0].model_copy(update={"prev_hash": "b" * 64})
    reloaded._entries[0] = broken
    assert reloaded.verify_chain() is False
    shifted = broken.model_copy(update={"index": 4, "prev_hash": "0" * 64})
    reloaded._entries[0] = shifted
    assert reloaded.verify_chain() is False


def test_lazy_package_attributes_fail_closed() -> None:
    import fx1.data as data
    import fx1.eval as evaluation

    assert data.sources.__name__ == "fx1.data.sources"
    with pytest.raises(AttributeError):
        _ = data.not_a_real_export
    with pytest.raises(AttributeError, match="no attribute"):
        _ = evaluation.not_a_real_export
    assert "DEFAULT_BANK" in evaluation.__dir__()


def test_quality_audit_records_a_splitter_that_accepts_an_illegal_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fx1.data.quality as quality_mod
    import fx1.data.quality_audit as quality

    real = quality_mod.frozen_split

    def loose(
        examples: list[dict[str, object]],
        root: object,
        *,
        val_fraction: float = 0.05,
        seed: int = 17,
    ) -> object:
        if val_fraction == 0.0:
            return real(examples, root, val_fraction=0.05, seed=seed)
        return real(examples, root, val_fraction=val_fraction, seed=seed)

    monkeypatch.setattr(quality_mod, "frozen_split", loose)
    report = quality.quality_audit()
    assert report["bound_fails"] == [0.0]
    assert report["bounds_enforced"] is False
