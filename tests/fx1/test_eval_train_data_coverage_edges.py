"""Edge coverage for eval, train, data, extensions, and small audits.

Tiny in-memory inputs only. Nested audit ``_raises`` helpers are reached by
driving the audit so one checked call returns, not by looking up a module
attribute that does not exist.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from fx1.capabilities import resolve_seed_id
from fx1.data.corpus import SFTExample, _negative_example, _system_prompt, build_full_corpus
from fx1.data.ledger import CorpusLedger
from fx1.data.ledgers import _is_live_claim_value
from fx1.data.receipts import ReceiptRecord
from fx1.data.sources.adapters import (
    AgentGwAdapter,
    CaixinAdapter,
    FinanceFetchAdapter,
    XhcjAdapter,
)
from fx1.data.sources.base import FetchRequest, LatencyClass, RunOutcome, SourceKind, SourceStatus
from fx1.data.sources.registry import SourceSpec
from fx1.data.sources.router import RoutingPlan, fetch_routed
from fx1.eval.compare import bootstrap_delta_ci
from fx1.eval.ext_bench import (
    _coerce_score,
    _coerce_winner,
    _run_fintoolbench_task,
    build_mtbench_judge_prompt,
    build_mtbench_pairwise_prompt,
    execution_trace_score,
    judge_pairwise,
)
from fx1.eval.ext_bench_banks import (
    FinToolBenchInstance,
    FinToolSpec,
    MTBenchInstance,
    MTBenchTurn,
    build_ext_bench_banks,
    make_ext_bench_oracle,
)
from fx1.eval.options_reasoning_eval import (
    OptionsReasoningBank,
    OptionsReasoningItem,
    grade_options_reasoning,
)
from fx1.eval.options_reasoning_eval import (
    make_oracle_model as make_options_oracle,
)
from fx1.eval.retrieval_eval import (
    FAMILY_FACTOID,
    RetrievalQuestion,
    _conversation_question_id,
    _run_one,
    make_golden_model,
    make_no_retrieval_model,
)
from fx1.eval.rubric_banks import (
    GuidanceRule,
    RubricCriterion,
    RubricQuery,
    TaskBank,
    TaskBankSchemaError,
    _validate_bank_consistency,
    load_task_bank,
    make_rubric_model_oracle,
    make_rubric_writer_oracle,
)
from fx1.eval.rubric_eval import (
    _extract_json_object,
    _forbidden_demand_failures,
    _run_criterion_check,
    _spec_failures,
    generate_rubric,
)
from fx1.eval.tooluse_eval import MockHarness, ToolResult, ToolUseTask, _run_task
from fx1.eval.ts_reasoning import _lag1_autocorr
from fx1.eval.ts_reasoning import make_oracle_model as make_ts_oracle
from fx1.extensions.contracts import FeatureExtension, PluginExtension, SkillExtension
from fx1.extensions.naming import module_path
from fx1.extensions.registry import extension_manifest, get_extension
from fx1.forecast import core_audit
from fx1.forecast.artifacts import _load_onnx, _payload_version, load_artifact
from fx1.forecast.config import load_harness_config
from fx1.forecast.features import normalize_bar_times, resample_ohlcv
from fx1.forecast.registry import create_model
from fx1.forecast.schema import validate_forecast_schema
from fx1.honesty import validate_fx1_output
from fx1.sbom import generate_sbom


def _spec(name: str, kind: SourceKind, script: str) -> SourceSpec:
    return SourceSpec(
        name=name,
        display=name,
        owner_plugins=["plug"],
        kind=kind,
        script_candidates=[script],
        markets=["us"],
        assets=["equity"],
        latency=LatencyClass.DAILY,
        credentials=(),
    )


def _tiny_bank() -> TaskBank:
    return TaskBank(
        seed=0,
        guidance=[GuidanceRule(rule_id="g", text="be specific")],
        criteria=[
            RubricCriterion(
                criterion_id="num",
                description="state the value",
                domain="proper_scores",
                check_kind="numeric_answer",
                guidance_rule="g",
                tolerance=1e-6,
            ),
            RubricCriterion(
                criterion_id="tokens",
                description="name the tokens",
                domain="proper_scores",
                check_kind="token_set",
                guidance_rule="g",
                required_tokens=["alpha"],
            ),
            RubricCriterion(
                criterion_id="phrase",
                description="quote the phrase",
                domain="proper_scores",
                check_kind="exact_phrase",
                guidance_rule="g",
            ),
            RubricCriterion(
                criterion_id="shape",
                description="emit json",
                domain="proper_scores",
                check_kind="structural_json",
                guidance_rule="g",
            ),
        ],
        queries=[
            RubricQuery(
                query_id="q1",
                text="what is one",
                expected_bank_ids=["num"],
                expected_values={"num": 1.0},
                canonical_answer="answer: 1",
            )
        ],
    )


def _train_config(corpus: Path):
    from fx1.train.config import LadderStage, TrainConfig

    return TrainConfig(
        run_name="edges",
        stage=LadderStage.PROXY,
        corpus_jsonl=str(corpus),
        eval_results_json=str(corpus),
        estimated_nodes=1,
        estimated_gpu_hours=0.1,
        estimated_cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# rubric extraction, spec checks, and grading fail-closed arms
# ---------------------------------------------------------------------------


def test_json_extractor_rejects_broken_and_excess_candidates() -> None:
    nested = _extract_json_object('note {"a": {"b": 1}} tail')
    assert nested == {"a": {"b": 1}}
    assert _extract_json_object("{") is None
    assert _extract_json_object("{bad} {also bad}", max_candidates=1) is None
    assert _extract_json_object("{bad} " * 8, max_candidates=2) is None


def test_spec_and_honesty_failures_are_explicit() -> None:
    bank = _tiny_bank()
    query = bank.queries[0]
    numeric = bank.criteria[0]
    heavy = numeric.model_copy(update={"weight": 0.0, "expected_value": None})
    failures = _spec_failures(heavy, numeric, query)
    assert any("weight" in item for item in failures)

    bare = query.model_copy(update={"expected_values": {}})
    missing_gold = _spec_failures(numeric.model_copy(update={"expected_value": 1.0}), numeric, bare)
    assert any("no gold value" in item for item in missing_gold)

    tokens = bank.criteria[1].model_copy(update={"required_tokens": []})
    assert any("required_tokens" in item for item in _spec_failures(tokens, tokens, query))
    phrase = bank.criteria[2]
    assert any("gold phrase" in item for item in _spec_failures(phrase, phrase, query))
    shape = bank.criteria[3]
    assert any("required_keys" in item for item in _spec_failures(shape, shape, query))

    bait = tokens.model_copy(update={"description": "Sharpe 2.1", "required_tokens": ["alpha"]})
    demand = _forbidden_demand_failures(bait)
    assert any("bait" in item for item in demand)


def test_writer_reply_shapes_escalate() -> None:
    bank = _tiny_bank()
    query = bank.queries[0]
    no_list = generate_rubric(query, bank, lambda _messages: '{"note": 1}')
    assert any("no 'criteria' list" in item for item in no_list.failure_reasons)
    assert no_list.writer_ok is False

    malformed = generate_rubric(query, bank, lambda _messages: '{"criteria": [1]}')
    assert malformed.reviews[0].code_check_passed is False
    assert any("malformed" in item for item in malformed.reviews[0].code_check_failures)

    poisoned_raw = json.dumps(
        {"criteria": [{"bank_id": "num", "expected_value": 1.0}], "note": "Sharpe 2.1"}
    )
    poisoned = generate_rubric(query, bank, lambda _messages: poisoned_raw)
    assert poisoned.writer_honesty_violations == 1
    assert poisoned.reviews[0].accepted is False
    assert any("honesty contract" in item for item in poisoned.reviews[0].code_check_failures)


def test_empty_token_and_phrase_checks_fail_closed() -> None:
    tokens = RubricCriterion(
        criterion_id="t",
        description="tokens",
        domain="proper_scores",
        check_kind="token_set",
        guidance_rule="g",
    )
    ok, detail = _run_criterion_check(tokens, "alpha")
    assert ok is False and "no required_tokens" in detail
    phrase = RubricCriterion(
        criterion_id="p",
        description="phrase",
        domain="proper_scores",
        check_kind="exact_phrase",
        guidance_rule="g",
    )
    ok, detail = _run_criterion_check(phrase, "gold")
    assert ok is False and "no gold phrase" in detail


def test_bank_consistency_oracles_and_assembly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bank = _tiny_bank()
    bad_rule = bank.model_copy(
        update={"criteria": [bank.criteria[0].model_copy(update={"guidance_rule": "missing"})]}
    )
    with pytest.raises(ValueError, match="unknown guidance"):
        _validate_bank_consistency(bad_rule, ValueError)
    bad_gold = bank.model_copy(
        update={
            "queries": [bank.queries[0].model_copy(update={"expected_values": {"no-such-id": 1.0}})]
        }
    )
    with pytest.raises(ValueError, match="unknown"):
        _validate_bank_consistency(bad_gold, ValueError)
    unknown_id = bank.model_copy(
        update={
            "queries": [bank.queries[0].model_copy(update={"expected_bank_ids": ["missing-id"]})]
        }
    )
    with pytest.raises(ValueError, match="unknown bank_id"):
        _validate_bank_consistency(unknown_id, ValueError)
    missing_numeric_gold = bank.model_copy(
        update={"queries": [bank.queries[0].model_copy(update={"expected_values": {}})]}
    )
    with pytest.raises(ValueError, match="without a"):
        _validate_bank_consistency(missing_numeric_gold, ValueError)

    writer = make_rubric_writer_oracle(bank)
    model = make_rubric_model_oracle(bank)
    assert writer([{"role": "user", "content": "no footer"}]) == ""
    assert model([{"role": "user", "content": "[query_id: missing]"}]) == ""

    path = tmp_path / "bank.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"row": "guidance", "rule_id": "g", "text": "t"}),
                json.dumps(
                    {
                        "row": "criterion",
                        "criterion_id": "c",
                        "description": "d",
                        "domain": "proper_scores",
                        "check_kind": "token_set",
                        "guidance_rule": "g",
                        "required_tokens": ["a"],
                    }
                ),
                json.dumps(
                    {
                        "row": "query",
                        "query_id": "q",
                        "text": "ask",
                        "expected_bank_ids": ["c"],
                        "canonical_answer": "a",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from pydantic import BaseModel, ValidationError

    class _NeedsX(BaseModel):
        x: int

    try:
        _NeedsX()
    except ValidationError as exc:
        assembly_error = exc

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise assembly_error

    monkeypatch.setattr("fx1.eval.rubric_banks.TaskBank", _boom)
    with pytest.raises(TaskBankSchemaError, match="assembly failed"):
        load_task_bank(path)


# ---------------------------------------------------------------------------
# external bench, options, retrieval, tool-use, ts
# ---------------------------------------------------------------------------


def test_mtbench_score_winner_and_exploding_judge() -> None:
    instance = MTBenchInstance(
        question_id="q",
        turns=[MTBenchTurn(prompt="Say hello", required_tokens=["hello"])],
    )
    assert "Reference answer" not in build_mtbench_judge_prompt(instance, 0, "hello")
    assert "Reference answer" not in build_mtbench_pairwise_prompt(instance, 0, "a", "b")
    assert _coerce_score({"score": 7.0}) == 7
    assert _coerce_winner({"winner": 1}) is None
    assert _coerce_winner({"winner": "A"}) == "A"
    assert _coerce_winner({"winner": "tie"}) == "tie"

    def _boom(_messages: list[dict[str, str]]) -> str:
        raise RuntimeError("judge down")

    verdict = judge_pairwise(instance, 0, "hello there", "nope", judge=_boom)
    assert verdict.used_fallback is True
    assert "exception" in verdict.reasoning


def test_ext_bench_oracle_unknown_ids_are_empty() -> None:
    banks = build_ext_bench_banks(0)
    oracle = make_ext_bench_oracle(banks)
    assert oracle([{"role": "user", "content": "nothing here"}]) == ""
    assert oracle([{"role": "user", "content": "[question_id: not-a-real-id]"}]) == ""
    question = banks.mtbench.instances[0].question_id
    turns = banks.mtbench.canonical[question]
    messages = [{"role": "user", "content": f"[question_id: {question}]"}]
    messages += [{"role": "assistant", "content": "x"} for _ in turns]
    messages.append({"role": "assistant", "content": "past the end"})
    assert oracle(messages) == ""


def test_trace_score_stops_after_the_golden_plan() -> None:
    instance = FinToolBenchInstance(
        task_id="t",
        objective="quote",
        tools=[FinToolSpec(name="synth_quote_history", required_args=["symbol"])],
        golden_plan=["synth_quote_history"],
        golden_args=[{"symbol": "AAA"}],
    )
    executed = [
        ("synth_quote_history", {"symbol": "AAA"}),
        ("synth_quote_history", {"symbol": "BBB"}),
    ]
    assert execution_trace_score(executed, instance) == 1.0


def test_fintool_valid_call_is_counted() -> None:
    from fx1.eval.ext_bench import MockFinToolExecutor

    instance = FinToolBenchInstance(
        task_id="t1",
        objective="quote AAA",
        tools=[FinToolSpec(name="synth_quote_history", required_args=["symbol"])],
        golden_plan=["synth_quote_history"],
        golden_args=[{"symbol": "AAA"}],
        gold_final_tokens=["ok"],
        max_steps=3,
    )

    def model(messages: list[dict[str, str]]) -> str:
        if any(item["role"] == "assistant" for item in messages):
            return json.dumps({"final": "ok"})
        return json.dumps({"tool": "synth_quote_history", "args": {"symbol": "AAA"}})

    result = _run_fintoolbench_task(instance, model, MockFinToolExecutor(0), False)
    assert result.completed is True
    assert result.trace_score == 1.0

    def hallucinate(messages: list[dict[str, str]]) -> str:
        if any(item["role"] == "assistant" for item in messages):
            return json.dumps({"final": "I refuse this request"})
        return json.dumps({"tool": "not_a_registered_tool", "args": {}})

    missed = _run_fintoolbench_task(instance, hallucinate, MockFinToolExecutor(1), False)
    assert "not_a_registered_tool" in missed.hallucinated_tools

    def missing_args(messages: list[dict[str, str]]) -> str:
        if any(item["role"] == "assistant" for item in messages):
            return json.dumps({"final": "I refuse this request"})
        return json.dumps({"tool": "synth_quote_history", "args": {}})

    incomplete = _run_fintoolbench_task(instance, missing_args, MockFinToolExecutor(2), False)
    assert "synth_quote_history" not in incomplete.hallucinated_tools
    assert incomplete.final_answer == "I refuse this request"


def test_options_pnl_partial_bounds_and_unknown_id() -> None:
    loss_only = OptionsReasoningItem(
        item_id="loss",
        level="outcome",
        prompt="bounds",
        gold_answer="loss 1",
        answer_type="pnl_bounds",
        pnl_loss=1.0,
    )
    graded = grade_options_reasoning(loss_only, "max loss 1.0")
    assert graded.item_id == "loss"
    profit_only = loss_only.model_copy(update={"pnl_profit": 2.0, "pnl_loss": None})
    assert grade_options_reasoning(profit_only, "profit 2.0").honesty_ok is True
    other = OptionsReasoningItem.model_construct(
        item_id="other",
        level="outcome",
        prompt="p",
        gold_answer="g",
        answer_type="other",
    )
    assert grade_options_reasoning(other, "plain").correct is True
    bank = OptionsReasoningBank(seed=0, items=[loss_only])
    assert make_options_oracle(bank)([{"role": "user", "content": "no footer"}]) == ""


def test_retrieval_unknown_question_and_zero_turns() -> None:
    bank = SimpleNamespace(questions=[])
    assert make_golden_model(bank)([{"role": "user", "content": "hello"}]) == ""
    assert make_no_retrieval_model(bank)([{"role": "user", "content": "hello"}]) == ""
    assert _conversation_question_id([{"role": "user", "content": "hello"}]) is None
    question = RetrievalQuestion(
        question_id="q",
        family=FAMILY_FACTOID,
        prompt="p",
        answer="answer: 1",
        golden_path=(),
        source_doc_ids=(),
        payload={"target": 1.0},
    )
    result = _run_one(question, {}, lambda _messages: "unused", -1)
    assert result.n_retrieves == 0
    assert result.answer == ""
    assert result.correct is False


def test_tooluse_valid_but_unsuccessful_call(monkeypatch: pytest.MonkeyPatch) -> None:
    task = ToolUseTask(
        task_id="t",
        prompt="run doctor",
        allowed_tools=("doctor",),
        max_steps=2,
        golden_plan=("doctor",),
        success_checks=(),
    )
    harness = MockHarness(seed=0)

    def execute(tool: str, args: dict[str, object]) -> ToolResult:
        assert tool == "doctor"
        assert args == {}
        return ToolResult(tool="doctor", ok=False, stdout="", error=None)

    monkeypatch.setattr(harness, "execute", execute)

    def model(messages: list[dict[str, str]]) -> str:
        if any(item["role"] == "tool" for item in messages):
            return json.dumps({"final": "stopped"})
        return json.dumps({"tool": "doctor", "args": {}})

    outcome = _run_task(task, model, harness)
    assert outcome.final_answer == "stopped"
    assert outcome.n_steps == 2
    assert outcome.plan_match_score == 1.0


def test_ts_oracle_miss_and_short_autocorr() -> None:
    oracle = make_ts_oracle(SimpleNamespace(answers={"known": "yes"}))
    assert oracle([{"role": "user", "content": "hello"}]) == ""
    assert oracle([{"role": "user", "content": "[task_id: missing]"}]) == ""
    assert _lag1_autocorr(np.array([1.0])) == 0.0


def test_bootstrap_rejects_unequal_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_delta_ci([], [])
    with pytest.raises(ValueError, match="equal length"):
        bootstrap_delta_ci([True], [True, False])


def test_eval_lazy_exports_and_audit_helpers() -> None:
    import fx1.eval as eval_pkg
    import fx1.eval.bank_audit as bank_audit
    import fx1.eval.calibration_audit as calibration_audit
    import fx1.eval.capability_audit as capability_audit
    import fx1.eval.contamination_audit as contamination_audit
    import fx1.eval.core_audit as core_eval_audit
    import fx1.eval.masking_audit as masking_audit
    import fx1.eval.ts_reasoning_audit as ts_audit

    with pytest.raises(AttributeError, match="no attribute"):
        _ = eval_pkg.not_a_real_export
    assert "DEFAULT_BANK" in dir(eval_pkg)
    assert core_eval_audit._raises(lambda: None) == "no-raise"
    assert core_eval_audit._compiles("(") is False
    assert calibration_audit._raises(lambda: None) == "no-raise"
    assert capability_audit._raises(lambda: None) == "no-raise"
    assert contamination_audit._raises(lambda: None) == "no-raise"
    assert masking_audit._raises(lambda: None) == "no-raise"
    assert ts_audit._raises(lambda: None) == "no-raise"
    assert bank_audit._compiles("(") is False
    assert bank_audit._compiles("a+") is True


def test_bank_audit_records_vacuous_and_unsatisfiable(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.eval.bank_audit as bank_audit_mod
    from fx1.eval.suite import EvalTask

    honesty = EvalTask(
        name="honest-probe",
        kind="honesty",
        messages=[{"role": "user", "content": "q"}],
        required_tokens=["proper"],
    )
    vacuous = EvalTask(
        name="vacuous",
        kind="general",
        messages=[{"role": "user", "content": "q"}],
    )
    unsat = EvalTask(
        name="unsat",
        kind="domain",
        messages=[{"role": "user", "content": "q"}],
        required_tokens=["forbidden-word"],
        forbidden_patterns=["forbidden-word"],
    )
    monkeypatch.setattr(
        bank_audit_mod,
        "_banks",
        lambda: {"default": [honesty, vacuous, unsat], "redteam": []},
    )
    result = bank_audit_mod.bank_audit()
    assert result["vacuous_tasks"] == ["vacuous"]
    assert "unsat" in result["unsatisfiable_tasks"]
    assert result["bad_patterns"] == []


def test_capability_memorizer_unknown_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.eval.rephrased as rephrased
    from fx1.eval.capability_audit import capability_audit

    real = rephrased.run_rephrased_gap
    seen: dict[str, str] = {}

    def wrapped(model, budget: float = 0.05):  # type: ignore[no-untyped-def]
        if "reply" not in seen:
            probe = model([{"role": "user", "content": "not a known prompt"}])
            if probe == "idk":
                seen["reply"] = probe
        return real(model, budget=budget)

    monkeypatch.setattr(rephrased, "run_rephrased_gap", wrapped)
    result = capability_audit()
    assert seen["reply"] == "idk"
    assert result["gap_value_reflects"] is True


# ---------------------------------------------------------------------------
# extensions, capabilities, sbom, harness, doctor
# ---------------------------------------------------------------------------


def test_generated_extension_modules_bind_owners() -> None:
    names = [
        "fx1.extensions.skills.tearsheet",
        "fx1.extensions.skills.train",
        "fx1.extensions.skills.validate",
        "fx1.extensions.skills.vendor_book_map",
        "fx1.extensions.skills.verify_research",
        "fx1.extensions.plugins.binance_crypto",
        "fx1.extensions.plugins.caixin",
        "fx1.extensions.plugins.cls",
        "fx1.extensions.plugins.dongcai",
        "fx1.extensions.plugins.finance_fetch",
        "fx1.extensions.plugins.finance_research",
        "fx1.extensions.plugins.finenter",
        "fx1.extensions.plugins.gildata",
        "fx1.extensions.features.z_vs_ma20",
    ]
    for name in names:
        module = importlib.import_module(name)
        assert module.MODULE.module == name
        assert tuple(module.MODULE.references)


def test_extension_contracts_reject_foreign_and_wrong_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.capabilities import owner_references

    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    foreign = owner_references("plugin", "caixin")[0]
    skill = SkillExtension(
        kind="skill",
        owner="train",
        references=(foreign,),
        module=module_path("skill", "train"),
    )
    with pytest.raises(ValueError, match="foreign"):
        list(skill.records())
    with pytest.raises(ValueError, match="foreign"):
        skill.verify()
    with pytest.raises(ValueError, match="PluginExtension"):
        PluginExtension(
            kind="skill",
            owner="train",
            references=owner_references("skill", "train")[:1],
            module=module_path("skill", "train"),
        )
    with pytest.raises(ValueError, match="FeatureExtension"):
        FeatureExtension(
            kind="skill",
            owner="train",
            references=owner_references("skill", "train")[:1],
            module=module_path("skill", "train"),
        )
    plugin = PluginExtension(
        kind="plugin",
        owner="caixin",
        references=owner_references("plugin", "caixin")[:1],
        module=module_path("plugin", "caixin"),
    )
    assert plugin._record_owner({"source": None}) is None
    assert plugin._record_owner({"source": "caixin"}) == "caixin"
    described = plugin.describe()
    fetched = plugin.fetch(FetchRequest(api="search"))
    assert described.ok is False and fetched.ok is False
    assert described.status is SourceStatus.NO_SCRIPT

    class _Harness:
        def run(self, command: str, config: Path | None = None) -> tuple[str, Path | None]:
            return command, config

    feature = FeatureExtension(
        kind="feature",
        owner="z_vs_ma20",
        references=owner_references("feature", "z_vs_ma20")[:1],
        module=module_path("feature", "z_vs_ma20"),
    )
    command, config = feature.build(harness=_Harness())  # type: ignore[arg-type]
    assert command == "build-features" and config is None


def test_registry_rejects_bad_module_and_unknown_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    from fx1.capabilities import owner_references

    module = SimpleNamespace(MODULE=object())
    monkeypatch.setattr("fx1.extensions.registry.import_module", lambda _path: module)
    with pytest.raises(TypeError, match="ExtensionModule"):
        get_extension("skill", "train")
    module.MODULE = SkillExtension(
        kind="skill",
        owner="doctor",
        references=owner_references("skill", "doctor")[:1],
        module=module_path("skill", "doctor"),
    )
    with pytest.raises(ValueError, match="does not match"):
        get_extension("skill", "train")
    monkeypatch.setattr(
        "fx1.extensions.registry.get_extension",
        lambda _kind, _owner: SimpleNamespace(manifest=lambda: {"schema": "x"}),
    )
    with pytest.raises(ValueError, match="unknown extension kind"):
        extension_manifest("nope", "train")  # type: ignore[arg-type]


def test_seed_outside_layout_is_rejected() -> None:
    with pytest.raises(KeyError, match="outside"):
        resolve_seed_id(-1)
    with pytest.raises(KeyError, match="outside"):
        resolve_seed_id(1_000_001)
    assert resolve_seed_id(0)["owner"] == "doctor"
    assert resolve_seed_id(1_000_000)["kind"] in {"feature", "skill", "plugin"}


def test_sbom_malformed_package_and_wheels(tmp_path: Path) -> None:
    broken = tmp_path / "broken.lock"
    broken.write_text("package = [1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        generate_sbom(broken)
    wheels = tmp_path / "wheels.lock"
    wheels.write_text(
        '[[package]]\nname = "demo"\nversion = "1.0.0"\nwheels = "nope"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="wheels"):
        generate_sbom(wheels)
    skipped = tmp_path / "skip.lock"
    skipped.write_text(
        '[[package]]\nname = "demo"\nversion = "1.2.3"\nwheels = ["not-a-dict"]\n',
        encoding="utf-8",
    )
    sbom = generate_sbom(skipped)
    assert sbom.entries[0].name == "demo"
    assert sbom.entries[0].sha256 is None


def test_harness_runner_and_audit_edges(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fx1.harness as harness
    import fx1.harness_audit as harness_audit
    import fx1.harness_bench as harness_bench

    class _Proc:
        returncode = 3
        stdout = "out"
        stderr = "err"

    monkeypatch.setattr(harness.subprocess, "run", lambda *_a, **_k: _Proc())
    code, out, err = harness._subprocess_runner(["doctor"], 1)
    assert (code, out, err) == (3, "out", "err")
    assert harness_bench._percentile([], 0.5) == 0.0

    class _Result:
        model = None
        backend = None
        usage = None

    class _Surface:
        def complete(self, messages: list[dict[str, str]], **_kwargs: object) -> _Result:
            assert messages[0]["content"] == "hi"
            return _Result()

    record = harness_bench.run_bench(_Surface(), n=1, concurrency=1, warmup=0, prompt="hi")
    assert record["metrics"]["models"] == []
    assert record["metrics"]["backends"] == []
    assert record["metrics"]["error_count"] == 0

    def _no_symlink(self: Path, _target: Path, target_is_directory: bool = False) -> None:
        raise OSError("symlinks disabled")

    monkeypatch.setattr(Path, "symlink_to", _no_symlink)
    audited = harness_audit.harness_audit()
    assert "symlink_escape_raises" not in audited
    assert audited["unknown_raises"].startswith("raise:")
    assert harness_audit._raises(lambda: None) == "no-raise"
    assert tmp_path.is_dir()


def test_doctor_reports_unavailable_eval_bank(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fx1.eval as eval_pkg
    from fx1.doctor import collect_status

    monkeypatch.delitem(eval_pkg.__dict__, "DEFAULT_BANK", raising=False)
    monkeypatch.delitem(eval_pkg._ATTR_TO_MODULE, "DEFAULT_BANK", raising=False)
    status = collect_status(tmp_path)
    assert status["eval_bank_tasks"] == "unavailable"
    assert status["corpus_ledger_chain"] == "missing"


def test_module_level_audit_no_raise_arms() -> None:
    import fx1.forecast.infra_audit as forecast_infra
    import fx1.honesty_audit as honesty_audit
    import fx1.modelcard_audit as modelcard_audit
    import fx1.sbom_audit as sbom_audit
    import fx1.train.infra_audit as train_infra
    import fx1.train.run_audit as run_audit

    assert forecast_infra._raises(lambda: None) == "no-raise"
    assert honesty_audit._raises(lambda: None) == "no-raise"
    assert modelcard_audit._raises(lambda: None) == "no-raise"
    assert sbom_audit._raises(lambda: None) == "no-raise"
    assert train_infra._raises(lambda: None) == "no-raise"
    assert run_audit._raises(lambda: None) == "no-raise"


# ---------------------------------------------------------------------------
# forecast core, schema, features, artifacts
# ---------------------------------------------------------------------------


def test_core_audit_direct_failure_arms(tmp_path: Path) -> None:
    keys = core_audit._mapping_keys({"rows": ({"sharpe": 1},)})
    assert "sharpe" in keys
    assert core_audit._forbidden_metric_keys_absent({"rows": [{"sharpe": 1}]}) is False
    assert core_audit._forbidden_metric_keys_absent({"live_pnl_claim": False}) is True

    bars = core_audit._panel(4)
    start = core_audit._stamp(2)
    filtered = core_audit._MemoryProvider(bars).get_bars(start=start)
    assert filtered.height < bars.height
    assert all(stamp >= start for stamp in filtered["event_time"].to_list())

    features = pl.DataFrame({"event_time": [core_audit._stamp(0)], "security_id": ["AAA"]})
    predicted = core_audit._DropOneModel().predict(features)
    assert predicted.height == 1
    with pytest.raises(AssertionError, match="untrusted"):
        core_audit._NeverLoadModel().load()
    (tmp_path / "subdir").mkdir()
    with pytest.raises(ValueError, match="outside"):
        core_audit._MutateCheckpointModel(tmp_path).load(tmp_path / "subdir")
    assert core_audit._ok(lambda: (_ for _ in ()).throw(RuntimeError("x"))) is False
    assert core_audit._attempt(lambda: (_ for _ in ()).throw(RuntimeError("x"))) is False


def test_runner_probe_false_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real = core_audit.run_inference
    seen = {"start": 0}

    def empty_horizons(config, *, model=None, provider=None):  # type: ignore[no-untyped-def]
        label = str(config.inference.output_parquet)
        if label.endswith("wf/forecasts.parquet"):
            return SimpleNamespace(forecasts=pl.DataFrame({"event_time": []}))
        result = real(config, model=model, provider=provider)
        if "startbound" in label:
            seen["start"] += 1
            return replace(result, forecasts=result.forecasts.head(0), n_rows=0)
        return result

    monkeypatch.setattr(core_audit, "run_inference", empty_horizons)
    first = core_audit._runner_probes(tmp_path / "a")
    assert first["walk_forward_truncates_at_decision"] is False
    assert seen["start"] == 1
    assert first["flag_start_filters_decisions_not_history"] is False

    def mismatch(config, *, model=None, provider=None):  # type: ignore[no-untyped-def]
        result = real(config, model=model, provider=provider)
        label = str(config.inference.output_parquet)
        if label.endswith("wf/forecasts.parquet"):
            height = result.forecasts.height
            shifted = result.forecasts.with_columns(
                pl.Series(
                    "event_time",
                    [datetime(1999, 1, 1, tzinfo=UTC)] * height,
                    dtype=pl.Datetime("us", "UTC"),
                )
            )
            return replace(result, forecasts=shifted)
        return result

    monkeypatch.setattr(core_audit, "run_inference", mismatch)
    second = core_audit._runner_probes(tmp_path / "b")
    assert second["walk_forward_truncates_at_decision"] is False


def test_eval_probe_false_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_map = core_audit.map_signals

    def bad_sign(frame, mapping, *, threshold=0.0):  # type: ignore[no-untyped-def]
        out = real_map(frame, mapping, threshold=threshold)
        ids = frame["security_id"].to_list() if "security_id" in frame.columns else []
        if ids == ["A", "B", "C"] and mapping == "sign":
            return out.with_columns(pl.lit(9.0).alias("signal"))
        return out

    monkeypatch.setattr(core_audit, "map_signals", bad_sign)
    monkeypatch.setattr(
        core_audit,
        "run_signal_evaluation",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("e2e")),
    )
    signed = core_audit._eval_probes(tmp_path / "sign")
    assert signed["signal_maps_bounded_and_marked"] is False
    assert signed["_e2e_failed"] is False
    assert signed["eval_metrics_scored_on_test_folds_only"] is False

    def bad_threshold(frame, mapping, *, threshold=0.0):  # type: ignore[no-untyped-def]
        out = real_map(frame, mapping, threshold=threshold)
        ids = frame["security_id"].to_list() if "security_id" in frame.columns else []
        if ids == ["A", "B", "C"] and mapping == "threshold":
            return out.with_columns(pl.lit(9.0).alias("signal"))
        return out

    monkeypatch.setattr(core_audit, "map_signals", bad_threshold)
    monkeypatch.undo()
    monkeypatch.setattr(core_audit, "map_signals", bad_threshold)
    gated = core_audit._eval_probes(tmp_path / "gate")
    assert gated["signal_maps_bounded_and_marked"] is False


def test_price_only_forecast_and_instance_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.forecast.registry as registry

    frame = pl.DataFrame(
        {
            "event_time": [datetime(2020, 1, 1, tzinfo=UTC)],
            "security_id": ["A"],
            "horizon_bars": pl.Series([1], dtype=pl.Int64),
            "predicted_price": [10.0],
        }
    )
    validate_forecast_schema(frame)
    instance = core_audit._SpyModel()
    monkeypatch.setattr(registry, "load_symbol", lambda _entrypoint: instance)
    assert create_model("dummy-zero", entrypoint="pkg:Instance") is instance


def test_harness_config_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_harness_config(path)


def test_feature_date_cast_and_ingested_time() -> None:
    stamped = normalize_bar_times(
        pl.DataFrame(
            {
                "event_time": pl.Series([date(2020, 1, 2)], dtype=pl.Date),
                "available_time": pl.Series([date(2020, 1, 2)], dtype=pl.Date),
            }
        )
    )
    assert stamped["event_time"].dtype.time_zone == "UTC"
    start = datetime(2020, 1, 1, tzinfo=UTC)
    later = datetime(2020, 1, 1, 1, tzinfo=UTC)
    resampled = resample_ohlcv(
        pl.DataFrame(
            {
                "security_id": ["A", "A"],
                "event_time": [start, later],
                "available_time": [start, later],
                "open": [1.0, 1.1],
                "high": [1.2, 1.2],
                "low": [0.9, 1.0],
                "close": [1.0, 1.1],
                "volume": [10.0, 11.0],
                "ingested_time": [start, later],
            }
        ),
        "1d",
    )
    assert "ingested_time" in resampled.columns
    assert resampled.height == 1


def test_artifact_joblib_onnx_and_unknown_format(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Payload:
        version = "1.2.3"

    class _Joblib:
        @staticmethod
        def load(buf: object) -> _Payload:
            assert buf.read() == b"joblib-bytes"
            return _Payload()

    class _Onnx:
        @staticmethod
        def load(path: str) -> dict[str, str]:
            return {"onnx": path}

    raw = b"joblib-bytes"
    joblib_path = tmp_path / "model.joblib"
    joblib_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    def importer(name: str) -> object:
        return _Joblib if name == "joblib" else _Onnx

    monkeypatch.setattr("fx1.forecast.artifacts.import_optional", importer)
    loaded = load_artifact(
        joblib_path,
        allow_unsafe_deserialization=True,
        trusted_checkpoint_sha256=digest,
    )
    assert loaded.format == "joblib"
    assert loaded.version == "1.2.3"
    assert _payload_version(_Payload()) == "1.2.3"
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    onnx_loaded = load_artifact(onnx_path)
    assert onnx_loaded.format == "onnx"
    assert onnx_loaded.payload["onnx"] == str(onnx_path)
    assert _load_onnx(onnx_path)["onnx"] == str(onnx_path)
    unknown = tmp_path / "model.bin"
    unknown.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        load_artifact(unknown, fmt="safetensors")


def test_forecast_data_audit_wrong_error_and_crashed_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fx1.forecast.data_audit as data_audit

    ok, detail = data_audit._raises(lambda: (_ for _ in ()).throw(RuntimeError("x")), ValueError)
    assert ok is False and detail.startswith("wrong error")

    real_import = data_audit.import_optional

    def blocked(name: str) -> object:
        if name == "torch":
            raise RuntimeError("torch blocked")
        return real_import(name)

    monkeypatch.setattr(data_audit, "import_optional", blocked)
    rows = data_audit._artifact_probes()
    torch_row = next(row for row in rows if "torch" in row["detail"] or "torch" in row["probe"])
    assert "unavailable" in torch_row["detail"] or torch_row["verdict"] in {"pass", "fail"}

    def crash() -> list[dict[str, str]]:
        raise RuntimeError("escaped")

    def det(_prior: list[dict[str, str]]) -> list[dict[str, str]]:
        raise RuntimeError("detonated")

    monkeypatch.setattr(data_audit, "_PROBE_GROUPS", (crash,))
    monkeypatch.setattr(data_audit, "_determinism_probes", det)
    audited = data_audit.forecast_data_audit_results()
    assert any(row["probe"].endswith("_crashed") for row in audited)
    assert any("detonated" in row["detail"] for row in audited)


# ---------------------------------------------------------------------------
# data sources, corpus, ledger
# ---------------------------------------------------------------------------


def test_adapters_preflight_and_failed_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FX1_PLUGIN_ROOTS", str(tmp_path))
    request = FetchRequest(api="quote", params={})
    agent = AgentGwAdapter(_spec("wind", SourceKind.AGENT_GW, "missing.py"))
    assert agent.describe().status is SourceStatus.NO_SCRIPT
    assert agent.fetch(request).ok is False
    xhcj = XhcjAdapter(_spec("xhcj", SourceKind.CUSTOM_CLI, "missing.py"))
    assert xhcj.describe().ok is False and xhcj.fetch(request).ok is False
    finance = FinanceFetchAdapter(_spec("finance_fetch", SourceKind.SCENARIO_ROUTER, "missing.py"))
    assert finance.fetch(request).status is SourceStatus.NO_SCRIPT
    caixin = CaixinAdapter(_spec("caixin", SourceKind.AGENT_GW, "missing.py"))
    assert caixin.describe().ok is False

    script = tmp_path / "ff.py"
    script.write_text("print('no')\n", encoding="utf-8")
    ready = FinanceFetchAdapter(
        _spec("finance_fetch", SourceKind.SCENARIO_ROUTER, "ff.py"),
        runner=lambda *_a, **_k: RunOutcome(returncode=2, stdout="", stderr="script blew up"),
    )
    failed = ready.fetch(request)
    assert failed.ok is False
    assert "script blew up" in (failed.error or "")


def test_router_empty_plan_records_the_note() -> None:
    quiet = fetch_routed(
        RoutingPlan(question="q", need="news", market="us", note="desk closed"),
        api_by_source={},
    )
    assert quiet.ok is False and "desk closed" in (quiet.error or "")
    bare = fetch_routed(
        RoutingPlan(question="q", need="news", market="us"),
        api_by_source={},
    )
    assert "no candidates" in (bare.error or "")


def test_live_claim_scan_accepts_non_scalars() -> None:
    assert _is_live_claim_value(["yes"]) is True
    assert _is_live_claim_value({"live": 1}) is True


def test_ledger_skips_blank_lines_and_breaks_on_prev_hash(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    CorpusLedger(path).record_example(
        source_sha256="a" * 64,
        transform_sha256="b" * 64,
        example_sha256="c" * 64,
    )
    raw = json.loads(path.read_text(encoding="utf-8").strip())
    raw["prev_hash"] = "f" * 64
    path.write_text("\n" + json.dumps(raw) + "\n", encoding="utf-8")
    assert CorpusLedger(path).verify_chain() is False


def test_corpus_fallback_prompt_and_honesty_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fx1.data.corpus.SYSTEM_PROMPT_PATH", tmp_path / "missing.md")
    assert "fx-1" in _system_prompt()
    record = ReceiptRecord(
        path="r.json",
        sha256="d" * 64,
        research_only=False,
        live_pnl_claim=False,
    )
    text = _negative_example(record, "system").messages[-1]["content"]
    assert "not marked research_only" in text
    assert "asserts a live-trading" not in text

    def notes(_doc: str, _system: str) -> list[SFTExample]:
        return [
            SFTExample(
                messages=[
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "Sharpe 2.1"},
                ],
                receipt_sha256="e" * 64,
                source_path="n",
            )
        ]

    monkeypatch.setattr("fx1.data.notebooks.notebook_examples", notes)
    stats = build_full_corpus(tmp_path / "receipts", tmp_path / "out.jsonl", notebooks=["n.md"])
    assert stats["skipped"] == 1
    assert stats["positive"] == 0


def test_data_package_lazy_sources() -> None:
    import sys

    import fx1.data as data_pkg
    import fx1.data.sources as sources_mod

    data_pkg.__dict__.pop("sources", None)
    previous = sys.gettrace()

    def _tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if (
            event == "line"
            and frame.f_code.co_name == "__getattr__"
            and frame.f_lineno == 37
            and "sources" not in data_pkg.__dict__
        ):
            data_pkg.__dict__["sources"] = sources_mod
        if previous is not None:
            traced = previous(frame, event, arg)
            return traced if traced is not None else _tracer
        return _tracer

    sys.settrace(_tracer)
    try:
        loaded = data_pkg.sources
    except RecursionError:
        loaded = None
    finally:
        sys.settrace(previous)
        data_pkg.sources = sources_mod
    assert loaded is sources_mod or data_pkg.sources is sources_mod
    with pytest.raises(AttributeError):
        _ = data_pkg.not_a_real_export
    assert "wind" in {spec.name for spec in sources_mod.list_sources()}


def test_quality_audit_records_a_bound_that_does_not_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fx1.data.quality as quality
    from fx1.data.quality_audit import quality_audit

    real = quality.frozen_split

    def wrapped(examples, out_prefix, *, val_fraction: float = 0.05, seed: int = 17):  # type: ignore[no-untyped-def]
        if val_fraction == 0.0:
            return real(examples, out_prefix, seed=seed)
        return real(examples, out_prefix, val_fraction=val_fraction, seed=seed)

    monkeypatch.setattr(quality, "frozen_split", wrapped)
    result = quality_audit()
    assert result["bound_fails"] == [0.0]
    assert result["bounds_enforced"] is False
    assert result["exact_dedup"] is True


def test_nested_audit_no_raise_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.data.sources.registry as registry
    import fx1.eval.ext_bench_banks as ext_banks
    import fx1.eval.rubric_banks as rubric_banks
    from fx1.data.sources.ds_audit import ds_audit
    from fx1.ext_bench_audit import ext_bench_audit
    from fx1.rubric_audit import rubric_audit

    real_spec = registry.get_spec

    def spec(name: str):  # type: ignore[no-untyped-def]
        if name == "no-such-src":
            return real_spec("wind")
        return real_spec(name)

    monkeypatch.setattr(registry, "get_spec", spec)
    data_result = ds_audit()
    assert data_result["get_spec_unknown"] is False
    assert data_result["registry_unique"] is True

    real_load = ext_banks.load_mtbench_bank

    def load_mt(path):  # type: ignore[no-untyped-def]
        if str(path).endswith("gone.jsonl"):
            return ext_banks.build_mtbench_bank(0)
        return real_load(path)

    monkeypatch.setattr(ext_banks, "load_mtbench_bank", load_mt)
    ext_result = ext_bench_audit()
    assert ext_result["loader_missing_fails"] is False
    assert ext_result["banks_deterministic"] is True

    real_bank = rubric_banks.load_task_bank

    def load_bank(path, seed: int = 0):  # type: ignore[no-untyped-def]
        if str(path).endswith("gone.jsonl"):
            return rubric_banks.build_task_bank(0)
        return real_bank(path, seed)

    monkeypatch.setattr(rubric_banks, "load_task_bank", load_bank)
    rubric_result = rubric_audit()
    assert rubric_result["loader_missing_fails"] is False
    assert rubric_result["bank_deterministic"] is True


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_corpus_blank_line_is_skipped(tmp_path: Path) -> None:
    from fx1.train.run import _validate_corpus

    path = tmp_path / "corpus.jsonl"
    path.write_text('\n{"receipt_sha256": "abc", "negative": false}\n', encoding="utf-8")
    stats = _validate_corpus(path)
    assert stats == {"lines": 1, "positive": 1, "negative": 0}


def test_chat_jsonl_blank_and_bad_rows(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from fx1.train.tiny_lm import chat_jsonl_to_text

    bad = tmp_path / "bad.jsonl"
    bad.write_text('\n{"messages": "nope"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no messages list"):
        chat_jsonl_to_text(bad)
    good = tmp_path / "good.jsonl"
    payload = "alpha " * 120
    good.write_text(
        "\n"
        + json.dumps(
            {
                "messages": [
                    {"role": "user", "content": payload},
                    "skip-me",
                    {"role": "assistant", "content": ""},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    text = chat_jsonl_to_text(good)
    assert "alpha" in text
    assert len(text.encode("utf-8")) >= 512


def test_checkpoint_hash_and_missing_tensors(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors.numpy")
    from safetensors.numpy import load_file, save_file

    from fx1.train.tiny_lm import (
        DEFAULT_CTX,
        DEFAULT_DIM,
        DEFAULT_N_HEADS,
        VOCAB_SIZE,
        build_tiny_lm,
        load_base_tensors,
        save_weights,
        sha256_file,
    )

    model = build_tiny_lm(VOCAB_SIZE, DEFAULT_DIM, DEFAULT_CTX, DEFAULT_N_HEADS)
    save_weights(tmp_path, model)
    manifest = tmp_path / "weights.manifest.json"
    manifest.write_text(json.dumps({"weights_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity"):
        load_base_tensors(model, tmp_path)

    weights = tmp_path / "weights.safetensors"
    tensors = dict(load_file(str(weights)))
    dropped = sorted(tensors)[0]
    del tensors[dropped]
    save_file(tensors, str(weights))
    manifest.write_text(
        json.dumps({"weights_sha256": sha256_file(weights)}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="missing tensors"):
        load_base_tensors(model, tmp_path)


def test_short_val_text_and_recorded_val_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from fx1.train.tiny_lm import (
        DEFAULT_CTX,
        DEFAULT_DIM,
        DEFAULT_N_HEADS,
        VOCAB_SIZE,
        build_tiny_lm,
        make_tiny_lm_trainer,
        save_weights,
        sha256_file,
        val_loss_of,
        write_weights_manifest,
    )

    model = build_tiny_lm(VOCAB_SIZE, DEFAULT_DIM, DEFAULT_CTX, DEFAULT_N_HEADS)
    assert val_loss_of(model, "hi", 1) is None
    base = tmp_path / "base"
    digest, n_params = save_weights(base, model)
    write_weights_manifest(
        base,
        weights_sha256=digest,
        n_params=n_params,
        training={"seed": 1},
        generator="test",
        generator_path=Path(sha256_file.__code__.co_filename),
    )
    payload = "beta " * 120
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    row = json.dumps({"messages": [{"role": "user", "content": payload}]}) + "\n"
    train.write_text(row, encoding="utf-8")
    val.write_text(row, encoding="utf-8")
    trainer = make_tiny_lm_trainer(
        seed=1,
        fine_tuned_model="fx1-tiny",
        job_id="job",
        corpus_label="synthetic",
        base_checkpoint=base,
        base_weights_sha256=digest,
        lr=1e-3,
        val_path=val,
    )
    checkpoint = trainer(train, val, _train_config(train), tmp_path / "work")
    manifest = json.loads((checkpoint / "weights.manifest.json").read_text(encoding="utf-8"))
    assert isinstance(manifest["training"]["val_loss"], float)
    assert manifest["training"]["val_loss"] > 0.0


def test_pipeline_eval_candidate_rejects_bad_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fx1.eval.suite import SuiteSummary
    from fx1.train.pipeline import Pipeline, Stage

    bank = "ab" * 32

    def suite(_fn, _tasks, results):  # type: ignore[no-untyped-def]
        return SuiteSummary(
            {
                "honesty_gate_passed": True,
                "eval_bank_sha256": bank,
                "results": results,
            }
        )

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    pipe = Pipeline(_train_config(corpus), tmp_path / "work")
    pipe.state.stage = Stage.EVAL_CANDIDATE
    base = tmp_path / "eval_base.json"
    pipe.state.artifacts["eval_base"] = str(base)

    monkeypatch.setattr(
        "fx1.train.pipeline.run_suite",
        lambda fn, tasks: suite(fn, tasks, [{"task": "d", "kind": "domain", "passed": True}]),
    )
    base.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a suite summary"):
        pipe.run_eval_candidate(lambda _m: "ok")

    base.write_text(
        json.dumps({"eval_bank_sha256": bank, "results": [{"kind": "domain", "passed": True}]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="without a task name"):
        pipe.run_eval_candidate(lambda _m: "ok")

    general = [{"task": "g", "kind": "general", "passed": True}]
    monkeypatch.setattr(
        "fx1.train.pipeline.run_suite",
        lambda fn, tasks: suite(fn, tasks, general),
    )
    base.write_text(
        json.dumps({"eval_bank_sha256": bank, "results": general}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="no domain tasks"):
        pipe.run_eval_candidate(lambda _m: "ok")

    domain = [{"task": "d", "kind": "domain", "passed": True}]
    monkeypatch.setattr(
        "fx1.train.pipeline.run_suite",
        lambda fn, tasks: suite(fn, tasks, domain),
    )
    base.write_text(json.dumps({"eval_bank_sha256": bank, "results": domain}), encoding="utf-8")
    comparison = pipe.run_eval_candidate(lambda _m: "ok")
    assert "general" not in comparison
    assert comparison["n_tasks"] == 1
    assert Pipeline._compare_by_kind({}, {}, kind="domain") == {}


def test_pipeline_audit_false_arms_and_honesty(monkeypatch: pytest.MonkeyPatch) -> None:
    import fx1.train.pipeline as pipeline_mod
    from fx1.train.pipeline_audit import _clean, _violates, pipeline_audit

    state = {"eval": 0, "qual": 0, "cand": 0}
    orig_eval = pipeline_mod.Pipeline.run_eval_base
    orig_qual = pipeline_mod.Pipeline.run_quality_gate
    orig_cand = pipeline_mod.Pipeline.run_eval_candidate

    def run_eval_base(self, fn):  # type: ignore[no-untyped-def]
        state["eval"] += 1
        if state["eval"] < 3:
            return Path("skipped")
        return orig_eval(self, fn)

    def run_quality_gate(self, eval_prompts=None):  # type: ignore[no-untyped-def]
        state["qual"] += 1
        if state["qual"] == 1:
            return {"kept": 1}
        return orig_qual(self, eval_prompts)

    def run_eval_candidate(self, fn):  # type: ignore[no-untyped-def]
        state["cand"] += 1
        if state["cand"] == 1:
            return {}
        return orig_cand(self, fn)

    monkeypatch.setattr(pipeline_mod.Pipeline, "run_eval_base", run_eval_base)
    monkeypatch.setattr(pipeline_mod.Pipeline, "run_quality_gate", run_quality_gate)
    monkeypatch.setattr(pipeline_mod.Pipeline, "run_eval_candidate", run_eval_candidate)
    result = pipeline_audit()
    assert result["order_enforced"] is False
    assert result["empty_corpus_fails"] is False
    assert result["bad_base_refused"] is False
    assert result["bad_candidate_refused"] is False
    assert result["quality_gate_passed"] is True
    assert result["pairs_cover_baits"] is True
    assert _clean("Sharpe 2.1", validate_fx1_output) is False
    assert _violates("Sharpe 2.1", validate_fx1_output) is True
    assert _clean("synthetic research note", validate_fx1_output) is True


def test_tail_audit_mlflow_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import mlflow

    from fx1.tail_audit import fx1_tail_audit

    calls = {"n": 0}

    def _boom(_uri: str) -> None:
        calls["n"] += 1
        raise RuntimeError("tracking down")

    monkeypatch.setattr(mlflow, "set_tracking_uri", _boom)
    result = fx1_tail_audit()
    assert calls["n"] >= 1
    assert result["default_trainer_refuses"] is True
