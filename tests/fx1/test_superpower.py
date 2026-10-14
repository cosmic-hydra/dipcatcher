"""/superpower: capability registry, planning, coordination, approvals, honesty.

Offline: harness, researcher, model, and store seams are injected. The
registry tests assert the fail-closed property that matters most — every
harness capability maps to a registered command, and consequential commands
are exactly the ones gated behind approval.
"""

import json
import threading
from pathlib import Path

import pytest

from fx1.flash.store import FlashStore
from fx1.harness import HARNESS_REGISTRY
from fx1.interactive.approvals import ApprovalGate, console_ask_fn
from fx1.interactive.superpower import (
    CAPABILITIES,
    KIND_HARNESS,
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    describe_capabilities,
    plan_goal,
    run_plan,
)
from fx1.webresearch.engine import ResearchReport, SourceDoc

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeHarnessResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeHarness:
    def __init__(self, stdout: str = "DATA_LABEL=SYNTHETIC\nSYNTHETIC\nran ok\n") -> None:
        self.calls: list[tuple[str, list[str] | None]] = []
        self._stdout = stdout

    def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
        self.calls.append((name, extra_args))
        return FakeHarnessResult(0, self._stdout, "")


class FakeResearcher:
    def __init__(self, report: ResearchReport | None = None) -> None:
        self.calls: list[str] = []
        self._report = report or ResearchReport(
            goal="g",
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )

    def run(self, goal, *, budget=None, queries=None, cancel_event=None, on_progress=None):  # noqa: ANN001
        self.calls.append(goal)
        if on_progress is not None:
            from fx1.webresearch.engine import ResearchProgress

            on_progress(ResearchProgress(event="done", detail="fake"))
        return self._report


@pytest.fixture()
def store(tmp_path: Path) -> FlashStore:
    return FlashStore(tmp_path / "flash" / "entries.jsonl")


def _plan(goal: str, ids: list[str]) -> SuperpowerPlan:
    return SuperpowerPlan(
        goal=goal,
        capabilities=[PlannedCapability(i, "test") for i in ids],
        source="test",
    )


# ---------------------------------------------------------------------------
# capability registry
# ---------------------------------------------------------------------------


def test_every_registered_harness_command_is_a_capability() -> None:
    registered = {c.name for c in HARNESS_REGISTRY}
    harness_caps = {k for k, v in CAPABILITIES.items() if v["kind"] == KIND_HARNESS}
    assert registered == harness_caps


def test_consequential_capabilities_are_training_surfaces() -> None:
    consequential = {k for k, v in CAPABILITIES.items() if v.get("consequential")}
    assert consequential == {"train", "optimize", "paper", "live"}


def test_front_door_capabilities_present() -> None:
    for cap in ("web_research", "flash_retrieve", "flash_save", "synthesize"):
        assert cap in CAPABILITIES


def test_describe_capabilities_lists_kinds_and_flags() -> None:
    text = describe_capabilities()
    assert "web_research" in text
    assert "approval-required" in text
    assert "[harness" in text


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------


def test_plan_selects_research_for_benchmark_goal() -> None:
    plan = plan_goal("run the conformal volatility benchmark family")
    ids = plan.ids()
    assert "research" in ids
    assert "paper" not in ids and "train" not in ids
    assert plan.source == "deterministic"
    # mandatory spine
    assert ids[0] == "flash_retrieve"
    assert ids[-1] == "synthesize"
    assert "flash_save" in ids


def test_plan_selects_web_for_current_events_goal() -> None:
    plan = plan_goal("what is the latest news on the Fed rate decision")
    assert "web_research" in plan.ids()


def test_plan_falls_back_to_web_when_no_harness_match() -> None:
    plan = plan_goal("summarize the history of the dutch tulip market")
    ids = plan.ids()
    assert "web_research" in ids
    assert not any(CAPABILITIES[i]["kind"] == KIND_HARNESS for i in ids)


def test_plan_selects_the_live_desk() -> None:
    plan = plan_goal("arm the live desk for live trading")
    assert "live" in plan.ids()


def test_plan_caps_harness_selections() -> None:
    goal = (
        "ingest data, build features and labels, run the research benchmark "
        "family, backtest it, then train and optimize a model"
    )
    plan = plan_goal(goal, max_harness=3)
    harness_ids = [i for i in plan.ids() if CAPABILITIES[i]["kind"] == KIND_HARNESS]
    assert len(harness_ids) <= 3


def test_plan_rejects_negative_harness_limit() -> None:
    with pytest.raises(ValueError, match="max_harness must be nonnegative"):
        plan_goal("ingest, benchmark, backtest, train and optimize", max_harness=-1)


def test_plan_describe_explains_choices() -> None:
    plan = plan_goal("check the northset order book identities")
    text = plan.describe()
    assert "northset" in text
    assert "planner: deterministic" in text


def test_model_refinement_validates_ids() -> None:
    def model_fn(prompt: str) -> str:
        assert "northset" in prompt  # registry listing reached the model
        return (
            '{"capabilities": ["northset", "totally-made-up"], '
            '"rationale": "goal names order book identities"}'
        )

    plan = plan_goal("check the order book identities", model_fn=model_fn)
    assert plan.source == "model+validated"
    ids = plan.ids()
    assert "northset" in ids
    assert "totally-made-up" not in ids  # unknown ids dropped, fail-closed
    assert ids[0] == "flash_retrieve"  # spine re-added
    assert ids[-1] == "synthesize"


@pytest.mark.parametrize("max_harness", [0, 1, 3])
def test_model_refinement_cannot_expand_past_harness_limit(max_harness: int) -> None:
    response = json.dumps({"capabilities": list(CAPABILITIES)})
    plan = plan_goal("check a benchmark", model_fn=lambda _: response, max_harness=max_harness)
    harness_ids = [i for i in plan.ids() if CAPABILITIES[i]["kind"] == KIND_HARNESS]
    expected = [i for i, meta in CAPABILITIES.items() if meta["kind"] == KIND_HARNESS]
    assert plan.source == "model+validated"
    assert harness_ids == expected[:max_harness]
    assert len(plan.ids()) == len(set(plan.ids()))
    assert plan.ids()[0] == "flash_retrieve"
    assert plan.ids()[-1] == "synthesize"


def test_model_refinement_deduplicates_before_applying_limit() -> None:
    response = json.dumps(
        {
            "capabilities": [
                "synthesize",
                "doctor",
                "doctor",
                "flash_retrieve",
                "research",
                "totally-made-up",
                "train",
                "synthesize",
                "flash_retrieve",
            ]
        }
    )
    plan = plan_goal("check a benchmark", model_fn=lambda _: response, max_harness=2)
    assert plan.ids() == ["flash_retrieve", "doctor", "research", "synthesize"]


def test_model_refinement_can_reselect_registered_capabilities() -> None:
    goal = "check the order book identities"
    assert "doctor" not in plan_goal(goal).ids()
    plan = plan_goal(goal, model_fn=lambda _: '{"capabilities": ["doctor"]}')
    assert plan.source == "model+validated"
    assert plan.ids() == ["flash_retrieve", "doctor", "synthesize"]


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"rationale": "no selection supplied"}',
        '{"capabilities": "research"}',
        '{"capabilities": ["research", 1]}',
        '{"capabilities": ["research"], "rationale": {"execute": "train"}}',
    ],
)
def test_malformed_refinement_preserves_bounded_fallback(response: str) -> None:
    goal = "ingest data, benchmark, backtest, then train and optimize"
    expected = plan_goal(goal, max_harness=1)
    actual = plan_goal(goal, model_fn=lambda _: response, max_harness=1)
    assert actual == expected


def test_model_unknown_selection_cannot_become_executable() -> None:
    plan = plan_goal(
        "check a benchmark",
        model_fn=lambda _: '{"capabilities": ["unknown-command", "!rm -rf /"]}',
    )
    assert plan.ids() == ["flash_retrieve", "synthesize"]


def test_model_garbage_falls_back_to_deterministic() -> None:
    plan = plan_goal(
        "run the conformal volatility benchmark",
        model_fn=lambda prompt: "I cannot output JSON, sorry",
    )
    assert plan.source == "deterministic"
    assert "research" in plan.ids()


def test_model_crash_falls_back_to_deterministic() -> None:
    def boom(prompt: str) -> str:
        raise RuntimeError("endpoint down")

    plan = plan_goal("run the conformal benchmark", model_fn=boom)
    assert plan.source == "deterministic"
    assert "research" in plan.ids()


# ---------------------------------------------------------------------------
# coordination
# ---------------------------------------------------------------------------


def test_run_plan_coordinates_memory_web_harness_synthesis(store: FlashStore) -> None:
    store.add(
        text="prior finding: conformal coverage held at 90% on synthetic panels",
        task="conformal volatility benchmark",
        tags=["conformal"],
    )
    harness = FakeHarness()
    researcher = FakeResearcher()
    prompts: list[str] = []

    def model_fn(prompt: str) -> str:
        prompts.append(prompt)
        return "Coverage held at the nominal 90% level (SYNTHETIC panel). Sources listed above."

    plan = _plan(
        "run the conformal volatility benchmark",
        ["flash_retrieve", "research", "web_research", "flash_save", "synthesize"],
    )
    ctx = SuperpowerContext(
        model_fn=model_fn,
        harness=harness,
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
    )
    result = run_plan(plan, ctx)

    assert harness.calls == [("research", None)]
    assert researcher.calls == [plan.goal]
    by_id = {r.id: r for r in result.runs}
    assert by_id["flash_retrieve"].status == "ran"
    assert "1 relevant memory entries" in by_id["flash_retrieve"].detail
    assert by_id["web_research"].status == "ran"
    assert by_id["research"].status == "ran"
    assert "DATA_LABEL=SYNTHETIC" in by_id["research"].detail
    assert by_id["flash_save"].status == "ran"
    assert by_id["synthesize"].status == "ran"
    assert result.honesty_violation is None
    assert "Coverage held" in result.summary
    # synthesis prompt carried goal + evidence + capability explanations
    assert "conformal volatility benchmark" in prompts[0]
    assert "HEURISTIC WEB VERIFICATION" in prompts[0]
    assert "prior finding" in prompts[0]
    # the run was persisted to flash with provenance tags
    saved = [e for e in store.all() if "superpower" in e.tags]
    assert len(saved) == 1
    assert "conformal volatility benchmark" in saved[0].task
    # explain() lists what ran and why
    explanation = result.explain()
    assert "capabilities used and why" in explanation
    assert "[ran] research" in explanation


def test_run_plan_denies_consequential_without_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    gate = ApprovalGate(ask_fn=lambda question: False)
    plan = _plan("run a paper trading shadow run", ["paper", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Paper run skipped: not approved.",
        harness=harness,
        approvals=gate,
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    by_id = {r.id: r for r in result.runs}
    assert by_id["paper"].status == "approval-denied"
    assert harness.calls == []  # never executed
    assert gate.denied() == 1


def test_refined_consequential_selection_still_requires_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    gate = ApprovalGate(ask_fn=lambda question: False)
    plan = plan_goal(
        "check a benchmark",
        model_fn=lambda _: '{"capabilities": ["train", "train", "doctor", "unknown"]}',
        max_harness=2,
    )
    result = run_plan(
        plan,
        SuperpowerContext(harness=harness, approvals=gate, flash_store=store),
    )
    assert harness.calls == [("doctor", None)]
    assert {r.id: r.status for r in result.runs}["train"] == "approval-denied"
    assert gate.denied() == 1


def test_run_plan_does_not_repeat_duplicate_harness_steps(store: FlashStore) -> None:
    harness = FakeHarness()
    plan = _plan("check data", ["doctor", "doctor", "synthesize"])
    run_plan(plan, SuperpowerContext(harness=harness, flash_store=store))
    assert harness.calls == [("doctor", None)]


def test_run_plan_allows_consequential_with_approval(store: FlashStore) -> None:
    harness = FakeHarness()
    questions: list[str] = []

    def approve(question: str) -> bool:
        questions.append(question)
        return True

    plan = _plan("train the ranking family", ["train", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Training dispatched.",
        harness=harness,
        approvals=ApprovalGate(ask_fn=approve),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert harness.calls == [("train", None)]
    assert {r.id: r for r in result.runs}["train"].status == "ran"
    assert any("consequential" in q for q in questions)


def test_run_plan_honesty_gate_withholds_violating_summary(store: FlashStore) -> None:
    plan = _plan("summarize performance", ["synthesize"])
    ctx = SuperpowerContext(
        model_fn=lambda prompt: "Great news: Sharpe 2.1 achieved live.",
        harness=FakeHarness(),
        approvals=ApprovalGate(),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert result.honesty_violation is not None
    assert "forbidden" in result.honesty_violation
    assert "withheld" in result.summary
    assert "Sharpe 2.1" not in result.summary  # violating text never emitted
    assert {r.id: r for r in result.runs}["synthesize"].status == "failed"


def test_run_plan_without_model_gives_offline_summary(store: FlashStore) -> None:
    researcher = FakeResearcher()
    plan = _plan("check inflation", ["flash_retrieve", "web_research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=FakeHarness(),
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    assert "no model endpoint" in result.summary
    assert "web_research" in result.summary
    assert result.honesty_violation is None


def test_run_plan_cancel_skips_remaining_work(store: FlashStore) -> None:
    cancel = threading.Event()
    cancel.set()
    harness = FakeHarness()
    researcher = FakeResearcher()
    plan = _plan("g", ["flash_retrieve", "web_research", "research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=harness,
        approvals=ApprovalGate(),
        researcher=researcher,
        flash_store=store,
        cancel_event=cancel,
    )
    result = run_plan(plan, ctx)
    assert researcher.calls == []  # web research never started
    assert harness.calls == []  # harness never started
    statuses = {r.id: r.status for r in result.runs}
    assert statuses["research"] == "skipped"


def test_run_plan_harness_failure_recorded_not_fatal(store: FlashStore) -> None:
    class FailingHarness:
        def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
            return FakeHarnessResult(2, "", "boom")

    plan = _plan("verify receipts", ["verify-research", "synthesize"])
    ctx = SuperpowerContext(
        model_fn=None,
        harness=FailingHarness(),
        approvals=ApprovalGate(),
        flash_store=store,
    )
    result = run_plan(plan, ctx)
    by_id = {r.id: r for r in result.runs}
    assert by_id["verify-research"].status == "failed"
    assert by_id["synthesize"].status == "ran"  # synthesis still reports honestly


def test_run_plan_unknown_capability_id_skipped(store: FlashStore) -> None:
    plan = _plan("g", ["definitely-not-registered", "synthesize"])
    ctx = SuperpowerContext(model_fn=None, harness=FakeHarness(), flash_store=store)
    result = run_plan(plan, ctx)
    assert all(r.id != "definitely-not-registered" for r in result.runs)


# ---------------------------------------------------------------------------
# approvals
# ---------------------------------------------------------------------------


def test_approval_gate_defaults_to_deny() -> None:
    gate = ApprovalGate()
    assert gate.ask("do the thing?") is False
    assert gate.denied() == 1
    assert gate.records[0].question == "do the thing?"


def test_approval_gate_records_decisions() -> None:
    gate = ApprovalGate(ask_fn=lambda q: True)
    assert gate.ask("a?", kind="consequential") is True
    assert gate.granted() == 1
    assert gate.records[0].kind == "consequential"
    assert gate.records[0].timestamp


def test_console_ask_fn_yes_no_eof() -> None:
    assert console_ask_fn(lambda prompt: "y")("q?") is True
    assert console_ask_fn(lambda prompt: "yes")("q?") is True
    assert console_ask_fn(lambda prompt: "n")("q?") is False
    assert console_ask_fn(lambda prompt: "")("q?") is False

    def eof(prompt: str) -> str:
        raise EOFError

    assert console_ask_fn(eof)("q?") is False
