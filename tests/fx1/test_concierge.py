"""Concierge console: slash dispatch, background jobs, chat tool loop.

Offline: backend, harness, researcher, store, and approvals are injected.
The chat tool loop is driven by a scripted fake backend so tool execution,
approval gating, and message bookkeeping are all observable.
"""

import io
import threading
from pathlib import Path
from typing import Any

import pytest

from fx1.flash.store import FlashStore
from fx1.interactive import concierge as concierge_module
from fx1.interactive import profiles
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import BackgroundJob, Concierge, run_console
from fx1.serve.backends import ToolCompletion
from fx1.webresearch.engine import ResearchReport, SourceDoc

# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


class FakeBackend:
    """Scripted complete_with_tools responses; records every call."""

    def __init__(self, script: list[ToolCompletion]) -> None:
        self.script = list(script)
        self.tool_calls_seen: list[list[dict[str, Any]]] = []
        self.complete_calls: list[list[dict[str, Any]]] = []

    def complete_with_tools(
        self,
        messages: list[dict[str, Any]],
        *,
        sampling: Any = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> ToolCompletion:
        self.tool_calls_seen.append(list(messages))
        return self.script.pop(0)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.complete_calls.append(list(messages))
        return "plain model answer"


class FakeHarnessResult:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeHarness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str] | None]] = []

    def run(self, name: str, extra_args: list[str] | None = None) -> FakeHarnessResult:
        self.calls.append((name, extra_args))
        return FakeHarnessResult(0, f"DATA_LABEL=SYNTHETIC\n{name} ran\n", "")


class FakeResearcher:
    """run() waits for cancel (bounded), records redirect() calls."""

    def __init__(self, *, block: bool = False, report: ResearchReport | None = None) -> None:
        self.redirects: list[str] = []
        self.goals: list[str] = []
        self._block = block
        self._report = report

    def redirect(self, new_goal: str) -> None:
        self.redirects.append(new_goal)

    def run(
        self,
        goal: str,
        *,
        budget: Any = None,
        queries: Any = None,
        cancel_event: threading.Event | None = None,
        on_progress: Any = None,
    ) -> ResearchReport:
        self.goals.append(goal)
        if on_progress is not None:
            from fx1.webresearch.engine import ResearchProgress

            on_progress(ResearchProgress(event="started", detail=goal))
        if self._block and cancel_event is not None:
            cancel_event.wait(3.0)
        if self._report is not None:
            return self._report
        return ResearchReport(
            goal=goal,
            cancelled=bool(cancel_event is not None and cancel_event.is_set()),
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )


@pytest.fixture()
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "fx1-config"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    monkeypatch.setenv("NO_COLOR", "1")
    return tmp_path


def make_concierge(
    tmp_path: Path,
    *,
    backend: FakeBackend | None = None,
    researcher: Any = None,
    harness: Any = None,
    approvals: ApprovalGate | None = None,
) -> tuple[Concierge, list[str]]:
    out: list[str] = []
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    concierge = Concierge(
        store=store,
        researcher=researcher if researcher is not None else FakeResearcher(),
        harness=harness if harness is not None else FakeHarness(),
        approvals=approvals if approvals is not None else ApprovalGate(),
        backend_factory=(lambda **kwargs: backend) if backend is not None else None,
        output_fn=out.append,
        animated=False,
    )
    return concierge, out


def joined(out: list[str]) -> str:
    return "\n".join(str(line) for line in out)


# ---------------------------------------------------------------------------
# basic dispatch
# ---------------------------------------------------------------------------


def test_help_and_exit(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    assert concierge.handle_line("/help") is False
    assert "/superpower" in joined(out)
    assert "/trade" in joined(out)
    assert concierge.handle_line("/exit") is True
    assert concierge.handle_line("quit") is True


def test_trade_readiness_refuses_live_orders(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    assert concierge.handle_line("/trade") is False
    text = joined(out)
    assert "Live orders: disabled" in text
    assert "would_promote_live=false" in text
    assert concierge.handle_line("/trade help") is False
    assert "dipcatcher trade session" in joined(out)
    assert concierge.handle_line("/trade live") is False


def test_unknown_command_reports_error(env: Path, tmp_path: Path) -> None:
    concierge, _ = make_concierge(tmp_path)
    assert concierge.handle_line("/definitely-not-a-command") is False  # survives


def test_model_command(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/model")
    assert "active model: fx1" in joined(out)


def test_chat_offline_is_honest(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("hello there")
    text = joined(out)
    assert "no model endpoint configured" in text
    assert "/keys set" in text


@pytest.mark.parametrize(
    "command, model", [("/keys set", "fx1"), ("/keys set fx1-lite", "fx1-lite")]
)
def test_keys_set_prompts_for_masked_key_and_endpoint(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    model: str,
) -> None:
    key_prompts: list[str] = []
    url_prompts: list[str] = []

    def password(prompt: str) -> str:
        key_prompts.append(prompt)
        return "test-private-api-key"

    def endpoint(prompt: str) -> str:
        url_prompts.append(prompt)
        return "https://models.example.test/v1/chat/completions"

    monkeypatch.setattr("getpass.getpass", password)
    concierge, out = make_concierge(tmp_path)
    concierge._input = endpoint  # noqa: SLF001

    assert concierge.handle_line(command) is False

    assert profiles.resolve_endpoint(model) == (
        "test-private-api-key",
        "https://models.example.test/v1/chat/completions",
    )
    assert key_prompts == [f"› {model} API key: "]
    assert len(url_prompts) == 1 and f"{model} endpoint URL" in url_prompts[0]
    assert "models.example.test" in joined(out)
    assert "test-private-api-key" not in joined(out)


def test_keys_set_blank_url_preserves_existing_endpoint(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_url = "https://configured.example.test/v1/chat/completions"
    profiles.set_endpoint("fx1", "old-private-key", base_url)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "new-private-key")
    concierge, _ = make_concierge(tmp_path)
    concierge._input = lambda prompt: ""  # noqa: SLF001

    concierge.handle_line("/keys set")

    assert profiles.resolve_endpoint("fx1") == ("new-private-key", base_url)


def test_keys_set_invalid_url_keeps_existing_profile(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = profiles.set_endpoint("fx1", "old-private-key", "https://configured.example.test")
    original = path.read_bytes()
    monkeypatch.setattr("getpass.getpass", lambda prompt: "new-private-key")
    concierge, out = make_concierge(tmp_path)
    concierge._input = lambda prompt: "file:///invalid-endpoint"  # noqa: SLF001

    assert concierge.handle_line("/keys set") is False

    assert path.read_bytes() == original
    assert "invalid base URL" in capsys.readouterr().err
    assert "new-private-key" not in joined(out)


@pytest.mark.parametrize(
    "stage, error", [("key", EOFError), ("url", EOFError), ("url", KeyboardInterrupt)]
)
def test_keys_set_cancel_keeps_existing_profile(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error: type[BaseException],
) -> None:
    path = profiles.set_endpoint("fx1", "old-private-key", "https://configured.example.test")
    original = path.read_bytes()

    def cancelled(prompt: str) -> str:
        raise error

    monkeypatch.setattr(
        "getpass.getpass", cancelled if stage == "key" else lambda prompt: "new-private-key"
    )
    concierge, out = make_concierge(tmp_path)
    concierge._input = cancelled  # noqa: SLF001

    assert concierge.handle_line("/keys set") is False

    assert path.read_bytes() == original
    assert "cancelled; no changes saved" in joined(out)
    assert "new-private-key" not in joined(out)


# ---------------------------------------------------------------------------
# chat tool loop
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_chat_tool_loop_executes_flash_retrieve(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    store_path = tmp_path / "flash" / "entries.jsonl"
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "flash_retrieve", {"query": "inflation"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Answered from memory.", tool_calls=None, finish_reason="stop"),
        ]
    )
    out: list[str] = []
    store = FlashStore(store_path)
    entry = store.add(text="US inflation was 3.4% in August 2026", task="inflation", tags=["macro"])
    concierge = Concierge(
        store=store,
        researcher=FakeResearcher(),
        harness=FakeHarness(),
        backend_factory=lambda **kwargs: backend,
        output_fn=out.append,
        animated=False,
    )
    concierge.handle_line("what do we know about inflation?")
    assert "Answered from memory." in joined(out)
    # the tool result reached the model as a role=tool message
    second_call_messages = backend.tool_calls_seen[1]
    tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
    assert tool_msgs and tool_msgs[0]["tool_call_id"] == "c1"
    assert "US inflation was 3.4%" in tool_msgs[0]["content"]
    # retrieval marked the entry used
    assert store.get(entry.id).uses == 1


def test_chat_harness_tool_denied_without_approval(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "train"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(
                content="Training requires approval.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(
        tmp_path, backend=backend, harness=harness, approvals=ApprovalGate()
    )
    concierge.handle_line("train the ranking family")
    assert harness.calls == []  # consequential command never ran
    assert "not approved" in joined(out)


def test_chat_harness_tool_runs_registered_command(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Doctor ran.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("check lab health")
    assert harness.calls == [("doctor", [])]
    assert "DATA_LABEL=SYNTHETIC" in joined(out)


def test_chat_harness_tool_refuses_unregistered(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    harness = FakeHarness()
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(_tool_call("c1", "harness_run", {"name": "rm -rf /"}),),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Refused.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, _ = make_concierge(tmp_path, backend=backend, harness=harness)
    concierge.handle_line("do something dangerous")
    assert harness.calls == []
    tool_msg = [m for m in backend.tool_calls_seen[1] if m.get("role") == "tool"][0]
    assert "not a registered harness command" in tool_msg["content"]


def test_chat_flash_save_tool(env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    backend = FakeBackend(
        [
            ToolCompletion(
                content=None,
                tool_calls=(
                    _tool_call(
                        "c1",
                        "flash_save",
                        {"text": "Fed held rates in September 2026", "task": "rates"},
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content="Saved.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge = Concierge(
        store=store,
        researcher=FakeResearcher(),
        harness=FakeHarness(),
        backend_factory=lambda **kwargs: backend,
        output_fn=lambda *a, **k: None,
        animated=False,
    )
    concierge.handle_line("remember that the Fed held rates")
    entries = store.all()
    assert len(entries) == 1
    assert entries[0].text == "Fed held rates in September 2026"
    assert "model-saved" in entries[0].tags


def test_chat_backend_fault_rolls_back_and_survives(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")

    class DeadBackend:
        def complete_with_tools(self, messages: list[dict], **kwargs: Any) -> ToolCompletion:
            raise RuntimeError("hosted_k3 endpoint unreachable")

    concierge, _ = make_concierge(tmp_path)
    concierge._backend_factory = lambda **kwargs: DeadBackend()  # noqa: SLF001
    concierge.handle_line("hello")  # must not raise
    assert concierge._messages == []  # failed turn rolled back  # noqa: SLF001


@pytest.mark.parametrize(
    "answer",
    [
        "The strategy achieved Sharpe: 2.35 over the panel.",
        "The model guarantees nothing, but achieved live trading profits.",
        "On synthetic data the model achieves 0.99 recovery accuracy.",
    ],
)
def test_chat_honesty_refusal_preserves_prior_turn_and_can_continue(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    answer: str,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend(
        [
            ToolCompletion(content="Ready to help.", tool_calls=None, finish_reason="stop"),
            ToolCompletion(content=answer, tool_calls=None, finish_reason="stop"),
            ToolCompletion(content="Use proper scores.", tool_calls=None, finish_reason="stop"),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("hello")
    prior = list(concierge._messages)  # noqa: SLF001

    assert concierge.handle_line("summarize the result") is False
    assert concierge._messages == prior  # noqa: SLF001
    assert "withheld by the fx-1 honesty gate" in joined(out)
    assert answer not in joined(out)
    captured = capsys.readouterr()
    assert answer not in captured.out + captured.err

    assert concierge.handle_line("which scores should we use?") is False
    assert backend.tool_calls_seen[-1] == [
        *prior,
        {"role": "user", "content": "which scores should we use?"},
    ]
    assert out[-1] == "Use proper scores."


@pytest.mark.parametrize(
    "answer",
    [
        "The held-out CRPS is 0.12 and pinball loss is 0.08.",
        "On SYNTHETIC data the model achieves 0.99 recovery accuracy.",
    ],
)
def test_chat_honesty_accepts_proper_scores_and_labeled_synthetic_results(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    backend = FakeBackend([ToolCompletion(content=answer, tool_calls=None, finish_reason="stop")])
    concierge, out = make_concierge(tmp_path, backend=backend)

    concierge.handle_line("summarize the result")

    assert out == [answer]
    assert concierge._messages[-1] == {"role": "assistant", "content": answer}  # noqa: SLF001


@pytest.mark.parametrize("unsafe_tool_preamble", [False, True])
def test_chat_honesty_gate_covers_tool_turns(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_tool_preamble: bool,
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    forbidden = "The strategy achieved Sharpe: 2.35 over the panel."
    script = [
        ToolCompletion(
            content=forbidden if unsafe_tool_preamble else None,
            tool_calls=(_tool_call("c1", "harness_run", {"name": "doctor"}),),
            finish_reason="tool_calls",
        ),
    ]
    if not unsafe_tool_preamble:
        script.append(ToolCompletion(content=forbidden, tool_calls=None, finish_reason="stop"))
    script.append(ToolCompletion(content="Ready again.", tool_calls=None, finish_reason="stop"))
    backend = FakeBackend(script)
    harness = FakeHarness()
    concierge, out = make_concierge(tmp_path, backend=backend, harness=harness)

    concierge.handle_line("check lab health")

    assert harness.calls == ([] if unsafe_tool_preamble else [("doctor", [])])
    assert forbidden not in joined(out)
    assert "withheld by the fx-1 honesty gate" in joined(out)
    retained = list(concierge._messages)  # noqa: SLF001
    if unsafe_tool_preamble:
        assert retained == []
    else:
        assert [message["role"] for message in retained] == [
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        assert "DATA_LABEL=SYNTHETIC" in retained[2]["content"]
        assert "withheld by the fx-1 honesty gate" in retained[3]["content"]
    assert forbidden not in str(retained)
    concierge.handle_line("try again")
    assert backend.tool_calls_seen[-1] == [*retained, {"role": "user", "content": "try again"}]
    assert out[-1] == "Ready again."


def test_chat_honesty_refusal_preserves_completed_flash_save_for_followup(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    forbidden = "The strategy achieved Sharpe: 2.35 over the panel."
    finding = "Use proper scores to compare the forecasts."
    backend = FakeBackend(
        [
            ToolCompletion(content="Ready to help.", tool_calls=None, finish_reason="stop"),
            ToolCompletion(
                content="I will save the research note.",
                tool_calls=(
                    _tool_call(
                        "save1", "flash_save", {"text": finding, "task": "forecast scoring"}
                    ),
                ),
                finish_reason="tool_calls",
            ),
            ToolCompletion(content=forbidden, tool_calls=None, finish_reason="stop"),
            ToolCompletion(
                content="The note is already saved.", tool_calls=None, finish_reason="stop"
            ),
        ]
    )
    concierge, out = make_concierge(tmp_path, backend=backend)
    concierge.handle_line("hello")
    prior = list(concierge._messages)  # noqa: SLF001

    concierge.handle_line("remember the scoring guidance")

    entries = concierge._store.all()  # noqa: SLF001
    assert len(entries) == 1 and entries[0].text == finding
    retained = list(concierge._messages)  # noqa: SLF001
    assert retained[: len(prior)] == prior
    assert retained[-3]["tool_calls"][0]["id"] == "save1"
    assert retained[-2] == {
        "role": "tool",
        "tool_call_id": "save1",
        "content": f"saved flash entry {entries[0].id}",
    }
    assert "withheld by the fx-1 honesty gate" in retained[-1]["content"]
    assert forbidden not in str(retained) and forbidden not in joined(out)

    concierge.handle_line("was the note saved?")

    assert backend.tool_calls_seen[-1] == [
        *retained,
        {"role": "user", "content": "was the note saved?"},
    ]
    assert len(concierge._store.all()) == 1  # noqa: SLF001
    assert out[-1] == "The note is already saved."


# ---------------------------------------------------------------------------
# /research background lifecycle
# ---------------------------------------------------------------------------


def test_research_background_completes_and_announces(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(
        report=ResearchReport(
            goal="fed rates",
            started_at="t0",
            finished_at="t1",
            sources=[SourceDoc(url="https://a.example/x", title="A", status=200, text="t")],
        )
    )
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research fed rates --budget-s 30 --max-sources 3")
    assert "running in the background" in joined(out)
    concierge.join_job(timeout_s=5)
    text = joined(out)
    assert "# Web research: fed rates" in text
    assert "HEURISTIC WEB VERIFICATION" in text
    assert researcher.goals == ["fed rates"]


def test_research_sync_mode(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path, researcher=FakeResearcher())
    concierge.handle_line("/research inflation now --sync")
    text = joined(out)
    assert "# Web research: inflation now" in text


def test_research_queries_option(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher()
    concierge, _ = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research goal --queries one|two --sync")
    assert researcher.goals == ["goal"]


def test_cancel_stops_background_research(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research long task")
    concierge.handle_line("/status")
    assert "running" in joined(out)
    concierge.handle_line("/cancel")
    assert "cancel requested" in joined(out)
    concierge.join_job(timeout_s=5)
    assert "(cancelled)" in joined(out)


def test_second_background_job_cannot_orphan_first(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research first goal")
    first = concierge._job  # noqa: SLF001
    assert first is not None

    concierge.handle_line("/research second goal")

    assert concierge._job is first  # noqa: SLF001
    assert "already running" in joined(out)
    concierge.handle_line("/cancel")
    concierge.join_job(timeout_s=5)


def test_shutdown_cancels_and_joins_owned_job(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, _ = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research long task")
    job = concierge._job  # noqa: SLF001
    assert job is not None and job.running

    assert concierge.shutdown(timeout_s=5)
    assert job.cancel_event.is_set()
    assert job.done_event.is_set()
    assert job.thread is not None and not job.thread.is_alive()


def test_redirect_retargets_running_research(env: Path, tmp_path: Path) -> None:
    researcher = FakeResearcher(block=True)
    concierge, out = make_concierge(tmp_path, researcher=researcher)
    concierge.handle_line("/research old goal")
    concierge.handle_line("/redirect new goal")
    assert researcher.redirects == ["new goal"]
    assert "redirect requested" in joined(out)
    concierge.handle_line("/cancel")
    concierge.join_job(timeout_s=5)


def test_status_without_job(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/status")
    assert "no background job" in joined(out)


def test_background_job_error_announced(env: Path, tmp_path: Path) -> None:
    class ExplodingResearcher(FakeResearcher):
        def run(self, goal: str, **kwargs: Any) -> ResearchReport:
            raise RuntimeError("search backend exploded")

    concierge, out = make_concierge(tmp_path, researcher=ExplodingResearcher())
    concierge.handle_line("/research anything --sync")
    assert "failed" in joined(out)
    assert "exploded" in joined(out)


# ---------------------------------------------------------------------------
# /superpower through the console
# ---------------------------------------------------------------------------


def test_superpower_help_lists_capabilities(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/superpower help")
    text = joined(out)
    assert "available /superpower capabilities" in text
    assert "web_research" in text


def test_superpower_sync_offline_pipeline(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    researcher = FakeResearcher()
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    store.add(
        text="prior conformal benchmark findings on record",
        task="conformal volatility benchmark",
        tags=["conformal"],
    )
    concierge, out = make_concierge(tmp_path, researcher=researcher, harness=harness)
    concierge._store = store  # noqa: SLF001
    concierge.handle_line("/superpower run the conformal volatility benchmark --sync")
    text = joined(out)
    assert "plan for: run the conformal volatility benchmark" in text
    assert "planner: deterministic" in text
    assert "[ran] research" in text  # harness ran and was explained
    assert harness.calls == [("research", None)]
    # planner coordinated: a pure lab-benchmark goal does not trigger web research
    assert researcher.goals == []
    # offline synthesis is honest about the missing model
    assert "no model endpoint" in text
    # run persisted to flash
    assert any("superpower" in e.tags for e in store.all())


def test_superpower_background_job(env: Path, tmp_path: Path) -> None:
    concierge, out = make_concierge(tmp_path)
    concierge.handle_line("/superpower summarize tulip history")
    assert "running in the background" in joined(out)
    concierge.join_job(timeout_s=10)
    text = joined(out)
    assert "capabilities used and why" in text
    assert "web_research" in text or "deep web research" in text


def test_superpower_denies_consequential(env: Path, tmp_path: Path) -> None:
    harness = FakeHarness()
    concierge, out = make_concierge(
        tmp_path, harness=harness, approvals=ApprovalGate(ask_fn=lambda q: False)
    )
    concierge.handle_line("/superpower train and optimize a model --sync")
    assert harness.calls == []
    assert "approval-denied" in joined(out)


# ---------------------------------------------------------------------------
# /flash through the console
# ---------------------------------------------------------------------------


def test_flash_add_list_search_correct_remove(env: Path, tmp_path: Path) -> None:
    store = FlashStore(tmp_path / "flash" / "entries.jsonl")
    concierge, out = make_concierge(tmp_path)
    concierge._store = store  # noqa: SLF001

    concierge.handle_line(
        '/flash add "US inflation was 3.4% in August 2026" --task inflation '
        "--tags macro,us --source https://x.example/rates --uncertainty low"
    )
    text = joined(out)
    assert "saved flash entry" in text
    entry = store.all()[0]
    assert entry.task == "inflation"
    assert entry.tags == ["macro", "us"]
    assert entry.sources[0].url == "https://x.example/rates"
    assert entry.uncertainty == "low"

    out.clear()
    concierge.handle_line("/flash list")
    assert "US inflation" in joined(out)

    out.clear()
    concierge.handle_line("/flash search inflation rate")
    assert f"{entry.id[:8]}" in joined(out)
    assert "score=" in joined(out)

    out.clear()
    concierge.handle_line(f"/flash correct {entry.id} --uncertainty high")
    assert "corrected" in joined(out)
    assert store.get(entry.id).uncertainty == "high"

    out.clear()
    concierge.handle_line(f"/flash show {entry.id}")
    assert '"uncertainty": "high"' in joined(out)

    out.clear()
    concierge.handle_line(f"/flash remove {entry.id}")
    assert "removed" in joined(out)
    assert store.get(entry.id) is None


def test_flash_bad_uncertainty_reports_error(env: Path, tmp_path: Path) -> None:
    concierge, _ = make_concierge(tmp_path)
    concierge.handle_line("/flash add finding --uncertainty catastrophic")
    entries = concierge._store.all()  # noqa: SLF001
    assert entries == []  # rejected, nothing stored


# ---------------------------------------------------------------------------
# console driver
# ---------------------------------------------------------------------------


def test_run_console_processes_scripted_lines(env: Path, tmp_path: Path) -> None:
    lines = iter(["/help", "/model", "exit"])
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    concierge, cout = make_concierge(tmp_path)
    run_console(
        input_fn=input_fn,
        output_fn=out.append,
        concierge_factory=lambda: concierge,
    )
    banner_text = joined(out)
    assert "dipcatcher concierge" in banner_text
    session_text = joined(cout)
    assert "/superpower" in session_text  # /help rendered
    assert "active model: fx1" in session_text  # /model rendered


def test_run_console_eof_exits_cleanly(env: Path, tmp_path: Path) -> None:
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        raise EOFError

    concierge, _ = make_concierge(tmp_path)
    run_console(input_fn=input_fn, output_fn=out.append, concierge_factory=lambda: concierge)
    assert "dipcatcher concierge" in joined(out)


def test_run_console_prints_one_prompt_per_line_across_idle_polls(
    env: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class TerminalInput(io.StringIO):
        def isatty(self) -> bool:
            return True

    stdin = TerminalInput("/model\n/exit\n")
    readiness = iter([False, False, True, False, True])
    monkeypatch.setattr(concierge_module.sys, "stdin", stdin)
    monkeypatch.setattr(concierge_module.sys, "platform", "linux")
    monkeypatch.setattr(
        concierge_module.select,
        "select",
        lambda *args: ([stdin] if next(readiness) else [], [], []),
    )
    concierge, out = make_concierge(tmp_path)

    run_console(concierge_factory=lambda: concierge, output_fn=lambda text: None)

    assert capsys.readouterr().out == "dip › dip › "
    assert "active model: fx1" in joined(out)


def test_run_console_windows_uses_blocking_input_without_select(
    env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(concierge_module.sys, "platform", "win32")
    monkeypatch.setattr(concierge_module.sys.stdin, "isatty", lambda: True)

    def unexpected_select(*args: Any) -> Any:
        pytest.fail("Windows console stdin cannot be polled with select")

    monkeypatch.setattr(concierge_module.select, "select", unexpected_select)
    lines = iter(["/model", "/exit"])
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        return next(lines)

    concierge, out = make_concierge(tmp_path)
    startup: list[str] = []

    run_console(concierge_factory=lambda: concierge, input_fn=input_fn, output_fn=startup.append)

    assert prompts == ["dip › ", "dip › "]
    assert "active model: fx1" in joined(out)
    assert "background progress updates after you submit a line" in joined(startup)


@pytest.mark.parametrize("configured", [False, True])
def test_run_console_banner_shows_actual_endpoint_and_respects_no_color(
    env: Path, tmp_path: Path, configured: bool
) -> None:
    if configured:
        profiles.set_endpoint("fx1-lite", "test-private-key", "https://lite.example.test")
    concierge, _ = make_concierge(tmp_path)
    concierge.model = "fx1-lite"
    out: list[str] = []

    def input_fn(prompt: str) -> str:
        raise EOFError

    run_console(input_fn=input_fn, output_fn=out.append, concierge_factory=lambda: concierge)

    text = joined(out)
    assert "model fx1-lite" in text
    assert ("lite.example.test" if configured else "unconfigured") in text
    assert ("/keys set fx1-lite" in text) is not configured
    assert "api.moonshot.ai" not in text
    assert "test-private-key" not in text
    assert "\x1b" not in text


def test_background_job_drain_and_cancel() -> None:
    release = threading.Event()

    def run_fn(cancel: threading.Event, on_event: Any) -> str:
        on_event("working", "step 1")
        cancel.wait(3.0)
        release.set()
        return "done-after-cancel"

    job = BackgroundJob(label="test", run_fn=run_fn).start()
    job.cancel()
    assert job.done_event.wait(5.0)
    assert release.is_set()
    events = job.drain()
    assert ("working", "step 1") in events
    assert job.result == "done-after-cancel"
    assert job.drain() == []  # drained once
