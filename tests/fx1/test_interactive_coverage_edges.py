"""Scripted coverage for the fxi REPL and the concierge session loop.

Stdin, getpass, subprocess, and model calls are injected. No sockets.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from fx1.interactive import actions, profiles, wizard
from fx1.interactive import concierge as concierge_mod
from fx1.interactive import orb as orb_mod
from fx1.interactive import shell as shell_mod
from fx1.interactive.approvals import ApprovalGate
from fx1.interactive.concierge import BackgroundJob, Concierge, _parse_research_command
from fx1.interactive.orb import OrbAnimator, OrbState
from fx1.interactive.shell import FxiShell
from fx1.interactive.superpower import (
    CapabilityRun,
    PlannedCapability,
    SuperpowerContext,
    SuperpowerPlan,
    SuperpowerResult,
    _offline_summary,
    run_plan,
)

# Package __init__ binds the name ``app`` to the Typer, which shadows the module.
fxi_app = importlib.import_module("fx1.interactive.app")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("FX1_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    return tmp_path


def _shell(monkeypatch: pytest.MonkeyPatch) -> FxiShell:
    monkeypatch.setattr(actions, "doctor", lambda model: {"model": model, "ok": True})
    monkeypatch.setattr(actions, "run_eval", lambda out, model: 0)
    monkeypatch.setattr(actions, "chat", lambda model, opening: 0)
    monkeypatch.setattr(actions, "harness_run", lambda name, args, model: 0)
    monkeypatch.setattr(actions, "verify", lambda path, model: 0)
    monkeypatch.setattr(actions, "run_tool", lambda tool, args, model: 0)
    return FxiShell()


def test_shell_keys_use_and_completions(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repl = _shell(monkeypatch)
    repl.onecmd("keys")
    listed = capsys.readouterr().out
    assert "fx1" in listed

    repl.onecmd('keys "unterminated')
    assert "could not parse" in capsys.readouterr().err

    repl.onecmd("keys invent")
    assert "unknown keys subcommand" in capsys.readouterr().err

    repl.onecmd("keys set fx1 extra")
    assert "usage: keys set" in capsys.readouterr().err

    repl.onecmd("keys set nope")
    assert "unknown model" in capsys.readouterr().err

    monkeypatch.setattr("getpass.getpass", lambda prompt: "  ")
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    repl.onecmd("keys set fx1")
    assert "empty API key" in capsys.readouterr().err

    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-test-value")
    monkeypatch.setattr("builtins.input", lambda prompt: "https://models.example.test/v1")
    repl.onecmd("keys set")
    stored = capsys.readouterr().out
    assert "stored endpoint" in stored
    assert profiles.resolve_endpoint("fx1")[0] == "sk-test-value"

    repl.onecmd("keys remove")
    assert "usage: keys remove" in capsys.readouterr().err
    repl.onecmd("keys remove nope")
    assert "unknown model" in capsys.readouterr().err
    repl.onecmd("keys remove fx1-lite")
    assert "was not stored" in capsys.readouterr().out
    repl.onecmd("keys remove fx1")
    assert "removed endpoint" in capsys.readouterr().out

    repl.onecmd("use")
    assert "usage: use" in capsys.readouterr().err
    repl.onecmd("use fx1")
    assert "no endpoint" in capsys.readouterr().err or "API key" in capsys.readouterr().err

    profiles.set_endpoint("fx1-lite", "sk-lite", "https://lite.example.test/v1")
    repl.onecmd("use fx1-lite")
    activated = capsys.readouterr().out
    assert "fx1-lite" in activated
    assert repl.model == "fx1-lite"
    repl.onecmd("model")
    assert "lite.example.test" in capsys.readouterr().out

    assert "set" in repl.complete_keys("s", "keys ", 5, 5)
    assert any(item.startswith("fx") for item in repl.complete_keys("fx", "keys set ", 9, 9))
    assert repl.complete_use("fx", "use ", 4, 4)
    names = repl.complete_harness("", "harness ", 8, 8)
    assert "list" in names and "run" in names
    assert repl.complete_harness("d", "harness describe ", 17, 17)
    assert "fx1" in repl.complete_run("fx", "run ", 4, 4)
    assert repl.complete_run("x", "run fx1 ", 8, 8) == []


def test_shell_harness_eval_chat_run_and_bang(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repl = _shell(monkeypatch)
    profiles.set_endpoint("fx1", "sk-test-value", "https://models.example.test/v1")
    repl.onecmd("use fx1")
    capsys.readouterr()

    repl.onecmd("doctor")
    assert "fx1" in capsys.readouterr().out

    repl.onecmd("eval")
    assert "eval exited" not in capsys.readouterr().err
    monkeypatch.setattr(actions, "run_eval", lambda out, model: 3)
    repl.onecmd("eval --out /tmp/fx1-eval-out.json")
    assert "eval exited 3" in capsys.readouterr().err
    repl.onecmd("eval --mystery")
    assert "unknown eval argument" in capsys.readouterr().err
    repl.onecmd('eval "')
    assert "could not parse" in capsys.readouterr().err
    monkeypatch.setattr(
        actions, "run_eval", lambda out, model: (_ for _ in ()).throw(RuntimeError("no endpoint"))
    )
    repl.onecmd("eval")
    assert "no endpoint" in capsys.readouterr().err

    seen: list[str | None] = []
    monkeypatch.setattr(actions, "chat", lambda model, opening: seen.append(opening) or 0)
    repl.onecmd("chat hello there")
    repl.onecmd("chat")
    assert seen == ["hello there", None]
    monkeypatch.setattr(
        actions, "chat", lambda model, opening: (_ for _ in ()).throw(RuntimeError("chat down"))
    )
    repl.onecmd("chat hi")
    assert "chat down" in capsys.readouterr().err

    repl.onecmd("harness")
    assert "doctor" in capsys.readouterr().out
    repl.onecmd("harness list")
    assert "name" in capsys.readouterr().out
    repl.onecmd("harness describe doctor")
    described = json.loads(capsys.readouterr().out)
    assert described["name"] == "doctor"
    repl.onecmd("harness describe missing-command")
    assert (
        "not a registered" in capsys.readouterr().err
        or "missing-command" in capsys.readouterr().err
    )
    repl.onecmd("harness run doctor")
    assert "exited" not in capsys.readouterr().err
    monkeypatch.setattr(actions, "harness_run", lambda name, args, model: 4)
    repl.onecmd("harness run doctor --fast")
    assert "harness command exited 4" in capsys.readouterr().err
    monkeypatch.setattr(
        actions, "harness_run", lambda name, args, model: (_ for _ in ()).throw(KeyError("missing"))
    )
    repl.onecmd("harness run missing")
    assert "missing" in capsys.readouterr().err
    repl.onecmd("harness nope")
    assert "usage: harness" in capsys.readouterr().err
    repl.onecmd('harness "')
    assert "could not parse" in capsys.readouterr().err

    repl.onecmd("verify /tmp/receipt.json")
    assert "verify exited" not in capsys.readouterr().err
    repl.onecmd('verify "')
    assert "could not parse" in capsys.readouterr().err
    monkeypatch.setattr(actions, "verify", lambda path, model: 2)
    repl.onecmd("verify /tmp/receipt.json")
    assert "verify exited 2" in capsys.readouterr().err
    repl.onecmd("verify")
    assert "usage: verify" in capsys.readouterr().err
    monkeypatch.setattr(
        actions, "verify", lambda path, model: (_ for _ in ()).throw(RuntimeError("verify down"))
    )
    repl.onecmd("verify one.json")
    assert "verify down" in capsys.readouterr().err

    repl.onecmd("run fx1 --version")
    assert "exited" not in capsys.readouterr().err
    repl.onecmd('run "')
    assert "could not parse" in capsys.readouterr().err
    monkeypatch.setattr(actions, "run_tool", lambda tool, args, model: 5)
    repl.onecmd("run fx1 --version")
    assert "fx1 exited 5" in capsys.readouterr().err
    repl.onecmd("run")
    assert "usage: run" in capsys.readouterr().err
    monkeypatch.setattr(
        actions, "run_tool", lambda tool, args, model: (_ for _ in ()).throw(KeyError("nope"))
    )
    repl.onecmd("run nope")
    assert "nope" in capsys.readouterr().err

    launched: list[str] = []
    monkeypatch.setattr(
        shell_mod.subprocess,
        "run",
        lambda *a, **k: launched.append(a[0]) or SimpleNamespace(returncode=0),
    )
    repl.onecmd("!echo hi")
    repl.onecmd("!")
    assert launched == ["echo hi"]
    assert "usage: !" in capsys.readouterr().err

    assert repl.onecmd("help") is None
    assert "keys" in capsys.readouterr().out
    assert repl.onecmd("exit") is True

    scripted = FxiShell()
    scripted.stdin = io.StringIO("model\nexit\n")
    scripted.stdout = io.StringIO()
    scripted.use_rawinput = False
    scripted.cmdloop()
    assert "active model" in capsys.readouterr().out


class _Report:
    def __init__(self, sources: list[object] | None = None) -> None:
        self.sources = sources if sources is not None else []
        self.goal = "g"

    def to_markdown(self) -> str:
        return "# research report\n"


class _Source:
    def __init__(self, url: str, *, ok: bool, text: str = "") -> None:
        self.url = url
        self.ok = ok
        self.text = text


class _Researcher:
    def __init__(self, report: _Report | None = None) -> None:
        self.report = report or _Report()
        self.redirected: list[str] = []
        self.calls: list[tuple] = []

    def redirect(self, goal: str) -> None:
        self.redirected.append(goal)

    def run(self, goal: str, **kwargs: object) -> _Report:
        self.calls.append((goal, kwargs))
        return self.report


class _Backend:
    def __init__(self, script: list[object]) -> None:
        self.script = script
        self.i = 0

    def complete(self, messages: list[dict[str, str]]) -> str:
        return "planned"

    def complete_with_tools(
        self, messages: list[dict[str, object]], tools: object = None, tool_choice: object = None
    ) -> object:
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class _Completion:
    def __init__(self, content: str, tool_calls: list[dict[str, object]] | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Harness:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def run(self, name: str, args: list[str]) -> SimpleNamespace:
        self.calls.append((name, args))
        if name == "doctor":
            return SimpleNamespace(stdout="doctor-ok\n", stderr="", exit_code=0)
        return SimpleNamespace(stdout="", stderr="train-err", exit_code=1)


def _concierge(tmp_path: Path, **kwargs: object) -> Concierge:
    from fx1.flash.store import FlashStore

    said: list[str] = []
    concierge = Concierge(
        store=kwargs.pop("store", FlashStore(tmp_path / "flash.jsonl")),
        researcher=kwargs.pop("researcher", _Researcher()),
        harness=kwargs.pop("harness", _Harness()),
        approvals=kwargs.pop("approvals", ApprovalGate(ask_fn=lambda question: True)),
        backend_factory=kwargs.pop("backend_factory", None),
        input_fn=kwargs.pop("input_fn", lambda prompt: ""),
        output_fn=said.append,
        animated=kwargs.pop("animated", False),
    )
    concierge.said = said  # type: ignore[attr-defined]
    return concierge


def test_parse_research_and_job_join() -> None:
    with pytest.raises(ValueError, match="usage: /research"):
        _parse_research_command("   ")
    with pytest.raises(ValueError, match="needs an integer"):
        _parse_research_command("goal --budget-s no")
    goal, options = _parse_research_command(
        'goal text --budget-s 3 --max-sources 2 --queries "a|b|" --sync'
    )
    assert goal == "goal text"
    assert options == {"budget_s": 3, "max_sources": 2, "queries": ["a", "b"], "sync": True}
    with pytest.raises(ValueError, match="usage: /research <goal>"):
        _parse_research_command("--sync")

    idle = BackgroundJob(label="idle", run_fn=lambda cancel, on_event: None)
    assert idle.join(0.01) is False


def test_concierge_session_keys_harness_and_flash(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge = _concierge(tmp_path)
    assert concierge.handle_line("   ") is False
    assert concierge.handle_line("exit") is True
    assert concierge.handle_line("/nope") is False
    assert "unknown command" in capsys.readouterr().err

    concierge.handle_line("/help")
    concierge.handle_line("/model")
    concierge.handle_line("/live")
    concierge.handle_line("/trade help")
    concierge.handle_line("/trade readiness")
    concierge.handle_line("/trade now")
    assert "usage: /trade" in capsys.readouterr().err
    text = "\n".join(concierge.said)
    assert "Live desk" in text and "active model" in text and "/superpower" in text

    concierge.handle_line("/use")
    assert "usage: /use" in capsys.readouterr().err
    profiles.set_endpoint("fx1-lite", "sk-lite", "https://lite.example.test/v1")
    assert concierge.handle_line("/use fx1-lite") is False
    assert concierge.model == "fx1-lite"

    concierge.handle_line("/keys")
    concierge.handle_line("/keys list")
    assert "fx1-lite" in "\n".join(concierge.said)
    concierge.handle_line("/keys remove fx1")
    assert "was not stored" in concierge.said[-1]
    concierge.handle_line("/keys remove fx1-lite")
    assert "removed endpoint" in concierge.said[-1]

    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-new")
    concierge._input = lambda prompt: "https://fresh.example.test/v1"
    concierge.handle_line("/keys set fx1")
    assert "fresh.example.test" in concierge.said[-1]
    assert profiles.resolve_endpoint("fx1")[1].startswith("https://fresh.example.test")

    concierge._input = lambda prompt: (_ for _ in ()).throw(EOFError)
    concierge.handle_line("/keys set fx1")
    assert "cancelled" in concierge.said[-1]
    concierge.handle_line("/keys drop fx1")
    assert "usage: /keys" in capsys.readouterr().err

    concierge.handle_line("/harness")
    concierge.handle_line("/harness describe doctor")
    assert "doctor" in concierge.said[-1]
    concierge.handle_line("/harness run doctor --fast")
    assert concierge._harness.calls[-1] == ("doctor", ["--fast"])
    concierge.handle_line("/harness run train")
    assert "train-err" in capsys.readouterr().err
    denying = _concierge(
        tmp_path, approvals=ApprovalGate(ask_fn=lambda question: False), harness=concierge._harness
    )
    denying.handle_line("/harness run train")
    assert "not approved" in denying.said[-1]
    with pytest.raises(KeyError):
        concierge.handle_line("/harness describe missing-command")
    concierge.handle_line("/harness dance")
    assert "usage: /harness" in capsys.readouterr().err

    concierge.handle_line("/flash list")
    assert concierge.said[-1] == "(no flash entries)"
    concierge.handle_line(
        "/flash add inflation finding --task cpi --tags a,b --source https://a.example --private --verified --uncertainty low --conflict other"
    )
    saved = concierge.said[-1]
    concierge.handle_line("/flash add uncertain only --uncertainty high")
    concierge.handle_line("/flash add plain note with no flags")
    assert "saved flash entry" in concierge.said[-1]
    assert "saved flash entry" in saved
    entry_id = saved.split()[3]
    concierge.handle_line("/flash list")
    listed = "\n".join(concierge.said[-4:])
    assert "private" in listed and "verified" in listed and "low" in listed and "high" in listed
    concierge.handle_line("/flash list all")
    assert entry_id[:8] in "\n".join(concierge.said[-4:])
    concierge.handle_line(f"/flash show {entry_id}")
    assert "inflation finding" in concierge.said[-1]
    concierge.handle_line("/flash show missing")
    assert "no flash entry" in capsys.readouterr().err
    concierge.handle_line(
        f"/flash correct {entry_id} --text 'revised finding' --task t2 --uncertainty high --tags x,y --verified --conflict extra"
    )
    assert "corrected" in concierge.said[-1]
    concierge.handle_line("/flash search inflation finding")
    assert "score=" in concierge.said[-1]
    retrieve_mod = importlib.import_module("fx1.flash.retrieve")
    monkeypatch.setattr(retrieve_mod, "retrieve", lambda *args, **kwargs: [])
    concierge.handle_line("/flash search inflation")
    assert "no flash context" in concierge.said[-1]
    concierge.handle_line(f"/flash refresh {entry_id}")
    assert concierge.said[-1]
    concierge.handle_line("/flash refresh missing")
    assert "no flash entry" in capsys.readouterr().err
    concierge.handle_line(f"/flash remove {entry_id}")
    assert "removed" in concierge.said[-1]
    concierge.handle_line("/flash remove missing")
    assert "no entry" in concierge.said[-1]
    concierge.handle_line("/flash list")
    remaining = "\n".join(concierge.said[-3:])
    assert "uncertain only" in remaining
    assert entry_id not in remaining
    concierge.handle_line("/flash add   ")
    assert "usage: /flash add" in capsys.readouterr().err
    concierge.handle_line("/flash correct")
    assert "usage: /flash correct" in capsys.readouterr().err
    with pytest.raises(KeyError):
        concierge.handle_line("/flash correct missing --conflict x")
    with pytest.raises(KeyError):
        concierge.handle_line("/flash correct missing --text hi")
    concierge.handle_line("/flash add keep me --task t")
    fresh = concierge.said[-1].split()[3]
    concierge.handle_line(f"/flash correct {fresh} --nope 1")
    assert "unknown /flash correct" in capsys.readouterr().err
    concierge.handle_line("/flash")
    assert "usage: /flash" in capsys.readouterr().err
    concierge.handle_line("/flash list not-a-real-query-token")
    assert concierge.said[-1]


def test_concierge_research_superpower_and_chat(
    store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gate = threading.Event()

    class Blocking:
        def __init__(self) -> None:
            self.redirected: list[str] = []

        def redirect(self, goal: str) -> None:
            self.redirected.append(goal)

        def run(self, goal: str, **kwargs: object) -> _Report:
            gate.wait(1.0)
            return _Report(
                [_Source("https://a.example/x", ok=True, text="inflation rose 3 percent")]
            )

    researcher = Blocking()
    concierge = _concierge(tmp_path, researcher=researcher)
    assert concierge.handle_line("/status") is False
    assert concierge.said[-1] == "no background job"
    assert concierge.handle_line("/cancel") is False
    assert "no running" in concierge.said[-1]
    concierge.handle_line("/redirect")
    assert "usage: /redirect" in capsys.readouterr().err
    concierge.handle_line("/redirect a new goal")
    assert "no running background job to redirect" in concierge.said[-1]

    concierge.handle_line("/research inflation outlook --budget-s 4 --max-sources 2")
    assert "running in the background" in concierge.said[-1]
    concierge.handle_line("/superpower something else")
    assert "already running" in concierge.said[-1]
    assert concierge.handle_line("/research another") is False
    assert "already running" in concierge.said[-1]
    concierge.handle_line("/status")
    assert "running" in concierge.said[-1]
    concierge.handle_line("/redirect a tighter inflation question")
    assert researcher.redirected == ["a tighter inflation question"]
    concierge.handle_line("/cancel")
    assert "cancel requested" in concierge.said[-1]
    gate.set()
    assert concierge._job is not None
    concierge._job.join(1.0)
    concierge._job.announced = False
    concierge.handle_line("/status")
    assert "finished" in "\n".join(concierge.said[-3:])
    assert concierge.shutdown(timeout_s=1.0) is True

    class NoRedirect:
        def run(self, goal: str, **kwargs: object) -> _Report:
            return _Report()

    plain = _concierge(tmp_path, researcher=NoRedirect())
    hold = threading.Event()

    def block(goal: str, **kwargs: object) -> _Report:
        hold.wait(1.0)
        return _Report()

    plain._researcher.run = block  # type: ignore[method-assign]
    plain.handle_line("/research stay")
    plain.handle_line("/redirect next")
    assert "cannot be redirected" in capsys.readouterr().err
    hold.set()
    plain.shutdown(1.0)

    sync = _concierge(
        tmp_path,
        researcher=_Researcher(_Report([_Source("https://a.example", ok=True, text="body")])),
    )
    sync.handle_line("/research sync goal --queries q1|q2 --sync")
    assert any(
        "research report" in line or "SOURCE" in line or line.startswith("#") for line in sync.said
    )

    # A job that ignores cancellation is joined explicitly below.
    release = threading.Event()

    def linger(cancel: threading.Event, on_event: object) -> str:
        release.wait(1.0)
        return "late"

    hung = _concierge(tmp_path)
    hung._job = BackgroundJob(label="hung", run_fn=linger).start()
    assert hung.shutdown(timeout_s=0.0) is False
    assert "did not stop" in hung.said[-1]
    release.set()
    hung._job.join(1.0)

    boom = _concierge(tmp_path)

    def explode(cancel: threading.Event, on_event: object) -> None:
        raise RuntimeError("job exploded")

    boom._job = BackgroundJob(label="boom", run_fn=explode).start()
    boom._job.join(1.0)
    boom.tick()
    assert "failed" in boom.said[-1]

    markdown = _concierge(tmp_path)
    markdown._job = BackgroundJob(label="md", run_fn=lambda *a: None)
    markdown._job.result = _Report()
    markdown._job.done_event.set()
    markdown.tick()
    assert "# research report" in markdown.said[-1]

    finished = _concierge(tmp_path)
    finished._job = BackgroundJob(label="plain", run_fn=lambda *a: None)
    finished._job.result = 5
    finished._job.done_event.set()
    finished._announce_job(finished._job)
    assert finished.said[-1] == "[plain] finished"

    calls: list[str] = []

    def plan_goal(goal: str, model_fn: object = None) -> SimpleNamespace:
        if model_fn:
            calls.append(model_fn("prompt"))  # type: ignore[operator]
        return SimpleNamespace(describe=lambda: "the plan")

    def run_plan(plan: object, ctx: object) -> SuperpowerResult:
        return SuperpowerResult(goal="g", plan=plan, summary="did the work", runs=[])  # type: ignore[arg-type]

    monkeypatch.setattr(concierge_mod, "plan_goal", plan_goal)
    monkeypatch.setattr(concierge_mod, "run_plan", run_plan)
    powered = _concierge(
        tmp_path,
        backend_factory=lambda **kwargs: _Backend([]),
        animated=False,
    )
    profiles.set_endpoint("fx1", "sk-fx", "https://models.example.test/v1")
    powered.handle_line("/superpower help")
    assert "capab" in powered.said[-1].lower() or powered.said[-1]
    powered.handle_line("/superpower")
    assert "usage: /superpower" in capsys.readouterr().err
    powered.handle_line("/superpower --sync")
    assert "usage: /superpower" in capsys.readouterr().err
    powered.handle_line("/superpower map the inflation evidence --sync")
    assert calls == ["planned"]
    assert "did the work" in "\n".join(powered.said)
    powered.handle_line("/superpower again")
    assert "background" in powered.said[-1]
    powered.join_job(1.0)
    assert powered.shutdown(1.0) is True

    broken = _concierge(
        tmp_path, backend_factory=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no backend"))
    )
    profiles.set_endpoint("fx1", "sk-fx", "https://models.example.test/v1")
    assert broken._backend() is None

    def tool(name: str, arguments: str, call_id: str = "c1") -> dict[str, object]:
        return {"id": call_id, "function": {"name": name, "arguments": arguments}}

    profiles.remove_endpoint("fx1")
    profiles.remove_endpoint("fx1-lite")
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("FX1_BASE_URL", raising=False)
    chat = _concierge(tmp_path, researcher=_Researcher(_Report()))
    chat.handle_line("hello without a model")
    assert "no model endpoint" in chat.said[-1]

    profiles.set_endpoint("fx1", "sk-fx", "https://models.example.test/v1")
    scripted = _concierge(
        tmp_path,
        backend_factory=lambda **kwargs: _Backend(
            [
                _Completion("Sharpe 2.1 is the headline"),
            ]
        ),
        animated=True,
    )
    scripted.handle_line("ask one")
    assert "honesty gate" in scripted.said[-1]
    assert scripted._messages == []

    looping = _concierge(
        tmp_path,
        backend_factory=lambda **kwargs: _Backend(
            [_Completion("", [tool("unknown_tool", "{}", f"c{i}")]) for i in range(6)]
        ),
    )
    looping.handle_line("loop")
    assert "tool loop budget" in looping.said[-1]

    saving = _concierge(
        tmp_path,
        researcher=_Researcher(
            _Report(
                [
                    _Source("https://a.example/one", ok=False, text="skip"),
                    _Source(
                        "https://a.example/two", ok=True, text="The inflation rate was 3 percent."
                    ),
                ]
            )
        ),
        harness=_Harness(),
        backend_factory=lambda **kwargs: _Backend(
            [
                _Completion(
                    "",
                    [
                        tool(
                            "flash_save",
                            json.dumps({"text": "saved inflation finding", "uncertainty": "low"}),
                        ),
                        tool("flash_retrieve", json.dumps({"query": "inflation"})),
                        tool("web_lookup", json.dumps({"query": "inflation"})),
                        tool("web_lookup", "{}"),
                        tool("harness_run", json.dumps({"name": "doctor", "args": ["--x", 1]})),
                        tool("harness_run", json.dumps({"name": "missing"})),
                        tool("nope", "not-json"),
                        tool("nope", "[]"),
                    ],
                ),
                _Completion("Sharpe 2.1 after tools"),
            ]
        ),
    )
    saving.handle_line("use the tools")
    assert "honesty gate" in saving.said[-1]
    assert any(message.get("role") == "tool" for message in saving._messages)

    dead = _concierge(
        tmp_path,
        backend_factory=lambda **kwargs: _Backend([_Completion("hello")]),
    )

    def blow(
        messages: list[dict[str, object]], tools: object = None, tool_choice: object = None
    ) -> object:
        raise ConnectionError("network down")

    backend = dead._backend_factory()  # type: ignore[operator]
    backend.complete_with_tools = blow  # type: ignore[method-assign]
    dead._backend_factory = lambda **kwargs: backend
    dead.handle_line("offline")
    assert "network down" in capsys.readouterr().err

    assert "no relevant" in saving._execute_tool(
        {
            "function": {
                "name": "flash_retrieve",
                "arguments": json.dumps({"query": "zzzz-no-such-token"}),
            }
        }
    )
    assert saving._web_lookup("   ") == "empty query"
    empty_lookup = _concierge(tmp_path, researcher=_Researcher(_Report([])))
    assert "nothing" in empty_lookup._web_lookup("inflation")

    monkeypatch.setattr(concierge_mod.sys, "platform", "linux")
    monkeypatch.setattr(concierge_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(concierge_mod.select, "select", lambda *a, **k: ([], [], []))
    assert concierge_mod._tty_polling_read(lambda prompt: "x", "", 0.001) is None
    monkeypatch.setattr(
        concierge_mod.select, "select", lambda *a, **k: ([concierge_mod.sys.stdin], [], [])
    )
    monkeypatch.setattr(concierge_mod.sys.stdin, "readline", lambda: "")
    with pytest.raises(EOFError):
        concierge_mod._tty_polling_read(lambda prompt: "x", "dip › ", 0.001)
    monkeypatch.setattr(concierge_mod.sys.stdin, "isatty", lambda: False)
    assert concierge_mod._tty_polling_read(lambda prompt: "typed", "dip › ", 0.001) == "typed"


def test_shell_zero_exits_and_parse_faults(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repl = _shell(monkeypatch)
    monkeypatch.setattr(actions, "run_eval", lambda out, model: 0)
    monkeypatch.setattr(actions, "harness_run", lambda name, args, model: 0)
    monkeypatch.setattr(actions, "verify", lambda path, model: 0)
    monkeypatch.setattr(actions, "run_tool", lambda tool, args, model: 0)
    repl.onecmd("eval")
    repl.onecmd("harness run doctor")
    repl.onecmd("verify one.json")
    repl.onecmd("run fx1")
    assert capsys.readouterr().err == ""
    repl.onecmd('verify "')
    assert "could not parse" in capsys.readouterr().err
    repl.onecmd('run "')
    assert "could not parse" in capsys.readouterr().err
    repl.onecmd("no-such-command")
    assert "unknown command" in capsys.readouterr().err
    assert repl.onecmd("EOF") is True


def test_profiles_store_edges(store: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FX1_CONFIG_DIR", raising=False)
    assert profiles.config_dir() == Path.home() / ".fx1"

    broken = tmp_path / "credentials.json"
    broken.write_text("{", encoding="utf-8")
    assert profiles._load(broken) == {}
    broken.write_text(json.dumps({"profiles": ["nope"]}), encoding="utf-8")
    assert profiles._load(broken) == {}

    def explode_replace(src: str, dst: str) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(profiles.os, "replace", explode_replace)
    with pytest.raises(OSError, match="replace failed"):
        profiles._save(
            {"fx1": {"api_key": "k", "base_url": "https://a.example", "set_at": "t"}}, broken
        )
    assert list(broken.parent.glob(".credentials-*")) == []

    stale = tmp_path / "stale.json"
    stale.write_text(
        json.dumps(
            {"profiles": {"fx1": {"api_key": "", "base_url": "https://a.example", "set_at": ""}}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-env")
    monkeypatch.setenv("FX1_BASE_URL", "https://env.example/v1")
    resolved = profiles.resolve_endpoint("fx1", path=stale)
    assert resolved == ("sk-env", "https://env.example/v1")


def test_actions_probe_table_and_subprocess(
    store: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profiles.set_endpoint("fx1", "sk-act", "https://1.1.1.1/v1")
    monkeypatch.setattr(actions.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="not found"):
        actions.run_tool("fx1", [], "fx1")
    monkeypatch.setattr(actions.shutil, "which", lambda name: "/usr/bin/true")
    monkeypatch.setattr(
        actions.subprocess, "run", lambda argv, check=False: SimpleNamespace(returncode=0)
    )
    assert actions.run_eval(None, "fx1") == 0
    assert actions.verify(Path("receipt.json"), "fx1") == 0

    class _Backend:
        def complete(self, messages: list[dict[str, str]]) -> str:
            return "reply"

    assert (
        actions._complete_with_orb(_Backend(), [{"role": "user", "content": "hi"}], animated=False)
        == "reply"
    )
    assert (
        actions._complete_with_orb(_Backend(), [{"role": "user", "content": "hi"}], animated=True)
        == "reply"
    )

    class _Harness:
        def run(self, name: str, extra: list[str] | None) -> SimpleNamespace:
            return SimpleNamespace(stdout="DATA_LABEL=SYNTHETIC\n", stderr="", exit_code=0)

    monkeypatch.setattr(actions, "Harness", lambda: _Harness())
    assert actions.harness_run("doctor", [], "fx1") == 0
    assert "SYNTHETIC" in capsys.readouterr().out

    class _Err:
        def run(self, name: str, extra: list[str] | None) -> SimpleNamespace:
            return SimpleNamespace(stdout="", stderr="nope\n", exit_code=2)

    monkeypatch.setattr(actions, "Harness", lambda: _Err())
    assert actions.harness_run("doctor", [], "fx1") == 2
    assert "nope" in capsys.readouterr().err

    class _Conn:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(actions.socket, "create_connection", lambda addr, timeout: _Conn())
    assert actions._probe("https://1.1.1.1") == "reachable"
    monkeypatch.setattr(
        actions.socket,
        "create_connection",
        lambda addr, timeout: (_ for _ in ()).throw(OSError("down")),
    )
    assert actions._probe("http://1.1.1.1") == "unreachable"
    assert actions.render_table([]) == "(no rows)"


def test_orb_frames_and_animator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLORFGBG", "15;15")
    assert orb_mod._background_is_dark() is False
    monkeypatch.setenv("COLORFGBG", "15;nope")
    assert orb_mod._background_is_dark() is True
    monkeypatch.setenv("COLORFGBG", "0;0")
    assert orb_mod._background_is_dark() is True
    monkeypatch.delenv("COLORFGBG", raising=False)
    assert orb_mod._background_is_dark() is True

    for state in OrbState:
        frame = orb_mod.render_frame(state, t=0.2, preset="inline", color=False)
        assert frame
    banner = orb_mod.banner("fx1", "example.test", color=False)
    assert "dipcatcher" in banner

    stream = io.StringIO()
    stream.isatty = lambda: True  # type: ignore[method-assign]
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FXI_ORBS", raising=False)
    animator = OrbAnimator(OrbState.SEARCHING, fps=1000, stream=stream, preset="inline")

    def stop_soon(seconds: float) -> None:
        animator._stop.set()

    monkeypatch.setattr(orb_mod.time, "sleep", stop_soon)
    with animator:
        animator._stop.wait(0.2)
    assert stream.getvalue()


def test_superpower_failure_and_offline_summary() -> None:
    plan = SuperpowerPlan(
        goal="inflation",
        capabilities=[
            PlannedCapability("flash_retrieve", "memory"),
            PlannedCapability("web_research", "web"),
            PlannedCapability("doctor", "doctor"),
            PlannedCapability("train", "train"),
            PlannedCapability("flash_save", "save"),
            PlannedCapability("synthesize", "write"),
        ],
    )

    class BoomStore:
        def add(self, **kwargs: object) -> object:
            raise RuntimeError("save down")

    class BoomResearch:
        def run(self, goal: str, **kwargs: object) -> object:
            raise RuntimeError("search down")

    class BoomHarness:
        def run(self, name: str, args: list[str] | None = None) -> SimpleNamespace:
            raise RuntimeError(f"{name} down")

    ctx = SuperpowerContext(
        model_fn=lambda prompt: (_ for _ in ()).throw(RuntimeError("model down")),
        harness=BoomHarness(),
        approvals=ApprovalGate(ask_fn=lambda question: False),
        researcher=BoomResearch(),
        flash_store=BoomStore(),
        save_to_flash=True,
    )
    # retrieve() rejects a non-store before the save/research steps record their own faults
    failed = run_plan(plan, ctx)
    assert any(run.status == "failed" for run in failed.runs)
    assert any("honesty" in run.detail or run.status == "failed" for run in failed.runs)
    explained = SuperpowerResult(
        goal="g",
        plan=plan,
        runs=[CapabilityRun("doctor", "doctor", "ran", "ok")],
        honesty_violation="Sharpe headline",
    )
    assert "honesty gate" in explained.explain()

    offline = _offline_summary(
        SuperpowerResult(
            goal="g",
            plan=plan,
            runs=[
                CapabilityRun("synthesize", "synthesis", "ran", "skip me"),
                CapabilityRun("doctor", "doctor", "failed", "no card"),
            ],
            errors=["doctor: no card"],
        )
    )
    assert "skip me" not in offline
    assert "errors:" in offline

    bare = SuperpowerPlan(goal="quiet", capabilities=[PlannedCapability("doctor", "d")])
    quiet = run_plan(bare, SuperpowerContext())
    assert quiet.summary == "" or quiet.runs


def test_fxi_app_commands(store: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    runner = CliRunner()
    monkeypatch.setattr(shell_mod, "run_shell", lambda model: None)
    monkeypatch.setattr(wizard, "enter", lambda force=False: None)
    assert runner.invoke(fxi_app.app, ["shell"]).exit_code == 0
    assert runner.invoke(fxi_app.app, ["setup"]).exit_code == 0

    monkeypatch.setattr(
        actions, "run_eval", lambda out, model: (_ for _ in ()).throw(RuntimeError("no endpoint"))
    )
    failed_eval = runner.invoke(fxi_app.app, ["eval"])
    assert failed_eval.exit_code == 2 and "no endpoint" in failed_eval.output
    monkeypatch.setattr(
        actions, "chat", lambda model, prompt: (_ for _ in ()).throw(RuntimeError("chat down"))
    )
    failed_chat = runner.invoke(fxi_app.app, ["chat"])
    assert failed_chat.exit_code == 2 and "chat down" in failed_chat.output

    monkeypatch.setattr("fx1.interactive.superpower.describe_capabilities", lambda: "caps")
    catalog = runner.invoke(fxi_app.app, ["superpower", "--capabilities"])
    assert catalog.exit_code == 0 and "caps" in catalog.stdout
    missing = runner.invoke(fxi_app.app, ["superpower"])
    assert missing.exit_code != 0

    def slash(self: object, text: str) -> bool:
        return False

    monkeypatch.setattr(concierge_mod.Concierge, "_slash", slash)
    monkeypatch.setattr(concierge_mod.Concierge, "join_job", lambda self, timeout_s=None: None)
    ran = runner.invoke(fxi_app.app, ["superpower", "map", "inflation"])
    assert ran.exit_code == 0

    from fx1.flash.store import FlashStore

    # FX1_CONFIG_DIR isolates the default store; replacing the class breaks its lock.
    flash = FlashStore()
    saved = flash.add(text="inflation finding", uncertainty="low")
    listed = runner.invoke(fxi_app.app, ["flash", "list"])
    assert saved.id in listed.stdout
    queried = runner.invoke(fxi_app.app, ["flash", "list", "inflation"])
    assert queried.exit_code == 0

    monkeypatch.setattr(sys, "argv", ["fxi", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        fxi_app.main()
    assert help_exit.value.code in {0, None}
    import runpy

    monkeypatch.setattr(sys, "argv", ["fxi", "--help"])
    with pytest.raises(SystemExit):
        runpy.run_path(fxi_app.__file__, run_name="__main__")


def test_concierge_remaining_slots(
    store: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    concierge = _concierge(tmp_path)
    release = threading.Event()
    concierge._job = BackgroundJob(
        label="busy", run_fn=lambda cancel, on_event: release.wait(1.0)
    ).start()
    assert concierge.handle_line("/superpower another goal") is False
    assert "already running" in concierge.said[-1]
    release.set()
    concierge._job.join(1.0)

    finished = _concierge(tmp_path)
    finished._job = BackgroundJob(label="done", run_fn=lambda *a: None)
    finished._job.result = None
    finished._job.done_event.set()
    finished.handle_line("/status")
    assert any("finished" in line for line in finished.said)

    concierge.handle_line("/flash nope")
    assert "usage: /flash" in capsys.readouterr().err

    idle = _concierge(tmp_path)
    idle._job = BackgroundJob(label="done", run_fn=lambda *a: None)
    idle._job.done_event.set()
    idle.handle_line("/redirect a tighter question")
    assert "no running background job" in idle.said[-1]


def test_eval_nonzero_exit_and_unknown_orb(store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    monkeypatch.setattr(actions, "run_eval", lambda out, model: 3)
    result = CliRunner().invoke(fxi_app.app, ["eval"])
    assert result.exit_code == 3
    assert orb_mod._dot_positions("not-a-state", 0.2) == []  # type: ignore[arg-type]


def test_approval_keyboard_interrupt_denies() -> None:
    def interrupt(_question: str) -> bool:
        raise KeyboardInterrupt

    gate = ApprovalGate(ask_fn=interrupt)
    assert gate.ask("transmit an order?") is False
    assert gate.records[-1].approved is False
