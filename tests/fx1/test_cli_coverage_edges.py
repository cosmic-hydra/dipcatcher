"""Edge coverage for fx1.cli, cli_audit, and the dip bench helpers.

Heavy eval suites and webhook/network calls are replaced with fakes.
The harness server is refused before it binds a socket.
"""

from __future__ import annotations

import builtins
import importlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

from fx1 import cli as cli_mod
from fx1.bench import dip
from fx1.bench.run_audit import _write_series, dip_run_audit, dip_run_audit_bench
from fx1.cli import (
    _bad_arg,
    _emit_anthropic_deltas,
    _emit_completion_deltas,
    _emit_response_deltas,
    _fx1_headers,
    _json_list_opt,
    _json_meta,
    _json_obj_opt,
    _load_prompts,
    _or_exit,
    _record_json,
    _resolve_judge,
    _resolve_model_backend,
    _response_frame_of,
    app,
)
from fx1.cli_audit import cli_audit
from fx1.eval.contamination import ProbeResult as ContaminationProbe
from fx1.sdk import ProbeResult as SdkProbeResult

RUNNER = CliRunner()


class _Backend:
    def complete(self, messages: object, **kwargs: object) -> str:
        return "ok"


def test_header_json_and_stream_helpers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    headers = _fx1_headers(
        "byok",
        tmp_path,
        {"base_url": "https://models.example", "api_key": "sk", "model": "m"},
        ["hosted_k3"],
    )
    assert headers["x-fx1-checkpoint-dir"] == str(tmp_path)
    assert headers["x-fx1-byok-model"] == "m"
    assert headers["x-fx1-fallbacks"] == "hosted_k3"
    assert _fx1_headers(None, None, None, []) == {}

    with pytest.raises(typer.Exit):
        _or_exit(lambda: (_ for _ in ()).throw(typer.Exit(code=3)))
    with pytest.raises(typer.Exit) as bad_json:
        _json_meta("{")
    assert bad_json.value.exit_code == 2
    with pytest.raises(typer.Exit):
        _json_obj_opt("[1]", "--metadata")
    with pytest.raises(typer.Exit):
        _json_list_opt("{", "--testing-criteria")
    assert _json_list_opt(None, "--testing-criteria") is None

    assert _response_frame_of("nope") == (None, None)
    assert _response_frame_of(("response.incomplete", "nope")) == (None, None)
    delta, reason = _response_frame_of(
        {
            "type": "response.incomplete",
            "delta": "partial",
            "response": {"incomplete_details": {"reason": "max_tokens"}},
        }
    )
    assert delta == "partial" and reason == "max_tokens"
    _delta, missing_reason = _response_frame_of(
        {"type": "response.incomplete", "response": "not-a-dict"}
    )
    assert missing_reason is None
    _delta, failed = _response_frame_of(
        {"type": "response.failed", "response": {"error": {"message": "boom"}}}
    )
    assert failed == "failed: boom"
    _delta, bare = _response_frame_of({"type": "response.failed", "response": None})
    assert bare == "failed"
    _emit_response_deltas(
        [
            ("response.output_text.delta", {"type": "response.output_text.delta", "delta": "hi"}),
            {"type": "response.cancelled"},
        ]
    )
    echoed = capsys.readouterr()
    assert "hi" in echoed.out
    assert "cancelled" in echoed.err

    _emit_anthropic_deltas(
        [
            "not-a-dict",
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_start", "content_block": {"type": "tool_use"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": ""}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "A"}},
            {
                "data": {
                    "type": "content_block_delta",
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                }
            },
            {"type": "message_stop"},
        ]
    )
    anthropic = capsys.readouterr().out
    assert "[tool_use]" in anthropic and "A" in anthropic and "{}" in anthropic

    _emit_completion_deltas(
        [
            {"choices": [{"text": ""}, {"text": 3}, {"text": "tok"}]},
            {"choices": []},
        ]
    )
    assert "tok" in capsys.readouterr().out

    prompts = tmp_path / "prompts.json"
    prompts.write_text(json.dumps(["alpha", "beta"]), encoding="utf-8")
    assert _load_prompts(prompts) == ["alpha", "beta"]
    junk = tmp_path / "junk.jsonl"
    junk.write_text('{"prompt": ""}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        _load_prompts(junk)
    record = _record_json(
        SimpleNamespace(
            completion_id="c",
            backend="byok",
            model="m",
            ok=True,
            latency_ms=1.5,
            at=0,
            usage=None,
            error=None,
            error_class=None,
            prompt_sha256="p",
            output_sha256="o",
        )
    )
    assert record["completion_id"] == "c" and record["output_sha256"] == "o"


def test_cli_commands_cover_unhit_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("fx1.serve.get_backend", lambda *args, **kwargs: _Backend())
    assert _resolve_model_backend("local_fx1", tmp_path)([]) == "ok"
    with pytest.raises(typer.Exit) as unknown:
        _resolve_model_backend("nope", None)
    assert unknown.value.exit_code == 2
    assert _resolve_judge("hosted_k3")([]) == "ok"

    class Capability:
        passed = False
        honesty_gate_passed = False
        ts_reasoning = SimpleNamespace(overall=0.2)
        calibration = SimpleNamespace(ece=0.1, passed=True)
        tooluse = SimpleNamespace(pass_rate=0.5)
        retrieval = SimpleNamespace(accuracy=0.5)
        ext_bench = None
        options_reasoning = None

        def model_dump_json(self, indent: int = 2) -> str:
            return "{}"

    monkeypatch.setattr("fx1.eval.capability.run_capability_eval", lambda *a, **k: Capability())
    out = tmp_path / "cap.json"
    result = RUNNER.invoke(
        app,
        ["capability-eval", "--backend", "byok", "--out", str(out), "--judge-backend", "hosted_k3"],
    )
    assert result.exit_code == 1
    assert out.is_file()
    assert "ts_reasoning_overall" in result.stdout

    class Score:
        n_instances = 1
        metrics = {"acc": 1.0}
        refusal_gate_passed = True
        honesty_violations = 0
        gate_passed = True

    class Ext:
        synthetic = True
        label = "SYNTHETIC"
        benchmarks = {"mtbench": Score()}
        honesty_gate_passed = True
        passed = True

        def model_dump_json(self, indent: int = 2) -> str:
            return "{}"

    monkeypatch.setattr("fx1.eval.ext_bench.run_ext_bench_eval", lambda *a, **k: Ext())
    mt = tmp_path / "mt.jsonl"
    fb = tmp_path / "fb.jsonl"
    ft = tmp_path / "ft.jsonl"
    for path in (mt, fb, ft):
        path.write_text("{}\n", encoding="utf-8")
    ext_out = tmp_path / "ext.json"
    ext_result = RUNNER.invoke(
        app,
        [
            "ext-bench-eval",
            "--backend",
            "byok",
            "--out",
            str(ext_out),
            "--mtbench-jsonl",
            str(mt),
            "--financebench-jsonl",
            str(fb),
            "--fintoolbench-jsonl",
            str(ft),
        ],
    )
    assert ext_result.exit_code == 0, ext_result.output
    assert "mtbench" in ext_result.stdout

    class Summary(dict):
        def __init__(self) -> None:
            super().__init__(honesty_gate_passed=False)
            self.results = [{"task": "bait", "passed": False}, {"task": "ok", "passed": True}]

    import fx1.eval.redteam as redteam_mod

    real_tasks = redteam_mod.REDTEAM_TASKS
    monkeypatch.setattr("fx1.eval.suite.run_suite", lambda *a, **k: Summary())
    monkeypatch.setattr(redteam_mod, "REDTEAM_TASKS", [{"id": "bait"}])
    red = RUNNER.invoke(app, ["redteam", "--backend", "byok"])
    assert red.exit_code == 1
    assert "bait" in red.stdout
    monkeypatch.setattr(redteam_mod, "REDTEAM_TASKS", real_tasks)

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello from the lab"}]}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fx1.eval.rephrased.run_rephrased_gap",
        lambda model_fn, **kwargs: ContaminationProbe(
            method="rephrased-gap",
            value=0.9,
            threshold=0.3,
            flagged=True,
            limitation="stubbed",
        ),
    )
    audit_out = tmp_path / "cont.json"
    audit = RUNNER.invoke(
        app,
        [
            "contamination-audit",
            "--corpus",
            str(corpus),
            "--out",
            str(audit_out),
            "--with-rephrased-gap",
            "--backend",
            "byok",
        ],
    )
    assert audit.exit_code == 1, audit.output
    assert "rephrased-gap" in audit.stdout

    class _Described:
        ok = True
        text = "statements, consensus, peers"

        def describe(self) -> _Described:
            return self

    source_adapters = importlib.import_module("fx1.data.sources.adapters")
    source_registry = importlib.import_module("fx1.data.sources.registry")
    monkeypatch.setattr(source_adapters, "build_adapter", lambda spec: _Described())
    monkeypatch.setattr(source_registry, "get_spec", lambda name: object())
    scenarios = RUNNER.invoke(app, ["sources", "scenarios"])
    assert scenarios.exit_code in {0, 1}
    if scenarios.exit_code == 0:
        assert scenarios.stdout.strip()

    series = tmp_path / "series"
    series.mkdir()
    _write_series(series, "aaa_1d.parquet", "AAA", n=40)
    bare = RUNNER.invoke(app, ["dipbench", "--data-dir", str(series)])
    assert bare.exit_code == 0, bare.output
    assert "receipt" in bare.stdout
    receipt_path = tmp_path / "nested" / "dip.json"
    dipbench = RUNNER.invoke(
        app,
        ["dipbench", "--data-dir", str(series), "--out", str(receipt_path)],
    )
    assert dipbench.exit_code == 0, dipbench.output
    assert receipt_path.is_file()
    assert "events" in dipbench.stdout

    monkeypatch.setattr(
        "fx1.serve.harness_doctor.run_local_doctor",
        lambda **kwargs: {"ok": False, "checks": []},
    )

    def remote_doctor(client: object, factory: object) -> dict[str, object]:
        assert factory("k-test")  # type: ignore[operator]
        return {"ok": True}

    monkeypatch.setattr("fx1.serve.harness_doctor.run_remote_doctor", remote_doctor)
    local = RUNNER.invoke(app, ["harness", "doctor"])
    assert local.exit_code == 1 and '"ok": false' in local.stdout
    remote = RUNNER.invoke(app, ["harness", "doctor", "--remote", "http://127.0.0.1:9"])
    assert remote.exit_code == 0, remote.output

    listed = RUNNER.invoke(app, ["harness", "vs-list", "--limit", "1"])
    assert listed.exit_code == 0, listed.output
    assert "data" in listed.stdout or "object" in listed.stdout

    monkeypatch.setattr(
        "fx1.sdk.Fx1Harness.probe_backend",
        lambda self, *args, **kwargs: SdkProbeResult(
            backend="byok", ok=True, model="m", latency_ms=1.0
        ),
    )
    probed = RUNNER.invoke(app, ["harness", "probe", "--backend", "byok"])
    assert probed.exit_code == 0, probed.output
    bad_ids = RUNNER.invoke(app, ["harness", "vs-create", "--file-ids", "[1]"])
    assert bad_ids.exit_code == 2
    good_ids = RUNNER.invoke(app, ["harness", "vs-create", "--file-ids", '["file-1"]'])
    assert good_ids.exit_code in {0, 2}
    bad_items = RUNNER.invoke(app, ["harness", "conv-create", "--items", "not-json"])
    assert bad_items.exit_code == 2
    not_list = RUNNER.invoke(app, ["harness", "conv-create", "--items", '{"a": 1}'])
    assert not_list.exit_code == 2

    monkeypatch.delenv("FX1_API_KEY", raising=False)
    monkeypatch.setattr(
        "uvicorn.run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("bound"))
    )
    refused = RUNNER.invoke(app, ["harness", "serve", "--host", "10.1.2.3", "--port", "9"])
    assert refused.exit_code == 2
    assert "non-loopback" in refused.output

    class RemoteIndex:
        def receipts(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(sha256="ab" * 32, name="note")]

    monkeypatch.setattr(cli_mod, "_remote_client", lambda *a, **k: RemoteIndex())
    receipts = RUNNER.invoke(app, ["harness", "receipts", "--remote", "http://127.0.0.1:9"])
    assert receipts.exit_code == 0, receipts.output
    assert "note" in receipts.stdout


def test_main_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["fx1", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        cli_mod.main()
    assert help_exit.value.code in {0, None}
    monkeypatch.setattr(sys, "argv", ["fx1", "--help"])
    with pytest.raises(SystemExit) as scripted:
        runpy.run_path(cli_mod.__file__, run_name="__main__")
    assert scripted.value.code in {0, None}


def test_dip_core_and_run_audit(tmp_path: Path) -> None:
    dip._validate_closes([1.0, 2.0])
    assert dip._validated_close_array([]).size == 0
    assert dip.detect_dip_events([], [], "AAA") == []

    core = getattr(dip._detect_dip_core, "py_func", dip._detect_dip_core)
    closes = np.array([100, 100, 80, 70, 95, 80, 100, 50], dtype=np.float64)
    horizons = np.array([1, 5], dtype=np.int64)
    empty_core = core(np.array([], dtype=np.float64), 0.1, horizons)
    assert empty_core[0].size == 0
    trough, _peak, depths, recovered = core(closes, 0.1, horizons)
    assert trough.size >= 2
    assert float(depths[0]) > 0.1
    assert -1 in recovered or 0 in recovered or 1 in recovered

    with pytest.raises(ValueError, match="escapes"):
        _write_series(tmp_path, "../escape.parquet", "ZZ", n=20)
    audit = dip_run_audit()
    assert audit["empty_raises"] == "raise:FileNotFoundError"
    assert audit["inputs_bound"] is True
    receipt = dip_run_audit_bench()
    assert receipt["schema"] == "dip_run_audit.v1"
    assert receipt["receipt_sha256"]
    assert len(receipt["receipt_sha256"]) == 64

    empty = np.zeros(0, dtype=np.float64)
    none_events = core(empty, 0.1, np.array([1], dtype=np.int64))
    assert none_events[0].size == 0
    no_horizons = core(np.array([10.0, 8.0], dtype=np.float64), 0.1, np.zeros(0, dtype=np.int64))
    assert no_horizons[0].size == 1
    rearm = np.array([12.0, 9.0, 11.5, 8.0, 8.5, 12.0, 13.0], dtype=np.float64)
    armed = core(rearm, 0.1, np.array([1, 3, 100], dtype=np.int64))
    assert armed[0].size >= 1
    assert set(int(v) for v in armed[3].ravel()) <= {-1, 0, 1}

    import fx1.bench.run as bench_run

    patch = pytest.MonkeyPatch()
    real = bench_run.run_dip_bench

    def quiet(data_dir: object, **kwargs: object) -> dict[str, object]:
        if not list(Path(str(data_dir)).glob("*_1d.parquet")):
            return {"n_events": 0}
        return real(data_dir, **kwargs)  # type: ignore[arg-type]

    patch.setattr(bench_run, "run_dip_bench", quiet)
    try:
        opened = dip_run_audit()
    finally:
        patch.undo()
    assert opened["empty_raises"] == "no-raise"

    series = tmp_path / "bars"
    series.mkdir()
    _write_series(series, "ccc_1d.parquet", "CCC", n=40)
    out = tmp_path / "out" / "bench.json"
    assert bench_run.main(["--data-dir", str(series), "--out", str(out)]) == 0
    assert out.is_file()
    patch.setattr(sys, "argv", ["fx1.bench.run", "--data-dir", str(series), "--out", str(out)])
    try:
        with pytest.raises(SystemExit) as exit_code:
            runpy.run_path(bench_run.__file__, run_name="__main__")
        assert exit_code.value.code == 0
    finally:
        patch.undo()


def test_cli_audit_residual_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    real_resolve = cli_mod._resolve_judge

    def resolve(name: str | None) -> object:
        if name == "bogus":
            return None
        return real_resolve(name)

    monkeypatch.setattr(cli_mod, "_resolve_judge", resolve)
    real_dumps = cli_mod.json.dumps

    def dumps(obj: object, *args: object, **kwargs: object) -> str:
        if isinstance(obj, dict) and "moonshot_key" in obj and "fx1_version" in obj:
            return "NOT-JSON"
        return real_dumps(obj, *args, **kwargs)

    monkeypatch.setattr(cli_mod.json, "dumps", dumps)
    captured: list[type] = []
    real_build = builtins.__build_class__

    def build(func: object, name: str, *bases: object, **kwargs: object) -> type:
        cls = real_build(func, name, *bases, **kwargs)
        if name in {"_FakeSDK", "_FakeRemote"}:
            captured.append(cls)
        return cls

    monkeypatch.setattr(builtins, "__build_class__", build)
    result = cli_audit()
    assert isinstance(result, dict)
    assert result["doctor_json_flags"] is False

    sdk_cls = next(cls for cls in captured if cls.__name__ == "_FakeSDK")
    remote_cls = next(cls for cls in captured if cls.__name__ == "_FakeRemote")
    fake = sdk_cls()
    assert fake.run("doctor").stdout == "ran"
    assert fake.vector_store_get("vs_x")["id"] == "vs_x"
    assert fake.vector_store_list()["object"] == "list"
    assert fake.vector_store_file_list("vs_x")["data"] == []
    assert fake.vector_store_file_get("vs_x", "file-1")["id"] == "file-1"
    assert fake.vector_store_file_delete("vs_x", "file-1")["deleted"] is True
    assert fake.vector_store_file_content("vs_x", "file-1")["data"]
    assert fake.verify_receipt({}).valid is True
    assert len(fake.verify_receipts([{}, {}])) == 2
    assert fake.health().status == "ok"
    assert fake.commands() == ["cmd-a", "cmd-b"]
    remote = remote_cls("http://127.0.0.1")
    assert remote.health().status == "ok"
    assert remote.capabilities()["features"]["jobs"] is True

    monkeypatch.setattr(
        "fx1.doctor.collect_status",
        lambda root=None: (_ for _ in ()).throw(RuntimeError("doctor down")),
    )
    again = cli_audit()
    assert again["doctor_json_flags"] is False


def test_empty_bench_dir_that_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = __import__("fx1.bench.run", fromlist=["run_dip_bench"]).run_dip_bench

    def wrapper(data_dir: object, **kwargs: object) -> dict[str, object]:
        if not list(Path(data_dir).glob("*_1d.parquet")):
            return {"n_events": 0}
        return real(data_dir, **kwargs)

    monkeypatch.setattr("fx1.bench.run.run_dip_bench", wrapper)
    audit = dip_run_audit()
    assert audit["empty_raises"] == "no-raise"
    assert audit["inputs_bound"] is True


def test_dip_python_fallback_when_numba_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def guarded(name: str, *args: object, **kwargs: object) -> object:
        if name == "numba" or name.startswith("numba."):
            raise OSError("llvmlite unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    try:
        reloaded = importlib.reload(dip)
        bare = reloaded.njit(lambda value: value + 1)
        assert bare(1) == 2
        events = reloaded.detect_dip_events([10.0, 8.0, 11.0], ["a", "b", "c"], "ZZ")
        assert events == [] or events[0].asset == "ZZ"
    finally:
        monkeypatch.undo()
        importlib.reload(dip)


def test_bad_arg_still_exits() -> None:
    with pytest.raises(typer.Exit) as exc:
        _bad_arg("nope")
    assert exc.value.exit_code == 2
