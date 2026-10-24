"""Production CLI contract: help, version, usage, and fail-closed live.

Subprocess checks cover the module entrypoints. CliRunner checks cover every
registered Typer command. Nothing here opens a socket or places an order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from fx1 import __version__
from fx1.cli import app as fx1_app
from fx1.cli_errors import user_error_message as fx1_user_error_message
from fx1.interactive.app import app as fxi_app
from quant_fund.cli.main import app as dipcatcher_app
from quant_fund.data.quality.cli import main as data_quality_main
from quant_fund.live.gateway_arm import LIVE_CAPITAL_PHRASE
from quant_fund.market_sim.__main__ import _parser as market_sim_parser
from quant_fund.mc_engine.cli import app as mc_app
from quant_fund.utils.cli_errors import user_error_message, version_requested

_ROOT = Path(__file__).resolve().parents[3]
_SRC = _ROOT / "src"
_RUNNER = CliRunner()
_VERSION = {
    "dipcatcher": f"dipcatcher {__version__}\n",
    "fx1": f"fx1 {__version__}\n",
    "fxi": f"fxi {__version__}\n",
    "mc-engine": f"mc-engine {__version__}\n",
}

_INDEX_START = "<!-- cli-index:start -->"
_INDEX_END = "<!-- cli-index:end -->"


def _command_paths(typer_app: object) -> list[str]:
    paths: list[str] = []

    def walk(app: object, prefix: tuple[str, ...]) -> None:
        for command in app.registered_commands:  # type: ignore[attr-defined]
            name = command.name
            if not name and command.callback is not None:
                name = command.callback.__name__.replace("_", "-")
            if not name:
                continue
            paths.append(" ".join((*prefix, name)))
        for group in app.registered_groups:  # type: ignore[attr-defined]
            name = group.name or group.typer_instance.info.name or ""
            if not name:
                continue
            paths.append(" ".join((*prefix, name)))
            walk(group.typer_instance, (*prefix, name))

    walk(typer_app, ())
    return sorted(set(paths))


def _blob(result: object) -> str:
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    output = getattr(result, "output", "") or ""
    return stdout + stderr + output


def _assert_usage(result: object, argv: str) -> None:
    assert result.exit_code == 2, f"{argv} exited {result.exit_code}: {_blob(result)}"
    assert "Traceback" not in _blob(result)


def _assert_surface(typer_app: object, program: str) -> None:
    version = _VERSION[program]
    root_help = _RUNNER.invoke(typer_app, ["--help"])
    assert root_help.exit_code == 0, _blob(root_help)
    assert "Usage" in _blob(root_help)
    assert "--version" in _blob(root_help)
    assert "Traceback" not in _blob(root_help)
    for flag in (["--version"], ["-V"]):
        shown = _RUNNER.invoke(typer_app, flag)
        assert shown.exit_code == 0, f"{program} {' '.join(flag)}: {_blob(shown)}"
        assert shown.stdout == version
    _assert_usage(
        _RUNNER.invoke(typer_app, ["--not-a-real-option"]), f"{program} --not-a-real-option"
    )
    for path in _command_paths(typer_app):
        argv = path.split()
        shown_help = _RUNNER.invoke(typer_app, [*argv, "--help"])
        assert shown_help.exit_code == 0, f"{program} {path} --help: {_blob(shown_help)}"
        assert "Usage" in _blob(shown_help)
        assert "Traceback" not in _blob(shown_help)
        shown_version = _RUNNER.invoke(typer_app, [*argv, "--version"])
        assert shown_version.exit_code == 0, path
        assert shown_version.stdout == version
        _assert_usage(
            _RUNNER.invoke(typer_app, [*argv, "--not-a-real-option"]),
            f"{program} {path} --not-a-real-option",
        )


def test_dipcatcher_surface() -> None:
    _assert_surface(dipcatcher_app, "dipcatcher")


def test_fx1_surface() -> None:
    _assert_surface(fx1_app, "fx1")


def test_fxi_surface() -> None:
    _assert_surface(fxi_app, "fxi")


def test_mc_engine_surface() -> None:
    _assert_surface(mc_app, "mc-engine")


def test_mc_engine_help_interface_is_offline() -> None:
    result = _RUNNER.invoke(mc_app, ["help-interface"])
    assert result.exit_code == 0
    assert "ScenarioGenerator" in result.stdout
    assert "Traceback" not in _blob(result)


def test_fx1_doctor_is_offline_json(tmp_path: Path) -> None:
    result = _RUNNER.invoke(fx1_app, ["doctor", "--root", str(tmp_path)])
    assert result.exit_code == 0, _blob(result)
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert "Traceback" not in _blob(result)


def test_fxi_keys_list_reads_home_and_rejects_unknown_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    listed = _RUNNER.invoke(fxi_app, ["keys", "list"])
    assert listed.exit_code == 0, _blob(listed)
    assert "unset" in listed.stdout
    assert not (tmp_path / ".dipcatcher").exists()
    rejected = _RUNNER.invoke(fxi_app, ["keys", "set", "not-a-model", "--value", "not-a-key"])
    _assert_usage(rejected, "fxi keys set")
    assert not (tmp_path / ".fx1" / "credentials.json").exists()


def test_doctor_user_errors_have_no_traceback(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    missing_result = _RUNNER.invoke(dipcatcher_app, ["doctor", "--config", str(missing)])
    _assert_usage(missing_result, "doctor missing")
    assert f"file not found: {missing}" in (missing_result.stderr or "")

    invalid = tmp_path / "broken.yaml"
    invalid.write_text(":\n  - :\n", encoding="utf-8")
    invalid_result = _RUNNER.invoke(dipcatcher_app, ["doctor", "--config", str(invalid)])
    _assert_usage(invalid_result, "doctor yaml")
    assert "invalid YAML:" in (invalid_result.stderr or "")

    empty = tmp_path / "unknown.yaml"
    empty.write_text("not_a_real_key: 1\n", encoding="utf-8")
    empty_result = _RUNNER.invoke(dipcatcher_app, ["doctor", "--config", str(empty)])
    _assert_usage(empty_result, "doctor config")
    assert "invalid config:" in (empty_result.stderr or "")

    binary = tmp_path / "binary.yaml"
    binary.write_bytes(b"\xff\xfe not utf-8")
    binary_result = _RUNNER.invoke(dipcatcher_app, ["doctor", "--config", str(binary)])
    _assert_usage(binary_result, "doctor encoding")
    assert "UTF-8" in (binary_result.stderr or "")

    directory = tmp_path / "dir.yaml"
    directory.mkdir()
    directory_result = _RUNNER.invoke(dipcatcher_app, ["doctor", "--config", str(directory)])
    _assert_usage(directory_result, "doctor directory")
    assert "expected a file:" in (directory_result.stderr or "")


def test_user_error_messages_cover_the_claimed_cases() -> None:
    missing = FileNotFoundError(2, "No such file", "missing.yaml")
    assert user_error_message(missing) == "file not found: missing.yaml"
    denied = PermissionError(13, "Permission denied", "secret.yaml")
    assert user_error_message(denied) == "permission denied: secret.yaml"
    assert (
        user_error_message(IsADirectoryError(21, "Is a directory", "dir")) == "expected a file: dir"
    )
    assert (
        user_error_message(NotADirectoryError(20, "Not a directory", "file"))
        == "expected a directory: file"
    )
    try:
        b"\xff".decode("utf-8")
    except UnicodeDecodeError as exc:
        assert user_error_message(exc) == "file is not valid UTF-8"
    try:
        json.loads("{")
    except json.JSONDecodeError as exc:
        assert user_error_message(exc) == f"invalid JSON: {exc.msg} (line {exc.lineno})"
    assert user_error_message(ValueError("Config demo.yaml must be a mapping")) == (
        "Config demo.yaml must be a mapping"
    )
    assert user_error_message(ValueError("not a config problem")) is None
    try:
        yaml.safe_load(":\n  - :\n")
    except yaml.YAMLError as exc:
        message = user_error_message(exc)
        assert message is not None and message.startswith("invalid YAML:")
    assert user_error_message(RuntimeError("boom")) is None
    assert fx1_user_error_message(RuntimeError("boom")) is None
    assert version_requested(["--version"])
    assert version_requested(["doctor", "-V"])
    assert not version_requested(["--", "--version"])
    assert not version_requested(["train", "--", "--version"])


def test_unexpected_errors_still_propagate() -> None:
    from typer import Typer

    from quant_fund.utils.cli_errors import DipcatcherTyperGroup

    mini = Typer(cls=DipcatcherTyperGroup)

    @mini.command("burst")
    def burst() -> None:
        raise BrokenPipeError()

    @mini.command("crash")
    def crash() -> None:
        raise RuntimeError("boom")

    piped = _RUNNER.invoke(mini, ["burst"])
    assert piped.exit_code == 1
    assert "Traceback" not in _blob(piped)
    crashed = _RUNNER.invoke(mini, ["crash"])
    assert crashed.exit_code != 0
    assert isinstance(crashed.exception, RuntimeError)


def _env(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["HOME"] = str(home)
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    env["COLUMNS"] = "80"
    env["TERMINAL_WIDTH"] = "80"
    env.pop("FORCE_COLOR", None)
    env.pop("PY_COLORS", None)
    for name in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        env.pop(name, None)
    return env


def _run(
    args: list[str], home: Path, *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=_ROOT,
        env=_env(home),
        input=stdin.decode() if stdin is not None else None,
        text=True,
        capture_output=True,
        check=False,
    )


def test_module_entrypoints_help_and_version_are_stable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    cases = [
        (["-m", "dipcatcher_cli"], "dipcatcher"),
        (["-m", "quant_fund.cli"], "dipcatcher"),
        (["-m", "quant_fund.cli.main"], "dipcatcher"),
        (["-m", "fx1.cli"], "fx1"),
        (["-m", "fx1.interactive.app"], "fxi"),
        (["-m", "quant_fund.mc_engine.cli"], "mc-engine"),
        (["-m", "quant_fund.audit"], "verify-ledger"),
        (["-m", "quant_fund.audit.cli"], "verify-ledger"),
        (["-m", "quant_fund.data.quality"], "data-quality"),
        (["-m", "quant_fund.data.quality.cli"], "data-quality"),
        (["-m", "quant_fund.parity"], "parity"),
        (["-m", "quant_fund.market_sim"], "market-sim"),
        (["-m", "quant_fund.robustness"], "robustness"),
    ]
    for args, program in cases:
        version = _run([*args, "--version"], home)
        assert version.returncode == 0, (args, version.stderr)
        assert version.stdout == f"{program} {__version__}\n"
        assert "RuntimeWarning" not in version.stderr
        assert "Traceback" not in version.stderr
        short = _run([*args, "-V"], home)
        assert short.returncode == 0, (args, short.stderr)
        assert short.stdout == version.stdout
        first = _run([*args, "--help"], home)
        second = _run([*args, "--help"], home)
        assert first.returncode == 0, (args, first.stderr)
        assert first.stdout == second.stdout
        assert first.stderr == second.stderr
        assert "usage" in first.stdout.lower()
        assert "Traceback" not in first.stderr
        assert "RuntimeWarning" not in first.stderr
    assert not (home / ".dipcatcher").exists()


def test_redirected_dipcatcher_does_not_read_stdin(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run(["-m", "dipcatcher_cli", "--help"], home, stdin=b"do-not-echo\n")
    assert result.returncode == 0
    assert "do-not-echo" not in result.stdout
    assert "do-not-echo" not in result.stderr
    assert "Start a conversation" in result.stdout
    version = _run(["-m", "dipcatcher_cli", "--version"], home, stdin=b"do-not-echo\n")
    assert version.returncode == 0
    assert version.stdout == f"dipcatcher {__version__}\n"
    assert "do-not-echo" not in version.stdout


def test_market_sim_help_does_not_import_mlflow(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    script = (
        "import runpy, sys\n"
        "sys.argv = ['quant_fund.market_sim', '--help']\n"
        "try:\n"
        "    runpy.run_module('quant_fund.market_sim', run_name='__main__')\n"
        "except SystemExit as exc:\n"
        "    code = exc.code or 0\n"
        "else:\n"
        "    code = 0\n"
        "assert 'mlflow' not in sys.modules\n"
        "raise SystemExit(code)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        env=_env(home),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()
    assert "mlflow" not in result.stderr.lower()
    assert "Traceback" not in result.stderr


def test_robustness_help_does_not_run_the_smoke(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    result = _run(["-m", "quant_fund.robustness", "--help"], home)
    assert result.returncode == 0, result.stderr
    assert "extension_schema_version" not in result.stdout
    assert "smoke" in result.stdout
    missing = _run(["-m", "quant_fund.robustness", "nope"], home)
    assert missing.returncode == 2
    assert "Traceback" not in missing.stderr


def test_parity_usage_exits_2(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    missing = _run(["-m", "quant_fund.parity", "nope"], home)
    assert missing.returncode == 2
    assert "Traceback" not in missing.stderr
    help_text = _run(["-m", "quant_fund.parity", "--help"], home)
    assert help_text.returncode == 0
    assert "smoke" in help_text.stdout


def test_data_quality_pass_fail_and_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import math
    from datetime import UTC, datetime, timedelta

    import polars as pl

    base = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    n = 40
    close = [10.0 * math.exp(0.001 * i) for i in range(n)]
    frame = pl.DataFrame(
        {
            "symbol": ["AAPL"] * n,
            "event_time": [base + timedelta(minutes=i) for i in range(n)],
            "open": [value * 0.999 for value in close],
            "high": [value * 1.01 for value in close],
            "low": [value * 0.99 for value in close],
            "close": close,
            "volume": [100.0 + i for i in range(n)],
        }
    )
    path = tmp_path / "bars.parquet"
    frame.write_parquet(path)
    assert data_quality_main([str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True

    missing = tmp_path / "missing.parquet"
    assert data_quality_main([str(missing)]) == 2
    captured = capsys.readouterr()
    assert "file not found" in captured.err
    assert "Traceback" not in captured.err

    garbage = tmp_path / "garbage.parquet"
    garbage.write_bytes(b"this is not a parquet file")
    assert data_quality_main([str(garbage)]) == 2
    captured = capsys.readouterr()
    assert "cannot read parquet" in captured.err
    assert "Traceback" not in captured.err


def test_verify_ledger_missing_directory_is_not_a_traceback(tmp_path: Path) -> None:
    from quant_fund.audit.cli import main

    missing = tmp_path / "no-ledger"
    with pytest.raises(SystemExit) as caught:
        main([str(missing)])
    assert caught.value.code == 1
    home = tmp_path / "home"
    home.mkdir()
    help_run = _run(["-m", "quant_fund.audit.cli", "--help"], home)
    assert help_run.returncode == 0
    assert "Exit 0 is valid" in help_run.stdout
    bad = _run(["-m", "quant_fund.audit.cli", str(missing), "--nope"], tmp_path / "home")
    assert bad.returncode == 2
    assert "Traceback" not in bad.stderr
    report = _run(["-m", "quant_fund.audit.cli", str(missing)], tmp_path / "home")
    assert report.returncode == 1
    assert "ledger_missing" in report.stdout
    assert "Traceback" not in report.stderr


def test_argparse_subcommands_match_the_parsers() -> None:
    sim = market_sim_parser()
    names = sorted(sim._subparsers._group_actions[0].choices)  # noqa: SLF001
    assert names == ["bench", "report", "scenarios", "stress", "stylized"]


def _gateway(tmp_path: Path, venue: str) -> Path:
    path = tmp_path / "live.yaml"
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"venue: {venue}",
                "starting_cash: 100000",
                "calendar: us_rth",
                f"pretrade_config: {_ROOT / 'configs' / 'pretrade_risk.yaml'}",
                f"state_dir: {tmp_path / 'state'}",
                "cash_tolerance: 0.01",
                "position_tolerance: 0.000000001",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _refuse_transport(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls = {"n": 0}

    def transport(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
        calls["n"] += 1
        raise AssertionError("live transport was called")

    monkeypatch.setattr("quant_fund.live.brokers.urllib_transport", transport)
    monkeypatch.setattr("urllib.request.urlopen", transport)
    return calls


def test_live_alpaca_stays_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _refuse_transport(monkeypatch)
    for name in (
        "ALPACA_API_KEY_ID",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    config = _gateway(tmp_path, "alpaca-live")
    state = tmp_path / "state"
    common = ["--config", str(config), "--state-dir", str(state), "--actor", "ada"]
    unarmed = _RUNNER.invoke(
        dipcatcher_app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
        ],
    )
    assert unarmed.exit_code == 1
    assert json.loads(unarmed.stdout)["ok"] is False
    assert "not armed" in unarmed.stdout
    armed = _RUNNER.invoke(
        dipcatcher_app,
        ["live", "arm", *common, "--confirm-live-capital", LIVE_CAPITAL_PHRASE],
    )
    assert armed.exit_code == 0, _blob(armed)
    assert json.loads(armed.stdout)["live_capital_confirmed"] is True
    denied = _RUNNER.invoke(
        dipcatcher_app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
        ],
    )
    assert denied.exit_code == 1
    body = json.loads(denied.stdout)
    assert body["ok"] is False
    assert body["live_pnl_claim"] is False
    assert "credentials" in body["error"]
    monkeypatch.setenv("ALPACA_API_KEY_ID", "cli-test-key")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "cli-test-secret-value")
    refused = _RUNNER.invoke(
        dipcatcher_app,
        [
            "live",
            "submit",
            *common,
            "--symbol",
            "AAPL",
            "--side",
            "buy",
            "--qty",
            "1",
            "--limit",
            "100",
            "--mark",
            "100",
        ],
    )
    assert refused.exit_code == 1, _blob(refused)
    payload = json.loads(refused.stdout)
    assert payload["ok"] is False
    assert payload["research_evidence"] is False
    assert "alpaca-live is refused" in payload["error"]
    assert "honesty contract" in payload["error"]
    combined = _blob(refused)
    assert "cli-test-secret-value" not in combined
    assert "Traceback" not in combined
    assert calls["n"] == 0


def test_help_does_not_create_a_live_desk(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    for args in (
        ["-m", "dipcatcher_cli", "--help"],
        ["-m", "dipcatcher_cli", "live", "--help"],
        ["-m", "quant_fund.cli.main", "--version"],
    ):
        result = _run(args, home)
        assert result.returncode == 0, result.stderr
    assert not (home / ".dipcatcher").exists()


def _readme_index() -> dict[str, list[str]]:
    text = (_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index(_INDEX_START)
    end = text.index(_INDEX_END)
    sections: dict[str, list[str]] = {}
    current: str | None = None
    in_fence = False
    for line in text[start:end].splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
            in_fence = False
            continue
        if line.startswith("```") and current is not None:
            in_fence = not in_fence
            continue
        if in_fence and current is not None and line.strip():
            sections[current].append(line.strip())
    return sections


def test_readme_command_index_matches_the_clis() -> None:
    expected = {
        "dipcatcher": _command_paths(dipcatcher_app),
        "fx1": _command_paths(fx1_app),
        "fxi": _command_paths(fxi_app),
        "mc-engine": _command_paths(mc_app),
        "verify-ledger": ["verify-ledger"],
        "data-quality": ["data-quality"],
        "parity": ["smoke"],
        "market-sim": ["bench", "report", "scenarios", "stress", "stylized"],
        "robustness": ["smoke"],
    }
    assert _readme_index() == expected
