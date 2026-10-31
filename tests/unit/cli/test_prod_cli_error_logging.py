from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from quant_fund.reality.cli import FETCH_UNAVAILABLE_EXIT, reality_app
from quant_fund.utils.cli_errors import DipcatcherTyperGroup

_RUNNER = CliRunner()


def test_guarded_cli_logs_one_user_error_line_without_a_traceback() -> None:
    app = typer.Typer(cls=DipcatcherTyperGroup)

    @app.command("missing")
    def missing() -> None:
        raise FileNotFoundError(2, "No such file", "missing.yaml")

    @app.command("other")
    def other() -> None:
        pass

    result = _RUNNER.invoke(app, ["missing"])
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr == "error: file not found: missing.yaml\n"
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("sweep", "quant_fund.research.reality_sweep.run_sweep"),
        ("survivorship", "quant_fund.research.reality_survivorship.run_study"),
    ],
)
def test_research_lane_input_errors_are_clean_cli_diagnostics(
    command: str, target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise ValueError("invalid research input")

    monkeypatch.setattr(target, fail)
    result = _RUNNER.invoke(reality_app, [command])

    assert result.exit_code == 2
    assert "Invalid value: invalid research input" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize(
    ("command", "target", "prefix"),
    [
        (
            "sweep",
            "quant_fund.research.reality_sweep.run_sweep",
            "REALITY_SWEEP_FETCH_UNAVAILABLE",
        ),
        (
            "survivorship",
            "quant_fund.research.reality_survivorship.run_study",
            "REALITY_SURVIVORSHIP_FETCH_UNAVAILABLE",
        ),
    ],
)
def test_research_lane_fetch_errors_use_the_dedicated_exit_and_log_prefix(
    command: str, target: str, prefix: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("panel unavailable")

    monkeypatch.setattr(target, fail)
    result = _RUNNER.invoke(reality_app, [command])

    assert result.exit_code == FETCH_UNAVAILABLE_EXIT
    assert result.stdout == f"{prefix}: panel unavailable\n"
    assert result.stderr == ""
    assert "Traceback" not in result.output
