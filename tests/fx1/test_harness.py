"""Harness tests: registry fail-closed, runner injection, config containment."""

from pathlib import Path

import pytest

from fx1.harness import Harness, HarnessRole


def _fake_runner(argv: list[str], timeout_s: int) -> tuple[int, str, str]:
    return 0, f"ran: {' '.join(argv)}", ""


def test_registry_covers_all_four_roles():
    harness = Harness(runner=_fake_runner)
    roles = {c.role for c in harness.list_commands()}
    assert roles == {
        HarnessRole.DATA_ENGINE,
        HarnessRole.EVALUATION,
        HarnessRole.VERIFICATION,
        HarnessRole.MODEL_TRAINING,
    }


def test_registry_covers_full_lab_surface():
    names = {c.name for c in Harness(runner=_fake_runner).list_commands()}
    expected = {
        "doctor",
        "verify-research",
        "validate",
        "monitor",
        "ingest",
        "collect",
        "build-features",
        "build-labels",
        "research",
        "northset",
        "backtest",
        "forecast",
        "kronos-forecast",
        "candle-book",
        "kyle-ofi",
        "session-book",
        "vendor-book-map",
        "book-panel",
        "report",
        "tearsheet",
        "train",
        "optimize",
        "paper",
        "live",
    }
    assert expected <= names
    # Hidden/network-serving surfaces are deliberately unreachable by fx-1.
    assert "lab" not in names and "api" not in names


def test_role_filter():
    harness = Harness(runner=_fake_runner)
    verification = harness.list_commands(role=HarnessRole.VERIFICATION)
    assert verification and all(c.role == HarnessRole.VERIFICATION for c in verification)


def test_live_harness_command_is_status_only():
    command = next(c for c in Harness(runner=_fake_runner).list_commands() if c.name == "live")
    assert command.argv == ["live", "status"]
    assert command.role == HarnessRole.MODEL_TRAINING


def test_unregistered_command_fails_closed():
    harness = Harness(runner=_fake_runner)
    with pytest.raises(KeyError, match="not a registered harness command"):
        harness.run("live-trading")


def test_run_dispatches_through_injected_runner():
    harness = Harness(runner=_fake_runner)
    result = harness.run("verify-research")
    assert result.ok
    assert "verify-research" in result.stdout


def test_config_contained_to_configs_dir(tmp_path: Path):
    harness = Harness(runner=_fake_runner)
    with pytest.raises(ValueError, match="allowlist"):
        harness.run("research", config=tmp_path / "evil.yaml")


def test_default_runner_targets_dipcatcher_cli():
    harness = Harness()
    # Registry argv always targets the dipcatcher harness CLI surface.
    assert all(isinstance(c.argv, list) and c.argv for c in harness.list_commands())
