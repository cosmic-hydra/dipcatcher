"""Runtime research flags — CI/examples bound the live-optional catalog."""

from __future__ import annotations

import pytest

from quant_fund.research.research_flags import research_required_only


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("nope", False),
    ],
)
def test_research_required_only_env(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("DIPCATCHER_RESEARCH_REQUIRED_ONLY", raw)
    assert research_required_only() is expected


def test_research_required_only_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIPCATCHER_RESEARCH_REQUIRED_ONLY", raising=False)
    assert research_required_only() is False
