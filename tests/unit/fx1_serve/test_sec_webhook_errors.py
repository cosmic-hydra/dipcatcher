"""SYNTHETIC callback URL error redaction regressions."""

from __future__ import annotations

import traceback

import pytest

from fx1.serve.webhooks import check_callback_url, deliver_signed


@pytest.mark.parametrize(
    "url",
    [
        "https://user:SYNTHETIC-secret@example.com/hook",
        "https://example.com:SYNTHETIC-secret/hook",
    ],
)
def test_invalid_callback_url_does_not_expose_embedded_secret(url: str) -> None:
    with pytest.raises(ValueError) as caught:
        check_callback_url(url)
    assert "SYNTHETIC-secret" not in "".join(traceback.format_exception(caught.value))

    delivered, error, attempts = deliver_signed(url, None, b"{}")
    assert not delivered
    assert attempts == 0
    assert "SYNTHETIC-secret" not in (error or "")
