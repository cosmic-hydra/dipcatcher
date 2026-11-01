"""SYNTHETIC ingress checks for the local weights HTTP adapter."""

from __future__ import annotations

import io
from email.message import Message

import pytest

from fx1.serve.local_engine import _MAX_REQUEST_BYTES, _Handler


def _read(
    headers: list[tuple[str, str]], raw: bytes = b"{}"
) -> tuple[object, list[int], bool, int]:
    handler = object.__new__(_Handler)
    message = Message()
    for name, value in headers:
        message.add_header(name, value)
    handler.headers = message
    handler.rfile = io.BytesIO(raw)
    handler.close_connection = False
    statuses: list[int] = []
    handler._respond_json = lambda status, _body: statuses.append(status)  # type: ignore[method-assign]
    result = handler._read_body()
    return result, statuses, handler.close_connection, handler.rfile.tell()


@pytest.mark.parametrize(
    "headers",
    [
        [],
        [("Content-Length", "-1")],
        [("Content-Length", "+2")],
        [("Content-Length", "2"), ("Content-Length", "3")],
    ],
)
def test_rejects_invalid_content_length_without_reading(headers: list[tuple[str, str]]) -> None:
    result, statuses, close, read = _read(headers)
    assert result is None
    assert statuses == [400]
    assert close
    assert read == 0


def test_rejects_oversized_body_without_reading() -> None:
    result, statuses, close, read = _read([("Content-Length", str(_MAX_REQUEST_BYTES + 1))])
    assert result is None
    assert statuses == [413]
    assert close
    assert read == 0


def test_rejects_transfer_encoding_even_with_length() -> None:
    result, statuses, close, read = _read(
        [("Content-Length", "2"), ("Transfer-Encoding", "chunked")]
    )
    assert result is None
    assert statuses == [400]
    assert close
    assert read == 0


def test_accepts_bounded_json_body() -> None:
    assert _read([("Content-Length", "2")]) == ({}, [], False, 2)
