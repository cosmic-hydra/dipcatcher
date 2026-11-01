"""SYNTHETIC redirect regression for credentialed harness requests."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fx1.serve.client import _urllib_transport


def test_redirect_does_not_forward_api_key() -> None:
    received: list[str | None] = []

    class Destination(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            received.append(self.headers.get("X-API-Key"))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    class Redirect(BaseHTTPRequestHandler):
        location = ""

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", self.location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: object) -> None:
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)
    redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    Redirect.location = f"http://127.0.0.1:{destination.server_port}/stolen"
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in (destination, redirect)
    ]
    for thread in threads:
        thread.start()
    try:
        status, _headers, _body = _urllib_transport(
            "GET",
            f"http://127.0.0.1:{redirect.server_port}/source",
            None,
            {"X-API-Key": "SYNTHETIC-secret"},
            2.0,
        )
        assert status == 302
        assert received == []
    finally:
        for server in (destination, redirect):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
